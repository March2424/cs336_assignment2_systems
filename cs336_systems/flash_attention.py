import torch
import triton
import triton.language as tl
from math import ceil
from einops import rearrange, einsum
from typing import Tuple, Optional
from torch.autograd import Function
import math

class FlashAttentionV2Torch(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Q,K,V,is_causal = False):
        """
        FlashAttention-2前向传播实现
        参数:
            query: [batch_size, seq_len_q, head_dim]
            key: [batch_size, seq_len_k, head_dim]
            value: [batch_size, seq_len_k, head_dim]
            is_causal: 是否使用因果掩码(本任务可忽略)
        返回:
            output: [batch_size, seq_len_q, head_dim]
        """
        Bq,Bk = 16,16
        B,seq_len_q,d = Q.shape
        _,seq_len_k,d = K.shape
        Tq = ceil(seq_len_q / Bq)
        Tk = ceil(seq_len_k / Bk)
        O = torch.zeros_like(Q)
        L = torch.zeros(B,seq_len_q,device = Q.device,dtype = Q.dtype)

        for i in range(Tq):
            start_q = i * Bq
            end_q = min(start_q+Bq,seq_len_q)
            q_len = end_q - start_q # 当前块的真实长度
            Q_i = Q[:,start_q:end_q,:]
            oi = torch.zeros((B,q_len,d), dtype=Q.dtype, device=Q.device)
            li = torch.zeros((B,q_len,1), dtype=Q.dtype, device=Q.device)
            mi = torch.full((B, q_len,1), -float('inf'), dtype=Q.dtype, device=Q.device)

            for j in range(Tk):
                start_k = j * Bk
                end_k = min(start_k + Bk,seq_len_k)
                K_j,V_j = K[:,start_k:end_k,:],V[:,start_k:end_k,:]
                S_ij = einsum(Q_i,K_j,"b s1 d, b s2 d -> b s1 s2") / math.sqrt(d)
                m_ij = torch.max(mi,torch.amax(S_ij, dim=-1, keepdim=True))
                p_ij = torch.exp(S_ij-m_ij)
                exp_diff = torch.exp(mi-m_ij)
                li = exp_diff*li+torch.sum(p_ij,dim = -1,keepdim = True)
                oi = exp_diff * oi + einsum(p_ij,V_j,"b s1 s2, b s2 d->b s1 d")
                mi = m_ij
            O_i = oi / li
            L_i = mi + torch.log(li)
            O[:,start_q:end_q, :] = O_i.to(Q.dtype)
            L[:,start_q:end_q] = L_i.squeeze(-1)
        ctx.q_shape = Q.shape  
        ctx.k_shape = K.shape  
        ctx.v_shape = V.shape  
        ctx.is_causal = is_causal
        ctx.save_for_backward(L, Q, K, V, O)

        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    @torch.compile(fullgraph=True)
    def backward(ctx, dO):
        raise NotImplementedError


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    # Q，K序列总长度
    scale,
    D: tl.constexpr,
    # Head Dimension
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 程序索引
    # GPU看到的只是一段连续的一维内存地址 
    # Triton 通过步长 (Strides) 和make_block_ptr来帮助我们在这一维内存中精准地切出我们想要的数据块
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 用相应的批次索引偏移每个指针
    # # 乘以每个张量的批次步幅 (batch stride)
    # 创建一个分块指针对象，在 GPU 显存中定义一个虚拟的二维网格，并指定我们要读取这个网格中的block
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    # 加载当前线程块负责的Q
    q = tl.load(Q_block_ptr)

    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    T_k = tl.cdiv(N_KEYS,K_TILE_SIZE)
    offs_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    for j in range(T_k):
        kj,vj = tl.load(K_block_ptr), tl.load(V_block_ptr)
        s_ij = tl.dot(q,tl.trans(kj)) * scale
        if is_causal:
            # 计算当前 Key 块在整个序列中的真实索引
            offs_k = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            # 广播比较，生成掩码矩阵 (True 表示合法，False 表示需要被 Mask 掉的未来词)
            causal_mask = offs_q[:, None] >= offs_k[None, :]
            # 用 tl.where 将需要 Mask 的地方替换为 -inf
            s_ij = tl.where(causal_mask, s_ij, -float("inf"))
        m_ij = tl.max(s_ij, axis=-1)
        m_new = tl.maximum(m_i, m_ij)
        
        p_ij = tl.exp(s_ij - m_new[:,None])
        scaling_factor = tl.exp(m_i - m_new)
        m_i = m_new
        l_i = l_i * scaling_factor + tl.sum(p_ij,axis = -1)
        o_i = scaling_factor[:,None] * o_i +  tl.dot(p_ij.to(vj.dtype), vj)
        K_block_ptr = tl.advance(K_block_ptr,(K_TILE_SIZE,0))
        V_block_ptr = tl.advance(V_block_ptr,(K_TILE_SIZE, 0))
    o_i = (1 / l_i)[:, None] * o_i
    l_i = m_i + tl.log(l_i)
    tl.store(O_block_ptr, o_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, l_i, boundary_check=(0,))
    

class FlashAttentionV2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        ctx.save_for_backward(Q,K,V)
        Bq, Bk = 16,16
        B,seq_len_q,D = Q.shape
        _,seq_len_k,D = K.shape
        N_QUERIES = seq_len_q
        N_KEYS = seq_len_k
        Tq = triton.cdiv(N_QUERIES, Bq)
        O = torch.empty((B, N_QUERIES, D), device=Q.device)
        L = torch.empty((B, N_QUERIES), device=Q.device)
        scale = 1 / math.sqrt(D)
        grid = (Tq,B)
        flash_fwd_kernel[grid](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            N_QUERIES,
            N_KEYS,
            scale,
            D,
            Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            is_causal=is_causal,
        )

        ctx.save_for_backward(O, L, Q, K, V)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        return FlashAttentionV2Torch.backward(ctx, dO)
        




