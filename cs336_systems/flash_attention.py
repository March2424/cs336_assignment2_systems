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
        ctx.save_for_backward(L, Q, K, V, O)
        return O
    
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Part (a) 只需要实现前向传播")




