from cs336_basics.optimizer import AdamW
from multiprocessing import Manager
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from timeit import default_timer as timer
import pandas as pd

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class ToyModel(nn.Module):
    def  __init__(self, in_features:int, out_features:int):
        super().__init__()
        self.f1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.f2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
    
def data_parallel_main(rank, world_size, data, num_steps, result_queue):
    setup(rank,world_size)
    # 保证所有显卡上的模型在初始化那一瞬间，权重参数是完全一样
    torch.manual_seed(0)

    batch_size, d_model = data.shape
    local_batch_size = batch_size // world_size
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size

    device = torch.cuda.current_device()
    toy_model = ToyModel(d_model, d_model).to(device)
    toy_model.train()
    data = data[start_index:end_index].to(device)
    optimizer = AdamW(toy_model.parameters(), lr = 0.001)
    for _ in range(num_steps):
        optimizer.zero_grad()
        output = toy_model(data)
        loss = output.mean()
        loss.backward()
        for param in toy_model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
            param.grad.data /= world_size
        
        optimizer.step()

    if rank == 0:
        cpu_state = {k:v.detach().cpu() for k,v in toy_model.state_dict().items()}
        result_queue.put(cpu_state)
        
def train_single_process(data, num_steps):
    torch.manual_seed(0)
    batch_size, d_model = data.shape
    model = ToyModel(d_model, d_model).to("cuda")
    data = data.to("cuda")
    opt = AdamW(model.parameters(), lr=0.001)
    model.train()

    for _ in range(num_steps):
        opt.zero_grad()
        out = model(data)
        loss = out.mean()
        loss.backward()
        opt.step()
    
    return {k: v.detach().cpu() for k,v in model.state_dict().items()}