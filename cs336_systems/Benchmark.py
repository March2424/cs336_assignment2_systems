import torch
import torch.nn as nn
import timeit
import argparse
import numpy as np
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, safe_nvtx_range
from cs336_basics.optimizer import AdamW
from torch.profiler import ProfilerActivity
import contextlib

def benchmark():
    parser = argparse.ArgumentParser(description="End-to-End Benchmarking Script")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=768, help="Hidden dimension size")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--measure_steps", type=int, default=10, help="Number of steps to measure")
    parser.add_argument("--mode", type=str, choices=['fwd', 'fwd_bwd', 'full'], default='full', 
                        help="What to measure: forward only, forward+backward, or full optimizer step")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasicsTransformerLM(vocab_size = 10000,context_length = 256,d_model = 768, num_layers = 12,num_heads = 12, d_ff= 3072, rope_theta = 10000).to(device)
    
    optimizer = AdamW(model.parameters(), lr=1e-3)
    dummy_input = torch.randint(
    0, 10000, (args.batch_size, args.seq_len), device=device
)
    dummy_target = torch.randint(
    0, 10000, (args.batch_size, args.seq_len), device=device
)

    def run():
        if args.mode == 'full':
            optimizer.zero_grad()
        output = model(dummy_input)
        if args.mode in ['fwd_bwd', 'full']:
            loss = cross_entropy(output, dummy_target)
            loss.backward()
        
        if args.mode == 'full':
            optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        
    print(f"Starting benchmark... Mode: {args.mode}, Warmups: {args.warmup_steps}, Measurements: {args.measure_steps}")
    for _ in range(args.warmup_steps):
        run()
    
    timings = []
    if device.type == 'cuda':
        torch.cuda.synchronize()

    for _ in range(args.measure_steps):
        start_time = timeit.default_timer()
        
        run()
        
        end_time = timeit.default_timer()
        timings.append((end_time - start_time) * 1000)
    
    avg_time = np.mean(timings)
    std_time = np.std(timings)

    print("-" * 50)
    print(f"Results for mode: '{args.mode}'")
    print(f"Average Time: {avg_time:.2f} ms")
    print(f"Standard Dev: {std_time:.2f} ms")
    print(f"All timings (ms): {[round(t, 2) for t in timings]}")
    print("-" * 50)

if __name__ == "__main__":
    benchmark()