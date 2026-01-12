import rpl
import time
import numpy as np

def benchmark_op(name, func, iter=5):
    # Warmup
    func()
    
    start = time.time()
    for _ in range(iter):
        func()
    end = time.time()
    
    avg_time = (end - start) / iter
    print(f"{name}: {avg_time*1000:.2f} ms")

def main():
    print("========================================")
    print(" RPL Performance Benchmark (RPi 4)")
    print("========================================")
    
    # 1. Matrix Multiplication (1024x1024)
    N = 1024
    print(f"\n[1] Matrix Multiplication ({N}x{N})")
    
    a_cpu = rpl.Tensor(np.random.randn(N, N).astype(np.float32))
    b_cpu = rpl.Tensor(np.random.randn(N, N).astype(np.float32))
    
    # GPU Setup
    a_gpu = rpl.Tensor(np.random.randn(N, N).astype(np.float32))
    b_gpu = rpl.Tensor(np.random.randn(N, N).astype(np.float32))
    a_gpu.to_gpu()
    b_gpu.to_gpu()
    
    def cpu_matmul():
        res = a_cpu @ b_cpu
    
    def gpu_matmul():
        res = a_gpu @ b_gpu
        res.to_cpu() # Force sync to measure full roundtrip latency
    
    benchmark_op("CPU (NEON)", cpu_matmul)
    # Re-run GPU without sync to measure dispatch throughput
    def gpu_matmul_async():
        res = a_gpu @ b_gpu
        # No sync
        
    benchmark_op("GPU (GLES) [Async]", gpu_matmul_async)
    benchmark_op("GPU (GLES) [Sync]", gpu_matmul)


    # 2. Element-wise Activation (ReLU) - 4M Elements
    size = 2048 * 2048
    print(f"\n[2] ReLU Activation ({size:,} elements)")
    
    t_cpu = rpl.Tensor(np.random.randn(size).astype(np.float32))
    t_gpu = rpl.Tensor(np.random.randn(size).astype(np.float32))
    t_gpu.to_gpu()
    
    def cpu_relu():
        res = t_cpu.relu()
        
    def gpu_relu():
        res = t_gpu.relu()
        res.to_cpu()
        
    benchmark_op("CPU (NEON)", cpu_relu)
    benchmark_op("GPU (GLES)", gpu_relu)
    
    print("\nBenchmark Complete.")

if __name__ == "__main__":
    main()
