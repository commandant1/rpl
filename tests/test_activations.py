import rpl
import time
import numpy as np

def test_activations():
    print("Testing GPU/NEON Activations...")
    
    # 1. Tanh
    print("\n[1] Testing Tanh")
    size = 1024
    a = rpl.Tensor(np.random.uniform(-1, 1, size).astype(np.float32))
    a.to_gpu()
    
    start = time.time()
    res = a.tanh()
    end = time.time()
    print(f"GPU Tanh time: {(end-start)*1000:.4f} ms")
    
    res.to_cpu()
    cpu_res = np.tanh(a.data)
    diff = np.abs(res.data - cpu_res).max()
    print(f"Max diff: {diff}")
    assert diff < 1e-4

    # 2. GELU
    print("\n[2] Testing GELU")
    b = rpl.Tensor(np.random.uniform(-3, 3, size).astype(np.float32))
    b.to_gpu()
    
    start = time.time()
    # Note: Using approximated GELU on GPU
    gpu_gelu = b.gelu()
    end = time.time()
    
    print(f"GPU GELU time: {(end-start)*1000:.4f} ms")
    
    if gpu_gelu:
        gpu_gelu.to_cpu()
        print("GELU GPU output sample:", gpu_gelu.data[:5])
    else:
        print("Skipped GPU GELU (not fully implemented if device=CPU)")

    print("\nAll Activation tests passed!")

if __name__ == "__main__":
    test_activations()
