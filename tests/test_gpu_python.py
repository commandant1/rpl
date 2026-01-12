import rpl
import time
import numpy as np

def test_gpu_integration():
    print("Testing Python GPU Integration...")
    
    # 1. Test Data Transfer
    t1 = rpl.Tensor([1.0, 2.0, 3.0, 4.0])
    print(f"Original device: {t1.device}")
    
    t1.to_gpu()
    print(f"Device after to_gpu(): {t1.device}")
    
    t1.to_cpu()
    print(f"Device after to_cpu(): {t1.device}")
    print(f"Data check: {t1.data}")
    
    # 2. Test GPU Addition
    print("\nTesting GPU Addition...")
    size = 1024
    a = rpl.Tensor(np.ones(size))
    b = rpl.Tensor(np.ones(size) * 2.0)
    
    a.to_gpu()
    b.to_gpu()
    
    start_time = time.time()
    c = a + b
    end_time = time.time()
    
    print(f"GPU Add Time: {(end_time - start_time)*1000:.4f} ms")
    print(f"Result Device: {c.device}")
    
    c.to_cpu()
    print(f"First 5 elements: {c.data[:5]}")
    assert c.data[0] == 3.0
    
    # 3. Test GPU GEMM
    print("\nTesting GPU GEMM...")
    M, N, K = 128, 128, 128
    A = rpl.Tensor(np.random.randn(M, K))
    B = rpl.Tensor(np.eye(K, N)) # Identity
    
    A.to_gpu()
    B.to_gpu()
    
    start_time = time.time()
    C = A @ B
    end_time = time.time()
    
    print(f"GPU GEMM {M}x{K}x{N} Time: {(end_time - start_time)*1000:.4f} ms")
    
    C.to_cpu()
    # A @ I should be close to A
    diff = np.abs(C.data - A.data).mean()
    print(f"Mean difference (A@I - A): {diff}")
    assert diff < 1e-5
    
    # 4. Test GPU ReLU
    print("\nTesting GPU ReLU...")
    x = rpl.Tensor([-1.0, 0.5, -2.0, 3.0])
    x.to_gpu()
    y = x.relu()
    y.to_cpu()
    print(f"ReLU input: [-1.0, 0.5, -2.0, 3.0]")
    print(f"ReLU output: {y.data}")
    assert y.data[0] == 0.0
    assert y.data[3] == 3.0

    print("\nAll Python GPU tests passed!")

if __name__ == "__main__":
    test_gpu_integration()
