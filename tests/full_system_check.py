import rpl
import time
import numpy as np
import sys

def print_header(msg):
    print(f"\n{'='*60}")
    print(f" {msg}")
    print(f"{'='*60}")

def run_test(name, func):
    print(f"Running {name}...", end=" ", flush=True)
    try:
        start = time.time()
        func()
        end = time.time()
        print(f"PASSED ({end-start:.4f}s)")
        return True
    except Exception as e:
        print(f"FAILED")
        print(f"Error: {e}")
        return False

def test_cpu_basics():
    # Test basic tensor creation and CPU operations (uses NEON if compiled)
    t = rpl.Tensor([1.0, 2.0, 3.0, 4.0])
    assert t.shape == (4,)
    
    # Add
    t2 = t + t
    assert t2.data[0] == 2.0
    
    # Matmul (CPU)
    A = rpl.Tensor(np.eye(128).astype(np.float32))
    B = rpl.Tensor(np.ones((128, 128)).astype(np.float32))
    C = A @ B
    assert np.allclose(C.data, B.data)

def test_gpu_transfer():
    t = rpl.Tensor(np.random.randn(1024).astype(np.float32))
    orig_data = t.data.copy()
    
    # CPU -> GPU
    t.to_gpu()
    assert t.device == 1 # Device.GPU
    
    # GPU -> CPU
    t.to_cpu()
    assert t.device == 0 # Device.CPU
    
    # Verify data integrity
    assert np.allclose(t.data, orig_data)

def test_gpu_math():
    size = 1024*1024
    a = rpl.Tensor(np.ones(size).astype(np.float32))
    b = rpl.Tensor(np.ones(size).astype(np.float32) * 2.0)
    
    a.to_gpu()
    b.to_gpu()
    
    # Vector Add
    c = a + b
    c.to_cpu()
    assert c.data[0] == 3.0
    assert c.data[-1] == 3.0

def test_gpu_gemm():
    M, K, N = 256, 256, 256
    A = rpl.Tensor(np.random.randn(M, K).astype(np.float32))
    B = rpl.Tensor(np.eye(K, N).astype(np.float32))
    
    A.to_gpu()
    B.to_gpu()
    
    C = A @ B # Should dispatch to GPU GEMM
    C.to_cpu()
    
    # Tolerance might need to be looser for GPU float32
    diff = np.abs(C.data - A.data).mean()
    if diff > 1e-4:
        raise ValueError(f"GEMM Error too high: {diff}")

def test_gpu_activations():
    x = rpl.Tensor(np.linspace(-3, 3, 1000).astype(np.float32))
    x.to_gpu()
    
    # ReLU
    y_relu = x.relu()
    y_relu.to_cpu()
    assert y_relu.data[0] == 0.0 # -3 -> 0
    assert y_relu.data[-1] == 3.0 # 3 -> 3
    
    # Tanh
    y_tanh = x.tanh()
    y_tanh.to_cpu()
    x.to_cpu()
    ref_tanh = np.tanh(x.data)
    assert np.allclose(y_tanh.data, ref_tanh, atol=1e-3)
    
    # GELU
    x.to_gpu()
    y_gelu = x.gelu()
    if y_gelu:
        y_gelu.to_cpu()
        # Just check it runs and produces reasonable output
        assert not np.isnan(y_gelu.data).any()

def main():
    print_header("RPL Full System Verification")
    print(f"Library: {rpl.__file__}")
    
    tests = [
        ("CPU Basics (NEON)", test_cpu_basics),
        ("GPU Data Transfer", test_gpu_transfer),
        ("GPU Vector Math", test_gpu_math),
        ("GPU Matrix Mult (GEMM)", test_gpu_gemm),
        ("GPU Activations", test_gpu_activations),
    ]
    
    passed = 0
    for name, func in tests:
        if run_test(name, func):
            passed += 1
            
    print_header(f"Summary: {passed}/{len(tests)} Tests Passed")
    
    if passed == len(tests):
        print("✅ SUCCESS: System is ready for use.")
        sys.exit(0)
    else:
        print("❌ FAILURE: Some tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
