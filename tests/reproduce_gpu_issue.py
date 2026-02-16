import rpl
import numpy as np

def test_gpu_sync():
    print("Creating tensor on CPU...")
    t1 = rpl.Tensor([1.0, 2.0, 3.0, 4.0])
    print(f"Initial device: {t1.device} (Expect 0)")
    
    # Moving to GPU
    print("\nMoving to GPU...")
    t1.to_gpu()
    print(f"Device after to_gpu: {t1.device} (Expect 1)")
    
    # Check if data is preserved (implicitly uses from_gpu if device=1)
    # Actually repr or element access should trigger sync if needed
    print(f"Tensor data: {t1}")
    
    # Moving back to CPU
    print("\nMoving back to CPU...")
    t1.to_cpu()
    print(f"Device after to_cpu: {t1.device} (Expect 0)")
    
    if t1.device != 0:
        print("FAIL: Device not updated to CPU")
    else:
        print("SUCCESS: Device updated to CPU")

if __name__ == "__main__":
    try:
        test_gpu_sync()
    except Exception as e:
        print(f"Error: {e}")
