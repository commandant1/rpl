import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../python"))

import rpl
import numpy as np

def test_tensor():
    print("Testing Tensor...")
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    t = rpl.Tensor(data)
    print(f"Shape: {t.shape}")
    print(f"Data:\n{t.data}")
    assert t.shape == (2, 3)
    assert np.allclose(t.data, data)
    print("Tensor test: PASS")

def test_linear():
    print("\nTesting Linear Layer...")
    fc = rpl.nn.Linear(3, 2)
    x = rpl.Tensor([[1.0, 1.0, 1.0]])
    print(f"Weight data:\n{fc.weight.data}")
    print(f"Bias data:\n{fc.bias.data}")
    
    # Manual calculation check
    # out = x @ W^T + b
    # W is [2, 3], b is [2]
    expected = np.matmul(x.data, fc.weight.data.T) + fc.bias.data
    
    y = fc(x)
    print(f"Input after forward:\n{x.data}")
    print(f"Output shape: {y.shape}")
    print(f"Output data:\n{y.data}")
    print(f"Expected data:\n{expected}")
    
    assert y.shape == (1, 2)
    assert np.allclose(y.data, expected, atol=1e-5)
    print("Linear test: PASS")

if __name__ == "__main__":
    try:
        test_tensor()
        test_linear()
        print("\nAll Python binding tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
