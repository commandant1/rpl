# RPL Python API Reference

The RPL Python bindings provide a high-level interface to the underlying C core using `ctypes`. The API is designed to feel familiar to users of PyTorch.

## Core Module: `rpl`

### `rpl.Tensor`
The fundamental data structure.

```python
# Create from data
t = rpl.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

# Create from shape
t = rpl.Tensor(shape=(10, 10), requires_grad=False)

# Properties
print(t.shape) # Tuple
print(t.data)  # Numpy array (view)
print(t.grad)  # Numpy array (view or None)

# Operations
a = rpl.Tensor([1.0, 2.0])
b = rpl.Tensor([3.0, 4.0])
c = a + b
d = a @ b.T # Matrix multiplication
```

### Autograd
```python
loss.backward()  # Computes gradients for all tensors in the graph
t.zero_grad()    # Resets gradients
```

## Neural Network Module: `rpl.nn`

### Layers
- `nn.Linear(in_features, out_features)`
- `nn.ReLU()`
- `nn.Sigmoid()`

### Loss Functions
- `nn.MSELoss()`

### Example: MLP
```python
import rpl.nn as nn

class MLP:
    def __init__(self):
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 1)
        
    def __call__(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
```

## Optimizer Module: `rpl.optim`

### `rpl.optim.SGD`
```python
optimizer = rpl.optim.SGD(params=[w1, b1, w2, b2], lr=0.01, momentum=0.9)

optimizer.zero_grad()
# ... backward pass ...
optimizer.step()
```

## Data Module: `rpl.data`

### `rpl.data.DataLoader`
Iterates over datasets in batches.

```python
dataset = rpl.data.TensorDataset(x_train, y_train)
loader = rpl.data.DataLoader(dataset, batch_size=32, shuffle=True)

for x_batch, y_batch in loader:
    # training step
```

## Best Practices

1. **Memory Management**: RPL Python objects wrap C pointers. Memory is automatically freed when the Python object is garbage collected.
2. **Numpy Interop**: Use `.data` and `.grad` to get direct views into the tensor memory as Numpy arrays. Modifications to these arrays will affect the tensor data.
3. **Broadcasting**: The C core supports limited broadcasting (e.g., adding a bias vector to a batch of activations).
