import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

import rpl
import numpy as np

def check(name, condition, detail=""):
    if condition:
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name} {detail}")
        sys.exit(1)

# 1. Diamond Graph Autograd Correctness
print("--- Diamond Graph Check ---")
x = rpl.Tensor([3.0], requires_grad=True)
y = x + x
y.backward()
check("y = x + x -> dy/dx == 2.0", abs(x.grad[0] - 2.0) < 1e-5, f"got {x.grad[0]}")

# 2. Deep Graph Stack Overflow Check
print("--- Deep Graph Check ---")
z = rpl.Tensor([1.0], requires_grad=True)
curr = z
for _ in range(200):
    curr = curr * 1.01
curr.backward()
check("200-layer backward runs without stack overflow", True)

# 3. Parameter Collecting
print("--- Module Parameter Collecting ---")
class SimpleMLP(rpl.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = rpl.nn.Linear(10, 20)
        self.fc2 = rpl.nn.Linear(20, 5)
        self.drop = rpl.nn.Dropout(0.1)

    def forward(self, x):
        return self.fc2(self.drop(self.fc1(x).relu()))

model = SimpleMLP()
params = model.parameters()
check("model.parameters() collects all weights & biases", len(params) == 4)

# 4. Sequential
print("--- Sequential Check ---")
seq = rpl.nn.Sequential(
    rpl.nn.Linear(10, 20),
    rpl.nn.ReLU(),
    rpl.nn.Linear(20, 5)
)
params_seq = seq.parameters()
check("seq.parameters() collects weights & biases", len(params_seq) == 4)

# 5. Optimizers
print("--- Optimizers Check ---")
optimizer = rpl.optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()
x = rpl.randn(2, 10)
y = model(x)
loss = (y * y).sum(dim=1).sum(dim=0)
loss.backward()
optimizer.step()
check("Adam step runs successfully", True)

optimizerW = rpl.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
optimizerW.zero_grad()
loss.backward()
optimizerW.step()
check("AdamW step runs successfully", True)

optimizerLAMB = rpl.optim.LAMB(model.parameters(), lr=0.001, weight_decay=0.01)
optimizerLAMB.zero_grad()
loss.backward()
optimizerLAMB.step()
check("LAMB step runs successfully", True)

# 6. Loss Differentiability
print("--- Loss Differentiability Check ---")
pred = rpl.Tensor([[0.5, 0.5]], requires_grad=True)
target = rpl.Tensor([[1.0, 0.0]])
ce = rpl.nn.CrossEntropyLoss()
loss_val = ce(pred, target)
loss_val.backward()
check("CrossEntropyLoss computes gradient", pred.grad is not None and not np.isnan(pred.grad).any())

# 7. DataLoader Iterator protocol
print("--- DataLoader Check ---")
x_data = rpl.randn(10, 5)
y_data = rpl.randn(10, 1)
dataset = rpl.data.TensorDataset(x_data, y_data)
dataloader = rpl.data.DataLoader(dataset, batch_size=2, shuffle=True)

batches_count = 0
for bx, by in dataloader:
    check("DataLoader batch shapes", bx.shape == (2, 5) and by.shape == (2, 1))
    batches_count += 1
check("DataLoader iterated correct number of batches", batches_count == 5)

print("\n=== All new components test cases passed! ===")
