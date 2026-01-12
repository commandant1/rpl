import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../python"))

import rpl
import numpy as np

# A simple MLP training example in Python
def main():
    print("--- RPiTorch Python MLP Example ---")
    
    # Generate some synthetic data for XOR
    X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    # Hyperparameters
    lr = 0.1
    epochs = 1000
    
    # Model components
    fc1 = rpl.nn.Linear(2, 4)
    relu = rpl.nn.ReLU()
    fc2 = rpl.nn.Linear(4, 1)
    sigmoid = rpl.nn.Sigmoid()
    
    # Optimizer
    optimizer = rpl.optim.SGD([fc1.weight, fc1.bias, fc2.weight, fc2.bias], lr=lr)
    
    print("\nStarting training...")
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(X_data)):
            # 1. Prepare data
            x = rpl.Tensor([X_data[i]], requires_grad=False)
            target = rpl.Tensor([y_data[i]], requires_grad=False)
            
            # 2. Forward pass
            h1 = relu(fc1(x))
            out = sigmoid(fc2(h1))
            
            # 3. Compute loss
            loss = rpl.nn.mse_loss(out, target)
            epoch_loss += loss.data[0]
            
            # 4. Backward pass (Note: Custom autograd might need more integration)
            # For now, we manually zero and step if needed, or let the C engine handle it
            # Since rpl.Tensor.backward() is called on the output
            loss.backward()
            
            # 5. Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(X_data):.6f}")
            
    print("\nFinal Predictions:")
    for i in range(len(X_data)):
        x = rpl.Tensor([X_data[i]])
        h1 = relu(fc1(x))
        out = sigmoid(fc2(h1))
        print(f"In: {X_data[i]} -> Pred: {out.data[0][0]:.4f} (Target: {y_data[i][0]})")

if __name__ == "__main__":
    main()
