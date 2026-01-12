/*
 * train_xor.c
 * 
 * Demonstrates training a simple neural network to solve XOR in pure C.
 * Uses:
 *  - Tensors (input, weights, output)
 *  - Autograd (backward propagation)
 *  - MSE Loss
 *  - Simple SGD optimization loop
 */

#include "rpl.h"
#include <stdio.h>
#include <math.h>

#define MAX_EPOCHS 5000
#define LEARNING_RATE 0.1f

int main() {
    printf("======================================\n");
    printf(" RPL Training Example: XOR Problem\n");
    printf("======================================\n");

    // 1. Data Setup (XOR)
    // Inputs: [0,0], [0,1], [1,0], [1,1]
    uint32_t x_shape[] = {4, 2};
    Tensor* X = tensor_create(2, x_shape, false);
    float x_data[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    tensor_fill_buffer(X->data, 0.0f, 8); // Just init, redundant with memcpy
    // Quick fill
    for(int i=0; i<8; i++) X->data[i] = x_data[i];

    // Targets: [0], [1], [1], [0]
    uint32_t y_shape[] = {4, 1};
    Tensor* Y = tensor_create(2, y_shape, false);
    float y_data[] = {
        0.0f,
        1.0f,
        1.0f, 
        0.0f
    };
    for(int i=0; i<4; i++) Y->data[i] = y_data[i];
    
    // 2. Model Setup (MLP: 2 -> 4 -> 1)
    // Layer 1: Linear(2->4) + Bias
    uint32_t w1_shape[] = {2, 4}; // [in, hidden] - but logic might expect [hidden, in]? 
    // RPL core 'backward_matmul' logic and 'linear_forward' in rpl_nn (which is not included in core)
    // Let's use low-level Ops manually to see autograd in action.
    
    // MatMul convention: X [Batch, In] @ W [In, Out] -> [Batch, Out]
    // Or X [Batch, In] @ W^T [Out, In] -> [Batch, Out]
    
    // Let's create Weight 1: [2, 4]
    Tensor* W1 = tensor_create(2, w1_shape, true);
    tensor_randomize(W1); // Random init
    
    // Bias 1: [4] (broadcast? rpl seems basic, assume [1,4] or handled)
    // For simplicity, let's skip bias or manual add. Let's do simple X @ W1 first.
    
    uint32_t w2_shape[] = {4, 1};
    Tensor* W2 = tensor_create(2, w2_shape, true);
    tensor_randomize(W2);

    // 3. Training Loop
    printf("Training for %d epochs...\n", MAX_EPOCHS);
    
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        // --- Forward Pass ---
        
        // H1 = X @ W1
        Tensor* H1 = tensor_matmul(X, W1);
        
        // A1 = ReLU(H1)
        Tensor* A1 = tensor_relu(H1);
        
        // Output = A1 @ W2
        Tensor* Pred = tensor_matmul(A1, W2);
        
        // Loss = MSE(Pred, Y)
        Tensor* Loss = tensor_mse_loss(Pred, Y); // Returns tensor with gradient enabled
        
        // --- Backward Pass ---
        // Zero grads before backward
        tensor_zero_grad(W1);
        tensor_zero_grad(W2);
        
        // Init grad at loss
        tensor_fill_buffer(Loss->grad, 1.0f, 1);
        
        // Propagate
        tensor_backward(Loss);
        
        // --- Simple SGD Update ---
        // W1 = W1 - lr * W1.grad
        for(uint32_t i=0; i<W1->size; i++) {
            W1->data[i] -= LEARNING_RATE * W1->grad[i];
        }
        for(uint32_t i=0; i<W2->size; i++) {
            W2->data[i] -= LEARNING_RATE * W2->grad[i];
        }
        
        if (epoch % 500 == 0) {
            printf("Epoch %d: Loss = %.6f\n", epoch, Loss->data[0]);
        }
        
        // Cleanup intermediate tensors (graph)
        // In full framework, memory pool handles this.
        // For C manual graph, we must free local non-parameter tensors from this iteration
        // However, tensor_free might try to free parents? 
        // RPL's tensor_free is simple, just frees data.
        // We need to keep params (W1, W2, X, Y). Free H1, A1, Pred, Loss.
        // Important: free in reverse topological order? Or just free.
        tensor_free(Loss);
        tensor_free(Pred);
        tensor_free(A1);
        tensor_free(H1);
    }

    // 4. Verification
    printf("\nTesting Trained Model:\n");
    Tensor* H1 = tensor_matmul(X, W1);
    Tensor* A1 = tensor_relu(H1);
    Tensor* Pred = tensor_matmul(A1, W2);
    
    for(int i=0; i<4; i++) {
        printf("Input: [%.0f, %.0f] -> Pred: %.4f (Target: %.0f)\n",
               X->data[i*2], X->data[i*2+1], Pred->data[i], Y->data[i]);
    }
    
    // Cleanup
    tensor_free(Pred);
    tensor_free(A1);
    tensor_free(H1);
    tensor_free(W1);
    tensor_free(W2);
    tensor_free(X);
    tensor_free(Y);

    return 0;
}
