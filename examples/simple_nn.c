/*
 * Simple Neural Network in Pure C
 * 2-layer MLP for demonstration
 */

#include "rpi_ml.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    Tensor* w1;  // Input to hidden weights
    Tensor* b1;  // Hidden bias
    Tensor* w2;  // Hidden to output weights
    Tensor* b2;  // Output bias
} MLP;

MLP* mlp_create(uint32_t input_size, uint32_t hidden_size, uint32_t output_size) {
    MLP* model = (MLP*)malloc(sizeof(MLP));
    
    uint32_t w1_shape[2] = {input_size, hidden_size};
    uint32_t b1_shape[1] = {hidden_size};
    uint32_t w2_shape[2] = {hidden_size, output_size};
    uint32_t b2_shape[1] = {output_size};
    
    model->w1 = tensor_create(2, w1_shape, true);
    model->b1 = tensor_create(1, b1_shape, true);
    model->w2 = tensor_create(2, w2_shape, true);
    model->b2 = tensor_create(1, b2_shape, true);
    
    // Xavier initialization
    tensor_randomize(model->w1);
    tensor_randomize(model->w2);
    tensor_fill(model->b1, 0.0f);
    tensor_fill(model->b2, 0.0f);
    
    return model;
}

void mlp_free(MLP* model) {
    tensor_free(model->w1);
    tensor_free(model->b1);
    tensor_free(model->w2);
    tensor_free(model->b2);
    free(model);
}

Tensor* mlp_forward(MLP* model, Tensor* input, Tensor** intermediates) {
    // Layer 1: h = relu(input @ w1 + b1)
    intermediates[0] = tensor_matmul(input, model->w1);
    intermediates[1] = tensor_add(intermediates[0], model->b1);
    intermediates[2] = tensor_relu(intermediates[1]);
    
    // Layer 2: output = h @ w2 + b2
    intermediates[3] = tensor_matmul(intermediates[2], model->w2);
    Tensor* output = tensor_add(intermediates[3], model->b2);
    
    return output;
}

void sgd_step(MLP* model, float lr) {
    Tensor* params[] = {model->w1, model->b1, model->w2, model->b2};
    
    for (int i = 0; i < 4; i++) {
        if (params[i]->grad) {
            for (uint32_t j = 0; j < params[i]->size; j++) {
                params[i]->data[j] -= lr * params[i]->grad[j];
            }
        }
    }
}

void zero_grad(MLP* model) {
    tensor_zero_grad(model->w1);
    tensor_zero_grad(model->b1);
    tensor_zero_grad(model->w2);
    tensor_zero_grad(model->b2);
}

int main() {
    printf("===========================================\n");
    printf("Simple Neural Network in Pure C\n");
    printf("===========================================\n\n");
    
    // Create model
    uint32_t input_size = 10;
    uint32_t hidden_size = 20;
    uint32_t output_size = 5;
    
    MLP* model = mlp_create(input_size, hidden_size, output_size);
    printf("✓ Created MLP: %d → %d → %d\n", input_size, hidden_size, output_size);
    
    // Create training data
    uint32_t batch_size = 32;
    uint32_t x_shape[2] = {batch_size, input_size};
    uint32_t y_shape[2] = {batch_size, output_size};
    
    Tensor* x_train = tensor_create(2, x_shape, false);
    Tensor* y_train = tensor_create(2, y_shape, false);
    
    tensor_randomize(x_train);
    tensor_randomize(y_train);
    
    printf("✓ Generated training data: %d samples\n\n", batch_size);
    
    // Training loop
    float lr = 0.01f;
    int num_epochs = 10;
    
    printf("Training for %d epochs...\n", num_epochs);
    printf("%-8s %-12s\n", "Epoch", "Loss");
    printf("----------------------------------------\n");
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Forward pass
        Tensor* intermediates[4];
        Tensor* output = mlp_forward(model, x_train, intermediates);
        
        // Compute MSE loss
        float loss = 0.0f;
        for (uint32_t i = 0; i < output->size; i++) {
            float diff = output->data[i] - y_train->data[i];
            loss += diff * diff;
        }
        loss /= output->size;
        
        printf("%-8d %-12.6f\n", epoch + 1, loss);
        
        // Backward pass
        zero_grad(model);
        tensor_backward(output);
        
        // Update weights
        sgd_step(model, lr);
        
        // Cleanup all tensors in this graph iteration
        tensor_free(output);
        for (int i = 0; i < 4; i++) tensor_free(intermediates[i]);
    }
    
    printf("\n===========================================\n");
    printf("Training completed!\n");
    printf("===========================================\n");
    
    // Cleanup
    mlp_free(model);
    tensor_free(x_train);
    tensor_free(y_train);
    
    return 0;
}
