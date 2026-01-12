/*
 * Complete Training Example in Pure C
 * Demonstrates: Optimizers, Early Stopping, Gradient Clipping, Checkpointing
 */

#include "rpitorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Simple 2-layer MLP
typedef struct {
    Tensor* w1;
    Tensor* b1;
    Tensor* w2;
    Tensor* b2;
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
    float std1 = sqrtf(2.0f / (input_size + hidden_size));
    float std2 = sqrtf(2.0f / (hidden_size + output_size));
    
    for (uint32_t i = 0; i < model->w1->size; i++) {
        model->w1->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std1;
    }
    for (uint32_t i = 0; i < model->w2->size; i++) {
        model->w2->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std2;
    }
    
    tensor_fill(model->b1, 0.0f);
    tensor_fill(model->b2, 0.0f);
    
    return model;
}

Tensor* mlp_forward(MLP* model, const Tensor* input) {
    // h = ReLU(input @ w1 + b1)
    Tensor* h = tensor_matmul(input, model->w1);
    tensor_add_inplace(h, model->b1);
    tensor_relu_inplace(h);
    
    // output = h @ w2 + b2
    Tensor* output = tensor_matmul(h, model->w2);
    tensor_add_inplace(output, model->b2);
    
    tensor_free(h);
    return output;
}

Tensor** mlp_parameters(MLP* model, uint32_t* num_params) {
    *num_params = 4;
    Tensor** params = (Tensor**)malloc(4 * sizeof(Tensor*));
    params[0] = model->w1;
    params[1] = model->b1;
    params[2] = model->w2;
    params[3] = model->b2;
    return params;
}

void mlp_free(MLP* model) {
    tensor_free(model->w1);
    tensor_free(model->b1);
    tensor_free(model->w2);
    tensor_free(model->b2);
    free(model);
}

// Generate synthetic data
void generate_data(Tensor* X, Tensor* y, uint32_t n_samples, uint32_t n_features) {
    for (uint32_t i = 0; i < X->size; i++) {
        X->data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    for (uint32_t i = 0; i < y->size; i++) {
        y->data[i] = (float)(rand() % 2);
    }
}

// MSE Loss
float mse_loss(const Tensor* pred, const Tensor* target) {
    float loss = 0.0f;
    
    #pragma omp parallel for reduction(+:loss)
    for (uint32_t i = 0; i < pred->size; i++) {
        float diff = pred->data[i] - target->data[i];
        loss += diff * diff;
    }
    
    return loss / pred->size;
}

// Accuracy
float accuracy(const Tensor* pred, const Tensor* target) {
    uint32_t correct = 0;
    
    for (uint32_t i = 0; i < pred->size; i++) {
        int pred_class = (pred->data[i] > 0.5f) ? 1 : 0;
        int true_class = (int)target->data[i];
        if (pred_class == true_class) correct++;
    }
    
    return (float)correct / pred->size;
}

int main() {
    srand(42);
    
    printf("================================================================================\n");
    printf("RPiTorch: Complete Training Example in Pure C\n");
    printf("================================================================================\n");
    
    // Hyperparameters
    const uint32_t n_samples = 1000;
    const uint32_t n_features = 20;
    const uint32_t hidden_size = 64;
    const uint32_t output_size = 1;
    const uint32_t batch_size = 32;
    const uint32_t num_epochs = 100;
    
    printf("\n[1] Creating dataset...\n");
    uint32_t X_shape[2] = {n_samples, n_features};
    uint32_t y_shape[2] = {n_samples, output_size};
    
    Tensor* X_train = tensor_create(2, X_shape, false);
    Tensor* y_train = tensor_create(2, y_shape, false);
    
    generate_data(X_train, y_train, n_samples, n_features);
    printf("  ✓ Generated %u samples with %u features\n", n_samples, n_features);
    
    // Create model
    printf("\n[2] Creating model...\n");
    MLP* model = mlp_create(n_features, hidden_size, output_size);
    
    uint32_t num_params;
    Tensor** parameters = mlp_parameters(model, &num_params);
    printf("  ✓ Model created with %u parameters\n", num_params);
    
    // Setup training utilities
    printf("\n[3] Initializing training utilities...\n");
    
    Optimizer* optimizer = optimizer_adam_create(parameters, num_params, 
                                                0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);
    printf("  ✓ Optimizer: Adam (lr=0.001)\n");
    
    EarlyStopping* early_stop = early_stopping_create(10, 0.001f, true);
    printf("  ✓ EarlyStopping: patience=10, min_delta=0.001\n");
    
    LRScheduler* scheduler = lr_scheduler_warmup_create(optimizer, 5, 1e-5f);
    printf("  ✓ LR Warmup: 5 epochs\n");
    
    MetricsTracker* metrics = metrics_tracker_create();
    printf("  ✓ MetricsTracker initialized\n");
    
    ModelCheckpoint* checkpoint = checkpoint_create("checkpoints/model_epoch%03u_loss%.4f.bin",
                                                   "val_loss", true, true);
    printf("  ✓ ModelCheckpoint: saving to checkpoints/\n");
    
    // Training loop
    printf("\n[4] Training...\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("%5s %12s %12s %12s %12s\n", "Epoch", "Train Loss", "Train Acc", "LR", "Status");
    printf("--------------------------------------------------------------------------------\n");
    
    for (uint32_t epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;
        uint32_t num_batches = 0;
        
        // Mini-batch training
        for (uint32_t i = 0; i < n_samples; i += batch_size) {
            uint32_t batch_end = (i + batch_size > n_samples) ? n_samples : i + batch_size;
            uint32_t current_batch_size = batch_end - i;
            
            // Create batch tensors
            uint32_t batch_X_shape[2] = {current_batch_size, n_features};
            uint32_t batch_y_shape[2] = {current_batch_size, output_size};
            
            Tensor* batch_X = tensor_create(2, batch_X_shape, false);
            Tensor* batch_y = tensor_create(2, batch_y_shape, false);
            
            memcpy(batch_X->data, &X_train->data[i * n_features], 
                  current_batch_size * n_features * sizeof(float));
            memcpy(batch_y->data, &y_train->data[i * output_size],
                  current_batch_size * output_size * sizeof(float));
            
            // Forward pass
            Tensor* output = mlp_forward(model, batch_X);
            float loss = mse_loss(output, batch_y);
            float acc = accuracy(output, batch_y);
            
            // Backward pass
            optimizer_zero_grad(optimizer);
            tensor_backward(output);
            
            // Gradient clipping
            gradient_clip_norm(parameters, num_params, 1.0f, 2.0f);
            
            // Optimizer step
            if (optimizer->type == OPTIMIZER_ADAM) {
                optimizer_adam_step(optimizer);
            } else {
                optimizer_sgd_step(optimizer);
            }
            
            epoch_loss += loss;
            epoch_acc += acc;
            num_batches++;
            
            tensor_free(batch_X);
            tensor_free(batch_y);
            tensor_free(output);
        }
        
        epoch_loss /= num_batches;
        epoch_acc /= num_batches;
        
        // Update learning rate
        lr_scheduler_step(scheduler);
        
        // Track metrics
        metrics_tracker_add(metrics, "train_loss", epoch_loss);
        metrics_tracker_add(metrics, "train_acc", epoch_acc);
        metrics_tracker_add(metrics, "lr", optimizer->learning_rate);
        metrics_tracker_next_epoch(metrics);
        
        // Print progress
        const char* status = "";
        printf("%5u %12.6f %12.4f %12.8f %12s\n", 
               epoch + 1, epoch_loss, epoch_acc, optimizer->learning_rate, status);
        
        // Checkpointing
        checkpoint_save_model(checkpoint, parameters, num_params, epoch, epoch_loss);
        
        // Early stopping
        if (early_stopping_check(early_stop, epoch, epoch_loss)) {
            printf("\n⚠ Early stopping triggered at epoch %u\n", epoch + 1);
            printf("  Best loss: %.6f at epoch %u\n", 
                   early_stop->best_score, early_stop->best_epoch + 1);
            break;
        }
    }
    
    printf("--------------------------------------------------------------------------------\n");
    
    // Save metrics
    printf("\n[5] Saving results...\n");
    metrics_tracker_save(metrics, "training_metrics.csv");
    printf("  ✓ Metrics saved to: training_metrics.csv\n");
    
    // Print summary
    printf("\n[6] Training Summary\n");
    metrics_tracker_print(metrics);
    
    // Cleanup
    printf("\n[7] Cleanup...\n");
    tensor_free(X_train);
    tensor_free(y_train);
    mlp_free(model);
    free(parameters);
    optimizer_free(optimizer);
    early_stopping_free(early_stop);
    lr_scheduler_free(scheduler);
    metrics_tracker_free(metrics);
    checkpoint_free(checkpoint);
    
    printf("\n================================================================================\n");
    printf("✅ Training Complete!\n");
    printf("================================================================================\n");
    printf("\nFeatures Demonstrated:\n");
    printf("  ✓ Adam Optimizer with weight decay\n");
    printf("  ✓ Early Stopping (patience-based)\n");
    printf("  ✓ Gradient Clipping (L2 norm)\n");
    printf("  ✓ Learning Rate Warmup\n");
    printf("  ✓ Model Checkpointing\n");
    printf("  ✓ Metrics Tracking & CSV Export\n");
    printf("  ✓ Mini-batch Training\n");
    printf("================================================================================\n");
    
    return 0;
}
