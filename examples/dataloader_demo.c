/*
 * Advanced Data Loading Example
 * Demonstrates: Prefetching, Augmentation, Batch Processing
 */

#include "rpitorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void generate_synthetic_dataset(Tensor** X, Tensor** y, uint32_t n_samples) {
    uint32_t X_shape[4] = {n_samples, 3, 32, 32};  // NCHW format
    uint32_t y_shape[2] = {n_samples, 10};
    
    *X = tensor_create(4, X_shape, false);
    *y = tensor_create(2, y_shape, false);
    
    // Generate random data
    for (uint32_t i = 0; i < (*X)->size; i++) {
        (*X)->data[i] = (float)rand() / RAND_MAX;
    }
    
    for (uint32_t i = 0; i < n_samples; i++) {
        uint32_t label = rand() % 10;
        for (uint32_t j = 0; j < 10; j++) {
            (*y)->data[i * 10 + j] = (j == label) ? 1.0f : 0.0f;
        }
    }
}

int main() {
    srand(42);
    
    printf("================================================================================\n");
    printf("RPiTorch: Advanced Data Loading Demo\n");
    printf("================================================================================\n");
    
    // Create dataset
    printf("\n[1] Creating synthetic dataset...\n");
    const uint32_t n_samples = 10000;
    Tensor *X_train, *y_train;
    
    generate_synthetic_dataset(&X_train, &y_train, n_samples);
    printf("  ✓ Generated %u samples (3×32×32 images)\n", n_samples);
    
    // Create TensorDataset
    printf("\n[2] Creating TensorDataset...\n");
    Dataset* dataset = tensor_dataset_create(X_train, y_train);
    printf("  ✓ Dataset created with %u samples\n", dataset->num_samples);
    printf("  ✓ Sample shape: ");
    for (uint32_t i = 0; i < dataset->sample_dims; i++) {
        printf("%u ", dataset->sample_shape[i]);
    }
    printf("\n");
    
    // Create DataLoader with prefetching
    printf("\n[3] Creating DataLoader...\n");
    const uint32_t batch_size = 64;
    const uint32_t num_workers = 2;
    
    DataLoader* loader = dataloader_create(
        dataset,
        batch_size,
        true,   // shuffle
        false,  // drop_last
        num_workers
    );
    printf("  ✓ Batch size: %u\n", batch_size);
    printf("  ✓ Num workers: %u (prefetching enabled)\n", num_workers);
    printf("  ✓ Shuffle: enabled\n");
    printf("  ✓ Num batches: %u\n", loader->num_batches);
    
    // Setup data augmentation
    printf("\n[4] Configuring data augmentation...\n");
    AugmentationConfig aug_config = {
        .random_flip_h = true,
        .random_flip_v = false,
        .brightness_delta = 0.2f,
        .contrast_delta = 0.0f,
        .noise_std = 0.01f
    };
    dataloader_set_augmentation(loader, &aug_config);
    printf("  ✓ Random horizontal flip: enabled\n");
    printf("  ✓ Brightness adjustment: ±%.1f%%\n", aug_config.brightness_delta * 100);
    printf("  ✓ Gaussian noise: std=%.3f\n", aug_config.noise_std);
    
    // Benchmark data loading
    printf("\n[5] Benchmarking data loading...\n");
    printf("--------------------------------------------------------------------------------\n");
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    dataloader_start(loader);
    
    uint32_t batch_count = 0;
    uint64_t total_samples = 0;
    
    Tensor *batch_X, *batch_y;
    while (dataloader_next(loader, &batch_X, &batch_y)) {
        batch_count++;
        total_samples += batch_X->shape[0];
        
        // Simulate training step
        // (In real training, you would do forward/backward here)
        
        // Free batch tensors
        tensor_free(batch_X);
        tensor_free(batch_y);
        
        if (batch_count % 50 == 0) {
            printf("  Processed %u batches (%lu samples)...\n", 
                   batch_count, total_samples);
        }
    }
    
    dataloader_stop(loader);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + 
                    (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("--------------------------------------------------------------------------------\n");
    printf("\n[6] Performance Summary\n");
    printf("  Total batches: %u\n", batch_count);
    printf("  Total samples: %lu\n", total_samples);
    printf("  Time elapsed: %.3f seconds\n", elapsed);
    printf("  Throughput: %.1f samples/sec\n", total_samples / elapsed);
    printf("  Throughput: %.1f batches/sec\n", batch_count / elapsed);
    
    // Memory efficiency
    size_t dataset_memory = n_samples * (3 * 32 * 32 + 10) * sizeof(float);
    printf("\n[7] Memory Efficiency\n");
    printf("  Dataset size: %.1f MB\n", dataset_memory / (1024.0 * 1024.0));
    printf("  Batch size: %.1f KB\n", 
           (batch_size * (3 * 32 * 32 + 10) * sizeof(float)) / 1024.0);
    printf("  Prefetch queue: %u batches\n", MAX_PREFETCH_BATCHES);
    
    // Cleanup
    printf("\n[8] Cleanup...\n");
    dataloader_free(loader);
    dataset->free_fn(dataset);
    tensor_free(X_train);
    tensor_free(y_train);
    
    printf("\n================================================================================\n");
    printf("✅ Data Loading Demo Complete!\n");
    printf("================================================================================\n");
    printf("\nFeatures Demonstrated:\n");
    printf("  ✓ TensorDataset (in-memory)\n");
    printf("  ✓ DataLoader with multi-threaded prefetching\n");
    printf("  ✓ Batch shuffling\n");
    printf("  ✓ Data augmentation (NEON-optimized)\n");
    printf("  ✓ Efficient batch processing\n");
    printf("  ✓ Zero-copy memory operations\n");
    printf("================================================================================\n");
    
    return 0;
}
