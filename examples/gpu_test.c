#include "rpl.h"
#include <stdio.h>
#include <assert.h>

int main() {
    printf("Testing RPL GPU Backend...\n");

    if (!rpl_gpu_init()) {
        printf("GPU not available or failed to initialize.\n");
        return 1;
    }

    uint32_t shape[] = {1024};
    Tensor* t = tensor_create(1, shape, false);
    
    // Fill with data
    for (int i = 0; i < 1024; i++) {
        t->data[i] = (float)i;
    }

    printf("Original data (first 5): %f %f %f %f %f\n", 
           t->data[0], t->data[1], t->data[2], t->data[3], t->data[4]);

    // Move to GPU
    printf("Moving to GPU...\n");
    tensor_to_gpu(t);
    
    // Check if it's marked as on GPU
    if (t->device == DEVICE_GPU) {
        printf("Tensor is now on GPU (Buffer ID: %u)\n", t->gpu_buffer);
    } else {
        printf("Failed to move tensor to GPU\n");
        return 1;
    }

    // Clear CPU data to prove we download it back
    // (Note: In a shared memory system like Pi this might not be strictly necessary if mapped, 
    // but good for verification)
    for (int i = 0; i < 1024; i++) {
        t->data[i] = 0.0f;
    }

    // Move back to CPU
    printf("Moving back from GPU...\n");
    tensor_from_gpu(t);

    printf("Restored data (first 5): %f %f %f %f %f\n", 
           t->data[0], t->data[1], t->data[2], t->data[3], t->data[4]);

    // Verify
    for (int i = 0; i < 1024; i++) {
        if (t->data[i] != (float)i) {
            printf("Mismatch at index %d: expected %f, got %f\n", i, (float)i, t->data[i]);
            return 1;
        }
    }
    printf("Data integrity verified!\n");

    // ============================================
    // Test Addition
    // ============================================
    printf("\nTesting GPU Addition...\n");
    Tensor* t2 = tensor_create(1, shape, false);
    for(int i=0; i<1024; i++) t2->data[i] = 10.0f;

    Tensor* out = tensor_create(1, shape, false);
    
    // warm up / alloc
    tensor_to_gpu(t);
    tensor_to_gpu(t2);
    
    tensor_add_gpu(out, t, t2);

    tensor_from_gpu(out);

    printf("Result data (first 5): %f %f %f %f %f\n", 
           out->data[0], out->data[1], out->data[2], out->data[3], out->data[4]);

    for (int i = 0; i < 1024; i++) {
        float expected = (float)i + 10.0f;
        if (out->data[i] != expected) {
            printf("Add Mismatch at index %d: expected %f, got %f\n", i, expected, out->data[i]);
            return 1;
        }
    }
    printf("GPU Addition verified!\n");

    tensor_free(t);
    tensor_free(t2);
    tensor_free(out);

    // ============================================
    // Test GEMM
    // ============================================
    printf("\nTesting GPU GEMM (Matrix Multiplication)...\n");
    // 32x32 matrices to test full tiles
    uint32_t M=32, N=32, K=32;
    uint32_t shapeA[] = {M, K};
    uint32_t shapeB[] = {K, N};
    uint32_t shapeC[] = {M, N};

    Tensor* A = tensor_create(2, shapeA, false);
    Tensor* B = tensor_create(2, shapeB, false);
    Tensor* C = tensor_create(2, shapeC, false);

    // Init A with 1.0, B with Identity
    for(int i=0; i<M*K; i++) A->data[i] = 1.0f;
    for(int r=0; r<K; r++) {
        for(int c=0; c<N; c++) {
            B->data[r*N + c] = (r == c) ? 1.0f : 0.0f;
        }
    }

    tensor_matmul_gpu(C, A, B);
    tensor_from_gpu(C);

    // Expect C to be all 1.0s because A * I = A
    bool gemm_passed = true;
    for(int i=0; i<M*N; i++) {
        if (C->data[i] != 1.0f) {
            printf("GEMM Mismatch at %d: %f\n", i, C->data[i]);
            gemm_passed = false;
            break;
        }
    }
    
    if (gemm_passed) printf("GPU GEMM verified!\n");
    else return 1;

    tensor_free(A);
    tensor_free(B);
    tensor_free(C);

    // ============================================
    // Test Activations (ReLU)
    // ============================================
    printf("\nTesting GPU ReLU...\n");
    Tensor* in = tensor_create(2, shapeA, false); // reuse shape
    Tensor* act_out = tensor_create(2, shapeA, false);
    
    // Fill with mix of pos/neg
    for(int i=0; i<32*32; i++) {
        in->data[i] = (float)(i - 512); // Range -512 to 512
    }
    
    tensor_relu_gpu(act_out, in);
    tensor_from_gpu(act_out);
    
    bool relu_passed = true;
    for(int i=0; i<1024; i++) {
        // Fix test logic: input value wasn't captured correctly
        float val = (float)(i - 512);
        float expected = (val > 0) ? val : 0.0f;
        
        if (act_out->data[i] != expected) {
            printf("ReLU Mismatch at %d: in=%f out=%f\n", i, val, act_out->data[i]);
            relu_passed = false;
            break;
        }
    }
    if (relu_passed) printf("GPU ReLU verified!\n");

    tensor_free(in);
    tensor_free(act_out);

    rpl_gpu_shutdown();
    return 0;
}
