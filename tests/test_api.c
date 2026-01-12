#include "rpl.h"
#include <stdio.h>
#include <assert.h>

int main() {
    printf("Testing RPL API...\n");

    // 1. Tensor creation
    uint32_t shape[] = {2, 3};
    Tensor* t = tensor_create(2, shape, true);
    assert(t != NULL);
    assert(t->dims == 2);
    assert(t->size == 6);
    printf("Tensor creation: PASS\n");

    // 2. Linear Layer
    Linear* fc = linear_create(3, 4);
    assert(fc != NULL);
    printf("Linear creation: PASS\n");

    Tensor* input = tensor_create(2, (uint32_t[]){2, 3}, false);
    tensor_fill(input, 1.0f);
    
    Tensor* output = linear_forward(fc, input);
    assert(output != NULL);
    assert(output->shape[0] == 2);
    assert(output->shape[1] == 4);
    printf("Linear forward: PASS\n");

    // 3. Logistic Regression
    LogisticRegression* lr = logistic_regression_create(1.0f, 100, 1e-4f);
    assert(lr != NULL);
    printf("Logistic Regression creation: PASS\n");

    // 4. Data Loader
    Dataset* ds = tensor_dataset_create(input, output);
    DataLoader* dl = dataloader_create(ds, 1, false, false, 0);
    assert(dl != NULL);
    printf("DataLoader creation: PASS\n");

    // Cleanup
    dataloader_free(dl);
    // ds is freed by dataloader_free
    logistic_regression_free(lr);
    linear_free(fc);
    tensor_free(input);
    tensor_free(output);
    tensor_free(t);

    printf("All basic API tests passed!\n");
    return 0;
}
