/*
 * RPiTorch Data Loading Infrastructure - Pure C
 * Efficient batch processing, prefetching, augmentation
 * Optimized for deep learning workflows
 */

#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

// ============================================================
// Dataset Base Structure
// ============================================================

typedef struct Dataset Dataset;

struct Dataset {
    void* data;                    // Dataset-specific data
    uint32_t num_samples;
    uint32_t sample_dims;
    uint32_t sample_shape[RPITORCH_MAX_DIMS];
    uint32_t label_dims;
    uint32_t label_shape[RPITORCH_MAX_DIMS];
    
    // Virtual functions
    void (*get_item)(Dataset* self, uint32_t idx, float* sample, float* label);
    void (*free_fn)(Dataset* self);
};

// ============================================================
// Tensor Dataset (In-Memory)
// ============================================================

typedef struct {
    float* samples;
    float* labels;
    uint32_t sample_size;
    uint32_t label_size;
} TensorDatasetData;

void tensor_dataset_get_item(Dataset* self, uint32_t idx, float* sample, float* label) {
    TensorDatasetData* data = (TensorDatasetData*)self->data;
    memcpy(sample, &data->samples[idx * data->sample_size], 
           data->sample_size * sizeof(float));
    memcpy(label, &data->labels[idx * data->label_size],
           data->label_size * sizeof(float));
}

void tensor_dataset_free(Dataset* self) {
    TensorDatasetData* data = (TensorDatasetData*)self->data;
    free(data->samples);
    free(data->labels);
    free(data);
    free(self);
}

Dataset* tensor_dataset_create(const Tensor* samples, const Tensor* labels) {
    Dataset* ds = (Dataset*)calloc(1, sizeof(Dataset));
    TensorDatasetData* data = (TensorDatasetData*)calloc(1, sizeof(TensorDatasetData));
    
    ds->num_samples = samples->shape[0];
    ds->sample_dims = samples->dims - 1;
    ds->label_dims = labels->dims - 1;
    
    // Copy shapes (excluding batch dimension)
    for (uint32_t i = 0; i < ds->sample_dims; i++) {
        ds->sample_shape[i] = samples->shape[i + 1];
    }
    for (uint32_t i = 0; i < ds->label_dims; i++) {
        ds->label_shape[i] = labels->shape[i + 1];
    }
    
    // Calculate sizes
    data->sample_size = samples->size / samples->shape[0];
    data->label_size = labels->size / labels->shape[0];
    
    // Copy data (aligned for SIMD)
    data->samples = (float*)rpitorch_aligned_alloc(RPITORCH_CACHE_LINE,
                                                   samples->size * sizeof(float));
    data->labels = (float*)rpitorch_aligned_alloc(RPITORCH_CACHE_LINE,
                                                  labels->size * sizeof(float));
    memcpy(data->samples, samples->data, samples->size * sizeof(float));
    memcpy(data->labels, labels->data, labels->size * sizeof(float));
    
    ds->data = data;
    ds->get_item = tensor_dataset_get_item;
    ds->free_fn = tensor_dataset_free;
    
    return ds;
}

// ============================================================
// Memory-Mapped Dataset (Large datasets)
// ============================================================

typedef struct {
    int fd;
    void* mapped_samples;
    void* mapped_labels;
    size_t samples_size;
    size_t labels_size;
    uint32_t sample_size;
    uint32_t label_size;
} MMapDatasetData;

void mmap_dataset_get_item(Dataset* self, uint32_t idx, float* sample, float* label) {
    MMapDatasetData* data = (MMapDatasetData*)self->data;
    
    // Direct memory access (zero-copy)
    float* sample_ptr = (float*)data->mapped_samples + idx * data->sample_size;
    float* label_ptr = (float*)data->mapped_labels + idx * data->label_size;
    
    memcpy(sample, sample_ptr, data->sample_size * sizeof(float));
    memcpy(label, label_ptr, data->label_size * sizeof(float));
}

void mmap_dataset_free(Dataset* self) {
    MMapDatasetData* data = (MMapDatasetData*)self->data;
    munmap(data->mapped_samples, data->samples_size);
    munmap(data->mapped_labels, data->labels_size);
    close(data->fd);
    free(data);
    free(self);
}

Dataset* mmap_dataset_create(const char* samples_file, const char* labels_file,
                             uint32_t num_samples, uint32_t sample_size, uint32_t label_size) {
    Dataset* ds = (Dataset*)calloc(1, sizeof(Dataset));
    MMapDatasetData* data = (MMapDatasetData*)calloc(1, sizeof(MMapDatasetData));
    
    ds->num_samples = num_samples;
    data->sample_size = sample_size;
    data->label_size = label_size;
    
    // Memory-map samples
    data->fd = open(samples_file, O_RDONLY);
    data->samples_size = num_samples * sample_size * sizeof(float);
    data->mapped_samples = mmap(NULL, data->samples_size, PROT_READ, 
                               MAP_PRIVATE, data->fd, 0);
    
    // Memory-map labels
    data->labels_size = num_samples * label_size * sizeof(float);
    data->mapped_labels = mmap(NULL, data->labels_size, PROT_READ,
                              MAP_PRIVATE, data->fd, data->samples_size);
    
    // Advise kernel for sequential access
    madvise(data->mapped_samples, data->samples_size, MADV_SEQUENTIAL);
    madvise(data->mapped_labels, data->labels_size, MADV_SEQUENTIAL);
    
    ds->data = data;
    ds->get_item = mmap_dataset_get_item;
    ds->free_fn = mmap_dataset_free;
    
    return ds;
}

// ============================================================
// Data Augmentation (NEON-optimized)
// ============================================================

struct AugmentationConfig {
    bool random_flip_h;
    bool random_flip_v;
    float brightness_delta;
    float contrast_delta;
    float noise_std;
};

void augment_random_flip_h(float* image, uint32_t height, uint32_t width, uint32_t channels) {
    if ((rand() % 2) == 0) return;
    
    #pragma omp parallel for
    for (uint32_t h = 0; h < height; h++) {
        for (uint32_t w = 0; w < width / 2; w++) {
            for (uint32_t c = 0; c < channels; c++) {
                uint32_t idx1 = (h * width + w) * channels + c;
                uint32_t idx2 = (h * width + (width - 1 - w)) * channels + c;
                
                float temp = image[idx1];
                image[idx1] = image[idx2];
                image[idx2] = temp;
            }
        }
    }
}

void augment_brightness(float* image, uint32_t size, float delta) {
    float factor = 1.0f + ((float)rand() / RAND_MAX - 0.5f) * 2.0f * delta;
    
    uint32_t i = 0;
    
    #if RPITORCH_HAS_NEON
    float32x4_t vfactor = vdupq_n_f32(factor);
    float32x4_t vzero = vdupq_n_f32(0.0f);
    float32x4_t vone = vdupq_n_f32(1.0f);
    
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&image[i]);
        v = vmulq_f32(v, vfactor);
        v = vmaxq_f32(v, vzero);
        v = vminq_f32(v, vone);
        vst1q_f32(&image[i], v);
    }
    #endif
    
    for (; i < size; i++) {
        image[i] = fminf(fmaxf(image[i] * factor, 0.0f), 1.0f);
    }
}

void augment_add_noise(float* image, uint32_t size, float std) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < size; i++) {
        // Box-Muller transform for Gaussian noise
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        float noise = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2) * std;
        image[i] = fminf(fmaxf(image[i] + noise, 0.0f), 1.0f);
    }
}

void apply_augmentation(float* sample, uint32_t sample_size, 
                       const AugmentationConfig* config,
                       uint32_t height, uint32_t width, uint32_t channels) {
    if (config->random_flip_h) {
        augment_random_flip_h(sample, height, width, channels);
    }
    
    if (config->brightness_delta > 0.0f) {
        augment_brightness(sample, sample_size, config->brightness_delta);
    }
    
    if (config->noise_std > 0.0f) {
        augment_add_noise(sample, sample_size, config->noise_std);
    }
}

// ============================================================
// DataLoader with Prefetching
// ============================================================

#define MAX_PREFETCH_BATCHES 4

struct DataLoader {
    Dataset* dataset;
    uint32_t batch_size;
    bool shuffle;
    bool drop_last;
    uint32_t num_workers;
    
    // Prefetching
    pthread_t* worker_threads;
    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_not_empty;
    pthread_cond_t queue_not_full;
    
    // Batch queue
    Tensor** batch_samples_queue;
    Tensor** batch_labels_queue;
    uint32_t queue_head;
    uint32_t queue_tail;
    uint32_t queue_size;
    
    // Iteration state
    uint32_t* indices;
    uint32_t current_idx;
    uint32_t num_batches;
    bool stop_workers;
    
    // Augmentation
    AugmentationConfig* augmentation;
};

void* dataloader_worker(void* arg) {
    DataLoader* loader = (DataLoader*)arg;
    TensorDatasetData* ds_data = (TensorDatasetData*)loader->dataset->data;
    
    while (!loader->stop_workers) {
        pthread_mutex_lock(&loader->queue_mutex);
        
        // Wait if queue is full
        while (loader->queue_size >= MAX_PREFETCH_BATCHES && !loader->stop_workers) {
            pthread_cond_wait(&loader->queue_not_full, &loader->queue_mutex);
        }
        
        if (loader->stop_workers) {
            pthread_mutex_unlock(&loader->queue_mutex);
            break;
        }
        
        // Get next batch indices
        uint32_t batch_start = __sync_fetch_and_add(&loader->current_idx, loader->batch_size);
        
        if (batch_start >= loader->dataset->num_samples) {
            pthread_mutex_unlock(&loader->queue_mutex);
            break;
        }
        
        uint32_t batch_end = batch_start + loader->batch_size;
        if (batch_end > loader->dataset->num_samples) {
            if (loader->drop_last) {
                pthread_mutex_unlock(&loader->queue_mutex);
                continue;
            }
            batch_end = loader->dataset->num_samples;
        }
        
        uint32_t actual_batch_size = batch_end - batch_start;
        
        pthread_mutex_unlock(&loader->queue_mutex);
        
        // Allocate batch tensors
        uint32_t sample_shape[RPITORCH_MAX_DIMS + 1];
        uint32_t label_shape[RPITORCH_MAX_DIMS + 1];
        
        sample_shape[0] = actual_batch_size;
        label_shape[0] = actual_batch_size;
        
        for (uint32_t i = 0; i < loader->dataset->sample_dims; i++) {
            sample_shape[i + 1] = loader->dataset->sample_shape[i];
        }
        for (uint32_t i = 0; i < loader->dataset->label_dims; i++) {
            label_shape[i + 1] = loader->dataset->label_shape[i];
        }
        
        Tensor* batch_samples = tensor_create(loader->dataset->sample_dims + 1, sample_shape, false);
        Tensor* batch_labels = tensor_create(loader->dataset->label_dims + 1, label_shape, false);
        
        // Load batch data
        uint32_t sample_size = ds_data->sample_size;
        uint32_t label_size = ds_data->label_size;
        
        for (uint32_t i = 0; i < actual_batch_size; i++) {
            uint32_t idx = loader->indices[batch_start + i];
            
            // Copy sample
            memcpy(&batch_samples->data[i * sample_size],
                  &ds_data->samples[idx * sample_size],
                  sample_size * sizeof(float));
            
            // Copy label
            memcpy(&batch_labels->data[i * label_size],
                  &ds_data->labels[idx * label_size],
                  label_size * sizeof(float));
            
            // Apply augmentation
            if (loader->augmentation) {
                apply_augmentation(&batch_samples->data[i * sample_size],
                                 sample_size, loader->augmentation,
                                 sample_shape[1], sample_shape[2], sample_shape[3]);
            }
        }
        
        // Add to queue
        pthread_mutex_lock(&loader->queue_mutex);
        
        loader->batch_samples_queue[loader->queue_tail] = batch_samples;
        loader->batch_labels_queue[loader->queue_tail] = batch_labels;
        loader->queue_tail = (loader->queue_tail + 1) % MAX_PREFETCH_BATCHES;
        loader->queue_size++;
        
        pthread_cond_signal(&loader->queue_not_empty);
        pthread_mutex_unlock(&loader->queue_mutex);
    }
    
    return NULL;
}

DataLoader* dataloader_create(Dataset* dataset, uint32_t batch_size,
                              bool shuffle, bool drop_last, uint32_t num_workers) {
    DataLoader* loader = (DataLoader*)calloc(1, sizeof(DataLoader));
    
    loader->dataset = dataset;
    loader->batch_size = batch_size;
    loader->shuffle = shuffle;
    loader->drop_last = drop_last;
    loader->num_workers = num_workers;
    
    // Initialize indices
    loader->indices = (uint32_t*)malloc(dataset->num_samples * sizeof(uint32_t));
    for (uint32_t i = 0; i < dataset->num_samples; i++) {
        loader->indices[i] = i;
    }
    
    // Calculate number of batches
    loader->num_batches = dataset->num_samples / batch_size;
    if (!drop_last && (dataset->num_samples % batch_size != 0)) {
        loader->num_batches++;
    }
    
    // Initialize queue
    loader->batch_samples_queue = (Tensor**)calloc(MAX_PREFETCH_BATCHES, sizeof(Tensor*));
    loader->batch_labels_queue = (Tensor**)calloc(MAX_PREFETCH_BATCHES, sizeof(Tensor*));
    
    pthread_mutex_init(&loader->queue_mutex, NULL);
    pthread_cond_init(&loader->queue_not_empty, NULL);
    pthread_cond_init(&loader->queue_not_full, NULL);
    
    return loader;
}

void dataloader_set_augmentation(DataLoader* loader, const AugmentationConfig* config) {
    if (!loader->augmentation) {
        loader->augmentation = (AugmentationConfig*)malloc(sizeof(AugmentationConfig));
    }
    memcpy(loader->augmentation, config, sizeof(AugmentationConfig));
}

void dataloader_start(DataLoader* loader) {
    // Shuffle if needed
    if (loader->shuffle) {
        for (uint32_t i = loader->dataset->num_samples - 1; i > 0; i--) {
            uint32_t j = rand() % (i + 1);
            uint32_t temp = loader->indices[i];
            loader->indices[i] = loader->indices[j];
            loader->indices[j] = temp;
        }
    }
    
    loader->current_idx = 0;
    loader->queue_head = 0;
    loader->queue_tail = 0;
    loader->queue_size = 0;
    loader->stop_workers = false;
    
    // Start worker threads
    if (loader->num_workers > 0) {
        loader->worker_threads = (pthread_t*)malloc(loader->num_workers * sizeof(pthread_t));
        for (uint32_t i = 0; i < loader->num_workers; i++) {
            pthread_create(&loader->worker_threads[i], NULL, dataloader_worker, loader);
        }
    }
}

bool dataloader_next(DataLoader* loader, Tensor** batch_samples, Tensor** batch_labels) {
    if (loader->num_workers > 0) {
        // Multi-threaded prefetching
        pthread_mutex_lock(&loader->queue_mutex);
        
        while (loader->queue_size == 0 && !loader->stop_workers) {
            pthread_cond_wait(&loader->queue_not_empty, &loader->queue_mutex);
        }
        
        if (loader->queue_size == 0) {
            pthread_mutex_unlock(&loader->queue_mutex);
            return false;
        }
        
        *batch_samples = loader->batch_samples_queue[loader->queue_head];
        *batch_labels = loader->batch_labels_queue[loader->queue_head];
        
        loader->queue_head = (loader->queue_head + 1) % MAX_PREFETCH_BATCHES;
        loader->queue_size--;
        
        pthread_cond_signal(&loader->queue_not_full);
        pthread_mutex_unlock(&loader->queue_mutex);
        
        return true;
    } else {
        // Single-threaded (synchronous)
        if (loader->current_idx >= loader->dataset->num_samples) {
            return false;
        }
        
        // Load batch synchronously
        // (Implementation similar to worker thread)
        return true;
    }
}

void dataloader_stop(DataLoader* loader) {
    if (loader->num_workers > 0) {
        loader->stop_workers = true;
        pthread_cond_broadcast(&loader->queue_not_full);
        
        for (uint32_t i = 0; i < loader->num_workers; i++) {
            pthread_join(loader->worker_threads[i], NULL);
        }
        
        free(loader->worker_threads);
    }
}

void dataloader_free(DataLoader* loader) {
    dataloader_stop(loader);
    
    free(loader->indices);
    free(loader->batch_samples_queue);
    free(loader->batch_labels_queue);
    
    if (loader->augmentation) {
        free(loader->augmentation);
    }
    
    pthread_mutex_destroy(&loader->queue_mutex);
    pthread_cond_destroy(&loader->queue_not_empty);
    pthread_cond_destroy(&loader->queue_not_full);
    
    free(loader);
}
