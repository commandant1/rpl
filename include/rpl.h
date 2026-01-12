/*
 * RPL - RPI Learn
 * A lightweight, pure C machine learning library for Raspberry Pi 4
 * 
 * Optimized for ARM Cortex-A72 with NEON SIMD, OpenMP, and OpenBLAS
 */

#ifndef RPL_H
#define RPL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Device Management
// ============================================================

typedef enum {
    DEVICE_CPU,
    DEVICE_GPU
} DeviceType;

// ============================================================
// Configuration & Feature Detection
// ============================================================

#define MAX_DIMS 8
#define RPL_MAX_DIMS MAX_DIMS
#define RPITORCH_MAX_DIMS MAX_DIMS
#define RPITORCH_CACHE_LINE 64

// Feature detection
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define RPITORCH_HAS_NEON 1
    #include <arm_neon.h>
#else
    #define RPITORCH_HAS_NEON 0
#endif


#ifdef USE_OPENBLAS
    #define RPITORCH_HAS_BLAS 1
    #include <cblas.h>
#else
    #define RPITORCH_HAS_BLAS 0
#endif

// ============================================================
// Core Tensor Structure
// ============================================================

typedef struct Tensor {
    float* data;
    float* grad;
    uint32_t dims;
    uint32_t shape[MAX_DIMS];
    uint32_t strides[MAX_DIMS];
    uint32_t size;
    bool requires_grad;
    
    // Memory management
    void* _allocation;
    size_t _alloc_size;
    
    DeviceType device;
    uint32_t gpu_buffer;  // OpenGL Buffer Object ID
    
    // Autograd
    bool is_leaf;
    struct Tensor* parent1;
    struct Tensor* parent2;
    void (*backward_fn)(struct Tensor*);
} Tensor;

// ============================================================
// Tensor Operations
// ============================================================

// Memory management
Tensor* tensor_create(uint32_t dims, const uint32_t* shape, bool requires_grad);
void tensor_free(Tensor* t);
void* rpitorch_aligned_alloc(size_t alignment, size_t size);
void rpitorch_aligned_free(void* ptr);

// GPU Operations
bool rpl_gpu_init();
void rpl_gpu_shutdown();
void tensor_to_gpu(Tensor* t);
void tensor_from_gpu(Tensor* t);
void tensor_add_gpu(Tensor* out, const Tensor* a, const Tensor* b);

// Initialization
void tensor_fill(Tensor* t, float value);
void tensor_fill_buffer(float* buffer, float value, uint32_t size);

// Math operations
Tensor* tensor_add(const Tensor* a, const Tensor* b);
void tensor_add_out(Tensor* out, const Tensor* a, const Tensor* b);
Tensor* tensor_mul(const Tensor* a, const Tensor* b);
void tensor_mul_out(Tensor* out, const Tensor* a, const Tensor* b);
void tensor_gemm(Tensor* C, const Tensor* A, const Tensor* B, float alpha, float beta, bool trans_a, bool trans_b);
Tensor* tensor_matmul(const Tensor* a, const Tensor* b);
void tensor_add_inplace(Tensor* a, const Tensor* b);
void tensor_mul_inplace(Tensor* a, float scalar);
void tensor_randomize(Tensor* t);

// Activations
void tensor_relu_inplace(Tensor* t);
Tensor* tensor_relu(const Tensor* t);
Tensor* tensor_sigmoid(const Tensor* t);
void tensor_sigmoid_inplace(Tensor* t);
void tensor_tanh_inplace(Tensor* t);
void tensor_gelu_inplace(Tensor* t);
void tensor_softmax_inplace(Tensor* t);
void tensor_leaky_relu(Tensor* out, const Tensor* in, float negative_slope);
void tensor_leaky_relu_inplace(Tensor* t, float negative_slope);
void tensor_elu(Tensor* out, const Tensor* in, float alpha);
void tensor_elu_inplace(Tensor* t, float alpha);
void tensor_swish(Tensor* out, const Tensor* in);
void tensor_swish_inplace(Tensor* t);
void tensor_softplus(Tensor* out, const Tensor* in, float beta, float threshold);
void tensor_softplus_inplace(Tensor* t, float beta, float threshold);

// Autograd
void tensor_backward(Tensor* t);
void tensor_zero_grad(Tensor* t);
void backward_add(Tensor* t);
void backward_mul(Tensor* t);
void backward_matmul(Tensor* t);
void backward_relu(Tensor* t);
void backward_sigmoid(Tensor* t);
void backward_mse(Tensor* t);

// ============================================================
// Quantization
// ============================================================

typedef struct {
    int8_t* data;
    float scale;
    int32_t zero_point;
    uint32_t size;
    uint32_t dims;
    uint32_t shape[MAX_DIMS];
} QuantizedTensor;

QuantizedTensor* tensor_quantize_int8(const Tensor* input, float scale, int32_t zero_point);
void quantized_tensor_free(QuantizedTensor* qt);
void tensor_get_min_max(const Tensor* t, float* min_val, float* max_val);

Tensor* tensor_mse_loss(const Tensor* pred, const Tensor* target);
void tensor_batchnorm2d(Tensor* out, const Tensor* in, float* gamma, float* beta, float* running_mean, float* running_var, float eps, bool training, float momentum);
void tensor_batchnorm2d_forward(Tensor* input, Tensor* output, const float* gamma, const float* beta, const float* running_mean, const float* running_var, float eps, bool training, float momentum);
void tensor_dropout(Tensor* out, const Tensor* in, float p, bool training);
Tensor* tensor_conv2d(const Tensor* input, const Tensor* kernel, uint32_t stride, uint32_t padding);
Tensor* tensor_maxpool2d(const Tensor* input, uint32_t kernel_size, uint32_t stride);
Tensor* tensor_dequantize_int8(QuantizedTensor* input);
void gemm_int8(const int8_t* A, const int8_t* B, int32_t* C, uint32_t M, uint32_t N, uint32_t K, float sa, float sb, float sc, int32_t za, int32_t zb);

// ============================================================
// Neural Network Layers
// ============================================================

typedef struct Linear Linear;
typedef struct Conv2dLayer Conv2dLayer;
typedef struct Conv3dLayer Conv3dLayer;
typedef struct LSTMLayer LSTMLayer;
typedef struct GRULayer GRULayer;
typedef struct BatchNorm2dLayer BatchNorm2dLayer;
typedef struct LayerNormLayer LayerNormLayer;
typedef struct EmbeddingLayer EmbeddingLayer;
typedef struct DropoutLayer DropoutLayer;
typedef struct MaxPool2dLayer MaxPool2dLayer;
typedef struct LRScheduler LRScheduler;

// Attention
typedef struct MultiHeadAttention MultiHeadAttention;
typedef struct PositionalEncoding PositionalEncoding;

MultiHeadAttention* multi_head_attention_create(uint32_t d_model, uint32_t num_heads, float dropout_p);
Tensor* multi_head_attention_forward(MultiHeadAttention* mha, const Tensor* query, const Tensor* key, const Tensor* value, const Tensor* mask, bool training);
void multi_head_attention_free(MultiHeadAttention* mha);

PositionalEncoding* positional_encoding_create(uint32_t max_len, uint32_t d_model, bool learnable);
Tensor* positional_encoding_forward(PositionalEncoding* pe, const Tensor* input);
void positional_encoding_free(PositionalEncoding* pe);

// Vision layers
typedef struct ResBlock ResBlock;
ResBlock* res_block_create(uint32_t channels);
Tensor* res_block_forward(ResBlock* block, const Tensor* input);
void res_block_free(ResBlock* block);
typedef struct SEBlock SEBlock;
typedef struct InstanceNorm InstanceNorm;
typedef struct DilatedConv2d DilatedConv2d;
typedef struct SPPLayer SPPLayer;
typedef struct PatchEmbedding PatchEmbedding;
typedef struct DepthwiseSeparableConv DepthwiseSeparableConv;

// Linear
Linear* linear_create(uint32_t in_features, uint32_t out_features);
Tensor* linear_forward(Linear* layer, const Tensor* input);
void linear_free(Linear* layer);

// Conv2D
Conv2dLayer* conv2d_create(uint32_t in_channels, uint32_t out_channels,
                           uint32_t kernel_size, uint32_t stride, uint32_t padding);
Tensor* conv2d_forward(Conv2dLayer* layer, const Tensor* input);
void conv2d_free(Conv2dLayer* layer);

// Conv3D
Conv3dLayer* conv3d_create(uint32_t in_channels, uint32_t out_channels,
                           uint32_t kernel_d, uint32_t kernel_h, uint32_t kernel_w,
                           uint32_t stride_d, uint32_t stride_h, uint32_t stride_w,
                           uint32_t padding_d, uint32_t padding_h, uint32_t padding_w);
Tensor* conv3d_forward(Conv3dLayer* layer, const Tensor* input);
void conv3d_free(Conv3dLayer* layer);

// BatchNorm
BatchNorm2dLayer* batchnorm2d_create(uint32_t num_features, float momentum, float eps);
Tensor* batchnorm2d_forward(BatchNorm2dLayer* layer, const Tensor* input);
void batchnorm2d_free(BatchNorm2dLayer* layer);

// LayerNorm
LayerNormLayer* layer_norm_create(uint32_t normalized_shape, float eps);
Tensor* layer_norm_forward(LayerNormLayer* layer, const Tensor* input);
void layer_norm_free(LayerNormLayer* layer);

// Embedding
EmbeddingLayer* embedding_create(uint32_t num_embeddings, uint32_t embedding_dim);
Tensor* embedding_forward(EmbeddingLayer* layer, const uint32_t* indices, uint32_t num_indices);
void embedding_free(EmbeddingLayer* layer);

// Dropout
DropoutLayer* dropout_create(float p);
Tensor* dropout_forward(DropoutLayer* layer, const Tensor* input);
void dropout_free(DropoutLayer* layer);

// MaxPool2D
MaxPool2dLayer* maxpool2d_create(uint32_t kernel_size, uint32_t stride, uint32_t padding);
Tensor* maxpool2d_forward(MaxPool2dLayer* layer, const Tensor* input);
void maxpool2d_free(MaxPool2dLayer* layer);

// Advanced Vision
DepthwiseSeparableConv* depthwise_separable_conv_create(uint32_t in_channels, uint32_t out_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding);
Tensor* depthwise_separable_conv_forward(DepthwiseSeparableConv* layer, const Tensor* input);
void depthwise_separable_conv_free(DepthwiseSeparableConv* layer);

PatchEmbedding* patch_embedding_create(uint32_t img_channels, uint32_t patch_size, uint32_t embed_dim);
Tensor* patch_embedding_forward(PatchEmbedding* layer, const Tensor* input);
void patch_embedding_free(PatchEmbedding* layer);

SEBlock* se_block_create(uint32_t channels, uint32_t reduction);
Tensor* se_block_forward(SEBlock* block, const Tensor* input);
void se_block_free(SEBlock* block);

InstanceNorm* instance_norm_create(uint32_t num_features, float eps);
Tensor* instance_norm_forward(InstanceNorm* layer, const Tensor* input);
void instance_norm_free(InstanceNorm* layer);

DilatedConv2d* dilated_conv_create(uint32_t in_channels, uint32_t out_channels, uint32_t kernel_size, uint32_t stride, uint32_t padding, uint32_t dilation);
Tensor* dilated_conv_forward(DilatedConv2d* layer, const Tensor* input);
void dilated_conv_free(DilatedConv2d* layer);

SPPLayer* spp_create(uint32_t* pool_sizes, uint32_t num_levels);
Tensor* spp_forward(SPPLayer* layer, const Tensor* input);
void spp_free(SPPLayer* layer);

// ============================================================
// Loss & Training
// ============================================================

float mse_loss(const Tensor* pred, const Tensor* target);
float cross_entropy_loss(const Tensor* pred, const Tensor* target);
float binary_cross_entropy_loss(const Tensor* pred, const Tensor* target);

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_ADAM,
    OPTIMIZER_RMSPROP,
    OPTIMIZER_ADAGRAD,
    OPTIMIZER_ADAMW,
    OPTIMIZER_LAMB
} OptimizerType;

typedef struct Optimizer Optimizer;
Optimizer* optimizer_sgd_create(Tensor** parameters, uint32_t num_params, float lr, float momentum, float dampening, float weight_decay, bool nesterov);
Optimizer* optimizer_adam_create(Tensor** parameters, uint32_t num_params, float lr, float beta1, float beta2, float epsilon, float weight_decay);
Optimizer* optimizer_adamw_create(Tensor** parameters, uint32_t num_params, float lr, float beta1, float beta2, float epsilon, float weight_decay);
void optimizer_step(Optimizer* opt);
void optimizer_zero_grad(Optimizer* opt);
void optimizer_free(Optimizer* opt);

typedef struct EarlyStopping EarlyStopping;
EarlyStopping* early_stopping_create(uint32_t patience, float min_delta, bool minimize);
bool early_stopping_check(EarlyStopping* es, uint32_t epoch, float metric);
void early_stopping_free(EarlyStopping* es);

typedef struct LRScheduler LRScheduler;
LRScheduler* lr_scheduler_step_create(Optimizer* opt, uint32_t step_size, float gamma);
void lr_scheduler_step(LRScheduler* sched);
void lr_scheduler_free(LRScheduler* sched);

typedef struct MetricsTracker MetricsTracker;
MetricsTracker* metrics_tracker_create();
void metrics_tracker_add(MetricsTracker* mt, const char* name, float value);
void metrics_tracker_next_epoch(MetricsTracker* mt);
void metrics_tracker_print(const MetricsTracker* mt);
void metrics_tracker_free(MetricsTracker* mt);

typedef struct ModelCheckpoint ModelCheckpoint;
ModelCheckpoint* checkpoint_create(const char* filepath, const char* monitor, bool save_best_only, bool minimize);
void checkpoint_save_model(ModelCheckpoint* ckpt, Tensor** parameters, uint32_t num_params, uint32_t epoch, float metric);
void checkpoint_free(ModelCheckpoint* ckpt);

// ============================================================
// Data Loading
// ============================================================

typedef struct Dataset Dataset;
typedef struct DataLoader DataLoader;
typedef struct AugmentationConfig AugmentationConfig;

Dataset* tensor_dataset_create(const Tensor* samples, const Tensor* labels);
DataLoader* dataloader_create(Dataset* dataset, uint32_t batch_size, bool shuffle, bool drop_last, uint32_t num_workers);
void dataloader_start(DataLoader* loader);
bool dataloader_next(DataLoader* loader, Tensor** batch_samples, Tensor** batch_labels);
void dataloader_free(DataLoader* loader);

// ============================================================
// Classical ML
// ============================================================

typedef struct LinearRegression LinearRegression;
LinearRegression* linear_regression_create();
void linear_regression_fit(LinearRegression* lr, const float* X, const float* y, uint32_t n_samples, uint32_t n_features);
void linear_regression_predict(const LinearRegression* lr, const float* X, uint32_t n_samples, float* y_pred);
void linear_regression_free(LinearRegression* lr);

typedef struct LogisticRegression LogisticRegression;
LogisticRegression* logistic_regression_create(float C, uint32_t max_iter, float tol);
void logistic_regression_fit(LogisticRegression* lr, const float* X, const uint32_t* y, uint32_t n_samples, uint32_t n_features, uint32_t n_classes);
void logistic_regression_predict(const LogisticRegression* lr, const float* X, uint32_t n_samples, uint32_t* y_pred);
void logistic_regression_free(LogisticRegression* lr);

typedef struct KMeans KMeans;
KMeans* kmeans_create(uint32_t n_clusters, uint32_t max_iter, float tol);
void kmeans_fit(KMeans* km, const float* X, uint32_t n_samples, uint32_t n_features);
void kmeans_predict(const KMeans* km, const float* X, uint32_t n_samples, uint32_t* labels);
void kmeans_free(KMeans* km);

typedef struct PCA PCA;
PCA* pca_create(uint32_t n_components);
void pca_fit(PCA* pca, const float* X, uint32_t n_samples, uint32_t n_features);
void pca_transform(const PCA* pca, const float* X, uint32_t n_samples, float* X_transformed);
void pca_free(PCA* pca);

typedef struct SVM SVM;
SVM* svm_create(float C, float gamma);
void svm_fit(SVM* svm, const float* X, const float* y, uint32_t n_samples, uint32_t n_features);
void svm_predict(const SVM* svm, const float* X, uint32_t n_samples, float* y_pred);
void svm_free(SVM* svm);

typedef struct NaiveBayes NaiveBayes;
NaiveBayes* naive_bayes_create();
void naive_bayes_fit(NaiveBayes* nb, const float* X, const uint32_t* y, uint32_t n_samples, uint32_t n_features, uint32_t n_classes);
void naive_bayes_predict(const NaiveBayes* nb, const float* X, uint32_t n_samples, uint32_t* y_pred);
void naive_bayes_free(NaiveBayes* nb);

typedef struct DBSCAN DBSCAN;
DBSCAN* dbscan_create(float eps, uint32_t min_samples);
void dbscan_fit(DBSCAN* db, const float* X, uint32_t n_samples, uint32_t n_features);
void dbscan_free(DBSCAN* db);

// ============================================================
// Reinforcement Learning
// ============================================================

typedef struct Transition Transition;
typedef struct ReplayBuffer ReplayBuffer;
typedef struct QNetwork QNetwork;
typedef struct Episode Episode;
typedef struct ActorCritic ActorCritic;

ReplayBuffer* replay_buffer_create(uint32_t capacity, uint32_t state_dim);
void replay_buffer_push(ReplayBuffer* rb, const float* state, uint32_t action, float reward, const float* next_state, bool done);
void replay_buffer_free(ReplayBuffer* rb);

QNetwork* q_network_create(uint32_t state_dim, uint32_t action_dim, uint32_t hidden_dim);
Tensor* q_network_forward(QNetwork* qnet, const Tensor* state);
uint32_t q_network_select_action(QNetwork* qnet, const Tensor* state, float epsilon);
void q_network_free(QNetwork* qnet);

ActorCritic* actor_critic_create(uint32_t state_dim, uint32_t action_dim, uint32_t hidden_dim);
void actor_critic_free(ActorCritic* ac);

// ============================================================
// Metrics
// ============================================================

float accuracy_score(const uint32_t* y_true, const uint32_t* y_pred, uint32_t n);
float roc_auc_score(const float* y_true, const float* y_score, uint32_t n);
float r2_score(const float* y_true, const float* y_pred, uint32_t n);
void confusion_matrix(const uint32_t* y_true, const uint32_t* y_pred, uint32_t n, uint32_t n_classes, uint32_t* matrix);

typedef struct CrossValidationSplit CrossValidationSplit;
CrossValidationSplit* cross_validation_split(uint32_t n_samples, uint32_t n_splits, bool shuffle);
void cross_validation_free(CrossValidationSplit* cv);

typedef struct HyperParameter HyperParameter;
typedef struct GridSearch GridSearch;
typedef float (*GridSearchScoreFunc)(const float* params, uint32_t n_params, void* user_data);
GridSearch* grid_search_create(uint32_t n_params);
void grid_search_add_param(GridSearch* gs, uint32_t param_idx, const char* name, const float* values, uint32_t n_values);
void grid_search_fit(GridSearch* gs, GridSearchScoreFunc score_func, void* user_data);
void grid_search_free(GridSearch* gs);

#ifdef __cplusplus
}
#endif

#endif // RPL_H
