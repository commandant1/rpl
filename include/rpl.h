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

struct Tensor;
void tensor_free_gpu(struct Tensor* t);

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

// OMP size threshold: skip thread spawn for small tensors
// Thread creation costs ~50µs on Cortex-A72; at 1 FLOP/cycle,
// 4096 floats takes ~2.7µs — below this, OMP overhead dominates
#ifndef RPL_OMP_THRESHOLD
#define RPL_OMP_THRESHOLD 4096
#endif

// GPU dispatch threshold: for tensors smaller than this, GPU overhead
// (GLES compute shader kernel launch + SSBO round-trip latency ≈ 100–500µs)
// exceeds the cost of a NEON/OpenMP CPU kernel.  Only dispatch to GPU when
// the tensor is large enough to amortise that overhead.
// Rule of thumb on Raspberry Pi 4 VideoCore VI: GPU wins for >16 K elements.
// Override at build time: -DRPL_GPU_THRESHOLD=<n>
#ifndef RPL_GPU_THRESHOLD
#define RPL_GPU_THRESHOLD 16384
#endif

// Element-count check: used for pointwise/unary/binary ops.
#ifdef USE_GPU
#  define RPL_GPU_PREFERABLE(size) ((size) >= (uint32_t)RPL_GPU_THRESHOLD)
#else
#  define RPL_GPU_PREFERABLE(size) (false)
#endif

// FLOPs-based check for GEMM: GPU overhead is amortised only once the
// arithmetic intensity (M*N*K multiply-adds) is large enough.
// Default: require at least 4M FLOPs (e.g. 128x128x256 or 256x256x64).
#ifndef RPL_GPU_GEMM_FLOP_THRESHOLD
#define RPL_GPU_GEMM_FLOP_THRESHOLD 4194304ULL   /* 4M */
#endif
#ifdef USE_GPU
#  define RPL_GPU_GEMM_PREFERABLE(m,n,k) \
      (((uint64_t)(m) * (uint64_t)(n) * (uint64_t)(k)) >= RPL_GPU_GEMM_FLOP_THRESHOLD)
#else
#  define RPL_GPU_GEMM_PREFERABLE(m,n,k) (false)
#endif

// Hot-path attribute for frequently-called functions
#define RPL_HOT __attribute__((hot))
#define RPL_LIKELY(x)   __builtin_expect(!!(x), 1)
#define RPL_UNLIKELY(x) __builtin_expect(!!(x), 0)


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

typedef uint16_t rpl_half;

typedef struct HalfTensor {
    rpl_half* data;
    uint32_t dims;
    uint32_t shape[MAX_DIMS];
    uint32_t strides[MAX_DIMS];
    uint32_t size;
    
    void* _allocation;
    DeviceType device;
    uint32_t gpu_buffer;
} HalfTensor;

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
void tensor_free_gpu(Tensor* t);
void tensor_add_gpu(Tensor* out, const Tensor* a, const Tensor* b);
void tensor_sub_gpu(Tensor* out, const Tensor* a, const Tensor* b);
void tensor_mul_gpu(Tensor* out, const Tensor* a, const Tensor* b);
void tensor_div_gpu(Tensor* out, const Tensor* a, const Tensor* b);
void tensor_matmul_gpu(Tensor* C, const Tensor* A, const Tensor* B);
/* Full GEMM with optional transpose and scaling: C = alpha*op(A)@op(B) + beta*C */
void tensor_gemm_gpu(Tensor* C, const Tensor* A, const Tensor* B,
                     uint32_t M, uint32_t N, uint32_t K,
                     float alpha, float beta, bool trans_a, bool trans_b);
// Element-wise activations (out-of-place)
void tensor_relu_gpu(Tensor* out, const Tensor* in);
void tensor_sigmoid_gpu(Tensor* out, const Tensor* in);
void tensor_tanh_gpu(Tensor* out, const Tensor* in);
void tensor_gelu_gpu(Tensor* out, const Tensor* in);
void tensor_leaky_relu_gpu(Tensor* out, const Tensor* in, float negative_slope);
void tensor_swish_gpu(Tensor* out, const Tensor* in);
void tensor_elu_gpu(Tensor* out, const Tensor* in, float alpha);
// In-place activations
void tensor_relu_inplace_gpu(Tensor* t);
// Softmax (axis = dimension to normalise over; 1 for row-wise on 2-D tensors)
void tensor_softmax_gpu(Tensor* out, const Tensor* in, uint32_t axis);
// Conv2D via GL_TEXTURE_2D sampling
// in  : [C_in,  H,   W]   (NCHW, one image at a time)
// kern: [C_out, C_in, kH, kW]
// out : [C_out, out_H, out_W]
void tensor_conv2d_gpu(Tensor* out, const Tensor* in, const Tensor* kern,
                       int kH, int kW, int stride, int padding);
void tensor_selu_gpu(Tensor* out, const Tensor* in);
void tensor_mish_gpu(Tensor* out, const Tensor* in);
void tensor_hardswish_gpu(Tensor* out, const Tensor* in);
void tensor_hardsigmoid_gpu(Tensor* out, const Tensor* in);
void tensor_softplus_gpu(Tensor* out, const Tensor* in, float beta, float threshold);
void tensor_log_softmax_gpu(Tensor* out, const Tensor* in);
void tensor_scale_gpu(Tensor* t, float scalar);
// Math unary GPU ops (shared GLSL program, uniform int op selector)
void tensor_sin_gpu(Tensor* out, const Tensor* in);
void tensor_cos_gpu(Tensor* out, const Tensor* in);
void tensor_tan_gpu(Tensor* out, const Tensor* in);
void tensor_asin_gpu(Tensor* out, const Tensor* in);
void tensor_acos_gpu(Tensor* out, const Tensor* in);
void tensor_atan_gpu(Tensor* out, const Tensor* in);
void tensor_sinh_gpu(Tensor* out, const Tensor* in);
void tensor_cosh_gpu(Tensor* out, const Tensor* in);
void tensor_asinh_gpu(Tensor* out, const Tensor* in);
void tensor_acosh_gpu(Tensor* out, const Tensor* in);
void tensor_atanh_gpu(Tensor* out, const Tensor* in);
void tensor_exp_gpu(Tensor* out, const Tensor* in);
void tensor_exp2_gpu(Tensor* out, const Tensor* in);
void tensor_expm1_gpu(Tensor* out, const Tensor* in);
void tensor_log_gpu(Tensor* out, const Tensor* in);
void tensor_log2_gpu(Tensor* out, const Tensor* in);
void tensor_log10_gpu(Tensor* out, const Tensor* in);
void tensor_log1p_gpu(Tensor* out, const Tensor* in);
void tensor_sqrt_gpu(Tensor* out, const Tensor* in);
void tensor_rsqrt_gpu(Tensor* out, const Tensor* in);
void tensor_square_gpu(Tensor* out, const Tensor* in);
void tensor_cbrt_gpu(Tensor* out, const Tensor* in);
void tensor_reciprocal_gpu(Tensor* out, const Tensor* in);
void tensor_abs_gpu(Tensor* out, const Tensor* in);
void tensor_neg_gpu(Tensor* out, const Tensor* in);
void tensor_sign_gpu(Tensor* out, const Tensor* in);
void tensor_deg2rad_gpu(Tensor* out, const Tensor* in);
void tensor_rad2deg_gpu(Tensor* out, const Tensor* in);
void tensor_erf_gpu(Tensor* out, const Tensor* in);
void tensor_logit_gpu(Tensor* out, const Tensor* in);
void tensor_round_gpu(Tensor* out, const Tensor* in);
void tensor_floor_gpu(Tensor* out, const Tensor* in);
void tensor_ceil_gpu(Tensor* out, const Tensor* in);
void tensor_trunc_gpu(Tensor* out, const Tensor* in);
void tensor_frac_gpu(Tensor* out, const Tensor* in);
// Math binary GPU ops
void tensor_pow_gpu(Tensor* out, const Tensor* a, const Tensor* b);
void tensor_atan2_gpu(Tensor* out, const Tensor* a, const Tensor* b);
void tensor_hypot_gpu(Tensor* out, const Tensor* a, const Tensor* b);
void tensor_fmod_gpu(Tensor* out, const Tensor* a, const Tensor* b);
void tensor_remainder_gpu(Tensor* out, const Tensor* a, const Tensor* b);
void tensor_floor_divide_gpu(Tensor* out, const Tensor* a, const Tensor* b);
void tensor_maximum_gpu(Tensor* out, const Tensor* a, const Tensor* b);
void tensor_minimum_gpu(Tensor* out, const Tensor* a, const Tensor* b);
void tensor_logaddexp_gpu(Tensor* out, const Tensor* a, const Tensor* b);
void tensor_logaddexp2_gpu(Tensor* out, const Tensor* a, const Tensor* b);
// Clamp / remaining activations GPU
void tensor_clamp_gpu(Tensor* out, const Tensor* in, float lo, float hi);
void tensor_hardtanh_gpu(Tensor* out, const Tensor* in, float min_val, float max_val);
void tensor_celu_gpu(Tensor* out, const Tensor* in, float alpha);
void tensor_softsign_gpu(Tensor* out, const Tensor* in);
void tensor_rrelu_gpu(Tensor* out, const Tensor* in, float lower, float upper);
void tensor_threshold_gpu(Tensor* out, const Tensor* in, float threshold, float value);

// Half Precision
HalfTensor* tensor_to_half(const Tensor* t);
Tensor* tensor_from_half(const HalfTensor* t);
void half_tensor_free(HalfTensor* t);

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
void tensor_gelu(Tensor* out, const Tensor* in);
void tensor_selu(Tensor* out, const Tensor* in);
void tensor_selu_inplace(Tensor* t);
void tensor_mish(Tensor* out, const Tensor* in);
void tensor_mish_inplace(Tensor* t);
void tensor_hardswish(Tensor* out, const Tensor* in);
void tensor_hardswish_inplace(Tensor* t);
void tensor_hardsigmoid(Tensor* out, const Tensor* in);
void tensor_hardsigmoid_inplace(Tensor* t);
void tensor_hardtanh(Tensor* out, const Tensor* in, float min_val, float max_val);
void tensor_hardtanh_inplace(Tensor* t, float min_val, float max_val);
void tensor_celu(Tensor* out, const Tensor* in, float alpha);
void tensor_celu_inplace(Tensor* t, float alpha);
void tensor_softsign(Tensor* out, const Tensor* in);
void tensor_softsign_inplace(Tensor* t);
void tensor_log_softmax(Tensor* out, const Tensor* in);
void tensor_log_softmax_inplace(Tensor* t);
void tensor_prelu(Tensor* out, const Tensor* in, const Tensor* weight);
void tensor_rrelu(Tensor* out, const Tensor* in, float lower, float upper);
void tensor_rrelu_inplace(Tensor* t, float lower, float upper);
void tensor_threshold(Tensor* out, const Tensor* in, float threshold, float value);
void tensor_threshold_inplace(Tensor* t, float threshold, float value);

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

// ============================================================
// Math Operations (rpl_math.c)
// ============================================================

// Trig
Tensor* tensor_sin(const Tensor* t); void tensor_sin_inplace(Tensor* t);
Tensor* tensor_cos(const Tensor* t); void tensor_cos_inplace(Tensor* t);
Tensor* tensor_tan(const Tensor* t); void tensor_tan_inplace(Tensor* t);
Tensor* tensor_asin(const Tensor* t); void tensor_asin_inplace(Tensor* t);
Tensor* tensor_acos(const Tensor* t); void tensor_acos_inplace(Tensor* t);
Tensor* tensor_atan(const Tensor* t); void tensor_atan_inplace(Tensor* t);
Tensor* tensor_atan2(const Tensor* a, const Tensor* b);
Tensor* tensor_hypot(const Tensor* a, const Tensor* b);

// Hyperbolic
Tensor* tensor_sinh(const Tensor* t); void tensor_sinh_inplace(Tensor* t);
Tensor* tensor_cosh(const Tensor* t); void tensor_cosh_inplace(Tensor* t);
Tensor* tensor_asinh(const Tensor* t); void tensor_asinh_inplace(Tensor* t);
Tensor* tensor_acosh(const Tensor* t); void tensor_acosh_inplace(Tensor* t);
Tensor* tensor_atanh(const Tensor* t); void tensor_atanh_inplace(Tensor* t);

// Exp/Log
Tensor* tensor_exp(const Tensor* t); void tensor_exp_inplace(Tensor* t);
Tensor* tensor_expm1(const Tensor* t); void tensor_expm1_inplace(Tensor* t);
Tensor* tensor_exp2(const Tensor* t); void tensor_exp2_inplace(Tensor* t);
Tensor* tensor_log(const Tensor* t); void tensor_log_inplace(Tensor* t);
Tensor* tensor_log2(const Tensor* t); void tensor_log2_inplace(Tensor* t);
Tensor* tensor_log10(const Tensor* t); void tensor_log10_inplace(Tensor* t);
Tensor* tensor_log1p(const Tensor* t); void tensor_log1p_inplace(Tensor* t);
Tensor* tensor_logaddexp(const Tensor* a, const Tensor* b);
Tensor* tensor_logaddexp2(const Tensor* a, const Tensor* b);

// Rounding
Tensor* tensor_round_op(const Tensor* t); void tensor_round_op_inplace(Tensor* t);
Tensor* tensor_floor_op(const Tensor* t); void tensor_floor_op_inplace(Tensor* t);
Tensor* tensor_ceil_op(const Tensor* t); void tensor_ceil_op_inplace(Tensor* t);
Tensor* tensor_trunc_op(const Tensor* t); void tensor_trunc_op_inplace(Tensor* t);
Tensor* tensor_frac(const Tensor* t); void tensor_frac_inplace(Tensor* t);

// Power/Root
Tensor* tensor_pow_op(const Tensor* a, const Tensor* b);
Tensor* tensor_sqrt_op(const Tensor* t); void tensor_sqrt_op_inplace(Tensor* t);
Tensor* tensor_rsqrt(const Tensor* t); void tensor_rsqrt_inplace(Tensor* t);
Tensor* tensor_square(const Tensor* t); void tensor_square_inplace(Tensor* t);
Tensor* tensor_cbrt(const Tensor* t); void tensor_cbrt_inplace(Tensor* t);
Tensor* tensor_reciprocal(const Tensor* t); void tensor_reciprocal_inplace(Tensor* t);

// Abs/Sign/Clamp
Tensor* tensor_abs_op(const Tensor* t); void tensor_abs_op_inplace(Tensor* t);
Tensor* tensor_neg(const Tensor* t); void tensor_neg_inplace(Tensor* t);
Tensor* tensor_sign(const Tensor* t); void tensor_sign_inplace(Tensor* t);
Tensor* tensor_signbit_op(const Tensor* t); void tensor_signbit_op_inplace(Tensor* t);
Tensor* tensor_copysign_op(const Tensor* a, const Tensor* b);
Tensor* tensor_heaviside(const Tensor* a, const Tensor* b);
Tensor* tensor_clamp(const Tensor* t, float lo, float hi);
void tensor_clamp_inplace(Tensor* t, float lo, float hi);
Tensor* tensor_nan_to_num(const Tensor* t, float nan_v, float posinf_v, float neginf_v);
Tensor* tensor_lerp(const Tensor* a, const Tensor* b, float weight);

// Angular
Tensor* tensor_deg2rad(const Tensor* t); void tensor_deg2rad_inplace(Tensor* t);
Tensor* tensor_rad2deg(const Tensor* t); void tensor_rad2deg_inplace(Tensor* t);

// Special
Tensor* tensor_erf(const Tensor* t); void tensor_erf_inplace(Tensor* t);
Tensor* tensor_erfc(const Tensor* t); void tensor_erfc_inplace(Tensor* t);
Tensor* tensor_erfinv(const Tensor* t); void tensor_erfinv_inplace(Tensor* t);
Tensor* tensor_lgamma_op(const Tensor* t); void tensor_lgamma_op_inplace(Tensor* t);
Tensor* tensor_digamma(const Tensor* t); void tensor_digamma_inplace(Tensor* t);
Tensor* tensor_sinc(const Tensor* t); void tensor_sinc_inplace(Tensor* t);
Tensor* tensor_i0(const Tensor* t); void tensor_i0_inplace(Tensor* t);
Tensor* tensor_logit(const Tensor* t); void tensor_logit_inplace(Tensor* t);

// Binary math
Tensor* tensor_fmod_op(const Tensor* a, const Tensor* b);
Tensor* tensor_remainder_op(const Tensor* a, const Tensor* b);
Tensor* tensor_floor_divide(const Tensor* a, const Tensor* b);
Tensor* tensor_true_divide(const Tensor* a, const Tensor* b);
Tensor* tensor_sub(const Tensor* a, const Tensor* b);
Tensor* tensor_div(const Tensor* a, const Tensor* b);
Tensor* tensor_xlogy(const Tensor* a, const Tensor* b);
Tensor* tensor_addcdiv(const Tensor* input, const Tensor* t1, const Tensor* t2, float value);
Tensor* tensor_addcmul(const Tensor* input, const Tensor* t1, const Tensor* t2, float value);

// ============================================================
// Tensor Manipulation (rpl_manipulation.c)
// ============================================================

Tensor* tensor_reshape(const Tensor* t, uint32_t dims, const uint32_t* shape);
Tensor* tensor_squeeze(const Tensor* t);
Tensor* tensor_unsqueeze(const Tensor* t, int32_t dim);
Tensor* tensor_flatten(const Tensor* t, int32_t start_dim, int32_t end_dim);
Tensor* tensor_ravel(const Tensor* t);
Tensor* tensor_t_op(const Tensor* t);
Tensor* tensor_transpose(const Tensor* t, int32_t dim0, int32_t dim1);
Tensor* tensor_permute(const Tensor* t, const uint32_t* perm);
Tensor* tensor_movedim(const Tensor* t, int32_t src, int32_t dst);
Tensor* tensor_swapaxes(const Tensor* t, int32_t a, int32_t b);
Tensor* tensor_cat(const Tensor** tensors, uint32_t num, int32_t dim);
Tensor* tensor_stack(const Tensor** tensors, uint32_t num, int32_t dim);
Tensor* tensor_hstack(const Tensor** tensors, uint32_t num);
Tensor* tensor_vstack(const Tensor** tensors, uint32_t num);
Tensor** tensor_chunk(const Tensor* t, uint32_t chunks, int32_t dim, uint32_t* out_num);
Tensor** tensor_split(const Tensor* t, uint32_t sections, int32_t dim, uint32_t* out_num);
Tensor* tensor_index_select(const Tensor* t, int32_t dim, const uint32_t* indices, uint32_t n_idx);
Tensor* tensor_gather(const Tensor* t, int32_t dim, const Tensor* index);
Tensor* tensor_where_cond(const Tensor* cond, const Tensor* x, const Tensor* y);
Tensor* tensor_masked_select(const Tensor* t, const Tensor* mask, uint32_t* out_size);
Tensor* tensor_nonzero_indices(const Tensor* t, uint32_t* count);
Tensor* tensor_flip(const Tensor* t, const int32_t* dims, uint32_t n_dims);
Tensor* tensor_fliplr(const Tensor* t);
Tensor* tensor_flipud(const Tensor* t);
Tensor* tensor_roll(const Tensor* t, int32_t shift, int32_t dim);
Tensor* tensor_clone(const Tensor* t);
Tensor* tensor_tile(const Tensor* t, const uint32_t* reps, uint32_t n_reps);
Tensor* tensor_narrow(const Tensor* t, int32_t dim, uint32_t start, uint32_t length);

// ============================================================
// Reduction Operations (rpl_reduce.c)
// ============================================================

// Full reductions
float tensor_sum_all(const Tensor* t);
float tensor_prod_all(const Tensor* t);
float tensor_mean_all(const Tensor* t);
float tensor_var_all(const Tensor* t, bool unbiased);
float tensor_std_all(const Tensor* t, bool unbiased);
float tensor_max_all(const Tensor* t);
float tensor_min_all(const Tensor* t);
uint32_t tensor_argmax_all(const Tensor* t);
uint32_t tensor_argmin_all(const Tensor* t);
float tensor_norm_all(const Tensor* t, float p);
float tensor_logsumexp_all(const Tensor* t);
uint32_t tensor_count_nonzero_all(const Tensor* t);
float tensor_median_all(const Tensor* t);

// NaN-safe
float tensor_nansum_all(const Tensor* t);
float tensor_nanmean_all(const Tensor* t);
float tensor_nanprod_all(const Tensor* t);
float tensor_nanmax_all(const Tensor* t);
float tensor_nanmin_all(const Tensor* t);

// Axis reductions
Tensor* tensor_sum(const Tensor* t, int32_t dim);
Tensor* tensor_prod(const Tensor* t, int32_t dim);
Tensor* tensor_mean(const Tensor* t, int32_t dim);
Tensor* tensor_var(const Tensor* t, int32_t dim, bool unbiased);
Tensor* tensor_std(const Tensor* t, int32_t dim, bool unbiased);
Tensor* tensor_max_dim(const Tensor* t, int32_t dim);
Tensor* tensor_min_dim(const Tensor* t, int32_t dim);
Tensor* tensor_argmax_dim(const Tensor* t, int32_t dim);
Tensor* tensor_argmin_dim(const Tensor* t, int32_t dim);

// Cumulative
Tensor* tensor_cumsum(const Tensor* t, int32_t dim);
Tensor* tensor_cumprod(const Tensor* t, int32_t dim);
Tensor* tensor_cummax(const Tensor* t, int32_t dim);
Tensor* tensor_cummin(const Tensor* t, int32_t dim);

// Diff
Tensor* tensor_diff(const Tensor* t, int32_t dim);
bool tensor_all(const Tensor* t);
bool tensor_any(const Tensor* t);
float tensor_dist(const Tensor* a, const Tensor* b, float p);

// ============================================================
// Comparison & Logic (rpl_compare.c)
// ============================================================

Tensor* tensor_eq(const Tensor* a, const Tensor* b);
Tensor* tensor_ne(const Tensor* a, const Tensor* b);
Tensor* tensor_lt(const Tensor* a, const Tensor* b);
Tensor* tensor_le(const Tensor* a, const Tensor* b);
Tensor* tensor_gt(const Tensor* a, const Tensor* b);
Tensor* tensor_ge(const Tensor* a, const Tensor* b);
bool tensor_equal(const Tensor* a, const Tensor* b);
bool tensor_allclose(const Tensor* a, const Tensor* b, float rtol, float atol);
Tensor* tensor_isclose(const Tensor* a, const Tensor* b, float rtol, float atol);
Tensor* tensor_logical_and(const Tensor* a, const Tensor* b);
Tensor* tensor_logical_or(const Tensor* a, const Tensor* b);
Tensor* tensor_logical_not(const Tensor* t);
Tensor* tensor_logical_xor(const Tensor* a, const Tensor* b);
Tensor* tensor_isnan_op(const Tensor* t);
Tensor* tensor_isinf_op(const Tensor* t);
Tensor* tensor_isfinite_op(const Tensor* t);
Tensor* tensor_isposinf(const Tensor* t);
Tensor* tensor_isneginf(const Tensor* t);
Tensor* tensor_maximum(const Tensor* a, const Tensor* b);
Tensor* tensor_minimum(const Tensor* a, const Tensor* b);
Tensor* tensor_fmax(const Tensor* a, const Tensor* b);
Tensor* tensor_fmin(const Tensor* a, const Tensor* b);
Tensor* tensor_sort_op(const Tensor* t, int32_t dim, bool descending, Tensor** indices);
Tensor* tensor_argsort(const Tensor* t, int32_t dim, bool descending);
Tensor* tensor_topk(const Tensor* t, uint32_t k, int32_t dim, bool largest);
Tensor* tensor_unique(const Tensor* t, uint32_t* out_count);
Tensor* tensor_isin(const Tensor* elements, const Tensor* test);

// ============================================================
// Linear Algebra (rpl_linalg.c)
// ============================================================

float tensor_dot(const Tensor* a, const Tensor* b);
float tensor_vdot(const Tensor* a, const Tensor* b);
Tensor* tensor_inner(const Tensor* a, const Tensor* b);
Tensor* tensor_outer(const Tensor* a, const Tensor* b);
Tensor* tensor_mm(const Tensor* a, const Tensor* b);
Tensor* tensor_mv(const Tensor* mat, const Tensor* vec);
Tensor* tensor_bmm(const Tensor* a, const Tensor* b);
Tensor* tensor_addmm(const Tensor* input, const Tensor* m1, const Tensor* m2, float beta, float alpha);
Tensor* tensor_addr(const Tensor* input, const Tensor* v1, const Tensor* v2, float beta, float alpha);
float tensor_trace(const Tensor* t);
Tensor* tensor_diag(const Tensor* t, int32_t diagonal);
Tensor* tensor_tril(const Tensor* t, int32_t diagonal);
Tensor* tensor_triu(const Tensor* t, int32_t diagonal);
Tensor* tensor_eye(uint32_t n);
Tensor* tensor_cross(const Tensor* a, const Tensor* b);
float tensor_det(const Tensor* t);
Tensor* tensor_inverse(const Tensor* t);
Tensor* tensor_cholesky(const Tensor* t);
Tensor* tensor_matrix_power(const Tensor* t, int32_t n);
Tensor* tensor_kron(const Tensor* a, const Tensor* b);
Tensor* tensor_tensordot(const Tensor* a, const Tensor* b, uint32_t dims);

// ============================================================
// FFT (rpl_fft.c)
// ============================================================

Tensor* tensor_fft(const Tensor* t);
Tensor* tensor_ifft(const Tensor* t);
Tensor* tensor_rfft(const Tensor* t);
Tensor* tensor_irfft(const Tensor* t, uint32_t n);
Tensor* tensor_stft(const Tensor* t, uint32_t n_fft, uint32_t hop_length);

// ============================================================
// Random (rpl_random.c)
// ============================================================

void rpl_manual_seed(uint64_t seed);
void rpl_seed(void);
Tensor* tensor_zeros(uint32_t dims, const uint32_t* shape);
Tensor* tensor_ones(uint32_t dims, const uint32_t* shape);
Tensor* tensor_full(uint32_t dims, const uint32_t* shape, float value);
Tensor* tensor_zeros_like(const Tensor* t);
Tensor* tensor_ones_like(const Tensor* t);
Tensor* tensor_full_like(const Tensor* t, float value);
Tensor* tensor_empty(uint32_t dims, const uint32_t* shape);
Tensor* tensor_empty_like(const Tensor* t);
Tensor* tensor_arange(float start, float end, float step);
Tensor* tensor_linspace(float start, float end, uint32_t steps);
Tensor* tensor_logspace(float start, float end, uint32_t steps, float base);
Tensor* tensor_rand(uint32_t dims, const uint32_t* shape);
Tensor* tensor_randn(uint32_t dims, const uint32_t* shape);
Tensor* tensor_randint(int32_t low, int32_t high, uint32_t dims, const uint32_t* shape);
Tensor* tensor_randperm(uint32_t n);
Tensor* tensor_rand_like(const Tensor* t);
Tensor* tensor_randn_like(const Tensor* t);
Tensor* tensor_bernoulli(const Tensor* probs);
Tensor* tensor_normal(float mean, float std, uint32_t dims, const uint32_t* shape);
Tensor* tensor_poisson_sample(const Tensor* rates);
Tensor* tensor_multinomial(const Tensor* probs, uint32_t num_samples, bool replacement);
void tensor_meshgrid(const Tensor** inputs, uint32_t n_inputs, Tensor** outputs);

// ============================================================
// Utilities (rpl_util.c)
// ============================================================

uint32_t tensor_numel(const Tensor* t);
bool tensor_is_floating_point(const Tensor* t);
bool tensor_is_nonzero(const Tensor* t);
Tensor* tensor_contiguous(const Tensor* t);
Tensor* tensor_broadcast_to(const Tensor* t, uint32_t dims, const uint32_t* shape);
Tensor* tensor_atleast_1d(const Tensor* t);
Tensor* tensor_atleast_2d(const Tensor* t);
Tensor* tensor_atleast_3d(const Tensor* t);
Tensor* tensor_block_diag(const Tensor** tensors, uint32_t num);
Tensor* tensor_vander(const Tensor* x, uint32_t N, bool increasing);
Tensor* tensor_hann_window(uint32_t size);
Tensor* tensor_hamming_window(uint32_t size);
Tensor* tensor_blackman_window(uint32_t size);
Tensor* tensor_bartlett_window(uint32_t size);
Tensor* tensor_kaiser_window(uint32_t size, float beta);
Tensor* tensor_convolve(const Tensor* a, const Tensor* v);
Tensor* tensor_interp(const Tensor* x, const Tensor* xp, const Tensor* fp);
Tensor* tensor_bincount(const Tensor* t, uint32_t minlength);
Tensor* tensor_histc(const Tensor* t, uint32_t bins, float min_val, float max_val);
float tensor_trapezoid(const Tensor* y, float dx);
Tensor* tensor_corrcoef(const Tensor* t);
Tensor* tensor_cdist(const Tensor* x1, const Tensor* x2, float p);

#ifdef __cplusplus
}
#endif

#endif // RPL_H
