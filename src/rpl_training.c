/*
 * RPiTorch Training Infrastructure - Pure C Implementation
 * Optimizers, Early Stopping, Gradient Clipping, Checkpointing
 */

#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <time.h>

// ============================================================
// Optimizer Base Structure
// ============================================================



struct Optimizer {
    OptimizerType type;
    float learning_rate;
    float weight_decay;
    uint32_t num_params;
    Tensor** parameters;
    
    // Optimizer-specific state
    void* state;
};

// ============================================================
// SGD Optimizer
// ============================================================

struct SGDState {
    float momentum;
    float dampening;
    bool nesterov;
    Tensor** velocity;  // Momentum buffers
};

Optimizer* optimizer_sgd_create(Tensor** parameters, uint32_t num_params,
                                float lr, float momentum, float dampening,
                                float weight_decay, bool nesterov) {
    Optimizer* opt = (Optimizer*)calloc(1, sizeof(Optimizer));
    opt->type = OPTIMIZER_SGD;
    opt->learning_rate = lr;
    opt->weight_decay = weight_decay;
    opt->num_params = num_params;
    opt->parameters = parameters;
    
    struct SGDState* state = (struct SGDState*)calloc(1, sizeof(struct SGDState));
    state->momentum = momentum;
    state->dampening = dampening;
    state->nesterov = nesterov;
    
    if (momentum > 0.0f) {
        state->velocity = (Tensor**)calloc(num_params, sizeof(Tensor*));
        for (uint32_t i = 0; i < num_params; i++) {
            state->velocity[i] = tensor_create(parameters[i]->dims, 
                                              parameters[i]->shape, false);
            tensor_fill(state->velocity[i], 0.0f);
        }
    }
    
    opt->state = state;
    return opt;
}

void optimizer_sgd_step(Optimizer* opt) {
    struct SGDState* state = (struct SGDState*)opt->state;
    
    #pragma omp parallel for
    for (uint32_t p = 0; p < opt->num_params; p++) {
        Tensor* param = opt->parameters[p];
        if (!param->grad) continue;
        
        float* d_p = param->grad;
        
        // Weight decay
        if (opt->weight_decay != 0.0f) {
            #pragma omp simd
            for (uint32_t i = 0; i < param->size; i++) {
                d_p[i] += opt->weight_decay * param->data[i];
            }
        }
        
        // Momentum
        if (state->momentum != 0.0f) {
            Tensor* v = state->velocity[p];
            
            #pragma omp simd
            for (uint32_t i = 0; i < param->size; i++) {
                v->data[i] = state->momentum * v->data[i] + 
                            (1.0f - state->dampening) * d_p[i];
            }
            
            if (state->nesterov) {
                #pragma omp simd
                for (uint32_t i = 0; i < param->size; i++) {
                    d_p[i] += state->momentum * v->data[i];
                }
            } else {
                d_p = v->data;
            }
        }
        
        // Update parameters
        #pragma omp simd
        for (uint32_t i = 0; i < param->size; i++) {
            param->data[i] -= opt->learning_rate * d_p[i];
        }
    }
}

// ============================================================
// Adam Optimizer
// ============================================================

struct AdamState {
    float beta1;
    float beta2;
    float epsilon;
    uint64_t step;
    Tensor** m;  // First moment
    Tensor** v;  // Second moment
};

Optimizer* optimizer_adam_create(Tensor** parameters, uint32_t num_params,
                                float lr, float beta1, float beta2,
                                float epsilon, float weight_decay) {
    Optimizer* opt = (Optimizer*)calloc(1, sizeof(Optimizer));
    opt->type = OPTIMIZER_ADAM;
    opt->learning_rate = lr;
    opt->weight_decay = weight_decay;
    opt->num_params = num_params;
    opt->parameters = parameters;
    
    struct AdamState* state = (struct AdamState*)calloc(1, sizeof(struct AdamState));
    state->beta1 = beta1;
    state->beta2 = beta2;
    state->epsilon = epsilon;
    state->step = 0;
    
    state->m = (Tensor**)calloc(num_params, sizeof(Tensor*));
    state->v = (Tensor**)calloc(num_params, sizeof(Tensor*));
    
    for (uint32_t i = 0; i < num_params; i++) {
        state->m[i] = tensor_create(parameters[i]->dims, parameters[i]->shape, false);
        state->v[i] = tensor_create(parameters[i]->dims, parameters[i]->shape, false);
        tensor_fill(state->m[i], 0.0f);
        tensor_fill(state->v[i], 0.0f);
    }
    
    opt->state = state;
    return opt;
}

void optimizer_adam_step(Optimizer* opt) {
    struct AdamState* state = (struct AdamState*)opt->state;
    state->step++;
    
    float bias_correction1 = 1.0f - powf(state->beta1, state->step);
    float bias_correction2 = 1.0f - powf(state->beta2, state->step);
    float step_size = opt->learning_rate * sqrtf(bias_correction2) / bias_correction1;
    
    #pragma omp parallel for
    for (uint32_t p = 0; p < opt->num_params; p++) {
        Tensor* param = opt->parameters[p];
        if (!param->grad) continue;
        
        float* grad = param->grad;
        Tensor* m = state->m[p];
        Tensor* v = state->v[p];
        
        // Weight decay
        if (opt->weight_decay != 0.0f) {
            #pragma omp simd
            for (uint32_t i = 0; i < param->size; i++) {
                grad[i] += opt->weight_decay * param->data[i];
            }
        }
        
        // Update biased first moment estimate
        #pragma omp simd
        for (uint32_t i = 0; i < param->size; i++) {
            m->data[i] = state->beta1 * m->data[i] + (1.0f - state->beta1) * grad[i];
        }
        
        // Update biased second raw moment estimate
        #pragma omp simd
        for (uint32_t i = 0; i < param->size; i++) {
            v->data[i] = state->beta2 * v->data[i] + 
                        (1.0f - state->beta2) * grad[i] * grad[i];
        }
        
        // Update parameters
        #pragma omp simd
        for (uint32_t i = 0; i < param->size; i++) {
            param->data[i] -= step_size * m->data[i] / (sqrtf(v->data[i]) + state->epsilon);
        }
    }
}

// ============================================================
// RMSprop Optimizer
// ============================================================

struct RMSpropState {
    float alpha;     // Decay rate
    float epsilon;
    float momentum;
    Tensor** square_avg;  // Running average of squared gradients
    Tensor** momentum_buffer;  // Momentum buffers (if momentum > 0)
};

Optimizer* optimizer_rmsprop_create(Tensor** parameters, uint32_t num_params,
                                   float lr, float alpha, float epsilon, 
                                   float momentum, float weight_decay) {
    Optimizer* opt = (Optimizer*)calloc(1, sizeof(Optimizer));
    opt->type = OPTIMIZER_RMSPROP;
    opt->learning_rate = lr;
    opt->weight_decay = weight_decay;
    opt->num_params = num_params;
    opt->parameters = parameters;
    
    struct RMSpropState* state = (struct RMSpropState*)calloc(1, sizeof(struct RMSpropState));
    state->alpha = alpha;
    state->epsilon = epsilon;
    state->momentum = momentum;
    
    state->square_avg = (Tensor**)calloc(num_params, sizeof(Tensor*));
    if (momentum > 0.0f) {
        state->momentum_buffer = (Tensor**)calloc(num_params, sizeof(Tensor*));
    }
    
    for (uint32_t i = 0; i < num_params; i++) {
        state->square_avg[i] = tensor_create(parameters[i]->dims, parameters[i]->shape, false);
        tensor_fill(state->square_avg[i], 0.0f);
        
        if (momentum > 0.0f) {
            state->momentum_buffer[i] = tensor_create(parameters[i]->dims, parameters[i]->shape, false);
            tensor_fill(state->momentum_buffer[i], 0.0f);
        }
    }
    
    opt->state = state;
    return opt;
}

void optimizer_rmsprop_step(Optimizer* opt) {
    struct RMSpropState* state = (struct RMSpropState*)opt->state;
    
    #pragma omp parallel for
    for (uint32_t p = 0; p < opt->num_params; p++) {
        Tensor* param = opt->parameters[p];
        if (!param->grad) continue;
        
        float* grad = param->grad;
        Tensor* square_avg = state->square_avg[p];
        
        // Weight decay
        if (opt->weight_decay != 0.0f) {
            #pragma omp simd
            for (uint32_t i = 0; i < param->size; i++) {
                grad[i] += opt->weight_decay * param->data[i];
            }
        }
        
        // Update square average
        #pragma omp simd
        for (uint32_t i = 0; i < param->size; i++) {
            square_avg->data[i] = state->alpha * square_avg->data[i] + 
                                 (1.0f - state->alpha) * grad[i] * grad[i];
        }
        
        if (state->momentum > 0.0f) {
            Tensor* buf = state->momentum_buffer[p];
            
            #pragma omp simd
            for (uint32_t i = 0; i < param->size; i++) {
                buf->data[i] = state->momentum * buf->data[i] + 
                              opt->learning_rate * grad[i] / (sqrtf(square_avg->data[i]) + state->epsilon);
                param->data[i] -= buf->data[i];
            }
        } else {
            #pragma omp simd
            for (uint32_t i = 0; i < param->size; i++) {
                param->data[i] -= opt->learning_rate * grad[i] / 
                                 (sqrtf(square_avg->data[i]) + state->epsilon);
            }
        }
    }
}

// ============================================================
// Adagrad Optimizer
// ============================================================

struct AdagradState {
    float epsilon;
    Tensor** sum_squares;  // Accumulated squared gradients
};

Optimizer* optimizer_adagrad_create(Tensor** parameters, uint32_t num_params,
                                   float lr, float epsilon, float weight_decay) {
    Optimizer* opt = (Optimizer*)calloc(1, sizeof(Optimizer));
    opt->type = OPTIMIZER_ADAGRAD;
    opt->learning_rate = lr;
    opt->weight_decay = weight_decay;
    opt->num_params = num_params;
    opt->parameters = parameters;
    
    struct AdagradState* state = (struct AdagradState*)calloc(1, sizeof(struct AdagradState));
    state->epsilon = epsilon;
    
    state->sum_squares = (Tensor**)calloc(num_params, sizeof(Tensor*));
    for (uint32_t i = 0; i < num_params; i++) {
        state->sum_squares[i] = tensor_create(parameters[i]->dims, parameters[i]->shape, false);
        tensor_fill(state->sum_squares[i], 0.0f);
    }
    
    opt->state = state;
    return opt;
}

void optimizer_adagrad_step(Optimizer* opt) {
    struct AdagradState* state = (struct AdagradState*)opt->state;
    
    #pragma omp parallel for
    for (uint32_t p = 0; p < opt->num_params; p++) {
        Tensor* param = opt->parameters[p];
        if (!param->grad) continue;
        
        float* grad = param->grad;
        Tensor* sum_sq = state->sum_squares[p];
        
        // Weight decay
        if (opt->weight_decay != 0.0f) {
            #pragma omp simd
            for (uint32_t i = 0; i < param->size; i++) {
                grad[i] += opt->weight_decay * param->data[i];
            }
        }
        
        // Accumulate squared gradients
        #pragma omp simd
        for (uint32_t i = 0; i < param->size; i++) {
            sum_sq->data[i] += grad[i] * grad[i];
        }
        
        // Update parameters
        #pragma omp simd
        for (uint32_t i = 0; i < param->size; i++) {
            param->data[i] -= opt->learning_rate * grad[i] / 
                             (sqrtf(sum_sq->data[i]) + state->epsilon);
        }
    }
}

// ============================================================
// LAMB Optimizer (Layer-wise Adaptive Moments)
// ============================================================

struct LAMBState {
    float beta1;
    float beta2;
    float epsilon;
    uint64_t step;
    Tensor** m;  // First moment
    Tensor** v;  // Second moment
};

Optimizer* optimizer_lamb_create(Tensor** parameters, uint32_t num_params,
                                float lr, float beta1, float beta2,
                                float epsilon, float weight_decay) {
    Optimizer* opt = (Optimizer*)calloc(1, sizeof(Optimizer));
    opt->type = OPTIMIZER_LAMB;
    opt->learning_rate = lr;
    opt->weight_decay = weight_decay;
    opt->num_params = num_params;
    opt->parameters = parameters;
    
    struct LAMBState* state = (struct LAMBState*)calloc(1, sizeof(struct LAMBState));
    state->beta1 = beta1;
    state->beta2 = beta2;
    state->epsilon = epsilon;
    state->step = 0;
    
    state->m = (Tensor**)calloc(num_params, sizeof(Tensor*));
    state->v = (Tensor**)calloc(num_params, sizeof(Tensor*));
    
    for (uint32_t i = 0; i < num_params; i++) {
        state->m[i] = tensor_create(parameters[i]->dims, parameters[i]->shape, false);
        state->v[i] = tensor_create(parameters[i]->dims, parameters[i]->shape, false);
        tensor_fill(state->m[i], 0.0f);
        tensor_fill(state->v[i], 0.0f);
    }
    
    opt->state = state;
    return opt;
}

void optimizer_lamb_step(Optimizer* opt) {
    struct LAMBState* state = (struct LAMBState*)opt->state;
    state->step++;
    
    float bias_correction1 = 1.0f - powf(state->beta1, state->step);
    float bias_correction2 = 1.0f - powf(state->beta2, state->step);
    
    #pragma omp parallel for
    for (uint32_t p = 0; p < opt->num_params; p++) {
        Tensor* param = opt->parameters[p];
        if (!param->grad) continue;
        
        float* grad = param->grad;
        Tensor* m = state->m[p];
        Tensor* v = state->v[p];
        
        // Update biased first moment estimate
        #pragma omp simd
        for (uint32_t i = 0; i < param->size; i++) {
            m->data[i] = state->beta1 * m->data[i] + (1.0f - state->beta1) * grad[i];
        }
        
        // Update biased second raw moment estimate
        #pragma omp simd
        for (uint32_t i = 0; i < param->size; i++) {
            v->data[i] = state->beta2 * v->data[i] + 
                        (1.0f - state->beta2) * grad[i] * grad[i];
        }
        
        // Compute Adam step
        float* adam_step = (float*)malloc(param->size * sizeof(float));
        
        #pragma omp simd
        for (uint32_t i = 0; i < param->size; i++) {
            float m_hat = m->data[i] / bias_correction1;
            float v_hat = v->data[i] / bias_correction2;
            adam_step[i] = m_hat / (sqrtf(v_hat) + state->epsilon);
            
            if (opt->weight_decay != 0.0f) {
                adam_step[i] += opt->weight_decay * param->data[i];
            }
        }
        
        // Compute layer-wise trust ratio
        float param_norm = 0.0f, update_norm = 0.0f;
        
        for (uint32_t i = 0; i < param->size; i++) {
            param_norm += param->data[i] * param->data[i];
            update_norm += adam_step[i] * adam_step[i];
        }
        
        param_norm = sqrtf(param_norm);
        update_norm = sqrtf(update_norm);
        
        float trust_ratio = 1.0f;
        if (param_norm > 0.0f && update_norm > 0.0f) {
            trust_ratio = param_norm / update_norm;
        }
        
        // Update parameters with trust ratio
        #pragma omp simd
        for (uint32_t i = 0; i < param->size; i++) {
            param->data[i] -= opt->learning_rate * trust_ratio * adam_step[i];
        }
        
        free(adam_step);
    }
}

// ============================================================
// AdamW Optimizer (Adam with Decoupled Weight Decay)
// ============================================================

Optimizer* optimizer_adamw_create(Tensor** parameters, uint32_t num_params,
                                  float lr, float beta1, float beta2,
                                  float epsilon, float weight_decay) {
    // AdamW uses the same state structure as Adam
    Optimizer* opt = optimizer_adam_create(parameters, num_params, lr, beta1, beta2, epsilon, 0.0f);
    opt->type = OPTIMIZER_ADAMW;
    opt->weight_decay = weight_decay;
    return opt;
}

void optimizer_adamw_step(Optimizer* opt) {
    struct AdamState* state = (struct AdamState*)opt->state;
    state->step++;
    
    float bias_correction1 = 1.0f - powf(state->beta1, state->step);
    float bias_correction2 = 1.0f - powf(state->beta2, state->step);
    float step_size = opt->learning_rate * sqrtf(bias_correction2) / bias_correction1;
    
    #pragma omp parallel for
    for (uint32_t p = 0; p < opt->num_params; p++) {
        Tensor* param = opt->parameters[p];
        if (!param->grad) continue;
        
        float* grad = param->grad;
        Tensor* m = state->m[p];
        Tensor* v = state->v[p];
        
        // Update biased first moment estimate (NO weight decay in gradient)
        #pragma omp simd
        for (uint32_t i = 0; i < param->size; i++) {
            m->data[i] = state->beta1 * m->data[i] + (1.0f - state->beta1) * grad[i];
        }
        
        // Update biased second raw moment estimate
        #pragma omp simd
        for (uint32_t i = 0; i < param->size; i++) {
            v->data[i] = state->beta2 * v->data[i] + 
                        (1.0f - state->beta2) * grad[i] * grad[i];
        }
        
        // Update parameters with DECOUPLED weight decay
        #pragma omp simd
        for (uint32_t i = 0; i < param->size; i++) {
            // AdamW: weight decay applied directly to parameters, not gradients
            param->data[i] = param->data[i] * (1.0f - opt->learning_rate * opt->weight_decay) -
                            step_size * m->data[i] / (sqrtf(v->data[i]) + state->epsilon);
        }
    }
}

void optimizer_step(Optimizer* opt) {
    switch (opt->type) {
        case OPTIMIZER_SGD:
            optimizer_sgd_step(opt);
            break;
        case OPTIMIZER_ADAM:
            optimizer_adam_step(opt);
            break;
        case OPTIMIZER_RMSPROP:
            optimizer_rmsprop_step(opt);
            break;
        case OPTIMIZER_ADAGRAD:
            optimizer_adagrad_step(opt);
            break;
        case OPTIMIZER_ADAMW:
            optimizer_adamw_step(opt);
            break;
        case OPTIMIZER_LAMB:
            optimizer_lamb_step(opt);
            break;
    }
}

void optimizer_zero_grad(Optimizer* opt) {
    for (uint32_t i = 0; i < opt->num_params; i++) {
        tensor_zero_grad(opt->parameters[i]);
    }
}

void optimizer_free(Optimizer* opt) {
    if (!opt) return;
    
    if (opt->type == OPTIMIZER_SGD) {
        struct SGDState* state = (struct SGDState*)opt->state;
        if (state->velocity) {
            for (uint32_t i = 0; i < opt->num_params; i++) {
                tensor_free(state->velocity[i]);
            }
            free(state->velocity);
        }
        free(state);
    } else if (opt->type == OPTIMIZER_ADAM || opt->type == OPTIMIZER_ADAMW) {
        struct AdamState* state = (struct AdamState*)opt->state;
        for (uint32_t i = 0; i < opt->num_params; i++) {
            tensor_free(state->m[i]);
            tensor_free(state->v[i]);
        }
        free(state->m);
        free(state->v);
        free(state);
    } else if (opt->type == OPTIMIZER_RMSPROP) {
        struct RMSpropState* state = (struct RMSpropState*)opt->state;
        for (uint32_t i = 0; i < opt->num_params; i++) {
            tensor_free(state->square_avg[i]);
            if (state->momentum_buffer) {
                tensor_free(state->momentum_buffer[i]);
            }
        }
        free(state->square_avg);
        if (state->momentum_buffer) free(state->momentum_buffer);
        free(state);
    } else if (opt->type == OPTIMIZER_ADAGRAD) {
        struct AdagradState* state = (struct AdagradState*)opt->state;
        for (uint32_t i = 0; i < opt->num_params; i++) {
            tensor_free(state->sum_squares[i]);
        }
        free(state->sum_squares);
        free(state);
    } else if (opt->type == OPTIMIZER_LAMB) {
        struct LAMBState* state = (struct LAMBState*)opt->state;
        for (uint32_t i = 0; i < opt->num_params; i++) {
            tensor_free(state->m[i]);
            tensor_free(state->v[i]);
        }
        free(state->m);
        free(state->v);
        free(state);
    }
    
    free(opt);
}

// ============================================================
// Gradient Clipping
// ============================================================

float gradient_clip_norm(Tensor** parameters, uint32_t num_params, 
                        float max_norm, float norm_type) {
    float total_norm = 0.0f;
    
    if (norm_type == 2.0f) {
        // L2 norm
        #pragma omp parallel for reduction(+:total_norm)
        for (uint32_t i = 0; i < num_params; i++) {
            if (!parameters[i]->grad) continue;
            
            float param_norm = 0.0f;
            #pragma omp simd reduction(+:param_norm)
            for (uint32_t j = 0; j < parameters[i]->size; j++) {
                float g = parameters[i]->grad[j];
                param_norm += g * g;
            }
            total_norm += param_norm;
        }
        
        total_norm = sqrtf(total_norm);
        
        // Clip
        float clip_coef = max_norm / (total_norm + 1e-6f);
        if (clip_coef < 1.0f) {
            #pragma omp parallel for
            for (uint32_t i = 0; i < num_params; i++) {
                if (!parameters[i]->grad) continue;
                
                #pragma omp simd
                for (uint32_t j = 0; j < parameters[i]->size; j++) {
                    parameters[i]->grad[j] *= clip_coef;
                }
            }
        }
    }
    
    return total_norm;
}

// ============================================================
// Early Stopping
// ============================================================

struct EarlyStopping {
    float best_score;
    uint32_t best_epoch;
    uint32_t counter;
    uint32_t patience;
    float min_delta;
    bool minimize;
    bool should_stop;
};

EarlyStopping* early_stopping_create(uint32_t patience, float min_delta, bool minimize) {
    EarlyStopping* es = (EarlyStopping*)calloc(1, sizeof(EarlyStopping));
    es->patience = patience;
    es->min_delta = min_delta;
    es->minimize = minimize;
    es->best_score = minimize ? FLT_MAX : -FLT_MAX;
    es->counter = 0;
    es->should_stop = false;
    return es;
}

bool early_stopping_check(EarlyStopping* es, uint32_t epoch, float metric) {
    float score = es->minimize ? metric : -metric;
    
    if (score < es->best_score - es->min_delta) {
        es->best_score = score;
        es->best_epoch = epoch;
        es->counter = 0;
    } else {
        es->counter++;
        if (es->counter >= es->patience) {
            es->should_stop = true;
        }
    }
    
    return es->should_stop;
}

void early_stopping_free(EarlyStopping* es) {
    free(es);
}

// ============================================================
// Learning Rate Scheduler
// ============================================================

typedef enum {
    LR_STEP,
    LR_EXPONENTIAL,
    LR_COSINE,
    LR_WARMUP
} LRSchedulerType;

struct LRScheduler {
    LRSchedulerType type;
    Optimizer* optimizer;
    float base_lr;
    uint32_t current_epoch;
    
    // Type-specific parameters
    union {
        struct {
            uint32_t step_size;
            float gamma;
        } step;
        
        struct {
            float gamma;
        } exponential;
        
        struct {
            uint32_t T_max;
            float eta_min;
        } cosine;
        
        struct {
            uint32_t warmup_epochs;
            float warmup_start_lr;
        } warmup;
    } params;
};

LRScheduler* lr_scheduler_step_create(Optimizer* opt, uint32_t step_size, float gamma) {
    LRScheduler* sched = (LRScheduler*)calloc(1, sizeof(LRScheduler));
    sched->type = LR_STEP;
    sched->optimizer = opt;
    sched->base_lr = opt->learning_rate;
    sched->params.step.step_size = step_size;
    sched->params.step.gamma = gamma;
    return sched;
}

LRScheduler* lr_scheduler_warmup_create(Optimizer* opt, uint32_t warmup_epochs, 
                                       float warmup_start_lr) {
    LRScheduler* sched = (LRScheduler*)calloc(1, sizeof(LRScheduler));
    sched->type = LR_WARMUP;
    sched->optimizer = opt;
    sched->base_lr = opt->learning_rate;
    sched->params.warmup.warmup_epochs = warmup_epochs;
    sched->params.warmup.warmup_start_lr = warmup_start_lr;
    return sched;
}

void lr_scheduler_step(LRScheduler* sched) {
    sched->current_epoch++;
    
    switch (sched->type) {
        case LR_STEP:
            if (sched->current_epoch % sched->params.step.step_size == 0) {
                sched->optimizer->learning_rate *= sched->params.step.gamma;
            }
            break;
            
        case LR_EXPONENTIAL:
            sched->optimizer->learning_rate *= sched->params.exponential.gamma;
            break;
            
        case LR_COSINE: {
            float progress = (float)sched->current_epoch / sched->params.cosine.T_max;
            sched->optimizer->learning_rate = sched->params.cosine.eta_min + 
                (sched->base_lr - sched->params.cosine.eta_min) * 
                0.5f * (1.0f + cosf(M_PI * progress));
            break;
        }
            
        case LR_WARMUP:
            if (sched->current_epoch < sched->params.warmup.warmup_epochs) {
                float alpha = (float)sched->current_epoch / sched->params.warmup.warmup_epochs;
                sched->optimizer->learning_rate = sched->params.warmup.warmup_start_lr + 
                    alpha * (sched->base_lr - sched->params.warmup.warmup_start_lr);
            }
            break;
    }
}

void lr_scheduler_free(LRScheduler* sched) {
    free(sched);
}

// ============================================================
// Metrics Tracker
// ============================================================

#define MAX_METRICS 10
#define MAX_EPOCHS 10000

struct MetricsTracker {
    char names[MAX_METRICS][64];
    float values[MAX_METRICS][MAX_EPOCHS];
    uint32_t num_metrics;
    uint32_t num_epochs;
};

MetricsTracker* metrics_tracker_create() {
    MetricsTracker* mt = (MetricsTracker*)calloc(1, sizeof(MetricsTracker));
    return mt;
}

void metrics_tracker_add(MetricsTracker* mt, const char* name, float value) {
    // Find or create metric
    int idx = -1;
    for (uint32_t i = 0; i < mt->num_metrics; i++) {
        if (strcmp(mt->names[i], name) == 0) {
            idx = i;
            break;
        }
    }
    
    if (idx == -1) {
        idx = mt->num_metrics++;
        strncpy(mt->names[idx], name, 63);
    }
    
    mt->values[idx][mt->num_epochs] = value;
}

void metrics_tracker_next_epoch(MetricsTracker* mt) {
    mt->num_epochs++;
}

void metrics_tracker_print(const MetricsTracker* mt) {
    printf("\n=== Training Metrics ===\n");
    for (uint32_t i = 0; i < mt->num_metrics; i++) {
        printf("%s: ", mt->names[i]);
        
        float min_val = FLT_MAX, max_val = -FLT_MAX, last_val = 0.0f;
        for (uint32_t e = 0; e < mt->num_epochs; e++) {
            float v = mt->values[i][e];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
            last_val = v;
        }
        
        printf("min=%.6f, max=%.6f, last=%.6f\n", min_val, max_val, last_val);
    }
    printf("========================\n");
}

void metrics_tracker_save(const MetricsTracker* mt, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) return;
    
    fprintf(f, "epoch");
    for (uint32_t i = 0; i < mt->num_metrics; i++) {
        fprintf(f, ",%s", mt->names[i]);
    }
    fprintf(f, "\n");
    
    for (uint32_t e = 0; e < mt->num_epochs; e++) {
        fprintf(f, "%u", e);
        for (uint32_t i = 0; i < mt->num_metrics; i++) {
            fprintf(f, ",%.6f", mt->values[i][e]);
        }
        fprintf(f, "\n");
    }
    
    fclose(f);
}

void metrics_tracker_free(MetricsTracker* mt) {
    free(mt);
}

// ============================================================
// Model Checkpointing
// ============================================================

struct ModelCheckpoint {
    char filepath_pattern[256];
    char monitor_metric[64];
    bool save_best_only;
    bool minimize;
    float best_score;
};

ModelCheckpoint* checkpoint_create(const char* filepath, const char* monitor, 
                                  bool save_best_only, bool minimize) {
    ModelCheckpoint* ckpt = (ModelCheckpoint*)calloc(1, sizeof(ModelCheckpoint));
    strncpy(ckpt->filepath_pattern, filepath, 255);
    strncpy(ckpt->monitor_metric, monitor, 63);
    ckpt->save_best_only = save_best_only;
    ckpt->minimize = minimize;
    ckpt->best_score = minimize ? FLT_MAX : -FLT_MAX;
    return ckpt;
}

bool checkpoint_should_save(ModelCheckpoint* ckpt, float metric) {
    if (!ckpt->save_best_only) return true;
    
    float score = ckpt->minimize ? metric : -metric;
    if (score < ckpt->best_score) {
        ckpt->best_score = score;
        return true;
    }
    
    return false;
}

void checkpoint_save_model(ModelCheckpoint* ckpt, Tensor** parameters, 
                          uint32_t num_params, uint32_t epoch, float metric) {
    if (!checkpoint_should_save(ckpt, metric)) return;
    
    char filename[512];
    snprintf(filename, 511, ckpt->filepath_pattern, epoch, metric);
    
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    
    // Write header
    fwrite(&num_params, sizeof(uint32_t), 1, f);
    fwrite(&epoch, sizeof(uint32_t), 1, f);
    fwrite(&metric, sizeof(float), 1, f);
    
    // Write each parameter
    for (uint32_t i = 0; i < num_params; i++) {
        Tensor* p = parameters[i];
        fwrite(&p->dims, sizeof(uint32_t), 1, f);
        fwrite(p->shape, sizeof(uint32_t), p->dims, f);
        fwrite(&p->size, sizeof(uint32_t), 1, f);
        fwrite(p->data, sizeof(float), p->size, f);
    }
    
    fclose(f);
    printf("Checkpoint saved: %s\n", filename);
}

void checkpoint_free(ModelCheckpoint* ckpt) {
    free(ckpt);
}
