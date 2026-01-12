/*
 * RPiTorch Metrics and Evaluation
 * Confusion Matrix, ROC-AUC, R², Cross-Validation, Grid Search
 */

#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

// ============================================================
// Confusion Matrix
// ============================================================

void confusion_matrix(const uint32_t* y_true, const uint32_t* y_pred,
                     uint32_t n_samples, uint32_t n_classes,
                     uint32_t* matrix) {
    // Initialize matrix to zero
    memset(matrix, 0, n_classes * n_classes * sizeof(uint32_t));
    
    // Fill confusion matrix
    for (uint32_t i = 0; i < n_samples; i++) {
        uint32_t true_class = y_true[i];
        uint32_t pred_class = y_pred[i];
        
        if (true_class < n_classes && pred_class < n_classes) {
            matrix[true_class * n_classes + pred_class]++;
        }
    }
}

void print_confusion_matrix(const uint32_t* matrix, uint32_t n_classes,
                           const char** class_names) {
    printf("\nConfusion Matrix:\n");
    printf("%-15s", "True\\Pred");
    
    for (uint32_t i = 0; i < n_classes; i++) {
        if (class_names) {
            printf("%-10s", class_names[i]);
        } else {
            printf("Class%-5u", i);
        }
    }
    printf("\n");
    
    for (uint32_t i = 0; i < n_classes; i++) {
        if (class_names) {
            printf("%-15s", class_names[i]);
        } else {
            printf("Class %-9u", i);
        }
        
        for (uint32_t j = 0; j < n_classes; j++) {
            printf("%-10u", matrix[i * n_classes + j]);
        }
        printf("\n");
    }
    
    // Compute per-class metrics
    printf("\nPer-Class Metrics:\n");
    printf("%-15s %-12s %-12s %-12s\n", "Class", "Precision", "Recall", "F1-Score");
    
    for (uint32_t i = 0; i < n_classes; i++) {
        uint32_t tp = matrix[i * n_classes + i];
        uint32_t fp = 0, fn = 0;
        
        for (uint32_t j = 0; j < n_classes; j++) {
            if (j != i) {
                fp += matrix[j * n_classes + i];  // Predicted as i but actually j
                fn += matrix[i * n_classes + j];  // Actually i but predicted as j
            }
        }
        
        float precision = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.0f;
        float recall = (tp + fn > 0) ? (float)tp / (tp + fn) : 0.0f;
        float f1 = (precision + recall > 0) ? 2.0f * precision * recall / (precision + recall) : 0.0f;
        
        if (class_names) {
            printf("%-15s", class_names[i]);
        } else {
            printf("Class %-9u", i);
        }
        printf("%.4f       %.4f       %.4f\n", precision, recall, f1);
    }
}

// ============================================================
// ROC-AUC Score
// ============================================================

float roc_auc_score(const float* y_true, const float* y_score, uint32_t n_samples) {
    // For binary classification
    // Sort by score
    typedef struct {
        float score;
        float label;
    } ScoreLabel;
    
    ScoreLabel* pairs = (ScoreLabel*)malloc(n_samples * sizeof(ScoreLabel));
    for (uint32_t i = 0; i < n_samples; i++) {
        pairs[i].score = y_score[i];
        pairs[i].label = y_true[i];
    }
    
    // Sort by score (descending)
    for (uint32_t i = 0; i < n_samples - 1; i++) {
        for (uint32_t j = i + 1; j < n_samples; j++) {
            if (pairs[j].score > pairs[i].score) {
                ScoreLabel temp = pairs[i];
                pairs[i] = pairs[j];
                pairs[j] = temp;
            }
        }
    }
    
    // Compute AUC using trapezoidal rule
    uint32_t n_pos = 0, n_neg = 0;
    for (uint32_t i = 0; i < n_samples; i++) {
        if (pairs[i].label > 0.5f) n_pos++;
        else n_neg++;
    }
    
    if (n_pos == 0 || n_neg == 0) {
        free(pairs);
        return 0.5f;
    }
    
    float auc = 0.0f;
    uint32_t tp = 0, fp = 0;
    float prev_tp_rate = 0.0f, prev_fp_rate = 0.0f;
    
    for (uint32_t i = 0; i < n_samples; i++) {
        if (pairs[i].label > 0.5f) {
            tp++;
        } else {
            fp++;
        }
        
        float tp_rate = (float)tp / n_pos;
        float fp_rate = (float)fp / n_neg;
        
        // Trapezoidal area
        auc += (fp_rate - prev_fp_rate) * (tp_rate + prev_tp_rate) / 2.0f;
        
        prev_tp_rate = tp_rate;
        prev_fp_rate = fp_rate;
    }
    
    free(pairs);
    return auc;
}

// ============================================================
// R² Score (Coefficient of Determination)
// ============================================================

float r2_score(const float* y_true, const float* y_pred, uint32_t n_samples) {
    // Compute mean of y_true
    float mean = 0.0f;
    for (uint32_t i = 0; i < n_samples; i++) {
        mean += y_true[i];
    }
    mean /= n_samples;
    
    // Compute SS_tot and SS_res
    float ss_tot = 0.0f, ss_res = 0.0f;
    
    for (uint32_t i = 0; i < n_samples; i++) {
        float diff_tot = y_true[i] - mean;
        float diff_res = y_true[i] - y_pred[i];
        
        ss_tot += diff_tot * diff_tot;
        ss_res += diff_res * diff_res;
    }
    
    if (ss_tot == 0.0f) return 1.0f;
    
    return 1.0f - (ss_res / ss_tot);
}

// ============================================================
// Cross-Validation Split
// ============================================================

struct CrossValidationSplit {
    uint32_t n_splits;
    uint32_t n_samples;
    uint32_t** train_indices;
    uint32_t** test_indices;
    uint32_t* train_sizes;
    uint32_t* test_sizes;
};

CrossValidationSplit* cross_validation_split(uint32_t n_samples, uint32_t n_splits, bool shuffle) {
    CrossValidationSplit* cv = (CrossValidationSplit*)calloc(1, sizeof(CrossValidationSplit));
    cv->n_splits = n_splits;
    cv->n_samples = n_samples;
    
    cv->train_indices = (uint32_t**)malloc(n_splits * sizeof(uint32_t*));
    cv->test_indices = (uint32_t**)malloc(n_splits * sizeof(uint32_t*));
    cv->train_sizes = (uint32_t*)malloc(n_splits * sizeof(uint32_t));
    cv->test_sizes = (uint32_t*)malloc(n_splits * sizeof(uint32_t));
    
    // Create shuffled indices
    uint32_t* indices = (uint32_t*)malloc(n_samples * sizeof(uint32_t));
    for (uint32_t i = 0; i < n_samples; i++) {
        indices[i] = i;
    }
    
    if (shuffle) {
        for (uint32_t i = n_samples - 1; i > 0; i--) {
            uint32_t j = rand() % (i + 1);
            uint32_t temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
    }
    
    // Create splits
    uint32_t fold_size = n_samples / n_splits;
    
    for (uint32_t fold = 0; fold < n_splits; fold++) {
        uint32_t test_start = fold * fold_size;
        uint32_t test_end = (fold == n_splits - 1) ? n_samples : (fold + 1) * fold_size;
        
        cv->test_sizes[fold] = test_end - test_start;
        cv->train_sizes[fold] = n_samples - cv->test_sizes[fold];
        
        cv->test_indices[fold] = (uint32_t*)malloc(cv->test_sizes[fold] * sizeof(uint32_t));
        cv->train_indices[fold] = (uint32_t*)malloc(cv->train_sizes[fold] * sizeof(uint32_t));
        
        // Fill test indices
        for (uint32_t i = 0; i < cv->test_sizes[fold]; i++) {
            cv->test_indices[fold][i] = indices[test_start + i];
        }
        
        // Fill train indices
        uint32_t train_idx = 0;
        for (uint32_t i = 0; i < n_samples; i++) {
            if (i < test_start || i >= test_end) {
                cv->train_indices[fold][train_idx++] = indices[i];
            }
        }
    }
    
    free(indices);
    return cv;
}

void cross_validation_free(CrossValidationSplit* cv) {
    for (uint32_t i = 0; i < cv->n_splits; i++) {
        free(cv->train_indices[i]);
        free(cv->test_indices[i]);
    }
    free(cv->train_indices);
    free(cv->test_indices);
    free(cv->train_sizes);
    free(cv->test_sizes);
    free(cv);
}

// ============================================================
// Grid Search for Hyperparameter Tuning
// ============================================================

struct HyperParameter {
    char name[64];
    float* values;
    uint32_t n_values;
};

struct GridSearch {
    HyperParameter* params;
    uint32_t n_params;
    float* best_params;
    float best_score;
};

GridSearch* grid_search_create(uint32_t n_params) {
    GridSearch* gs = (GridSearch*)calloc(1, sizeof(GridSearch));
    gs->n_params = n_params;
    gs->params = (HyperParameter*)calloc(n_params, sizeof(HyperParameter));
    gs->best_params = (float*)calloc(n_params, sizeof(float));
    gs->best_score = -FLT_MAX;
    return gs;
}

void grid_search_add_param(GridSearch* gs, uint32_t param_idx, const char* name,
                          const float* values, uint32_t n_values) {
    HyperParameter* param = &gs->params[param_idx];
    strncpy(param->name, name, 63);
    param->n_values = n_values;
    param->values = (float*)malloc(n_values * sizeof(float));
    memcpy(param->values, values, n_values * sizeof(float));
}

typedef float (*GridSearchScoreFunc)(const float* params, uint32_t n_params, void* user_data);

void grid_search_fit(GridSearch* gs, GridSearchScoreFunc score_func, void* user_data) {
    // Compute total number of combinations
    uint32_t total_combinations = 1;
    for (uint32_t i = 0; i < gs->n_params; i++) {
        total_combinations *= gs->params[i].n_values;
    }
    
    printf("Grid Search: Testing %u combinations\n", total_combinations);
    
    // Current parameter combination
    float* current_params = (float*)malloc(gs->n_params * sizeof(float));
    uint32_t* indices = (uint32_t*)calloc(gs->n_params, sizeof(uint32_t));
    
    for (uint32_t combo = 0; combo < total_combinations; combo++) {
        // Set current parameters
        for (uint32_t p = 0; p < gs->n_params; p++) {
            current_params[p] = gs->params[p].values[indices[p]];
        }
        
        // Evaluate
        float score = score_func(current_params, gs->n_params, user_data);
        
        // Update best
        if (score > gs->best_score) {
            gs->best_score = score;
            memcpy(gs->best_params, current_params, gs->n_params * sizeof(float));
            
            printf("  New best score: %.4f with params: ", score);
            for (uint32_t p = 0; p < gs->n_params; p++) {
                printf("%s=%.4f ", gs->params[p].name, current_params[p]);
            }
            printf("\n");
        }
        
        // Increment indices (odometer-style)
        for (int32_t p = gs->n_params - 1; p >= 0; p--) {
            indices[p]++;
            if (indices[p] < gs->params[p].n_values) {
                break;
            }
            indices[p] = 0;
        }
    }
    
    free(current_params);
    free(indices);
    
    printf("\nGrid Search Complete!\n");
    printf("Best score: %.4f\n", gs->best_score);
    printf("Best parameters:\n");
    for (uint32_t p = 0; p < gs->n_params; p++) {
        printf("  %s = %.4f\n", gs->params[p].name, gs->best_params[p]);
    }
}

void grid_search_free(GridSearch* gs) {
    for (uint32_t i = 0; i < gs->n_params; i++) {
        free(gs->params[i].values);
    }
    free(gs->params);
    free(gs->best_params);
    free(gs);
}
