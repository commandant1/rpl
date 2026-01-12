/*
 * Classical ML Algorithms (scikit-learn equivalent)
 * K-Means, SVM, Random Forest, PCA, Logistic Regression, etc.
 */

#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ============================================================
// Logistic Regression
// ============================================================

struct LogisticRegression {
    uint32_t n_features;
    uint32_t n_classes;
    float* coef;      // [n_classes, n_features] or [n_features] for binary
    float* intercept; // [n_classes] or [1] for binary
    float C;          // Inverse regularization strength
    uint32_t max_iter;
    float tol;
    bool is_binary;
};

LogisticRegression* logistic_regression_create(float C, uint32_t max_iter, float tol) {
    LogisticRegression* lr = (LogisticRegression*)calloc(1, sizeof(LogisticRegression));
    lr->C = C;
    lr->max_iter = max_iter;
    lr->tol = tol;
    return lr;
}

void logistic_regression_fit(LogisticRegression* lr, const float* X, const uint32_t* y,
                             uint32_t n_samples, uint32_t n_features, uint32_t n_classes) {
    lr->n_features = n_features;
    lr->n_classes = n_classes;
    lr->is_binary = (n_classes == 2);
    
    uint32_t n_coef = lr->is_binary ? n_features : n_classes * n_features;
    uint32_t n_intercept = lr->is_binary ? 1 : n_classes;
    
    lr->coef = (float*)calloc(n_coef, sizeof(float));
    lr->intercept = (float*)calloc(n_intercept, sizeof(float));
    
    // SGD with L2 regularization
    float learning_rate = 0.01f;
    float reg_strength = 1.0f / (lr->C * n_samples);
    
    for (uint32_t iter = 0; iter < lr->max_iter; iter++) {
        float total_loss = 0.0f;
        
        // Shuffle indices
        uint32_t* indices = (uint32_t*)malloc(n_samples * sizeof(uint32_t));
        for (uint32_t i = 0; i < n_samples; i++) indices[i] = i;
        for (uint32_t i = n_samples - 1; i > 0; i--) {
            uint32_t j = rand() % (i + 1);
            uint32_t temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        // Mini-batch SGD
        for (uint32_t i = 0; i < n_samples; i++) {
            uint32_t idx = indices[i];
            const float* x = &X[idx * n_features];
            uint32_t label = y[idx];
            
            if (lr->is_binary) {
                // Binary logistic regression
                float z = lr->intercept[0];
                for (uint32_t f = 0; f < n_features; f++) {
                    z += lr->coef[f] * x[f];
                }
                
                float p = 1.0f / (1.0f + expf(-z));
                float error = p - (float)label;
                
                // Update weights
                lr->intercept[0] -= learning_rate * error;
                for (uint32_t f = 0; f < n_features; f++) {
                    lr->coef[f] -= learning_rate * (error * x[f] + reg_strength * lr->coef[f]);
                }
                
                total_loss += -((float)label * logf(p + 1e-10f) + (1.0f - label) * logf(1.0f - p + 1e-10f));
            } else {
                // Multiclass (softmax)
                float scores[n_classes];
                float max_score = -FLT_MAX;
                
                // Compute scores
                for (uint32_t c = 0; c < n_classes; c++) {
                    scores[c] = lr->intercept[c];
                    for (uint32_t f = 0; f < n_features; f++) {
                        scores[c] += lr->coef[c * n_features + f] * x[f];
                    }
                    if (scores[c] > max_score) max_score = scores[c];
                }
                
                // Softmax
                float sum_exp = 0.0f;
                for (uint32_t c = 0; c < n_classes; c++) {
                    scores[c] = expf(scores[c] - max_score);
                    sum_exp += scores[c];
                }
                for (uint32_t c = 0; c < n_classes; c++) {
                    scores[c] /= sum_exp;
                }
                
                // Update weights
                for (uint32_t c = 0; c < n_classes; c++) {
                    float error = scores[c] - ((c == label) ? 1.0f : 0.0f);
                    
                    lr->intercept[c] -= learning_rate * error;
                    for (uint32_t f = 0; f < n_features; f++) {
                        lr->coef[c * n_features + f] -= learning_rate * 
                            (error * x[f] + reg_strength * lr->coef[c * n_features + f]);
                    }
                }
                
                total_loss += -logf(scores[label] + 1e-10f);
            }
        }
        
        free(indices);
        
        // Check convergence
        if (iter > 0 && total_loss / n_samples < lr->tol) {
            break;
        }
        
        // Decay learning rate
        learning_rate *= 0.99f;
    }
}

void logistic_regression_predict(const LogisticRegression* lr, const float* X,
                                 uint32_t n_samples, uint32_t* y_pred) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < n_samples; i++) {
        const float* x = &X[i * lr->n_features];
        
        if (lr->is_binary) {
            float z = lr->intercept[0];
            for (uint32_t f = 0; f < lr->n_features; f++) {
                z += lr->coef[f] * x[f];
            }
            float p = 1.0f / (1.0f + expf(-z));
            y_pred[i] = (p >= 0.5f) ? 1 : 0;
        } else {
            float max_score = -FLT_MAX;
            uint32_t best_class = 0;
            
            for (uint32_t c = 0; c < lr->n_classes; c++) {
                float score = lr->intercept[c];
                for (uint32_t f = 0; f < lr->n_features; f++) {
                    score += lr->coef[c * lr->n_features + f] * x[f];
                }
                
                if (score > max_score) {
                    max_score = score;
                    best_class = c;
                }
            }
            
            y_pred[i] = best_class;
        }
    }
}

float* logistic_regression_predict_proba(const LogisticRegression* lr, const float* X,
                                        uint32_t n_samples) {
    float* proba = (float*)malloc(n_samples * lr->n_classes * sizeof(float));
    
    #pragma omp parallel for
    for (uint32_t i = 0; i < n_samples; i++) {
        const float* x = &X[i * lr->n_features];
        
        if (lr->is_binary) {
            float z = lr->intercept[0];
            for (uint32_t f = 0; f < lr->n_features; f++) {
                z += lr->coef[f] * x[f];
            }
            float p = 1.0f / (1.0f + expf(-z));
            proba[i * 2] = 1.0f - p;
            proba[i * 2 + 1] = p;
        } else {
            float scores[lr->n_classes];
            float max_score = -FLT_MAX;
            
            for (uint32_t c = 0; c < lr->n_classes; c++) {
                scores[c] = lr->intercept[c];
                for (uint32_t f = 0; f < lr->n_features; f++) {
                    scores[c] += lr->coef[c * lr->n_features + f] * x[f];
                }
                if (scores[c] > max_score) max_score = scores[c];
            }
            
            // Softmax
            float sum_exp = 0.0f;
            for (uint32_t c = 0; c < lr->n_classes; c++) {
                scores[c] = expf(scores[c] - max_score);
                sum_exp += scores[c];
            }
            
            for (uint32_t c = 0; c < lr->n_classes; c++) {
                proba[i * lr->n_classes + c] = scores[c] / sum_exp;
            }
        }
    }
    
    return proba;
}

void logistic_regression_free(LogisticRegression* lr) {
    free(lr->coef);
    free(lr->intercept);
    free(lr);
}

// ============================================================
// K-Means Clustering
// ============================================================

struct KMeans {
    uint32_t n_clusters;
    uint32_t n_features;
    uint32_t max_iter;
    float tol;
    
    float* centroids;  // [n_clusters, n_features]
    uint32_t* labels;
    float inertia;
    uint32_t n_iter;
};

KMeans* kmeans_create(uint32_t n_clusters, uint32_t max_iter, float tol) {
    KMeans* km = (KMeans*)calloc(1, sizeof(KMeans));
    km->n_clusters = n_clusters;
    km->max_iter = max_iter;
    km->tol = tol;
    return km;
}

void kmeans_fit(KMeans* km, const float* X, uint32_t n_samples, uint32_t n_features) {
    km->n_features = n_features;
    km->centroids = (float*)malloc(km->n_clusters * n_features * sizeof(float));
    km->labels = (uint32_t*)malloc(n_samples * sizeof(uint32_t));
    
    // Initialize centroids (k-means++)
    km->centroids[0] = X[rand() % n_samples];
    
    for (uint32_t k = 1; k < km->n_clusters; k++) {
        float* distances = (float*)malloc(n_samples * sizeof(float));
        
        #pragma omp parallel for
        for (uint32_t i = 0; i < n_samples; i++) {
            float min_dist = FLT_MAX;
            for (uint32_t j = 0; j < k; j++) {
                float dist = 0.0f;
                for (uint32_t f = 0; f < n_features; f++) {
                    float diff = X[i * n_features + f] - km->centroids[j * n_features + f];
                    dist += diff * diff;
                }
                if (dist < min_dist) min_dist = dist;
            }
            distances[i] = min_dist;
        }
        
        // Select next centroid proportional to distance
        float sum_dist = 0.0f;
        for (uint32_t i = 0; i < n_samples; i++) sum_dist += distances[i];
        
        float r = ((float)rand() / RAND_MAX) * sum_dist;
        float cumsum = 0.0f;
        for (uint32_t i = 0; i < n_samples; i++) {
            cumsum += distances[i];
            if (cumsum >= r) {
                memcpy(&km->centroids[k * n_features], &X[i * n_features], 
                      n_features * sizeof(float));
                break;
            }
        }
        
        free(distances);
    }
    
    // Lloyd's algorithm
    for (km->n_iter = 0; km->n_iter < km->max_iter; km->n_iter++) {
        // Assignment step
        #pragma omp parallel for
        for (uint32_t i = 0; i < n_samples; i++) {
            float min_dist = FLT_MAX;
            uint32_t best_cluster = 0;
            
            for (uint32_t k = 0; k < km->n_clusters; k++) {
                float dist = 0.0f;
                for (uint32_t f = 0; f < n_features; f++) {
                    float diff = X[i * n_features + f] - km->centroids[k * n_features + f];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = k;
                }
            }
            
            km->labels[i] = best_cluster;
        }
        
        // Update step
        float* new_centroids = (float*)calloc(km->n_clusters * n_features, sizeof(float));
        uint32_t* counts = (uint32_t*)calloc(km->n_clusters, sizeof(uint32_t));
        
        for (uint32_t i = 0; i < n_samples; i++) {
            uint32_t cluster = km->labels[i];
            counts[cluster]++;
            for (uint32_t f = 0; f < n_features; f++) {
                new_centroids[cluster * n_features + f] += X[i * n_features + f];
            }
        }
        
        for (uint32_t k = 0; k < km->n_clusters; k++) {
            if (counts[k] > 0) {
                for (uint32_t f = 0; f < n_features; f++) {
                    new_centroids[k * n_features + f] /= counts[k];
                }
            }
        }
        
        // Check convergence
        float shift = 0.0f;
        for (uint32_t i = 0; i < km->n_clusters * n_features; i++) {
            float diff = new_centroids[i] - km->centroids[i];
            shift += diff * diff;
        }
        
        memcpy(km->centroids, new_centroids, km->n_clusters * n_features * sizeof(float));
        free(new_centroids);
        free(counts);
        
        if (sqrtf(shift) < km->tol) break;
    }
    
    // Compute inertia
    km->inertia = 0.0f;
    for (uint32_t i = 0; i < n_samples; i++) {
        uint32_t cluster = km->labels[i];
        for (uint32_t f = 0; f < n_features; f++) {
            float diff = X[i * n_features + f] - km->centroids[cluster * n_features + f];
            km->inertia += diff * diff;
        }
    }
}

void kmeans_predict(const KMeans* km, const float* X, uint32_t n_samples, uint32_t* labels) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < n_samples; i++) {
        float min_dist = FLT_MAX;
        uint32_t best_cluster = 0;
        
        for (uint32_t k = 0; k < km->n_clusters; k++) {
            float dist = 0.0f;
            for (uint32_t f = 0; f < km->n_features; f++) {
                float diff = X[i * km->n_features + f] - km->centroids[k * km->n_features + f];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = k;
            }
        }
        
        labels[i] = best_cluster;
    }
}

void kmeans_free(KMeans* km) {
    free(km->centroids);
    free(km->labels);
    free(km);
}

// ============================================================
// Principal Component Analysis (PCA)
// ============================================================

struct PCA {
    uint32_t n_components;
    uint32_t n_features;
    
    float* components;      // [n_components, n_features]
    float* explained_variance;
    float* mean;
};

PCA* pca_create(uint32_t n_components) {
    PCA* pca = (PCA*)calloc(1, sizeof(PCA));
    pca->n_components = n_components;
    return pca;
}

void pca_fit(PCA* pca, const float* X, uint32_t n_samples, uint32_t n_features) {
    pca->n_features = n_features;
    pca->mean = (float*)calloc(n_features, sizeof(float));
    pca->components = (float*)malloc(pca->n_components * n_features * sizeof(float));
    pca->explained_variance = (float*)malloc(pca->n_components * sizeof(float));
    
    // Compute mean
    for (uint32_t i = 0; i < n_samples; i++) {
        for (uint32_t f = 0; f < n_features; f++) {
            pca->mean[f] += X[i * n_features + f];
        }
    }
    for (uint32_t f = 0; f < n_features; f++) {
        pca->mean[f] /= n_samples;
    }
    
    // Center data
    float* X_centered = (float*)malloc(n_samples * n_features * sizeof(float));
    for (uint32_t i = 0; i < n_samples; i++) {
        for (uint32_t f = 0; f < n_features; f++) {
            X_centered[i * n_features + f] = X[i * n_features + f] - pca->mean[f];
        }
    }
    
    // Compute covariance matrix
    float* cov = (float*)calloc(n_features * n_features, sizeof(float));
    
    #pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < n_features; i++) {
        for (uint32_t j = 0; j < n_features; j++) {
            float sum = 0.0f;
            for (uint32_t s = 0; s < n_samples; s++) {
                sum += X_centered[s * n_features + i] * X_centered[s * n_features + j];
            }
            cov[i * n_features + j] = sum / (n_samples - 1);
        }
    }
    
    // Power iteration for top eigenvectors
    for (uint32_t k = 0; k < pca->n_components; k++) {
        float* v = (float*)malloc(n_features * sizeof(float));
        
        // Random initialization
        for (uint32_t i = 0; i < n_features; i++) {
            v[i] = (float)rand() / RAND_MAX;
        }
        
        // Power iteration
        for (uint32_t iter = 0; iter < 100; iter++) {
            float* v_new = (float*)calloc(n_features, sizeof(float));
            
            // v_new = cov @ v
            for (uint32_t i = 0; i < n_features; i++) {
                for (uint32_t j = 0; j < n_features; j++) {
                    v_new[i] += cov[i * n_features + j] * v[j];
                }
            }
            
            // Normalize
            float norm = 0.0f;
            for (uint32_t i = 0; i < n_features; i++) {
                norm += v_new[i] * v_new[i];
            }
            norm = sqrtf(norm);
            
            for (uint32_t i = 0; i < n_features; i++) {
                v_new[i] /= norm;
            }
            
            memcpy(v, v_new, n_features * sizeof(float));
            free(v_new);
        }
        
        // Store component
        memcpy(&pca->components[k * n_features], v, n_features * sizeof(float));
        
        // Compute eigenvalue
        float eigenvalue = 0.0f;
        for (uint32_t i = 0; i < n_features; i++) {
            for (uint32_t j = 0; j < n_features; j++) {
                eigenvalue += v[i] * cov[i * n_features + j] * v[j];
            }
        }
        pca->explained_variance[k] = eigenvalue;
        
        // Deflate covariance matrix
        for (uint32_t i = 0; i < n_features; i++) {
            for (uint32_t j = 0; j < n_features; j++) {
                cov[i * n_features + j] -= eigenvalue * v[i] * v[j];
            }
        }
        
        free(v);
    }
    
    free(X_centered);
    free(cov);
}

void pca_transform(const PCA* pca, const float* X, uint32_t n_samples, float* X_transformed) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < n_samples; i++) {
        for (uint32_t k = 0; k < pca->n_components; k++) {
            float sum = 0.0f;
            for (uint32_t f = 0; f < pca->n_features; f++) {
                sum += (X[i * pca->n_features + f] - pca->mean[f]) * 
                      pca->components[k * pca->n_features + f];
            }
            X_transformed[i * pca->n_components + k] = sum;
        }
    }
}

void pca_free(PCA* pca) {
    free(pca->components);
    free(pca->explained_variance);
    free(pca->mean);
    free(pca);
}

// ============================================================
// Support Vector Machine (SVM) - Simplified SMO
// ============================================================

struct SVM {
    uint32_t n_features;
    uint32_t n_support;
    float* support_vectors;  // [n_support, n_features]
    float* alphas;          // [n_support]
    float b;                // Bias
    float C;                // Regularization
    float gamma;            // RBF kernel parameter
};

SVM* svm_create(float C, float gamma) {
    SVM* svm = (SVM*)calloc(1, sizeof(SVM));
    svm->C = C;
    svm->gamma = gamma;
    return svm;
}

float svm_rbf_kernel(const float* x1, const float* x2, uint32_t n_features, float gamma) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_features; i++) {
        float diff = x1[i] - x2[i];
        sum += diff * diff;
    }
    return expf(-gamma * sum);
}

void svm_fit(SVM* svm, const float* X, const float* y, uint32_t n_samples, uint32_t n_features) {
    svm->n_features = n_features;
    
    // Simplified: use all samples as support vectors
    svm->n_support = n_samples;
    svm->support_vectors = (float*)malloc(n_samples * n_features * sizeof(float));
    svm->alphas = (float*)calloc(n_samples, sizeof(float));
    memcpy(svm->support_vectors, X, n_samples * n_features * sizeof(float));
    
    // Simplified SMO algorithm
    float tol = 1e-3f;
    uint32_t max_passes = 100;
    uint32_t passes = 0;
    
    while (passes < max_passes) {
        uint32_t num_changed = 0;
        
        for (uint32_t i = 0; i < n_samples; i++) {
            // Compute E_i
            float E_i = -y[i];
            for (uint32_t j = 0; j < n_samples; j++) {
                E_i += svm->alphas[j] * y[j] * 
                      svm_rbf_kernel(&svm->support_vectors[i * n_features],
                                    &svm->support_vectors[j * n_features],
                                    n_features, svm->gamma);
            }
            E_i += svm->b;
            
            if ((y[i] * E_i < -tol && svm->alphas[i] < svm->C) ||
                (y[i] * E_i > tol && svm->alphas[i] > 0)) {
                
                // Select j randomly
                uint32_t j = rand() % n_samples;
                if (j == i) j = (j + 1) % n_samples;
                
                // Simple update
                float old_alpha_i = svm->alphas[i];
                float old_alpha_j = svm->alphas[j];
                
                svm->alphas[i] += 0.01f * y[i] * E_i;
                svm->alphas[i] = fmaxf(0.0f, fminf(svm->C, svm->alphas[i]));
                
                num_changed++;
            }
        }
        
        if (num_changed == 0) {
            passes++;
        } else {
            passes = 0;
        }
    }
    
    // Compute bias
    svm->b = 0.0f;
    uint32_t count = 0;
    for (uint32_t i = 0; i < n_samples; i++) {
        if (svm->alphas[i] > 0 && svm->alphas[i] < svm->C) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < n_samples; j++) {
                sum += svm->alphas[j] * y[j] *
                      svm_rbf_kernel(&svm->support_vectors[i * n_features],
                                    &svm->support_vectors[j * n_features],
                                    n_features, svm->gamma);
            }
            svm->b += y[i] - sum;
            count++;
        }
    }
    if (count > 0) svm->b /= count;
}

void svm_predict(const SVM* svm, const float* X, uint32_t n_samples, float* y_pred) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < n_samples; i++) {
        float sum = svm->b;
        for (uint32_t j = 0; j < svm->n_support; j++) {
            sum += svm->alphas[j] *
                  svm_rbf_kernel(&X[i * svm->n_features],
                                &svm->support_vectors[j * svm->n_features],
                                svm->n_features, svm->gamma);
        }
        y_pred[i] = (sum >= 0) ? 1.0f : -1.0f;
    }
}

void svm_free(SVM* svm) {
    free(svm->support_vectors);
    free(svm->alphas);
    free(svm);
}

// ============================================================
// Naive Bayes (Gaussian)
// ============================================================

struct NaiveBayes {
    uint32_t n_features;
    uint32_t n_classes;
    float* class_priors;    // [n_classes]
    float* means;          // [n_classes, n_features]
    float* variances;      // [n_classes, n_features]
};

NaiveBayes* naive_bayes_create() {
    return (NaiveBayes*)calloc(1, sizeof(NaiveBayes));
}

void naive_bayes_fit(NaiveBayes* nb, const float* X, const uint32_t* y,
                    uint32_t n_samples, uint32_t n_features, uint32_t n_classes) {
    nb->n_features = n_features;
    nb->n_classes = n_classes;
    
    nb->class_priors = (float*)calloc(n_classes, sizeof(float));
    nb->means = (float*)calloc(n_classes * n_features, sizeof(float));
    nb->variances = (float*)calloc(n_classes * n_features, sizeof(float));
    
    uint32_t* class_counts = (uint32_t*)calloc(n_classes, sizeof(uint32_t));
    
    // Count samples per class and compute means
    for (uint32_t i = 0; i < n_samples; i++) {
        uint32_t c = y[i];
        class_counts[c]++;
        
        for (uint32_t f = 0; f < n_features; f++) {
            nb->means[c * n_features + f] += X[i * n_features + f];
        }
    }
    
    for (uint32_t c = 0; c < n_classes; c++) {
        nb->class_priors[c] = (float)class_counts[c] / n_samples;
        
        for (uint32_t f = 0; f < n_features; f++) {
            nb->means[c * n_features + f] /= class_counts[c];
        }
    }
    
    // Compute variances
    for (uint32_t i = 0; i < n_samples; i++) {
        uint32_t c = y[i];
        
        for (uint32_t f = 0; f < n_features; f++) {
            float diff = X[i * n_features + f] - nb->means[c * n_features + f];
            nb->variances[c * n_features + f] += diff * diff;
        }
    }
    
    for (uint32_t c = 0; c < n_classes; c++) {
        for (uint32_t f = 0; f < n_features; f++) {
            nb->variances[c * n_features + f] /= class_counts[c];
            nb->variances[c * n_features + f] += 1e-9f;  // Smoothing
        }
    }
    
    free(class_counts);
}

void naive_bayes_predict(const NaiveBayes* nb, const float* X, uint32_t n_samples, uint32_t* y_pred) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < n_samples; i++) {
        float max_prob = -FLT_MAX;
        uint32_t best_class = 0;
        
        for (uint32_t c = 0; c < nb->n_classes; c++) {
            float log_prob = logf(nb->class_priors[c]);
            
            for (uint32_t f = 0; f < nb->n_features; f++) {
                float x = X[i * nb->n_features + f];
                float mean = nb->means[c * nb->n_features + f];
                float var = nb->variances[c * nb->n_features + f];
                
                float diff = x - mean;
                log_prob += -0.5f * (logf(2.0f * M_PI * var) + diff * diff / var);
            }
            
            if (log_prob > max_prob) {
                max_prob = log_prob;
                best_class = c;
            }
        }
        
        y_pred[i] = best_class;
    }
}

void naive_bayes_free(NaiveBayes* nb) {
    free(nb->class_priors);
    free(nb->means);
    free(nb->variances);
    free(nb);
}

// ============================================================
// DBSCAN Clustering
// ============================================================

struct DBSCAN {
    float eps;
    uint32_t min_samples;
    uint32_t* labels;
    uint32_t n_clusters;
};

DBSCAN* dbscan_create(float eps, uint32_t min_samples) {
    DBSCAN* db = (DBSCAN*)calloc(1, sizeof(DBSCAN));
    db->eps = eps;
    db->min_samples = min_samples;
    return db;
}

void dbscan_fit(DBSCAN* db, const float* X, uint32_t n_samples, uint32_t n_features) {
    db->labels = (uint32_t*)malloc(n_samples * sizeof(uint32_t));
    for (uint32_t i = 0; i < n_samples; i++) {
        db->labels[i] = UINT32_MAX;  // Unvisited
    }
    
    uint32_t cluster_id = 0;
    
    for (uint32_t i = 0; i < n_samples; i++) {
        if (db->labels[i] != UINT32_MAX) continue;
        
        // Find neighbors
        uint32_t* neighbors = (uint32_t*)malloc(n_samples * sizeof(uint32_t));
        uint32_t n_neighbors = 0;
        
        for (uint32_t j = 0; j < n_samples; j++) {
            float dist = 0.0f;
            for (uint32_t f = 0; f < n_features; f++) {
                float diff = X[i * n_features + f] - X[j * n_features + f];
                dist += diff * diff;
            }
            dist = sqrtf(dist);
            
            if (dist <= db->eps) {
                neighbors[n_neighbors++] = j;
            }
        }
        
        if (n_neighbors < db->min_samples) {
            db->labels[i] = UINT32_MAX - 1;  // Noise
            free(neighbors);
            continue;
        }
        
        // Start new cluster
        db->labels[i] = cluster_id;
        
        // Expand cluster
        for (uint32_t k = 0; k < n_neighbors; k++) {
            uint32_t neighbor = neighbors[k];
            
            if (db->labels[neighbor] == UINT32_MAX - 1) {
                db->labels[neighbor] = cluster_id;
            }
            
            if (db->labels[neighbor] != UINT32_MAX) continue;
            
            db->labels[neighbor] = cluster_id;
        }
        
        free(neighbors);
        cluster_id++;
    }
    
    db->n_clusters = cluster_id;
}

void dbscan_free(DBSCAN* db) {
    free(db->labels);
    free(db);
}

// ============================================================
// Linear Regression
// ============================================================

struct LinearRegression {
    uint32_t n_features;
    float* coef;
    float intercept;
};

LinearRegression* linear_regression_create() {
    return (LinearRegression*)calloc(1, sizeof(LinearRegression));
}

void linear_regression_fit(LinearRegression* lr, const float* X, const float* y,
                           uint32_t n_samples, uint32_t n_features) {
    lr->n_features = n_features;
    lr->coef = (float*)malloc(n_features * sizeof(float));
    
    // Normal equation: coef = (X^T X)^{-1} X^T y
    // For simplicity, using gradient descent
    
    memset(lr->coef, 0, n_features * sizeof(float));
    lr->intercept = 0.0f;
    
    float learning_rate = 0.01f;
    uint32_t max_iter = 1000;
    
    for (uint32_t iter = 0; iter < max_iter; iter++) {
        float* grad_coef = (float*)calloc(n_features, sizeof(float));
        float grad_intercept = 0.0f;
        
        // Compute gradients
        for (uint32_t i = 0; i < n_samples; i++) {
            float pred = lr->intercept;
            for (uint32_t f = 0; f < n_features; f++) {
                pred += lr->coef[f] * X[i * n_features + f];
            }
            
            float error = pred - y[i];
            grad_intercept += error;
            
            for (uint32_t f = 0; f < n_features; f++) {
                grad_coef[f] += error * X[i * n_features + f];
            }
        }
        
        // Update parameters
        lr->intercept -= learning_rate * grad_intercept / n_samples;
        for (uint32_t f = 0; f < n_features; f++) {
            lr->coef[f] -= learning_rate * grad_coef[f] / n_samples;
        }
        
        free(grad_coef);
    }
}

void linear_regression_predict(const LinearRegression* lr, const float* X,
                               uint32_t n_samples, float* y_pred) {
    #pragma omp parallel for
    for (uint32_t i = 0; i < n_samples; i++) {
        y_pred[i] = lr->intercept;
        for (uint32_t f = 0; f < lr->n_features; f++) {
            y_pred[i] += lr->coef[f] * X[i * lr->n_features + f];
        }
    }
}

void linear_regression_free(LinearRegression* lr) {
    free(lr->coef);
    free(lr);
}
