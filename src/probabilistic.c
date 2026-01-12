/*
 * Markov Chains and Hidden Markov Models
 * Probabilistic sequence models for RPiTorch
 */

#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// ============================================================
// Markov Chain
// ============================================================

typedef struct {
    int num_states;
    float* transition;      // P[i*num_states + j] = P(X_t+1=j | X_t=i)
    float* initial;         // Initial distribution
    int current_state;
} MarkovChain;

MarkovChain* markov_create(int num_states) {
    MarkovChain* mc = (MarkovChain*)malloc(sizeof(MarkovChain));
    mc->num_states = num_states;
    mc->transition = (float*)calloc(num_states * num_states, sizeof(float));
    mc->initial = (float*)calloc(num_states, sizeof(float));
    mc->current_state = 0;
    return mc;
}

void markov_free(MarkovChain* mc) {
    if (mc) {
        free(mc->transition);
        free(mc->initial);
        free(mc);
    }
}

// Sample next state from current state
int markov_step(MarkovChain* mc) {
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    
    for (int j = 0; j < mc->num_states; j++) {
        cumsum += mc->transition[mc->current_state * mc->num_states + j];
        if (r < cumsum) {
            mc->current_state = j;
            return j;
        }
    }
    
    return mc->num_states - 1;
}

// Compute stationary distribution (solve π = πP)
void markov_stationary(MarkovChain* mc, float* stationary) {
    // Power iteration method
    float* current = (float*)malloc(mc->num_states * sizeof(float));
    float* next = (float*)malloc(mc->num_states * sizeof(float));
    
    // Initialize uniform
    for (int i = 0; i < mc->num_states; i++) {
        current[i] = 1.0f / mc->num_states;
    }
    
    // Iterate until convergence
    for (int iter = 0; iter < 1000; iter++) {
        // next = current * P
        for (int j = 0; j < mc->num_states; j++) {
            next[j] = 0.0f;
            for (int i = 0; i < mc->num_states; i++) {
                next[j] += current[i] * mc->transition[i * mc->num_states + j];
            }
        }
        
        // Check convergence
        float diff = 0.0f;
        for (int i = 0; i < mc->num_states; i++) {
            diff += fabsf(next[i] - current[i]);
        }
        
        if (diff < 1e-6f) break;
        
        memcpy(current, next, mc->num_states * sizeof(float));
    }
    
    memcpy(stationary, current, mc->num_states * sizeof(float));
    free(current);
    free(next);
}

// ============================================================
// Hidden Markov Model
// ============================================================

typedef struct {
    int num_states;
    int num_observations;
    
    float* transition;      // A[i][j] = P(s_t+1=j | s_t=i)
    float* emission;        // B[i][k] = P(o_t=k | s_t=i)
    float* initial;         // π[i] = P(s_0=i)
} HMM;

HMM* hmm_create(int num_states, int num_observations) {
    HMM* hmm = (HMM*)malloc(sizeof(HMM));
    hmm->num_states = num_states;
    hmm->num_observations = num_observations;
    hmm->transition = (float*)calloc(num_states * num_states, sizeof(float));
    hmm->emission = (float*)calloc(num_states * num_observations, sizeof(float));
    hmm->initial = (float*)calloc(num_states, sizeof(float));
    return hmm;
}

void hmm_free(HMM* hmm) {
    if (hmm) {
        free(hmm->transition);
        free(hmm->emission);
        free(hmm->initial);
        free(hmm);
    }
}

// Forward algorithm: compute P(observations | model)
float hmm_forward(HMM* hmm, int* observations, int T) {
    int N = hmm->num_states;
    
    // Alpha[t][i] = P(o_1,...,o_t, s_t=i | model)
    float* alpha = (float*)malloc(T * N * sizeof(float));
    
    // Initialize: alpha[0][i] = π[i] * B[i][o_0]
    for (int i = 0; i < N; i++) {
        alpha[i] = hmm->initial[i] * hmm->emission[i * hmm->num_observations + observations[0]];
    }
    
    // Recursion
    for (int t = 1; t < T; t++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int i = 0; i < N; i++) {
                sum += alpha[(t-1)*N + i] * hmm->transition[i*N + j];
            }
            alpha[t*N + j] = sum * hmm->emission[j * hmm->num_observations + observations[t]];
        }
    }
    
    // Termination: sum over final states
    float prob = 0.0f;
    for (int i = 0; i < N; i++) {
        prob += alpha[(T-1)*N + i];
    }
    
    free(alpha);
    return prob;
}

// Viterbi algorithm: find most likely state sequence
void hmm_viterbi(HMM* hmm, int* observations, int T, int* states) {
    int N = hmm->num_states;
    
    // Delta[t][i] = max probability of state sequence ending in state i at time t
    float* delta = (float*)malloc(T * N * sizeof(float));
    int* psi = (int*)malloc(T * N * sizeof(int));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        delta[i] = hmm->initial[i] * hmm->emission[i * hmm->num_observations + observations[0]];
        psi[i] = 0;
    }
    
    // Recursion
    for (int t = 1; t < T; t++) {
        for (int j = 0; j < N; j++) {
            float max_val = -1.0f;
            int max_idx = 0;
            
            for (int i = 0; i < N; i++) {
                float val = delta[(t-1)*N + i] * hmm->transition[i*N + j];
                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }
            
            delta[t*N + j] = max_val * hmm->emission[j * hmm->num_observations + observations[t]];
            psi[t*N + j] = max_idx;
        }
    }
    
    // Backtrack
    float max_val = -1.0f;
    int max_idx = 0;
    for (int i = 0; i < N; i++) {
        if (delta[(T-1)*N + i] > max_val) {
            max_val = delta[(T-1)*N + i];
            max_idx = i;
        }
    }
    
    states[T-1] = max_idx;
    for (int t = T-2; t >= 0; t--) {
        states[t] = psi[(t+1)*N + states[t+1]];
    }
    
    free(delta);
    free(psi);
}

// Baum-Welch algorithm: train HMM parameters
void hmm_train(HMM* hmm, int** sequences, int* lengths, int num_seq, int max_iter) {
    int N = hmm->num_states;
    int M = hmm->num_observations;
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Accumulate statistics
        float* new_initial = (float*)calloc(N, sizeof(float));
        float* new_transition = (float*)calloc(N * N, sizeof(float));
        float* new_emission = (float*)calloc(N * M, sizeof(float));
        
        for (int seq = 0; seq < num_seq; seq++) {
            int T = lengths[seq];
            int* obs = sequences[seq];
            
            // Forward-backward
            float* alpha = (float*)malloc(T * N * sizeof(float));
            float* beta = (float*)malloc(T * N * sizeof(float));
            
            // Forward pass
            for (int i = 0; i < N; i++) {
                alpha[i] = hmm->initial[i] * hmm->emission[i*M + obs[0]];
            }
            for (int t = 1; t < T; t++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int i = 0; i < N; i++) {
                        sum += alpha[(t-1)*N + i] * hmm->transition[i*N + j];
                    }
                    alpha[t*N + j] = sum * hmm->emission[j*M + obs[t]];
                }
            }
            
            // Backward pass
            for (int i = 0; i < N; i++) {
                beta[(T-1)*N + i] = 1.0f;
            }
            for (int t = T-2; t >= 0; t--) {
                for (int i = 0; i < N; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; j++) {
                        sum += hmm->transition[i*N + j] * 
                               hmm->emission[j*M + obs[t+1]] * 
                               beta[(t+1)*N + j];
                    }
                    beta[t*N + i] = sum;
                }
            }
            
            // Update statistics (simplified)
            for (int t = 0; t < T; t++) {
                float norm = 0.0f;
                for (int i = 0; i < N; i++) {
                    norm += alpha[t*N + i] * beta[t*N + i];
                }
                
                for (int i = 0; i < N; i++) {
                    float gamma = alpha[t*N + i] * beta[t*N + i] / norm;
                    if (t == 0) new_initial[i] += gamma;
                    new_emission[i*M + obs[t]] += gamma;
                }
            }
            
            free(alpha);
            free(beta);
        }
        
        // Normalize and update
        float sum = 0.0f;
        for (int i = 0; i < N; i++) sum += new_initial[i];
        for (int i = 0; i < N; i++) hmm->initial[i] = new_initial[i] / sum;
        
        for (int i = 0; i < N; i++) {
            sum = 0.0f;
            for (int k = 0; k < M; k++) sum += new_emission[i*M + k];
            for (int k = 0; k < M; k++) hmm->emission[i*M + k] = new_emission[i*M + k] / sum;
        }
        
        free(new_initial);
        free(new_transition);
        free(new_emission);
    }
}
