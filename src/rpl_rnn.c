/*
 * RPL Recurrent Neural Networks
 * Implementations for LSTM and GRU cells.
 */

#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================
// Sigmoid Helper
// ============================================================
static inline float sigmoidf_local(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// ============================================================
// LSTM
// ============================================================

struct LSTMLayer {
    uint32_t input_size;
    uint32_t hidden_size;
    
    Tensor* weight_ih; // [4 * hidden_size, input_size]
    Tensor* weight_hh; // [4 * hidden_size, hidden_size]
    Tensor* bias_ih;   // [4 * hidden_size]
    Tensor* bias_hh;   // [4 * hidden_size]
};

LSTMLayer* lstm_create(uint32_t input_size, uint32_t hidden_size) {
    LSTMLayer* layer = (LSTMLayer*)calloc(1, sizeof(LSTMLayer));
    layer->input_size = input_size;
    layer->hidden_size = hidden_size;
    
    uint32_t dim_ih[2] = {4 * hidden_size, input_size};
    layer->weight_ih = tensor_create(2, dim_ih, true);
    
    uint32_t dim_hh[2] = {4 * hidden_size, hidden_size};
    layer->weight_hh = tensor_create(2, dim_hh, true);
    
    uint32_t dim_b[1] = {4 * hidden_size};
    layer->bias_ih = tensor_create(1, dim_b, true);
    layer->bias_hh = tensor_create(1, dim_b, true);
    
    float stdv = 1.0f / sqrtf((float)hidden_size);
    for (uint32_t i = 0; i < layer->weight_ih->size; i++)
        layer->weight_ih->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * stdv;
    for (uint32_t i = 0; i < layer->weight_hh->size; i++)
        layer->weight_hh->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * stdv;
        
    tensor_fill(layer->bias_ih, 0.0f);
    tensor_fill(layer->bias_hh, 0.0f);
    
    return layer;
}

void tensor_lstm_forward(LSTMLayer* layer, const Tensor* input, Tensor* h_0, Tensor* c_0, Tensor* out_h, Tensor* out_c) {
    // input: [seq_len, batch_size, input_size]
    // h_0/c_0: [batch_size, hidden_size]
    // out_h/out_c: [seq_len, batch_size, hidden_size]
    
    uint32_t seq_len = input->shape[0];
    uint32_t batch = input->shape[1];
    uint32_t hidden = layer->hidden_size;
    uint32_t input_sz = layer->input_size;
    
    float* h_prev = (float*)malloc(batch * hidden * sizeof(float));
    float* c_prev = (float*)malloc(batch * hidden * sizeof(float));
    
    if (h_0) memcpy(h_prev, h_0->data, batch * hidden * sizeof(float));
    else memset(h_prev, 0, batch * hidden * sizeof(float));
        
    if (c_0) memcpy(c_prev, c_0->data, batch * hidden * sizeof(float));
    else memset(c_prev, 0, batch * hidden * sizeof(float));
    
    float* gates = (float*)malloc(batch * 4 * hidden * sizeof(float));
    
    for (uint32_t t = 0; t < seq_len; t++) {
        // Compute W_ih * x + b_ih
        memset(gates, 0, batch * 4 * hidden * sizeof(float));
        
        for (uint32_t b = 0; b < batch; b++) {
            const float* x = &input->data[(t * batch + b) * input_sz];
            float* g = &gates[b * 4 * hidden];
            
            for (uint32_t i = 0; i < 4 * hidden; i++) {
                g[i] = layer->bias_ih->data[i] + layer->bias_hh->data[i];
                for (uint32_t j = 0; j < input_sz; j++) {
                    g[i] += layer->weight_ih->data[i * input_sz + j] * x[j];
                }
                for (uint32_t j = 0; j < hidden; j++) {
                    g[i] += layer->weight_hh->data[i * hidden + j] * h_prev[b * hidden + j];
                }
            }
        }
        
        for (uint32_t b = 0; b < batch; b++) {
            float* g = &gates[b * 4 * hidden];
            float* h_cur = &out_h->data[(t * batch + b) * hidden];
            float* c_cur = &out_c->data[(t * batch + b) * hidden];
            
            for (uint32_t i = 0; i < hidden; i++) {
                float i_t = sigmoidf_local(g[0 * hidden + i]); // input gate
                float f_t = sigmoidf_local(g[1 * hidden + i]); // forget gate
                float g_t = tanhf(g[2 * hidden + i]);          // cell gate
                float o_t = sigmoidf_local(g[3 * hidden + i]); // output gate
                
                c_cur[i] = f_t * c_prev[b * hidden + i] + i_t * g_t;
                h_cur[i] = o_t * tanhf(c_cur[i]);
                
                c_prev[b * hidden + i] = c_cur[i];
                h_prev[b * hidden + i] = h_cur[i];
            }
        }
    }
    
    free(gates);
    free(h_prev);
    free(c_prev);
}

void lstm_free(LSTMLayer* layer) {
    tensor_free(layer->weight_ih);
    tensor_free(layer->weight_hh);
    tensor_free(layer->bias_ih);
    tensor_free(layer->bias_hh);
    free(layer);
}

// ============================================================
// GRU
// ============================================================

struct GRULayer {
    uint32_t input_size;
    uint32_t hidden_size;
    
    Tensor* weight_ih; // [3 * hidden_size, input_size]
    Tensor* weight_hh; // [3 * hidden_size, hidden_size]
    Tensor* bias_ih;   // [3 * hidden_size]
    Tensor* bias_hh;   // [3 * hidden_size]
};

GRULayer* gru_create(uint32_t input_size, uint32_t hidden_size) {
    GRULayer* layer = (GRULayer*)calloc(1, sizeof(GRULayer));
    layer->input_size = input_size;
    layer->hidden_size = hidden_size;
    
    uint32_t dim_ih[2] = {3 * hidden_size, input_size};
    layer->weight_ih = tensor_create(2, dim_ih, true);
    
    uint32_t dim_hh[2] = {3 * hidden_size, hidden_size};
    layer->weight_hh = tensor_create(2, dim_hh, true);
    
    uint32_t dim_b[1] = {3 * hidden_size};
    layer->bias_ih = tensor_create(1, dim_b, true);
    layer->bias_hh = tensor_create(1, dim_b, true);
    
    float stdv = 1.0f / sqrtf((float)hidden_size);
    for (uint32_t i = 0; i < layer->weight_ih->size; i++)
        layer->weight_ih->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * stdv;
    for (uint32_t i = 0; i < layer->weight_hh->size; i++)
        layer->weight_hh->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * stdv;
        
    tensor_fill(layer->bias_ih, 0.0f);
    tensor_fill(layer->bias_hh, 0.0f);
    
    return layer;
}

void tensor_gru_forward(GRULayer* layer, const Tensor* input, Tensor* h_0, Tensor* out_h) {
    // input: [seq_len, batch_size, input_size]
    uint32_t seq_len = input->shape[0];
    uint32_t batch = input->shape[1];
    uint32_t hidden = layer->hidden_size;
    uint32_t input_sz = layer->input_size;
    
    float* h_prev = (float*)malloc(batch * hidden * sizeof(float));
    if (h_0) memcpy(h_prev, h_0->data, batch * hidden * sizeof(float));
    else memset(h_prev, 0, batch * hidden * sizeof(float));
    
    float* gates_ih = (float*)malloc(batch * 3 * hidden * sizeof(float));
    float* gates_hh = (float*)malloc(batch * 3 * hidden * sizeof(float));
    
    for (uint32_t t = 0; t < seq_len; t++) {
        memset(gates_ih, 0, batch * 3 * hidden * sizeof(float));
        memset(gates_hh, 0, batch * 3 * hidden * sizeof(float));
        
        for (uint32_t b = 0; b < batch; b++) {
            const float* x = &input->data[(t * batch + b) * input_sz];
            float* g_ih = &gates_ih[b * 3 * hidden];
            float* g_hh = &gates_hh[b * 3 * hidden];
            
            for (uint32_t i = 0; i < 3 * hidden; i++) {
                g_ih[i] = layer->bias_ih->data[i];
                for (uint32_t j = 0; j < input_sz; j++) {
                    g_ih[i] += layer->weight_ih->data[i * input_sz + j] * x[j];
                }
                
                g_hh[i] = layer->bias_hh->data[i];
                for (uint32_t j = 0; j < hidden; j++) {
                    g_hh[i] += layer->weight_hh->data[i * hidden + j] * h_prev[b * hidden + j];
                }
            }
        }
        
        for (uint32_t b = 0; b < batch; b++) {
            float* g_ih = &gates_ih[b * 3 * hidden];
            float* g_hh = &gates_hh[b * 3 * hidden];
            float* h_cur = &out_h->data[(t * batch + b) * hidden];
            
            for (uint32_t i = 0; i < hidden; i++) {
                float r_t = sigmoidf_local(g_ih[0 * hidden + i] + g_hh[0 * hidden + i]); // reset gate
                float z_t = sigmoidf_local(g_ih[1 * hidden + i] + g_hh[1 * hidden + i]); // update gate
                float n_t = tanhf(g_ih[2 * hidden + i] + r_t * g_hh[2 * hidden + i]);    // new gate
                
                h_cur[i] = (1.0f - z_t) * n_t + z_t * h_prev[b * hidden + i];
                h_prev[b * hidden + i] = h_cur[i];
            }
        }
    }
    
    free(gates_ih);
    free(gates_hh);
    free(h_prev);
}

void gru_free(GRULayer* layer) {
    tensor_free(layer->weight_ih);
    tensor_free(layer->weight_hh);
    tensor_free(layer->bias_ih);
    tensor_free(layer->bias_hh);
    free(layer);
}
