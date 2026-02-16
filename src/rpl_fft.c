/*
 * RPL FFT — Cooley-Tukey radix-2 FFT/IFFT, RFFT, STFT
 */
#include "rpl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Internal complex FFT (in-place, Cooley-Tukey radix-2)
// real/imag interleaved: data[2*i] = real, data[2*i+1] = imag
static void fft_internal(float* data, uint32_t n, int sign) {
    // Bit-reversal permutation
    for (uint32_t i = 1, j = 0; i < n; i++) {
        uint32_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = data[2*i]; data[2*i] = data[2*j]; data[2*j] = tr;
            float ti = data[2*i+1]; data[2*i+1] = data[2*j+1]; data[2*j+1] = ti;
        }
    }
    // Butterfly
    for (uint32_t len = 2; len <= n; len <<= 1) {
        float ang = sign * 2.0f * (float)M_PI / len;
        float wr_step = cosf(ang), wi_step = sinf(ang);
        for (uint32_t i = 0; i < n; i += len) {
            float wr = 1.0f, wi = 0.0f;
            for (uint32_t j = 0; j < len/2; j++) {
                uint32_t u = i+j, v = i+j+len/2;
                float tr = data[2*v]*wr - data[2*v+1]*wi;
                float ti = data[2*v]*wi + data[2*v+1]*wr;
                data[2*v] = data[2*u] - tr;   data[2*v+1] = data[2*u+1] - ti;
                data[2*u] += tr;               data[2*u+1] += ti;
                float nwr = wr*wr_step - wi*wi_step;
                wi = wr*wi_step + wi*wr_step;
                wr = nwr;
            }
        }
    }
}

// FFT: input is (N,2) tensor [real,imag], output is (N,2) tensor
Tensor* tensor_fft(const Tensor* t) {
    uint32_t n = t->shape[0];
    // Ensure power of 2
    uint32_t np2 = 1; while (np2 < n) np2 <<= 1;
    uint32_t s[2] = {np2, 2};
    Tensor* out = tensor_create(2, s, false);
    tensor_fill(out, 0);
    // Copy input
    if (t->dims == 1) {
        for (uint32_t i = 0; i < n; i++) { out->data[2*i] = t->data[i]; out->data[2*i+1] = 0; }
    } else {
        for (uint32_t i = 0; i < n; i++) { out->data[2*i] = t->data[2*i]; out->data[2*i+1] = t->data[2*i+1]; }
    }
    fft_internal(out->data, np2, -1);
    return out;
}

Tensor* tensor_ifft(const Tensor* t) {
    uint32_t n = t->shape[0];
    uint32_t s[2] = {n, 2};
    Tensor* out = tensor_create(2, s, false);
    memcpy(out->data, t->data, n*2*sizeof(float));
    fft_internal(out->data, n, 1);
    float inv = 1.0f / n;
    for (uint32_t i = 0; i < 2*n; i++) out->data[i] *= inv;
    return out;
}

// RFFT: real input, output (N/2+1, 2)
Tensor* tensor_rfft(const Tensor* t) {
    Tensor* full = tensor_fft(t);
    uint32_t np2 = full->shape[0];
    uint32_t out_n = np2/2 + 1;
    uint32_t s[2] = {out_n, 2};
    Tensor* out = tensor_create(2, s, false);
    memcpy(out->data, full->data, out_n*2*sizeof(float));
    tensor_free(full);
    return out;
}

// IRFFT: (N/2+1, 2) complex -> N real
Tensor* tensor_irfft(const Tensor* t, uint32_t n) {
    uint32_t s2[2] = {n, 2};
    Tensor* full = tensor_create(2, s2, false);
    uint32_t half = t->shape[0];
    memcpy(full->data, t->data, half*2*sizeof(float));
    // Conjugate symmetry
    for (uint32_t i = half; i < n; i++) {
        full->data[2*i] = t->data[2*(n-i)];
        full->data[2*i+1] = -t->data[2*(n-i)+1];
    }
    fft_internal(full->data, n, 1);
    float inv = 1.0f / n;
    uint32_t s1[1] = {n};
    Tensor* out = tensor_create(1, s1, false);
    for (uint32_t i = 0; i < n; i++) out->data[i] = full->data[2*i] * inv;
    tensor_free(full);
    return out;
}

// STFT (simplified: Hann window, hop_length, n_fft)
Tensor* tensor_stft(const Tensor* t, uint32_t n_fft, uint32_t hop_length) {
    uint32_t n_frames = (t->size - n_fft) / hop_length + 1;
    uint32_t freq_bins = n_fft / 2 + 1;
    uint32_t s[3] = {freq_bins, n_frames, 2};
    Tensor* out = tensor_create(3, s, false);
    
    // Precompute Hann window
    float* win = (float*)malloc(n_fft * sizeof(float));
    for (uint32_t i = 0; i < n_fft; i++)
        win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (n_fft - 1)));
    
    float* buf = (float*)malloc(n_fft * 2 * sizeof(float));
    uint32_t np2 = 1; while (np2 < n_fft) np2 <<= 1;
    float* fft_buf = (float*)malloc(np2 * 2 * sizeof(float));
    
    for (uint32_t f = 0; f < n_frames; f++) {
        memset(fft_buf, 0, np2 * 2 * sizeof(float));
        for (uint32_t i = 0; i < n_fft; i++) {
            fft_buf[2*i] = t->data[f*hop_length+i] * win[i];
        }
        fft_internal(fft_buf, np2, -1);
        for (uint32_t b = 0; b < freq_bins; b++) {
            out->data[(b*n_frames+f)*2] = fft_buf[2*b];
            out->data[(b*n_frames+f)*2+1] = fft_buf[2*b+1];
        }
    }
    free(win); free(buf); free(fft_buf);
    return out;
}
