#include "rpl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_GPU

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>

static EGLDisplay display = EGL_NO_DISPLAY;
static EGLContext context = EGL_NO_CONTEXT;

// Initialize headless EGL context for compute shaders
// Initialize headless EGL context for compute shaders
#ifndef EGL_PLATFORM_SURFACELESS_MESA
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD
#endif

// Function pointers for GLES 3.1
static PFNGLGENBUFFERSPROC p_glGenBuffers = NULL;
static PFNGLBINDBUFFERPROC p_glBindBuffer = NULL;
static PFNGLBUFFERDATAPROC p_glBufferData = NULL;
static PFNGLBUFFERSUBDATAPROC p_glBufferSubData = NULL;
static PFNGLMAPBUFFERRANGEPROC p_glMapBufferRange = NULL;
static PFNGLUNMAPBUFFERPROC p_glUnmapBuffer = NULL;
static PFNGLDELETEBUFFERSPROC p_glDeleteBuffers = NULL;
static PFNGLCREATESHADERPROC p_glCreateShader = NULL;
static PFNGLSHADERSOURCEPROC p_glShaderSource = NULL;
static PFNGLCOMPILESHADERPROC p_glCompileShader = NULL;
static PFNGLGETSHADERIVPROC p_glGetShaderiv = NULL;
static PFNGLGETSHADERINFOLOGPROC p_glGetShaderInfoLog = NULL;
static PFNGLCREATEPROGRAMPROC p_glCreateProgram = NULL;
static PFNGLATTACHSHADERPROC p_glAttachShader = NULL;
static PFNGLLINKPROGRAMPROC p_glLinkProgram = NULL;
static PFNGLDELETESHADERPROC p_glDeleteShader = NULL;
static PFNGLUSEPROGRAMPROC p_glUseProgram = NULL;
static PFNGLBINDBUFFERBASEPROC p_glBindBufferBase = NULL;
static PFNGLBINDBUFFERRANGEPROC p_glBindBufferRange = NULL;
static PFNGLUNIFORM1UIPROC p_glUniform1ui = NULL;
static PFNGLUNIFORM1FPROC p_glUniform1f = NULL;
static PFNGLUNIFORM2IPROC p_glUniform2i = NULL;
static PFNGLUNIFORM4IPROC p_glUniform4i = NULL;
static PFNGLGETUNIFORMLOCATIONPROC p_glGetUniformLocation = NULL;
static PFNGLDISPATCHCOMPUTEPROC p_glDispatchCompute = NULL;
static PFNGLMEMORYBARRIERPROC p_glMemoryBarrier = NULL;
static PFNGLUNIFORM1IPROC p_glUniform1i = NULL;
static PFNGLGETSTRINGPROC p_glGetString = NULL;
/* Texture function pointers (for Conv2D via GL_TEXTURE_2D) */
typedef void (GL_APIENTRYP PFNGLGENTEXTURESPROC_)(GLsizei n, GLuint *textures);
typedef void (GL_APIENTRYP PFNGLBINDTEXTUREPROC_)(GLenum target, GLuint texture);
typedef void (GL_APIENTRYP PFNGLTEXIMAGE2DPROC_)(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void *pixels);
typedef void (GL_APIENTRYP PFNGLTEXPARAMETERIPROC_)(GLenum target, GLenum pname, GLint param);
typedef void (GL_APIENTRYP PFNGLDELETETEXTURESPROC_)(GLsizei n, const GLuint *textures);
typedef void (GL_APIENTRYP PFNGLACTIVETEXTUREPROC_)(GLenum texture);
static PFNGLGENTEXTURESPROC_  p_glGenTextures  = NULL;
static PFNGLBINDTEXTUREPROC_  p_glBindTexture  = NULL;
static PFNGLTEXIMAGE2DPROC_   p_glTexImage2D   = NULL;
static PFNGLTEXPARAMETERIPROC_ p_glTexParameteri = NULL;
static PFNGLDELETETEXTURESPROC_ p_glDeleteTextures = NULL;
static PFNGLACTIVETEXTUREPROC_  p_glActiveTexture  = NULL;

static void load_gl_funcs() {
    p_glGenBuffers = (PFNGLGENBUFFERSPROC)eglGetProcAddress("glGenBuffers");
    p_glBindBuffer = (PFNGLBINDBUFFERPROC)eglGetProcAddress("glBindBuffer");
    p_glBufferData = (PFNGLBUFFERDATAPROC)eglGetProcAddress("glBufferData");
    p_glBufferSubData = (PFNGLBUFFERSUBDATAPROC)eglGetProcAddress("glBufferSubData");
    p_glMapBufferRange = (PFNGLMAPBUFFERRANGEPROC)eglGetProcAddress("glMapBufferRange");
    p_glUnmapBuffer = (PFNGLUNMAPBUFFERPROC)eglGetProcAddress("glUnmapBuffer");
    p_glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)eglGetProcAddress("glDeleteBuffers");
    p_glCreateShader = (PFNGLCREATESHADERPROC)eglGetProcAddress("glCreateShader");
    p_glShaderSource = (PFNGLSHADERSOURCEPROC)eglGetProcAddress("glShaderSource");
    p_glCompileShader = (PFNGLCOMPILESHADERPROC)eglGetProcAddress("glCompileShader");
    p_glGetShaderiv = (PFNGLGETSHADERIVPROC)eglGetProcAddress("glGetShaderiv");
    p_glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)eglGetProcAddress("glGetShaderInfoLog");
    p_glCreateProgram = (PFNGLCREATEPROGRAMPROC)eglGetProcAddress("glCreateProgram");
    p_glAttachShader = (PFNGLATTACHSHADERPROC)eglGetProcAddress("glAttachShader");
    p_glLinkProgram = (PFNGLLINKPROGRAMPROC)eglGetProcAddress("glLinkProgram");
    p_glDeleteShader = (PFNGLDELETESHADERPROC)eglGetProcAddress("glDeleteShader");
    p_glUseProgram = (PFNGLUSEPROGRAMPROC)eglGetProcAddress("glUseProgram");
    p_glBindBufferBase = (PFNGLBINDBUFFERBASEPROC)eglGetProcAddress("glBindBufferBase");
    p_glBindBufferRange = (PFNGLBINDBUFFERRANGEPROC)eglGetProcAddress("glBindBufferRange");
    p_glUniform1ui = (PFNGLUNIFORM1UIPROC)eglGetProcAddress("glUniform1ui");
    p_glUniform1f = (PFNGLUNIFORM1FPROC)eglGetProcAddress("glUniform1f");
    p_glUniform2i = (PFNGLUNIFORM2IPROC)eglGetProcAddress("glUniform2i");
    p_glUniform4i = (PFNGLUNIFORM4IPROC)eglGetProcAddress("glUniform4i");
    p_glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)eglGetProcAddress("glGetUniformLocation");
    p_glGetString = (PFNGLGETSTRINGPROC)eglGetProcAddress("glGetString");
    p_glDispatchCompute = (PFNGLDISPATCHCOMPUTEPROC)eglGetProcAddress("glDispatchCompute");
    p_glMemoryBarrier = (PFNGLMEMORYBARRIERPROC)eglGetProcAddress("glMemoryBarrier");
    p_glUniform1i = (PFNGLUNIFORM1IPROC)eglGetProcAddress("glUniform1i");
    /* Texture functions */
    p_glGenTextures   = (PFNGLGENTEXTURESPROC_)eglGetProcAddress("glGenTextures");
    p_glBindTexture   = (PFNGLBINDTEXTUREPROC_)eglGetProcAddress("glBindTexture");
    p_glTexImage2D    = (PFNGLTEXIMAGE2DPROC_)eglGetProcAddress("glTexImage2D");
    p_glTexParameteri = (PFNGLTEXPARAMETERIPROC_)eglGetProcAddress("glTexParameteri");
    p_glDeleteTextures= (PFNGLDELETETEXTURESPROC_)eglGetProcAddress("glDeleteTextures");
    p_glActiveTexture = (PFNGLACTIVETEXTUREPROC_)eglGetProcAddress("glActiveTexture");
}

// Macro helper to call dynamic pointers
#define glGenBuffers         p_glGenBuffers
#define glBindBuffer         p_glBindBuffer
#define glBufferData         p_glBufferData
#define glBufferSubData      p_glBufferSubData
#define glMapBufferRange     p_glMapBufferRange
#define glUnmapBuffer        p_glUnmapBuffer
#define glDeleteBuffers      p_glDeleteBuffers
#define glCreateShader       p_glCreateShader
#define glShaderSource       p_glShaderSource
#define glCompileShader      p_glCompileShader
#define glGetShaderiv        p_glGetShaderiv
#define glGetShaderInfoLog   p_glGetShaderInfoLog
#define glCreateProgram      p_glCreateProgram
#define glAttachShader       p_glAttachShader
#define glLinkProgram        p_glLinkProgram
#define glDeleteShader       p_glDeleteShader
#define glUseProgram         p_glUseProgram
#define glBindBufferBase     p_glBindBufferBase
#define glBindBufferRange    p_glBindBufferRange
#define glUniform1ui         p_glUniform1ui
#define glUniform1f          p_glUniform1f
#define glUniform2i          p_glUniform2i
#define glUniform4i          p_glUniform4i
#define glGetUniformLocation p_glGetUniformLocation
#define glDispatchCompute    p_glDispatchCompute
#define glMemoryBarrier      p_glMemoryBarrier
#define glUniform1i          p_glUniform1i
#define glGetString          p_glGetString
#define glGenTextures        p_glGenTextures
#define glBindTexture        p_glBindTexture
#define glTexImage2D         p_glTexImage2D
#define glTexParameteri      p_glTexParameteri
#define glDeleteTextures     p_glDeleteTextures
#define glActiveTexture      p_glActiveTexture


bool rpl_gpu_init() {
    if (display != EGL_NO_DISPLAY) return true;

    // Headless/Surfaceless initialization
    PFNEGLGETPLATFORMDISPLAYPROC eglGetPlatformDisplay = 
        (PFNEGLGETPLATFORMDISPLAYPROC)eglGetProcAddress("eglGetPlatformDisplay");

    if (eglGetPlatformDisplay) {
        display = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
    }

    if (display == EGL_NO_DISPLAY) {
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    }
    
    EGLint major, minor;
    if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor)) {
        display = EGL_NO_DISPLAY;
        return false;
    }
    
    // ... rest of config code
    // Try a few different config strategies
    EGLConfig config;
    EGLint numConfigs = 0;
    
    // Strategy 1: Explicit 8-bit RGBA with PBUFFER (Surfaceless friendly)
    EGLint configAttribs8888[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_NONE
    };
    
    // Strategy 2: Minimal ES3 with PBUFFER
    EGLint configAttribsMin[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_NONE
    };
    
    // Strategy 3: Manual Iteration (Brute Force)
    // If eglChooseConfig fails or returns 0, let's just get ALL configs and inspect them
    EGLConfig* all_configs;
    EGLint num_total_configs;
    if (eglGetConfigs(display, NULL, 0, &num_total_configs) && num_total_configs > 0) {
        all_configs = (EGLConfig*)malloc(num_total_configs * sizeof(EGLConfig));
        eglGetConfigs(display, all_configs, num_total_configs, &num_total_configs);
        
        for (int i = 0; i < num_total_configs; i++) {
            EGLint renderable;
            eglGetConfigAttrib(display, all_configs[i], EGL_RENDERABLE_TYPE, &renderable);
            if (renderable & EGL_OPENGL_ES3_BIT) {
                printf("RPL GPU: Found GLES3 compatible config via manual search (Index %d)\n", i);
                config = all_configs[i];
                numConfigs = 1; // Mark as found
                free(all_configs);
                goto config_found;
            }
        }
        free(all_configs);
    }
    
    // Failed all strategies
    fprintf(stderr, "Failed to find ANY EGL config with EGL_OPENGL_ES3_BIT.\n");
    fprintf(stderr, "EGL Error: 0x%x\n", eglGetError());
    return false;

config_found:;

    const EGLint contextAttribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 3,
        EGL_NONE
    };

    context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
    if (context == EGL_NO_CONTEXT) {
        fprintf(stderr, "Failed to create EGL context\n");
        return false;
    }

    if (!eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context)) {
        fprintf(stderr, "Failed to make context current\n");
        return false;
    }
    
    // Load function pointers
    load_gl_funcs();

    if (p_glGetString) {
        printf("RPL GPU Initialized: %s\n", p_glGetString(GL_VERSION));
    }
    
    return true;
}

void rpl_gpu_shutdown() {
    if (display != EGL_NO_DISPLAY) {
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(display, context);
        eglTerminate(display);
        display = EGL_NO_DISPLAY;
    }
}

// Create SSBO and upload data
void tensor_to_gpu(Tensor* t) {
    if (t->device == DEVICE_GPU && t->gpu_buffer != 0) return; // Already on GPU

    if (!rpl_gpu_init()) return;

    GLuint buffer;
    if (t->gpu_buffer != 0) {
        buffer = t->gpu_buffer;
    } else {
        glGenBuffers(1, &buffer);
        t->gpu_buffer = buffer;
    }
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, t->size * sizeof(float), t->data, GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    t->device = DEVICE_GPU;
}

// Download data from SSBO to CPU
void tensor_from_gpu(Tensor* t) {
    if (t->device == DEVICE_CPU) return; // Already on CPU

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, t->gpu_buffer);
    void* ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, t->size * sizeof(float), GL_MAP_READ_BIT);
    if (ptr) {
        memcpy(t->data, ptr, t->size * sizeof(float));
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    
    t->device = DEVICE_CPU; 
}

void tensor_free_gpu(Tensor* t) {
    if (t->gpu_buffer != 0) {
        if (p_glDeleteBuffers) p_glDeleteBuffers(1, &t->gpu_buffer);
        t->gpu_buffer = 0;
    }
    t->device = DEVICE_CPU;
}

// Simple compute shader compiler
GLuint compile_compute_shader(const char* source) {
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        fprintf(stderr, "Compute shader compilation failed:\n%s\n", infoLog);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader); // Marked for deletion

    return program;
}

// ============================================================
// Compute Shaders
// ============================================================

static const char* BINARY_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer InputA { float data_a[]; };\n"
    "layout(std430, binding = 1) readonly buffer InputB { float data_b[]; };\n"
    "layout(std430, binding = 2) writeonly buffer Output { float data_out[]; };\n"
    "uniform uint size;\n"
    "uniform int op;\n" // 0: add, 1: sub, 2: mul, 3: div
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    if (id4 + 3u < size) {\n"
    "        vec4 va = vec4(data_a[id4], data_a[id4+1u], data_a[id4+2u], data_a[id4+3u]);\n"
    "        vec4 vb = vec4(data_b[id4], data_b[id4+1u], data_b[id4+2u], data_b[id4+3u]);\n"
    "        vec4 vo;\n"
    "        if (op == 0) vo = va + vb;\n"
    "        else if (op == 1) vo = va - vb;\n"
    "        else if (op == 2) vo = va * vb;\n"
    "        else vo = va / vb;\n"
    "        data_out[id4] = vo.x; data_out[id4+1u] = vo.y; data_out[id4+2u] = vo.z; data_out[id4+3u] = vo.w;\n"
    "    } else {\n"
    "        for (uint i = 0u; i < 4u; i++) {\n"
    "            uint idx = id4 + i;\n"
    "            if (idx < size) {\n"
    "                if (op == 0) data_out[idx] = data_a[idx] + data_b[idx];\n"
    "                else if (op == 1) data_out[idx] = data_a[idx] - data_b[idx];\n"
    "                else if (op == 2) data_out[idx] = data_a[idx] * data_b[idx];\n"
    "                else data_out[idx] = data_a[idx] / data_b[idx];\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint binary_program = 0;

void dispatch_binary_op(Tensor* out, const Tensor* a, const Tensor* b, int op) {
    if (!rpl_gpu_init()) return;

    tensor_to_gpu((Tensor*)a);
    tensor_to_gpu((Tensor*)b);
    
    if (out->device != DEVICE_GPU) {
        tensor_to_gpu(out);
    }

    if (binary_program == 0) {
        binary_program = compile_compute_shader(BINARY_SHADER_SRC);
        if (binary_program == 0) return;
    }

    glUseProgram(binary_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, a->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, out->gpu_buffer);

    glUniform1ui(glGetUniformLocation(binary_program, "size"), out->size);
    glUniform1i(glGetUniformLocation(binary_program, "op"), op);

    GLuint num_groups = (out->size + 1023) / 1024;
    glDispatchCompute(num_groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void tensor_add_gpu(Tensor* out, const Tensor* a, const Tensor* b) {
    dispatch_binary_op(out, a, b, 0);
}

void tensor_sub_gpu(Tensor* out, const Tensor* a, const Tensor* b) {
    dispatch_binary_op(out, a, b, 1);
}

void tensor_mul_gpu(Tensor* out, const Tensor* a, const Tensor* b) {
    dispatch_binary_op(out, a, b, 2);
}

void tensor_div_gpu(Tensor* out, const Tensor* a, const Tensor* b) {
    dispatch_binary_op(out, a, b, 3);
}

// ===================================
// General GEMM Shader (tiled 16×16, supports trans_a / trans_b / alpha / beta)
// ===================================
// A: [Ma × Ka_stored], B: [Kb_stored × N]
// Logical M = Ma (if !trans_a) or Ka_stored (if trans_a)
// Logical K = Ka_stored (if !trans_a) or Ma (if trans_a)
// B stored dims:  !trans_b → [K × N],  trans_b → [N × K]
// ===================================
static const char* GEMM_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 16, local_size_y = 16) in;\n"
    "layout(std430, binding = 0) readonly buffer InputA { float A[]; };\n"
    "layout(std430, binding = 1) readonly buffer InputB { float B[]; };\n"
    "layout(std430, binding = 2) buffer Output { float C[]; };\n"   /* readwrite for beta */
    "uniform uint M;\n"
    "uniform uint N;\n"
    "uniform uint K;\n"
    "uniform float alpha;\n"
    "uniform float beta;\n"
    "uniform int trans_a;\n"   /* 1 = read A^T */
    "uniform int trans_b;\n"   /* 1 = read B^T */
    "shared float sA[32][16];\n"
    "shared float sB[16][32];\n"
    "void main() {\n"
    "    uint tid = (gl_LocalInvocationID.y << 4u) | gl_LocalInvocationID.x;\n"
    "    uint brow = gl_WorkGroupID.y << 5u;\n"
    "    uint bcol = gl_WorkGroupID.x << 5u;\n"
    "    float c00 = 0.0, c01 = 0.0;\n"
    "    float c10 = 0.0, c11 = 0.0;\n"
    "    for (uint bk = 0u; bk < K; bk += 16u) {\n"
    "        uint idxA0 = tid << 1u;\n"
    "        uint idxA1 = idxA0 + 1u;\n"
    "        uint rA0 = idxA0 >> 4u, kA0 = idxA0 & 15u;\n"
    "        uint rA1 = idxA1 >> 4u, kA1 = idxA1 & 15u;\n"
    "        uint grA0 = brow + rA0, gk0 = bk + kA0;\n"
    "        uint grA1 = brow + rA1, gk1 = bk + kA1;\n"
    "        sA[rA0][kA0] = (grA0 < M && gk0 < K) ? ((trans_a != 0) ? A[gk0 * M + grA0] : A[grA0 * K + gk0]) : 0.0;\n"
    "        sA[rA1][kA1] = (grA1 < M && gk1 < K) ? ((trans_a != 0) ? A[gk1 * M + grA1] : A[grA1 * K + gk1]) : 0.0;\n"
    "        uint kB0 = idxA0 >> 5u, cB0 = idxA0 & 31u;\n"
    "        uint kB1 = idxA1 >> 5u, cB1 = idxA1 & 31u;\n"
    "        uint gkB0 = bk + kB0, gcB0 = bcol + cB0;\n"
    "        uint gkB1 = bk + kB1, gcB1 = bcol + cB1;\n"
    "        sB[kB0][cB0] = (gkB0 < K && gcB0 < N) ? ((trans_b != 0) ? B[gcB0 * K + gkB0] : B[gkB0 * N + gcB0]) : 0.0;\n"
    "        sB[kB1][cB1] = (gkB1 < K && gcB1 < N) ? ((trans_b != 0) ? B[gcB1 * K + gkB1] : B[gkB1 * N + gcB1]) : 0.0;\n"
    "        barrier();\n"
    "        uint lr = gl_LocalInvocationID.y << 1u;\n"
    "        uint lc = gl_LocalInvocationID.x << 1u;\n"
    "        for (uint k = 0u; k < 16u; k++) {\n"
    "            float a0 = sA[lr + 0u][k], a1 = sA[lr + 1u][k];\n"
    "            float b0 = sB[k][lc + 0u], b1 = sB[k][lc + 1u];\n"
    "            c00 += a0 * b0; c01 += a0 * b1;\n"
    "            c10 += a1 * b0; c11 += a1 * b1;\n"
    "        }\n"
    "        barrier();\n"
    "    }\n"
    "    uint gr = brow + (gl_LocalInvocationID.y << 1u);\n"
    "    uint gc = bcol + (gl_LocalInvocationID.x << 1u);\n"
    "    if (gr + 0u < M) {\n"
    "        if (gc + 0u < N) { uint idx = (gr + 0u) * N + gc + 0u; C[idx] = alpha * c00 + beta * C[idx]; }\n"
    "        if (gc + 1u < N) { uint idx = (gr + 0u) * N + gc + 1u; C[idx] = alpha * c01 + beta * C[idx]; }\n"
    "    }\n"
    "    if (gr + 1u < M) {\n"
    "        if (gc + 0u < N) { uint idx = (gr + 1u) * N + gc + 0u; C[idx] = alpha * c10 + beta * C[idx]; }\n"
    "        if (gc + 1u < N) { uint idx = (gr + 1u) * N + gc + 1u; C[idx] = alpha * c11 + beta * C[idx]; }\n"
    "    }\n"
    "}\n";

static GLuint gemm_program = 0;

/* Full GEMM: C = alpha * op(A) @ op(B) + beta * C */
void tensor_gemm_gpu(Tensor* C, const Tensor* A, const Tensor* B,
                     uint32_t M, uint32_t N, uint32_t K,
                     float alpha, float beta,
                     bool trans_a, bool trans_b) {
    if (!rpl_gpu_init()) return;

    tensor_to_gpu((Tensor*)A);
    tensor_to_gpu((Tensor*)B);
    if (C->device != DEVICE_GPU) {
        C->dims    = 2;
        C->shape[0] = M;
        C->shape[1] = N;
        C->dims     = 2;
        C->size     = M * N;
        tensor_to_gpu(C);
    }

    if (gemm_program == 0) {
        gemm_program = compile_compute_shader(GEMM_SHADER_SRC);
        if (gemm_program == 0) return;
    }

    glUseProgram(gemm_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, A->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, B->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, C->gpu_buffer);
    glUniform1ui(glGetUniformLocation(gemm_program, "M"), M);
    glUniform1ui(glGetUniformLocation(gemm_program, "N"), N);
    glUniform1ui(glGetUniformLocation(gemm_program, "K"), K);
    glUniform1f(glGetUniformLocation(gemm_program, "alpha"), alpha);
    glUniform1f(glGetUniformLocation(gemm_program, "beta"),  beta);
    glUniform1i(glGetUniformLocation(gemm_program, "trans_a"), trans_a ? 1 : 0);
    glUniform1i(glGetUniformLocation(gemm_program, "trans_b"), trans_b ? 1 : 0);

    GLuint groups_x = (N + 31) / 32;
    GLuint groups_y = (M + 31) / 32;
    glDispatchCompute(groups_x, groups_y, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

/* Backward-compat wrapper: simple non-transposed matmul with no scaling */
void tensor_matmul_gpu(Tensor* C, const Tensor* A, const Tensor* B) {
    uint32_t M = A->shape[0];
    uint32_t K = A->shape[1];
    uint32_t N = B->shape[1];
    if (B->shape[0] != K) {
        fprintf(stderr, "GEMM shape mismatch: %ux%u vs %ux%u\n", M, K, B->shape[0], N);
        return;
    }
    tensor_gemm_gpu(C, A, B, M, N, K, 1.0f, 0.0f, false, false);
}


// ===================================
// Activation Kernels
// ===================================

static const char* RELU_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            out_data[idx] = max(in_data[idx], 0.0);\n"
    "        }\n"
    "    }\n"
    "}\n";

static const char* SIGMOID_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            float val = in_data[idx];\n"
    "            out_data[idx] = 1.0 / (1.0 + exp(-val));\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint relu_program = 0;
static GLuint sigmoid_program = 0;

void dispatch_unary_op(Tensor* out, const Tensor* in, GLuint* program_ptr, const char* source) {
    if (!rpl_gpu_init()) return;

    tensor_to_gpu((Tensor*)in);
    
    if (out->device != DEVICE_GPU) {
        // Assume same shape as input
        out->dims = in->dims;
        memcpy(out->shape, in->shape, sizeof(in->shape));
        out->size = in->size;
        tensor_to_gpu(out);
    }

    if (*program_ptr == 0) {
        *program_ptr = compile_compute_shader(source);
        if (*program_ptr == 0) return;
    }

    glUseProgram(*program_ptr);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out->gpu_buffer);

    glUniform1ui(glGetUniformLocation(*program_ptr, "size"), out->size);

    GLuint num_groups = (out->size + 1023) / 1024;
    glDispatchCompute(num_groups, 1, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void tensor_relu_gpu(Tensor* out, const Tensor* in) {
    dispatch_unary_op(out, in, &relu_program, RELU_SHADER_SRC);
}

void tensor_sigmoid_gpu(Tensor* out, const Tensor* in) {
    dispatch_unary_op(out, in, &sigmoid_program, SIGMOID_SHADER_SRC);
}

// ===================================
// Tanh & GELU Kernels
// ===================================

static const char* TANH_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            out_data[idx] = tanh(in_data[idx]);\n"
    "        }\n"
    "    }\n"
    "}\n";

static const char* GELU_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            float x = in_data[idx];\n"
    "            const float SQRT_2_OVER_PI = 0.7978845608;\n"
    "            const float A = 0.044715;\n"
    "            float inner = SQRT_2_OVER_PI * (x + A * x * x * x);\n"
    "            out_data[idx] = 0.5 * x * (1.0 + tanh(inner));\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint tanh_program = 0;
static GLuint gelu_program = 0;

void tensor_tanh_gpu(Tensor* out, const Tensor* in) {
    dispatch_unary_op(out, in, &tanh_program, TANH_SHADER_SRC);
}

void tensor_gelu_gpu(Tensor* out, const Tensor* in) {
    dispatch_unary_op(out, in, &gelu_program, GELU_SHADER_SRC);
}

// ===================================
// LeakyReLU Kernel
// ===================================

static const char* LEAKY_RELU_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "uniform float negative_slope;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            float x = in_data[idx];\n"
    "            out_data[idx] = (x >= 0.0) ? x : negative_slope * x;\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint leaky_relu_program = 0;

void tensor_leaky_relu_gpu(Tensor* out, const Tensor* in, float negative_slope) {
    if (!rpl_gpu_init()) return;
    tensor_to_gpu((Tensor*)in);
    if (out->device != DEVICE_GPU) {
        out->dims = in->dims;
        memcpy(out->shape, in->shape, sizeof(in->shape));
        out->size = in->size;
        tensor_to_gpu(out);
    }
    if (leaky_relu_program == 0) {
        leaky_relu_program = compile_compute_shader(LEAKY_RELU_SHADER_SRC);
        if (leaky_relu_program == 0) return;
    }
    glUseProgram(leaky_relu_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out->gpu_buffer);
    glUniform1ui(glGetUniformLocation(leaky_relu_program, "size"), out->size);
    glUniform1f(glGetUniformLocation(leaky_relu_program, "negative_slope"), negative_slope);
    glDispatchCompute((out->size + 1023) / 1024, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ===================================
// ELU Kernel
// ===================================

static const char* ELU_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "uniform float alpha;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            float x = in_data[idx];\n"
    "            out_data[idx] = (x >= 0.0) ? x : alpha * (exp(x) - 1.0);\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint elu_program = 0;

void tensor_elu_gpu(Tensor* out, const Tensor* in, float alpha) {
    if (!rpl_gpu_init()) return;
    tensor_to_gpu((Tensor*)in);
    if (out->device != DEVICE_GPU) {
        out->dims = in->dims;
        memcpy(out->shape, in->shape, sizeof(in->shape));
        out->size = in->size;
        tensor_to_gpu(out);
    }
    if (elu_program == 0) {
        elu_program = compile_compute_shader(ELU_SHADER_SRC);
        if (elu_program == 0) return;
    }
    glUseProgram(elu_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out->gpu_buffer);
    glUniform1ui(glGetUniformLocation(elu_program, "size"), out->size);
    glUniform1f(glGetUniformLocation(elu_program, "alpha"), alpha);
    glDispatchCompute((out->size + 1023) / 1024, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ===================================
// Swish / SiLU Kernel: x * sigmoid(x)
// ===================================

static const char* SWISH_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            float x = in_data[idx];\n"
    "            float sig = 1.0 / (1.0 + exp(-x));\n"
    "            out_data[idx] = x * sig;\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint swish_program = 0;

void tensor_swish_gpu(Tensor* out, const Tensor* in) {
    dispatch_unary_op(out, in, &swish_program, SWISH_SHADER_SRC);
}

// ===================================
// SELU Kernel
// ===================================
// lambda=1.0507009873554804934, alpha=1.6732632423543772848

static const char* SELU_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "void main() {\n"
    "    const float lam = 1.0507009873554804934;\n"
    "    const float alp = 1.6732632423543772848;\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            float x = in_data[idx];\n"
    "            out_data[idx] = (x >= 0.0) ? lam * x : lam * alp * (exp(x) - 1.0);\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint selu_program = 0;

void tensor_selu_gpu(Tensor* out, const Tensor* in) {
    dispatch_unary_op(out, in, &selu_program, SELU_SHADER_SRC);
}

// ===================================
// Mish Kernel: x * tanh(ln(1+exp(x)))
// ===================================

static const char* MISH_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            float x = in_data[idx];\n"
    "            float sp = log(1.0 + exp(x));  // softplus\n"
    "            out_data[idx] = x * tanh(sp);\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint mish_program = 0;

void tensor_mish_gpu(Tensor* out, const Tensor* in) {
    dispatch_unary_op(out, in, &mish_program, MISH_SHADER_SRC);
}

// ===================================
// Hardswish Kernel
// ===================================

static const char* HARDSWISH_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            float x = in_data[idx];\n"
    "            float clip = clamp(x + 3.0, 0.0, 6.0);\n"
    "            out_data[idx] = x * clip / 6.0;\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint hardswish_program = 0;

void tensor_hardswish_gpu(Tensor* out, const Tensor* in) {
    dispatch_unary_op(out, in, &hardswish_program, HARDSWISH_SHADER_SRC);
}

// ===================================
// Hardsigmoid Kernel
// ===================================

static const char* HARDSIGMOID_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            float x = in_data[idx];\n"
    "            out_data[idx] = clamp(x / 6.0 + 0.5, 0.0, 1.0);\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint hardsigmoid_program = 0;

void tensor_hardsigmoid_gpu(Tensor* out, const Tensor* in) {
    dispatch_unary_op(out, in, &hardsigmoid_program, HARDSIGMOID_SHADER_SRC);
}

// ===================================
// Softplus Kernel
// ===================================

static const char* SOFTPLUS_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "uniform float beta;\n"
    "uniform float threshold;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            float x = in_data[idx];\n"
    "            float bx = beta * x;\n"
    "            out_data[idx] = (bx > threshold) ? x : log(1.0 + exp(bx)) / beta;\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint softplus_program = 0;

void tensor_softplus_gpu(Tensor* out, const Tensor* in, float beta, float threshold) {
    if (!rpl_gpu_init()) return;
    tensor_to_gpu((Tensor*)in);
    if (out->device != DEVICE_GPU) {
        out->dims = in->dims;
        memcpy(out->shape, in->shape, sizeof(in->shape));
        out->size = in->size;
        tensor_to_gpu(out);
    }
    if (softplus_program == 0) {
        softplus_program = compile_compute_shader(SOFTPLUS_SHADER_SRC);
        if (softplus_program == 0) return;
    }
    glUseProgram(softplus_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out->gpu_buffer);
    glUniform1ui(glGetUniformLocation(softplus_program, "size"), out->size);
    glUniform1f(glGetUniformLocation(softplus_program, "beta"), beta);
    glUniform1f(glGetUniformLocation(softplus_program, "threshold"), threshold);
    glDispatchCompute((out->size + 1023) / 1024, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ===================================
// Log-Softmax Kernel (row-wise)
// ===================================

static const char* LOG_SOFTMAX_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 32) in;\n"           /* 32 rows per workgroup */
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint num_rows;\n"
    "uniform uint row_size;\n"
    "void main() {\n"
    "    uint row = gl_GlobalInvocationID.x;\n"
    "    if (row >= num_rows) return;\n"
    "    uint base = row * row_size;\n"
    "    /* find max for numerical stability */\n"
    "    float max_val = in_data[base];\n"
    "    for (uint i = 1u; i < row_size; i++)\n"
    "        if (in_data[base + i] > max_val) max_val = in_data[base + i];\n"
    "    /* log-sum-exp */\n"
    "    float sum = 0.0;\n"
    "    for (uint i = 0u; i < row_size; i++)\n"
    "        sum += exp(in_data[base + i] - max_val);\n"
    "    float log_sum = max_val + log(sum);\n"
    "    for (uint i = 0u; i < row_size; i++)\n"
    "        out_data[base + i] = in_data[base + i] - log_sum;\n"
    "}\n";

// Softmax (output = exp(x)/sum(exp(x))) — same structure
static const char* SOFTMAX_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 32) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint num_rows;\n"
    "uniform uint row_size;\n"
    "void main() {\n"
    "    uint row = gl_GlobalInvocationID.x;\n"
    "    if (row >= num_rows) return;\n"
    "    uint base = row * row_size;\n"
    "    float max_val = in_data[base];\n"
    "    for (uint i = 1u; i < row_size; i++)\n"
    "        if (in_data[base + i] > max_val) max_val = in_data[base + i];\n"
    "    float sum = 0.0;\n"
    "    for (uint i = 0u; i < row_size; i++)\n"
    "        sum += exp(in_data[base + i] - max_val);\n"
    "    for (uint i = 0u; i < row_size; i++)\n"
    "        out_data[base + i] = exp(in_data[base + i] - max_val) / sum;\n"
    "}\n";

static GLuint softmax_program = 0;
static GLuint log_softmax_program = 0;

/* axis ignored for now — always normalises over last dim, treating 2-D [N,C] as N rows of C */
void tensor_softmax_gpu(Tensor* out, const Tensor* in, uint32_t axis) {
    (void)axis;
    if (!rpl_gpu_init()) return;
    tensor_to_gpu((Tensor*)in);
    if (out->device != DEVICE_GPU) {
        out->dims = in->dims;
        memcpy(out->shape, in->shape, sizeof(in->shape));
        out->size = in->size;
        tensor_to_gpu(out);
    }
    if (softmax_program == 0) {
        softmax_program = compile_compute_shader(SOFTMAX_SHADER_SRC);
        if (softmax_program == 0) return;
    }
    uint32_t row_size = in->shape[in->dims - 1];
    uint32_t num_rows = in->size / row_size;
    glUseProgram(softmax_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out->gpu_buffer);
    glUniform1ui(glGetUniformLocation(softmax_program, "num_rows"), num_rows);
    glUniform1ui(glGetUniformLocation(softmax_program, "row_size"), row_size);
    glDispatchCompute((num_rows + 31) / 32, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void tensor_log_softmax_gpu(Tensor* out, const Tensor* in) {
    if (!rpl_gpu_init()) return;
    tensor_to_gpu((Tensor*)in);
    if (out->device != DEVICE_GPU) {
        out->dims = in->dims;
        memcpy(out->shape, in->shape, sizeof(in->shape));
        out->size = in->size;
        tensor_to_gpu(out);
    }
    if (log_softmax_program == 0) {
        log_softmax_program = compile_compute_shader(LOG_SOFTMAX_SHADER_SRC);
        if (log_softmax_program == 0) return;
    }
    uint32_t row_size = in->shape[in->dims - 1];
    uint32_t num_rows = in->size / row_size;
    glUseProgram(log_softmax_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out->gpu_buffer);
    glUniform1ui(glGetUniformLocation(log_softmax_program, "num_rows"), num_rows);
    glUniform1ui(glGetUniformLocation(log_softmax_program, "row_size"), row_size);
    glDispatchCompute((num_rows + 31) / 32, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ===================================
// In-place ReLU  (single SSBO binding)
// ===================================

static const char* RELU_INPLACE_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 128) in;\n"
    "layout(std430, binding = 0) buffer Data { float v[]; };\n"      /* read-write */
    "uniform uint size;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    if (id4 + 3u < size) {\n"
    "        vec4 val = vec4(v[id4], v[id4+1u], v[id4+2u], v[id4+3u]);\n"
    "        vec4 res = max(vec4(0.0), val);\n"
    "        v[id4] = res.x; v[id4+1u] = res.y; v[id4+2u] = res.z; v[id4+3u] = res.w;\n"
    "    } else {\n"
    "        for (uint i = 0u; i < 4u; i++) {\n"
    "            uint idx = id4 + i;\n"
    "            if (idx < size) v[idx] = max(0.0, v[idx]);\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint relu_inplace_program = 0;

void tensor_relu_inplace_gpu(Tensor* t) {
    if (!rpl_gpu_init()) return;
    tensor_to_gpu(t);
    if (relu_inplace_program == 0) {
        relu_inplace_program = compile_compute_shader(RELU_INPLACE_SHADER_SRC);
        if (relu_inplace_program == 0) return;
    }
    glUseProgram(relu_inplace_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, t->gpu_buffer);
    glUniform1ui(glGetUniformLocation(relu_inplace_program, "size"), t->size);
    glDispatchCompute((t->size + 511) / 512, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ===================================
// Scalar-multiply inplace kernel
// ===================================

static const char* SCALE_INPLACE_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) buffer Data { float v[]; };\n"
    "uniform uint size;\n"
    "uniform float scalar;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    if (id4 + 3u < size) {\n"
    "        vec4 val = vec4(v[id4], v[id4+1u], v[id4+2u], v[id4+3u]) * scalar;\n"
    "        v[id4] = val.x; v[id4+1u] = val.y; v[id4+2u] = val.z; v[id4+3u] = val.w;\n"
    "    } else {\n"
    "        for (uint i = 0u; i < 4u; i++) {\n"
    "            uint idx = id4 + i;\n"
    "            if (idx < size) v[idx] *= scalar;\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint scale_inplace_program = 0;

void tensor_scale_gpu(Tensor* t, float scalar) {
    if (!rpl_gpu_init()) return;
    tensor_to_gpu(t);
    if (scale_inplace_program == 0) {
        scale_inplace_program = compile_compute_shader(SCALE_INPLACE_SHADER_SRC);
        if (scale_inplace_program == 0) return;
    }
    glUseProgram(scale_inplace_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, t->gpu_buffer);
    glUniform1ui(glGetUniformLocation(scale_inplace_program, "size"), t->size);
    glUniform1f(glGetUniformLocation(scale_inplace_program, "scalar"), scalar);
    glDispatchCompute((t->size + 1023) / 1024, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ===================================
// Conv2D via GL_TEXTURE_2D  (NCHW, 1 image)
// ===================================
//
// Upload the input channel as a GL_TEXTURE_2D (GL_R32F).
// The compute shader samples it with texelFetch() which uses integer
// coordinates — no filtering hardware needed, just the texture cache.
//
// Shader receives:
//   binding 0 (texture unit 0): usampler2D-style GL_TEXTURE_2D for ONE input channel
//   binding 1 (SSBO):            kernel weights [C_out, C_in, kH, kW]
//   binding 2 (SSBO):            output buffer  [C_out, out_H, out_W]
//
// Each compute invocation computes ONE output element (oc, out_y, out_x).
// Dispatch: (C_out * out_H * out_W + 63) / 64 work-groups.
// The shader decomposes the linear invocation index back to (oc, oy, ox).

static const char* CONV2D_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 16, local_size_y = 16) in;\n"
    "layout(std430, binding = 0) readonly buffer InputBuf { float in_data[]; };\n"
    "layout(std430, binding = 1) readonly buffer KernBuf { float kern_data[]; };\n"
    "layout(std430, binding = 2) writeonly buffer OutputBuf { float out_data[]; };\n"
    "uniform int batch;\n"
    "uniform int C_in;\n"
    "uniform int C_out;\n"
    "uniform int in_H;\n"
    "uniform int in_W;\n"
    "uniform int out_H;\n"
    "uniform int out_W;\n"
    "uniform int kH;\n"
    "uniform int kW;\n"
    "uniform int stride;\n"
    "uniform int padding;\n"
    "void main() {\n"
    "    uint ox0 = gl_GlobalInvocationID.x * 2u;\n"
    "    uint oy = gl_GlobalInvocationID.y;\n"
    "    uint oc_b = gl_GlobalInvocationID.z;\n"
    "    if (ox0 >= uint(out_W) || oy >= uint(out_H) || oc_b >= uint(batch * C_out)) return;\n"
    "    int b = int(oc_b) / C_out;\n"
    "    int oc = int(oc_b) % C_out;\n"
    "    float sum0 = 0.0;\n"
    "    float sum1 = 0.0;\n"
    "    bool has1 = (ox0 + 1u < uint(out_W));\n"
    "    int in_slice = C_in * in_H * in_W;\n"
    "    int b_in_offset = b * in_slice;\n"
    "    for (int ic = 0; ic < C_in; ic++) {\n"
    "        int c_offset = b_in_offset + ic * in_H * in_W;\n"
    "        int k_offset = (oc * C_in + ic) * kH * kW;\n"
    "        for (int ky = 0; ky < kH; ky++) {\n"
    "            int iy = int(oy) * stride - padding + ky;\n"
    "            if (iy >= 0 && iy < in_H) {\n"
    "                int row_offset = c_offset + iy * in_W;\n"
    "                int k_row_offset = k_offset + ky * kW;\n"
    "                for (int kx = 0; kx < kW; kx++) {\n"
    "                    float w = kern_data[k_row_offset + kx];\n"
    "                    int ix0 = int(ox0) * stride - padding + kx;\n"
    "                    if (ix0 >= 0 && ix0 < in_W) {\n"
    "                        sum0 += in_data[row_offset + ix0] * w;\n"
    "                    }\n"
    "                    if (has1) {\n"
    "                        int ix1 = int(ox0 + 1u) * stride - padding + kx;\n"
    "                        if (ix1 >= 0 && ix1 < in_W) {\n"
    "                            sum1 += in_data[row_offset + ix1] * w;\n"
    "                        }\n"
    "                    }\n"
    "                }\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    int out_slice = C_out * out_H * out_W;\n"
    "    int base_out = b * out_slice + oc * out_H * out_W + int(oy) * out_W;\n"
    "    out_data[base_out + int(ox0)] = sum0;\n"
    "    if (has1) out_data[base_out + int(ox0 + 1u)] = sum1;\n"
    "}\n";

static GLuint conv2d_program = 0;

void tensor_conv2d_gpu(Tensor* out, const Tensor* in, const Tensor* kern,
                       int kH, int kW, int stride, int padding) {
    if (!rpl_gpu_init()) return;

    int batch, C_in, H, W;
    if (in->dims == 4) {
        batch = (int)in->shape[0];
        C_in  = (int)in->shape[1];
        H     = (int)in->shape[2];
        W     = (int)in->shape[3];
    } else {
        batch = 1;
        C_in  = (int)in->shape[0];
        H     = (int)in->shape[1];
        W     = (int)in->shape[2];
    }
    int C_out = (int)kern->shape[0];
    int out_H = (H + 2*padding - kH) / stride + 1;
    int out_W = (W + 2*padding - kW) / stride + 1;

    tensor_to_gpu((Tensor*)in);
    tensor_to_gpu((Tensor*)kern);

    if (out->device != DEVICE_GPU) {
        out->dims    = (batch > 1) ? 4 : 3;
        if (batch > 1) {
            out->shape[0] = batch;
            out->shape[1] = C_out;
            out->shape[2] = out_H;
            out->shape[3] = out_W;
        } else {
            out->shape[0] = C_out;
            out->shape[1] = out_H;
            out->shape[2] = out_W;
        }
        out->size = (uint32_t)(batch * C_out * out_H * out_W);
        tensor_to_gpu(out);
    }

    if (conv2d_program == 0) {
        conv2d_program = compile_compute_shader(CONV2D_SHADER_SRC);
        if (conv2d_program == 0) return;
    }

    glUseProgram(conv2d_program);
    glUniform1i(glGetUniformLocation(conv2d_program, "batch"),    batch);
    glUniform1i(glGetUniformLocation(conv2d_program, "C_in"),     C_in);
    glUniform1i(glGetUniformLocation(conv2d_program, "C_out"),    C_out);
    glUniform1i(glGetUniformLocation(conv2d_program, "in_H"),     H);
    glUniform1i(glGetUniformLocation(conv2d_program, "in_W"),     W);
    glUniform1i(glGetUniformLocation(conv2d_program, "out_H"),    out_H);
    glUniform1i(glGetUniformLocation(conv2d_program, "out_W"),    out_W);
    glUniform1i(glGetUniformLocation(conv2d_program, "kH"),       kH);
    glUniform1i(glGetUniformLocation(conv2d_program, "kW"),       kW);
    glUniform1i(glGetUniformLocation(conv2d_program, "stride"),   stride);
    glUniform1i(glGetUniformLocation(conv2d_program, "padding"),  padding);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, kern->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, out->gpu_buffer);

    GLuint gx = (GLuint)((out_W + 31) / 32);
    GLuint gy = (GLuint)((out_H + 15) / 16);
    GLuint gz = (GLuint)(batch * C_out);

    glDispatchCompute(gx, gy, gz);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ===================================
// Unified Math Unary Shader
// op enum (uniform int op):
//  0=sin  1=cos  2=tan  3=asin  4=acos  5=atan
//  6=sinh 7=cosh 8=asinh 9=acosh 10=atanh
// 11=exp  12=exp2 13=expm1 14=log  15=log2  16=log10  17=log1p
// 18=sqrt 19=rsqrt 20=square 21=cbrt 22=reciprocal
// 23=abs  24=neg  25=sign  26=deg2rad 27=rad2deg
// 28=erf  29=logit 30=round 31=floor 32=ceil 33=trunc 34=frac
// ===================================

static const char* MATH_UNARY_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "uniform int op;\n"
    "#define PI 3.14159265358979323846\n"
    "float eval_op(float x, int op) {\n"
    "    if      (op ==  0) return sin(x);\n"
    "    else if (op ==  1) return cos(x);\n"
    "    else if (op ==  2) return tan(x);\n"
    "    else if (op ==  3) return asin(x);\n"
    "    else if (op ==  4) return acos(x);\n"
    "    else if (op ==  5) return atan(x);\n"
    "    else if (op ==  6) return sinh(x);\n"
    "    else if (op ==  7) return cosh(x);\n"
    "    else if (op ==  8) return asinh(x);\n"
    "    else if (op ==  9) return acosh(x);\n"
    "    else if (op == 10) return atanh(x);\n"
    "    else if (op == 11) return exp(x);\n"
    "    else if (op == 12) return exp2(x);\n"
    "    else if (op == 13) return exp(x) - 1.0;\n"
    "    else if (op == 14) return log(x);\n"
    "    else if (op == 15) return log2(x);\n"
    "    else if (op == 16) return log(x) / log(10.0);\n"
    "    else if (op == 17) return log(1.0 + x);\n"
    "    else if (op == 18) return sqrt(x);\n"
    "    else if (op == 19) return inversesqrt(x);\n"
    "    else if (op == 20) return x * x;\n"
    "    else if (op == 21) return sign(x) * exp(log(abs(x)) / 3.0);\n"
    "    else if (op == 22) return 1.0 / x;\n"
    "    else if (op == 23) return abs(x);\n"
    "    else if (op == 24) return -x;\n"
    "    else if (op == 25) return sign(x);\n"
    "    else if (op == 26) return x * float(PI / 180.0);\n"
    "    else if (op == 27) return x * float(180.0 / PI);\n"
    "    else if (op == 28) {\n"
    "        float t = 1.0 / (1.0 + 0.3275911 * abs(x));\n"
    "        float poly = t*(0.254829592+t*(-0.284496736+t*(1.421413741+t*(-1.453152027+t*1.061405429))));\n"
    "        return sign(x) * (1.0 - poly * exp(-x*x));\n"
    "    }\n"
    "    else if (op == 29) return log(x / (1.0 - x));\n"
    "    else if (op == 30) return floor(x + 0.5);\n"
    "    else if (op == 31) return floor(x);\n"
    "    else if (op == 32) return ceil(x);\n"
    "    else if (op == 33) return trunc(x);\n"
    "    else if (op == 34) return x - trunc(x);\n"
    "    return 0.0;\n"
    "}\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            out_data[idx] = eval_op(in_data[idx], op);\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint math_unary_program = 0;

/* Helper: dispatch math unary shader with op code */
static void dispatch_math_unary(Tensor* out, const Tensor* in, int op) {
    if (!rpl_gpu_init()) return;
    tensor_to_gpu((Tensor*)in);
    if (out->device != DEVICE_GPU) {
        out->dims = in->dims;
        memcpy(out->shape, in->shape, sizeof(in->shape));
        out->size = in->size;
        tensor_to_gpu(out);
    }
    if (math_unary_program == 0) {
        math_unary_program = compile_compute_shader(MATH_UNARY_SHADER_SRC);
        if (math_unary_program == 0) return;
    }
    glUseProgram(math_unary_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out->gpu_buffer);
    glUniform1ui(glGetUniformLocation(math_unary_program, "size"), out->size);
    glUniform1i(glGetUniformLocation(math_unary_program, "op"), op);
    glDispatchCompute((out->size + 1023) / 1024, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void tensor_sin_gpu(Tensor* out, const Tensor* in)       { dispatch_math_unary(out, in,  0); }
void tensor_cos_gpu(Tensor* out, const Tensor* in)       { dispatch_math_unary(out, in,  1); }
void tensor_tan_gpu(Tensor* out, const Tensor* in)       { dispatch_math_unary(out, in,  2); }
void tensor_asin_gpu(Tensor* out, const Tensor* in)      { dispatch_math_unary(out, in,  3); }
void tensor_acos_gpu(Tensor* out, const Tensor* in)      { dispatch_math_unary(out, in,  4); }
void tensor_atan_gpu(Tensor* out, const Tensor* in)      { dispatch_math_unary(out, in,  5); }
void tensor_sinh_gpu(Tensor* out, const Tensor* in)      { dispatch_math_unary(out, in,  6); }
void tensor_cosh_gpu(Tensor* out, const Tensor* in)      { dispatch_math_unary(out, in,  7); }
void tensor_asinh_gpu(Tensor* out, const Tensor* in)     { dispatch_math_unary(out, in,  8); }
void tensor_acosh_gpu(Tensor* out, const Tensor* in)     { dispatch_math_unary(out, in,  9); }
void tensor_atanh_gpu(Tensor* out, const Tensor* in)     { dispatch_math_unary(out, in, 10); }
void tensor_exp_gpu(Tensor* out, const Tensor* in)       { dispatch_math_unary(out, in, 11); }
void tensor_exp2_gpu(Tensor* out, const Tensor* in)      { dispatch_math_unary(out, in, 12); }
void tensor_expm1_gpu(Tensor* out, const Tensor* in)     { dispatch_math_unary(out, in, 13); }
void tensor_log_gpu(Tensor* out, const Tensor* in)       { dispatch_math_unary(out, in, 14); }
void tensor_log2_gpu(Tensor* out, const Tensor* in)      { dispatch_math_unary(out, in, 15); }
void tensor_log10_gpu(Tensor* out, const Tensor* in)     { dispatch_math_unary(out, in, 16); }
void tensor_log1p_gpu(Tensor* out, const Tensor* in)     { dispatch_math_unary(out, in, 17); }
void tensor_sqrt_gpu(Tensor* out, const Tensor* in)      { dispatch_math_unary(out, in, 18); }
void tensor_rsqrt_gpu(Tensor* out, const Tensor* in)     { dispatch_math_unary(out, in, 19); }
void tensor_square_gpu(Tensor* out, const Tensor* in)    { dispatch_math_unary(out, in, 20); }
void tensor_cbrt_gpu(Tensor* out, const Tensor* in)      { dispatch_math_unary(out, in, 21); }
void tensor_reciprocal_gpu(Tensor* out, const Tensor* in){ dispatch_math_unary(out, in, 22); }
void tensor_abs_gpu(Tensor* out, const Tensor* in)       { dispatch_math_unary(out, in, 23); }
void tensor_neg_gpu(Tensor* out, const Tensor* in)       { dispatch_math_unary(out, in, 24); }
void tensor_sign_gpu(Tensor* out, const Tensor* in)      { dispatch_math_unary(out, in, 25); }
void tensor_deg2rad_gpu(Tensor* out, const Tensor* in)   { dispatch_math_unary(out, in, 26); }
void tensor_rad2deg_gpu(Tensor* out, const Tensor* in)   { dispatch_math_unary(out, in, 27); }
void tensor_erf_gpu(Tensor* out, const Tensor* in)       { dispatch_math_unary(out, in, 28); }
void tensor_logit_gpu(Tensor* out, const Tensor* in)     { dispatch_math_unary(out, in, 29); }
void tensor_round_gpu(Tensor* out, const Tensor* in)     { dispatch_math_unary(out, in, 30); }
void tensor_floor_gpu(Tensor* out, const Tensor* in)     { dispatch_math_unary(out, in, 31); }
void tensor_ceil_gpu(Tensor* out, const Tensor* in)      { dispatch_math_unary(out, in, 32); }
void tensor_trunc_gpu(Tensor* out, const Tensor* in)     { dispatch_math_unary(out, in, 33); }
void tensor_frac_gpu(Tensor* out, const Tensor* in)      { dispatch_math_unary(out, in, 34); }

// ===================================
// Unified Math Binary Shader
// op: 0=pow 1=atan2 2=hypot 3=fmod 4=remainder
//     5=floor_divide 6=maximum 7=minimum 8=logaddexp 9=logaddexp2
// ===================================

static const char* MATH_BINARY_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer InputA { float a[]; };\n"
    "layout(std430, binding = 1) readonly buffer InputB { float b[]; };\n"
    "layout(std430, binding = 2) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size_a;\n"
    "uniform uint size_b;\n"
    "uniform int op;\n"
    "float eval_bin_op(float x, float y, int op) {\n"
    "    if      (op == 0) return pow(abs(x), y) * sign(x);\n"
    "    else if (op == 1) return atan(x, y);\n"
    "    else if (op == 2) return sqrt(x*x + y*y);\n"
    "    else if (op == 3) return x - trunc(x/y)*y;\n"
    "    else if (op == 4) { float q = x/y; float n = (q >= 0.0) ? floor(q+0.5) : ceil(q-0.5); return x - n*y; }\n"
    "    else if (op == 5) return floor(x/y);\n"
    "    else if (op == 6) return max(x, y);\n"
    "    else if (op == 7) return min(x, y);\n"
    "    else if (op == 8) { float mx = max(x,y); return mx + log(exp(x-mx)+exp(y-mx)); }\n"
    "    else if (op == 9) { float mx = max(x,y); return mx + log2(exp2(x-mx)+exp2(y-mx)); }\n"
    "    return 0.0;\n"
    "}\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size_a) {\n"
    "            out_data[idx] = eval_bin_op(a[idx], b[idx % size_b], op);\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint math_binary_program = 0;

static void dispatch_math_binary(Tensor* out, const Tensor* a, const Tensor* b, int op) {
    if (!rpl_gpu_init()) return;
    tensor_to_gpu((Tensor*)a);
    tensor_to_gpu((Tensor*)b);
    if (out->device != DEVICE_GPU) {
        out->dims = a->dims;
        memcpy(out->shape, a->shape, sizeof(a->shape));
        out->size = a->size;
        tensor_to_gpu(out);
    }
    if (math_binary_program == 0) {
        math_binary_program = compile_compute_shader(MATH_BINARY_SHADER_SRC);
        if (math_binary_program == 0) return;
    }
    glUseProgram(math_binary_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, a->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, out->gpu_buffer);
    glUniform1ui(glGetUniformLocation(math_binary_program, "size_a"), a->size);
    glUniform1ui(glGetUniformLocation(math_binary_program, "size_b"), b->size);
    glUniform1i(glGetUniformLocation(math_binary_program, "op"), op);
    glDispatchCompute((a->size + 1023) / 1024, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void tensor_pow_gpu(Tensor* out, const Tensor* a, const Tensor* b)          { dispatch_math_binary(out, a, b, 0); }
void tensor_atan2_gpu(Tensor* out, const Tensor* a, const Tensor* b)        { dispatch_math_binary(out, a, b, 1); }
void tensor_hypot_gpu(Tensor* out, const Tensor* a, const Tensor* b)        { dispatch_math_binary(out, a, b, 2); }
void tensor_fmod_gpu(Tensor* out, const Tensor* a, const Tensor* b)         { dispatch_math_binary(out, a, b, 3); }
void tensor_remainder_gpu(Tensor* out, const Tensor* a, const Tensor* b)    { dispatch_math_binary(out, a, b, 4); }
void tensor_floor_divide_gpu(Tensor* out, const Tensor* a, const Tensor* b) { dispatch_math_binary(out, a, b, 5); }
void tensor_maximum_gpu(Tensor* out, const Tensor* a, const Tensor* b)      { dispatch_math_binary(out, a, b, 6); }
void tensor_minimum_gpu(Tensor* out, const Tensor* a, const Tensor* b)      { dispatch_math_binary(out, a, b, 7); }
void tensor_logaddexp_gpu(Tensor* out, const Tensor* a, const Tensor* b)    { dispatch_math_binary(out, a, b, 8); }
void tensor_logaddexp2_gpu(Tensor* out, const Tensor* a, const Tensor* b)   { dispatch_math_binary(out, a, b, 9); }

// ===================================
// Clamp Shader
// ===================================

static const char* CLAMP_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "uniform float lo;\n"
    "uniform float hi;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) out_data[idx] = clamp(in_data[idx], lo, hi);\n"
    "    }\n"
    "}\n";

static GLuint clamp_program = 0;

void tensor_clamp_gpu(Tensor* out, const Tensor* in, float lo, float hi) {
    if (!rpl_gpu_init()) return;
    tensor_to_gpu((Tensor*)in);
    if (out->device != DEVICE_GPU) {
        out->dims = in->dims;
        memcpy(out->shape, in->shape, sizeof(in->shape));
        out->size = in->size;
        tensor_to_gpu(out);
    }
    if (clamp_program == 0) {
        clamp_program = compile_compute_shader(CLAMP_SHADER_SRC);
        if (clamp_program == 0) return;
    }
    glUseProgram(clamp_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out->gpu_buffer);
    glUniform1ui(glGetUniformLocation(clamp_program, "size"), out->size);
    glUniform1f(glGetUniformLocation(clamp_program, "lo"), lo);
    glUniform1f(glGetUniformLocation(clamp_program, "hi"), hi);
    glDispatchCompute((out->size + 1023) / 1024, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// Hardtanh reuses clamp shader
void tensor_hardtanh_gpu(Tensor* out, const Tensor* in, float min_val, float max_val) {
    tensor_clamp_gpu(out, in, min_val, max_val);
}

// ===================================
// CELU Shader: max(0,x) + min(0, alpha*(exp(x/alpha)-1))
// ===================================

static const char* CELU_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "uniform float alpha;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            float x = in_data[idx];\n"
    "            float pos = max(0.0, x);\n"
    "            float neg = min(0.0, alpha * (exp(x / alpha) - 1.0));\n"
    "            out_data[idx] = pos + neg;\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint celu_program = 0;

void tensor_celu_gpu(Tensor* out, const Tensor* in, float alpha) {
    if (!rpl_gpu_init()) return;
    tensor_to_gpu((Tensor*)in);
    if (out->device != DEVICE_GPU) {
        out->dims = in->dims;
        memcpy(out->shape, in->shape, sizeof(in->shape));
        out->size = in->size;
        tensor_to_gpu(out);
    }
    if (celu_program == 0) {
        celu_program = compile_compute_shader(CELU_SHADER_SRC);
        if (celu_program == 0) return;
    }
    glUseProgram(celu_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out->gpu_buffer);
    glUniform1ui(glGetUniformLocation(celu_program, "size"), out->size);
    glUniform1f(glGetUniformLocation(celu_program, "alpha"), alpha);
    glDispatchCompute((out->size + 1023) / 1024, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ===================================
// Softsign Shader: x / (1 + |x|)
// ===================================

static const char* SOFTSIGN_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            float x = in_data[idx];\n"
    "            out_data[idx] = x / (1.0 + abs(x));\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint softsign_program = 0;

void tensor_softsign_gpu(Tensor* out, const Tensor* in) {
    dispatch_unary_op(out, in, &softsign_program, SOFTSIGN_SHADER_SRC);
}

// ===================================
// RReLU Shader: x if x>=0, else slope*x
// (eval mode: slope = (lower+upper)/2)
// ===================================

static const char* RRELU_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "uniform float slope;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            float x = in_data[idx];\n"
    "            out_data[idx] = (x >= 0.0) ? x : slope * x;\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint rrelu_program = 0;

void tensor_rrelu_gpu(Tensor* out, const Tensor* in, float lower, float upper) {
    float slope = (lower + upper) * 0.5f;
    if (!rpl_gpu_init()) return;
    tensor_to_gpu((Tensor*)in);
    if (out->device != DEVICE_GPU) {
        out->dims = in->dims;
        memcpy(out->shape, in->shape, sizeof(in->shape));
        out->size = in->size;
        tensor_to_gpu(out);
    }
    if (rrelu_program == 0) {
        rrelu_program = compile_compute_shader(RRELU_SHADER_SRC);
        if (rrelu_program == 0) return;
    }
    glUseProgram(rrelu_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out->gpu_buffer);
    glUniform1ui(glGetUniformLocation(rrelu_program, "size"), out->size);
    glUniform1f(glGetUniformLocation(rrelu_program, "slope"), slope);
    glDispatchCompute((out->size + 1023) / 1024, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ===================================
// Threshold Shader: x if x > threshold, else value
// ===================================

static const char* THRESHOLD_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "uniform float threshold;\n"
    "uniform float value;\n"
    "void main() {\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint idx = id4 + k;\n"
    "        if (idx < size) {\n"
    "            float x = in_data[idx];\n"
    "            out_data[idx] = (x > threshold) ? x : value;\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint threshold_program = 0;

void tensor_threshold_gpu(Tensor* out, const Tensor* in, float threshold, float value) {
    if (!rpl_gpu_init()) return;
    tensor_to_gpu((Tensor*)in);
    if (out->device != DEVICE_GPU) {
        out->dims = in->dims;
        memcpy(out->shape, in->shape, sizeof(in->shape));
        out->size = in->size;
        tensor_to_gpu(out);
    }
    if (threshold_program == 0) {
        threshold_program = compile_compute_shader(THRESHOLD_SHADER_SRC);
        if (threshold_program == 0) return;
    }
    glUseProgram(threshold_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out->gpu_buffer);
    glUniform1ui(glGetUniformLocation(threshold_program, "size"), out->size);
    glUniform1f(glGetUniformLocation(threshold_program, "threshold"), threshold);
    glUniform1f(glGetUniformLocation(threshold_program, "value"), value);
    glDispatchCompute((out->size + 1023) / 1024, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ===================================
// RMSNorm GPU Kernel
// ===================================

static const char* RMSNORM_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) readonly buffer Weight { float weight[]; };\n"
    "layout(std430, binding = 2) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint num_rows;\n"
    "uniform uint row_size;\n"
    "uniform float eps;\n"
    "shared float s_sum[256];\n"
    "void main() {\n"
    "    uint row = gl_GlobalInvocationID.y;\n"
    "    if (row >= num_rows) return;\n"
    "    uint base = row * row_size;\n"
    "    uint tid = gl_LocalInvocationID.x;\n"
    "    float local_sum = 0.0;\n"
    "    for (uint i = tid; i < row_size; i += 256u) {\n"
    "        float val = in_data[base + i];\n"
    "        local_sum += val * val;\n"
    "    }\n"
    "    s_sum[tid] = local_sum;\n"
    "    barrier();\n"
    "    for (uint s = 128u; s > 0u; s >>= 1u) {\n"
    "        if (tid < s) {\n"
    "            s_sum[tid] += s_sum[tid + s];\n"
    "        }\n"
    "        barrier();\n"
    "    }\n"
    "    float ms = s_sum[0] / float(row_size);\n"
    "    float rms_inv = 1.0 / sqrt(ms + eps);\n"
    "    for (uint i = tid; i < row_size; i += 256u) {\n"
    "        out_data[base + i] = in_data[base + i] * rms_inv * weight[i];\n"
    "    }\n"
    "}\n";

static GLuint rmsnorm_program = 0;

void tensor_rmsnorm_gpu(Tensor* out, const Tensor* in, const Tensor* weight, float eps) {
    if (!rpl_gpu_init()) return;
    tensor_to_gpu((Tensor*)in);
    tensor_to_gpu((Tensor*)weight);
    if (out->device != DEVICE_GPU) {
        out->dims = in->dims;
        memcpy(out->shape, in->shape, sizeof(in->shape));
        out->size = in->size;
        tensor_to_gpu(out);
    }
    if (rmsnorm_program == 0) {
        rmsnorm_program = compile_compute_shader(RMSNORM_SHADER_SRC);
        if (rmsnorm_program == 0) return;
    }
    uint32_t row_size = in->shape[in->dims - 1];
    uint32_t num_rows = in->size / row_size;
    glUseProgram(rmsnorm_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, weight->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, out->gpu_buffer);
    glUniform1ui(glGetUniformLocation(rmsnorm_program, "num_rows"), num_rows);
    glUniform1ui(glGetUniformLocation(rmsnorm_program, "row_size"), row_size);
    glUniform1f(glGetUniformLocation(rmsnorm_program, "eps"), eps);
    glDispatchCompute(1, num_rows, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ===================================
// RoPE GPU Kernel
// ===================================

static const char* ROPE_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 128) in;\n"
    "layout(std430, binding = 0) buffer QData { float q_data[]; };\n"
    "layout(std430, binding = 1) buffer KData { float k_data[]; };\n"
    "uniform uint seq_len;\n"
    "uniform uint num_heads_q;\n"
    "uniform uint num_heads_k;\n"
    "uniform uint dim_head;\n"
    "uniform float theta;\n"
    "void main() {\n"
    "    uint d = gl_GlobalInvocationID.x; // process half dimension (dim_head / 2)\n"
    "    uint h_q = gl_GlobalInvocationID.y; // head id q\n"
    "    uint s = gl_GlobalInvocationID.z; // seq index b * seq_len + seq_pos\n"
    "    if (d >= dim_head / 2u || h_q >= num_heads_q) return;\n"
    "\n"
    "    uint seq_pos = s % seq_len;\n"
    "    float freq = 1.0 / pow(theta, float(2u * d) / float(dim_head));\n"
    "    float val = float(seq_pos) * freq;\n"
    "    float cos_val = cos(val);\n"
    "    float sin_val = sin(val);\n"
    "\n"
    "    // Q transform\n"
    "    uint q_idx = (s * num_heads_q + h_q) * dim_head + d;\n"
    "    float q0 = q_data[q_idx];\n"
    "    float q1 = q_data[q_idx + dim_head / 2u];\n"
    "    q_data[q_idx] = q0 * cos_val - q1 * sin_val;\n"
    "    q_data[q_idx + dim_head / 2u] = q0 * sin_val + q1 * cos_val;\n"
    "\n"
    "    // K transform\n"
    "    if (h_q < num_heads_k) {\n"
    "        uint k_idx = (s * num_heads_k + h_q) * dim_head + d;\n"
    "        float k0 = k_data[k_idx];\n"
    "        float k1 = k_data[k_idx + dim_head / 2u];\n"
    "        k_data[k_idx] = k0 * cos_val - k1 * sin_val;\n"
    "        k_data[k_idx + dim_head / 2u] = k0 * sin_val + k1 * cos_val;\n"
    "    }\n"
    "}\n";

static GLuint rope_program = 0;

void tensor_rope_gpu(Tensor* q, Tensor* k, uint32_t dim_head, float theta) {
    if (!rpl_gpu_init()) return;
    tensor_to_gpu(q);
    tensor_to_gpu(k);
    if (rope_program == 0) {
        rope_program = compile_compute_shader(ROPE_SHADER_SRC);
        if (rope_program == 0) return;
    }
    uint32_t seq_len = q->shape[1];
    uint32_t num_heads_q = q->shape[2];
    uint32_t num_heads_k = k->shape[2];
    uint32_t batch_size = q->shape[0];
    if (q->dims < 3) {
        seq_len = q->shape[0];
        num_heads_q = q->shape[1] / dim_head;
        batch_size = 1;
    }
    glUseProgram(rope_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, q->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, k->gpu_buffer);
    glUniform1ui(glGetUniformLocation(rope_program, "seq_len"), seq_len);
    glUniform1ui(glGetUniformLocation(rope_program, "num_heads_q"), num_heads_q);
    glUniform1ui(glGetUniformLocation(rope_program, "num_heads_k"), num_heads_k);
    glUniform1ui(glGetUniformLocation(rope_program, "dim_head"), dim_head);
    glUniform1f(glGetUniformLocation(rope_program, "theta"), theta);
    
    GLuint groups_x = ((dim_head / 2) + 127) / 128;
    glDispatchCompute(groups_x, num_heads_q, batch_size * seq_len);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ===================================
// Gated Attention GPU Kernel
// ===================================

static const char* GATED_ATTN_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 32) in;\n"
    "layout(std430, binding = 0) readonly buffer BufQ { float Q[]; };\n"
    "layout(std430, binding = 1) readonly buffer BufK { float K[]; };\n"
    "layout(std430, binding = 2) readonly buffer BufV { float V[]; };\n"
    "layout(std430, binding = 3) readonly buffer BufGate { float Gate[]; };\n"
    "layout(std430, binding = 4) writeonly buffer BufOut { float Out[]; };\n"
    "uniform uint seq_len_q;\n"
    "uniform uint seq_len_k;\n"
    "uniform uint num_heads_q;\n"
    "uniform uint num_heads_kv;\n"
    "uniform uint d_k;\n"
    "uniform float scale;\n"
    "uniform int has_gate;\n"
    "void main() {\n"
    "    uint i = gl_GlobalInvocationID.y; // seq_q index\n"
    "    uint h_q = gl_GlobalInvocationID.z; // head q index\n"
    "    if (i >= seq_len_q || h_q >= num_heads_q) return;\n"
    "    uint b = gl_GlobalInvocationID.x; // batch block\n" // simplification, only 1 batch supported locally
    "    uint group_size = num_heads_q / num_heads_kv;\n"
    "    uint h_kv = h_q / group_size;\n"
    "    uint q_base = ((b * seq_len_q + i) * num_heads_q + h_q) * d_k;\n"
    "    // Compute scores locally (up to max seq len ~ 4096 in this naive kernel, dynamically allocated in global for large, but we use a small shared or direct loop here)\n"
    "    float max_val = -1e9;\n"
    "    // For simplicity, we write a naive O(N^2) serial scan per thread. For true GPU, requires shared mem tile matmul.\n"
    "    // This is a placeholder for small sequences.\n"
    "}\n";

static GLuint gated_attn_program = 0;

void tensor_gated_attention_gpu(Tensor* out, const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* gate, const Tensor* mask) {
    // Stub implementation. For performance, complex attention needs tiled GEMMs.
    // Falling back to CPU if called.
    (void)out; (void)Q; (void)K; (void)V; (void)gate; (void)mask;
    fprintf(stderr, "tensor_gated_attention_gpu not fully implemented, falling back.\n");
}

// ===================================
// Gated DeltaNet GPU Kernel (Naive)
// ===================================

static const char* GATED_DELTANET_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 1) in;\n"
    "void main() {}\n";

static GLuint gated_deltanet_program = 0;

void tensor_gated_deltanet_gpu(Tensor* out, const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* gate, const Tensor* beta) {
    (void)out; (void)Q; (void)K; (void)V; (void)gate; (void)beta;
    fprintf(stderr, "tensor_gated_deltanet_gpu not fully implemented, falling back.\n");
}

// ===================================
// SwiGLU GPU Kernel (Swish(x[:d/2]) * x[d/2:])
// ===================================

static const char* SWIGLU_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint pre_dim;\n"
    "uniform uint half_dim;\n"
    "uniform uint stride;\n"
    "void main() {\n"
    "    uint total_out = pre_dim * half_dim * stride;\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint id = id4 + k;\n"
    "        if (id < total_out) {\n"
    "            uint p = id / (half_dim * stride);\n"
    "            uint rem = id % (half_dim * stride);\n"
    "            uint h = rem / stride;\n"
    "            uint s = rem % stride;\n"
    "            uint in_idx1 = p * (half_dim * 2u) * stride + h * stride + s;\n"
    "            uint in_idx2 = p * (half_dim * 2u) * stride + (h + half_dim) * stride + s;\n"
    "            float x = in_data[in_idx1];\n"
    "            float gate = in_data[in_idx2];\n"
    "            float sigmoid = 1.0 / (1.0 + exp(-x));\n"
    "            out_data[id] = x * sigmoid * gate;\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint swiglu_program = 0;

void tensor_swiglu_gpu(Tensor* out, const Tensor* in, int32_t dim) {
    if (!rpl_gpu_init()) return;
    tensor_to_gpu((Tensor*)in);
    
    if (dim < 0) dim += in->dims;
    uint32_t stride = 1;
    for (uint32_t i = dim + 1; i < in->dims; i++) stride *= in->shape[i];
    uint32_t half_dim = in->shape[dim] / 2;
    uint32_t pre_dim = 1;
    for (uint32_t i = 0; i < dim; i++) pre_dim *= in->shape[i];
    
    if (out->device != DEVICE_GPU) {
        out->dims = in->dims;
        memcpy(out->shape, in->shape, sizeof(in->shape));
        out->shape[dim] = half_dim;
        out->size = in->size / 2;
        tensor_to_gpu(out);
    }
    if (swiglu_program == 0) {
        swiglu_program = compile_compute_shader(SWIGLU_SHADER_SRC);
        if (swiglu_program == 0) return;
    }
    glUseProgram(swiglu_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out->gpu_buffer);
    glUniform1ui(glGetUniformLocation(swiglu_program, "pre_dim"), pre_dim);
    glUniform1ui(glGetUniformLocation(swiglu_program, "half_dim"), half_dim);
    glUniform1ui(glGetUniformLocation(swiglu_program, "stride"), stride);
    uint32_t out_size = pre_dim * half_dim * stride;
    glDispatchCompute((out_size + 1023) / 1024, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}


// ===================================
// GeGLU GPU Kernel (GELU(x[:d/2]) * x[d/2:])
// ===================================

static const char* GEGLU_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint pre_dim;\n"
    "uniform uint half_dim;\n"
    "uniform uint stride;\n"
    "void main() {\n"
    "    uint total_out = pre_dim * half_dim * stride;\n"
    "    uint id4 = gl_GlobalInvocationID.x << 2u;\n"
    "    for (uint k = 0u; k < 4u; ++k) {\n"
    "        uint id = id4 + k;\n"
    "        if (id < total_out) {\n"
    "            uint p = id / (half_dim * stride);\n"
    "            uint rem = id % (half_dim * stride);\n"
    "            uint h = rem / stride;\n"
    "            uint s = rem % stride;\n"
    "            uint in_idx1 = p * (half_dim * 2u) * stride + h * stride + s;\n"
    "            uint in_idx2 = p * (half_dim * 2u) * stride + (h + half_dim) * stride + s;\n"
    "            float x = in_data[in_idx1];\n"
    "            float gate = in_data[in_idx2];\n"
    "            float inner = 0.7978845608 * (x + 0.044715 * x * x * x);\n"
    "            float gelu = 0.5 * x * (1.0 + tanh(inner));\n"
    "            out_data[id] = gelu * gate;\n"
    "        }\n"
    "    }\n"
    "}\n";

static GLuint geglu_program = 0;

void tensor_geglu_gpu(Tensor* out, const Tensor* in, int32_t dim) {
    if (!rpl_gpu_init()) return;
    tensor_to_gpu((Tensor*)in);
    
    if (dim < 0) dim += in->dims;
    uint32_t stride = 1;
    for (uint32_t i = dim + 1; i < in->dims; i++) stride *= in->shape[i];
    uint32_t half_dim = in->shape[dim] / 2;
    uint32_t pre_dim = 1;
    for (uint32_t i = 0; i < dim; i++) pre_dim *= in->shape[i];
    
    if (out->device != DEVICE_GPU) {
        out->dims = in->dims;
        memcpy(out->shape, in->shape, sizeof(in->shape));
        out->shape[dim] = half_dim;
        out->size = in->size / 2;
        tensor_to_gpu(out);
    }
    if (geglu_program == 0) {
        geglu_program = compile_compute_shader(GEGLU_SHADER_SRC);
        if (geglu_program == 0) return;
    }
    glUseProgram(geglu_program);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, in->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out->gpu_buffer);
    glUniform1ui(glGetUniformLocation(geglu_program, "pre_dim"), pre_dim);
    glUniform1ui(glGetUniformLocation(geglu_program, "half_dim"), half_dim);
    glUniform1ui(glGetUniformLocation(geglu_program, "stride"), stride);
    uint32_t out_size = pre_dim * half_dim * stride;
    glDispatchCompute((out_size + 1023) / 1024, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

#else
/* ================================================================
 * Non-GPU build: empty stubs so the library links without -DUSE_GPU
 * ================================================================ */
bool rpl_gpu_init()  { return false; }
void rpl_gpu_shutdown() {}
void tensor_to_gpu(Tensor* t)   { (void)t; }
void tensor_from_gpu(Tensor* t) { (void)t; }
void tensor_free_gpu(Tensor* t) { (void)t; }
void tensor_sub_gpu(Tensor* out, const Tensor* a, const Tensor* b)  { (void)out;(void)a;(void)b; }
void tensor_add_gpu(Tensor* out, const Tensor* a, const Tensor* b)  { (void)out;(void)a;(void)b; }
void tensor_mul_gpu(Tensor* out, const Tensor* a, const Tensor* b)  { (void)out;(void)a;(void)b; }
void tensor_div_gpu(Tensor* out, const Tensor* a, const Tensor* b)  { (void)out;(void)a;(void)b; }
void tensor_matmul_gpu(Tensor* C, const Tensor* A, const Tensor* B) { (void)C;(void)A;(void)B; }
void tensor_gemm_gpu(Tensor* C, const Tensor* A, const Tensor* B,
                     uint32_t M, uint32_t N, uint32_t K,
                     float alpha, float beta, bool trans_a, bool trans_b) {
    (void)C;(void)A;(void)B;(void)M;(void)N;(void)K;
    (void)alpha;(void)beta;(void)trans_a;(void)trans_b;
}
void tensor_relu_gpu(Tensor* out, const Tensor* in)       { (void)out;(void)in; }
void tensor_relu_inplace_gpu(Tensor* t)                   { (void)t; }
void tensor_sigmoid_gpu(Tensor* out, const Tensor* in)    { (void)out;(void)in; }
void tensor_tanh_gpu(Tensor* out, const Tensor* in)       { (void)out;(void)in; }
void tensor_gelu_gpu(Tensor* out, const Tensor* in)       { (void)out;(void)in; }
void tensor_leaky_relu_gpu(Tensor* out, const Tensor* in, float s) { (void)out;(void)in;(void)s; }
void tensor_swish_gpu(Tensor* out, const Tensor* in)      { (void)out;(void)in; }
void tensor_elu_gpu(Tensor* out, const Tensor* in, float a) { (void)out;(void)in;(void)a; }
void tensor_selu_gpu(Tensor* out, const Tensor* in)       { (void)out;(void)in; }
void tensor_mish_gpu(Tensor* out, const Tensor* in)       { (void)out;(void)in; }
void tensor_hardswish_gpu(Tensor* out, const Tensor* in)  { (void)out;(void)in; }
void tensor_hardsigmoid_gpu(Tensor* out, const Tensor* in){ (void)out;(void)in; }
void tensor_softplus_gpu(Tensor* out, const Tensor* in, float b, float th) { (void)out;(void)in;(void)b;(void)th; }
void tensor_softmax_gpu(Tensor* out, const Tensor* in, uint32_t ax){ (void)out;(void)in;(void)ax; }
void tensor_log_softmax_gpu(Tensor* out, const Tensor* in){ (void)out;(void)in; }
void tensor_scale_gpu(Tensor* t, float s)                 { (void)t;(void)s; }
void tensor_conv2d_gpu(Tensor* out, const Tensor* in, const Tensor* k,
                       int kH, int kW, int st, int pad)
{ (void)out;(void)in;(void)k;(void)kH;(void)kW;(void)st;(void)pad; }

/* Math unary stubs */
void tensor_sin_gpu(Tensor* o, const Tensor* i)        { (void)o;(void)i; }
void tensor_cos_gpu(Tensor* o, const Tensor* i)        { (void)o;(void)i; }
void tensor_tan_gpu(Tensor* o, const Tensor* i)        { (void)o;(void)i; }
void tensor_asin_gpu(Tensor* o, const Tensor* i)       { (void)o;(void)i; }
void tensor_acos_gpu(Tensor* o, const Tensor* i)       { (void)o;(void)i; }
void tensor_atan_gpu(Tensor* o, const Tensor* i)       { (void)o;(void)i; }
void tensor_sinh_gpu(Tensor* o, const Tensor* i)       { (void)o;(void)i; }
void tensor_cosh_gpu(Tensor* o, const Tensor* i)       { (void)o;(void)i; }
void tensor_asinh_gpu(Tensor* o, const Tensor* i)      { (void)o;(void)i; }
void tensor_acosh_gpu(Tensor* o, const Tensor* i)      { (void)o;(void)i; }
void tensor_atanh_gpu(Tensor* o, const Tensor* i)      { (void)o;(void)i; }
void tensor_exp_gpu(Tensor* o, const Tensor* i)        { (void)o;(void)i; }
void tensor_exp2_gpu(Tensor* o, const Tensor* i)       { (void)o;(void)i; }
void tensor_expm1_gpu(Tensor* o, const Tensor* i)      { (void)o;(void)i; }
void tensor_log_gpu(Tensor* o, const Tensor* i)        { (void)o;(void)i; }
void tensor_log2_gpu(Tensor* o, const Tensor* i)       { (void)o;(void)i; }
void tensor_log10_gpu(Tensor* o, const Tensor* i)      { (void)o;(void)i; }
void tensor_log1p_gpu(Tensor* o, const Tensor* i)      { (void)o;(void)i; }
void tensor_sqrt_gpu(Tensor* o, const Tensor* i)       { (void)o;(void)i; }
void tensor_rsqrt_gpu(Tensor* o, const Tensor* i)      { (void)o;(void)i; }
void tensor_square_gpu(Tensor* o, const Tensor* i)     { (void)o;(void)i; }
void tensor_cbrt_gpu(Tensor* o, const Tensor* i)       { (void)o;(void)i; }
void tensor_reciprocal_gpu(Tensor* o, const Tensor* i) { (void)o;(void)i; }
void tensor_abs_gpu(Tensor* o, const Tensor* i)        { (void)o;(void)i; }
void tensor_neg_gpu(Tensor* o, const Tensor* i)        { (void)o;(void)i; }
void tensor_sign_gpu(Tensor* o, const Tensor* i)       { (void)o;(void)i; }
void tensor_deg2rad_gpu(Tensor* o, const Tensor* i)    { (void)o;(void)i; }
void tensor_rad2deg_gpu(Tensor* o, const Tensor* i)    { (void)o;(void)i; }
void tensor_erf_gpu(Tensor* o, const Tensor* i)        { (void)o;(void)i; }
void tensor_logit_gpu(Tensor* o, const Tensor* i)      { (void)o;(void)i; }
void tensor_round_gpu(Tensor* o, const Tensor* i)      { (void)o;(void)i; }
void tensor_floor_gpu(Tensor* o, const Tensor* i)      { (void)o;(void)i; }
void tensor_ceil_gpu(Tensor* o, const Tensor* i)       { (void)o;(void)i; }
void tensor_trunc_gpu(Tensor* o, const Tensor* i)      { (void)o;(void)i; }
void tensor_frac_gpu(Tensor* o, const Tensor* i)       { (void)o;(void)i; }
/* Math binary stubs */
void tensor_pow_gpu(Tensor* o, const Tensor* a, const Tensor* b)          { (void)o;(void)a;(void)b; }
void tensor_atan2_gpu(Tensor* o, const Tensor* a, const Tensor* b)        { (void)o;(void)a;(void)b; }
void tensor_hypot_gpu(Tensor* o, const Tensor* a, const Tensor* b)        { (void)o;(void)a;(void)b; }
void tensor_fmod_gpu(Tensor* o, const Tensor* a, const Tensor* b)         { (void)o;(void)a;(void)b; }
void tensor_remainder_gpu(Tensor* o, const Tensor* a, const Tensor* b)    { (void)o;(void)a;(void)b; }
void tensor_floor_divide_gpu(Tensor* o, const Tensor* a, const Tensor* b) { (void)o;(void)a;(void)b; }
void tensor_maximum_gpu(Tensor* o, const Tensor* a, const Tensor* b)      { (void)o;(void)a;(void)b; }
void tensor_minimum_gpu(Tensor* o, const Tensor* a, const Tensor* b)      { (void)o;(void)a;(void)b; }
void tensor_logaddexp_gpu(Tensor* o, const Tensor* a, const Tensor* b)    { (void)o;(void)a;(void)b; }
void tensor_logaddexp2_gpu(Tensor* o, const Tensor* a, const Tensor* b)   { (void)o;(void)a;(void)b; }
/* Clamp / activation stubs */
void tensor_clamp_gpu(Tensor* o, const Tensor* i, float lo, float hi)     { (void)o;(void)i;(void)lo;(void)hi; }
void tensor_hardtanh_gpu(Tensor* o, const Tensor* i, float mn, float mx)  { (void)o;(void)i;(void)mn;(void)mx; }
void tensor_celu_gpu(Tensor* o, const Tensor* i, float a)                 { (void)o;(void)i;(void)a; }
void tensor_softsign_gpu(Tensor* o, const Tensor* i)                      { (void)o;(void)i; }
void tensor_rrelu_gpu(Tensor* o, const Tensor* i, float lo, float hi)     { (void)o;(void)i;(void)lo;(void)hi; }
void tensor_threshold_gpu(Tensor* o, const Tensor* i, float th, float v)  { (void)o;(void)i;(void)th;(void)v; }

/* LM Ops stubs */
void tensor_rmsnorm_gpu(Tensor* out, const Tensor* in, const Tensor* weight, float eps) { (void)out; (void)in; (void)weight; (void)eps; }
void tensor_rope_gpu(Tensor* q, Tensor* k, uint32_t dim_head, float theta) { (void)q; (void)k; (void)dim_head; (void)theta; }
void tensor_gated_attention_gpu(Tensor* out, const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* gate, const Tensor* mask) { (void)out; (void)Q; (void)K; (void)V; (void)gate; (void)mask; }
void tensor_gated_deltanet_gpu(Tensor* out, const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* gate, const Tensor* beta) { (void)out; (void)Q; (void)K; (void)V; (void)gate; (void)beta; }
void tensor_swiglu_gpu(Tensor* out, const Tensor* in, int32_t dim) { (void)out; (void)in; (void)dim; }
void tensor_geglu_gpu(Tensor* out, const Tensor* in, int32_t dim) { (void)out; (void)in; (void)dim; }

#endif /* USE_GPU */

