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
    printf("RPL Debug: Tensor %p moved to GPU. Device field: %d\n", (void*)t, t->device);
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        if (op == 0) data_out[id] = data_a[id] + data_b[id];\n"
    "        else if (op == 1) data_out[id] = data_a[id] - data_b[id];\n"
    "        else if (op == 2) data_out[id] = data_a[id] * data_b[id];\n"
    "        else if (op == 3) data_out[id] = data_a[id] / data_b[id];\n"
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

    GLuint num_groups = (out->size + 255) / 256;
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
    "shared float tileA[16][16];\n"
    "shared float tileB[16][16];\n"
    "void main() {\n"
    "    uint row = gl_GlobalInvocationID.y;\n"
    "    uint col = gl_GlobalInvocationID.x;\n"
    "    uint localRow = gl_LocalInvocationID.y;\n"
    "    uint localCol = gl_LocalInvocationID.x;\n"
    "    float sum = 0.0;\n"
    "    for (uint t = 0u; t < (K + 15u) / 16u; t++) {\n"
    "        uint tk = t * 16u;\n"
    /* Load tileA: logical element A[row, tk+localCol] */
    "        uint ak = tk + localCol;\n"
    "        if (row < M && ak < K) {\n"
    "            tileA[localRow][localCol] = (trans_a != 0)\n"
    "                ? A[ak * M + row]\n"   /* A stored [K×M], read transposed */
    "                : A[row * K + ak];\n"  /* A stored [M×K], normal */
    "        } else { tileA[localRow][localCol] = 0.0; }\n"
    /* Load tileB: logical element B[tk+localRow, col] */
    "        uint bk = tk + localRow;\n"
    "        if (col < N && bk < K) {\n"
    "            tileB[localRow][localCol] = (trans_b != 0)\n"
    "                ? B[col * K + bk]\n"   /* B stored [N×K], read transposed */
    "                : B[bk * N + col];\n"  /* B stored [K×N], normal */
    "        } else { tileB[localRow][localCol] = 0.0; }\n"
    "        memoryBarrierShared();\n"
    "        barrier();\n"
    "        for (uint k = 0u; k < 16u; k++) { sum += tileA[localRow][k] * tileB[k][localCol]; }\n"
    "        barrier();\n"
    "    }\n"
    "    if (row < M && col < N) {\n"
    "        uint idx = row * N + col;\n"
    "        C[idx] = alpha * sum + beta * C[idx];\n"
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

    GLuint groups_x = (N + 15) / 16;
    GLuint groups_y = (M + 15) / 16;
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        float val = in_data[id];\n"
    "        out_data[id] = max(val, 0.0);\n"
    "    }\n"
    "}\n";

static const char* SIGMOID_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "void main() {\n"
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        float val = in_data[id];\n"
    "        out_data[id] = 1.0 / (1.0 + exp(-val));\n"
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

    GLuint num_groups = (out->size + 255) / 256;
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        float val = in_data[id];\n"
    "        out_data[id] = tanh(val);\n"
    "    }\n"
    "}\n";

static const char* GELU_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer Input { float in_data[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Output { float out_data[]; };\n"
    "uniform uint size;\n"
    "void main() {\n"
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        float x = in_data[id];\n"
    "        // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))\n"
    "        const float SQRT_2_OVER_PI = 0.7978845608;\n"
    "        const float A = 0.044715;\n"
    "        float inner = SQRT_2_OVER_PI * (x + A * x * x * x);\n"
    "        float res = 0.5 * x * (1.0 + tanh(inner));\n"
    "        out_data[id] = res;\n"
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        float x = in_data[id];\n"
    "        out_data[id] = (x >= 0.0) ? x : negative_slope * x;\n"
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
    glDispatchCompute((out->size + 255) / 256, 1, 1);
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        float x = in_data[id];\n"
    "        out_data[id] = (x >= 0.0) ? x : alpha * (exp(x) - 1.0);\n"
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
    glDispatchCompute((out->size + 255) / 256, 1, 1);
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        float x = in_data[id];\n"
    "        float sig = 1.0 / (1.0 + exp(-x));\n"
    "        out_data[id] = x * sig;\n"
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        float x = in_data[id];\n"
    "        const float lam = 1.0507009873554804934;\n"
    "        const float alp = 1.6732632423543772848;\n"
    "        out_data[id] = (x >= 0.0) ? lam * x : lam * alp * (exp(x) - 1.0);\n"
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        float x = in_data[id];\n"
    "        float sp = log(1.0 + exp(x));  // softplus\n"
    "        out_data[id] = x * tanh(sp);\n"
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        float x = in_data[id];\n"
    "        float clip = clamp(x + 3.0, 0.0, 6.0);\n"
    "        out_data[id] = x * clip / 6.0;\n"
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        float x = in_data[id];\n"
    "        out_data[id] = clamp(x / 6.0 + 0.5, 0.0, 1.0);\n"
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        float x = in_data[id];\n"
    "        float bx = beta * x;\n"
    "        out_data[id] = (bx > threshold) ? x : log(1.0 + exp(bx)) / beta;\n"
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
    glDispatchCompute((out->size + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ===================================
// Log-Softmax Kernel (row-wise)
// ===================================

static const char* LOG_SOFTMAX_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 1) in;\n"           /* one invocation per row */
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
    "layout(local_size_x = 1) in;\n"
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
    glDispatchCompute(num_rows, 1, 1);
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
    glDispatchCompute(num_rows, 1, 1);
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) v[id] = max(0.0, v[id]);\n"
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
    glDispatchCompute((t->size + 127) / 128, 1, 1);
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) v[id] *= scalar;\n"
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
    glDispatchCompute((t->size + 255) / 256, 1, 1);
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
    "layout(local_size_x = 64) in;\n"
    "layout(binding = 0) uniform highp sampler2D input_tex;\n"     /* one channel per draw */
    "layout(std430, binding = 1) readonly buffer Kern { float kern[]; };\n"
    "layout(std430, binding = 2) buffer Out { float out_data[]; };\n"
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
    "    uint gid = gl_GlobalInvocationID.x;\n"
    "    uint total = uint(C_out * out_H * out_W);\n"
    "    if (gid >= total) return;\n"
    "    int oc  = int(gid) / (out_H * out_W);\n"
    "    int rem = int(gid) % (out_H * out_W);\n"
    "    int oy  = rem / out_W;\n"
    "    int ox  = rem % out_W;\n"
    "    float sum = 0.0;\n"
    "    for (int ic = 0; ic < C_in; ic++) {\n"
    "        for (int ky = 0; ky < kH; ky++) {\n"
    "            for (int kx = 0; kx < kW; kx++) {\n"
    "                int iy = oy * stride - padding + ky;\n"
    "                int ix = ox * stride - padding + kx;\n"
    "                float inp = 0.0;\n"
    "                if (iy >= 0 && iy < in_H && ix >= 0 && ix < in_W) {\n"
    "                    /* tex coords: (ix, iy) = (col, row) in GL convention */\n"
    "                    inp = texelFetch(input_tex, ivec2(ix + ic * in_W, iy), 0).r;\n"
    "                }\n"
    "                int kidx = ((oc * C_in + ic) * kH + ky) * kW + kx;\n"
    "                sum += inp * kern[kidx];\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    out_data[int(gid)] = sum;\n"
    "}\n";

static GLuint conv2d_program = 0;

/*
 * Upload input tensor as a wide 2-D texture: width = C_in * W, height = H.
 * Each horizontal stripe of width W corresponds to one input channel.
 * Returns the texture ID (caller must delete).
 */
static GLuint upload_input_texture(const Tensor* in, int C_in, int H, int W) {
    GLuint tex;
    glGenTextures(1, &tex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    /* Pack all channels side by side: tex_width = C_in * W, tex_height = H */
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F,
                 C_in * W, H, 0,
                 GL_RED, GL_FLOAT, in->data);
    return tex;
}

void tensor_conv2d_gpu(Tensor* out, const Tensor* in, const Tensor* kern,
                       int kH, int kW, int stride, int padding) {
    if (!rpl_gpu_init()) return;

    /* Determine batch and spatial dims.
     * Support both [C_in, H, W]  (batch=1, dims=3)
     * and          [batch, C_in, H, W]  (dims=4).
     */
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

    /* Make sure kernel is in CPU memory for texture upload, then push to SSBO */
    tensor_from_gpu((Tensor*)kern);
    tensor_to_gpu((Tensor*)kern);

    /* Prepare output SSBO — size = batch * C_out * out_H * out_W */
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

    /* Ensure input is in CPU memory so we can slice it per batch */
    tensor_from_gpu((Tensor*)in);

    glUseProgram(conv2d_program);
    glUniform1i(glGetUniformLocation(conv2d_program, "C_in"),    C_in);
    glUniform1i(glGetUniformLocation(conv2d_program, "C_out"),   C_out);
    glUniform1i(glGetUniformLocation(conv2d_program, "in_H"),    H);
    glUniform1i(glGetUniformLocation(conv2d_program, "in_W"),    W);
    glUniform1i(glGetUniformLocation(conv2d_program, "out_H"),   out_H);
    glUniform1i(glGetUniformLocation(conv2d_program, "out_W"),   out_W);
    glUniform1i(glGetUniformLocation(conv2d_program, "kH"),      kH);
    glUniform1i(glGetUniformLocation(conv2d_program, "kW"),      kW);
    glUniform1i(glGetUniformLocation(conv2d_program, "stride"),  stride);
    glUniform1i(glGetUniformLocation(conv2d_program, "padding"), padding);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, kern->gpu_buffer);
    glUniform1i(glGetUniformLocation(conv2d_program, "input_tex"), 0);

    GLuint total = (GLuint)(C_out * out_H * out_W);
    int    slice_in  = C_in  * H * W;
    int    slice_out = C_out * out_H * out_W;

    for (int b = 0; b < batch; b++) {
        /* Upload this batch element's slice as a texture */
        const float* in_ptr = in->data + b * slice_in;

        /* Build a temporary staging tensor pointing at this slice */
        Tensor slice;
        slice.data       = (float*)in_ptr;   /* read-only slice */
        slice.dims       = 3;
        slice.shape[0]   = C_in;
        slice.shape[1]   = H;
        slice.shape[2]   = W;
        slice.size       = (uint32_t)slice_in;
        slice.device     = DEVICE_CPU;
        slice.gpu_buffer = 0;

        GLuint tex = upload_input_texture(&slice, C_in, H, W);

        /* Point Output SSBO at the correct batch offset.
         * We use glBindBufferRange to address the sub-range. */
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2,
                          out->gpu_buffer,
                          (GLintptr)((size_t)b * slice_out * sizeof(float)),
                          (GLsizeiptr)((size_t)slice_out * sizeof(float)));

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);

        glDispatchCompute((total + 63) / 64, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        glDeleteTextures(1, &tex);
    }
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
    "void main() {\n"
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id >= size) return;\n"
    "    float x = in_data[id];\n"
    "    float r = 0.0;\n"
    "    if      (op ==  0) r = sin(x);\n"
    "    else if (op ==  1) r = cos(x);\n"
    "    else if (op ==  2) r = tan(x);\n"
    "    else if (op ==  3) r = asin(x);\n"
    "    else if (op ==  4) r = acos(x);\n"
    "    else if (op ==  5) r = atan(x);\n"
    "    else if (op ==  6) r = sinh(x);\n"
    "    else if (op ==  7) r = cosh(x);\n"
    "    else if (op ==  8) r = asinh(x);\n"
    "    else if (op ==  9) r = acosh(x);\n"
    "    else if (op == 10) r = atanh(x);\n"
    "    else if (op == 11) r = exp(x);\n"
    "    else if (op == 12) r = exp2(x);\n"
    "    else if (op == 13) r = exp(x) - 1.0;\n"
    "    else if (op == 14) r = log(x);\n"
    "    else if (op == 15) r = log2(x);\n"
    "    else if (op == 16) r = log(x) / log(10.0);\n"
    "    else if (op == 17) r = log(1.0 + x);\n"
    "    else if (op == 18) r = sqrt(x);\n"
    "    else if (op == 19) r = inversesqrt(x);\n"
    "    else if (op == 20) r = x * x;\n"
    "    else if (op == 21) r = sign(x) * exp(log(abs(x)) / 3.0);\n"
    "    else if (op == 22) r = 1.0 / x;\n"
    "    else if (op == 23) r = abs(x);\n"
    "    else if (op == 24) r = -x;\n"
    "    else if (op == 25) r = sign(x);\n"
    "    else if (op == 26) r = x * float(PI / 180.0);\n"
    "    else if (op == 27) r = x * float(180.0 / PI);\n"
    "    else if (op == 28) {\n"
    "        /* erf approximation: Abramowitz & Stegun 7.1.26 */\n"
    "        float t = 1.0 / (1.0 + 0.3275911 * abs(x));\n"
    "        float poly = t*(0.254829592+t*(-0.284496736+t*(1.421413741+t*(-1.453152027+t*1.061405429))));\n"
    "        r = sign(x) * (1.0 - poly * exp(-x*x));\n"
    "    }\n"
    "    else if (op == 29) r = log(x / (1.0 - x));\n"
    "    else if (op == 30) r = floor(x + 0.5);\n"
    "    else if (op == 31) r = floor(x);\n"
    "    else if (op == 32) r = ceil(x);\n"
    "    else if (op == 33) r = trunc(x);\n"
    "    else if (op == 34) r = x - trunc(x);\n"
    "    out_data[id] = r;\n"
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
    glDispatchCompute((out->size + 255) / 256, 1, 1);
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
    "void main() {\n"
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id >= size_a) return;\n"
    "    float x = a[id];\n"
    "    float y = b[id % size_b];\n"
    "    float r = 0.0;\n"
    "    if      (op == 0) r = pow(abs(x), y) * sign(x);\n"
    "    else if (op == 1) r = atan(x, y);\n"
    "    else if (op == 2) r = sqrt(x*x + y*y);\n"
    "    else if (op == 3) r = x - trunc(x/y)*y;\n"
    "    else if (op == 4) { float q = x/y; float n = (q >= 0.0) ? floor(q+0.5) : ceil(q-0.5); r = x - n*y; }\n"
    "    else if (op == 5) r = floor(x/y);\n"
    "    else if (op == 6) r = max(x, y);\n"
    "    else if (op == 7) r = min(x, y);\n"
    "    else if (op == 8) { float mx = max(x,y); r = mx + log(exp(x-mx)+exp(y-mx)); }\n"
    "    else if (op == 9) { float mx = max(x,y); r = mx + log2(exp2(x-mx)+exp2(y-mx)); }\n"
    "    out_data[id] = r;\n"
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
    glDispatchCompute((a->size + 255) / 256, 1, 1);
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) out_data[id] = clamp(in_data[id], lo, hi);\n"
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
    glDispatchCompute((out->size + 255) / 256, 1, 1);
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        float x = in_data[id];\n"
    "        float pos = max(0.0, x);\n"
    "        float neg = min(0.0, alpha * (exp(x / alpha) - 1.0));\n"
    "        out_data[id] = pos + neg;\n"
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
    glDispatchCompute((out->size + 255) / 256, 1, 1);
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        float x = in_data[id];\n"
    "        out_data[id] = x / (1.0 + abs(x));\n"
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        float x = in_data[id];\n"
    "        out_data[id] = (x >= 0.0) ? x : slope * x;\n"
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
    glDispatchCompute((out->size + 255) / 256, 1, 1);
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
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        float x = in_data[id];\n"
    "        out_data[id] = (x > threshold) ? x : value;\n"
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
    glDispatchCompute((out->size + 255) / 256, 1, 1);
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

#endif /* USE_GPU */

