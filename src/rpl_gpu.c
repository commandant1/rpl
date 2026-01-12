#include "rpl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_GPU

#include <EGL/egl.h>
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
static PFNGLMAPBUFFERRANGEPROC p_glMapBufferRange = NULL;
static PFNGLUNMAPBUFFERPROC p_glUnmapBuffer = NULL;
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
static PFNGLUNIFORM1UIPROC p_glUniform1ui = NULL;
static PFNGLGETUNIFORMLOCATIONPROC p_glGetUniformLocation = NULL;
static PFNGLDISPATCHCOMPUTEPROC p_glDispatchCompute = NULL;
static PFNGLMEMORYBARRIERPROC p_glMemoryBarrier = NULL;
static PFNGLGETSTRINGPROC p_glGetString = NULL;

static void load_gl_funcs() {
    p_glGenBuffers = (PFNGLGENBUFFERSPROC)eglGetProcAddress("glGenBuffers");
    p_glBindBuffer = (PFNGLBINDBUFFERPROC)eglGetProcAddress("glBindBuffer");
    p_glBufferData = (PFNGLBUFFERDATAPROC)eglGetProcAddress("glBufferData");
    p_glMapBufferRange = (PFNGLMAPBUFFERRANGEPROC)eglGetProcAddress("glMapBufferRange");
    p_glUnmapBuffer = (PFNGLUNMAPBUFFERPROC)eglGetProcAddress("glUnmapBuffer");
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
    p_glUniform1ui = (PFNGLUNIFORM1UIPROC)eglGetProcAddress("glUniform1ui");
    p_glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)eglGetProcAddress("glGetUniformLocation");
    p_glDispatchCompute = (PFNGLDISPATCHCOMPUTEPROC)eglGetProcAddress("glDispatchCompute");
    p_glMemoryBarrier = (PFNGLMEMORYBARRIERPROC)eglGetProcAddress("glMemoryBarrier");
    p_glGetString = (PFNGLGETSTRINGPROC)eglGetProcAddress("glGetString");
}

// Macro helper to call dynamic pointers
#define glGenBuffers p_glGenBuffers
#define glBindBuffer p_glBindBuffer
#define glBufferData p_glBufferData
#define glMapBufferRange p_glMapBufferRange
#define glUnmapBuffer p_glUnmapBuffer
#define glCreateShader p_glCreateShader
#define glShaderSource p_glShaderSource
#define glCompileShader p_glCompileShader
#define glGetShaderiv p_glGetShaderiv
#define glGetShaderInfoLog p_glGetShaderInfoLog
#define glCreateProgram p_glCreateProgram
#define glAttachShader p_glAttachShader
#define glLinkProgram p_glLinkProgram
#define glDeleteShader p_glDeleteShader
#define glUseProgram p_glUseProgram
#define glBindBufferBase p_glBindBufferBase
#define glUniform1ui p_glUniform1ui
#define glGetUniformLocation p_glGetUniformLocation
#define glDispatchCompute p_glDispatchCompute
#define glMemoryBarrier p_glMemoryBarrier
#define glGetString p_glGetString

bool rpl_gpu_init() {
    if (display != EGL_NO_DISPLAY) return true; // Already initialized

    // Try default display first
    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    
    EGLint major, minor;
    if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor)) {
        // Fallback for headless: Surfaceless
        // Some systems require explicit PFN loading, but simple linking might work if supported
        // Or simpler: Just tell user to export EGL_PLATFORM=surfaceless
        
        // Let's print detailed error
        fprintf(stderr, "Default EGL init failed. Trying fallback...\n");
        
        // Terminate previous attempt
        if (display != EGL_NO_DISPLAY) eglTerminate(display);
        display = EGL_NO_DISPLAY;
        
        // Try getting display via generic method (might need extensions)
        // For now, simpler approach: Just error out but suggest environment var
        fprintf(stderr, "RPL requires a valid EGL display.\n");
        fprintf(stderr, "On Raspberry Pi/Headless, try running: export EGL_PLATFORM=surfaceless\n");
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
    } else {
        printf("RPL GPU Initialized (Function loading failed?)\n");
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
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, t->size * sizeof(float), t->data, GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    t->gpu_buffer = buffer;
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
    
    // Optional: Keep buffer on GPU? For now, let's treat it as sync
    // t->device = DEVICE_CPU; 
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

static const char* ADD_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 256) in;\n"
    "layout(std430, binding = 0) readonly buffer InputA { float data_a[]; };\n"
    "layout(std430, binding = 1) readonly buffer InputB { float data_b[]; };\n"
    "layout(std430, binding = 2) writeonly buffer Output { float data_out[]; };\n"
    "uniform uint size;\n"
    "void main() {\n"
    "    uint id = gl_GlobalInvocationID.x;\n"
    "    if (id < size) {\n"
    "        data_out[id] = data_a[id] + data_b[id];\n"
    "    }\n"
    "}\n";

static GLuint add_program = 0;

void tensor_add_gpu(Tensor* out, const Tensor* a, const Tensor* b) {
    if (!rpl_gpu_init()) return;

    // Ensure inputs are on GPU
    tensor_to_gpu((Tensor*)a);
    tensor_to_gpu((Tensor*)b);
    
    // Allocate output on GPU if not already
    if (out->device != DEVICE_GPU) {
        tensor_to_gpu(out); // This allocates specific size based on out->size
    }

    // Compile shader if needed
    if (add_program == 0) {
        add_program = compile_compute_shader(ADD_SHADER_SRC);
        if (add_program == 0) return;
    }

    glUseProgram(add_program);

    // Bind buffers
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, a->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b->gpu_buffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, out->gpu_buffer);

    // Set uniforms
    glUniform1ui(glGetUniformLocation(add_program, "size"), out->size);

    // Dispatch
    // local_size_x is 256, so we need ceil(size / 256) groups
    GLuint num_groups = (out->size + 255) / 256;
    glDispatchCompute(num_groups, 1, 1);

    // Barrier to ensure completion before next read/write
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

static const char* GEMM_SHADER_SRC =
    "#version 310 es\n"
    "layout(local_size_x = 16, local_size_y = 16) in;\n"
    "layout(std430, binding = 0) readonly buffer InputA { float A[]; };\n"
    "layout(std430, binding = 1) readonly buffer InputB { float B[]; };\n"
    "layout(std430, binding = 2) writeonly buffer Output { float C[]; };\n"
    "uniform uint M;\n"
    "uniform uint N;\n"
    "uniform uint K;\n"
    "shared float tileA[16][16];\n"
    "shared float tileB[16][16];\n"
    "void main() {\n"
    "    uint row = gl_GlobalInvocationID.y;\n"
    "    uint col = gl_GlobalInvocationID.x;\n"
    "    uint localRow = gl_LocalInvocationID.y;\n"
    "    uint localCol = gl_LocalInvocationID.x;\n"
    "    float sum = 0.0;\n"
    "    for (uint t = 0u; t < (K + 15u) / 16u; t++) {\n"
    "        uint tilingK = t * 16u;\n"
    "        if (row < M && tilingK + localCol < K)\n"
    "            tileA[localRow][localCol] = A[row * K + tilingK + localCol];\n"
    "        else\n"
    "            tileA[localRow][localCol] = 0.0;\n"
    "        if (col < N && tilingK + localRow < K)\n"
    "            tileB[localRow][localCol] = B[(tilingK + localRow) * N + col];\n"
    "        else\n"
    "            tileB[localRow][localCol] = 0.0;\n"
    "        memoryBarrierShared();\n"
    "        barrier();\n"
    "        for (uint k = 0u; k < 16u; k++) {\n"
    "            sum += tileA[localRow][k] * tileB[k][localCol];\n"
    "        }\n"
    "        barrier();\n"
    "    }\n"
    "    if (row < M && col < N) {\n"
    "        C[row * N + col] = sum;\n"
    "    }\n"
    "}\n";

static GLuint gemm_program = 0;

void tensor_matmul_gpu(Tensor* C, const Tensor* A, const Tensor* B) {
    if (!rpl_gpu_init()) return;

    // A: [M, K], B: [K, N], C: [M, N]
    // Assume 2D tensors for now
    uint32_t M = A->shape[0];
    uint32_t K = A->shape[1];
    uint32_t N = B->shape[1];

    if (B->shape[0] != K) {
        fprintf(stderr, "GEMM Shape mismatch: %ux%u vs %ux%u\n", M, K, B->shape[0], N);
        return;
    }

    tensor_to_gpu((Tensor*)A);
    tensor_to_gpu((Tensor*)B);
    
    // Ensure C is ready
    if (C->device != DEVICE_GPU) {
        // Update C shape if not already correct (simplified logic)
        C->shape[0] = M;
        C->shape[1] = N;
        C->size = M * N;
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

    // Dispatch (16x16 tiles)
    GLuint num_groups_x = (N + 15) / 16;
    GLuint num_groups_y = (M + 15) / 16;
    glDispatchCompute(num_groups_x, num_groups_y, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
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

#else

// Stubs for non-GPU builds
bool rpl_gpu_init() { return false; }
void rpl_gpu_shutdown() {}
void tensor_to_gpu(Tensor* t) {}
void tensor_from_gpu(Tensor* t) {}
void tensor_add_gpu(Tensor* out, const Tensor* a, const Tensor* b) {}
void tensor_matmul_gpu(Tensor* C, const Tensor* A, const Tensor* B) {}
void tensor_relu_gpu(Tensor* out, const Tensor* in) {}
void tensor_sigmoid_gpu(Tensor* out, const Tensor* in) {}
void tensor_tanh_gpu(Tensor* out, const Tensor* in) {}
void tensor_gelu_gpu(Tensor* out, const Tensor* in) {}

#endif
