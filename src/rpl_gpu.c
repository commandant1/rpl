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
bool rpl_gpu_init() {
    if (display != EGL_NO_DISPLAY) return true; // Already initialized

    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) {
        fprintf(stderr, "Failed to get EGL display\n");
        return false;
    }

    EGLint major, minor;
    if (!eglInitialize(display, &major, &minor)) {
        fprintf(stderr, "Failed to initialize EGL\n");
        return false;
    }

    EGLint configAttribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_NONE
    };

    EGLConfig config;
    EGLint numConfigs;
    if (!eglChooseConfig(display, configAttribs, &config, 1, &numConfigs) || numConfigs == 0) {
        fprintf(stderr, "Failed to choose EGL config\n");
        return false;
    }

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

    printf("RPL GPU Initialized: %s\n", glGetString(GL_VERSION));
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

#else

// Stubs for non-GPU builds
bool rpl_gpu_init() { return false; }
void rpl_gpu_shutdown() {}
void tensor_to_gpu(Tensor* t) {}
void tensor_from_gpu(Tensor* t) {}
void tensor_add_gpu(Tensor* out, const Tensor* a, const Tensor* b) {}

#endif
