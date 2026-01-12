# Raspberry Pi 4 Hardware Architecture & Optimization Guide

## GPU Architecture: Broadcom VideoCore VI
RPL's GPU backend is specifically tuned for the **VideoCore VI** GPU found in the Raspberry Pi 4 Model B (and Compute Module 4).

### Key Specifications
*   **Architecture**: VideoCore VI (built into BCM2711 SoC)
*   **API Support**: OpenGL ES 3.1 (Supports Compute Shaders)
*   **Driver Stack**: Mesa V3D (Open Source)
*   **Clock Speed**: ~500 MHz (Stock)
*   **Shading Units**: ~64 QPU (Quad Processing Units) shader cores
*   **Performance**: ~10-30 GFLOPS (FP32)

### RPL Optimization Strategy
Due to the modest shading power and shared memory architecture, RPL employs specific strategies:

1.  **Compute Shaders**: We use GLES 3.1 Compute Shaders to bypass the graphics pipeline and treat the GPU as a parallel vector processor.
2.  **Tiled Compute**: Our Matrix Multiplication (GEMM) uses **16x16 tiling**. This aligns perfectly with the QPU's 16-way SIMD architecture, maximizing register utilization and minimizing expensive global memory creation.
3.  **Headless EGL**: The library is designed to initialize the GPU context without an X11/Wayland display server (`surfaceless` platform), making it ideal for edge devices and servers.

## CPU Architecture: ARM Cortex-A72
*   **Cores**: 4x ARM Cortex-A72 (ARMv8-A)
*   **SIMD**: NEON (128-bit wide vector registers)
*   **Clock**: 1.5 GHz (up to 2.0 GHz+ with overclocking)

### RPL Optimizations
*   **NEON Intrinsics**: Critical paths (Add, Mul, ReLU) are hand-written using `arm_neon.h` intrinsics to process 4 floats per cycle per core.
*   **Hybrid Threading**: We combine OpenMP (4 threads) with NEON (4x SIMD) to theoretically process 16 floats per cycle across the CPU.

## Performance Tuning Tips
1.  **Overclocking**: Overclocking the GPU (`gpu_freq`) in `/boot/config.txt` can linearly improve matrix multiplication performance. Ensure adequate cooling.
2.  **Memory**: The GPU shares system RAM. Higher RAM clock speeds (if adjustable) or lower system load improves bandwidth validation.
3.  **FP16**: Use `to_half()` to store tensors in half-precision, reducing memory pressure by 50%.
