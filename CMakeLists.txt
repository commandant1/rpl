cmake_minimum_required(VERSION 3.10)
project(rpl C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -fopenmp -march=native -fno-math-errno -freciprocal-math -fno-trapping-math -fno-signed-zeros -fassociative-math -funroll-loops -ftree-vectorize")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)  # LTO

# Optional: OpenBLAS support
if(USE_OPENBLAS)
    add_definitions(-DUSE_OPENBLAS)
    find_package(OpenBLAS REQUIRED)
    include_directories(${OpenBLAS_INCLUDE_DIR})
    set(LIBS ${LIBS} ${OpenBLAS_LIBRARIES})
endif()

# Optional: GPU Support (for Raspberry Pi 4 VideoCore VI)
option(USE_GPU "Enable GPU support using GLES Compute Shaders" ON)
if(USE_GPU)
    add_definitions(-DUSE_GPU)
    set(LIBS ${LIBS} EGL GLESv2)
endif()

# Header directories
include_directories(include)

# Source files
set(SOURCES
    src/rpl_core.c
    src/rpl_nn.c
    src/rpl_activations.c
    src/rpl_attention.c
    src/rpl_quantization.c
    src/rpl_data.c
    src/rpl_sklearn.c
    src/rpl_metrics.c
    src/rpl_vision.c
    src/rpl_rl.c
    src/rpl_training.c
    src/ops_extended.c
    src/conv_winograd.c
    src/gemm_optimized.c
    src/probabilistic.c
    src/rpl_gpu.c
    src/rpl_half.c
    src/rpl_math.c
    src/rpl_manipulation.c
    src/rpl_reduce.c
    src/rpl_compare.c
    src/rpl_linalg.c
    src/rpl_fft.c
    src/rpl_random.c
    src/rpl_util.c
)

# Shared library
add_library(rpl SHARED ${SOURCES})
target_link_libraries(rpl m pthread ${LIBS})

# Static library
add_library(rpl_static STATIC ${SOURCES})
target_link_libraries(rpl_static m pthread ${LIBS})

# C Training Example (XOR)
add_executable(rpl_train_xor examples/train_xor.c)
target_link_libraries(rpl_train_xor PRIVATE rpl)

# Test suite
add_executable(test_new_api tests/test_new_api.c)
target_link_libraries(test_new_api PRIVATE rpl m)


# Installation
# GPU Test Executable (Only if GPU is enabled)
if (USE_GPU)
    add_executable(rpl_gpu_test examples/gpu_test.c)
    target_link_libraries(rpl_gpu_test PRIVATE rpl ${LIBS})
    
    install(TARGETS rpl rpl_static rpl_gpu_test rpl_train_xor
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
else()
    install(TARGETS rpl rpl_static rpl_train_xor
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
endif()
install(FILES include/rpl.h
    DESTINATION include
)

if(USE_GPU)
    # Only run test if not cross-compiling
    if(NOT CMAKE_CROSSCOMPILING)
        add_test(NAME GPU_Test COMMAND rpl_gpu_test)
    endif()
endif()
