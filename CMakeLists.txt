cmake_minimum_required(VERSION 3.10)
project(rpl C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -fopenmp -march=native")

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

# Installation
install(TARGETS rpl rpl_static rpl_gpu_test rpl_train_xor
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)
install(FILES include/rpl.h
    DESTINATION include
)

if(USE_GPU)
    add_executable(rpl_gpu_test examples/gpu_test.c)
    target_link_libraries(rpl_gpu_test PRIVATE rpl EGL GLESv2)
    # Only run test if not cross-compiling
    if(NOT CMAKE_CROSSCOMPILING)
        add_test(NAME GPU_Test COMMAND rpl_gpu_test)
    endif()
endif()
