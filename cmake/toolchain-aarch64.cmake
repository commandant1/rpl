# CMake Toolchain File for Cross-Compiling to aarch64 (Raspberry Pi 4/5)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Specify the cross compiler
# Users might need to adjust these paths or add them to PATH
find_program(CMAKE_C_COMPILER NAMES aarch64-linux-gnu-gcc clang)
find_program(CMAKE_CXX_COMPILER NAMES aarch64-linux-gnu-g++ clang++)

if(CMAKE_C_COMPILER MATCHES "clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --target=aarch64-linux-gnu")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --target=aarch64-linux-gnu")
endif()

# Sysroot setup (optional, but good if user has one)
# set(CMAKE_SYSROOT /path/to/sysroot)

# Search paths for libraries
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# RPi-specific flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=cortex-a72 -mfpu=neon-fp-armv8 -mfloat-abi=hard -funsafe-math-optimizations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=cortex-a72 -mfpu=neon-fp-armv8 -mfloat-abi=hard -funsafe-math-optimizations")
