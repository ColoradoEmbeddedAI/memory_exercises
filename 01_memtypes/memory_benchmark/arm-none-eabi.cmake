# arm-none-eabi.cmake
# CMake toolchain file for ARM Cortex-M4 bare-metal cross-compilation.
# Copied from lab0_hello_world_soln/arm-none-eabi.cmake -- generic, not
# ExecuTorch-specific, so it's vendored here directly instead of pointed at
# via ~/executorch/... (this project has no ExecuTorch dependency).

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER   arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)

# Cortex-M4 with hardware FPU (FPv4-SP, single-precision only)
set(CMAKE_C_FLAGS_INIT
    "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mno-unaligned-access -fno-pic -fno-pie")
set(CMAKE_ASM_FLAGS_INIT
    "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard")

# Prevent CMake from linking a test executable against the host toolchain
# during compiler detection -- we are cross-compiling, output is not runnable.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
