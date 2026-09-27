# arm-none-eabi.cmake
# CMake toolchain file for ARM Cortex-M4 bare-metal cross-compilation.
# Based on ../../04_memmovement/dma_bandwidth_lab/arm-none-eabi.cmake, with
# one addition: the float ABI is a parameter instead of hard-coded, because
# this lab is largely *about* the float ABI. Set FLOAT_ABI from the preset
# (see CMakePresets.json) -- cache variables given on the command line /
# preset are already defined when CMake reads the toolchain file.
#
#   FLOAT_ABI=hard    FPU instructions, float args passed in s0-s15 (default)
#   FLOAT_ABI=softfp  FPU instructions, float args passed in r0-r3
#   FLOAT_ABI=soft    no FPU instructions at all; every float op is a call
#                     into libgcc's software emulation (__aeabi_fmul, ...)
#
# The ABI must match across *every* object in the link, including the
# assembler startup file and the newlib/libgcc multilib GCC selects -- which
# is why it is set here for both C and ASM, not per-file in CMakeLists.txt.

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER   arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)

if(NOT DEFINED FLOAT_ABI)
    set(FLOAT_ABI hard)
endif()

if(FLOAT_ABI STREQUAL "soft")
    # No -mfpu: with the soft ABI GCC never emits FPU instructions anyway,
    # and omitting it makes GCC pick the thumb/v7e-m/nofp multilib.
    set(_fpu_flags "-mfloat-abi=soft")
elseif(FLOAT_ABI STREQUAL "softfp" OR FLOAT_ABI STREQUAL "hard")
    # Cortex-M4 with hardware FPU (FPv4-SP, single-precision only)
    set(_fpu_flags "-mfpu=fpv4-sp-d16 -mfloat-abi=${FLOAT_ABI}")
else()
    message(FATAL_ERROR "FLOAT_ABI must be hard, softfp or soft (got '${FLOAT_ABI}')")
endif()

set(CMAKE_C_FLAGS_INIT
    "-mcpu=cortex-m4 -mthumb ${_fpu_flags} -mno-unaligned-access -fno-pic -fno-pie")
set(CMAKE_ASM_FLAGS_INIT
    "-mcpu=cortex-m4 -mthumb ${_fpu_flags}")

# Prevent CMake from linking a test executable against the host toolchain
# during compiler detection -- we are cross-compiling, output is not runnable.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
