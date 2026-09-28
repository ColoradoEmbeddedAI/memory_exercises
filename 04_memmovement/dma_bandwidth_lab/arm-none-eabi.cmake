# arm-none-eabi.cmake
# CMake toolchain file for ARM Cortex-M4 bare-metal cross-compilation.
# Copied from lab0_hello_world_soln/arm-none-eabi.cmake -- generic, not
# ExecuTorch-specific, so it's vendored here directly instead of pointed at
# via ~/executorch/... (this project has no ExecuTorch dependency).

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Locate the toolchain. On Ubuntu (apt gcc-arm-none-eabi + libnewlib-arm-none-eabi)
# it's on PATH. On macOS the Arm GNU Toolchain (Arm's .pkg installer, or
# `brew install --cask gcc-arm-embedded`) lives under
# /Applications/ArmGNUToolchain/<version>/ and isn't always on PATH, so look
# there too (newest version first). Override with -DARM_TOOLCHAIN_DIR=<bin dir>
# or the ARM_TOOLCHAIN_DIR environment variable.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES ARM_TOOLCHAIN_DIR)
file(GLOB _arm_gnu_bins "/Applications/ArmGNUToolchain/*/arm-none-eabi/bin")
list(SORT _arm_gnu_bins COMPARE NATURAL ORDER DESCENDING)
find_program(ARM_NONE_EABI_GCC arm-none-eabi-gcc
    HINTS ${ARM_TOOLCHAIN_DIR} $ENV{ARM_TOOLCHAIN_DIR}
    PATHS ${_arm_gnu_bins})
if(NOT ARM_NONE_EABI_GCC)
    message(FATAL_ERROR
        "arm-none-eabi-gcc not found. Install it with\n"
        "  Ubuntu: sudo apt install gcc-arm-none-eabi libnewlib-arm-none-eabi\n"
        "  macOS:  brew install --cask gcc-arm-embedded\n"
        "or pass -DARM_TOOLCHAIN_DIR=/path/to/toolchain/bin")
endif()
get_filename_component(_arm_bin "${ARM_NONE_EABI_GCC}" DIRECTORY)

# The link uses newlib-nano (-specs=nano.specs). Homebrew's arm-none-eabi-gcc
# *formula* ships without newlib, which otherwise fails late and cryptically.
execute_process(COMMAND "${ARM_NONE_EABI_GCC}" -print-file-name=nano.specs
    OUTPUT_VARIABLE _nano_specs OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT IS_ABSOLUTE "${_nano_specs}")
    message(FATAL_ERROR
        "${ARM_NONE_EABI_GCC} has no newlib (nano.specs not found).\n"
        "  Ubuntu: sudo apt install libnewlib-arm-none-eabi\n"
        "  macOS:  brew uninstall arm-none-eabi-gcc && brew install --cask gcc-arm-embedded")
endif()

set(CMAKE_C_COMPILER   ${_arm_bin}/arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER ${_arm_bin}/arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER ${_arm_bin}/arm-none-eabi-gcc)

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
