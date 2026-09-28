cmake_minimum_required(VERSION 3.20)
project(cortex_m_perf_lab C ASM)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR
        "Configure with one of the presets:\n"
        "  cmake --preset f32-hard     # float32 kernel, hard-float ABI\n"
        "  cmake --preset f32-soft     # float32 kernel, soft-float ABI\n"
        "  cmake --preset q15-scalar   # q15 scalar kernel\n"
        "  cmake --preset q15-simd     # q15 SIMD (SMLAD) kernel")
endif()

# Which single kernel this build times (see src/main.c). Set by the preset.
set(KERNEL "" CACHE STRING "Kernel to time: f32, q15_scalar or q15_simd")
set_property(CACHE KERNEL PROPERTY STRINGS f32 q15_scalar q15_simd)
if(NOT KERNEL MATCHES "^(f32|q15_scalar|q15_simd)$")
    message(FATAL_ERROR "KERNEL must be f32, q15_scalar or q15_simd (got '${KERNEL}')")
endif()
string(TOUPPER "${KERNEL}" KERNEL_UPPER)

# Optimization comes from CMAKE_BUILD_TYPE=RelWithDebInfo (-O2 -g -DNDEBUG)
# in CMakePresets.json. Don't add -O2 here as well: CMake appends the
# build-type flags *after* CMAKE_C_FLAGS, so a build type like MinSizeRel
# would silently override it with -Os -- and this lab's cycle counts
# depend on the optimization level.
set(CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections -mno-unaligned-access"
    CACHE STRING "" FORCE)

add_executable(firmware.elf
    startup_stm32f407.s
    src/syscalls.c
    src/main.c
)

target_include_directories(firmware.elf PRIVATE
    ${CMAKE_SOURCE_DIR}   # for stm32f407.h
)

target_compile_definitions(firmware.elf PRIVATE KERNEL_${KERNEL_UPPER})

target_link_libraries(firmware.elf PRIVATE c m gcc)

# -nostartfiles excludes crti.o/crtn.o; _init/_fini stubs are provided in
# startup_stm32f407.s instead.
target_link_options(firmware.elf PRIVATE
    -nostartfiles
    -specs=nano.specs
    -T${CMAKE_SOURCE_DIR}/stm32f407.ld
    -Wl,--gc-sections
    -Wl,-Map=${CMAKE_BINARY_DIR}/firmware.map,--cref
)

# Use the binutils that sit next to the compiler found by the toolchain file
# (CMake sets CMAKE_OBJCOPY/CMAKE_OBJDUMP from it) rather than whatever is on
# PATH -- on macOS the Arm toolchain often isn't on PATH at all.
get_filename_component(ARM_TOOLCHAIN_BIN "${CMAKE_C_COMPILER}" DIRECTORY)
find_program(ARM_SIZE arm-none-eabi-size HINTS "${ARM_TOOLCHAIN_BIN}" REQUIRED)

add_custom_command(TARGET firmware.elf POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary firmware.elf firmware.bin
    COMMENT "Generating firmware.bin"
)
add_custom_command(TARGET firmware.elf POST_BUILD
    COMMAND ${ARM_SIZE} firmware.elf
    COMMENT "Firmware size (KERNEL=${KERNEL}, FLOAT_ABI=${FLOAT_ABI}):"
)

add_custom_target(flash
    COMMAND openocd
        -f interface/stlink.cfg
        -f target/stm32f4x.cfg
        -c "transport select hla_swd"
        -c "program ${CMAKE_BINARY_DIR}/firmware.bin 0x08000000 verify reset exit"
    DEPENDS firmware.elf
    COMMENT "Flashing firmware (KERNEL=${KERNEL}, FLOAT_ABI=${FLOAT_ABI}) via OpenOCD (chip runs after)"
    USES_TERMINAL
)

add_custom_target(disasm
    COMMAND ${CMAKE_OBJDUMP} -d -S ${CMAKE_BINARY_DIR}/firmware.elf
    DEPENDS firmware.elf
    COMMENT "Disassembly:"
    USES_TERMINAL
)

# Just this build's timed kernel (bench_dot_<KERNEL>) -- its inner loop is
# the code the lab's cycle counts describe.
add_custom_target(disasm-kernel
    COMMAND ${CMAKE_OBJDUMP} -d --no-show-raw-insn
        --disassemble=bench_dot_${KERNEL} ${CMAKE_BINARY_DIR}/firmware.elf
    DEPENDS firmware.elf
    COMMENT "Kernel disassembly (KERNEL=${KERNEL}, FLOAT_ABI=${FLOAT_ABI}):"
    USES_TERMINAL
)
