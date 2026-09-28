cmake_minimum_required(VERSION 3.20)
project(memory_benchmark C ASM)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR
        "Configure with the stm32 preset:\n"
        "  cmake --preset stm32")
endif()

set(CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} -O2 -g -ffunction-sections -fdata-sections -mno-unaligned-access"
    CACHE STRING "" FORCE)

add_executable(firmware.elf
    startup_stm32f407.s
    src/syscalls.c
    src/main.c
)

target_include_directories(firmware.elf PRIVATE
    ${CMAKE_SOURCE_DIR}   # for stm32f407.h
)

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
    COMMENT "Firmware size:"
)

add_custom_target(flash
    COMMAND openocd
        -f interface/stlink.cfg
        -f target/stm32f4x.cfg
        -c "transport select hla_swd"
        -c "program ${CMAKE_BINARY_DIR}/firmware.bin 0x08000000 verify reset exit"
    DEPENDS firmware.elf
    COMMENT "Flashing firmware via OpenOCD (chip runs after)"
    USES_TERMINAL
)

add_custom_target(disasm
    COMMAND ${CMAKE_OBJDUMP} -d -S ${CMAKE_BINARY_DIR}/firmware.elf
    DEPENDS firmware.elf
    COMMENT "Disassembly:"
    USES_TERMINAL
)
