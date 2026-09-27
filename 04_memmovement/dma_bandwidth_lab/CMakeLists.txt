cmake_minimum_required(VERSION 3.20)
project(dma_bandwidth_lab C ASM)

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

add_custom_command(TARGET firmware.elf POST_BUILD
    COMMAND arm-none-eabi-objcopy -O binary firmware.elf firmware.bin
    COMMENT "Generating firmware.bin"
)
add_custom_command(TARGET firmware.elf POST_BUILD
    COMMAND arm-none-eabi-size firmware.elf
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
    COMMAND arm-none-eabi-objdump -d -S ${CMAKE_BINARY_DIR}/firmware.elf
    DEPENDS firmware.elf
    COMMENT "Disassembly:"
    USES_TERMINAL
)
