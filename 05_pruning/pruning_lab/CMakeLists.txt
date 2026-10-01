# Pruning lab firmware. ExecuTorch import pattern and flags follow
# ../../../lab0_hello_world_soln/CMakeLists.txt; see README.md.

cmake_minimum_required(VERSION 3.20)
project(pruning_lab CXX C ASM)

if(NOT DEFINED EXECUTORCH_BUILD_DIR)
    message(FATAL_ERROR "EXECUTORCH_BUILD_DIR must be set. Use: cmake --preset stm32")
endif()

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/generated/model_pte.h)
    message(FATAL_ERROR
        "generated/model_pte.h is missing. Pick a variant first:\n"
        "  python3 python/export_variants.py --variant A")
endif()

# ExecuTorch headers come from the source tree, not the build tree.
set(EXECUTORCH_SRC_DIR "$ENV{HOME}/executorch")
set(ET_LIB_DIR ${EXECUTORCH_BUILD_DIR})

# ------------------------------------------------------------------ #
#  Op check: the shared cortex-m4 tree's libexecutorch_selected_kernels #
#  only contains the ops in the EXECUTORCH_SELECT_OPS_LIST it was      #
#  configured with. A missing op links fine and only fails on the      #
#  board (load_method -> 0x14 OperatorMissing), so check up front.     #
#  These are the ops every variant's exported program uses             #
#  (export_variants.py prints them).                                   #
# ------------------------------------------------------------------ #
set(PRUNING_LAB_OPS
    aten::addmm.out
    aten::convolution.out
    aten::max_pool2d_with_indices.out
    aten::mean.out
    aten::permute_copy.out
    aten::relu.out
)
set(ET_CACHE ${EXECUTORCH_BUILD_DIR}/CMakeCache.txt)
if(EXISTS ${ET_CACHE})
    file(STRINGS ${ET_CACHE} ET_OPS_LINE REGEX "^EXECUTORCH_SELECT_OPS_LIST:")
    string(REGEX REPLACE "^[^=]*=" "" ET_OPS "${ET_OPS_LINE}")
    string(REPLACE "," ";" ET_OPS "${ET_OPS}")
    set(ET_MISSING_OPS "")
    foreach(op ${PRUNING_LAB_OPS})
        if(NOT op IN_LIST ET_OPS)
            list(APPEND ET_MISSING_OPS ${op})
        endif()
    endforeach()
    if(ET_MISSING_OPS)
        string(REPLACE ";" "\n  " ET_MISSING_TEXT "${ET_MISSING_OPS}")
        message(FATAL_ERROR
            "The ExecuTorch build in ${EXECUTORCH_BUILD_DIR} is missing ops this lab needs:\n"
            "  ${ET_MISSING_TEXT}\n"
            "Rebuild it with a larger op list -- see README.md, \"ExecuTorch kernels\".")
    endif()
else()
    message(WARNING
        "Can't find ${ET_CACHE}, so the op list wasn't checked. If load_method "
        "fails with 0x14 (OperatorMissing), see README.md, \"ExecuTorch kernels\".")
endif()

# ------------------------------------------------------------------ #
#  Imported ExecuTorch libraries                                       #
#                                                                      #
#  All from the shared cortex-m4 tree. Its EXECUTORCH_SELECT_OPS_LIST  #
#  already covers every op these models use (convolution,             #
#  max_pool2d_with_indices, relu, mean, permute_copy, addmm) -- see    #
#  README.md.                                                          #
# ------------------------------------------------------------------ #
add_library(executorch_core STATIC IMPORTED)
set_target_properties(executorch_core PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/libexecutorch_core.a)

add_library(executorch STATIC IMPORTED)
set_target_properties(executorch PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/libexecutorch.a)

add_library(executorch_selected_kernels STATIC IMPORTED)
set_target_properties(executorch_selected_kernels PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/libexecutorch_selected_kernels.a)

add_library(portable_kernels STATIC IMPORTED)
set_target_properties(portable_kernels PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/kernels/portable/libportable_kernels.a)

add_library(extension_data_loader STATIC IMPORTED)
set_target_properties(extension_data_loader PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/extension/data_loader/libextension_data_loader.a)

add_library(flatccrt STATIC IMPORTED)
set_target_properties(flatccrt PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/third-party/flatcc_ep/lib/libflatccrt.a)

set(ET_INCLUDES
    ${EXECUTORCH_SRC_DIR}
    ${EXECUTORCH_SRC_DIR}/executorch/third-party/flatbuffers/include
    ${ET_LIB_DIR}/third-party/flatcc_ep/include
    $ENV{HOME}/executorch/et-env/lib/python3.12/site-packages/torch/include
)

# ------------------------------------------------------------------ #
#  STM32F407 firmware                                                  #
# ------------------------------------------------------------------ #
set(CPU_FLAGS "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard")

set(CMAKE_C_FLAGS
    "${CPU_FLAGS} -Os -g -ffunction-sections -fdata-sections -mno-unaligned-access -DET_LOG_ENABLED=0"
    CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS
    "${CPU_FLAGS} -Os -g -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -std=c++17 -mno-unaligned-access -DET_LOG_ENABLED=0"
    CACHE STRING "" FORCE)
set(CMAKE_ASM_FLAGS "${CPU_FLAGS} -g" CACHE STRING "" FORCE)

add_executable(firmware.elf
    startup_stm32f407.s
    src/syscalls.c
    src/main.cpp
)

target_include_directories(firmware.elf PRIVATE
    ${ET_INCLUDES}
    ${CMAKE_SOURCE_DIR}             # stm32f407.h
    ${CMAKE_SOURCE_DIR}/generated   # model_pte.h, test_images.h
)

# --whole-archive: op registration happens in static initializers with
# no directly referenced symbols; without it the linker drops them and
# load_method() fails with OperatorMissing (0x14).
target_link_libraries(firmware.elf PRIVATE
    -Wl,--whole-archive
    executorch_selected_kernels
    -Wl,--no-whole-archive
    portable_kernels
    executorch
    executorch_core
    extension_data_loader
    flatccrt
    c m gcc
)

# -nostartfiles excludes crti.o/crtn.o; _init/_fini stubs are provided
# in startup_stm32f407.s instead.
target_link_options(firmware.elf PRIVATE
    -nostartfiles
    -specs=nano.specs
    -T${CMAKE_SOURCE_DIR}/stm32f407.ld
    -Wl,--gc-sections
    -Wl,-Map=${CMAKE_BINARY_DIR}/firmware.map
)

add_custom_command(TARGET firmware.elf POST_BUILD
    COMMAND arm-none-eabi-objcopy -O binary firmware.elf firmware.bin
    COMMAND arm-none-eabi-size firmware.elf
    COMMENT "Generating firmware.bin; firmware size:"
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
