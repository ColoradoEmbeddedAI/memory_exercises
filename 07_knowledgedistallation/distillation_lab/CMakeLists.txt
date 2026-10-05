# Distillation lab firmware (Part 5, optional). The quantization lab's
# CMakeLists.txt (../../06_quantization/quantization_lab/), minus its avgpool
# fix (an MLP has no pooling). Same ExecuTorch tree, with the Cortex-M
# backend and CMSIS-NN. See README.md.

cmake_minimum_required(VERSION 3.20)
project(distillation_lab CXX C ASM)

if(NOT DEFINED EXECUTORCH_BUILD_DIR)
    message(FATAL_ERROR "EXECUTORCH_BUILD_DIR must be set. Use: cmake --preset stm32")
endif()

if(NOT EXISTS ${CMAKE_SOURCE_DIR}/generated/model_pte.h)
    message(FATAL_ERROR
        "generated/model_pte.h is missing. Export a build first:\n"
        "  python3 python/export_mlp.py --model teacher --precision int8")
endif()

# ExecuTorch headers come from the source tree, not the build tree.
set(EXECUTORCH_SRC_DIR "$ENV{HOME}/executorch")
set(ET_LIB_DIR ${EXECUTORCH_BUILD_DIR})
# CMSIS-NN source checkout, built below.
set(CMSIS_NN_DIR "$ENV{HOME}/executorch/CMSIS-NN" CACHE PATH "CMSIS-NN source checkout")

# ------------------------------------------------------------------ #
#  Check the ExecuTorch tree before anything links against it.         #
#  1. It must be built with the Cortex-M backend                       #
#     (EXECUTORCH_BUILD_CORTEX_M=ON), which makes cortex_m_ops_lib.    #
#  2. Its EXECUTORCH_SELECT_OPS_LIST must include every portable op    #
#     the fp32 builds use. A missing op links fine and only fails on   #
#     the board (load_method -> 0x14 OperatorMissing), so check up     #
#     front.                                                           #
# ------------------------------------------------------------------ #
if(NOT EXISTS ${ET_LIB_DIR}/backends/cortex_m/libcortex_m_ops_lib.a)
    message(FATAL_ERROR
        "${ET_LIB_DIR} has no Cortex-M backend (backends/cortex_m/libcortex_m_ops_lib.a).\n"
        "Build the cortex-m4-cmsis tree -- see the quantization lab's README.md.")
endif()

set(LAB_PORTABLE_OPS
    aten::addmm.out
    aten::permute_copy.out
    aten::relu.out
    dim_order_ops::_clone_dim_order.out
)
set(ET_CACHE ${EXECUTORCH_BUILD_DIR}/CMakeCache.txt)
if(EXISTS ${ET_CACHE})
    file(STRINGS ${ET_CACHE} ET_OPS_LINE REGEX "^EXECUTORCH_SELECT_OPS_LIST:")
    string(REGEX REPLACE "^[^=]*=" "" ET_OPS "${ET_OPS_LINE}")
    string(REPLACE "," ";" ET_OPS "${ET_OPS}")
    set(ET_MISSING_OPS "")
    foreach(op ${LAB_PORTABLE_OPS})
        if(NOT op IN_LIST ET_OPS)
            list(APPEND ET_MISSING_OPS ${op})
        endif()
    endforeach()
    if(ET_MISSING_OPS)
        string(REPLACE ";" "\n  " ET_MISSING_TEXT "${ET_MISSING_OPS}")
        message(FATAL_ERROR
            "The ExecuTorch build in ${EXECUTORCH_BUILD_DIR} is missing ops this lab needs:\n"
            "  ${ET_MISSING_TEXT}\n"
            "Rebuild it with the op list in the quantization lab's README.md.")
    endif()
endif()

# ------------------------------------------------------------------ #
#  Imported ExecuTorch libraries                                       #
# ------------------------------------------------------------------ #
function(et_import name path)
    add_library(${name} STATIC IMPORTED)
    set_target_properties(${name} PROPERTIES IMPORTED_LOCATION ${ET_LIB_DIR}/${path})
endfunction()

et_import(executorch_core              libexecutorch_core.a)
et_import(executorch                   libexecutorch.a)
et_import(executorch_selected_kernels  libexecutorch_selected_kernels.a)
et_import(portable_kernels             kernels/portable/libportable_kernels.a)
et_import(kernels_util_all_deps        kernels/portable/cpu/util/libkernels_util_all_deps.a)
et_import(cortex_m_ops_lib             backends/cortex_m/libcortex_m_ops_lib.a)
et_import(cortex_m_kernels             backends/cortex_m/libcortex_m_kernels.a)
et_import(extension_data_loader        extension/data_loader/libextension_data_loader.a)
et_import(flatccrt                     third-party/flatcc_ep/lib/libflatccrt.a)

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

# ------------------------------------------------------------------ #
#  CMSIS-NN, compiled here from source instead of linking the copy in  #
#  the ExecuTorch tree, so this lab controls one flag:                 #
#                                                                      #
#  Every project in this course compiles with -mno-unaligned-access   #
#  (the FlatBuffer workaround in src/syscalls.c). CMSIS-NN reads 4     #
#  int8 values at a time with memcpy(&word, ptr, 4). With unaligned    #
#  access allowed, GCC turns that into one LDR. Without it, GCC can't  #
#  assume the pointer is aligned, so it calls memcpy -- syscalls.c's   #
#  byte-at-a-time memcpy -- about once per SMLAD.                      #
#                                                                      #
#  CMSIS_NN_UNALIGNED=ON (default) builds CMSIS-NN with                #
#  -munaligned-access. The Cortex-M4 handles unaligned LDR/LDRH/STR in #
#  hardware (CCR.UNALIGN_TRP is 0 out of reset, and nothing here sets  #
#  it). Everything else keeps -mno-unaligned-access. The quantization  #
#  activity measured the difference (5.3x on a conv model).            #
# ------------------------------------------------------------------ #
option(CMSIS_NN_UNALIGNED "Compile CMSIS-NN with -munaligned-access" ON)
add_subdirectory(${CMSIS_NN_DIR} ${CMAKE_BINARY_DIR}/cmsis_nn EXCLUDE_FROM_ALL)
if(CMSIS_NN_UNALIGNED)
    target_compile_options(cmsis-nn PRIVATE -munaligned-access)
endif()
message(STATUS "CMSIS-NN unaligned access: ${CMSIS_NN_UNALIGNED}")

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
# load_method() fails with OperatorMissing (0x14). Both operator libraries
# need it: the portable one (fp32 ops) and the Cortex-M one (int8 ops).
target_link_libraries(firmware.elf PRIVATE
    -Wl,--whole-archive
    executorch_selected_kernels
    cortex_m_ops_lib
    -Wl,--no-whole-archive
    cortex_m_kernels
    cmsis-nn
    portable_kernels
    kernels_util_all_deps
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
