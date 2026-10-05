# quantization_lab — Activity companion (PyTorch PT2E + ExecuTorch Cortex-M/CMSIS-NN firmware)

Companion project for
[`../lecture-6-activity-quantization-handout.md`](../lecture-6-activity-quantization-handout.md),
which walks through every step.

It measures what int8 quantization actually changes on the STM32F407:

- **Flash:** the `.pte` size embedded in firmware, and what's in it
- **SRAM:** the planned activation arena, plus the kernels' scratch memory
- **Latency:** `method.execute()` timed with the DWT cycle counter, plus a per-op profile
- **Accuracy:** on 200 embedded test images on the board, checked image by image against the laptop; on all 10,000 test images on the laptop

It covers two variants of Model B2 from the pruning activity, each at two precisions. Both checkpoints are pre-trained and provided in `checkpoints/`, copied from `../../05_pruning/pruning_lab/`.

| Build | Variant | Precision |
|---|---|---|
| `A-fp32` | A: dense, `ModelB2(16, 32)` | fp32, portable kernels (the pruning lab's export) |
| `A-int8` | A | int8, PTQ with `CortexMQuantizer`, CMSIS-NN kernels |
| `C-fp32` | C: structured-pruned, `ModelB2(8, 16)` | fp32 |
| `C-int8` | C | int8 |

```
1x32x32 -> Conv(1->16) -> MaxPool -> ReLU -> Conv(16->32) -> MaxPool -> ReLU -> GAP -> FC(32->10)
```

That's the pruning lab's order (pool before ReLU). int8 builds default to conv → ReLU → pool instead: same weights, same function, but it quantizes correctly. Handout Part A, Step 3 shows what goes wrong otherwise.

## Layout

```
quantization_lab/
├── README.md                  this file
├── python/
│   ├── model.py               ModelB2(c1, c2, order), variant loading, MAC count
│   ├── data.py                Fashion-MNIST; the 200 embedded test images; calibration sets
│   ├── quantize.py            the PT2E -> CortexMQuantizer -> CMSIS-NN -> .pte pipeline
│   ├── export_model.py        one build -> pte/, generated/model_pte.h, test_images.h, host preds
│   ├── compare_preds.py       board UART capture vs host predictions, logits side by side
│   ├── eval_quant.py          int8 accuracy on 10k test images for calibration choices (Part B1)
│   ├── quant_sim.py           backend-independent fake-quant simulator (Parts B2, B3, C1)
│   ├── inspect_graph.py       zero points, lowered ops, memory plan, dead constants (B4, C2)
│   ├── memory_calc.py         memory-budgeting calculator, --int8 for 1 byte/element
│   └── train_baseline.py      from the pruning lab; this lab uses its evaluate()
├── checkpoints/variant_{A,C}.pt   pre-trained (tracked; provided to students)
├── src/main.cpp               firmware: load, 200-image predictions, DWT timing, per-op profile
├── src/cortex_m_avgpool_fix.cpp   replaces a buggy upstream kernel (see "Known upstream issues")
├── src/syscalls.c             newlib stubs (from lab0_hello_world_soln)
├── scripts/run_build.sh       export + build + flash + capture UART + compare, one build
├── CMakeLists.txt / CMakePresets.json / arm-none-eabi.cmake
├── stm32f407.ld / startup_stm32f407.s / stm32f407.h
│
├── data/        (generated) Fashion-MNIST download, ~30 MB
├── pte/         (generated) <build>.pte
├── generated/   (generated) model_pte.h, test_images.h -- the firmware's model and images
├── results/     (generated) <build>.txt UART captures, host_<build>.json host predictions
└── build/       (generated) cmake output
```

## Prerequisites

Everything from the pruning lab (`arm-none-eabi-gcc` 13.x, OpenOCD, CMake, Ninja, the ExecuTorch venv at `~/executorch/et-env`, a 3.3 V USB-UART adapter on PA2), plus three one-time additions for the Cortex-M backend. Together they take about 15 minutes.

### 1. The TOSA serialization package (Python)

The Cortex-M quantizer imports code shared with ExecuTorch's Arm backend, which imports `tosa_serializer`. It isn't on PyPI; ExecuTorch's `examples/arm/setup.sh` builds it from source. Just that piece:

```bash
source ~/executorch/et-env/bin/activate
cd ~/executorch
git clone https://git.gitlab.arm.com/tosa/tosa-tools.git
cd tosa-tools && git checkout v2025.11.2
# scikit-build-core 0.12+ rejects this package's pyproject.toml; pin the build backend.
echo "scikit-build-core==0.11.6" > /tmp/skb.txt
PIP_CONSTRAINT=/tmp/skb.txt CMAKE_POLICY_VERSION_MINIMUM=3.5 BUILD_PYBIND=1 \
    pip install --no-dependencies ./serialization
python3 -c "from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer; print('ok')"
```

### 2. CMSIS-NN source

```bash
cd ~/executorch
git clone --depth 1 -b v7.0.0 https://github.com/ARM-software/CMSIS-NN.git
```

v7.0.0 is the version ExecuTorch release/1.2's Cortex-M backend pins. The ExecuTorch build below uses it, and so does this lab's firmware, which compiles CMSIS-NN itself (see "Why this lab compiles CMSIS-NN" below).

### 3. ExecuTorch with CMSIS-NN

A **separate** ExecuTorch tree, `~/executorch/executorch-build/cortex-m4-cmsis`, with the Cortex-M backend turned on. The shared `cortex-m4` tree that other labs use is left alone.

```bash
source ~/executorch/et-env/bin/activate
cd ~/executorch

OPS="aten::addmm.out,aten::convolution.out,aten::max_pool2d_with_indices.out,\
aten::mean.out,aten::permute_copy.out,aten::relu.out,\
aten::_native_batch_norm_legit_no_training.out,aten::batch_norm.out,\
aten::adaptive_avg_pool2d.out,aten::view_copy.out,\
dim_order_ops::_clone_dim_order.out,dim_order_ops::_to_dim_order_copy.out,\
aten::mul.out,aten::add.out"

cmake -S executorch -B executorch-build/cortex-m4-cmsis -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$HOME/executorch/executorch/cmake/toolchains/arm-none-eabi.cmake \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DEXECUTORCH_BUILD_CORTEX_M=ON \
  -DCMSIS_NN_LOCAL_PATH=$HOME/executorch/CMSIS-NN \
  -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_ENABLE_LOGGING=OFF \
  -DEXECUTORCH_PAL_DEFAULT=posix \
  -DEXECUTORCH_BUILD_CPUINFO=OFF -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF \
  -DEXECUTORCH_XNNPACK_ENABLE_KLEIDI=OFF -DEXECUTORCH_XNNPACK_SHARED_WORKSPACE=OFF \
  -DMAX_KERNEL_NUM=128 \
  -DEXECUTORCH_SELECT_OPS_LIST="$OPS"
cmake --build executorch-build/cortex-m4-cmsis -j$(nproc)
```

What each unusual option is for:

| Option | Why |
|---|---|
| `EXECUTORCH_BUILD_CORTEX_M=ON` | builds `libcortex_m_ops_lib.a` (the `cortex_m::` ops) and `libcortex_m_kernels.a` (their CMSIS-NN wrappers) |
| `CMSIS_NN_LOCAL_PATH` | without it, ExecuTorch fetches CMSIS-NN into the build directory and the generate step fails ("INTERFACE_INCLUDE_DIRECTORIES … prefixed in the build directory") |
| `CPUINFO`, `PTHREADPOOL`, … `=OFF` | they default to ON when configured from scratch and don't compile for bare metal; the shared `cortex-m4` tree has them off |
| `MAX_KERNEL_NUM=128` | the kernel registry is a static array; the default (2,000 kernels, 24 KB of `.bss`) doesn't fit next to the 112 KB arena. 128 matches the shared tree |
| the op list | every portable op this lab's builds use. The fp32 builds need all of the first six; the int8 builds need none of them except `aten::relu` in the pool-relu experiment. The rest keep the list a superset of the other labs' |

Do the configure with the venv active: ExecuTorch's code generation runs Python, and the system Python doesn't have ExecuTorch's packages.

`cmake --preset stm32` checks that the tree has the Cortex-M backend and these ops, and stops with a pointer back here if not.

## Prep

```bash
source ~/executorch/et-env/bin/activate
cd quantization_lab
python3 python/data.py                                     # Fashion-MNIST (~30 MB)
python3 python/export_model.py --variant A --precision int8   # creates generated/ so cmake can configure
cmake --preset stm32 && cmake --build --preset stm32
```

A first build that links cleanly means the toolchain, ExecuTorch and CMSIS-NN paths are fine. Then run one build on the board (`scripts/run_build.sh A int8`, under 10 s) to check the UART.

## Exercise

Board, one build at a time:

```bash
scripts/run_build.sh A int8                     # UART on /dev/ttyUSB0; override with UART=/dev/ttyUSBn
scripts/run_build.sh A int8 --order pool-relu   # any extra arguments go to export_model.py
```

Run times: A-fp32 about 50 s (1.4 s per inference: 20 predictions, 11 timing runs, one profiled run), C-fp32 about 20 s, int8 builds under 10 s.

The script does the steps below, then runs `compare_preds.py`. By hand (or on macOS, where `stty -F` differs):

```bash
python3 python/export_model.py --variant A --precision int8
cmake --build --preset stm32 --target flash
picocom -b 115200 --imap lfcrlf /dev/ttyUSB0      # then reset the board; save the output to results/A-int8.txt
python3 python/compare_preds.py A-int8
```

What the board prints (A-int8, real capture):

```
--- Quantization lab: STM32F407VG ---
Build A-int8: dense baseline, int8, relu-pool, calib 200
Clock                  : 168000000 Hz (HSE -> PLL)
.pte size (Flash)      : 18104 bytes
Planned arena (SRAM)   : 20480 bytes
Input                  : float32 1x1x32x32, NHWC (channels-last)
method_pool used (CCM) : 4460 bytes

=== Predictions: 200 test images ===
preds   0: 61361758124391294882949913227952845839630048742552
preds  50: 97513008776549826708112038351728677313063323612505
preds 100: 97260187589335537741747754302931412861100943748282
preds 150: 94066809474350793412481824765009415466510995460720
Board accuracy         : 172/200 correct
Board matches host     : 198/200
  image 33: board Coat, host Shirt
    logits: -2.328 -7.682 -5.587 -1.629 -1.397 -11.872 -1.397 -20.950 -8.846 -7.682
  image 87: board Dress, host Pullover
    logits: -3.492 -1.862 -0.233 0.233 -0.233 -12.337 -1.164 -22.580 -8.147 -8.147

=== Latency: method.execute() ===
Runs                   : 1 warm-up
                         10 timed
Cycles min/median/max  :    6530351 /   6531092 /   6531092
Median latency         : 38.87 ms

=== Per-op profile: Method::step() ===
   0       67738  cortex_m::quantize_per_tensor
   1     2382305  cortex_m::quantized_depthwise_conv2d
   2      218722  cortex_m::quantized_max_pool2d
   3     3738633  cortex_m::quantized_conv2d
   4       96690  cortex_m::quantized_max_pool2d
   5       22559  cortex_m::quantized_avg_pool2d
   6        3116  cortex_m::quantized_linear
   7        1386  cortex_m::dequantize_per_tensor
  total    6531149
Kernel scratch (peak)  : 576 bytes (temp_pool, CCM)

--- Done ---
```

`Board matches host` should be 192/200 or better for an int8 build: the CMSIS-NN kernels round a few intermediate values differently from PyTorch, which flips near-tied images. For fp32 builds it should be 20/20. Anything far below means a bug: see handout Part A, Step 3.

Laptop only:

```bash
python3 python/inspect_graph.py --variant A     # zero points, lowered ops, memory plan, dead constants
python3 python/eval_quant.py --sweep            # Part B1 (~30 s)
python3 python/quant_sim.py --compare granularity
python3 python/quant_sim.py --compare range
python3 python/quant_sim.py --clip-curve conv2 --bits 4
python3 python/quant_sim.py --sensitivity
```

## Memory budget

| Region | Holds | Size |
|---|---|---:|
| Flash | runtime, portable and CMSIS-NN kernels, about 222 KiB; the `.pte`, 7.6–25.0 KiB; 200 test images as uint8, 200 KiB | ≈ 430–448 KiB of 1 MiB |
| Main SRAM | `planned_pool`, 112 KiB; `.data`/`.bss`, about 3.8 KiB; stack, the rest (about 12 KiB) | 128 KiB |
| CCM | `method_pool` 16 KiB (about 4.4 KiB used); `temp_pool` 8 KiB (576 B used); input staging buffer 4 KiB | 28 KiB of 64 KiB |

As in the pruning lab, `planned_pool` is sized for the largest build (A-fp32, 114,688 B), so `arm-none-eabi-size` reports the same SRAM for every build. The `Planned arena` line on the UART is the number that changes.

`temp_pool` is new: CMSIS-NN's conv kernels unpack each input patch into an int16 buffer before the SMLAD loop, and the avgpool kernel needs one int32 accumulator per channel. ExecuTorch gives kernels a temp allocator for this and resets it after every kernel. That scratch isn't part of the planned arena, so the firmware reports its peak separately.

The test images are stored as uint8 (1 KiB each, not 4 KiB as floats) and converted with a 256-entry lookup table that PyTorch computed. The board's input tensor is bit-for-bit the one the laptop used, so any host/board disagreement comes from the model, not the input.

## Why this lab compiles CMSIS-NN

Every project in this course compiles with `-mno-unaligned-access`, the workaround for the FlatBuffer parser's unaligned-access faults (`src/syscalls.c` also replaces `memcpy` with a byte-at-a-time version for the same reason). CMSIS-NN reads four int8 values at a time with `memcpy(&word, ptr, 4)`. When unaligned access is allowed, GCC compiles that into a single `LDR`. Under `-mno-unaligned-access`, GCC can't assume the pointer is aligned, so it emits a call to `memcpy`, here the byte-at-a-time one: 11 calls alongside the 12 `SMLAD`s of the conv inner kernel (`arm_nn_mat_mult_kernel_s8_s16`).

So the lab builds CMSIS-NN from `~/executorch/CMSIS-NN` with `-munaligned-access` (`CMSIS_NN_UNALIGNED=ON`, the default). The Cortex-M4 handles unaligned `LDR`/`LDRH`/`STR` in hardware: `CCR.UNALIGN_TRP` is 0 out of reset, and nothing here sets it. Everything else keeps `-mno-unaligned-access`. The copy of CMSIS-NN inside the ExecuTorch tree was compiled with the toolchain file's flags, so it has the slow reads, and the lab doesn't link it.

The difference, A-int8 on the board:

| `CMSIS_NN_UNALIGNED` | conv2 cycles | cycles/MAC | Total | Latency |
|---|---:|---:|---:|---:|
| OFF (like the ExecuTorch tree) | 30,922,141 | 26.2 | 34,581,782 | 205.8 ms |
| ON (default) | 3,738,633 | 3.17 | 6,531,092 | 38.9 ms |

To compare yourself (handout Part A, Step 6):

```bash
cmake --preset stm32 -DCMSIS_NN_UNALIGNED=OFF && scripts/run_build.sh A int8
cmake --preset stm32 -DCMSIS_NN_UNALIGNED=ON  && scripts/run_build.sh A int8
```

## Known upstream issues (ExecuTorch release/1.2, Cortex-M backend)

The backend is marked WIP upstream and is tested on Cortex-M55 simulators. Three things don't work as-is on a Cortex-M4, and the lab works around each one.

1. **`quantized_avg_pool2d` fails on Cortex-M4 (error 0x1, Internal).** The kernel calls CMSIS-NN's `arm_avgpool_s8` without a scratch buffer. On the M55 (MVE) build of CMSIS-NN the function needs none; on the M4 (DSP) build it needs `channels × 4` bytes and returns an argument error. Every model with an average pool (Model B2's GAP) fails. **Workaround:** `src/cortex_m_avgpool_fix.cpp` is a copy of the kernel that allocates the buffer from the temp allocator. It defines the same function, so the linker takes it instead of the original in `libcortex_m_kernels.a`. Nothing in `~/executorch` is modified.
2. **A ReLU that doesn't directly follow a conv or linear stays a portable `aten::relu` on int8 data**, and the portable kernel clamps at 0 instead of the zero point. In the pool-then-ReLU order, that's wrong: 23.5% accuracy on the board. **Workaround:** int8 builds use conv → ReLU → pool, where the quantizer fuses ReLU into the conv. This is handout Part A, Step 3, on purpose.
3. **The lowering leaves dead constants in the `.pte`**: the original NCHW int8 weights (only the NHWC copies are used), the per-channel scale and zero-point tensors (int64 zero points), and the FC's original bias (it's folded into the kernel sum). That's 5,752 B of A-int8's 18,104 B. **Workaround (optional):** `export_model.py --strip-unused`. Off by default, so students see the toolchain's real output (handout Q1).

The firmware also prints which instruction failed whenever `execute()` fails. That's how issue 1 was found.

## Implementation notes

- **Clock, UART, DWT and ExecuTorch bring-up** are the pruning lab's, unchanged.
- **The input stays fp32.** Every build takes a 1×1×32×32 float tensor; int8 builds quantize it with their first op (`cortex_m::quantize_per_tensor`, 67,738 cycles) and dequantize the logits at the end. The firmware reads the input's dim order from the program (NHWC for int8, NCHW for fp32). With one channel those are the same bytes in memory.
- **1 warm-up and 10 timed runs**, as in the pruning lab. int8 spread is under 1,000 cycles out of 6.5 million.
- **The per-op profile** runs one more inference with `Method::step()` (an ExecuTorch API marked experimental), timing each instruction. The sum is within 100 cycles of `execute()`.
- **fp32 builds classify 20 images, int8 builds 200.** fp32 A takes 1.4 s per image; 200 would take almost 5 minutes. fp32 accuracy comes from the laptop, and fp32 board/host agreement was 20/20 on every run.
- **Fixed calibration set:** the first 200 images of the training split (`data.py` `calibration_set()`), the same split the checkpoints were trained on. Seeds are fixed, so exports are reproducible.
