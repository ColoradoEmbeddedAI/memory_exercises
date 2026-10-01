# pruning_lab — Activity companion (PyTorch + ExecuTorch firmware)

Companion project for
[`../lecture5-activity-pruning-handout.md`](../lecture5-activity-pruning-handout.md),
which walks through every step.

It measures what pruning actually changes on the STM32F407:

- **Flash:** the `.pte` size embedded in firmware
- **SRAM:** the planned activation arena ExecuTorch allocates
- **Latency:** `method.execute()` timed with the DWT cycle counter
- **Accuracy:** measured on the laptop, since 10 embedded images can't measure it

It covers three variants of one model. All three are pre-trained and provided in `checkpoints/`.

| ID | Variant |
|---|---|
| A | Dense baseline, Model B2 |
| B | **Unstructured:** 70% of weights zeroed (global L1), fine-tuned, `prune.remove()` |
| C | **Structured:** half the conv filters (smallest L1 norm) **physically removed**: `ModelB2(8, 16)`, fine-tuned |

**Model B2** is the memory-budgeting activity's Model B at 2× width, meaning twice the channels in each conv layer (conv1 8 → 16, conv2 16 → 32). Its max-pool comes before the ReLU:

```
1x32x32 -> Conv(1->16) -> MaxPool -> ReLU -> Conv(16->32) -> MaxPool -> ReLU -> GAP -> FC(32->10)
```

Variant C, `ModelB2(8, 16)`, is exactly Model B's shape. The data is Fashion-MNIST, zero-padded to 32×32.

## Layout

```
pruning_lab/
├── README.md                  this file
├── python/
│   ├── model.py               ModelB2(c1, c2), variant loading, analytic MAC count
│   ├── data.py                Fashion-MNIST download/pad/normalize, fixed 55k/5k/10k split
│   ├── train_baseline.py      variant A (30 epochs, ~10 min on a laptop CPU)
│   ├── make_variants.py       variants B and C from A (~2 min)
│   ├── export_variants.py     .pte per variant + generated/model_pte.h, test_images.h
│   ├── measure_host.py        the laptop-side results table
│   └── memory_calc.py         memory-budgeting calculator, with B2 and C added
├── checkpoints/variant_{A,B,C}.pt   pre-trained (tracked; provided to students)
├── src/main.cpp               firmware: load, 10-image export check, DWT timing
├── src/syscalls.c             newlib stubs (from lab0_hello_world_soln)
├── scripts/run_variant.sh     export + build + flash + capture UART, one variant
├── CMakeLists.txt / CMakePresets.json / arm-none-eabi.cmake
├── stm32f407.ld / startup_stm32f407.s / stm32f407.h
│
├── data/        (generated) Fashion-MNIST download, ~30 MB
├── pte/         (generated) variant_*.pte
├── generated/   (generated) model_pte.h, test_images.h -- the firmware's model
├── results/     (generated) run_variant.sh UART captures
└── build/       (generated) cmake output
```

## Prerequisites

The same setup as every other ExecuTorch project in this course:

- `arm-none-eabi-gcc` 13.x, OpenOCD, CMake, and Ninja
- ExecuTorch **release/1.2**, built for Cortex-M4 at `~/executorch/executorch-build/cortex-m4`, **with this lab's six ops in its op list**. `cmake --preset stm32` checks this; see "ExecuTorch kernels" below if it fails.
- The ExecuTorch Python venv at `~/executorch/et-env`

No `torchvision` is needed; `data.py` reads the raw dataset files itself.

You also need a 3.3 V USB-UART adapter on the Discovery board's P2 header (USART2: PA2 = TX, plus GND), at 115200 8N1, as in the data-movement activity.

## Before class

```bash
source ~/executorch/et-env/bin/activate
cd pruning_lab
python3 python/data.py                          # downloads Fashion-MNIST (~30 MB)
python3 python/export_variants.py --variant A   # creates generated/ so cmake can configure
cmake --preset stm32 && cmake --build --preset stm32
```

A first build that links cleanly means the toolchain and ExecuTorch paths are fine.

## In class

Laptop table (about 1 minute; exports all three variants and evaluates each on the 10k test set):

```bash
python3 python/measure_host.py
python3 python/measure_host.py --keys B     # a checkpoint's tensors, names and sizes
python3 python/memory_calc.py               # Part 2 pair-model estimates (handout Q9)
```

Board, one variant at a time (about 40 s each for A and B, 20 s for C):

```bash
scripts/run_variant.sh A          # UART on /dev/ttyUSB0; override with UART=/dev/ttyUSBn
```

The script does exactly the steps below. Students without the script (or on macOS, where `stty -F` differs) can do them by hand:

```bash
python3 python/export_variants.py --variant A
cmake --build --preset stm32 --target flash
picocom -b 115200 --imap lfcrlf /dev/ttyUSB0      # then reset the board
```

What the board prints (variant A, real capture):

```
--- Pruning lab: STM32F407VG ---
Variant A: dense baseline
Clock                  : 168000000 Hz (HSE -> PLL)
.pte size (Flash)      : 25640 bytes
Planned arena (SRAM)   : 114688 bytes
method_pool used (CCM) : 4257 bytes

=== Sanity check: 10 test images ===
  image 0: label T-shirt/top, board T-shirt/top (= host)
  ...
Board matches host     : 10/10  (export check)
Board matches label    : 10/10  (NOT an accuracy measurement)

=== Latency: method.execute() ===
Runs                   : 1 warm-up
                         10 timed
Cycles min/median/max  :  235234330 / 235234342 / 235234396
Median latency         : 1400.2 ms

--- Done ---
```

`Board matches host` should always be 10/10. Anything less means the export or the firmware is wrong, not that the model is inaccurate.

## Memory budget

| Region | Holds | Size |
|---|---|---:|
| Flash | runtime and kernels, about 143 KiB; the `.pte`, 10.5–25.0 KiB; 10 test images, 40 KiB | ≈195–210 KiB of 1 MiB |
| Main SRAM | `planned_pool`, 112 KiB; `.data`/`.bss`, about 3.5 KiB; stack, the rest (about 12.5 KiB) | 128 KiB |
| CCM | `method_pool` (16 KiB, about 4.3 KiB used); input staging buffer (4 KiB) | 20 KiB of 64 KiB |

`planned_pool` is sized for the largest variant (A and B need 114,688 B), so one build fits every variant. The downside is that `arm-none-eabi-size` reports the same SRAM usage for every variant. The **`Planned arena` line on the UART** is the number that changes.

The linker script asserts that at least 8 KiB of SRAM is left for the stack.

Why the pool comes before the ReLU: with the memory-budgeting activity's original ReLU-then-pool order, B2's planned arena is **131,072 B**. That's all 128 KiB of main SRAM, so the model can't run at all. Pooling first means the 64 KiB conv1 output is consumed once instead of twice, which brings the arena to 114,688 B.

## ExecuTorch kernels

The firmware links the shared `~/executorch/executorch-build/cortex-m4` ExecuTorch tree, like the other ExecuTorch projects in this course. You don't build anything ExecuTorch-specific for this lab, **as long as that tree includes this lab's ops.**

The op kernels compiled into that tree are fixed by the `EXECUTORCH_SELECT_OPS_LIST` it was configured with. This lab's models use six ops (`export_variants.py` prints each variant's list):

```
aten::addmm.out, aten::convolution.out, aten::max_pool2d_with_indices.out,
aten::mean.out, aten::permute_copy.out, aten::relu.out
```

**Checking:** `cmake --preset stm32` checks this for you. It reads the tree's `CMakeCache.txt` and stops with a list of any missing ops. To look yourself:

```bash
grep EXECUTORCH_SELECT_OPS_LIST ~/executorch/executorch-build/cortex-m4/CMakeCache.txt
```

If your tree was set up for the wake-word lab, it already has all six. If it was set up only for the first ExecuTorch examples (`mul`, `add`, …), it won't.

**Rebuilding with a larger op list** (a few minutes). The list below is a superset of both this lab's ops and the wake-word lab's, so other projects that link the same tree keep working. If your current list (from the `grep` above) has other ops, add them too: anything left off is removed from the tree.

```bash
source ~/executorch/et-env/bin/activate

OPS="aten::addmm.out,aten::convolution.out,aten::max_pool2d_with_indices.out,\
aten::mean.out,aten::permute_copy.out,aten::relu.out,\
aten::_native_batch_norm_legit_no_training.out,aten::batch_norm.out,\
aten::adaptive_avg_pool2d.out,aten::view_copy.out,\
dim_order_ops::_clone_dim_order.out,dim_order_ops::_to_dim_order_copy.out,\
aten::mul.out,aten::add.out"

# Reconfigure the existing tree; every other option stays as it was cached.
cmake -S ~/executorch/executorch -B ~/executorch/executorch-build/cortex-m4 \
  -DEXECUTORCH_SELECT_OPS_LIST="$OPS"
cmake --build ~/executorch/executorch-build/cortex-m4 -j$(nproc)
```

Then reconfigure this project so the check runs again: `rm -rf build && cmake --preset stm32`.

The op list is baked in at configure time, so it takes a fresh `cmake -S … -B …` as above; rebuilding alone doesn't pick up a new list. If you don't have a `cortex-m4` tree at all, set one up first with the ExecuTorch setup instructions from the earlier labs, then use the op list above.

If `load_method` still fails with `0x14` (OperatorMissing) on the board, some op isn't in the tree. Compare `export_variants.py`'s printed op list against the tree's `EXECUTORCH_SELECT_OPS_LIST`.

## Regenerating the checkpoints (optional; homework)

```bash
python3 python/train_baseline.py     # A: 30 epochs, Adam 3e-3, cosine schedule
python3 python/make_variants.py      # B and C from A (all fine-tunes: Adam 1e-3, cosine, 5 epochs)
```

Seeds are fixed, but CPU thread scheduling makes PyTorch training not bit-exact across machines. Expect accuracies within a few tenths of a percent of the table above. The board results depend almost entirely on the architecture, not the weights, so those don't move.

## Implementation notes

- **Clock, UART and DWT** are copied from `../../04_memmovement/dma_bandwidth_lab`: HSE → PLL → 168 MHz, 5 flash wait states plus the ART Accelerator, and USART2 at 115200 8N1.
- **ExecuTorch bring-up** follows `lab0_hello_world_soln/src/main.cpp`. The firmware reads the planned arena size from `MethodMeta` at load time and refuses to run if it doesn't fit `planned_pool`.
- **1 warm-up and 10 timed runs, not 5 + 50.** A full-width inference takes 1.4 s (the portable conv kernel costs about 177 cycles/MAC). Run-to-run spread is about 70 cycles out of 235 million, so extra runs only add waiting time.
- **`syscalls.c`'s `memcpy`/`memset` are byte-at-a-time** (lab0's workaround for unaligned-access faults in the FlatBuffer parser). That's a fixed cost in every variant, and none of it is inside the timed region except whatever the kernels themselves copy.
- **Test images:** the first test-set image of each class, embedded as normalized fp32 in `generated/test_images.h`, 40 KiB of Flash.
