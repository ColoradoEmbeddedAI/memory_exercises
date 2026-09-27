# cortex_m_perf_lab — Activity companion firmware

Buildable, flashable STM32F407VG firmware for
[`../lecture3-activity-fpu-simd-handout.md`](../lecture3-activity-fpu-simd-handout.md)
(hand out to students) /
[`../lecture3-activity-fpu-simd-answers.md`](../lecture3-activity-fpu-simd-answers.md)
(instructor reference, with every question answered against real captured
hardware output).

Each build times **one** dot-product kernel with the Cortex-M4's DWT
cycle counter and prints one result over USART2. Four CMake presets
select the kernel and float ABI:

| Preset | Kernel | Float ABI | Builds into |
|---|---|---|---|
| `f32-hard` | float32 dot product | hard (`-mfpu=fpv4-sp-d16 -mfloat-abi=hard`) | `build/f32-hard/` |
| `f32-soft` | float32 dot product | soft (`-mfloat-abi=soft`, no FPU instructions) | `build/f32-soft/` |
| `q15-scalar` | int16 (q15) dot product, one MAC per instruction | hard | `build/q15-scalar/` |
| `q15-simd` | int16 (q15) dot product using `SMLAD`, two MACs per instruction | hard | `build/q15-simd/` |

The activity runs each configuration twice: first under **Renode** (for
an exact instruction count), then on the **real board** (for real
cycles). See "Running under Renode" below for what the simulator can and
can't tell you.

This is a complete, bare-metal project matching the rest of this repo's
convention: direct register access, no HAL, no CMSIS, `cmake --preset
...` (same layout as
[`../../04_memmovement/dma_bandwidth_lab/`](../../04_memmovement/dma_bandwidth_lab/)
and [`../../01_memtypes/memory_benchmark/`](../../01_memtypes/memory_benchmark/)).

## Implementation notes

- **Clock, UART and DWT** plumbing is copied from `dma_bandwidth_lab`:
  HSE → PLL → **168 MHz**, 5 flash wait states + ART Accelerator,
  USART2 at 115200 8N1, direct register access via `stm32f407.h`, and
  inline `cpsid`/`cpsie` asm around each timed region.
- **`SMLAD`** comes from GCC's ACLE intrinsic `__smlad()` in
  `<arm_acle.h>`. It's the same instruction as CMSIS's `__SMLAD()`, with
  no CMSIS headers needed.
- **No `printf`, and no float arithmetic in the reporting path.**
  `report()` formats MACs/cycle etc. with integer fixed-point math. In the
  soft-float build every float operation is a library call, and the
  reporting code should not be one of the things being emulated. The only
  float operation outside the timed region is the one sanity-check
  conversion of the float32 result.
- **One kernel per build.** `KERNEL` (a CMake cache variable, set by the
  preset) becomes `KERNEL_F32` / `KERNEL_Q15_SCALAR` / `KERNEL_Q15_SIMD`,
  and `src/main.c` compiles only that kernel. Each kernel is marked
  `noinline, noclone`, so it appears in the disassembly as one function
  under its source name.
- **Result checks.** The float32 build prints its dot product ×1000 as an
  integer (expect ~1024000). Each q15 build checks its kernel against an
  untimed reference loop and prints `match` or `MISMATCH!`. A SIMD kernel
  that's fast but wrong is not a speedup.
- **Build-config line.** It reports the float ABI actually compiled
  (`hard` / `softfp` / `soft`, from `__ARM_PCS_VFP` and `__ARM_FP`), so a
  wrongly flashed build is obvious.

## Building

The float ABI is compile-time, and it has to match across **every**
object in the link: `main.c`, the startup assembly, and the newlib/libgcc
variant GCC picks. So it is set in the toolchain file
(`arm-none-eabi.cmake`, `FLOAT_ABI` variable), and each preset gets its
own subdirectory of `build/`.

```bash
cmake --preset f32-hard   && cmake --build --preset f32-hard
cmake --preset f32-soft   && cmake --build --preset f32-soft
cmake --preset q15-scalar && cmake --build --preset q15-scalar
cmake --preset q15-simd   && cmake --build --preset q15-simd
```

All presets compile at **`-O2`** (`CMAKE_BUILD_TYPE=RelWithDebInfo`).
Don't switch to `MinSizeRel`. CMake appends the build type's flags after
`CMAKE_C_FLAGS`, so its `-Os` would silently override any `-O2` set in
`CMakeLists.txt`, and this lab's numbers depend on the optimization
level.

The toolchain file also supports `FLOAT_ABI=softfp` (FPU instructions,
float arguments in core registers). There's no preset for it, but it can
be tried by hand, e.g. `cmake --preset f32-hard -B build/f32-softfp
-DFLOAT_ABI=softfp`.

## Running under Renode

```bash
scripts/renode_run.sh f32-hard      # or f32-soft, q15-scalar, q15-simd
```

The script assumes Renode's portable release at `~/renode_portable`,
with its venv at `~/renode_portable/renode-venv/` (override the
directory with `RENODE_HOME`). It uses
`platforms/stm32f4_discovery_full.repl`, copied from `memory_benchmark`,
which adds the DWT, CCM RAM, and the correct SRAM/Flash sizes to Renode's
stock board. A run takes about 10 seconds.

Renode's DWT model derives `CYCCNT` from simulated time. The script sets
the simulated CPU to 168 MIPS, so the counter advances by exactly one per
executed instruction. Every "cycles" figure the firmware prints under
Renode is therefore an **instruction count**, and the script ends with
the number to record:

```
>>> Instructions per MAC (Renode, exact): 8.00
```

**What Renode is good for here:** exact instruction counts per MAC,
including the instructions executed inside the soft-float library calls;
checking that each build runs and computes the right answer; perfect
repeatability; and working without a board.

**What it can't do:** Renode doesn't model Cortex-M4 timing (2-cycle
loads, the 3-cycle `VFMA`, branch refill, Flash wait states, the ART
cache, bus contention), so it cannot give real cycles. The activity
uses that gap on purpose: board cycles ÷ Renode instructions = **CPI**,
the cost of the hardware's timing.

## Running on the board

Connect a 3.3V USB-UART adapter to the Discovery board's P2 header
(adapter RX → PA2 = USART2 TX, adapter GND → GND). The firmware prints
once, at boot, so in a **second terminal**, start picocom *before*
flashing, logging to a file (exit with `Ctrl-A` then `Ctrl-X`):

```bash
picocom -b 115200 --imap lfcrlf --logfile f32-hard.txt /dev/ttyUSB0
```

Then flash from the first terminal:

```bash
cmake --build build/f32-hard --target flash
```

Press the board's RESET button with picocom running to rerun. If
`/dev/ttyUSB0` doesn't exist, check `ls /dev/ttyUSB* /dev/ttyACM*`. If it
exists but won't open, add yourself to the `dialout` group.

Output looks like this. The labels and formatting are real; the numbers
are placeholders:

```
--- Cortex-M Performance Lab: STM32F407VG ---
Clock = 168000000 Hz (HSE -> PLL)

Build config: FLOAT_ABI=hard   -- FPU instructions ENABLED, float args in FPU registers
Kernel:       q15 SIMD (SMLAD) dot product (1024 elements x 500 reps)

q15 SIMD (SMLAD)     :   <cycles> cycles |   <time> us | <x.xxx> MACs/cycle | <y.yy> cycles/MAC
  (result = 2556800, reference = 2556800  -- match)

--- Done ---
```

`cycles/MAC` is the inverse of `MACs/cycle`, printed as well because the
soft-float result is around 0.01 MACs/cycle, where three decimals of
MACs/cycle hide a lot. PD12 (green LED) blinks afterward as a heartbeat
for anyone without a UART adapter handy.

## Looking at the inner loops

```bash
cmake --build build/f32-hard --target disasm-kernel   # this build's kernel only
cmake --build build/f32-hard --target disasm          # everything, with source
```

The inner loops GCC 13.2 (`arm-none-eabi-gcc 13.2.rel1`) generates at
`-O2`, for reference. Your compiler version may schedule them slightly
differently:

```
bench_dot_f32 (f32-hard) -- 8 instructions, 1 MAC per iteration
  add.w    ip, r2, r3, lsl #2      ; &fa[i]
  add.w    r1, r0, r3, lsl #2      ; &fb[i]
  vldr     s13, [ip]
  vldr     s14, [r1]
  adds     r3, #1
  cmp.w    r3, #1024
  vfma.f32 s15, s13, s14           ; the one "useful" instruction
  bne.n    <loop>

bench_dot_f32 (f32-soft) -- 10 instructions + two library calls, 1 MAC
  ldr.w    r0, [r7, r4, lsl #2]
  ldr.w    r1, [r6, r4, lsl #2]
  bl       __aeabi_fmul            ; call libgcc software multiply: 32 instructions
  mov      r1, r0
  mov      r0, r5
  bl       __addsf3                ; call libgcc software add: 48 instructions
  adds     r4, #1
  cmp.w    r4, #1024
  mov      r5, r0
  bne.n    <loop>

bench_dot_q15_scalar (q15-scalar) -- 6 instructions, 1 MAC per iteration
  ldrh.w   ip, [r2, r3, lsl #1]
  ldrh.w   lr, [r0, r3, lsl #1]
  adds     r3, #1
  cmp.w    r3, #1024
  smlabb   r1, lr, ip, r1          ; DSP-extension MAC: one 16-bit x 16-bit multiply
  bne.n    <loop>

bench_dot_q15_simd (q15-simd) -- 5 instructions, 2 MACs per iteration
  ldr.w    r2, [r3], #4            ; two int16 elements per load
  ldr.w    r4, [r0], #4
  cmp      r5, r3
  smlad    r1, r2, r4, r1          ; two 16-bit x 16-bit multiplies (2 MACs), one instruction
  bne.n    <loop>
```

These match Renode's instruction counts: 8, 10 + 80, 6, and 5 per 2 MACs
(2.5 per MAC). Things worth noticing:

- **The "scalar" q15 loop already uses a DSP instruction.** GCC chose
  `SMLABB` (signed multiply-accumulate of the bottom 16-bit half of
  one register by the bottom 16-bit half of another: one multiply, one
  MAC) on its own.
  So the q15 comparison is not really "plain C vs DSP". It compares one
  MAC per instruction (plus two halfword loads) against two MACs per
  instruction (plus two word loads).
- **Loop overhead dominates every kernel.** The `volatile` arrays that
  keep the benchmark honest also stop GCC from unrolling the loops or
  merging loads. In the hard-float loop, only 1 of 8 instructions is the
  MAC.

## Memory budget

| Buffer | Built in | Region | Size |
|---|---|---|---|
| `fa`, `fb` | `f32-*` | main SRAM | 4 KB each (`N_FLOAT` = 1024 floats) |
| `qa`, `qb` | `q15-*` | main SRAM | 2 KB each (`N_Q15` = 1024 int16) |

All four builds use 1024-element vectors × 500 repetitions, so each
measurement covers the same 512,000 MACs.

At most 8 KB of SRAM, with no CCM RAM or DMA. The operands sit in
zero-wait-state SRAM on purpose. This lab measures the compute side
(this lecture); the effect of operands arriving slowly is a future lecture's subject.
Instruction fetch still comes from Flash through the ART Accelerator
(covered previously). The loops are small enough to stay in its cache after the
first iteration.

## Troubleshooting

- **No output in picocom.** It was probably started after flashing.
  The firmware prints once, at boot, so press RESET with picocom
  running.
- **`Build config:` or `Kernel:` isn't what you expected.** You flashed
  another preset's binary. Each preset's `flash` target flashes *its own*
  build directory.
- **`renode_run.sh` says renode not found.** Check `~/renode_portable`
  exists, or set `RENODE_HOME`. If Renode itself fails, the script prints
  its log.
- **Link errors mentioning `uses VFP register arguments` / `does not`.**
  Something in the link was compiled with a different float ABI (for
  example, an object or library you added by hand). Every object must
  match. That's why the ABI lives in the toolchain file.
- **`MISMATCH!` on a q15 result line.** The kernel computed the wrong
  answer, for the SIMD kernel most likely because `qa`/`qb` lost their
  4-byte alignment. Its timing means nothing until this is fixed.
