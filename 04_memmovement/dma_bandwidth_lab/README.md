# dma_bandwidth_lab — Activity companion firmware

Buildable, flashable STM32F407VG firmware for
[`../lecture4-activity-dma-bandwidth-handout.md`](../lecture4-activity-dma-bandwidth-handout.md)
(hand out to students) /
[`../lecture4-activity-dma-bandwidth-answers.md`](../lecture4-activity-dma-bandwidth-answers.md)
(instructor reference, with every task answered against real captured
hardware output).

Measures, using the Cortex-M4's DWT cycle counter:

- **Part 1 — DMA overlap:** a blocking CPU copy-then-compute sequence
  vs. a DMA-driven copy running concurrently with the same compute work.
- **Part 2 — Bus contention:** the same CPU read-modify-write workload,
  with and without a background DMA transfer active, once with the
  workload buffer in main SRAM and once in CCM RAM.

Prints results over USART2. This supersedes the old top-level
`dma_bandwidth_lab.c` sketch with a complete, bare-metal project matching
the rest of this repo's convention: direct register access, no HAL, no
CMSIS, `cmake --preset stm32` (same layout as
[`../../01_memtypes/memory_benchmark/`](../../01_memtypes/memory_benchmark/)).

## What changed from the old HAL sketch

- `HAL_Init()`/`SystemClock_Config()`/CMSIS `CoreDebug`/`DWT` structs /
  `__disable_irq()`/`__enable_irq()` intrinsics → direct register access
  via `stm32f407.h` and inline `cpsid`/`cpsie` asm — copied almost
  verbatim from `memory_benchmark`'s `clock_init()`/DWT/UART plumbing,
  which already solved this for a bare-metal STM32F407 project in this
  repo.
- `DMA_HandleTypeDef` + `HAL_DMA_Init()`/`HAL_DMA_Start()`/
  `HAL_DMA_PollForTransfer()` → `dma2_clock_init()` +
  `dma_m2m_start()`/`dma_m2m_done()`/`dma_m2m_wait()` in `src/main.c`,
  driving `DMA2_Stream0`'s registers directly (see `stm32f407.h`'s "DMA
  (memory-to-memory)" section for the register map and bit definitions,
  and the comment above `dma_m2m_start()` for how each HAL call maps to
  raw register writes). Same constraints as the HAL version: DMA2 only
  (DMA1 has no M2M path on the F4), FIFO mode required (M2M disallows
  direct mode), circular mode not permitted.
- `printf` → this project doesn't retarget `printf` at all (see
  `syscalls.c`); `uart_puts()`/`uart_put_uint()`/`uart_put_signed_pct1()`
  in `src/main.c` format output by hand, the same reasoning as
  `memory_benchmark`'s `report()`.
- Clock: HSE → PLL → **168 MHz**, identical to `memory_benchmark`, for
  the same reason (this course's simpler 16 MHz HSI-only demos aren't
  representative of a real deployment clock, and this activity's cycle
  counts should be comparable to Topic 1's benchmark numbers).

## A bug this port found by actually running on hardware

The original HAL sketch's `blocking_copy_then_compute()` and
`dma_overlap_copy_and_compute()` each wrote out their own copy of the
compute loop (`for (r...) for (i...) compute_buf[i] = compute_buf[i] +
1u;`), textually identical to the loop inside `run_workload()` that
`compute_alone()` calls. At `-O2`, GCC compiled the two copies
differently: `run_workload()`'s loop got a tight post-increment
addressing mode, while the duplicated loop — inlined into `main()`
alongside a `memcpy()` call and IRQ-disable bookkeeping — kept an
explicit array-index register and compiled one instruction longer per
iteration. That ~2 cycles/iteration difference, over 204,800 iterations
per measurement, added roughly 400,000 cycles of pure measurement
artifact to `blocking` and `overlap` relative to the `compute_alone`
reference they're supposed to be compared against — enough to make the
overlap benefit from Part 1 nearly disappear in the printed numbers, for
a reason that had nothing to do with DMA or bus contention.

This version factors the compute loop into one `compute_workload()`
function that all three measurements call (see the comment above it in
`src/main.c`), so they compile identically and are actually comparable.
**The lesson generalizes past this one lab:** two textually-identical
loops are not guaranteed to compile to the same code once surrounding
context differs, and a "why doesn't this match the simple model"
investigation should check the disassembly before concluding the
hardware (or the theory) is what's misbehaving. See the answer key for
the before/after numbers this produced on real hardware.

## Building and flashing

```bash
cmake --preset stm32
cmake --build --preset stm32
cmake --build build --target flash
```

Then connect a 3.3V USB-UART adapter to the Discovery board's P2 header
(USART2: PA2 = TX) and open a terminal at 115200 8N1:

```bash
picocom -b 115200 --imap lfcrlf /dev/ttyUSB0
```

Expect output in this shape (labels and formatting are real; the numeric
columns are **not** — see the handout for how to form a prediction before
running, and the answer key for real captured numbers):

```
--- Data Movement & Bandwidth Lab: STM32F407VG ---
Clock = 168000000 Hz (HSE -> PLL)

=== Part 1: DMA Overlap ===
Copy alone                          :    <cycles> cycles
Compute alone                       :    <cycles> cycles
Naive sum prediction (copy+compute) :    <cycles> cycles
Blocking, sequential                :    <cycles> cycles
Overlapped (DMA copy || compute)    :    <cycles> cycles

=== Part 2: Bus Contention ===
Background DMA duration (alone)     :    <cycles> cycles

SRAM workload, DMA idle    :    <cycles> cycles
SRAM workload, DMA active  :    <cycles> cycles  (<+/-X.X>%)
CCM  workload, DMA idle    :    <cycles> cycles
CCM  workload, DMA active  :    <cycles> cycles  (<+/-X.X>%)

--- Done ---
```

PD12 (green LED) blinks afterward as a heartbeat for anyone without a
UART adapter handy.

```bash
cmake --build build --target disasm   # optional: full disassembly
```

## `Background DMA duration (alone)` — for the tuning check

This line (not present in the old HAL sketch) exists specifically to
support the handout's Part 2 tuning check: it prints how long the background transfer
takes running by itself, right before the four workload measurements
that need it to still be running. Compare it against the `SRAM workload,
DMA idle` line printed immediately after — if the background duration
isn't comfortably larger than the workload duration, you'll also see a
`[warning] background DMA finished before the workload did` line, and
`N_BG` (or `N_WORKLOAD_REPS`) needs retuning (see the handout).

## SRAM budget

This lab's buffers, and where they land:

| Buffer | Region | Size |
|---|---|---|
| `src_buf`, `dst_buf` | main SRAM | 8 KB each (`N_XFER` = 2048 words) |
| `compute_buf` | main SRAM | 2 KB (`N_COMPUTE` = 512 words) |
| `bg_src`, `bg_dst` | main SRAM | ~46.9 KB each (`N_BG` = 12000 words) |
| `workload_sram` | main SRAM | 2 KB (`N_WORKLOAD` = 512 words) |
| `workload_ccm` | **CCM RAM** | 2 KB |

Main-SRAM total is about 116 KB of the F407VG's 128 KB, leaving headroom
for the stack — `N_BG` is already tuned close to the practical ceiling
for this buffer layout (see "A bug this port found," above, for why
`N_BG` can't just be made enormous instead of tuning `N_WORKLOAD_REPS`
down: the background DMA's own throughput here is only modestly higher
than the CPU's read-modify-write loop, so outlasting a *long* CPU
workload would need a *very* large `N_BG` — bigger than this chip's SRAM
allows). If you resize any buffer and the link fails, check
`build/firmware.map` (Topic 1's `.map`-file technique).

## Not covered here: Renode

Unlike `memory_benchmark`, this project has no Renode platform/scripts.
Renode's stock STM32F4 platform doesn't model AHB bus arbitration or DMA
bandwidth at all — exactly the mechanism Part 2 exists to measure — so a
simulated run would report identical numbers with and without background
DMA regardless of memory region, which is worse than not running it:
real hardware (or at minimum a cycle-accurate bus model this course
doesn't have access to) is the only way to answer this activity's
question. See `memory_benchmark/README.md`'s "Running under Renode?"
section for the same argument made in more detail for Topic 1's simpler
per-region latency question.
