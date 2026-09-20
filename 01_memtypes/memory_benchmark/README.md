# memory_benchmark — Activity 2 companion firmware

Buildable, flashable STM32F407VG firmware for
[`../lecture1-activity-memory-benchmark-handout.md`](../lecture1-activity-memory-benchmark-handout.md)
(hand out to students) /
[`../lecture1-activity-memory-benchmark-answers.md`](../lecture1-activity-memory-benchmark-answers.md)
(instructor reference, with every task answered — including real captured
Renode output).
Times SRAM, CCM RAM, and Flash reads/writes using the Cortex-M4's DWT
cycle counter and prints the results over USART2.

This supersedes the old standalone `memory_benchmark.c` sketch (which
assumed HAL/CMSIS and STM32CubeIDE) with a complete, bare-metal project
matching the rest of this repo's convention: direct register access, no
HAL, `cmake --preset stm32`.

## What changed from the original sketch

- `HAL_Init()` / `SystemClock_Config()` / CMSIS `CoreDebug`/`DWT` structs
  / `__disable_irq()`/`__enable_irq()` intrinsics → replaced with direct
  register access via `stm32f407.h` (same pattern as
  `wakeword_et_dev/src/hal.c` and `lab0_hello_world_soln`) and inline
  `cpsid`/`cpsie` asm.
- `printf`/`SystemCoreClock` → this repo's newlib stubs
  (`src/syscalls.c`) don't implement a real `snprintf`/`vsnprintf` (no
  `FILE*` infrastructure on bare-metal), and pulling in float formatting
  isn't worth it here. `report()` in `src/main.c` does its own
  fixed-point (2 decimal place) integer formatting and writes to USART2
  by hand instead.
- Clock: HSE → PLL → **168 MHz**, not this repo's usual 16 MHz HSI-only
  config. At 16 MHz the STM32F407 needs zero flash wait states, which
  would make Flash reads look identical to SRAM and defeat the point of
  the activity. `clock_init()` in `src/main.c` sets up the same
  HSE→PLL→168 MHz path as `wakeword_et_dev/src/hal.c`, plus the 5 flash
  wait states RM0090 requires at that clock.

## CCM RAM: already wired up here

This project's `stm32f407.ld` has a working `.ccmram` output section —
`ccm_buf` in `src/main.c` needs to actually land in CCM RAM for the
benchmark to mean anything. (Activity 1's `map_file_demo/` used to be a
comparable from-scratch project with CCM RAM deliberately left unused,
for a before/after diff — it's now just the pre-built
`firmware.elf`/`firmware.map` from `lab0_hello_world_soln`, whose own CCM
RAM section is fully wired up too, so there's no longer a "before"
example to diff against in this repo.)

`startup_stm32f407.s` zeroes `.ccmram` the same way it zeroes `.bss`
(step 3b) since the region is `NOLOAD` — no flash image to copy from.

**If you're adding this to a project of your own that doesn't have it
yet:** depending on how your starter project was generated, the default
`.ld` file may define a `CCMRAM` memory region (STM32CubeIDE/CubeMX-
generated projects for the F407 usually do) but **not** have an output
section that actually places anything there. Check your `.ld` file for a
`MEMORY` block resembling:

```ld
MEMORY
{
  CCMRAM (xrw)    : ORIGIN = 0x10000000,   LENGTH = 64K
  RAM    (xrw)    : ORIGIN = 0x20000000,   LENGTH = 128K
  FLASH  (rx)     : ORIGIN = 0x08000000,   LENGTH = 1024K
}
```

If the `CCMRAM` region exists but there's no section placing data into
it, add this section block (typically placed near the `.bss` section
definition):

```ld
  .ccmram (NOLOAD) :
  {
    . = ALIGN(4);
    _sccmram = .;
    *(.ccmram)
    *(.ccmram*)
    . = ALIGN(4);
    _eccmram = .;
  } >CCMRAM
```

If the `MEMORY` block doesn't define `CCMRAM` at all, add the region
line shown above alongside your existing `RAM` and `FLASH` entries. Any
variable declared with `__attribute__((section(".ccmram")))` then links
into CCM RAM. After the change, confirm in a `.map` file (Activity 1)
that the CCM buffer's address falls in the `0x1000 0000` range — and
that its startup code zeroes the region the same way `.bss` is zeroed,
since `NOLOAD` means there's no flash image to copy from (compare
`startup_stm32f407.s` step 3b in this project).

## What the inner loop actually costs (for forming a prediction)

`bench_write`/`bench_read` manually unroll their inner loop 10-wide (see
the comment above `bench_write()` in `src/main.c`): each pass through the
loop body performs **10 memory accesses**, not 1, before paying for the
loop's own bookkeeping (increment/compare/branch) once. Check yourself
with `cmake --build build --target disasm` and look for
`bench_write.constprop.0`/`bench_read.constprop.0` — the write loop's
body is 10 back-to-back `str.w` instructions (one per unrolled word)
followed by a single `adds`/`cmp.w`/`bne.n`:

```
;; bench_write inner loop -- 10 stores per iteration, one branch
str.w ip, [r2, #0]
str.w ip, [r2, #4]
...                    ; 10 stores total, offsets 0..36
str.w ip, [r2, #36]
add.w r2, r2, #40       ; advance by 10 words
adds  r3, #10
cmp.w r3, #1020
bne.n <loop start>
```

A short straight-line tail (4 more stores/loads, executed once per rep,
not once per iteration) handles the remainder: `N_WORDS = 1024` isn't a
multiple of 10, so the unrolled loop covers words 0–1019 and the tail
covers 1020–1023 — every word still gets touched exactly once per rep,
same as before unrolling.

`bench_read`'s accumulation (`sink += buf[i+0]; ... sink += buf[i+9];`)
gives the compiler 10 partial sums to combine, which is more live values
than it has spare registers for — the disassembly shows 2 of the 10
reads getting spilled to the stack and reloaded before the final add.
That's 4 extra stack (SRAM) ops per **10** real reads, versus the old
`volatile`-accumulator version's 2 extra stack ops per **1** real read
(see below) — an order-of-magnitude improvement, not eliminated
entirely. `sink` is a plain (non-`volatile`) register variable, with a
`__asm volatile ("" : "+r" (sink))` compiler barrier after the loop —
not `volatile` on the variable itself — so the loop can't be optimized
away without forcing memory traffic on every accumulation the way
`volatile` would (see the comment in `bench_read()`).

Earlier, un-unrolled versions of this benchmark had a worse version of
the same problem: a `volatile sink` accumulator forced a stack spill
*and* reload, both always in main SRAM, on **every single iteration** —
2 of every 3 memory ops in the loop were then guaranteed-fast SRAM
traffic regardless of which region was under test, diluting the
CCM-vs-SRAM difference and understating the Flash penalty by roughly a
third. Unrolling doesn't just amortize loop overhead — it also shrinks
this kind of accumulator-spill overhead relative to the "real" accesses
being measured.

That means the floor for every benchmark's `cycles/access` is closer to
"1" than the un-unrolled version's 4-instruction loop implied, but still
not exactly 1 — it's whatever 10 zero-wait-state loads/stores plus one
`adds`/`cmp`/`bne` (amortized over 10) actually costs on this core, plus
`bench_read`'s residual 2-spill overhead (also amortized over 10).
Cortex-M4, roughly: single load/store ≈ 2 cycles at zero wait states,
each ALU op ≈ 1 cycle, a taken backward branch costs a pipeline refill
(commonly a couple of cycles on a 3-stage pipeline, but now paid once per
10 accesses instead of once per 1). Treat that as a ballpark for a
prediction, not a substitute for the DWT number the board actually
reports (Cortex-M4 pipelining/dual-issue can beat a naive
instruction-by-instruction sum).

### Try it yourself: disassemble `bench_write` and count instructions

1. Build first if you haven't (`cmake --preset stm32 && cmake --build --preset stm32`).

2. Find the real symbol name. GCC's `-O2` constant-propagation clones
   `bench_write` per call site and renames the clone — it will **not**
   be called plain `bench_write` in the binary:
   ```bash
   arm-none-eabi-nm build/firmware.elf | grep bench_write
   ```
   ```
   0800067c t bench_write.constprop.0
   ```

3. Disassemble just that one function (not the whole `disasm` target's
   output) using the exact name from step 2:
   ```bash
   arm-none-eabi-objdump -d --disassemble=bench_write.constprop.0 build/firmware.elf
   ```

4. Find the loop body. Scan for a `bne.n` near the end that branches
   *backward* to a label earlier in the listing — that backward edge is
   the unrolled loop. Everything from the branch target down to the
   `bne.n` itself (inclusive) is one pass through the loop body, and one
   pass covers **10 accesses** (see the unroll comment above
   `bench_write()` in `src/main.c`).

5. Count instructions in that block and split them into two piles:
   - **Memory instructions**: `str`/`str.w` (also `ldr`/`ldr.w` in
     `bench_read`'s case).
   - **Everything else**: `add.w`/`adds`/`cmp.w`/`bne.n` — loop
     bookkeeping and, in `bench_write`'s case, computing each `i+k`
     value to store (that's what the `add.w ip, r3, #k` instructions
     before each `str.w` are — the benchmark writes distinct payloads
     per word, not a constant, so the compiler can't skip this).

6. Divide both counts by 10 to get **instructions per access** — a
   memory-op count and an overhead count. On this build (13.2.1,
   `-O2`), the loop body is 23 instructions total: 10 `str`/`str.w` (one
   per unrolled word, confirming exactly 1 memory op per access) and 13
   everything-else, i.e. **1.0 memory instruction + 1.3 overhead
   instructions per access** — down from the un-unrolled version's 1
   memory instruction + 3 overhead instructions per access. Recount for
   your own toolchain version rather than trusting this number verbatim;
   codegen shifts between GCC versions.

7. *(Optional, for a numeric prediction)* Apply rough Cortex-M4
   per-instruction costs — zero-wait-state `str`/`ldr` ≈ 2 cycles, each
   `add`/`cmp` ≈ 1 cycle, the (once-per-pass) taken `bne.n` ≈ a couple of
   cycles for the pipeline refill — to the per-access instruction counts
   from step 6 to get a floor estimate for `cycles/access`, then flash
   and compare against what the board actually reports. Expect the real
   number to beat a naive instruction-by-instruction sum somewhat —
   Cortex-M4 can dual-issue and pipeline adjacent instructions.

For Flash: at 168 MHz, RM0090 Table 10 requires 5 wait states for the
2.7–3.6V range (`clock_init()` sets `FLASH_ACR_LATENCY(5)`), so a cold,
uncached access could cost up to `1 + 5 = 6` cycles just for the memory
stage. The ART Accelerator (prefetch + I/D-cache, enabled via
`FLASH_ACR_PRFTEN|ICEN|DCEN`) exists specifically to hide most of that on
repeated/sequential access — which is exactly this benchmark's pattern
(the same 4 KB `flash_buf` reread 2000 times, plus an untimed warm-up pass
before the timed one). Expect Flash read to land closer to SRAM/CCM RAM
than the naive 6-cycle worst case would suggest, not a clean multiple of
it — how close is the actual measurement.

## Access counts

- `N_WORDS = 1024` (4 KB buffer), `N_REPS = 2000`
- Each of the 5 timed benchmarks (SRAM write/read, CCM RAM write/read,
  Flash read) performs `N_WORDS × N_REPS = 2,048,000` word accesses —
  about 10.24M word accesses (~41 MB of traffic) across the whole run.
- Plus one untimed 1024-word warm-up read of `flash_buf` before the timed
  Flash measurement (not counted in any reported total).

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
columns are **not** — see "What the inner loop actually costs" above for
how to form an actual prediction, and just run it for the real numbers,
which is the point of the activity):

```
--- Memory Benchmark: STM32F407VG ---
Clock = 168000000 Hz (HSE -> PLL)
Buffer size = 1024 words (4096 bytes), 2000 reps

SRAM write    :    <cycles> cycles total | <us> us total | <cycles/access> cycles/access
SRAM read     :    <cycles> cycles total | <us> us total | <cycles/access> cycles/access
CCM RAM write :    <cycles> cycles total | <us> us total | <cycles/access> cycles/access
CCM RAM read  :    <cycles> cycles total | <us> us total | <cycles/access> cycles/access
Flash read    :    <cycles> cycles total | <us> us total | <cycles/access> cycles/access

--- Done ---
```

PD12 (green LED) blinks afterward as a heartbeat for anyone without a
UART adapter handy.

```bash
cmake --build build --target disasm   # optional: full disassembly
```

## Running under Renode

`platforms/`, `scripts/`, and `tests/` hold Renode assets for running this
firmware without physical hardware. This assumes Renode is already
installed at `~/renode_portable` (a portable release, e.g. v1.16.1) with
a `renode-venv` virtualenv holding `robotframework`/`psutil` for
`renode-test` — if you need to set that up from scratch, any course
Renode-installation guide for this board (e.g. the one accompanying a
`first_renode`-style blink/UART project, if your course has one) covers
it; the steps aren't specific to this project.

`platforms/stm32f4_discovery_full.repl` starts from Renode's stock
`platforms/boards/stm32f4_discovery.repl` and corrects it to match this
project's actual memory map: adds the `ccmram` region (missing from the
stock platform entirely — required here since `ccm_buf` needs to land in
it), resizes `sram`/`flash` from the stock board's 256 KB/2 MB down to
the real STM32F407VG's 128 KB/1 MB (matching `stm32f407.ld`), and adds a
`Miscellaneous.DWT` peripheral at `0xE0001000` (the stock platform has
none at all — some other Cortex-M boards' Renode platform files do wire
one in, e.g. `platforms/cpus/sam4s.repl`; this one didn't).

**Automated capture** (recommended — no second terminal needed):
```bash
cd memory_benchmark
cmake --preset stm32 && cmake --build --preset stm32
export PATH="$HOME/renode_portable/renode-venv/bin:$HOME/renode_portable:$PATH"
renode-test tests/test_benchmark.robot
```
This boots the firmware, waits for each of the five report lines on the
emulated USART2, and prints them to the console as it finds them.

**Interactive** (to watch it boot live): `renode --console
scripts/run_benchmark.resc`, then in a second terminal, `nc localhost
3456` (or any TCP-capable serial terminal, e.g. `picocom -b 115200
socket://localhost:3456`).

**Not useful for comparing memory regions, by the way.** The `Miscellaneous.DWT`
peripheral above does give `DWT->CYCCNT` a real, nonzero, deterministic
value — but it counts a fixed cost per *instruction*, with no model of
which memory region a load/store actually targets (no AHB arbitration,
no CCM RAM dedicated-bus modeling, no Flash wait states — `sram`,
`ccmram`, and `flash` are all just `Memory.MappedMemory` here). In
practice that means every read benchmark reports the identical cycle
count and every write benchmark reports the identical cycle count,
regardless of region. These Renode assets are useful for confirming the
firmware's control flow/clock/UART behavior without hardware, or as a
DWT-enabled platform-file starting point for a different project where
memory-region timing isn't the point — not for the SRAM-vs-CCM-vs-Flash
comparison this project exists to make. See
`../lecture1-activity-memory-benchmark-answers.md` ("Running under
Renode?") for the actual captured numbers demonstrating this.

## Extension: measuring Flash write cost directly

The main benchmark only times Flash **reads** — `flash_buf` is a `const`
array, already sitting in `.rodata` (XIP, no copy step). Timing an actual
Flash **write** needs a completely different path, because Flash writes
don't work like SRAM/CCM RAM stores at all: you can't just `str` a word
into a Flash address. You have to go through the Flash interface
peripheral's control registers (`FLASH_KEYR`/`FLASH_SR`/`FLASH_CR`,
RM0090 §3.6), which is what the handout's extensions section (Section 6)
is asking for. This section is the reference implementation for that
extension — read it, don't just copy it in, and see the endurance
warning at the bottom before running it.

### Why registers, not a store instruction

- **`FLASH_CR` starts locked.** Every reset sets the `LOCK` bit in
  `FLASH_CR`, specifically so a stray write can't corrupt the code
  currently executing. Unlocking is a deliberate two-step handshake via
  `FLASH_KEYR`, not encryption — any wrong value, wrong order, or a third
  write re-locks it.
- **You can't flip a bit from 0 back to 1 with a plain write.** Flash
  cells only go one direction when programmed (1→0). To write new data
  you must first **erase** the whole sector (resets it to all `0xFF`),
  *then* **program** the words you want — this is the asymmetric
  read/write cost from Slide 6 of the lecture, made concrete.
- **Both operations are asynchronous.** The CPU sets a bit and then has
  to poll `FLASH_SR`'s `BSY` bit until the Flash interface finishes —
  unlike an SRAM/CCM RAM store, which completes in the same cycle it's
  issued (modulo wait states).

### Register definitions to add to `stm32f407.h`

The project's `FLASH_t` struct only declares `ACR` (all that the main
benchmark needs for wait-state configuration). Extend it with the rest
of the Flash interface register map (RM0090 §3.9, register map table),
keeping the same "direct register access, no HAL" style as the rest of
this header:

```c
/* ── Flash interface (extended for write-cost benchmarking) ─────────────── */
typedef struct {
    volatile uint32_t ACR;      /* 0x00 access control (wait states)   */
    volatile uint32_t KEYR;     /* 0x04 unlock key register            */
    volatile uint32_t OPTKEYR;  /* 0x08 option byte unlock (unused)    */
    volatile uint32_t SR;       /* 0x0C status register                */
    volatile uint32_t CR;       /* 0x10 control register               */
} FLASH_t;

#define FLASH   ((FLASH_t *)FLASH_BASE)

#define FLASH_ACR_LATENCY(n)    ((n) & 0x7U)
#define FLASH_ACR_PRFTEN        (1U << 8)
#define FLASH_ACR_ICEN          (1U << 9)
#define FLASH_ACR_DCEN          (1U << 10)

#define FLASH_KEY1              0x45670123UL
#define FLASH_KEY2              0xCDEF89ABUL

#define FLASH_SR_EOP            (1U << 0)
#define FLASH_SR_WRPERR         (1U << 4)
#define FLASH_SR_PGAERR         (1U << 5)
#define FLASH_SR_PGPERR         (1U << 6)
#define FLASH_SR_PGSERR         (1U << 7)
#define FLASH_SR_BSY            (1U << 16)

#define FLASH_CR_PG             (1U << 0)   /* programming enable       */
#define FLASH_CR_SER            (1U << 1)   /* sector erase enable      */
#define FLASH_CR_SNB(n)         (((n) & 0xFU) << 3)  /* sector 0..11    */
#define FLASH_CR_PSIZE_X32      (2U << 8)   /* program 32-bit at a time */
#define FLASH_CR_STRT           (1U << 16)  /* start erase              */
#define FLASH_CR_LOCK           (1U << 31)
```

### Erase-then-program sequence

Use **sector 11** (`0x080E0000`–`0x080FFFFF`, the last 128 KB sector) as
scratch — it's the sector furthest from where this small firmware's code
actually lives, so erasing it won't brick the board mid-run. This
mirrors `DWT_Init()`/`DWT_Cycles()`/`report()`'s naming in `src/main.c`:

```c
#define FLASH_SCRATCH_SECTOR   11
#define FLASH_SCRATCH_ADDR     0x080E0000UL

static void flash_wait_busy(void) {
    while (FLASH->SR & FLASH_SR_BSY) { }
}

static void flash_unlock(void) {
    if (FLASH->CR & FLASH_CR_LOCK) {
        FLASH->KEYR = FLASH_KEY1;
        FLASH->KEYR = FLASH_KEY2;
    }
}

static void flash_lock(void) {
    FLASH->CR |= FLASH_CR_LOCK;
}

static void flash_erase_scratch_sector(void) {
    flash_wait_busy();
    FLASH->CR = (FLASH->CR & ~FLASH_CR_SNB(0xF)) | FLASH_CR_SER
              | FLASH_CR_SNB(FLASH_SCRATCH_SECTOR);
    FLASH->CR |= FLASH_CR_STRT;
    flash_wait_busy();
    FLASH->CR &= ~FLASH_CR_SER;
}

/* Programs exactly one 32-bit word and returns the cycle count for the
 * program step alone (erase and unlock are untimed setup, same as the
 * main benchmark excludes cache warm-up from its Flash read timing). */
static uint32_t flash_program_one_word(uint32_t value) {
    volatile uint32_t *dst = (volatile uint32_t *)FLASH_SCRATCH_ADDR;

    flash_wait_busy();
    FLASH->CR = (FLASH->CR & ~(3U << 8)) | FLASH_CR_PSIZE_X32;
    FLASH->CR |= FLASH_CR_PG;

    uint32_t start = DWT_Cycles();
    *dst = value;
    flash_wait_busy();
    uint32_t end = DWT_Cycles();

    FLASH->CR &= ~FLASH_CR_PG;
    return end - start;
}
```

### Timing it and comparing against SRAM

```c
flash_unlock();
flash_erase_scratch_sector();               /* untimed — see warning below */
uint32_t program_cycles = flash_program_one_word(0xA5A5A5A5UL);
flash_lock();

report("Flash program (1 word)", program_cycles, 1, 1);
```

Compare `program_cycles` directly against this activity's `SRAM write`
`cycles/access` result — expect Flash programming to land far higher
(microseconds vs. nanoseconds-scale), since it's an internal
charge-pump/programming-cycle operation on the Flash array itself, not a
bus write.

### Endurance warning

Each call to `flash_erase_scratch_sector()` consumes one of that sector's
finite erase cycles (RM0090 quotes on the order of 10,000 cycles typical
endurance). **Call the erase step once, not in a loop** — unlike the
read/write benchmarks elsewhere in this project, which deliberately
repeat `N_REPS = 2000` times to get a stable average, this measurement
should be a single before/after data point.
