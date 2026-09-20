# Activity 2: Measuring Memory Speed
### Embedded AI — Memory Hierarchy & Types
### Board: STM32F4DISCOVERY (STM32F407VG)

Goal: measure, in CPU cycles, the actual cost of reading and writing SRAM, CCM RAM, and reading Flash — then compare against predictions made *before* running the code.

Companion project: `memory_benchmark/` (sibling of this file) — a complete, buildable, flashable STM32F407VG firmware, not just a sketch. Build instructions are in this file, but you can look at `memory_benchmark/README.md` for full implementation details after completing this activity.

## 1. Background you need first

- The STM32F407VG has three distinct memories worth comparing: main SRAM (0x2000 0000, DMA-accessible), CCM RAM (0x1000 0000, CPU-only, no DMA), and Flash (0x0800 0000, non-volatile, asymmetric read/write cost).
- We will **not** write to Flash in this activity since it wears out the sector (limited erase cycles). We measure Flash **reads** only.
- We use the Cortex-M4's built-in **DWT cycle counter** (part of the debug/trace unit) rather than a timer peripheral or wall-clock timing — it counts CPU clock cycles directly with no peripheral setup overhead.
- **The loop costs more than the memory access — and is manually unrolled to reduce that cost.** A naive one-access-per-iteration loop pays for an increment/compare/branch on every single memory access. `bench_write`/`bench_read` unroll 10-wide instead (10 loads or stores per pass through the loop body, then one `adds`/`cmp`/`bne`), so that bookkeeping is amortized over 10 accesses rather than 1. `memory_benchmark/README.md` ("What the inner loop actually costs") has the actual disassembly, if you want to see exactly how.

## 2. Before running anything: predict actual cycle counts

Some facts about the loop shape, so your guess can be informed rather than a pure guess:

- Each benchmark performs `1024 × 2000 = 2,048,000` word accesses — all reads, for the three read benchmarks; all writes, for the two write benchmarks.
- The loop doing this is unrolled 10-wide: each pass through the loop body performs **10 reads or 10 writes**, then pays for the loop's own bookkeeping (index increment, bound check, branch) once.
- That bookkeeping costs **about 14 cycles per pass** (10 unrolled accesses). The full instruction-by-instruction derivation of that number is in `memory_benchmark/README.md` ("What the inner loop actually costs"), if you want to see where it comes from — not required for this task.

**Task:** using the above, plus what we know about zero-wait-state SRAM/CCM RAM access (roughly 2 cycles for a single load/store on this core) and Flash's wait-state requirement at 168 MHz (up to `1 + 5 = 6` cycles for a cold, uncached access — though the ART Accelerator exists specifically to hide much of that on repeated access, like this benchmark's pattern), predict an actual `cycles/access` number for each of the five benchmarks:

| Operation | Predicted cycles/access |
|---|---|
| SRAM write | |
| SRAM read | |
| CCM RAM write | |
| CCM RAM read | |
| Flash read | |

## 3. Building and running (on real hardware)

```bash
cd memory_benchmark
cmake --preset stm32 && cmake --build --preset stm32
cmake --build build --target flash
```

Connect a 3.3V USB-UART adapter to the Discovery board's P2 header (USART2: PA2 = TX) and open a terminal at 115200 8N1:

```bash
picocom -b 115200 --imap lfcrlf /dev/ttyUSB0
```

Build notes:
- The project already builds at `-O2` with the fixed 168 MHz clock described in Section 1.
- No debugger needs to be attached during measurement — a debugger halting the core mid-loop would corrupt the cycle count, so just flash and reset, then watch the UART. PD12 (green LED) blinks afterward as a heartbeat for anyone without a UART adapter handy.

**Task:** record the five report lines exactly as printed:

```
SRAM write    : ________________________________________________
SRAM read     : ________________________________________________
CCM RAM write : ________________________________________________
CCM RAM read  : ________________________________________________
Flash read    : ________________________________________________
```

## 4. Running under Renode?

The same firmware also boots under [Renode](https://renode.io/) (no physical board needed) — this project's `platforms/`, `scripts/`, and `tests/` hold everything needed to run it there; see `memory_benchmark/README.md` ("Running under Renode") for the commands.

It's not useful for *this* activity, though, so it isn't a required step here — worth explaining why rather than just skipping it. Renode's stock STM32F4 Discovery platform doesn't model the DWT cycle counter at all; this project's platform file adds a real one (Renode does ship a working `DWT` peripheral, just not wired into that particular board by default). But that DWT model counts a fixed cost per *instruction*, with no awareness of which memory region a load or store actually targets — Renode doesn't model AHB bus arbitration, CCM RAM's dedicated bus, or Flash wait states at all. So every read benchmark reports the identical cycle count, and every write benchmark reports the identical cycle count, regardless of whether the region under test was SRAM, CCM RAM, or Flash — the exact distinction this whole activity exists to measure is invisible to it.

So: this is here in case you want to see the firmware run without hardware, or if you want a working DWT-enabled Renode platform as a starting point for a *different* project where cycle-accurate memory-region timing doesn't matter (e.g. verifying control flow, or timing something that isn't memory-region-dependent). For measuring SRAM vs. CCM RAM vs. Flash specifically, real hardware (Section 3) is the only tool here that can actually answer the question.

If you're curious to confirm this yourself:
```bash
cd memory_benchmark
export PATH="$HOME/renode_portable/renode-venv/bin:$HOME/renode_portable:$PATH"
renode-test tests/test_benchmark.robot
```

## 5. Discussion / write-up

1. How close were your predictions? Where were you most surprised?
2. What's the write vs. read ratio for SRAM? For CCM RAM? Are they meaningfully different from each other?
3. Flash read cycles per access — does it look like a flat per-access cost, or does it seem to vary with access pattern (try sequential vs. a strided/non-sequential access pattern as a stretch extension)?
4. Given these numbers, if you were choosing where to place a large read-only lookup table your model uses at inference time, would you copy it to CCM RAM first or read it directly from Flash? Under what conditions would your answer change (hint: think about how many times it's read per inference)?
5. CCM RAM isn't reachable by DMA. Given today's measurements, when would that limitation actually matter in practice, and when would it not?
6. Section 4 explained why Renode couldn't answer this activity's core question, even though the same firmware boots and runs correctly there. What does that tell you, in general, about what a functional simulator is (and isn't) good for — and what would it take for you to trust a simulator's *timing* numbers, not just its functional correctness?

## 6. Extensions (optional)

- Compare `-O0` vs `-O2` results directly and explain the difference in your own words.
- Add a strided-access variant (e.g., stride of 16 words) to the Flash read benchmark and see whether the ART Accelerator's prefetch buffer helps sequential access more than strided access.
- Measure actual Flash **write** cost once, in a scratch sector, via the Flash controller's key/erase/program registers (`FLASH->KEYR` unlock sequence, `FLASH->CR` sector erase, then word programming — RM0090 §3.6), and compare the single-word program time against the SRAM write time from this activity. This exercises the sector's limited erase-cycle endurance — run it sparingly, not looped like the other benchmarks. See `memory_benchmark/README.md` ("Extension: measuring Flash write cost directly") for the register definitions and a working erase/program/time code sequence.
