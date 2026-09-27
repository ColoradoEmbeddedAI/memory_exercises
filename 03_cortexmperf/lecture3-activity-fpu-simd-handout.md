# Activity: FPU and SIMD Performance
### Embedded AI — Cortex-M Performance Characteristics
### Board: STM32F4DISCOVERY (STM32F407VG, Cortex-M4F)

Goal: measure, rather than assume, (1) how much a hardware FPU actually buys you over software floating-point emulation on the *same* silicon, and (2) how much a single SIMD instruction (`SMLAD`) buys you over a scalar loop for MAC-heavy int16 code.

Companion project: [`cortex_m_perf_lab/`](cortex_m_perf_lab/), a buildable, flashable bare-metal firmware project (see its `README.md` for full detail).

## How this activity works: simulate, then measure

You'll run four configurations, one at a time. Each build times exactly **one** dot-product kernel and prints one result. For each configuration you first run the firmware in the **Renode** simulator, then on the **real board**.

**What Renode gives you.** Renode executes the exact same firmware binary instruction by instruction. The lab's script sets it up so the firmware's cycle counter counts *instructions*, so Renode reports **exactly how many instructions each MAC takes**. That is your prediction: if every instruction took one cycle, cycles/MAC would equal instructions/MAC. Renode is exact, repeatable, and needs no hardware.

**What Renode doesn't give you.** Renode does not model the Cortex-M4's timing: loads that take 2 cycles, a floating-point multiply-accumulate that takes 3, the pipeline refill after a branch, or Flash wait states. So it cannot tell you real cycles. Under Renode, ignore the firmware's `us`, `MACs/cycle` and `cycles` figures and record only the `>>> Instructions per MAC` line.

**Putting them together.** The board gives you real cycles/MAC. Dividing by Renode's instructions/MAC gives the average **cycles per instruction (CPI)**:

```
CPI = (board cycles/MAC) ÷ (Renode instructions/MAC)
```

CPI = 1.0 would mean Renode's prediction was perfect. Anything above 1.0 is the cost of the hardware's timing: the part only the real board can show.

## Setup

### What you need

- The ARM cross-compiler and build tools from earlier labs: `arm-none-eabi-gcc`, `cmake` (3.20 or newer) and `ninja`
- `openocd` (to flash the board) and `picocom` (to read its output)
- Renode, installed at `~/renode_portable`, with its Python venv at `~/renode_portable/renode-venv/`
- The Discovery board connected by USB (for flashing), plus a 3.3V USB-UART adapter (for its output)

All commands below are run from the lab project directory:

```bash
cd memory_lectures/03_cortexmperf/cortex_m_perf_lab
```

### Building

Each configuration has a CMake preset. Configure it once, then build:

```bash
cmake --preset f32-hard            # configure: creates build/f32-hard/
cmake --build --preset f32-hard    # compile and link
```

A successful build ends by printing the firmware's size, and leaves `build/f32-hard/firmware.elf` (used by Renode) and `build/f32-hard/firmware.bin` (flashed to the board). Replace `f32-hard` with `f32-soft`, `q15-scalar` or `q15-simd` for the other rounds; each gets its own subdirectory of `build/`, so building one never overwrites another. If you change anything and want to rebuild, just rerun `cmake --build --preset <preset>`.

### Running under Renode (no board needed)

```bash
scripts/renode_run.sh f32-hard
```

- **No venv activation needed.** The script adds `~/renode_portable` and its venv to `PATH` itself. (If Renode is installed somewhere else, set `RENODE_HOME` to that directory first.)
- **No board or UART adapter needed.** Renode simulates the board, including its UART. The script captures the simulated UART output and prints it **in your terminal** when the run finishes. That takes about 5–10 seconds, and nothing appears until it's done.
- **What to record:** the last line, for example `>>> Instructions per MAC (Renode, exact): 8.00`. The firmware's own output above it says "cycles" and "us", but under Renode those are instruction counts, not real timing, so ignore them.

If the script says `no .../firmware.elf`, build that preset first. If it says `renode not found`, check that `~/renode_portable` exists. If Renode itself fails, the script prints Renode's log.

### Running on the board

The board's output comes over its UART, so you need the USB-UART adapter and a terminal program. Connect the adapter to the Discovery board's P2 header: adapter RX → **PA2** (USART2 TX), adapter GND → **GND**.

The firmware prints its result once, as soon as it starts, so **start picocom before you flash**, in a second terminal, logging to a file:

```bash
picocom -b 115200 --imap lfcrlf --logfile f32-hard.txt /dev/ttyUSB0
```

Then flash from your first terminal:

```bash
cmake --build build/f32-hard --target flash
```

The flash terminal shows OpenOCD's programming progress, ending with `shutdown command invoked`. The **results appear in picocom** (and are saved in `f32-hard.txt`) a moment later, once the board resets and runs. Exit picocom with `Ctrl-A` then `Ctrl-X`. Use a new log file for each configuration. To rerun without reflashing, press the board's black RESET button while picocom is running.

If `/dev/ttyUSB0` doesn't exist, look for the adapter with `ls /dev/ttyUSB* /dev/ttyACM*`. If it exists but won't open, add yourself to the `dialout` group (`sudo usermod -aG dialout $USER`, then log out and back in). If picocom shows nothing after flashing, it was probably started too late; press RESET.

**Every board run, check two lines before recording anything:** `Build config:` names the float ABI you meant to build, and the `result` line says `match` (q15) or shows ~1024000 (float32).

## The four configurations

| Round | Preset | Kernel | Float ABI | Vector length | Repetitions | Total MACs timed |
|---|---|---|---|---|---|---|
| 1 | `f32-hard` | float32 dot product | hard: FPU instructions used | 1,024 | 500 | 512,000 |
| 2 | `f32-soft` | float32 dot product | soft: no FPU instructions; every float operation is a call into a software emulation library | 1,024 | 500 | 512,000 |
| 3 | `q15-scalar` | int16 (q15) dot product, one MAC per instruction | hard | 1,024 | 500 | 512,000 |
| 4 | `q15-simd` | int16 (q15) dot product using `SMLAD`, two MACs per instruction | hard | 1,024 | 500 | 512,000 |

Each kernel computes the dot product of two vectors of the given length (`sum += a[i] * b[i]`, one MAC per element), stored in on-chip SRAM. It repeats the whole dot product the given number of times, and the total run is timed as one measurement with the Cortex-M4's cycle counter, interrupts disabled. So `total MACs = vector length × repetitions`, and the firmware prints `cycles/MAC = total cycles ÷ total MACs`. (The firmware's `Kernel:` line repeats the length and repetitions, e.g. `1024 elements x 500 reps`.) All four kernels use the same vector length and repetitions, so every measurement covers the same 512,000 MACs.

The lecture's simple model says: **≈1 MAC/cycle** for float with the hardware FPU, **≈2 MACs/cycle** for int16 with `SMLAD`, and software-emulated float one to two orders of magnitude slower. Keep those numbers in mind as you go.

Each round is the same four steps:

```bash
cmake --preset <preset> && cmake --build --preset <preset>     # 1. build
scripts/renode_run.sh <preset>                                 # 2. simulate (the prediction), output in this terminal
# 3. in a second terminal: picocom ... --logfile <preset>.txt /dev/ttyUSB0, then here:
cmake --build build/<preset> --target flash                    #    measure on the board, output in picocom
# 4. record, then answer the round's questions
```

## Round 1: float32, hard FPU (`f32-hard`)

**Renode prediction.** Renode reports **8.00 instructions per MAC**. At one cycle per instruction:

```
predicted cycles/MAC = 8.00
predicted MACs/cycle = 1 / 8.00 = 0.125
predicted time       = 512,000 MACs × 8.00 cycles/MAC ÷ 168,000,000 cycles/s = 24.4 ms
```

That's already 8x below the simple model's 1 MAC/cycle, before any hardware timing effects.

**Board measurement.**

| | Instructions/MAC (Renode) | Cycles/MAC (board) | CPI (board ÷ Renode) |
|---|---|---|---|
| f32-hard | 8.00 | | |

**Questions**

1. The simple model assumed ≈1 MAC/cycle, but Renode shows the loop runs 8 instructions per MAC. A dot-product loop has to do more than multiply-accumulate: what might the other 7 instructions be doing? Then, on the board, CPI came out above 1. Name something about real hardware that can make an instruction take more than one cycle.

## Round 2: float32, soft float (`f32-soft`)

**Renode prediction.** Renode reports **89.99 instructions per MAC**: each float multiply and add is now a call into a software library routine, and Renode counts the instructions executed inside those calls too.

```
predicted cycles/MAC   = 89.99
predicted MACs/cycle   = 1 / 89.99 = 0.0111
predicted time         = 512,000 × 89.99 ÷ 168,000,000 = 274.3 ms
predicted soft ÷ hard  = 89.99 ÷ 8.00 = 11.2x slower
```

**Board measurement.**

| | Instructions/MAC (Renode) | Cycles/MAC (board) | CPI (board ÷ Renode) |
|---|---|---|---|
| f32-soft | 89.99 | | |

Soft ÷ hard slowdown on the board (Round 2 cycles/MAC ÷ Round 1 cycles/MAC): ______ x

**Questions**

2. Renode predicted soft float would be 11.2x slower than hard float. What did the board show? Compare the two rounds' CPIs: why might the much longer soft-float code run at a *lower* CPI than the short hard-float loop? Which ratio, Renode's or the board's, would you report as "the cost of not using the FPU"?
3. Software floating-point emulation cost is often underestimated by people new to embedded systems. Why do you think that is, coming from a background where "floating point is basically free"?
4. If you were stuck targeting a Cortex-M0 (no FPU at all, not just a disabled one), what would these results imply for a design decision between float32 and quantized int8/int16 weights, independent of any memory savings from quantization?
5. A third float ABI, **softfp**, uses FPU instructions for the arithmetic but passes float *function arguments* in core (integer) registers instead of FPU registers. This dot-product loop calls no functions. Predict what Renode would report for a softfp build of this kernel, and what the board would measure. Would it land near hard, near soft, or in between?

## Round 3: q15 scalar (`q15-scalar`)

**Renode prediction.** Renode reports **6.00 instructions per MAC**.

```
predicted cycles/MAC = 6.00
predicted MACs/cycle = 1 / 6.00 = 0.167
predicted time       = 512,000 MACs × 6.00 ÷ 168,000,000 = 18.3 ms
```

**Board measurement.**

| | Instructions/MAC (Renode) | Cycles/MAC (board) | CPI (board ÷ Renode) |
|---|---|---|---|
| q15-scalar | 6.00 | | |

**Questions**

6. Compare with Round 1. The int16 loop needs fewer instructions per MAC than the float loop, and it has a lower CPI. Which result is faster per MAC, integer or float? Given that the loop structure is the same (load, load, multiply-accumulate, count, compare, branch), why might an integer MAC be cheaper than a float MAC on this core?

## Round 4: q15 SIMD (`q15-simd`)

**Renode prediction.** Renode reports **2.50 instructions per MAC**: each loop iteration now does two MACs with a single `SMLAD`.

```
predicted cycles/MAC  = 2.50
predicted MACs/cycle  = 1 / 2.50 = 0.40
predicted time        = 512,000 × 2.50 ÷ 168,000,000 = 7.6 ms
predicted speedup     = 6.00 ÷ 2.50 = 2.4x over q15 scalar (at CPI = 1)
```

A better prediction uses what the board has already taught you: the two loops have the same structure, so assume Round 3's CPI carries over:

```
calibrated cycles/MAC = 2.50 instructions/MAC × CPI(Round 3)
                      = 2.50 × 1.3 = 3.25        (e.g., with a Round 3 CPI of about 1.3)
```

**Board measurement.**

| | Instructions/MAC (Renode) | Cycles/MAC (board) | CPI (board ÷ Renode) |
|---|---|---|---|
| q15-simd | 2.50 | | |

SIMD speedup on the board (Round 3 cycles/MAC ÷ Round 4 cycles/MAC): ______ x

**Questions**

7. How close did SIMD come to the simple model's 2 MACs/cycle? Of Renode's 2.50 instructions per MAC, `SMLAD` itself accounts for only 0.5 (one instruction per two MACs). What must the other 2.0 be doing? If the loop were unrolled so the compare-and-branch overhead mostly disappeared, what would still limit this kernel?
8. All three of Rounds 1, 3 and 4 used the hard float ABI. Predict: if you rebuilt the two q15 kernels with the **soft** ABI, would Renode's count or the board's cycles change? Why or why not? What does your answer say about which hardware `SMLAD` uses, versus the hardware the FPU provides?
9. CMSIS-DSP's q15 kernels use exactly this kind of instruction. Given your measured speedup, and that an int16 value takes 2 bytes where a float32 takes 4, make the case (in a few sentences) for why quantizing a model to int16 might be attractive on this board for reasons beyond just memory footprint. The quantization unit later in the course will answer this in full.

## Wrap-up

| Round | Preset | Instructions/MAC (Renode) | Cycles/MAC (board) | CPI |
|---|---|---|---|---|
| 1 | f32-hard | 8.00 | | |
| 2 | f32-soft | 89.99 | | |
| 3 | q15-scalar | 6.00 | | |
| 4 | q15-simd | 2.50 | | |

Suppose you have to estimate, *before writing any code*, the latency of a conv layer with a 3×3 kernel, 16 → 16 channels, and a 32×32 output:

```
MACs        = H_out × W_out × out_ch × (k × k × in_ch) = 32 × 32 × 16 × (3 × 3 × 16) = 2,359,296
latency (s) = MACs × cycles_per_MAC ÷ 168,000,000
```

10. Which of your measured cycles/MAC would you use for this estimate, and what would make you more or less confident in it for a real, larger kernel (not just a tight dot-product loop)?
11. How good a predictor was Renode: on its own (CPI = 1), and once calibrated with a CPI from a similar kernel (Round 4)? When would you rely on a simulator like this, and when must you measure on the real board?
