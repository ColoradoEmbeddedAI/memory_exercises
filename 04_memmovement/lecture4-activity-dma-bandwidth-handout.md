# Activity: DMA Overlap & Bus Contention
### Embedded AI — Data Movement & Bandwidth Bottlenecks
### Board: STM32F4DISCOVERY (STM32F407VG)

Goal: measure, rather than assume, (1) whether DMA overlap actually delivers on its "free data movement" promise, and (2) whether bus contention between DMA and CPU is real and measurable — and whether CCM RAM's DMA restriction is actually useful for avoiding it.

Companion project: [`dma_bandwidth_lab/`](dma_bandwidth_lab/) — a buildable, flashable firmware project (see its `README.md` for full detail).

This activity deliberately generates its own synthetic bus traffic with a memory-to-memory DMA transfer rather than a real sensor peripheral, so it runs standalone on the Discovery board without extra wiring.

## Part 1: Does DMA Overlap Actually Help?

### Background

The theory: if a DMA transfer runs concurrently with independent CPU compute, the total time should approach `max(transfer_time, compute_time)` rather than `transfer_time + compute_time`. This activity tests that directly.

### What This Part Measures

The firmware runs four back-to-back measurements, each timed with interrupts disabled so nothing else can preempt the clock:

1. **Copy alone** — a DMA memory-to-memory transfer of 2,048 words (8 KB), timed by itself, with no CPU work happening at the same time.
2. **Compute alone** — a CPU read-modify-write loop over a 512-word buffer, repeated 400 times, timed by itself, with no DMA activity.
3. **Blocking, sequential** — the CPU copies the same 2,048 words itself (a plain array-to-array copy), then immediately runs the same 400-repetition compute loop — one after the other, no overlap.
4. **Overlapped (DMA copy || compute)** — the DMA transfer is started, then the same compute loop runs while it's still in flight; the measurement waits for the DMA to finish (in case compute finishes first) before stopping the clock.

The firmware also prints a **naive sum prediction**, computed as `Copy alone + Compute alone`, for comparison against the two combined measurements above.

## Part 2: Is Bus Contention Real?

### Background

We previously introduced CCM RAM as CPU-only — no DMA access. Here we test whether that restriction is actually an advantage: does a CPU workload running out of CCM RAM avoid a slowdown that the same workload in main SRAM experiences, when a DMA transfer is active in the background?

### What This Part Measures

The firmware runs a background DMA transfer of 12,000 words (~46.9 KB) and, separately, a CPU read-modify-write workload over a 512-word buffer repeated 8 times, in four combinations:

- the workload buffer in **main SRAM**, background DMA **idle**
- the workload buffer in **main SRAM**, background DMA **active**
- the workload buffer in **CCM RAM**, background DMA **idle**
- the workload buffer in **CCM RAM**, background DMA **active**

Each "active" measurement starts the background DMA transfer, times the CPU workload while it's running, then checks whether the DMA had already finished (printing a `[warning]` if so, since that would mean no contention was actually possible) before letting it fully finish. The firmware also times the background transfer by itself (`Background DMA duration (alone)`) as a reference for that check.

## Results

Build and flash the firmware:

```bash
cd dma_bandwidth_lab
cmake --preset stm32
cmake --build --preset stm32
cmake --build build --target flash
```

Then connect a 3.3V USB-UART adapter to the Discovery board's P2 header (USART2: PA2 = TX) and open a terminal at 115200 8N1 (e.g. `picocom -b 115200 --imap lfcrlf /dev/ttyUSB0`) to read the results. Reset the board (or re-run the `flash` target) to rerun the measurements.

The firmware prints all of Part 1's and Part 2's results in one continuous stream, starting the moment the board boots — there's no pause between sections, so capture the whole output in one run.

### Part 1 results

Record the five lines the firmware prints for Part 1:

| Quantity | Cycles |
|---|---|
| Copy alone | |
| Compute alone | |
| Naive sum prediction (copy+compute) | |
| Blocking, sequential | |
| Overlapped (DMA copy \|\| compute) | |

### Part 2 results

Tuning check first: compare `Background DMA duration (alone)` against `SRAM workload, DMA idle`, printed just after it — the background transfer should clearly outlast the workload. If you see a `[warning] background DMA finished before the workload did`, increase `N_BG` (or decrease `N_WORKLOAD_REPS`) in `dma_bandwidth_lab/src/main.c` and rebuild. Increasing `N_BG` uses more SRAM for `bg_src`/`bg_dst` — if you run out, use Topic 1's `.map`-file technique (`build/firmware.map`) to check your actual SRAM budget and back off `N_WORKLOAD_REPS` instead.

Once the tuning checks out, record the four measurements:

| Configuration | Cycles | % change vs. that memory's own DMA-idle baseline |
|---|---|---|
| SRAM, DMA idle | | — (baseline) |
| SRAM, DMA active | | |
| CCM, DMA idle | | — (baseline) |
| CCM, DMA active | | |

## Discussion

1. How close was the overlapped result to `max(Copy alone, Compute alone)`? If it's somewhat higher than that theoretical floor, what real-world costs does the simple `max()` model leave out (hint: look at what `dma_overlap_copy_and_compute()` does before and after the compute loop, in `dma_bandwidth_lab/src/main.c`)?
2. At what relative sizes of `N_XFER` and `N_COMPUTE`/`N_COMPUTE_REPS` (both defined near the top of `dma_bandwidth_lab/src/main.c`) would overlap provide the *least* benefit in absolute cycles? Which one dominates in that case? Given the numbers you actually measured, which regime are the current defaults in?
3. Where in an actual streaming inference pipeline (Topic 4's double-buffering pattern, covered in lecture) would you place the "compute" work and the "transfer" work to get this benefit for real?
4. Compare CCM's "DMA active" number to CCM's own "DMA idle" number, and separately to SRAM's "DMA active" number — which one does it resemble, and why does that make sense given CCM RAM's bus topology? If CCM's contention penalty came out small but nonzero rather than exactly zero, what could explain a nonzero result even for a CPU-only memory (hint: think about what else is happening on the AHB matrix besides the specific transfer you configured, and about measurement noise/interrupt jitter even with interrupts disabled around the timed region).
5. Compare the SRAM contention penalty here to the magnitude of the DMA overlap benefit you found in Part 1. Under what system design would the Part 1 benefit and the Part 2 penalty be in direct tension — i.e., using DMA to overlap data movement with compute is exactly what creates the contention risk from Part 2. How would you resolve that tension in a real design?
6. Given today's results, revise (or confirm) the lecture's "design implication": which buffers in a real inference pipeline belong in CCM RAM, and which ones *can't* go there because they need DMA to reach them at all?
