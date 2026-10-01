# Activity: What Does Pruning Buy You on an MCU?
### Embedded AI — Pruning
### Board: STM32F4DISCOVERY (STM32F407VG) — 1 MB Flash, 128 KB SRAM (+64 KB CCM)

This activity answers two questions about pruning, by measuring on the STM32F407:

1. Does pruning **save space**, in Flash or SRAM?
2. Does pruning make inference **faster**?

We do it together in lecture. This handout follows the same steps in the same order, so you can redo it on your own or catch up on a step you missed. **Predict before you measure**, every time.

Companion project: [`pruning_lab/`](pruning_lab/). Its `README.md` has the full detail. Everything is fp32; quantization comes in the next lecture.

## 0. Setup

- The ExecuTorch Python environment from earlier labs:
  ```bash
  source ~/executorch/et-env/bin/activate
  python3 -c "import torch, executorch; print('ok')"
  ```
- The Fashion-MNIST dataset, downloaded once (about 30 MB):
  ```bash
  cd pruning_lab
  python3 python/data.py
  ```
- A 3.3 V USB-UART adapter on the Discovery board's P2 header (USART2: PA2 = TX, plus GND), as in the data-movement activity.
- A first firmware build, to check your toolchain:
  ```bash
  python3 python/export_variants.py --variant A
  cmake --preset stm32 && cmake --build --preset stm32
  ```
  `cmake --preset stm32` also checks that your ExecuTorch Cortex-M4 build includes the six ops this lab's models use. If it stops with "missing ops this lab needs", follow "ExecuTorch kernels" in `pruning_lab/README.md` to rebuild it with a larger op list (a few minutes, once).

All commands below run from `pruning_lab/` with the environment active.

All three model variants are **already trained** (`checkpoints/`). You won't train anything. `python/train_baseline.py` and `python/make_variants.py` show exactly how each one was made.

### The model: Model B2

Model B from the memory-budgeting activity, at 2× width: the same layers, with twice as many channels (filters) in each conv layer, conv1 8 → 16 and conv2 16 → 32:

```
1x32x32 -> Conv(1->16, 3x3) -> MaxPool -> ReLU -> Conv(16->32, 3x3) -> MaxPool -> ReLU -> GAP -> FC(32->10)
```

Max-pool comes before ReLU. That's the same function (the two commute), but pooling first uses less activation memory. The model is trained on Fashion-MNIST, zero-padded to 32×32.

| Variant | How it was made | Used in |
|---|---|---|
| A | Dense baseline | both parts |
| B | **Unstructured:** 70% of all weights zeroed by global magnitude (`prune.global_unstructured`, L1), fine-tuned with the masks attached, then `prune.remove()` | Part 1 |
| C | **Structured:** the half of each conv layer's filters with the smallest L1 norm **physically removed**. A new, narrower `ModelB2(8, 16)` is built from the kept filters and the matching downstream slices, then fine-tuned | Part 2 |

**`prune` and `prune.remove()`:** PyTorch's pruning functions don't overwrite a layer's weight. They rename it `weight_orig`, add a 0/1 `weight_mask`, and recompute `weight = weight_orig * weight_mask` on every forward pass. That's what keeps pruned weights at zero during fine-tuning. `prune.remove()` writes the masked weights back into a plain `weight` and deletes the rest. Call it before saving or exporting: without it, the exported `.pte` carries both `weight_orig` and the mask (46,576 B instead of 25,640 B for this model), plus an extra multiply per layer on every inference.

### The two measurement tools

**`python3 python/measure_host.py`** (laptop only, about 1 minute). For each variant it:

1. loads the pre-trained checkpoint and counts its parameters and nonzeros;
2. exports it through ExecuTorch (`torch.export` → `to_edge` → `to_executorch`, the same flow as the memory-budgeting activity), writing `pte/variant_X.pte`;
3. measures that variant's three file sizes: its `torch.save` checkpoint, its `.pte`, and its `.pte` after gzip;
4. reads the planned activation arena from the exported program, and computes MACs and the `memory_calc.py` estimate analytically;
5. evaluates accuracy with PyTorch on the 10,000-image test set.

It prints one row per variant:

| Column | Meaning |
|---|---|
| `params`, `nonzero` | parameters, and how many are nonzero |
| `MACs` | multiply-accumulates per inference |
| `ckpt` | `torch.save` checkpoint size |
| `.pte` | the ExecuTorch program, which is what goes into Flash |
| `.pte.gz` | the `.pte` after gzip, a rough measure of its information content |
| `planned` | ExecuTorch's planned activation arena |
| `pair est.` | `memory_calc.py`'s pair-model peak activation estimate |
| `test acc` | accuracy on 10,000 test images |

**`scripts/run_variant.sh X`** (board, about 40 s) exports variant X, builds, flashes, and prints the board's report. Use `UART=/dev/ttyUSBn` if your adapter isn't `/dev/ttyUSB0`. The report includes:

- `Board matches host`: the board's predictions on 10 test images compared with PyTorch's. This checks the export; it is **not** an accuracy measurement. If it's ever less than 10/10, stop and ask.
- `.pte size` and `Planned arena`: the program size, and the activation arena the runtime allocates
- `Cycles min/median/max` and `Median latency`: `method.execute()` timed with the DWT cycle counter at 168 MHz

Without the script:

```bash
python3 python/export_variants.py --variant X
cmake --build --preset stm32 --target flash
picocom -b 115200 --imap lfcrlf /dev/ttyUSB0      # then press the board's reset button
```

---

## Part 1: Does unstructured pruning save space or time?

### Step 1 — Predict

Before running anything, write down B's `.pte` size and board latency **relative to A**: "same", "≈ half", "≈ 3× faster", and so on.

| | `.pte` size vs. A | Latency vs. A |
|---|---|---|
| B | | |

### Step 2 — Does B take less space?

```bash
python3 python/measure_host.py
```

Record rows A and B:

| Variant | Nonzero params | Checkpoint (B) | `.pte` (B) | gzipped `.pte` (B) | Test accuracy |
|---|---|---|---|---|---|
| A | | | | | |
| B | | | | | |

**Q1.** Why are A's and B's `.pte` files the same size, even though 70% of B's weights are zero?

**Q2.** What does the gzip result tell you about the **information content** of B compared with its **storage**?

### Step 3 — Could a sparse format help?

**Q3.** For N weights of b bytes each, with a fraction s of them zero, the storage costs are: dense N·b; bitmask N/8 + (1−s)·N·b; CSR with 16-bit indices (1−s)·N·(b+2), plus 2 bytes per row pointer. Apply these to B's actual weights: 5,072 weights, 1,522 nonzero, spread over 58 rows (16 conv1 filters, 32 conv2 filters, 10 fc outputs). Work out the weight bytes for dense, bitmask, and CSR with 16-bit indices (include the row pointers), **in fp32 and in int8**. Which format wins in each case, and by how much?

**Q4.** Suppose B's weights were stored as a bitmask in Flash and decoded to dense in SRAM before each inference. Using B's planned arena (from `measure_host.py`, or from the board in Step 4) and the 128 KB of main SRAM, would the decoded weights fit? Would decoding one layer at a time help?

### Step 4 — Is B faster?

```bash
scripts/run_variant.sh A
scripts/run_variant.sh B
```

| Variant | Matches host | Planned arena (B) | Median cycles | Median ms |
|---|---|---|---|---|
| A | | | | |
| B | | | | |

**Q5.** Why is B no faster than A? B's cycle count may differ from A's by a few hundred cycles. Is that from skipped multiplies? How can you tell?

**Q6.** If a sparse kernel's cost per useful MAC is k times a dense kernel's, the sparse kernel wins only when s > 1 − 1/k. What k would make B's 70% sparsity break even? Why is comparing against this lab's 177 cycles/MAC portable kernel the wrong baseline?

### Step 5 — Wrap-up

**Q7.** Answer the two questions for unstructured pruning in one sentence each, and fill in the first two rows of the scorecard in Part 2, Step 5.

---

## Part 2: Structured (channel) pruning

### Step 1 — Zeroing vs. removing

**Q8.** Suppose you ran `prune.ln_structured(conv, "weight", amount=0.5, n=1, dim=0)` and then `prune.remove()` on A's two conv layers. That zeroes the same half of the filters that variant C drops. Relative to A, what would that model's `.pte` size, planned arena, and board latency be, and why? What does variant C do that this doesn't?

### Step 2 — Predict

Compute C's parameters and MACs by hand. (Hint: conv2 loses half its input channels *and* half its output channels.) Then predict C's `.pte` size, planned arena, and latency relative to A.

| | Params | MACs | `.pte` vs. A | Arena vs. A | Latency vs. A |
|---|---|---|---|---|---|
| A | 5,130 | 1,327,424 | — | — | — |
| C | | | | | |

### Step 3 — Measure

```bash
python3 python/memory_calc.py
scripts/run_variant.sh C
```

Row C of `measure_host.py`'s table (Part 1, Step 2) has the params, `.pte` size, and accuracy.

| Variant | Params | MACs | `memory_calc.py` peak (B) | Planned arena (B) | `.pte` (B) | Median cycles | Median ms | Test accuracy |
|---|---|---|---|---|---|---|---|---|
| A | | | | | | | | |
| C | | | | | | | | |

### Step 4 — Explain the results

**Q9.** Compare the planned arena with `memory_calc.py`'s estimate, for A and for C. The gap is the same fraction for both; what tensor does the pair model leave out? (The memory-budgeting activity's answer for Model B found the same thing.)

**Q10.** Compute A's latency divided by C's, and compare it with A's MACs divided by C's. Also compute cycles per MAC for each. What accounts for the difference? Name at least two things that don't shrink in proportion to MACs.

### Step 5 — Wrap-up

**Q11.** Which of A, B and C would you deploy, and why? Consider accuracy per KB of Flash, per KB of SRAM, and per millisecond. Would your answer change if the model had to run 10 times a second?

**Q12.** Fill in the compression scorecard. Use ✓ or ✗ and a few words for "what it needs".

| Technique | Flash (weights) | SRAM (peak activations) | Latency (dense kernels) | What it needs |
|---|---|---|---|---|
| Unstructured pruning, exported as-is (B) | | | | |
| Unstructured + sparse format | | | | |
| Structured, zeroed only (`ln_structured`, Q8) | | | | |
| Structured, channels removed (C) | | | | |

---

## Optional extension

Train the slimmed architecture `ModelB2(8, 16)` **from scratch**, with no pruning and no inherited weights, and compare its test accuracy with C's. Give it a comparable training budget: C saw 30 epochs as part of A, plus 5 of fine-tuning. The functions you need are all in `python/`; run this from `pruning_lab/python/`:

```python
from data import loaders
from model import ModelB2
from train_baseline import train_epochs, evaluate
train, val, test = loaders()
m = ModelB2(8, 16)
train_epochs(m, train, val, epochs=35, lr=3e-3)
print(evaluate(m, test))
```

If the two are about equal, what did pruning actually contribute?
