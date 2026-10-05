# Activity: Where Did the 4× Go?
### Embedded AI — Quantization
### Board: STM32F4DISCOVERY (STM32F407VG) — 1 MB Flash, 128 KB SRAM (+64 KB CCM)

int8 makes every number 4× smaller. This activity measures what that actually buys on the STM32F407:

1. How much smaller is the model in **Flash**, and its activations in **SRAM**?
2. How much **faster** is it on a Cortex-M4F, which has a hardware FPU?
3. How much **accuracy** does it cost, and does the board compute what the laptop says it should?

We do it together in lecture. This handout follows the same steps in the same order, so you can redo it on your own or catch up on a step you missed. **Predict before you measure**, every time.

Companion project: [`quantization_lab/`](quantization_lab/). Its `README.md` has the full detail.

## 0. Setup

- The ExecuTorch environment from earlier labs, plus three one-time additions for ExecuTorch's Cortex-M backend: the TOSA serialization package, a CMSIS-NN checkout, and an ExecuTorch build with CMSIS-NN. Follow "Prerequisites" in `quantization_lab/README.md` (about 15 minutes). Check with:
  ```bash
  source ~/executorch/et-env/bin/activate
  python3 -c "from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer; print('ok')"
  ls ~/executorch/executorch-build/cortex-m4-cmsis/backends/cortex_m/libcortex_m_ops_lib.a
  ```
- The Fashion-MNIST dataset, downloaded once (about 30 MB):
  ```bash
  cd quantization_lab
  python3 python/data.py
  ```
- The 3.3 V USB-UART adapter on the Discovery board's P2 header (USART2: PA2 = TX, plus GND), as in the pruning activity.
- A first build, to check your toolchain:
  ```bash
  python3 python/export_model.py --variant A --precision int8
  cmake --preset stm32 && cmake --build --preset stm32
  ```

All commands below run from `quantization_lab/` with the environment active.

### The model and the builds

Model B2 from the pruning activity, and two of its variants, already trained (`checkpoints/`):

```
1x32x32 -> Conv(1->16) -> MaxPool -> ReLU -> Conv(16->32) -> MaxPool -> ReLU -> GAP -> FC(32->10)
```

| Variant | What it is | Params | MACs |
|---|---|---:|---:|
| A | dense, 16/32 channels | 5,130 | 1,327,424 |
| C | structured-pruned: half of each conv layer's filters removed, `ModelB2(8, 16)` | 1,418 | 368,800 |

Each variant runs at two precisions, so there are four **builds**: `A-fp32`, `A-int8`, `C-fp32`, `C-int8`.

- **fp32:** exported exactly as in the pruning lab, and run on ExecuTorch's portable kernels.
- **int8:** post-training quantization (PTQ) with ExecuTorch's `CortexMQuantizer`, calibrated on 200 training images, then lowered so that supported ops run on Arm's **CMSIS-NN** int8 kernels. `python/quantize.py` has the whole pipeline in a few dozen lines.

### The tools

**`scripts/run_build.sh A int8`** (board, under 10 s for int8; about 50 s for A-fp32, 20 s for C-fp32) exports a build, compiles, flashes, saves the board's report to `results/A-int8.txt`, and runs `compare_preds.py` on it. Any extra arguments go to `export_model.py` (e.g. `--order pool-relu`). Use `UART=/dev/ttyUSBn` if your adapter isn't `/dev/ttyUSB0`. The report includes:

| Line | Meaning |
|---|---|
| `.pte size (Flash)` | the program, embedded in Flash |
| `Planned arena (SRAM)` | the activation memory the runtime allocates |
| `Board accuracy` | correct predictions on the 200 embedded test images (20 for fp32 builds, which take 1.4 s each) |
| `Board matches host` | how many of the board's predictions equal the laptop's for the same build |
| `Cycles min/median/max`, `Median latency` | `method.execute()`, 1 warm-up and 10 timed runs, DWT cycle counter at 168 MHz |
| `Per-op profile` | one more inference, run one operator at a time, with each op's cycles |
| `Kernel scratch (peak)` | temporary memory the kernels asked for, outside the planned arena |

**`python3 python/compare_preds.py A-int8`** compares the board's 200 predictions with the laptop's, lists the images that differ, and prints the host's and the board's logits side by side for up to three of them. For an int8 build, the "host" is the `convert_pt2e` model: PyTorch running the quantized graph in floating point.

**`python3 python/inspect_graph.py --variant A`** (laptop) prints four sections: (1) each activation's scale and zero point and how the weights were quantized; (2) the lowered ops in execution order, with the kernel library that runs each one; (3) every tensor in the planned arena; (4) what's stored in the `.pte`. Add `--precision fp32` for the fp32 program, or `--order pool-relu`.

**`python3 python/eval_quant.py`** and **`python3 python/quant_sim.py`** (laptop, Parts B and C) are described where they're used.

---

## Part A: fp32 vs. int8 on the board

### Step 1 — Predict

From the pruning activity, A-fp32 and C-fp32 are known. Fill in your predictions for the int8 builds **before running anything**.

| Build | `.pte` (B) | Planned arena (B) | Latency (ms) | Test accuracy |
|---|---:|---:|---:|---:|
| A-fp32 | 25,640 | 114,688 | 1,400 | 84.7% |
| A-int8 | | | | |
| C-fp32 | 10,792 | 57,344 | 406 | 79.9% |
| C-int8 | | | | |

### Step 2 — Quantize Model B2 as the pruning lab left it

The pruning lab's model pools before its ReLU (conv → max-pool → ReLU). Quantize it in that order:

```bash
scripts/run_build.sh A int8 --order pool-relu
```

`export_model.py` prints the host's accuracy on the 200 embedded images. Predict the board's accuracy, and how many of its predictions will match the host's, before the board reports.

| | Host accuracy (200) | Board accuracy (200) | Board matches host |
|---|---:|---:|---:|
| A-int8, pool-relu | | | |

**Q1.** Look at the logits `compare_preds.py` prints for the first disagreeing image. How do the board's logits compare with the host's? What value do several of them share, and what is special about that value? (Hint: `inspect_graph.py` section 1 shows the range the output's int8 codes cover.)

**Q2.** Run `python3 python/inspect_graph.py --variant A --order pool-relu` and look at section 2. Which ops are *not* CMSIS-NN? Using the ReLU input's scale and zero point from section 1, work out what real value a portable `relu`, which computes max(q, 0), clamps at. What should it clamp at? Do the same for the second ReLU.

### Step 3 — Fix it

conv → ReLU → pool computes the same function as conv → pool → ReLU (max-pool and ReLU commute), and the Cortex-M quantizer fuses a ReLU into the conv right before it. `export_model.py` uses this order for int8 by default:

```bash
scripts/run_build.sh A int8
python3 python/inspect_graph.py --variant A
```

| | Host accuracy (200) | Board accuracy (200) | Board matches host | Portable ops in the graph |
|---|---:|---:|---:|---|
| A-int8 | | | | |

**Q3.** In the pruning lab, pooling before the ReLU saved 16 KB of SRAM. Why doesn't that argument apply to the int8 model? (Compare the conv1 output's size in fp32 and int8.) And why does the fused version also do better *on the laptop* (`python3 python/eval_quant.py --order pool-relu` vs. `python3 python/eval_quant.py`)? Look at the ranges in `inspect_graph.py` section 1 for both orders.

### Step 4 — Measure the four builds

```bash
scripts/run_build.sh C int8
scripts/run_build.sh A fp32      # ~50 s
scripts/run_build.sh C fp32      # ~20 s
python3 python/eval_quant.py --variant A         # host int8 accuracy, 10k test images
python3 python/eval_quant.py --variant C
```

The fp32 test accuracies on 10k images are A 84.69%, C 79.87% (from the pruning activity).

| Build | `.pte` (B) | Planned arena (B) | Kernel scratch (B) | Median cycles | Median ms | Test acc (10k, host) | Board matches host |
|---|---:|---:|---:|---:|---:|---:|---:|
| A-fp32 | | | | | | 84.69% | /20 |
| A-int8 | | | | | | | /200 |
| C-fp32 | | | | | | 79.87% | /20 |
| C-int8 | | | | | | | /200 |

#### Flash

**Q4.** Compute the int8/fp32 `.pte` ratio for A and for C. Neither is 4×. Use `inspect_graph.py` section 4 (int8 and `--precision fp32`) to account for A-int8's bytes: how many are int8 weights, how many are other constants the graph uses, how many are constants the graph never reads, and how many are program structure? Why is C's ratio so much worse than A's? Then export with `--strip-unused` and check the new size.

#### SRAM

**Q5.** Compute the int8/fp32 planned-arena ratio. It's *more* than 4×. Use `inspect_graph.py` section 3 for both precisions to find the tensors that make up the fp32 arena at its peak. Which one has no int8 counterpart? Which tensor is still fp32 in the int8 arena, and why doesn't it make the arena bigger? Compare `python3 python/memory_calc.py --int8` with the int8 planned arena.

#### Accuracy and agreement

**Q6.** How much accuracy did int8 cost A and C on the 10k test set? How many images does each int8 build disagree on between host and board? For one disagreeing image, compare the host's and board's logits from `compare_preds.py`: how many output codes apart are they? (The output scale is in `inspect_graph.py` section 1.) Is this rounding or a bug? Contrast it with Step 2.

#### Latency

**Q7.** Compute the fp32/int8 latency ratio for A and C. Then use the per-op profiles to compute **cycles per MAC** for conv1 and conv2, fp32 and int8 (conv1: 147,456 MACs in A; conv2: 1,179,648).

| | conv1 fp32 | conv1 int8 | conv2 fp32 | conv2 int8 |
|---|---:|---:|---:|---:|
| cycles | | | | |
| cycles/MAC | | | | |

The Cortex-M performance activity measured a hard-FPU fp32 dot product at **11.0** cycles/MAC and a q15 `SMLAD` dot product at **3.5** cycles/MAC on this board. Use those two numbers to split A's speedup into "better kernels" and "int8 arithmetic." Why does conv1 get so much less benefit than conv2?

### Step 5 — One compiler flag

Every project in this course compiles with `-mno-unaligned-access`. This lab compiles CMSIS-NN without it. Predict what happens to A-int8's latency if CMSIS-NN uses the course-wide flag, then measure:

```bash
cmake --preset stm32 -DCMSIS_NN_UNALIGNED=OFF && scripts/run_build.sh A int8
cmake --preset stm32 -DCMSIS_NN_UNALIGNED=ON          # restore the default afterwards
```

| `CMSIS_NN_UNALIGNED` | conv2 cycles | conv2 cycles/MAC | Median ms | Board matches host |
|---|---:|---:|---:|---:|
| ON | | | | |
| OFF | | | | |

**Q8.** Read "Why this lab compiles CMSIS-NN" in `quantization_lab/README.md`. Explain the slowdown. (Optional: `arm-none-eabi-objdump -d build/stm32/firmware.elf | awk '/<arm_nn_mat_mult_kernel_s8_s16>:/,/^$/' | grep -E 'smlad|bl'` in each configuration.)

### Step 6 — Wrap-up

**Q9.** Total compression from A-fp32 to C-int8: ratios for `.pte`, planned arena and latency, and the accuracy cost. Which part came from pruning and which from quantization? Which builds meet a 10 Hz (100 ms) budget?

---

## Part B: calibration and granularity (laptop)

### B1 — Calibration data

`eval_quant.py` quantizes A with the same quantizer the board build uses, varying only the calibration images, and evaluates on all 10,000 test images (about 30 s for the whole table):

```bash
python3 python/eval_quant.py --sweep
```

Predict first: accuracy with 1 image, with 200 images of a single class (T-shirt/top), and with 200 images that skipped normalization.

| Calibration | Logit range | Test accuracy | Drop |
|---|---|---:|---:|
| 1 image | | | |
| 10 images | | | |
| 100 images | | | |
| 1,000 images | | | |
| 200, T-shirt/top only | | | |
| 200, not normalized | | | |

**Q10.** What's the smallest calibration set you'd trust for this model, and why? Look at the "logit range" column for the 1-image and T-shirt-only rows: what happens to an image whose winning logit is above the top of that range? Try another class with `python3 python/eval_quant.py --calib-class 1`; why might one class hurt much less than another? What does the not-normalized row's input range look like (`inspect_graph.py` doesn't take calibration flags, but you can reason from `data.py`'s MEAN and STD)?

### B2 — Per-tensor vs. per-channel weights

`quant_sim.py` simulates quantization in plain PyTorch, so it can try things the backend doesn't offer:

```bash
python3 python/quant_sim.py --compare granularity
python3 python/quant_sim.py --levels
```

**Q11.** At which weight bit width does per-channel start to matter? `--levels` lists how many distinct int8 codes each conv2 filter's weights use: which filters suffer under per-tensor, and why don't they hurt accuracy at 8 bits?

### B3 — Range estimators: clipping vs. rounding

```bash
python3 python/quant_sim.py --compare range
python3 python/quant_sim.py --clip-curve conv1 --bits 8
python3 python/quant_sim.py --clip-curve conv2 --bits 4
```

**Q12.** In the two clip curves, where is the total error lowest relative to the min/max clip value? Explain each in terms of rounding error and clipping error. At which activation bit widths does the range estimator matter?

### B4 — Symmetric or asymmetric?

ExecuTorch's Cortex-M documentation says the backend "implements **symmetric INT8 (8w8a)** quantization".

**Q13.** In `inspect_graph.py --variant A` section 1, what are the zero points of the weights, and of the activations? Is the scheme symmetric? What does Z = −128 mean for a post-ReLU tensor? Use `python3 python/quant_sim.py --abits 6 --act sym` vs. `--act asym` to estimate what symmetric activations would cost.

---

## Part C: sensitivity and graph reading (laptop)

### C1 — Per-layer sensitivity

```bash
python3 python/quant_sim.py --sensitivity
```

**Q14.** Which layer is most sensitive at 4 bits? Is it the first or last layer, as the rule of thumb says? If you had to keep one layer at higher precision on this backend, what would that cost in kernels and SRAM?

### C2 — Read the lowered graph

**Q15.** For A-int8 (default order) and for A-int8 with `--order pool-relu`, list every op and whether CMSIS-NN or a portable kernel runs it. Identify any fp32 fallback (dequantize → fp32 op → quantize) and any portable op running on int8 data. Suppose the GAP fell back to fp32: what tensor would have to exist in fp32, and how many bytes of SRAM would it take?

### C3 — Reflection

**Q16.** Which two findings from this activity will you apply first when you deploy a model of your own? One or two sentences each.

---

## Optional extensions

1. **QAT.** The PT2E flow has a QAT entry point. From `quantization_lab/python/`:
   ```python
   import torch
   from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
   from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
   from data import load_tensors, loaders
   from eval_quant import accuracy
   from model import load_variant
   from quantize import CL, example_input

   m = load_variant("A", "relu-pool")
   gm = torch.export.export(m.train(), (example_input("int8"),)).module()
   prep = prepare_qat_pt2e(gm, CortexMQuantizer())
   train, _, _ = loaders()
   opt = torch.optim.Adam(prep.parameters(), lr=1e-4)
   for i, (x, y) in enumerate(train):
       loss = torch.nn.functional.cross_entropy(prep(x.contiguous(memory_format=CL)), y)
       opt.zero_grad(); loss.backward(); opt.step()
       if i == 200: break
   q = convert_pt2e(prep)
   _, _, (xt, yt) = load_tensors()
   print(accuracy(q, xt, yt, channels_last=True))
   ```
   Compare with PTQ's 83.89%. Is there anything to recover at 8 bits?
2. **Strip and verify.** Run `scripts/run_build.sh A int8 --strip-unused`. Does the board still agree with the host? What changed in the report?
3. **Fewer bits.** How few weight bits does `quant_sim.py` need before per-channel C loses more than 2 points (`--variant C --wbits N`)?
