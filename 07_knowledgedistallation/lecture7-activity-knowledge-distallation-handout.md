# Activity: Distilling a Teacher into a Tiny Student
### Embedded AI — Knowledge Distillation
### Laptop CPU for Parts 0–6; STM32F4DISCOVERY (STM32F407VG) for the optional board step in Part 5

The knowledge distillation lecture made a set of claims: soft targets carry information that labels don't, temperature makes that information visible, T² keeps the loss balanced, and a teacher turns unlabeled data into training data. This activity tests each one on a model small enough to train in seconds:

1. Does a student trained on the teacher's outputs beat one trained on the labels?
2. How do the temperature T and the weight α matter, and does the answer depend on how much labeled data you have?
3. With only **1,000 labels** and 54,000 unlabeled images, how close can a distilled student get to one trained on all 55,000 labels?
4. Can a student learn to recognize a digit it has **never seen**?
5. Does the student fit the STM32F407, and what does distillation change on the board?
6. (Extension) Can KD recover our own pruned Model B2, and was that model limited by its capacity or by its training signal?

The activity is **optional and self-paced**, done on your own time, and it isn't graded. Each part has a command to run, a prediction to make first, and questions to answer. **Predict before you measure**, every time; the predictions are where most of the learning is. When you're done, compare your numbers with the reference results in the answer key.

Companion project: [`distillation_lab/`](distillation_lab/). Its `README.md` has the details of every script.

## 0. Setup

- The ExecuTorch Python environment from earlier labs, which has PyTorch:
  ```bash
  source ~/executorch/et-env/bin/activate
  ```
- MNIST, downloaded once (about 11 MB), and a check of the splits:
  ```bash
  cd distillation_lab
  python3 python/data.py
  ```
  Part 6 downloads Fashion-MNIST into `data/fashion/` the first time it runs (about 30 MB).
- **Part 5's board step only:** the STM32F4DISCOVERY, the 3.3 V USB-UART adapter on PA2 (as in the pruning and quantization activities), and the quantization lab's ExecuTorch tree with CMSIS-NN (`~/executorch/executorch-build/cortex-m4-cmsis`; see "Prerequisites" in the quantization lab's README). Everything else runs on the laptop.

All commands below run from `distillation_lab/` with the environment active.

### The teacher and the student

| | Architecture | Parameters | Provided as |
|---|---|---:|---|
| Teacher | LeNet-300-100: 784 → 300 → 100 → 10, ReLU, dropout 0.2 in training | 266,610 | `checkpoints/teacher_lenet300100.pt` (trained by `python/train_teacher.py`) |
| Student | MLP: 784 → 30 → 10, ReLU | 23,860 | trained by you, in every part |

Images are MNIST digits, flattened to 784 values and normalized with mean 0.1307 and standard deviation 0.3081.

### The splits

All fixed by seeds in `python/data.py`, so everyone gets the same images:

| Split | Images | What it is | Used in |
|---|---:|---|---|
| `train_full` | 55,000 | the 60k training images minus `val` | Parts 1, 2, 3d |
| `val` | 5,000 | for choosing T, α, and Part 4's bias | everywhere |
| `test` | 10,000 | the official test set: **report, never tune** | everywhere |
| `labeled_1k` | 1,000 | 100 per digit from `train_full` | Parts 2, 3 |
| `unlabeled_rest` | 54,000 | the rest of `train_full`, with labels hidden | Part 3c |
| `train_no3` | 49,372 | `train_full` with every 3 removed | Part 4 |

### The training recipe

Every student in Parts 1–5 is trained the same way, by `train()` in `python/kd.py`: Adam, learning rate 10⁻³, batch 128, **4,300 steps** (10 epochs of `train_full`), no augmentation, no learning-rate schedule. The loss is cross-entropy when there's no teacher (α = 0), and otherwise the lecture's loss:

L = α · T² · KL( p_teacher(T) ‖ p_student(T) ) + (1 − α) · CE( y, p_student(1) )

**Every configuration runs with 3 seeds** (0, 1, 2), and results are reported as mean ± standard deviation. The seed sets both the student's initialization and the batch order. Each part's script prints its results, and saves them to `results/part<N>.json`.

### The code

`python/kd.py` is short and worth reading before you start:

- `kd_loss_slide()` is the lecture's 10-line function, verbatim.
- `kd_loss()` is the same loss, computed per image, so that a batch can mix labeled and unlabeled images. An image with label −1 (hidden) gets the soft term alone.
- `train()` is the loop: one batch tensor goes to both the teacher (in eval mode, under `no_grad`) and the student.

---

## Part 0 — Warm-up (no training, about 10 seconds)

```bash
python3 python/part0_warmup.py
```

The script prints five things.

**1. Parameter counts.** Check both by hand: 784·300 + 300 + 300·100 + 100 + 100·10 + 10 for the teacher, and the same for 784-30-10.

**2. The teacher's test accuracy and confidence.** *Predict first:* on what fraction of test images does the teacher put more than 0.9999 on its top digit at T = 1? At T = 4?

**3. The teacher's softmax at T = 1, 4, and 10** for five test images (indices 0, 3, 7, 149, 412).

**4. The loss on one real batch**, for an untrained student, computed correctly and with each pitfall from the lecture's checklist: no T², `reduction="mean"`, swapped `kl_div` arguments, probabilities where log-probabilities belong, and the teacher left in train mode (dropout on).

**5. The soft term's gradient** with respect to the student's logits, at T = 1, 2, 4, 8, and 16, with and without the T² factor.

**Questions**

- **Q0.1** At T = 1, how different are the teacher's outputs from one-hot labels? What does T = 4 change? Use the confidence numbers from item 2.
- **Q0.2** In item 3, find the image where two digits are both visibly nonzero at T = 1. Which digits are they, and is that confusion plausible for a handwritten digit? What do the "confident" images (0 and 3) look like at T = 10?
- **Q0.3** Which pitfalls in item 4 change the loss by a large factor, and which only slightly? Would any of them crash, or show up as an obvious error during training? One of them gives a *negative* "KL divergence": why is that impossible for a real KL?
- **Q0.4** In item 5, how does the gradient scale with T without the T² factor, at small T and at large T? What does the T² factor do to it? Relate this to the lecture's claim that "gradients of the soft term scale as 1/T²".

---

## Part 1 — Baseline vs. KD, all 55,000 labels (about 30 seconds)

```bash
python3 python/part1_baseline_vs_kd.py
```

Two configurations on `train_full`, 3 seeds each:

- **CE:** cross-entropy only (α = 0)
- **KD:** T = 4, α = 0.9

The script prints test accuracy, accuracy on `train_full` itself, and the validation accuracy after each epoch, averaged over seeds.

*Predict first:* KD minus CE, in test-accuracy points.

**Questions**

- **Q1.1** How large is the KD improvement, and is it larger than the seed-to-seed spread?
- **Q1.2** Compare the two validation curves. Does KD converge faster, end higher, overfit less, or some combination?
- **Q1.3** Compare the two students' accuracy on `train_full`, the data they were trained on. What does the difference suggest the soft targets are doing? (Lecture: "Why does it work?")

---

## Part 2 — Temperature and α, in two regimes (about 8 minutes)

```bash
python3 python/part2_sweeps.py                 # both regimes
python3 python/part2_sweeps.py --regime full   # or one at a time, ~4 min each
python3 python/part2_sweeps.py --regime 1k
```

The same three experiments, run twice, first with `train_full` (55,000 labels) and then with `labeled_1k` (1,000 labels, 4,300 steps as always):

1. With α = 0.9, sweep T ∈ {1, 2, 4, 8, 16}.
2. With T = 4, sweep α ∈ {0, 0.25, 0.5, 0.75, 0.9, 1.0}.
3. Ablation: T = 8, α = 0.9 **without** the T² factor (`train(..., t2=False)`), compared with the same configuration with it.

Finally the script picks the (T, α) with the best mean **validation** accuracy in each regime, and reports that choice's test accuracy.

*Predict first:* the best T in each regime. Will they be the same?

**Questions**

- **Q2.1** Describe the accuracy-vs-T curve in each regime. Why might a small student do best at a low temperature when labels are plentiful, and a higher one when they're scarce? (Hint: at high T, matching the teacher means matching nearly all of its logits, not just the top few.)
- **Q2.2** What does α = 1.0 mean? How does it compare with α = 0.9 in each regime, and what does that tell you about how much the true labels matter here?
- **Q2.3** Explain the result of the T² ablation in each regime, using the lecture's explanation of the T² factor and your Part 0, item 5 numbers. Work out the effective weight of the soft term relative to the hard term at T = 8 without T². Is "removing T² hurts" always true, or does it depend on whether the soft term was helping?
- **Q2.4** Which (T, α) did validation choose in each regime? Why choose them on the validation set and not on the test set?

---

## Part 3 — Few labels, many images: the main result (about 1 minute)

```bash
python3 python/part3_low_data.py
python3 python/part3_low_data.py --T 8    # optional: if Part 2 suggests another T
```

Each configuration gets the same compute: 4,300 optimizer steps of 128 images (10 epochs of `train_full`), 3 seeds.

| Config | Training inputs | Targets |
|---|---|---|
| 3a | `labeled_1k` | labels only (α = 0) |
| 3b | `labeled_1k` | KD + labels (α = 0.9) |
| 3c | `labeled_1k` + `unlabeled_rest` | KD + labels on the labeled images (α = 0.9), KD only on the unlabeled ones (α = 1) |
| 3d | `train_full` | labels only (α = 0): every label available |

In 3c, batches are drawn from all 55,000 images, so about 2 images in each batch of 128 have labels. The script also saves each configuration's seed-0 student to `checkpoints/student_<cfg>.pt` for Part 5.

*Predict first:* the four test accuracies. Write them down: this is the lecture's second poll question.

**Questions**

- **Q3.1** Rank the four configurations. How much of the gap between 3a and 3d does 3c close? (The script prints it.)
- **Q3.2** Where did the student's knowledge come from in 3c, given that 54,000 of its 55,000 training images had no labels?
- **Q3.3** Why does 3b, with the teacher but only the 1,000 labeled images, gain so much less than 3c? What did 3c's extra images provide that the teacher's soft targets on 1,000 images couldn't?
- **Q3.4** Name a situation in embedded deployment where you'd have many unlabeled inputs but few labels (for example, a sensor you've just deployed). How would you use KD there? What would you need on the device, and what only on a server?

---

## Part 4 — Hinton's "missing digit" experiment (about 30 seconds)

```bash
python3 python/part4_missing_digit.py
```

The transfer set is `train_no3`: no 3s at all. 3 seeds each of:

- **KD:** T = 8, α = 1.0 (soft targets only)
- **CE:** labels only, for contrast

For each student the script reports overall test accuracy, accuracy on the 1,010 test-set 3s, and how many test images it predicted as 3. Then it applies Hinton's bias correction: add b to the student's class-3 output bias, for b = 0, 0.5, …, 10, keep the b with the best **validation** accuracy, and report test accuracy with that b.

*Predict first:* the KD student's accuracy on 3s, before the correction.

**Questions**

- **Q4.1** How can a student that never saw a 3 classify 3s? (Hint: what did the teacher's soft targets say about other digits' resemblance to a 3? Look back at Part 0, item 3.)
- **Q4.2** Hinton et al. found that the distilled student's errors on 3s were largely due to a too-low bias for class 3. Does the bias correction confirm that here? What does the correction trade away on the other digits?
- **Q4.3** Why does the CE-only student fail on 3s, even with a bias correction?

---

## Part 5 — Back to the board

### Step 1 — Does it fit? (laptop, instant)

```bash
python3 python/part5_fit.py
```

It prints the weight bytes of the teacher and the student in fp32 (4 B per parameter) and int8 (1 B per weight and 4 B per bias, the quantization activity's int32 biases), and their peak activation memory, against the F407's 1 MB of Flash and 128 KB of SRAM.

*Predict first:* does the teacher fit in fp32? In int8?

### Step 2 — Run them on the board (optional)

This step uses the quantization activity's int8 flow unchanged: PT2E post-training quantization with ExecuTorch's `CortexMQuantizer`, then lowering to CMSIS-NN kernels. Each command exports a build, compiles the firmware, flashes it, and captures the board's report over the UART (under 10 s each):

```bash
scripts/run_board.sh teacher int8
scripts/run_board.sh 3c int8      # the Part 3c student: 1,000 labels + KD
scripts/run_board.sh 3a int8      # the Part 3a student: 1,000 labels, CE only
scripts/run_board.sh 3c fp32      # the 3c student, fp32, portable kernels
```

`UART=/dev/ttyUSBn` picks another adapter. The board classifies 200 embedded MNIST test images (20 per digit), checks each prediction against the laptop's, and times `method.execute()` with the DWT cycle counter. Then rerun Step 1: `part5_fit.py` adds the board's numbers to its table.

*Predict first:* which numbers will differ between `3c-int8` and `3a-int8`?

### Step 3 — The teacher in fp32 (optional)

```bash
scripts/run_board.sh teacher fp32
```

*Predict first:* what happens?

**Questions**

- **Q5.1** Fill in a mini-scorecard comparing the teacher (fp32), the teacher (int8), the student (fp32), and the student (int8): Flash, SRAM, latency (if you ran the board), test accuracy.
- **Q5.2** Compare `3c-int8` and `3a-int8` line by line. What changed and what didn't? Which of the lecture's claims about distillation does this test?
- **Q5.3** Which compression technique(s) gave the student its size, and which gave it its accuracy?
- **Q5.4** (If you ran Steps 2–3.) `export_mlp.py` prints a planned arena for each build. Why is the fp32 student's arena so much larger than the int8 student's, when its activations are only 4× larger? (Look at the per-op profile.) What does the fp32 teacher's arena mean for SRAM, on top of its Flash problem?

---

## Part 6 (extension) — KD in the pruning pipeline (about 7 minutes; `--long` adds about 13)

```bash
python3 python/part6_pipeline.py
python3 python/part6_pipeline.py --long   # optional
```

Back to Model B2 and Fashion-MNIST, from the pruning and quantization activities. The **teacher** is variant A, the dense `ModelB2(16, 32)` (84.7%). The **student** is A with half of each conv layer's filters removed: `ModelB2(8, 16)`, exactly as the pruning lab made variant C, but before any fine-tuning.

- **Step 1** prints the training-set and test accuracy of A and of the provided C, and the slimmed model's accuracy before fine-tuning.
- **Step 2** recovers the slimmed model with the pruning lab's 5-epoch fine-tune (Adam 10⁻³, cosine schedule), 3 seeds each, three ways: **CE** with all labels (how C was made), **KD** with all labels (T = 4, α = 0.9), and **KD with every label hidden** (the soft term alone).
- **`--long`** runs one seed each of a 20-epoch fine-tune, CE vs. KD, and of `ModelB2(8, 16)` trained from scratch for 35 epochs (the pruning lab's extension), CE vs. KD.

*Predict first,* after Step 1 prints but before Step 2 finishes: will KD beat the CE fine-tune?

**Questions**

- **Q6.1** What do Step 1's training-set and test accuracies tell you about A and C? Compare them with the Part 1 student's training-set and test accuracy on MNIST, and with Part 3a's.
- **Q6.2** Does KD beat the CE fine-tune here? Explain the result using Q6.1, and contrast it with Part 3.
- **Q6.3** How close does KD come to the CE fine-tune **without any labels**? When would that matter in practice?
- **Q6.4** At the start of the lecture, we asked whether `ModelB2(8, 16)` was limited by its capacity or by its training signal. What's your answer, and what evidence supports it?
- **Q6.5** (With `--long`.) What does a longer fine-tune do, with and without KD? Compare the 20-epoch fine-tune with the 35-epoch from-scratch runs. Does this change the pruning activity's conclusion that pruning plus fine-tuning was no better than training from scratch?

---

## Wrap-up

Fill in one line per part: your prediction, the measured result, and one sentence on why. Then look back at the lecture's two poll questions and answer them with your own numbers.

| Part | Prediction | Measured | Why |
|---|---|---|---|
| 0: teacher confidence at T = 1 | | | |
| 1: KD − CE, full data | | | |
| 2: best T, full vs. 1k | | | |
| 3: 3a / 3b / 3c / 3d | | | |
| 4: 3s accuracy, before / after b | | | |
| 5: `3c-int8` vs. `3a-int8` | | | |
| 6: KD vs. CE fine-tune; KD without labels | | | |
