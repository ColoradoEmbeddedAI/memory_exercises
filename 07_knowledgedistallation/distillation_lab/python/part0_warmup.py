"""
part0_warmup.py -- Part 0: the teacher, the student, and the loss. No
training; about 10 seconds.

    python3 python/part0_warmup.py

Prints:
  1. parameter counts of the teacher and the student, by code and by formula
  2. the teacher's test accuracy, and how confident it is at T = 1 and T = 4
  3. the teacher's softmax at T = 1, 4 and 10 for five test images
  4. the loss on one real batch, correct and with each pitfall from the
     lecture's checklist
  5. how the soft term's gradient scales with T, with and without T^2

Writes results/part0.json.
"""

import warnings

import torch
import torch.nn.functional as F

from data import splits
from expt import pct, save
from kd import evaluate, kd_loss, kd_loss_slide
from models import StudentMLP, count_params, load_teacher

# Five test images: two the teacher is sure of, three where a second digit
# gets visible probability (index into the 10k test set).
IMAGES = [0, 3, 7, 149, 412]

# Pitfall 4 uses reduction="mean" on purpose; PyTorch warns about it.
warnings.filterwarnings("ignore", message="reduction: 'mean'")
TEMPS = [1, 4, 10]


def softmax_T(logits, T):
    return F.softmax(logits / T, dim=1)


def main():
    s = splits()
    torch.manual_seed(0)                    # the untrained student in items 4 and 5
    teacher, student = load_teacher(), StudentMLP()
    out = {}

    # 1. parameter counts
    by_formula = {"teacher": 784 * 300 + 300 + 300 * 100 + 100 + 100 * 10 + 10,
                  "student": 784 * 30 + 30 + 30 * 10 + 10}
    print("1. Parameters")
    for name, m in (("teacher", teacher), ("student", student)):
        print(f"   {name:8s} by code {count_params(m):7,d}   by formula {by_formula[name]:7,d}")
    ratio = count_params(teacher) / count_params(student)
    print(f"   the student is {ratio:.1f}x smaller")
    out["params"] = {"teacher": count_params(teacher), "student": count_params(student), "ratio": ratio}

    # 2. accuracy and confidence
    xt, yt = s["test"]
    with torch.no_grad():
        zt = teacher(xt)
    acc = evaluate(teacher, xt, yt)
    print(f"\n2. Teacher test accuracy: {pct(acc)}")
    conf = {}
    for T in (1, 4):
        p = softmax_T(zt, T)
        top = p.max(1).values
        conf[T] = {"median_max_p": top.median().item(),
                   "frac_above_0.9999": (top > 0.9999).float().mean().item(),
                   "mean_mass_off_top": (1 - top).mean().item()}
        print(f"   T = {T}: median max probability {conf[T]['median_max_p']:.4f}; "
              f"{pct(conf[T]['frac_above_0.9999'])} of test images above 0.9999; "
              f"average probability on the other 9 digits {conf[T]['mean_mass_off_top']:.4f}")
    out["teacher_test_acc"] = acc
    out["confidence"] = conf

    # 3. soft targets for five images
    print("\n3. Teacher softmax for five test images (digits 0..9)")
    soft = {}
    for i in IMAGES:
        print(f"   test[{i}], label {int(yt[i])}")
        soft[i] = {}
        for T in TEMPS:
            p = softmax_T(zt[i:i + 1], T)[0]
            soft[i][T] = [round(v, 5) for v in p.tolist()]
            print(f"     T = {T:2d}: " + " ".join(f"{v:.3f}" for v in p.tolist()))
    out["label"] = {i: int(yt[i]) for i in IMAGES}
    out["soft_targets"] = soft

    # 4. the loss and its pitfalls, on one real batch
    xb, yb = s["train_full"][0][:128], s["train_full"][1][:128]
    T, alpha = 4.0, 0.9
    with torch.no_grad():
        s_logits = student(xb)
        t_logits = teacher(xb)
        teacher.train()                     # pitfall: dropout on
        t_train = teacher(xb)
        teacher.eval()
    log_ps, pt = F.log_softmax(s_logits / T, 1), F.softmax(t_logits / T, 1)
    hard = F.cross_entropy(s_logits, yb)
    variants = {
        "correct (kd_loss_slide)": kd_loss_slide(s_logits, t_logits, yb, T, alpha),
        "correct (kd_loss, per image)": kd_loss(s_logits, t_logits, yb, T, alpha),
        "no T^2": alpha * F.kl_div(log_ps, pt, reduction="batchmean") + (1 - alpha) * hard,
        "reduction='mean' (not batchmean)":
            alpha * F.kl_div(log_ps, pt, reduction="mean") * T * T + (1 - alpha) * hard,
        "arguments swapped (teacher log-probs, student probs)":
            alpha * F.kl_div(F.log_softmax(t_logits / T, 1), F.softmax(s_logits / T, 1),
                             reduction="batchmean") * T * T + (1 - alpha) * hard,
        "probabilities where log-probabilities belong":
            alpha * F.kl_div(F.softmax(s_logits / T, 1), pt, reduction="batchmean") * T * T
            + (1 - alpha) * hard,
        "teacher in train mode (dropout on)": kd_loss_slide(s_logits, t_train, yb, T, alpha),
    }
    print(f"\n4. The loss on one batch of 128 (untrained student, T = {T:g}, alpha = {alpha})")
    for name, v in variants.items():
        print(f"   {name:55s} {v.item():9.4f}")
    drift = F.kl_div(F.log_softmax(t_train / T, 1), pt, reduction="batchmean").item()
    agree = (t_train.argmax(1) == t_logits.argmax(1)).float().mean().item()
    print(f"   teacher train vs eval mode: soft-target KL {drift:.4f}, same top digit on {pct(agree)}")
    out["loss_variants"] = {k: v.item() for k, v in variants.items()}
    out["teacher_train_mode"] = {"kl": drift, "same_top": agree}

    # 5. gradient of the soft term with respect to the student's logits
    print("\n5. Size of the soft term's gradient (mean |d loss / d student logit|)")
    print("      T    without T^2      with T^2")
    grads = {}
    for T in (1, 2, 4, 8, 16):
        z = s_logits.clone().requires_grad_(True)
        F.kl_div(F.log_softmax(z / T, 1), F.softmax(t_logits / T, 1), reduction="batchmean").backward()
        g = z.grad.abs().mean().item()
        grads[T] = {"without": g, "with": g * T * T}
        print(f"   {T:4d}   {g:12.6f}  {g * T * T:12.6f}")
    out["soft_grad"] = grads
    save("part0", out)


if __name__ == "__main__":
    main()
