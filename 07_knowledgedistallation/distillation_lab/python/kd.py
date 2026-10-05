"""
kd.py -- the distillation loss and the one training loop every part uses.

    kd_loss_slide(...)   the lecture's 10-line version, verbatim
    kd_loss(...)         the same loss, computed per image, so a batch can
                         mix labeled and unlabeled images (Part 3c); on a
                         fully labeled batch it equals kd_loss_slide
    train(...)           Adam, fixed number of steps, optional teacher
    evaluate(...)        accuracy on a split

The fixed recipe (README "The training recipe"): Adam, lr 1e-3, batch 128,
10 epochs of train_full = 4,300 steps, no augmentation, no lr schedule.
"""

import math
import os

import torch
import torch.nn.functional as F

from data import UNLABELED

torch.set_num_threads(int(os.environ.get("KD_THREADS", "4")))

STEPS_PER_EPOCH = math.ceil(55000 / 128)      # 430
DEFAULT_STEPS = 10 * STEPS_PER_EPOCH           # 4,300


def kd_loss_slide(s_logits, t_logits, y, T=4.0, alpha=0.9):
    soft = F.kl_div(F.log_softmax(s_logits / T, dim=1),
                    F.softmax(t_logits / T, dim=1),
                    reduction="batchmean") * (T * T)
    hard = F.cross_entropy(s_logits, y)
    return alpha * soft + (1 - alpha) * hard


def kd_loss(s_logits, t_logits, y, T=4.0, alpha=0.9, t2=True):
    """alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * CE(y, student),
    per image, then averaged over the batch. An image with y == UNLABELED has
    no hard term, so it gets the soft term alone (alpha = 1 for that image).
    t2=False drops the T^2 factor (the Part 2 ablation)."""
    soft = F.kl_div(F.log_softmax(s_logits / T, dim=1),
                    F.softmax(t_logits / T, dim=1),
                    reduction="none").sum(dim=1)        # KL per image
    if t2:
        soft = soft * (T * T)
    labeled = y != UNLABELED
    hard = F.cross_entropy(s_logits, y.clamp(min=0), reduction="none")
    a = torch.where(labeled, torch.tensor(alpha), torch.tensor(1.0))
    return (a * soft + (1 - a) * hard * labeled).mean()


def evaluate(model, x, y, batch: int = 5000) -> float:
    model.eval()
    with torch.no_grad():
        correct = sum((model(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item()
                      for i in range(0, len(x), batch))
    return correct / len(x)


def train(student, x, y, teacher=None, T=4.0, alpha=0.9, steps=DEFAULT_STEPS, lr=1e-3,
          batch=128, seed=0, val=None, eval_every=STEPS_PER_EPOCH, t2=True,
          reset=True, cosine=False):
    """Trains `student` in place for `steps` optimizer steps on (x, y) and
    returns a log {"step": [...], "val_acc": [...]} (empty without `val`).

    teacher=None or alpha=0 is plain cross-entropy. Otherwise the loss is
    kd_loss, with the teacher in eval mode and under no_grad, called on the
    *same* batch tensor as the student.

    Batches are drawn from a reshuffled pass over the data, cycling for as
    many passes as `steps` needs: 10 epochs of train_full, or 550 passes
    over labeled_1k (Part 3 holds steps fixed, not epochs).

    The seed sets both the student's initialization and the batch order.
    reset=False keeps the student's current weights (fine-tuning, Part 6);
    cosine=True anneals the lr to zero over the run, as the pruning lab's
    training did (Part 6 only; Parts 0-5 use a constant lr)."""
    torch.manual_seed(seed)
    if reset:
        for m in student.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    use_teacher = teacher is not None and alpha > 0
    if (y == UNLABELED).any() and not use_teacher:
        raise ValueError("unlabeled images need a teacher")
    if use_teacher:
        teacher.eval()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps) if cosine else None
    g = torch.Generator().manual_seed(seed)
    log = {"step": [], "val_acc": []}
    order, pos = torch.randperm(len(x), generator=g), 0
    for step in range(1, steps + 1):
        if pos + batch > len(order):
            order, pos = torch.randperm(len(x), generator=g), 0
        idx = order[pos:pos + batch]
        pos += batch
        xb, yb = x[idx], y[idx]
        student.train()
        s_logits = student(xb)
        if use_teacher:
            with torch.no_grad():
                t_logits = teacher(xb)
            loss = kd_loss(s_logits, t_logits, yb, T, alpha, t2)
        else:
            loss = F.cross_entropy(s_logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if val is not None and (step % eval_every == 0 or step == steps):
            log["step"].append(step)
            log["val_acc"].append(evaluate(student, *val))
    student.eval()
    return log
