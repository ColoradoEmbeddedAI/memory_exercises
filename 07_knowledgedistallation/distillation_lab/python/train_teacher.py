"""
train_teacher.py -- how checkpoints/teacher_lenet300100.pt was made.

    python3 python/train_teacher.py        # ~1 minute on a laptop CPU

You don't need to run this: the checkpoint is provided. LeNet-300-100 with
dropout 0.2, trained on train_full with the activity's recipe (Adam, lr
1e-3, batch 128), for 20 epochs instead of 10. Prints test accuracy and how
confident the teacher is, since an over-confident teacher (max softmax
~1.0 on everything) has little to teach beyond the labels.
"""

import torch

from data import splits
from kd import STEPS_PER_EPOCH, evaluate, train
from models import TEACHER_CKPT, LeNet300100, count_params

EPOCHS = 20
SEED = 0


def main():
    s = splits()
    teacher = LeNet300100()
    log = train(teacher, *s["train_full"], steps=EPOCHS * STEPS_PER_EPOCH, seed=SEED, val=s["val"])
    for step, acc in zip(log["step"], log["val_acc"]):
        print(f"epoch {step // STEPS_PER_EPOCH:2d}  val_acc={acc:.4f}")
    torch.save(teacher.state_dict(), TEACHER_CKPT)
    print(f"params {count_params(teacher)}")
    print(f"test accuracy {evaluate(teacher, *s['test']):.4f}")
    with torch.no_grad():
        p = torch.softmax(teacher(s["train_full"][0]), 1).max(1).values
    print(f"max softmax on train_full: median {p.median():.5f}, "
          f"{(p > 0.9999).float().mean():.1%} of images above 0.9999")
    print(f"saved {TEACHER_CKPT}")


if __name__ == "__main__":
    main()
