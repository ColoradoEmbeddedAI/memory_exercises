"""
part3_low_data.py -- Part 3: 1,000 labels, with and without 54,000
unlabeled images. About 1 minute.

    python3 python/part3_low_data.py          # T = 4
    python3 python/part3_low_data.py --T 8    # if Part 2 says so

Every configuration gets the same compute: 4,300 optimizer steps of 128
images (10 epochs of train_full), 3 seeds.

    3a  labeled_1k                    labels only (alpha = 0)
    3b  labeled_1k                    KD + labels (alpha = 0.9)
    3c  labeled_1k + unlabeled_rest   KD + labels on the 1k labeled images
                                      (alpha = 0.9), KD only on the 54k
                                      unlabeled ones (alpha = 1). Batches
                                      are drawn from all 55k, so about 2
                                      images in each batch of 128 have labels.
    3d  train_full                    labels only (alpha = 0): the
                                      reference, every label available

Saves each configuration's seed-0 student to checkpoints/student_<cfg>.pt
(Part 5 deploys them) and writes results/part3.json. With --T other than 4,
the files are student_<cfg>_T<T>.pt and results/part3_T<T>.json, so the
default run's students stay in place.
"""

import argparse
import os

import torch

from data import UNLABELED, splits
from expt import SEEDS, Timer, fmt, mean_std, save
from kd import evaluate, train
from models import CHECKPOINT_DIR, StudentMLP, load_teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=4.0)
    args = ap.parse_args()

    s = splits()
    teacher = load_teacher()
    mixed = (torch.cat([s["labeled_1k"][0], s["unlabeled_rest"][0]]),
             torch.cat([s["labeled_1k"][1], s["unlabeled_rest"][1]]))
    configs = {
        "3a": (s["labeled_1k"], dict(alpha=0.0)),
        "3b": (s["labeled_1k"], dict(teacher=teacher, T=args.T, alpha=0.9)),
        "3c": (mixed, dict(teacher=teacher, T=args.T, alpha=0.9)),
        "3d": (s["train_full"], dict(alpha=0.0)),
    }
    suffix = "" if args.T == 4.0 else f"_T{args.T:g}"
    timer = Timer()
    out = {"T": args.T}
    for name, ((x, y), kw) in configs.items():
        tests = []
        for seed in SEEDS:
            st = StudentMLP()
            train(st, x, y, seed=seed, **kw)
            tests.append(evaluate(st, *s["test"]))
            if seed == 0:
                torch.save(st.state_dict(), os.path.join(CHECKPOINT_DIR, f"student_{name}{suffix}.pt"))
        labeled = int((y != UNLABELED).sum())
        out[name] = {"images": len(x), "labeled": labeled, "test": tests}
        print(f"  {name}: {len(x):6d} images, {labeled:6d} labeled: test {fmt(tests)}  ({timer})", flush=True)

    m = {k: mean_std(out[k]["test"])[0] for k in ("3a", "3b", "3c", "3d")}
    closed = (m["3c"] - m["3a"]) / (m["3d"] - m["3a"])
    out["gap_closed_by_3c"] = closed
    print(f"\nPart 3 (T = {args.T:g}), test accuracy, 3 seeds")
    for k in ("3a", "3b", "3c", "3d"):
        print(f"  {k}  {fmt(out[k]['test'])}")
    print(f"3c closes {100 * closed:.0f}% of the gap between 3a and 3d")
    save(f"part3{suffix}", out)


if __name__ == "__main__":
    main()
