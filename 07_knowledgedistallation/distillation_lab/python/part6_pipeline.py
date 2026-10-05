"""
part6_pipeline.py -- Part 6 (extension): KD in the compression pipeline,
on Model B2 and Fashion-MNIST from the pruning and quantization activities.
About 7 minutes; --long adds about 13 more.

    python3 python/part6_pipeline.py
    python3 python/part6_pipeline.py --long

Teacher: variant A, the dense ModelB2(16, 32) (84.7%). Student: A
structurally pruned to ModelB2(8, 16) exactly as the pruning lab made C
(fashion_b2.slim()), before any fine-tuning.

  Step 1  capacity check: training-set vs test accuracy of A and C
  Step 2  recover the pruned model with a 5-epoch fine-tune (the pruning
          lab's: Adam 1e-3, cosine schedule), 3 seeds each:
            CE         all 55k labels, cross-entropy (how C was made)
            KD         all 55k labels, KD with T = 4, alpha = 0.9
            KD, no labels   the 55k images with labels hidden, KD only
  --long  one seed each of: a 20-epoch fine-tune, CE vs KD; and
          ModelB2(8, 16) trained from scratch for 35 epochs (lr 3e-3,
          the pruning lab's from-scratch extension), CE vs KD

Writes results/part6.json (part6_long.json for --long).
"""

import argparse

import torch

from data import UNLABELED
from expt import SEEDS, Timer, fmt, save
from fashion_b2 import ModelB2, fashion_splits, load_A, load_C, slim
from kd import STEPS_PER_EPOCH, evaluate, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true")
    args = ap.parse_args()

    s = fashion_splits()
    A, C = load_A(), load_C()
    x, y = s["train"]
    timer = Timer()

    if args.long:
        out = {}
        runs = [("finetune20_CE", lambda: slim(A), {}, 20, 1e-3, False),
                ("finetune20_KD", lambda: slim(A), dict(teacher=A, T=4.0, alpha=0.9), 20, 1e-3, False),
                ("scratch35_CE", lambda: ModelB2(8, 16), {}, 35, 3e-3, True),
                ("scratch35_KD", lambda: ModelB2(8, 16), dict(teacher=A, T=4.0, alpha=0.9), 35, 3e-3, True)]
        for name, make, kw, epochs, lr, reset in runs:
            m = make()
            train(m, x, y, steps=epochs * STEPS_PER_EPOCH, lr=lr, cosine=True, reset=reset, seed=0, **kw)
            out[name] = {"test": evaluate(m, *s["test"]), "train": evaluate(m, x, y)}
            print(f"  {name}: test {100 * out[name]['test']:.2f}%, train {100 * out[name]['train']:.2f}%  "
                  f"({timer})", flush=True)
        save("part6_long", out)
        return

    out = {"capacity": {}}
    print("Step 1: capacity check")
    for name, m in (("A", A), ("C", C)):
        tr, te = evaluate(m, x, y), evaluate(m, *s["test"])
        out["capacity"][name] = {"train": tr, "test": te}
        print(f"  {name}: train {100 * tr:.2f}%  test {100 * te:.2f}%  gap {100 * (tr - te):.2f} points")
    out["slimmed_before_finetune"] = evaluate(slim(A), *s["test"])
    print(f"  A slimmed to ModelB2(8, 16), before fine-tuning: test "
          f"{100 * out['slimmed_before_finetune']:.2f}%")

    print("\nStep 2: recover the pruned model (5-epoch fine-tune, 3 seeds)")
    hidden = torch.full_like(y, UNLABELED)
    configs = {"CE": (y, {}),
               "KD": (y, dict(teacher=A, T=4.0, alpha=0.9)),
               "KD, no labels": (hidden, dict(teacher=A, T=4.0, alpha=0.9))}
    for name, (labels, kw) in configs.items():
        tests, trains = [], []
        for seed in SEEDS:
            m = slim(A)
            train(m, x, labels, steps=5 * STEPS_PER_EPOCH, reset=False, cosine=True, seed=seed, **kw)
            tests.append(evaluate(m, *s["test"]))
            trains.append(evaluate(m, x, y))
        out[name] = {"test": tests, "train": trains}
        print(f"  {name:14s} test {fmt(tests)}  train {fmt(trains)}  ({timer})", flush=True)
    save("part6", out)


if __name__ == "__main__":
    main()
