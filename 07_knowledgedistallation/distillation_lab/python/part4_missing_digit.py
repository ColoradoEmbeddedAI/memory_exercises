"""
part4_missing_digit.py -- Part 4: Hinton's "missing digit" experiment. About
30 seconds.

    python3 python/part4_missing_digit.py

The transfer set is train_no3: train_full with every 3 removed. 3 seeds of:
    KD   T = 8, alpha = 1.0 (soft targets only)
    CE   labels only

For each student: overall test accuracy, and accuracy on the 1,010 test 3s.
Then the bias correction: add b to the student's class-3 output bias, for
b = 0, 0.5, ..., 10, and keep the b with the best *validation* accuracy
(val has 3s in it). Report test accuracy with that b. Writes
results/part4.json.
"""

import torch

from data import splits
from expt import SEEDS, Timer, fmt, save
from kd import evaluate, train
from models import StudentMLP, load_teacher

BIASES = [0.5 * k for k in range(21)]


def with_bias(st, b, fn):
    with torch.no_grad():
        st.fc2.bias[3] += b
        try:
            return fn()
        finally:
            st.fc2.bias[3] -= b


def main():
    s = splits()
    teacher = load_teacher()
    xt, yt = s["test"]
    threes = (xt[yt == 3], yt[yt == 3])
    configs = {"KD": dict(teacher=teacher, T=8.0, alpha=1.0), "CE": dict(alpha=0.0)}
    timer = Timer()
    out = {"n_test_3s": len(threes[0]), "biases": BIASES}
    for name, kw in configs.items():
        runs = []
        for seed in SEEDS:
            st = StudentMLP()
            train(st, *s["train_no3"], seed=seed, **kw)
            curve = [with_bias(st, b, lambda: (evaluate(st, *s["val"]), evaluate(st, *threes)))
                     for b in BIASES]
            k = max(range(len(BIASES)), key=lambda i: curve[i][0])
            b = BIASES[k]
            runs.append({
                "seed": seed,
                "test": evaluate(st, xt, yt), "test_3s": evaluate(st, *threes),
                "best_b": b,
                "test_b": with_bias(st, b, lambda: evaluate(st, xt, yt)),
                "test_3s_b": with_bias(st, b, lambda: evaluate(st, *threes)),
                "val_curve": [c[0] for c in curve], "threes_curve": [c[1] for c in curve],
            })
            with torch.no_grad():
                runs[-1]["predicted_3"] = int((st(xt).argmax(1) == 3).sum())
            r = runs[-1]
            print(f"  {name} seed {seed}: test {100 * r['test']:.2f}%, 3s {100 * r['test_3s']:.1f}%, "
                  f"{r['predicted_3']} test images predicted 3; best b = {b:g}: "
                  f"test {100 * r['test_b']:.2f}%, 3s {100 * r['test_3s_b']:.1f}%  ({timer})", flush=True)
        out[name] = runs

    print(f"\nPart 4: students that never saw a 3 (3 seeds; {out['n_test_3s']} test 3s)")
    print(f"{'':4s} {'test, all':>16s} {'test, 3s only':>16s} {'best b':>8s} "
          f"{'all, with b':>16s} {'3s, with b':>16s}")
    for name in configs:
        r = out[name]
        print(f"{name:4s} {fmt([x['test'] for x in r]):>16s} {fmt([x['test_3s'] for x in r]):>16s} "
              f"{'/'.join(format(x['best_b'], 'g') for x in r):>8s} "
              f"{fmt([x['test_b'] for x in r]):>16s} {fmt([x['test_3s_b'] for x in r]):>16s}")
    save("part4", out)


if __name__ == "__main__":
    main()
