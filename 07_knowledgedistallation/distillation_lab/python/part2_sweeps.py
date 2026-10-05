"""
part2_sweeps.py -- Part 2: temperature, alpha, and the T^2 factor, with
plenty of labels and with few. About 8 minutes for both regimes.

    python3 python/part2_sweeps.py                 # both regimes
    python3 python/part2_sweeps.py --regime full   # just one (~4 min)

Regimes (training data):
    full   train_full, 55,000 labeled images
    1k     labeled_1k, 1,000 labeled images (4,300 steps = 550 passes)

In each regime, 3 seeds per point:
    T sweep      alpha = 0.9, T in {1, 2, 4, 8, 16}
    alpha sweep  T = 4, alpha in {0, 0.25, 0.5, 0.75, 0.9, 1.0}
    ablation     T = 8, alpha = 0.9 without the T^2 factor
Then picks (T, alpha) by mean *validation* accuracy and reports that
choice's test accuracy. Writes results/part2.json.
"""

import argparse

from data import splits
from expt import SEEDS, Timer, fmt, mean_std, save
from kd import evaluate, train
from models import StudentMLP, load_teacher

TEMPS = [1, 2, 4, 8, 16]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
DATA = {"full": "train_full", "1k": "labeled_1k"}


def run_point(s, teacher, data, T, alpha, t2=True):
    vals, tests = [], []
    for seed in SEEDS:
        st = StudentMLP()
        train(st, *s[data], teacher=teacher if alpha > 0 else None, T=T, alpha=alpha, seed=seed, t2=t2)
        vals.append(evaluate(st, *s["val"]))
        tests.append(evaluate(st, *s["test"]))
    return {"T": T, "alpha": alpha, "t2": t2, "val": vals, "test": tests}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["full", "1k", "both"], default="both")
    args = ap.parse_args()
    regimes = list(DATA) if args.regime == "both" else [args.regime]

    s = splits()
    teacher = load_teacher()
    timer = Timer()
    out = {}
    for regime in regimes:
        data = DATA[regime]
        print(f"\n=== Regime {regime}: {data} ({len(s[data][0])} labeled images) ===")
        points = {}

        def point(T, alpha, t2=True):
            key = (T, alpha, t2)
            if key not in points:
                points[key] = run_point(s, teacher, data, T, alpha, t2)
                p = points[key]
                print(f"  T={T:<3g} alpha={alpha:<4g}{'' if t2 else ' no T^2'}: "
                      f"val {fmt(p['val'])}  test {fmt(p['test'])}  ({timer})", flush=True)
            return points[key]

        t_sweep = [point(T, 0.9) for T in TEMPS]
        a_sweep = [point(4, a) for a in ALPHAS]
        ablation = {"with": point(8, 0.9), "without": point(8, 0.9, t2=False)}
        # pick by validation accuracy among everything run with T^2
        candidates = [p for (T, a, t2), p in points.items() if t2]
        best = max(candidates, key=lambda p: mean_std(p["val"])[0])
        out[regime] = {"T_sweep": t_sweep, "alpha_sweep": a_sweep, "ablation": ablation,
                       "best_by_val": best}

        print(f"\nRegime {regime}: test accuracy, mean ± std over 3 seeds")
        print("  T sweep (alpha 0.9):   " + "  ".join(f"T={p['T']}: {fmt(p['test'])}" for p in t_sweep))
        print("  alpha sweep (T 4):     " + "  ".join(f"a={p['alpha']}: {fmt(p['test'])}" for p in a_sweep))
        print(f"  T=8, alpha=0.9:        with T^2 {fmt(ablation['with']['test'])}   "
              f"without {fmt(ablation['without']['test'])}")
        print(f"  best by validation:    T={best['T']}, alpha={best['alpha']}: "
              f"val {fmt(best['val'])}, test {fmt(best['test'])}")
    save("part2" if args.regime == "both" else f"part2_{args.regime}", out)


if __name__ == "__main__":
    main()
