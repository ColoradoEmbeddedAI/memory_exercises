"""
part1_baseline_vs_kd.py -- Part 1: the student with and without the teacher,
on all 55,000 labels. About 30 seconds.

    python3 python/part1_baseline_vs_kd.py

Two configurations, 3 seeds each, the fixed recipe (Adam 1e-3, batch 128,
10 epochs = 4,300 steps):
    CE   cross-entropy on the labels only (alpha = 0)
    KD   kd_loss with T = 4, alpha = 0.9

Prints test accuracy (mean ± std over seeds), training-set accuracy (how
well each student fits the labels it trained on), and validation accuracy
after every epoch, averaged over seeds. Writes results/part1.json.
"""

from data import splits
from expt import SEEDS, Timer, fmt, mean_std, save
from kd import STEPS_PER_EPOCH, evaluate, train
from models import StudentMLP, load_teacher

CONFIGS = {"CE": dict(teacher=None), "KD": dict(T=4.0, alpha=0.9)}


def main():
    s = splits()
    teacher = load_teacher()
    out = {}
    timer = Timer()
    for name, cfg in CONFIGS.items():
        kw = dict(cfg)
        if "T" in kw:
            kw["teacher"] = teacher
        runs = []
        for seed in SEEDS:
            st = StudentMLP()
            log = train(st, *s["train_full"], seed=seed, val=s["val"], **kw)
            runs.append({"seed": seed, "test": evaluate(st, *s["test"]),
                         "train": evaluate(st, *s["train_full"]), "val_curve": log["val_acc"]})
            print(f"  {name} seed {seed}: test {100 * runs[-1]['test']:.2f}%  ({timer})", flush=True)
        out[name] = runs

    print("\nPart 1 results (3 seeds)")
    print(f"{'':4s} {'test':>16s} {'train_full':>16s}")
    for name, runs in out.items():
        print(f"{name:4s} {fmt([r['test'] for r in runs]):>16s} {fmt([r['train'] for r in runs]):>16s}")
    gain = mean_std([r["test"] for r in out["KD"]])[0] - mean_std([r["test"] for r in out["CE"]])[0]
    print(f"KD - CE: {100 * gain:+.2f} points")

    print("\nValidation accuracy after each epoch (mean of 3 seeds)")
    print("epoch " + " ".join(f"{e:6d}" for e in range(1, 11)))
    for name, runs in out.items():
        curve = [mean_std([r["val_curve"][e] for r in runs])[0] for e in range(10)]
        print(f"{name:5s} " + " ".join(f"{100 * v:6.2f}" for v in curve))
    out["steps_per_epoch"] = STEPS_PER_EPOCH
    save("part1", out)


if __name__ == "__main__":
    main()
