"""
expt.py -- small helpers shared by the part scripts: seeds, mean +- std,
and results/<part>.json.
"""

import json
import os
import statistics
import time

from models import RESULTS_DIR

SEEDS = (0, 1, 2)


def pct(v: float) -> str:
    return f"{100 * v:.2f}%"


def mean_std(vals):
    """(mean, sample std) of a list of accuracies, as fractions."""
    m = statistics.mean(vals)
    return m, (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def fmt(vals) -> str:
    m, s = mean_std(vals)
    return f"{100 * m:.2f} ± {100 * s:.2f}%"


def save(name: str, obj) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"saved {path}")
    return path


def load(name: str):
    with open(os.path.join(RESULTS_DIR, f"{name}.json")) as f:
        return json.load(f)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __str__(self):
        return f"{time.time() - self.t0:.0f} s"
