"""
part5_fit.py -- Part 5: does it fit the STM32F407? Laptop only, instant.

    python3 python/part5_fit.py

For the teacher and the student, prints:
  - weight bytes in fp32 (4 B per parameter) and int8 (1 B per weight,
    4 B per bias: int32 biases, as in the quantization activity), against
    the board's 1 MiB of Flash
  - peak activation memory, the memory-budgeting way: for each layer, input
    plus output must be live at once; the peak is the largest such pair
    (fp32 = 4 B per value, int8 = 1 B)
Then, if results/<build>.txt board captures exist (Part 5, Step 3), the
measured .pte, planned arena, latency and board accuracy next to them.
Writes results/part5.json.
"""

import os
import re

from expt import save
from models import RESULTS_DIR, LeNet300100, StudentMLP, count_params

FLASH = 1024 * 1024
SRAM = 128 * 1024


def linear_shapes(model):
    return [(m.in_features, m.out_features) for m in model.modules() if m.__class__.__name__ == "Linear"]


def weight_bytes(model, precision):
    if precision == "fp32":
        return 4 * count_params(model)
    return sum(i * o + 4 * o for i, o in linear_shapes(model))


def peak_activations(model, bytes_per_value):
    return max((i + o) * bytes_per_value for i, o in linear_shapes(model))


def board(build):
    path = os.path.join(RESULTS_DIR, f"{build}.txt")
    if not os.path.exists(path):
        return None
    text = open(path).read()
    get = lambda pat: re.search(pat, text).group(1)
    return {"pte": int(get(r"\.pte size \(Flash\)\s+: (\d+)")),
            "arena": int(get(r"Planned arena \(SRAM\)\s+: (\d+)")),
            "ms": float(get(r"Median latency\s+: ([\d.]+) ms")),
            "correct": int(get(r"Board accuracy\s+: (\d+)/200"))}


def main():
    out = {}
    print(f"{'':10s} {'params':>8s} {'fp32 weights':>14s} {'int8 weights':>14s} "
          f"{'peak act fp32':>14s} {'peak act int8':>14s}")
    for name, m in (("teacher", LeNet300100()), ("student", StudentMLP())):
        r = {"params": count_params(m),
             "fp32_bytes": weight_bytes(m, "fp32"), "int8_bytes": weight_bytes(m, "int8"),
             "peak_fp32": peak_activations(m, 4), "peak_int8": peak_activations(m, 1)}
        out[name] = r
        print(f"{name:10s} {r['params']:8,d} {r['fp32_bytes']:14,d} {r['int8_bytes']:14,d} "
              f"{r['peak_fp32']:14,d} {r['peak_int8']:14,d}")
    print(f"\nFlash is {FLASH:,d} B, of which the runtime and kernels take about 225 KB and the "
          f"200 test images 157 KB.")
    for name in ("teacher", "student"):
        for p in ("fp32", "int8"):
            b = out[name][f"{p}_bytes"]
            print(f"  {name} {p}: {b:,d} B of weights = {100 * b / FLASH:.1f}% of Flash"
                  + ("  -> does not fit, before any code" if b > FLASH else ""))

    builds = ["teacher-int8", "3a-int8", "3c-int8", "3c-fp32"]
    measured = {b: board(b) for b in builds}
    if any(measured.values()):
        print("\nMeasured on the board (results/<build>.txt):")
        print(f"  {'build':14s} {'.pte':>9s} {'arena':>9s} {'latency':>10s} {'accuracy':>10s}")
        for b, r in measured.items():
            if r:
                print(f"  {b:14s} {r['pte']:9,d} {r['arena']:9,d} {r['ms']:8.2f} ms {r['correct']:6d}/200")
        out["board"] = {b: r for b, r in measured.items() if r}
    save("part5", out)


if __name__ == "__main__":
    main()
