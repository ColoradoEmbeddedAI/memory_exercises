"""
compare_preds.py -- compare the board's predictions with the laptop's.

    python3 python/compare_preds.py 3c-int8       # reads results/3c-int8.txt
    python3 python/compare_preds.py 3c-int8 --all # list every disagreement

Reads the board's UART capture (results/<build>.txt, written by
scripts/run_board.sh) and the host's predictions from the same export
(results/host_<build>.json, written by export_mlp.py), and prints:

  - accuracy on the embedded test images, board and host
  - host/board agreement, and which images disagree
  - for each disagreement the board printed logits for, the host's and
    the board's logits side by side

Copied from the quantization lab. For an int8 build, "host" is the convert_pt2e model: PyTorch evaluating the
quantize/dequantize graph in floating point. A few disagreements are
rounding: the CMSIS-NN kernels round intermediate results slightly
differently, and an image whose top two logits were nearly tied can flip.
Many disagreements mean the board is computing something else.
"""

import argparse
import json
import os
import re

from models import RESULTS_DIR

CLASS_NAMES = [str(d) for d in range(10)]


def parse_board(path):
    text = open(path).read()
    if "--- Done ---" not in text:
        raise SystemExit(f"{path}: incomplete capture (no '--- Done ---')")
    build = re.search(r"^Build (\S+):", text, re.M).group(1)
    preds = []
    for m in re.finditer(r"^preds\s+\d+: (\d+)$", text, re.M):
        preds += [int(c) for c in m.group(1)]
    logits = {}
    for m in re.finditer(r"^  image (\d+): .*\n    logits: (.*)$", text, re.M):
        logits[int(m.group(1))] = [float(v) for v in m.group(2).split()]
    return build, preds, logits


def fmt(row):
    return " ".join(f"{v:7.3f}" for v in row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build", help="e.g. 3c-int8")
    ap.add_argument("--all", action="store_true", help="list every disagreeing image")
    args = ap.parse_args()

    board_file = os.path.join(RESULTS_DIR, f"{args.build}.txt")
    host_file = os.path.join(RESULTS_DIR, f"host_{args.build}.json")
    build, board, board_logits = parse_board(board_file)
    host = json.load(open(host_file))
    if build != args.build:
        raise SystemExit(f"{board_file} is a capture of {build}, not {args.build}")

    n = len(board)
    labels = host["labels"][:n]
    hpred = host["host_preds"][:n]
    agree = [i for i in range(n) if board[i] == hpred[i]]
    differ = [i for i in range(n) if board[i] != hpred[i]]
    print(f"{build}: {host['description']}")
    print(f"  images             : {n}")
    print(f"  host accuracy      : {sum(h == l for h, l in zip(hpred, labels)) / n:.3f}")
    print(f"  board accuracy     : {sum(b == l for b, l in zip(board, labels)) / n:.3f}")
    print(f"  host/board agree   : {len(agree)}/{n} ({len(agree) / n:.1%})")
    if not differ:
        return
    shown = differ if args.all else differ[:10]
    print(f"  disagreeing images : {', '.join(map(str, shown))}"
          + ("" if args.all or len(differ) <= 10 else f", ... ({len(differ)} in all; --all lists them)"))
    for i in differ:
        if i not in board_logits:
            continue
        print(f"\n  image {i} (test[{host['test_indices'][i]}]), label {CLASS_NAMES[labels[i]]}: "
              f"host says {CLASS_NAMES[hpred[i]]}, board says {CLASS_NAMES[board[i]]}")
        print(f"    class       {' '.join(f'{c:>7d}' for c in range(10))}")
        print(f"    host  logits {fmt(host['host_logits'][i])}")
        print(f"    board logits {fmt(board_logits[i])}")


if __name__ == "__main__":
    main()
