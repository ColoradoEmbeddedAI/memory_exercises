"""
measure_host.py -- the laptop half of the pruning activity's measurements.

    python3 python/measure_host.py              # table for all three variants (~1 minute)
    python3 python/measure_host.py --keys B     # print a checkpoint's state_dict keys and shapes

For each variant, from its checkpoint:
  params / nonzero  parameters the model computes with, and how many are nonzero
  MACs              multiply-accumulates per inference (analytic; a dense kernel
                    executes every one, zero or not)
  ckpt              torch.save checkpoint size on disk
  .pte              ExecuTorch program size (this is what goes in Flash)
  .pte.gz           the .pte compressed with gzip level 9 -- a rough measure of
                    how much information the file really holds
  planned           ExecuTorch's memory-planned activation arena (the same
                    number executorch_memory_check.py printed in the
                    memory-budgeting activity)
  pair est.         memory_calc.py's pair-model peak activation estimate
  test acc          accuracy on the 10k Fashion-MNIST test set (PyTorch, laptop)

Also writes pte/variant_*.pte as a side effect (same as export_variants.py --all).
"""

import argparse
import gzip
import os

import torch

from data import loaders
from export_variants import export_variant, planned_bytes
from memory_calc import peak_activation_bytes
from model import VARIANTS, checkpoint_path, count_params, macs
from train_baseline import evaluate

torch.set_num_threads(6)


def print_keys(variant: str):
    state = torch.load(checkpoint_path(variant), map_location="cpu")
    total = 0
    for k, v in state.items():
        print(f"  {k:<22} {str(tuple(v.shape)):<18} {v.dtype}  {v.numel() * v.element_size():>7} bytes")
        total += v.numel() * v.element_size()
    print(f"  {'total tensor bytes':<22} {'':<18} {'':13}{total:>7} bytes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", choices=VARIANTS, help="print one checkpoint's state_dict and exit")
    args = ap.parse_args()
    if args.keys:
        print_keys(args.keys)
        return

    _, _, test = loaders()
    header = (f"{'':<3}{'params':>8}{'nonzero':>9}{'MACs':>11}{'ckpt':>9}{'.pte':>9}"
              f"{'.pte.gz':>9}{'planned':>9}{'pair est.':>10}{'test acc':>10}")
    rows = []
    for v in VARIANTS:
        model, prog = export_variant(v)
        pte = prog.buffer
        total, nonzero = count_params(model)
        c1 = model.conv1.out_channels
        c2 = model.conv2.out_channels
        rows.append(
            f"{v:<3}{total:>8}{nonzero:>9}{macs(model):>11,}"
            f"{os.path.getsize(checkpoint_path(v)):>9}{len(pte):>9}"
            f"{len(gzip.compress(pte, compresslevel=9)):>9}{planned_bytes(prog):>9}"
            f"{peak_activation_bytes(c1, c2):>10}{evaluate(model, test):>10.4f}")
        print(f"[{v}] done")

    print()
    print("Sizes in bytes. MACs per inference.")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
