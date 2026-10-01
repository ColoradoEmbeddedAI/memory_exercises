"""
memory_calc.py

Embedded AI course -- Pruning activity
Lab-local copy of ../../../02_membudget/memory_calc.py, the framework-free
self-check calculator from the memory-budgeting activity, with two models
added: Model B2 (variants A and B) and its structured-pruned version
(variant C).

Same simplified reuse model as before: for a purely sequential chain of
ops, peak activation memory is the max, over all ops, of (input tensor
bytes + output tensor bytes). Activation functions are folded into the
preceding op's output. Unlike the memory-budgeting version, B2 and C use
the conv -> pool -> ReLU order the pruning lab's model.py uses, so each
ReLU is folded into the POOL output here, not the conv output.

This is still the lecture's design-time approximation. The ExecuTorch
planner number (measure_host.py's "planned" column) is the ground truth,
and it comes out higher -- see the handout.

Usage:
    python3 memory_calc.py
"""

from dataclasses import dataclass
from typing import List, Tuple

BYTES_PER_ELEMENT = 4  # float32, matches what we're using pre-quantization


@dataclass
class Layer:
    name: str
    weight_params: int          # 0 for layers with no weights (pool, flatten, GAP, ...)
    out_shape: Tuple[int, ...]  # shape of this layer's OUTPUT tensor (activation fn folded in)


def elements(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def compute_budget(input_shape: Tuple[int, ...], layers: List[Layer], verbose: bool = True):
    shapes = [input_shape] + [l.out_shape for l in layers]
    sizes_bytes = [elements(s) * BYTES_PER_ELEMENT for s in shapes]
    total_weight_bytes = sum(l.weight_params for l in layers) * BYTES_PER_ELEMENT
    naive_total_activation = sum(sizes_bytes)
    pair_sums = [sizes_bytes[i] + sizes_bytes[i + 1] for i in range(len(layers))]
    peak_activation = max(pair_sums)
    peak_at = layers[pair_sums.index(peak_activation)].name

    if verbose:
        print(f"{'Layer':<10}{'Out shape':<18}{'Out bytes':>12}{'Weight bytes':>14}")
        print("-" * 54)
        print(f"{'input':<10}{str(input_shape):<18}{sizes_bytes[0]:>12}{'':>14}")
        for i, l in enumerate(layers):
            wbytes = l.weight_params * BYTES_PER_ELEMENT
            print(f"{l.name:<10}{str(l.out_shape):<18}{sizes_bytes[i + 1]:>12}{wbytes:>14}")
        print("-" * 54)
        print(f"Total weight memory        : {total_weight_bytes:>7} bytes ({total_weight_bytes/1024:.2f} KB)  -> Flash")
        print(f"Naive total activation mem : {naive_total_activation:>7} bytes ({naive_total_activation/1024:.2f} KB)")
        print(f"Peak activation memory     : {peak_activation:>7} bytes ({peak_activation/1024:.2f} KB)  -> SRAM  (peak at '{peak_at}')")
        print(f"Reuse savings              : {naive_total_activation - peak_activation:>7} bytes "
              f"({100 * (1 - peak_activation / naive_total_activation):.1f}% less than naive sum)")

    return total_weight_bytes, naive_total_activation, peak_activation


def model_b2_layers(c1: int, c2: int) -> List[Layer]:
    """Conv(1->c1) -> Pool -> ReLU -> Conv(c1->c2) -> Pool -> ReLU -> GAP -> FC(10).
    Each ReLU is folded into the pool before it."""
    return [
        Layer("conv1", weight_params=3 * 3 * 1 * c1 + c1, out_shape=(c1, 32, 32)),
        Layer("pool1", weight_params=0, out_shape=(c1, 16, 16)),
        Layer("conv2", weight_params=3 * 3 * c1 * c2 + c2, out_shape=(c2, 16, 16)),
        Layer("pool2", weight_params=0, out_shape=(c2, 8, 8)),
        Layer("gap", weight_params=0, out_shape=(c2, 1, 1)),
        Layer("fc", weight_params=c2 * 10 + 10, out_shape=(10,)),
    ]


def peak_activation_bytes(c1: int, c2: int) -> int:
    """Pair-model peak for ModelB2(c1, c2); used by measure_host.py."""
    return compute_budget((1, 32, 32), model_b2_layers(c1, c2), verbose=False)[2]


if __name__ == "__main__":
    print("=== Model B2: variants A and B (2x width) ===")
    print("Input: 1x32x32 -> Conv(16) -> Pool -> ReLU -> Conv(32) -> Pool -> ReLU -> GAP -> FC(10 classes)\n")
    compute_budget((1, 32, 32), model_b2_layers(16, 32))

    print()
    print("=== Model B2 slimmed: variant C (= Model B's shape) ===")
    print("Input: 1x32x32 -> Conv(8) -> Pool -> ReLU -> Conv(16) -> Pool -> ReLU -> GAP -> FC(10 classes)\n")
    compute_budget((1, 32, 32), model_b2_layers(8, 16))
