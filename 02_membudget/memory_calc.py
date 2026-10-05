"""
memory_calc.py

Embedded AI course — Memory Budgeting for Neural Nets, Activity 2
Framework-free self-check tool: NO PyTorch/ExecuTorch dependency required.

This implements exactly the simplified reuse model taught in lecture:
for a purely sequential chain of ops, peak activation memory
is the max, over all ops, of (input tensor bytes + output tensor bytes).
Activation functions (ReLU etc.) are treated as folded into the
preceding op's output, matching how we did the by-hand calculation on
the board.

Use this to check your by-hand arithmetic BEFORE moving on to the real
ExecuTorch validation step (executorch_memory_check.py). This script
does NOT tell you what ExecuTorch will actually do -- its planner is
more general than this simple pairwise model and may do better. This
is a warm-up sanity check, not the ground truth.

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


def compute_budget(input_shape: Tuple[int, ...], layers: List[Layer]):
    shapes = [input_shape] + [l.out_shape for l in layers]
    sizes_bytes = [elements(s) * BYTES_PER_ELEMENT for s in shapes]

    print(f"{'Layer':<10}{'Out shape':<18}{'Out bytes':>12}{'Weight bytes':>14}")
    print("-" * 54)
    print(f"{'input':<10}{str(input_shape):<18}{sizes_bytes[0]:>12}{'':>14}")

    total_weight_bytes = 0
    for i, l in enumerate(layers):
        wbytes = l.weight_params * BYTES_PER_ELEMENT
        total_weight_bytes += wbytes
        print(f"{l.name:<10}{str(l.out_shape):<18}{sizes_bytes[i + 1]:>12}{wbytes:>14}")

    naive_total_activation = sum(sizes_bytes)
    pair_sums = [sizes_bytes[i] + sizes_bytes[i + 1] for i in range(len(layers))]
    peak_activation = max(pair_sums)
    peak_at = layers[pair_sums.index(peak_activation)].name

    print("-" * 54)
    print(f"Total weight memory        : {total_weight_bytes:>7} bytes ({total_weight_bytes/1024:.2f} KB)  -> Flash")
    print(f"Naive total activation mem : {naive_total_activation:>7} bytes ({naive_total_activation/1024:.2f} KB)")
    print(f"Peak activation memory     : {peak_activation:>7} bytes ({peak_activation/1024:.2f} KB)  -> SRAM  (peak at '{peak_at}')")
    print(f"Reuse savings              : {naive_total_activation - peak_activation:>7} bytes "
          f"({100 * (1 - peak_activation / naive_total_activation):.1f}% less than naive sum)")

    return total_weight_bytes, naive_total_activation, peak_activation


if __name__ == "__main__":
    print("=== Model A: IMU Gesture MLP ===")
    print("Input: 60 features -> FC(48) -> FC(24) -> FC(6 classes)\n")
    compute_budget(
        input_shape=(60,),
        layers=[
            Layer("fc1", weight_params=60 * 48 + 48, out_shape=(48,)),
            Layer("fc2", weight_params=48 * 24 + 24, out_shape=(24,)),
            Layer("fc3", weight_params=24 * 6 + 6, out_shape=(6,)),
        ],
    )

    print()
    print("=== Model B: Tiny CNN Image Classifier ===")
    print("Input: 1x32x32 -> Conv(8) -> Pool -> Conv(16) -> Pool -> GAP -> FC(10 classes)\n")
    compute_budget(
        input_shape=(1, 32, 32),
        layers=[
            Layer("conv1", weight_params=3 * 3 * 1 * 8 + 8, out_shape=(8, 32, 32)),
            Layer("pool1", weight_params=0, out_shape=(8, 16, 16)),
            Layer("conv2", weight_params=3 * 3 * 8 * 16 + 16, out_shape=(16, 16, 16)),
            Layer("pool2", weight_params=0, out_shape=(16, 8, 8)),
            Layer("gap", weight_params=0, out_shape=(16, 1, 1)),
            Layer("fc", weight_params=16 * 10 + 10, out_shape=(10,)),
        ],
    )
