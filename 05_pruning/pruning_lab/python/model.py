"""
model.py -- Model B2 and helpers shared by every script in this lab.

Model B2 is Model B from the memory-budgeting activity
(../../../02_membudget/) at 2x width -- twice the channels in each conv
layer (conv1 8 -> 16, conv2 16 -> 32) -- with one change: max-pool comes
BEFORE ReLU. The two commute (max and ReLU are both monotone), so the math
is identical, but pooling first means the full-resolution conv output is
only ever consumed once -- the 16x32x32 tensor never has to exist twice.

    1x32x32 -> Conv(1->c1, 3x3, pad 1) -> MaxPool2 -> ReLU
            -> Conv(c1->c2, 3x3, pad 1) -> MaxPool2 -> ReLU
            -> AdaptiveAvgPool(1) -> Flatten -> Linear(c2->10)

ModelB2(16, 32) is the dense baseline (variant A) and unstructured-pruned model (B).
ModelB2(8, 16) is the structured-pruned model (variant C) -- exactly Model B's shape.
"""

import os

import torch
import torch.nn as nn

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(LAB_DIR, "checkpoints")
PTE_DIR = os.path.join(LAB_DIR, "pte")
GENERATED_DIR = os.path.join(LAB_DIR, "generated")

VARIANTS = ["A", "B", "C"]

VARIANT_DESCRIPTIONS = {
    "A": "dense baseline",
    "B": "unstructured: 70% of weights zeroed, fine-tuned",
    "C": "structured: 50% of channels removed, fine-tuned",
}


class ModelB2(nn.Module):
    def __init__(self, c1: int = 16, c2: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c2, 10)

    def forward(self, x):
        x = torch.relu(self.pool(self.conv1(x)))
        x = torch.relu(self.pool(self.conv2(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def prunable_modules(model: ModelB2):
    return [model.conv1, model.conv2, model.fc]


def macs(model: ModelB2) -> int:
    """Multiply-accumulates per inference, computed analytically per layer.
    Counts every weight position whether or not its value is zero -- that's
    what a dense kernel executes."""
    c1 = model.conv1.out_channels
    c2 = model.conv2.out_channels
    conv1 = 32 * 32 * c1 * (3 * 3 * 1)
    conv2 = 16 * 16 * c2 * (3 * 3 * c1)
    fc = c2 * 10
    return conv1 + conv2 + fc


def count_params(model: nn.Module):
    """(total, nonzero) over every conv/linear weight and bias."""
    total = 0
    nonzero = 0
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            for t in (mod.weight, mod.bias):
                total += t.numel()
                nonzero += int(torch.count_nonzero(t))
    return total, nonzero


def checkpoint_path(variant: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"variant_{variant}.pt")


def pte_path(variant: str) -> str:
    return os.path.join(PTE_DIR, f"variant_{variant}.pte")


def load_variant(variant: str) -> ModelB2:
    """Rebuild a variant from its checkpoint, in eval mode. Every pruned
    variant had prune.remove() called before saving, so each checkpoint is
    a plain state_dict with one 'weight' per layer."""
    state = torch.load(checkpoint_path(variant), map_location="cpu")
    if variant == "C":
        model = ModelB2(8, 16)
    else:
        model = ModelB2(16, 32)
    model.load_state_dict(state)
    model.eval()
    return model
