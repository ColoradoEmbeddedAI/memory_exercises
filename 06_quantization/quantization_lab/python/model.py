"""
model.py -- Model B2 and helpers shared by every script in this lab.

Model B2 is the pruning activity's model (../../05_pruning/pruning_lab/),
unchanged. Variants A and C are copied from there:

    A  ModelB2(16, 32)  dense baseline
    C  ModelB2(8, 16)   structured-pruned: half of each conv layer's
                        filters physically removed, fine-tuned

    1x32x32 -> Conv(1->c1, 3x3, pad 1) -> MaxPool2 -> ReLU
            -> Conv(c1->c2, 3x3, pad 1) -> MaxPool2 -> ReLU
            -> AdaptiveAvgPool(1) -> Flatten -> Linear(c2->10)

The pruning lab put max-pool BEFORE ReLU: the two commute, so the function
is identical, and in fp32 pooling first saves 16 KiB of planned SRAM.
This lab adds the other order as an option (order="relu-pool"), with the
same weights. The two orders compute exactly the same function in fp32,
but they quantize very differently -- that's handout Part A, Step 3.
"""

import os

import torch
import torch.nn as nn

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(LAB_DIR, "checkpoints")
PTE_DIR = os.path.join(LAB_DIR, "pte")
GENERATED_DIR = os.path.join(LAB_DIR, "generated")
RESULTS_DIR = os.path.join(LAB_DIR, "results")

VARIANTS = ["A", "C"]
ORDERS = ["pool-relu", "relu-pool"]

VARIANT_DESCRIPTIONS = {
    "A": "dense baseline",
    "C": "structured: 50% of channels removed",
}


class ModelB2(nn.Module):
    def __init__(self, c1: int = 16, c2: int = 32, order: str = "pool-relu"):
        super().__init__()
        assert order in ORDERS
        self.order = order
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c2, 10)

    def _block(self, x, conv):
        if self.order == "pool-relu":
            return torch.relu(self.pool(conv(x)))
        return self.pool(torch.relu(conv(x)))

    def forward(self, x):
        x = self._block(x, self.conv1)
        x = self._block(x, self.conv2)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def quant_layers(model: ModelB2):
    """The three layers with weights, in order."""
    return {"conv1": model.conv1, "conv2": model.conv2, "fc": model.fc}


def macs(model: ModelB2) -> int:
    """Multiply-accumulates per inference, computed analytically per layer."""
    c1 = model.conv1.out_channels
    c2 = model.conv2.out_channels
    conv1 = 32 * 32 * c1 * (3 * 3 * 1)
    conv2 = 16 * 16 * c2 * (3 * 3 * c1)
    fc = c2 * 10
    return conv1 + conv2 + fc


def count_params(model: nn.Module) -> int:
    return sum(t.numel() for mod in model.modules()
               if isinstance(mod, (nn.Conv2d, nn.Linear))
               for t in (mod.weight, mod.bias))


def checkpoint_path(variant: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"variant_{variant}.pt")


def build_name(variant: str, precision: str) -> str:
    """'A-int8', 'C-fp32', ... -- the names used for pte/, results/ and the UART report."""
    return f"{variant}-{precision}"


def load_variant(variant: str, order: str = "pool-relu") -> ModelB2:
    """Rebuild a variant from its checkpoint, in eval mode. The order only
    changes forward(); the state_dict is the same either way."""
    state = torch.load(checkpoint_path(variant), map_location="cpu")
    if variant == "C":
        model = ModelB2(8, 16, order)
    else:
        model = ModelB2(16, 32, order)
    model.load_state_dict(state)
    model.eval()
    return model
