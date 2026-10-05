"""
fashion_b2.py -- Model B2 on Fashion-MNIST, from the pruning and
quantization activities, for Part 6 (KD in the compression pipeline) and
the lecture's dark-knowledge figure.

    ModelB2(c1, c2)   1x32x32 -> Conv(1->c1) -> MaxPool -> ReLU
                      -> Conv(c1->c2) -> MaxPool -> ReLU -> GAP -> FC(c2->10)
    load_A()          checkpoints/b2_variant_A.pt, the dense baseline (84.7%)
    load_C()          checkpoints/b2_variant_C.pt, the pruning lab's C:
                      A slimmed to ModelB2(8, 16), fine-tuned 5 epochs with CE
    slim(a)           A -> ModelB2(8, 16), keeping the half of each conv's
                      filters with the largest L1 norm: exactly the pruning
                      lab's make_C(), before its fine-tune
    fashion_splits()  {train, val, test}: 55k / 5k / 10k, the pruning lab's
                      split, padded to 32x32 and normalized

The checkpoints and data/fashion/ are copies of the pruning lab's.
"""

import gzip
import os

import numpy as np
import torch
import torch.nn as nn

from models import CHECKPOINT_DIR, LAB_DIR

FASHION_DIR = os.path.join(LAB_DIR, "data", "fashion")
FASHION_MIRROR = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
F_MEAN, F_STD = 0.2860, 0.3530
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


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
        return self.fc(torch.flatten(self.gap(x), 1))


def _load(name, c1, c2):
    m = ModelB2(c1, c2)
    m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, name), map_location="cpu"))
    return m.eval()


def load_A():
    return _load("b2_variant_A.pt", 16, 32)


def load_C():
    return _load("b2_variant_C.pt", 8, 16)


def _top_filters(conv, keep):
    norms = conv.weight.detach().abs().sum(dim=(1, 2, 3))
    return torch.topk(norms, keep).indices.sort().values


def slim(a: ModelB2) -> ModelB2:
    """The pruning lab's structured pruning of A, without the fine-tune."""
    keep1, keep2 = _top_filters(a.conv1, 8), _top_filters(a.conv2, 16)
    c = ModelB2(8, 16)
    with torch.no_grad():
        c.conv1.weight.copy_(a.conv1.weight[keep1])
        c.conv1.bias.copy_(a.conv1.bias[keep1])
        c.conv2.weight.copy_(a.conv2.weight[keep2][:, keep1])
        c.conv2.bias.copy_(a.conv2.bias[keep2])
        c.fc.weight.copy_(a.fc.weight[:, keep2])
        c.fc.bias.copy_(a.fc.bias)
    return c.eval()


def _read_idx(fname):
    path = os.path.join(FASHION_DIR, fname)
    if not os.path.exists(path):
        os.makedirs(FASHION_DIR, exist_ok=True)
        import urllib.request
        print(f"Downloading Fashion-MNIST {fname} ...")
        urllib.request.urlretrieve(FASHION_MIRROR + fname, path)
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def _prepare(images):
    x = torch.from_numpy(images.astype(np.float32) / 255.0)
    x = torch.nn.functional.pad(x, (2, 2, 2, 2), value=0.0)
    return ((x - F_MEAN) / F_STD).unsqueeze(1)


_cache = {}


def fashion_splits():
    if _cache:
        return _cache
    x_all = _prepare(_read_idx("train-images-idx3-ubyte.gz"))
    y_all = torch.from_numpy(_read_idx("train-labels-idx1-ubyte.gz").astype(np.int64))
    perm = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(0))
    tr, va = perm[:55000], perm[55000:]
    _cache.update({
        "train": (x_all[tr], y_all[tr]),
        "val": (x_all[va], y_all[va]),
        "test": (_prepare(_read_idx("t10k-images-idx3-ubyte.gz")),
                 torch.from_numpy(_read_idx("t10k-labels-idx1-ubyte.gz").astype(np.int64))),
    })
    return _cache
