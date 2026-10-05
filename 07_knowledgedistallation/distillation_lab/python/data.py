"""
data.py -- MNIST, flattened to 784 values and normalized, with the fixed
splits every part of the activity uses.

Downloads the four raw IDX files into ../data/ on first use (about 11 MB).
No torchvision dependency (the ExecuTorch venv doesn't have it); the IDX
format is a few header bytes and then raw uint8.

Splits (all fixed by seeds, so every script and every student sees the same
images):
    train_full      55,000  the 60k training images minus val
    val              5,000  for choosing T, alpha, the Part 4 bias
    test            10,000  the official test set; report, never tune
    labeled_1k       1,000  100 per class from train_full
    unlabeled_rest  54,000  the rest of train_full; labels hidden (-1)
    train_no3      ~49,400  train_full with every digit 3 removed

Everything is returned as tensors: x float32 (N, 784), y int64 (N,). The
whole of MNIST is 220 MB as float32, which fits in memory, and slicing
tensors is much faster than a DataLoader for a model this small.
"""

import gzip
import os
import urllib.request

import numpy as np
import torch

from models import LAB_DIR

DATA_DIR = os.path.join(LAB_DIR, "data")
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

MEAN = 0.1307
STD = 0.3081
UNLABELED = -1          # the label value of an image whose label is hidden

SPLIT_SEED = 0
LABELED_SEED = 1
N_PER_CLASS = 100


def _download():
    os.makedirs(DATA_DIR, exist_ok=True)
    for fname in FILES.values():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(MIRROR + fname, path)


def _read_idx(fname: str) -> np.ndarray:
    with gzip.open(os.path.join(DATA_DIR, fname), "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def pixel_lut() -> torch.Tensor:
    """float32[256]: the normalized value of each uint8 pixel. Training and
    the firmware both convert pixels through this table, so the board's input
    is bit-for-bit what PyTorch sees."""
    return (torch.arange(256, dtype=torch.float32) / 255.0 - MEAN) / STD


def _prepare(images: np.ndarray) -> torch.Tensor:
    return pixel_lut()[torch.from_numpy(images.reshape(len(images), -1).copy()).long()]


_cache = {}


def splits() -> dict:
    """{name: (x, y)} for every split above. Cached per process."""
    if _cache:
        return _cache
    _download()
    x_all = _prepare(_read_idx(FILES["train_images"]))
    y_all = torch.from_numpy(_read_idx(FILES["train_labels"]).astype(np.int64))
    x_test = _prepare(_read_idx(FILES["test_images"]))
    y_test = torch.from_numpy(_read_idx(FILES["test_labels"]).astype(np.int64))

    perm = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(SPLIT_SEED))
    tr, va = perm[:55000], perm[55000:]
    x_tr, y_tr = x_all[tr], y_all[tr]

    g = torch.Generator().manual_seed(LABELED_SEED)
    lab = torch.cat([torch.nonzero(y_tr == c).flatten()[torch.randperm(int((y_tr == c).sum()), generator=g)[:N_PER_CLASS]]
                     for c in range(10)]).sort().values
    is_lab = torch.zeros(len(x_tr), dtype=torch.bool)
    is_lab[lab] = True

    _cache.update({
        "train_full": (x_tr, y_tr),
        "val": (x_all[va], y_all[va]),
        "test": (x_test, y_test),
        "labeled_1k": (x_tr[is_lab], y_tr[is_lab]),
        "unlabeled_rest": (x_tr[~is_lab], torch.full(((~is_lab).sum().item(),), UNLABELED)),
        "train_no3": (x_tr[y_tr != 3], y_tr[y_tr != 3]),
    })
    return _cache


# ---- the 200 test images embedded in the firmware (Part 5) ----

N_EVAL = 200
EVAL_SEED = 2026


def eval_set():
    """20 test images per class, fixed seed, shuffled. Returns (indices into
    the 10k test set, uint8 (200, 784) pixels, int64 labels, float inputs)."""
    _download()
    y = _read_idx(FILES["test_labels"]).astype(np.int64)
    rng = np.random.default_rng(EVAL_SEED)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), N_EVAL // 10, replace=False)
                          for c in range(10)])
    idx = rng.permutation(idx)
    u8 = _read_idx(FILES["test_images"])[idx].reshape(N_EVAL, 784)
    x = pixel_lut()[torch.from_numpy(u8).long()]
    return idx, u8, torch.from_numpy(y[idx]), x


if __name__ == "__main__":
    # python3 python/data.py -- download ahead of time and print the splits.
    for name, (x, y) in splits().items():
        labeled = int((y != UNLABELED).sum())
        print(f"{name:15s} {len(x):6d} images, {labeled:6d} labeled")
