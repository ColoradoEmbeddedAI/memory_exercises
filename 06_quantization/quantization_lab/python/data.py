"""
data.py -- Fashion-MNIST, zero-padded to 32x32 and normalized.

Downloads the four raw IDX files from the dataset's official mirror into
../data/ on first use (about 30 MB). No torchvision dependency: the
ExecuTorch venv doesn't have it, and the IDX format is trivial to parse.

Split: the 60k training images are split 55k train / 5k validation with a
fixed seed; accuracy is always reported on the separate 10k test set.

Copied from ../../05_pruning/pruning_lab/python/data.py, with three
additions for the quantization lab:
  eval_set()        the 200 fixed test images the board classifies
  pixel_lut()       uint8 pixel -> normalized float, exactly as training does
  calibration_set() the images PTQ calibration runs, with the handout's
                    deliberate mistakes (one class only, no normalization)
"""

import gzip
import os
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import LAB_DIR

DATA_DIR = os.path.join(LAB_DIR, "data")
MIRROR = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# Fashion-MNIST training-set pixel statistics (on the 0..1 scale, computed
# over the original 28x28 images).
MEAN = 0.2860
STD = 0.3530

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

SPLIT_SEED = 0


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


def _prepare(images: np.ndarray) -> torch.Tensor:
    """uint8 (N,28,28) -> float32 (N,1,32,32): zero-pad 2 pixels on each
    side (background stays black, as in the original images), then
    normalize. The padding happens before normalization, so padded pixels
    end up at (0 - MEAN) / STD, the same value as any other black pixel."""
    x = torch.from_numpy(images.astype(np.float32) / 255.0)
    x = torch.nn.functional.pad(x, (2, 2, 2, 2), value=0.0)
    x = _normalize(x)
    return x.unsqueeze(1)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN) / STD


def load_tensors():
    """Returns ((x_train, y_train), (x_val, y_val), (x_test, y_test))."""
    _download()
    x_all = _prepare(_read_idx(FILES["train_images"]))
    y_all = torch.from_numpy(_read_idx(FILES["train_labels"]).astype(np.int64))
    x_test = _prepare(_read_idx(FILES["test_images"]))
    y_test = torch.from_numpy(_read_idx(FILES["test_labels"]).astype(np.int64))

    perm = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(SPLIT_SEED))
    train_idx, val_idx = perm[:55000], perm[55000:]
    return ((x_all[train_idx], y_all[train_idx]),
            (x_all[val_idx], y_all[val_idx]),
            (x_test, y_test))


def loaders(batch_size: int = 128):
    (xtr, ytr), (xva, yva), (xte, yte) = load_tensors()
    train = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(1))
    val = DataLoader(TensorDataset(xva, yva), batch_size=1000)
    test = DataLoader(TensorDataset(xte, yte), batch_size=1000)
    return train, val, test


N_EVAL = 200
EVAL_SEED = 2026


def _padded_uint8(fname: str) -> np.ndarray:
    return np.pad(_read_idx(fname), ((0, 0), (2, 2), (2, 2)))


def pixel_lut() -> torch.Tensor:
    """float32[256]: the normalized value of each uint8 pixel, computed with
    the same float32 operations as _prepare(). The firmware stores its test
    images as uint8 and converts them with this table, so the board's input
    tensor is bit-for-bit the one PyTorch sees."""
    return _normalize(torch.from_numpy(np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0))


def eval_set():
    """The 200 test images embedded in the firmware: 20 per class, chosen
    with a fixed seed, in shuffled order. Returns (indices into the 10k test
    set, uint8 (200,32,32) padded images, int64 labels, float (200,1,32,32)
    normalized inputs)."""
    _download()
    y = _read_idx(FILES["test_labels"]).astype(np.int64)
    rng = np.random.default_rng(EVAL_SEED)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), N_EVAL // 10, replace=False)
                          for c in range(10)])
    idx = rng.permutation(idx)
    u8 = _padded_uint8(FILES["test_images"])[idx]
    x = pixel_lut()[torch.from_numpy(u8).long()].unsqueeze(1)
    return idx, u8, torch.from_numpy(y[idx]), x


def calibration_set(n: int, cls: int = None, normalize: bool = True) -> torch.Tensor:
    """The first n training-split images (the split train_baseline.py trained
    on), optionally only those of class `cls`. normalize=False skips the
    (x - MEAN) / STD step -- a deliberate preprocessing bug for handout Part B."""
    (x_train, y_train), _, _ = load_tensors()
    if cls is not None:
        x_train = x_train[y_train == cls]
    x = x_train[:n]
    if not normalize:
        x = x * STD + MEAN          # undo normalization: back to 0..1 pixels
    return x


if __name__ == "__main__":
    # python3 python/data.py -- download ahead of time (e.g. before class).
    (xtr, _), (xva, _), (xte, _) = load_tensors()
    print(f"Fashion-MNIST ready in {DATA_DIR}: {len(xtr)} train, {len(xva)} val, {len(xte)} test, "
          f"each {tuple(xtr.shape[1:])}")
