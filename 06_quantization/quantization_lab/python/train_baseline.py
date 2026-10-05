"""
train_baseline.py -- train variant A (dense Model B2) from scratch.

    python3 python/train_baseline.py            # 30 epochs, ~10 minutes on a laptop CPU
    python3 python/train_baseline.py --epochs 3 # quick smoke test

Writes checkpoints/variant_A.pt. Copied from the pruning lab, where
make_variants.py derives variant C from it. In this lab the scripts only use
evaluate(); the checkpoints are provided.
"""

import argparse

import torch
import torch.nn as nn

from data import loaders
from model import ModelB2, checkpoint_path

torch.set_num_threads(6)


def evaluate(model: nn.Module, loader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def train_epochs(model: nn.Module, train_loader, val_loader, epochs: int, lr: float,
                 label: str = ""):
    """Adam + cross-entropy, learning rate cosine-annealed to zero over the
    run. Used for the baseline and, with fewer epochs and a lower lr, for
    every fine-tune in make_variants.py. torch's prune module reapplies its
    masks on every forward pass, so fine-tuning a masked model needs
    nothing special here."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(y)
        val_acc = evaluate(model, val_loader)
        print(f"{label}epoch {epoch + 1:2d}/{epochs}  "
              f"train_loss={running / len(train_loader.dataset):.4f}  val_acc={val_acc:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    train, val, test = loaders()
    model = ModelB2(16, 32)
    train_epochs(model, train, val, args.epochs, args.lr, label="[A] ")
    print(f"[A] test accuracy: {evaluate(model, test):.4f}")

    torch.save(model.state_dict(), checkpoint_path("A"))
    print(f"[A] saved {checkpoint_path('A')}")


if __name__ == "__main__":
    main()
