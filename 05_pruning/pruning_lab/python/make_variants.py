"""
make_variants.py -- derive variants B and C from variant A's checkpoint.

    python3 python/make_variants.py          # both, ~2 minutes
    python3 python/make_variants.py --only C # just one

Students don't need to run this: checkpoints/variant_{A,B,C}.pt are provided.
It's here so you can see exactly how each variant was produced, and rerun
or modify it as homework.

  B  (unstructured) prune.global_unstructured over conv1/conv2/fc weights (L1, 70%, biases
     untouched) -> fine-tune 5 epochs with the masks attached ->
     prune.remove() on every pruned module.
  C  (structured) keep the half of each conv layer's filters with the
     largest L1 norm -- the same selection prune.ln_structured(n=1, dim=0,
     amount=0.5) makes -- and copy them, plus the matching input slices of
     each downstream layer, into a physically smaller ModelB2(8, 16) ->
     fine-tune 5 epochs. prune.ln_structured on its own would only zero
     those filters and leave every shape (and so the .pte, the arena and
     the latency) unchanged.
"""

import argparse
import copy

import torch
import torch.nn.utils.prune as prune

from data import loaders
from model import ModelB2, checkpoint_path, count_params, load_variant, prunable_modules
from train_baseline import evaluate, train_epochs

torch.set_num_threads(6)

FINETUNE_LR = 1e-3


def global_unstructured_70(model: ModelB2) -> ModelB2:
    """Attach 70% global L1 masks to conv1/conv2/fc weights (in place)."""
    prune.global_unstructured(
        [(m, "weight") for m in prunable_modules(model)],
        pruning_method=prune.L1Unstructured,
        amount=0.7,
    )
    return model


def make_B(a: ModelB2, train, val, test):
    torch.manual_seed(2)
    model = global_unstructured_70(copy.deepcopy(a))
    print(f"[B] after pruning, before fine-tune: test acc {evaluate(model, test):.4f}")
    train_epochs(model, train, val, epochs=5, lr=FINETUNE_LR, label="[B] ")

    # Bake the masks in before saving. Without this the state_dict holds
    # weight_orig AND weight_mask for every pruned layer, and an export
    # traces the hook: the .pte carries both tensors (~2x the weight bytes)
    # plus an extra multiply per layer on every inference. prune.remove() sets weight = weight_orig * mask,
    # deletes weight_orig/weight_mask, and removes the forward pre-hook.
    for m in prunable_modules(model):
        prune.remove(m, "weight")
    torch.save(model.state_dict(), checkpoint_path("B"))

    total, nonzero = count_params(model)
    print(f"[B] test acc {evaluate(model, test):.4f}  nonzero params {nonzero}/{total}")


def top_filters(conv: torch.nn.Conv2d, keep: int) -> torch.Tensor:
    """Indices of the `keep` filters with the largest L1 norm, in their
    original order -- the same filters prune.ln_structured(n=1) keeps."""
    norms = conv.weight.detach().abs().sum(dim=(1, 2, 3))
    return torch.topk(norms, keep).indices.sort().values


def make_C(a: ModelB2, train, val, test):
    keep1 = top_filters(a.conv1, 8)
    keep2 = top_filters(a.conv2, 16)

    c = ModelB2(8, 16)
    with torch.no_grad():
        c.conv1.weight.copy_(a.conv1.weight[keep1])
        c.conv1.bias.copy_(a.conv1.bias[keep1])
        # conv2 loses output filters (rows) AND the input channels that fed
        # from the removed conv1 filters (columns).
        c.conv2.weight.copy_(a.conv2.weight[keep2][:, keep1])
        c.conv2.bias.copy_(a.conv2.bias[keep2])
        # fc's inputs are conv2's channels (after GAP), so it loses columns.
        c.fc.weight.copy_(a.fc.weight[:, keep2])
        c.fc.bias.copy_(a.fc.bias)

    print(f"[C] after slimming, before fine-tune: test acc {evaluate(c, test):.4f}")
    torch.manual_seed(3)
    train_epochs(c, train, val, epochs=5, lr=FINETUNE_LR, label="[C] ")
    torch.save(c.state_dict(), checkpoint_path("C"))
    total, nonzero = count_params(c)
    print(f"[C] test acc {evaluate(c, test):.4f}  params {total}")
    print(f"[C] kept conv1 filters {keep1.tolist()}")
    print(f"[C] kept conv2 filters {keep2.tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["B", "C"])
    args = ap.parse_args()

    train, val, test = loaders()
    a = load_variant("A")
    print(f"[A] test acc {evaluate(a, test):.4f}")

    if args.only in (None, "B"):
        make_B(a, train, val, test)
    if args.only in (None, "C"):
        make_C(a, train, val, test)


if __name__ == "__main__":
    main()
