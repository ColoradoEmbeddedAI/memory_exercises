"""
quant_sim.py -- a fake-quantization simulator for Model B2, independent of
any backend. Pure PyTorch: torch.fake_quantize_per_tensor_affine and
torch.fake_quantize_per_channel_affine round values onto a b-bit integer
grid and back to float, so you can try choices the real backend doesn't
offer (per-tensor weights, 4 bits, other range estimators) and see what
they cost in accuracy.

    python3 python/quant_sim.py                              # 8-bit defaults, variant A
    python3 python/quant_sim.py --bits 4 --weights per-tensor
    python3 python/quant_sim.py --bits 6 --range mse --act sym
    python3 python/quant_sim.py --compare granularity        # handout B2 table
    python3 python/quant_sim.py --levels                     # handout B2: codes used per filter
    python3 python/quant_sim.py --compare range              # handout B3 table
    python3 python/quant_sim.py --clip-curve conv2 --bits 4  # handout B3: error vs clip value
    python3 python/quant_sim.py --sensitivity                # handout C1: one layer at a time

What gets quantized (the same places the int8 backend quantizes, with the
fusable conv -> ReLU -> pool order):
  weights      conv1, conv2, fc: symmetric, per-tensor or per-channel,
               range = max |w| (weights are known exactly; no estimator needed)
  activations  the input image, each conv block's output (after ReLU), the
               GAP output, and the logits: per-tensor, symmetric or
               asymmetric, range from --range on the calibration images
  bias         left in floating point. On the board it's int32 at scale
               S_w * S_x, which is fine-grained enough not to matter.

Activation ranges are measured on the fp32 model (all quantizers off), then
fixed -- the same thing an observer does during calibration.
"""

import argparse

import torch
import torch.nn as nn

from data import calibration_set, load_tensors
from model import VARIANTS, load_variant, quant_layers
from train_baseline import evaluate

torch.set_num_threads(6)

ACT_POINTS = ["input", "conv1", "conv2", "gap", "fc"]
RANGES = ["minmax", "pct", "mse"]


# ------------------------------------------------------------ quantizers ----

def qrange(bits: int, symmetric: bool):
    """Integer range. Symmetric uses the restricted range [-(2^(b-1)-1), 2^(b-1)-1]
    so that zero is exactly representable and the grid is symmetric."""
    if symmetric:
        return -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def act_qparams(lo: float, hi: float, bits: int, symmetric: bool):
    """(scale, zero point, qmin, qmax) covering [lo, hi]."""
    qmin, qmax = qrange(bits, symmetric)
    if symmetric:
        c = max(abs(lo), abs(hi), 1e-8)
        return c / qmax, 0, qmin, qmax
    lo, hi = min(lo, 0.0), max(hi, 0.0)       # zero must be on the grid
    scale = max(hi - lo, 1e-8) / (qmax - qmin)
    zp = int(round(qmin - lo / scale))
    return scale, max(qmin, min(qmax, zp)), qmin, qmax


def fq_act(x, qp):
    scale, zp, qmin, qmax = qp
    return torch.fake_quantize_per_tensor_affine(x, scale, zp, qmin, qmax)


def fq_weight(w: torch.Tensor, bits: int, per_channel: bool) -> torch.Tensor:
    qmin, qmax = qrange(bits, symmetric=True)
    if per_channel:
        c = w.detach().abs().flatten(1).amax(1).clamp(min=1e-8)
        return torch.fake_quantize_per_channel_affine(
            w, c / qmax, torch.zeros_like(c, dtype=torch.int32), 0, qmin, qmax)
    c = max(w.detach().abs().max().item(), 1e-8)
    return torch.fake_quantize_per_tensor_affine(w, c / qmax, 0, qmin, qmax)


def levels_used(w: torch.Tensor, bits: int, per_channel: bool) -> list:
    """For each output channel: how many distinct integer codes its weights
    land on after quantization."""
    qmin, qmax = qrange(bits, symmetric=True)
    if per_channel:
        scale = w.abs().flatten(1).amax(1).clamp(min=1e-8) / qmax
    else:
        scale = torch.full((w.shape[0],), w.abs().max().item() / qmax)
    q = torch.round(w.flatten(1) / scale[:, None]).clamp(qmin, qmax)
    return [len(torch.unique(row)) for row in q]


# --------------------------------------------------- range estimation ----

def range_minmax(x: torch.Tensor):
    return x.min().item(), x.max().item()


def range_pct(x: torch.Tensor, pct: float = 99.99):
    flat = x.flatten()
    if flat.numel() > 2_000_000:   # torch.quantile has an input-size limit
        flat = flat[torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(0))[:2_000_000]]
    lo = torch.quantile(flat, 1 - pct / 100).item()
    hi = torch.quantile(flat, pct / 100).item()
    return lo, hi


def quant_error(x: torch.Tensor, lo: float, hi: float, bits: int, symmetric: bool):
    """(total, rounding, clipping) mean squared error from fake-quantizing x
    over [lo, hi]. Rounding error comes from values inside the range,
    clipping error from values outside it."""
    qp = act_qparams(lo, hi, bits, symmetric)
    err = (fq_act(x, qp) - x) ** 2
    scale, zp, qmin, qmax = qp
    rlo, rhi = (qmin - zp) * scale, (qmax - zp) * scale
    inside = (x >= rlo) & (x <= rhi)
    n = x.numel()
    return err.sum().item() / n, err[inside].sum().item() / n, err[~inside].sum().item() / n


def range_mse(x: torch.Tensor, bits: int, symmetric: bool, steps: int = 100):
    """Grid search: shrink the min/max range by a factor a in (0, 1] and keep
    the a with the lowest total quantization error."""
    lo, hi = range_minmax(x)
    best = None
    for k in range(1, steps + 1):
        a = k / steps
        e = quant_error(x, lo * a, hi * a, bits, symmetric)[0]
        if best is None or e < best[0]:
            best = (e, lo * a, hi * a)
    return best[1], best[2]


# ------------------------------------------------------------- the model ----

class SimModel(nn.Module):
    """Model B2 (conv -> ReLU -> pool order) with a fake-quantizer at every
    point the int8 backend quantizes. Each quantizer can be on or off."""

    def __init__(self, fp32: nn.Module):
        super().__init__()
        self.m = fp32
        self.wq = {}          # layer name -> quantized weight tensor (None = fp32)
        self.aq = {}          # act point -> qparams (None = fp32)
        self.record = None    # dict to collect activations into, during calibration

    def _act(self, name, x):
        if self.record is not None:
            self.record.setdefault(name, []).append(x.detach())
        qp = self.aq.get(name)
        return fq_act(x, qp) if qp is not None else x

    def _w(self, name, layer):
        w = self.wq.get(name)
        return w if w is not None else layer.weight

    def forward(self, x):
        m = self.m
        x = self._act("input", x)
        x = m.pool(torch.relu(nn.functional.conv2d(x, self._w("conv1", m.conv1), m.conv1.bias, padding=1)))
        x = self._act("conv1", x)
        x = m.pool(torch.relu(nn.functional.conv2d(x, self._w("conv2", m.conv2), m.conv2.bias, padding=1)))
        x = self._act("conv2", x)
        x = torch.flatten(m.gap(x), 1)
        x = self._act("gap", x)
        x = nn.functional.linear(x, self._w("fc", m.fc), m.fc.bias)
        return self._act("fc", x)


def collect_activations(sim: SimModel, calib: torch.Tensor) -> dict:
    saved_aq, sim.aq = sim.aq, {}
    saved_wq, sim.wq = sim.wq, {}
    sim.record = {}
    with torch.no_grad():
        for i in range(0, len(calib), 100):
            sim(calib[i:i + 100])
    acts = {k: torch.cat(v) for k, v in sim.record.items()}
    sim.record, sim.aq, sim.wq = None, saved_aq, saved_wq
    return acts


def configure(sim, acts, wbits, abits, per_channel, symmetric, rng,
              layers=("conv1", "conv2", "fc"), points=tuple(ACT_POINTS)):
    """Turns on weight quantizers for `layers` and activation quantizers for
    `points`; everything else stays fp32."""
    sim.wq = {name: fq_weight(layer.weight.detach(), wbits, per_channel)
              for name, layer in quant_layers(sim.m).items() if name in layers}
    sim.aq = {}
    for p in points:
        x = acts[p]
        if rng == "minmax":
            lo, hi = range_minmax(x)
        elif rng == "pct":
            lo, hi = range_pct(x)
        else:
            lo, hi = range_mse(x, abits, symmetric)
        sim.aq[p] = act_qparams(lo, hi, abits, symmetric)


# ------------------------------------------------------------- commands ----

def run_one(sim, acts, test, args, **kw):
    cfg = dict(wbits=args.wbits or args.bits, abits=args.abits or args.bits,
               per_channel=args.weights == "per-channel", symmetric=args.act == "sym",
               rng=args.range)
    cfg.update(kw)
    configure(sim, acts, **cfg)
    return evaluate(sim, test)


def cmd_compare(sim, acts, test, args, fp32_acc):
    """granularity: weight bit widths x per-tensor/per-channel, activations
    held at 8 bits so only the weights change. range: activation bit widths x
    range estimator, weights held at 8 bits per-channel."""
    if args.compare == "granularity":
        rows = [(f"{b}-bit weights, {g}", dict(wbits=b, abits=8, per_channel=(g == "per-channel")))
                for b in (8, 6, 5, 4) for g in ("per-tensor", "per-channel")]
        print("activations: 8-bit asym minmax")
    else:
        rows = [(f"{b}-bit activations, {r}", dict(wbits=8, abits=b, rng=r))
                for b in (6, 5, 4) for r in RANGES]
        print(f"weights: 8-bit per-channel; activations {args.act}")
    print(f"{'configuration':<32}{'test acc':>10}{'drop':>9}")
    print("-" * 51)
    print(f"{'fp32':<32}{fp32_acc:>10.4f}")
    for name, kw in rows:
        if args.compare == "range":
            kw["per_channel"] = True
        acc = run_one(sim, acts, test, args, **kw)
        print(f"{name:<32}{acc:>10.4f}{100 * (acc - fp32_acc):>+8.2f}")


def cmd_sensitivity(sim, acts, test, fp32_acc, args):
    """Quantize one layer's weights AND its output activation; leave every
    other layer fp32. The input image is quantized with conv1."""
    per_layer = {"conv1": ("input", "conv1"), "conv2": ("conv2",), "fc": ("gap", "fc")}
    print(f"weights {args.weights}, activations {args.act} {args.range}")
    print(f"{'bits':<6}{'layer':<8}{'test acc':>10}{'drop':>9}")
    print("-" * 33)
    for bits in (8, 4):
        for layer, points in per_layer.items():
            configure(sim, acts, bits, bits, args.weights == "per-channel", args.act == "sym",
                      args.range, layers=(layer,), points=points)
            acc = evaluate(sim, test)
            print(f"{bits:<6}{layer:<8}{acc:>10.4f}{100 * (acc - fp32_acc):>+8.2f}")


def clip_curve(x: torch.Tensor, bits: int, symmetric: bool, steps: int = 40):
    """[(clip value c, total, rounding, clipping)] for c from max/steps to max."""
    lo, hi = range_minmax(x)
    out = []
    for k in range(1, steps + 1):
        a = k / steps
        out.append((hi * a, *quant_error(x, lo * a, hi * a, bits, symmetric)))
    return out


def cmd_clip_curve(acts, args):
    x = acts[args.clip_curve]
    bits = args.abits or args.bits
    sym = args.act == "sym"
    curve = clip_curve(x, bits, sym)
    best = min(curve, key=lambda r: r[1])
    pct_hi = range_pct(x)[1]
    what = "input image" if args.clip_curve == "input" else f"{args.clip_curve} output"
    print(f"{what}, {bits}-bit {args.act}: {x.numel():,} values, "
          f"max {range_minmax(x)[1]:.3f}, 99.99th percentile {pct_hi:.3f}")
    print(f"{'clip c':>8}{'total MSE':>12}{'rounding':>12}{'clipping':>12}")
    for c, tot, rnd, clp in curve:
        marks = (["lowest"] if c == best[0] else []) + (["min/max"] if c == curve[-1][0] else [])
        mark = f"  <- {' = '.join(marks)}" if marks else ""
        print(f"{c:>8.3f}{tot:>12.3e}{rnd:>12.3e}{clp:>12.3e}{mark}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", choices=VARIANTS, default="A")
    ap.add_argument("--bits", type=int, default=8, help="weights and activations")
    ap.add_argument("--wbits", type=int, help="weights only (overrides --bits)")
    ap.add_argument("--abits", type=int, help="activations only (overrides --bits)")
    ap.add_argument("--weights", choices=["per-channel", "per-tensor"], default="per-channel")
    ap.add_argument("--act", choices=["asym", "sym"], default="asym")
    ap.add_argument("--range", choices=RANGES, default="minmax")
    ap.add_argument("--calib-n", type=int, default=200)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--compare", choices=["granularity", "range"])
    g.add_argument("--sensitivity", action="store_true")
    g.add_argument("--clip-curve", choices=ACT_POINTS, metavar="POINT",
                   help=f"error vs clip value at one of {ACT_POINTS}")
    g.add_argument("--levels", action="store_true",
                   help="integer codes each conv2 filter uses, per-tensor vs per-channel")
    args = ap.parse_args()

    sim = SimModel(load_variant(args.variant, "relu-pool"))
    acts = collect_activations(sim, calibration_set(args.calib_n))
    if args.clip_curve:
        cmd_clip_curve(acts, args)
        return
    if args.levels:
        w = sim.m.conv2.weight.detach()
        bits = args.wbits or args.bits
        pt, pc = levels_used(w, bits, False), levels_used(w, bits, True)
        mx = w.abs().flatten(1).amax(1)
        print(f"conv2, {bits}-bit symmetric: distinct codes used per filter (of {2 ** bits - 1})")
        print(f"{'filter':>6}{'max|w|':>9}{'per-tensor':>12}{'per-channel':>13}")
        for j in range(len(pt)):
            print(f"{j:>6}{mx[j].item():>9.3f}{pt[j]:>12}{pc[j]:>13}")
        return

    _, _, (x_test, y_test) = load_tensors()
    test = [(x_test[i:i + 1000], y_test[i:i + 1000]) for i in range(0, len(x_test), 1000)]
    fp32_acc = evaluate(sim.m, test)
    if args.compare:
        cmd_compare(sim, acts, test, args, fp32_acc)
    elif args.sensitivity:
        cmd_sensitivity(sim, acts, test, fp32_acc, args)
    else:
        acc = run_one(sim, acts, test, args)
        print(f"variant {args.variant}: fp32 {fp32_acc:.4f}, simulated "
              f"{args.wbits or args.bits}-bit weights ({args.weights}) / "
              f"{args.abits or args.bits}-bit activations ({args.act}, {args.range}): "
              f"{acc:.4f} ({100 * (acc - fp32_acc):+.2f} points)")


if __name__ == "__main__":
    main()
