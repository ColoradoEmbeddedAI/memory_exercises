"""
eval_quant.py -- host-side accuracy of the real int8 model (CortexMQuantizer,
convert_pt2e) on the full 10k test set, for different calibration choices.

    python3 python/eval_quant.py                       # Part A's default: 200 images
    python3 python/eval_quant.py --calib-n 10
    python3 python/eval_quant.py --calib-class 0       # calibrate on T-shirts only
    python3 python/eval_quant.py --calib-no-normalize  # forgot to normalize
    python3 python/eval_quant.py --sweep               # handout Part B1: the whole table

This is the same quantizer and calibration export_model.py uses for the
board, evaluated in PyTorch (the convert_pt2e model), so no board is needed.
The board agrees with this model on 96-99% of images (Part A).
"""

import argparse

import torch

from data import CLASS_NAMES, calibration_set, load_tensors
from model import ORDERS, VARIANTS, load_variant
from quantize import CL, ptq

torch.set_num_threads(6)


def accuracy(model, x, y, channels_last=False):
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), 1000):
            xb = x[i:i + 1000]
            if channels_last:
                xb = xb.contiguous(memory_format=CL)
            correct += (model(xb).argmax(1) == y[i:i + 1000]).sum().item()
    return correct / len(x)


def logit_range(converted) -> str:
    """The real-valued range [lo, hi] calibration chose for the logits (the
    last quantizer): the int8 codes -128..127 cover exactly this, and any
    logit above hi is clipped to hi."""
    last = None
    for n in converted.graph.nodes:
        if n.target == torch.ops.quantized_decomposed.quantize_per_tensor.default:
            last = n
    s, z = last.args[1], last.args[2]
    return f"[{(-128 - z) * s:6.2f}, {(127 - z) * s:6.2f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", choices=VARIANTS, default="A")
    ap.add_argument("--order", choices=ORDERS, default="relu-pool")
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--calib-class", type=int, choices=range(10), metavar="K")
    ap.add_argument("--calib-no-normalize", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="calibration sizes 1/10/100/1000 plus the two bad calibrations")
    args = ap.parse_args()

    _, _, (x_test, y_test) = load_tensors()
    model = load_variant(args.variant, args.order)
    fp32 = accuracy(model, x_test, y_test)

    if args.sweep:
        configs = [(f"{n} images", dict(n=n)) for n in (1, 10, 100, 1000)]
        biased = args.calib_class if args.calib_class is not None else 0
        configs += [(f"200, class {biased} ({CLASS_NAMES[biased]}) only", dict(n=200, cls=biased)),
                    ("200, not normalized", dict(n=200, normalize=False))]
    else:
        name = f"{args.calib_n} images"
        if args.calib_class is not None:
            name += f", class {args.calib_class} only"
        if args.calib_no_normalize:
            name += ", not normalized"
        configs = [(name, dict(n=args.calib_n, cls=args.calib_class,
                               normalize=not args.calib_no_normalize))]

    print(f"Variant {args.variant} ({args.order}), int8 CortexMQuantizer, 10k test images")
    print(f"{'calibration':<34}{'logit range':>18}{'test acc':>10}{'drop':>8}")
    print("-" * 70)
    print(f"{'(fp32, no quantization)':<34}{'':>18}{fp32:>10.4f}")
    for name, c in configs:
        calib = calibration_set(c["n"], c.get("cls"), c.get("normalize", True))
        converted = ptq(model, calib)
        acc = accuracy(converted, x_test, y_test, channels_last=True)
        print(f"{name:<34}{logit_range(converted):>18}{acc:>10.4f}{100 * (acc - fp32):>+7.2f}")


if __name__ == "__main__":
    main()
