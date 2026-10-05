"""
inspect_graph.py -- read what the int8 export actually produced.

    python3 python/inspect_graph.py --variant A                     # int8, relu-pool
    python3 python/inspect_graph.py --variant A --order pool-relu
    python3 python/inspect_graph.py --variant A --precision fp32    # memory plan for comparison

Four sections:

  1. Quantization parameters (the convert_pt2e graph): every activation
     quantizer's scale, zero point and the real range it covers, and how
     the weights were quantized (per-channel or per-tensor, zero points).
     Handout B4: are activations symmetric?
  2. Lowered ops, in execution order: which kernel library runs each op
     (CMSIS-NN cortex_m:: or portable aten::), and the dtype it outputs.
     Flags a portable op running on int8 data, and an fp32 op between a
     dequantize and a quantize (a fallback). Handout C2.
  3. Memory plan: every tensor in the planned arena, its dtype, size,
     offset, and the instructions it's live across. Handout Q2.
  4. Constants: what's stored in the .pte, and how much of it the lowered
     graph never reads. Handout Q1.
"""

import argparse
import copy

import torch
from torch.export.exported_program import InputKind

from data import calibration_set
from model import ORDERS, VARIANTS, load_variant
from quantize import (constant_tensors, export_fp32, lower, planned_bytes, ptq,
                      strip_unused_constants, to_executorch)

Q_PT = torch.ops.quantized_decomposed.quantize_per_tensor.default
DQ_PT = torch.ops.quantized_decomposed.dequantize_per_tensor.default
DQ_PC = torch.ops.quantized_decomposed.dequantize_per_channel.default
# Portable ops that only move or reinterpret bytes. They're correct on int8
# data because they never look at the values.
DATA_MOVEMENT = {"view_copy", "permute_copy", "clone", "_clone_dim_order",
                 "_to_dim_order_copy", "slice_copy", "cat", "squeeze_copy", "unsqueeze_copy"}
_DTYPE = {torch.float32: "fp32", torch.int8: "int8", torch.int32: "int32",
          torch.int64: "int64", torch.uint8: "uint8"}


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- 1 ----

def print_qparams(converted):
    section("1. Quantization parameters (convert_pt2e graph)")
    print("Activations (per-tensor):")
    print(f"  {'tensor':<16}{'scale':>10}{'zero pt':>9}   real range covered by -128..127")
    for n in converted.graph.nodes:
        if n.target == Q_PT:
            s, z = n.args[1], n.args[2]
            name = "input image" if n.args[0].op == "placeholder" else n.args[0].name
            print(f"  {name:<16}{s:>10.5f}{z:>9}   [{(-128 - z) * s:8.3f}, {(127 - z) * s:8.3f}]")
    print("Weights and biases:")
    for n in converted.graph.nodes:
        if n.target in (DQ_PC, DQ_PT) and n.args[0].op == "get_attr":
            user = next(iter(n.users)).target
            layer = "conv" if user == torch.ops.aten.conv2d.default else "linear"
            dtype = n.args[-1]
            role = "bias" if dtype == torch.int32 else "weight"
            if n.target == DQ_PC:
                scales = getattr(converted, n.args[1].target)
                zps = getattr(converted, n.args[2].target)
                print(f"  {layer:<7}{role:<7} per-channel, {len(scales)} channels, "
                      f"scale {scales.min().item():.2e}..{scales.max().item():.2e}, "
                      f"zero points {'all 0' if (zps == 0).all() else zps.tolist()}, {_DTYPE[dtype]}")
            else:
                print(f"  {layer:<7}{role:<7} per-tensor, scale {n.args[1]:.2e}, "
                      f"zero point {n.args[2]}, {_DTYPE[dtype]}")


# ---------------------------------------------------------------- 2 ----

def print_ops(epm):
    section("2. Lowered ops, in execution order")
    ep = epm.exported_program()
    rows = []
    for n in ep.graph.nodes:
        if n.op != "call_function" or "getitem" in str(n.target):
            continue
        # e.g. "cortex_m::quantized_conv2d" or "aten::relu"
        ns, name = n.target._op._name.split("::")
        val = n.meta.get("val")
        if isinstance(val, (tuple, list)):
            val = val[0]
        dt = _DTYPE.get(val.dtype, str(val.dtype)) if val is not None else "?"
        shape = tuple(val.shape) if val is not None else ()
        ins = [a.meta["val"].dtype for a in n.all_input_nodes
               if a.op == "call_function" or (a.op == "placeholder" and a.name == "x")]
        rows.append((ns, name, dt, shape, ins))

    print(f"  {'#':>2}  {'library':<10}{'op':<38}{'out':<6}{'shape':<18}note")
    n_cmsis = n_port = 0
    for k, (ns, name, dt, shape, ins) in enumerate(rows):
        lib = "CMSIS-NN" if ns == "cortex_m" else "portable"
        note = ""
        if ns == "cortex_m":
            n_cmsis += 1
        else:
            n_port += 1
            if name in DATA_MOVEMENT:
                note = "data movement only (no arithmetic): fine on int8"
            elif torch.int8 in ins:
                note = "<- portable kernel on int8 data: it knows nothing about zero points"
            elif dt == "fp32" and k > 0 and rows[k - 1][1].startswith("dequantize"):
                note = "<- fp32 FALLBACK between dequantize and quantize"
        print(f"  {k:>2}  {lib:<10}{ns + '::' + name:<38}{dt:<6}{str(shape):<18}{note}")
    print(f"  {n_cmsis} CMSIS-NN ops, {n_port} portable ops "
          "(view_copy becomes a free memory view at runtime and isn't an instruction)")


# ---------------------------------------------------------------- 3 ----

def print_memory_plan(prog):
    section("3. Memory plan (the planned arena)")
    plan = prog.executorch_program.execution_plan[0]
    names = []
    for ins in plan.chains[0].instructions:
        a = ins.instr_args
        names.append(plan.operators[a.op_index].name.split("::")[-1] if hasattr(a, "op_index")
                     else type(a).__name__.replace("Arguments", "").lower())

    def tensor(i):
        t = plan.values[i].val
        if type(t).__name__ != "Tensor" or t.data_buffer_idx > 0 or t.allocation_info is None:
            return None
        return t

    first, last, producer = {}, {}, {}
    for i in plan.inputs:
        first[i], producer[i] = -1, "program input"
    for k, ins in enumerate(plan.chains[0].instructions):
        a = ins.instr_args
        args = list(a.args) if hasattr(a, "args") else [getattr(a, "move_from", None), getattr(a, "move_to", None)]
        for i in args:
            if i is None or tensor(i) is None:
                continue
            if i not in first:
                first[i], producer[i] = k, f"out of #{k} {names[k]}"
            last[i] = k
    for i in plan.outputs:
        last[i] = len(names)

    from quantize import _DTYPE_BYTES, _DTYPE_NAMES
    rows = []
    for i in first:
        t = tensor(i)
        n = 1
        for d in t.sizes:
            n *= d
        nbytes = n * _DTYPE_BYTES.get(int(t.scalar_type), 4)
        off = t.allocation_info.memory_offset_low + (t.allocation_info.memory_offset_high << 32)
        rows.append((off, nbytes, _DTYPE_NAMES.get(int(t.scalar_type)), tuple(t.sizes),
                     producer[i], first[i], last.get(i, first[i])))
    print(f"  {'offset':>7}{'bytes':>8}  {'dtype':<6}{'shape':<17}{'live':<9}tensor")
    for off, nb, dt, shape, prod, f, l in sorted(rows):
        live = f"{max(f, 0)}..{min(l, len(names) - 1)}"
        print(f"  {off:>7}{nb:>8}  {dt:<6}{str(shape):<17}{live:<9}{prod}")
    print(f"  planned arena: {planned_bytes(prog)} bytes")


# ---------------------------------------------------------------- 4 ----

def print_constants(prog, unused):
    section("4. Constants stored in the .pte")
    consts = constant_tensors(prog)
    total = sum(c[3] for c in consts)
    by_dtype = {}
    for _, _, dt, nb in consts:
        by_dtype[dt] = by_dtype.get(dt, 0) + nb
    print(f"  .pte size {len(prog.buffer)} bytes: {total} in {len(consts)} constant tensors "
          f"({', '.join(f'{dt} {nb}' for dt, nb in sorted(by_dtype.items()))}), "
          f"{len(prog.buffer) - total} program structure (ops, tensor metadata, alignment)")
    if unused is not None:
        nbytes = sum(b for _, b in unused)
        print(f"  never read by the lowered graph: {len(unused)} tensors, {nbytes} bytes")
        for name, b in unused:
            print(f"    {name:<28}{b:>6} bytes")
        print("  (export_model.py --strip-unused removes them)")


def unused_constants(epm):
    ep = epm.exported_program()
    ph = {n.target: n for n in ep.graph_module.graph.nodes if n.op == "placeholder"}
    out = []
    for s in ep.graph_signature.input_specs:
        if s.kind in (InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR) \
                and len(ph[s.arg.name].users) == 0:
            t = ep.state_dict.get(s.target, ep.constants.get(s.target))
            out.append((s.target, t.numel() * t.element_size()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", choices=VARIANTS, default="A")
    ap.add_argument("--precision", choices=["fp32", "int8"], default="int8")
    ap.add_argument("--order", choices=ORDERS,
                    help="default: pool-relu for fp32, relu-pool for int8")
    ap.add_argument("--calib-n", type=int, default=200)
    args = ap.parse_args()
    order = args.order or ("pool-relu" if args.precision == "fp32" else "relu-pool")
    model = load_variant(args.variant, order)
    print(f"Variant {args.variant}, {args.precision}, {order}")

    if args.precision == "fp32":
        prog = export_fp32(model)
        print_memory_plan(prog)
        print_constants(prog, None)
        return

    converted = ptq(model, calibration_set(args.calib_n))
    print_qparams(converted)
    epm = lower(converted)
    print_ops(epm)
    unused = unused_constants(epm)
    prog = to_executorch(copy.deepcopy(epm))
    print_memory_plan(prog)
    print_constants(prog, unused)


if __name__ == "__main__":
    main()
