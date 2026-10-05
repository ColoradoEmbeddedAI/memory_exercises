"""
quantize.py -- the PyTorch -> int8 -> ExecuTorch Cortex-M pipeline, shared by
export_model.py, eval_quant.py and inspect_graph.py.

    export     torch.export.export(model, channels-last example)
    prepare    prepare_pt2e(graph, CortexMQuantizer())  -- inserts observers
    calibrate  run calibration images through the prepared graph
    convert    convert_pt2e(...)  -- observers become quantize/dequantize ops;
               the result still runs in PyTorch (that's the "host" model)
    lower      to_edge(...) + CortexMPassManager  -- (dequantize, op, quantize)
               patterns become cortex_m::quantized_* CMSIS-NN ops; anything
               the passes don't recognize stays a portable aten:: op
    serialize  to_executorch() -> .pte bytes

The fp32 path is the pruning lab's: torch.export -> to_edge -> to_executorch,
portable kernels only.

The lowering calls follow ExecuTorch release/1.2's
examples/arm/cortex_m_mv2_example.ipynb. The Cortex-M backend is marked WIP
upstream; these calls may change in later releases.
"""

import copy

import torch
import executorch.exir as exir
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from torch.export.exported_program import InputKind

# Importing these registers the cortex_m:: ops (and their Python reference
# implementations) with PyTorch.
import executorch.backends.cortex_m.ops.operators  # noqa: F401
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

CL = torch.channels_last


def example_input(precision: str) -> torch.Tensor:
    """CMSIS-NN kernels work in channels-last (NHWC). For a 1-channel image
    NCHW and NHWC are the same bytes in memory; only the dim order recorded
    in the program differs."""
    x = torch.zeros(1, 1, 32, 32)
    return x.to(memory_format=CL) if precision == "int8" else x


# ---------------------------------------------------------------- fp32 ----

def export_fp32(model: torch.nn.Module):
    """Returns an ExecutorchProgramManager (its .buffer is the .pte)."""
    ep = torch.export.export(model.eval(), (example_input("fp32"),))
    return exir.to_edge(ep).to_executorch()


# ---------------------------------------------------------------- int8 ----

def ptq(model: torch.nn.Module, calib: torch.Tensor, quantizer=None, batch: int = 50):
    """Post-training quantization. Returns the convert_pt2e GraphModule: a
    PyTorch model with explicit quantize/dequantize ops, runnable on the
    laptop."""
    gm = torch.export.export(model.eval(), (example_input("int8"),)).module()
    prepared = prepare_pt2e(gm, quantizer or CortexMQuantizer())
    with torch.no_grad():
        for i in range(0, len(calib), batch):
            prepared(calib[i:i + batch].contiguous(memory_format=CL))
    return convert_pt2e(prepared)


def lower(converted: torch.fx.GraphModule):
    """convert_pt2e graph -> edge program with cortex_m:: ops. Returns the
    EdgeProgramManager; call .to_executorch() on it to serialize."""
    ep = torch.export.export(converted, (example_input("int8"),))
    config = EdgeCompileConfig(
        preserve_ops=[torch.ops.aten.linear.default],
        _check_ir_validity=False,
        _core_aten_ops_exception_list=[torch.ops.aten.max_pool2d.default],
    )
    epm = to_edge(ep, compile_config=config)
    # As in the upstream example: transform the ExportedProgram directly so
    # the passes can add constants (NHWC weights, requantize multipliers).
    epm._edge_programs["forward"] = CortexMPassManager(epm.exported_program()).transform()
    return epm


def strip_unused_constants(epm):
    """Removes constants the lowered graph no longer reads.

    The lowering passes write new constants (NHWC copies of the conv
    weights, requantize multipliers and shifts) but leave the originals in
    the program: the NCHW int8 weights and the per-channel scale and
    zero-point tensors. ExecuTorch's own remove_unused_parameters_pass runs
    before lowering and only looks at parameters, not buffers, so it misses
    them. This is the same idea, applied after lowering, to both kinds."""
    ep = epm.exported_program()
    placeholders = {n.target: n for n in ep.graph_module.graph.nodes if n.op == "placeholder"}
    unused = [s for s in ep.graph_signature.input_specs
              if s.kind in (InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR)
              and len(placeholders[s.arg.name].users) == 0]
    sig = copy.deepcopy(ep.graph_signature)
    for s in unused:
        sig.input_specs.remove(s)
        if s.target in ep._state_dict:
            del ep._state_dict[s.target]
        elif s.target in ep._constants:
            del ep._constants[s.target]
        ep.graph_module.graph.erase_node(placeholders[s.arg.name])
    ep._graph_signature = sig
    ep.graph_module.recompile()
    return [s.target for s in unused]


def to_executorch(epm):
    return epm.to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=False))


def export_int8(model, calib, strip_unused: bool = False):
    """Returns (converted GraphModule, ExecutorchProgramManager)."""
    converted = ptq(model, calib)
    epm = lower(converted)
    if strip_unused:
        strip_unused_constants(epm)
    return converted, to_executorch(epm)


# ---------------------------------------------------- program inspection ----

def planned_bytes(prog) -> int:
    """Size of the memory-planned activation arena (index 0 is reserved)."""
    return sum(prog.executorch_program.execution_plan[0].non_const_buffer_sizes)


def instruction_names(prog) -> list:
    """One name per instruction in the program, in execution order: the
    operator for a kernel call, or the instruction kind otherwise. The
    firmware's per-op profile prints these."""
    plan = prog.executorch_program.execution_plan[0]
    names = []
    for ins in plan.chains[0].instructions:
        args = ins.instr_args
        if hasattr(args, "op_index"):
            op = plan.operators[args.op_index]
            names.append(op.name)
        else:
            names.append(type(args).__name__.replace("Arguments", "").lower())
    return names


def program_ops(prog) -> list:
    plan = prog.executorch_program.execution_plan[0]
    return sorted(f"{op.name}.{op.overload}" for op in plan.operators)


_DTYPE_BYTES = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8, 5: 2, 6: 4, 7: 8, 11: 1, 15: 2}
_DTYPE_NAMES = {0: "uint8", 1: "int8", 2: "int16", 3: "int32", 4: "int64",
                5: "fp16", 6: "fp32", 7: "fp64", 11: "bool", 15: "bf16"}


def constant_tensors(prog) -> list:
    """[(value index, shape, dtype name, bytes)] for every constant tensor
    stored in the program."""
    plan = prog.executorch_program.execution_plan[0]
    out = []
    for i, v in enumerate(plan.values):
        t = v.val
        if type(t).__name__ == "Tensor" and t.data_buffer_idx > 0:
            n = 1
            for d in t.sizes:
                n *= d
            st = int(t.scalar_type)
            out.append((i, tuple(t.sizes), _DTYPE_NAMES.get(st, str(st)), n * _DTYPE_BYTES.get(st, 4)))
    return out
