"""
executorch_memory_check.py

Embedded AI course — Memory Budgeting for Neural Nets, Activity 2
Ground-truth validation: export the real model through ExecuTorch and
inspect what its memory planner actually allocates, to compare against
your by-hand calculation and the memory_calc.py self-check.

Requires: torch, executorch (install per your course's Week 1-3 setup;
this script targets the same environment you already used to get
examples running on the STM32 board).

NOTE ON API STABILITY: ExecuTorch's Python APIs evolve between
releases. The `generate_memory_trace` call below is the officially
documented inspection tool as of the ExecuTorch docs at the time this
was written (see "Memory Planning Inspection in ExecuTorch" in the
ExecuTorch documentation). If an import or call in this script fails
against your installed version, check that page for the current
signature rather than assuming the concept has changed -- the
underlying mechanism (a memory-planning pass that assigns tensors to
shared arenas by lifetime) has been stable even when exact APIs move.

Usage:
    python3 executorch_memory_check.py model_a
    python3 executorch_memory_check.py model_b
"""

import sys
import torch
import torch.nn as nn

from executorch.exir import to_edge
from torch.export import export


# ---- Model definitions, matching the architectures from the lecture ----

class ModelA(nn.Module):
    """IMU gesture MLP: 60 -> 48 -> 24 -> 6"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(60, 48)
        self.fc2 = nn.Linear(48, 24)
        self.fc3 = nn.Linear(24, 6)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ModelB(nn.Module):
    """Tiny CNN: 1x32x32 -> conv(8) -> pool -> conv(16) -> pool -> GAP -> fc(10)"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def export_and_inspect(model: nn.Module, example_input: torch.Tensor, name: str):
    model.eval()

    # 1. Export the graph
    exported_program = export(model, (example_input,))

    # 2. Lower to the Edge dialect
    edge_program = to_edge(exported_program)

    # 3. Lower to the ExecuTorch program (this is where memory planning happens)
    et_program = edge_program.to_executorch()

    # 4. Generate the memory planning trace (documented inspection tool)
    trace_path = f"{name}_memory_profile.json"
    try:
        from executorch.util.activation_memory_profiler import generate_memory_trace
        generate_memory_trace(
            executorch_program_manager=et_program,
            chrome_trace_filename=trace_path,
            enable_memory_offsets=True,
        )
        print(f"[{name}] Wrote {trace_path}")
        print(f"[{name}] Open chrome://tracing in a Chrome browser tab and load this file.")
        print(f"[{name}] The horizontal axis (labeled seconds) is actually bytes/MB of memory;")
        print(f"[{name}] the max extent you see is the planner's peak activation memory.")
    except ImportError as e:
        print(f"[{name}] Could not import generate_memory_trace ({e}).")
        print(f"[{name}] Check the 'Memory Planning Inspection' page in the ExecuTorch docs")
        print(f"[{name}] for the current API in your installed version.")

    # 5. OPTIONAL: try to print planned buffer sizes numerically, without
    #    the trace visualization. This reaches into lower-level structures
    #    that are more likely to change between ExecuTorch versions than
    #    the documented trace tool above -- treat this as a bonus, and
    #    fall back to the trace file if it doesn't work in your version.
    try:
        exec_program = et_program.executorch_program
        plan = exec_program.execution_plan[0]
        buf_sizes = list(plan.non_const_buffer_sizes)
        print(f"[{name}] Planned non-constant buffer sizes (bytes): {buf_sizes}")
        print(f"[{name}] (index 0 is typically reserved; look at index >= 1 for the")
        print(f"[{name}]  actual memory-planned activation arena sizes)")
        print(f"[{name}] Sum of planned buffers: {sum(buf_sizes)} bytes "
              f"({sum(buf_sizes)/1024:.2f} KB)")
    except Exception as e:
        print(f"[{name}] Numeric buffer size introspection not available in this "
              f"ExecuTorch version ({e}). Use the trace file above instead.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "model_a"

    if which == "model_a":
        export_and_inspect(ModelA(), torch.randn(1, 60), "model_a")
    elif which == "model_b":
        export_and_inspect(ModelB(), torch.randn(1, 1, 32, 32), "model_b")
    else:
        print("Usage: python3 executorch_memory_check.py [model_a|model_b]")
        sys.exit(1)
