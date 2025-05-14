#!/usr/bin/env python3
"""
ResNet GPU Memory & Dynamic-Profiling Script
===========================================

•   Traces a torchvision-ResNet with **torch.fx**  
•   Collects per-op CUDA run-time + memory statistics  
•   *Safely* removes all in-place additions / ReLUs if desired  
•   Outputs:
      – `memory_profile.png` — activation / gradient / weight timeline  
      – `dynamic_profiling.csv` — Table-A attributes from the μ-TWO paper
"""

import argparse
import csv
from types import MethodType

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.fx import Interpreter, symbolic_trace


# ──────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ──────────────────────────────────────────────────────────────────────────
def monkey_patch_residual_blocks():
    """Replace `out += identity` with `out = out + identity` in BasicBlock & Bottleneck."""
    import torchvision.models.resnet as res_mod

    def _basic_forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity   # safe
        out = self.relu(out)
        return out

    def _bottle_forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity   # safe
        out = self.relu(out)
        return out

    if hasattr(res_mod, "BasicBlock"):
        res_mod.BasicBlock.forward = _basic_forward
    if hasattr(res_mod, "Bottleneck"):
        res_mod.Bottleneck.forward = _bottle_forward


def disable_inplace_relu(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, torch.nn.ReLU):
            m.inplace = False


# ──────────────────────────────────────────────────────────────────────────
#  FX Interpreter with memory + time instrumentation
# ──────────────────────────────────────────────────────────────────────────
class MemoryProfiler(Interpreter):
    def __init__(self, module: torch.nn.Module, device: torch.device):
        gm = module if isinstance(module, torch.fx.GraphModule) else symbolic_trace(module)
        super().__init__(gm)
        self.device = device

        # ── Forward-pass bookkeeping ────────────────────────────
        self.forward_node_names: list[str] = ["baseline"]
        self.forward_mem: list[int] = []
        self.output_sizes: list[int] = [0]        # bytes of op outputs that need grad
        self.node_times: list[float] = []         # ms per op
        self.node_peak_mem: list[int] = []        # mem after op
        self.node_active_mem: list[int] = []      # currently equal to peak

        # ── Misc. bookkeeping ──────────────────────────────────
        self.param_sizes: dict[str, int] = {}     # bytes of params for each call_module
        self.backward_mem: list[tuple[str, int]] = []

    # ----------------------------------------------------------
    # Forward pass (per-node)
    # ----------------------------------------------------------
    def run(self, *args, **kwargs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        # baseline memory *after* weights & inputs reside on the GPU
        self.forward_mem.append(torch.cuda.memory_allocated(self.device))
        return super().run(*args, **kwargs)

    def run_node(self, n):
        if n.op in ("placeholder", "output"):
            return super().run_node(n)

        # ── start CUDA timing ──
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        result = super().run_node(n)

        # ── stop timing ──
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)        # milliseconds

        torch.cuda.synchronize(self.device)
        current_mem = torch.cuda.memory_allocated(self.device)

        # identify node
        name = n.target if n.op == "call_module" else f"{n.op}:{n.target}"
        self.forward_node_names.append(name)
        self.forward_mem.append(current_mem)
        self.node_times.append(elapsed)
        self.node_peak_mem.append(current_mem)
        self.node_active_mem.append(current_mem)  # same granularity

        # output tensor sizes
        tensors = []
        if isinstance(result, torch.Tensor):
            tensors = [result]
        elif isinstance(result, (list, tuple)):
            tensors = [t for t in result if isinstance(t, torch.Tensor)]
        out_bytes = sum(t.numel() * t.element_size() for t in tensors if t.requires_grad)
        self.output_sizes.append(out_bytes)

        for t in tensors:
            if t.requires_grad:
                t.register_hook(self._make_backward_hook(name))

        # param size
        if n.op == "call_module":
            mod = self.module.get_submodule(n.target)
            pbytes = sum(p.numel() * p.element_size() for p in mod.parameters(recurse=False))
            if pbytes:
                self.param_sizes[name] = pbytes

        return result

    # ----------------------------------------------------------
    # Backward helpers
    # ----------------------------------------------------------
    def _make_backward_hook(self, name):
        def hook(grad):
            torch.cuda.synchronize(self.device)
            mem = torch.cuda.memory_allocated(self.device)
            self.backward_mem.append((name, mem))
            return grad
        return hook

    def run_backward(self, output_tensor):
        loss = output_tensor.sum()
        loss.backward()
        torch.cuda.synchronize(self.device)
        self.backward_mem.append(("end", torch.cuda.memory_allocated(self.device)))


# ──────────────────────────────────────────────────────────────────────────
#  Profiling + plotting + CSV
# ──────────────────────────────────────────────────────────────────────────
def profile_resnet_memory(model, batch_size, image_size, device):
    model.to(device).train()
    inp = torch.randn(batch_size, 3, image_size, image_size, device=device)
    profiler = MemoryProfiler(model, device)
    out = profiler.run(inp)
    profiler.run_backward(out)

    # helper returns profiler so caller can access raw data
    return profiler


def plot_memory_profile(prof: MemoryProfiler, output_file: str):
    import numpy as np

    fwd_mem = np.array(prof.forward_mem, dtype=float)
    out_sizes = np.array(prof.output_sizes, dtype=float)          # len = len(fwd_mem)
    steps_fwd = len(out_sizes) - 1

    # activation timeline (fwd cum-sum then free in bwd)
    act_fwd = np.cumsum(out_sizes)
    act_bwd = [act_fwd[-1]]
    for i in range(steps_fwd):
        act_bwd.append(act_bwd[-1] - out_sizes[-(i + 1)])
    activation_MB = np.concatenate((act_fwd, act_bwd[1:])) / 2**20

    # gradient timeline (0 during fwd, grow during bwd)
    params_for_node = [0.0] * len(prof.forward_node_names)
    for n, sz in prof.param_sizes.items():
        idx = prof.forward_node_names.index(n)
        params_for_node[idx] = sz
    grad_bwd = [0.0]
    for i in range(steps_fwd):
        idx = steps_fwd - i
        grad_bwd.append(grad_bwd[-1] + params_for_node[idx])
    grad_MB = np.concatenate((np.zeros_like(act_fwd), np.array(grad_bwd[1:], dtype=float))) / 2**20

    # weight line
    weight_MB = sum(p.numel() * p.element_size() for p in prof.module.parameters()) / 2**20
    weight_line = np.full_like(activation_MB, weight_MB)

    # plot
    plt.figure(figsize=(8, 6))
    plt.plot(activation_MB, label="Activation")
    plt.plot(grad_MB, label="Gradient")
    plt.plot(weight_line, "--", label="Weights")
    plt.axvline(x=steps_fwd, color="gray", linestyle="--")
    plt.xlabel("Operation index (forward ▸ backward)")
    plt.ylabel("Memory (MB)")
    plt.title("GPU Memory Timeline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def write_dynamic_csv(prof: MemoryProfiler, csv_path: str):
    """Table-A CSV (μ-TWO) per forward node."""
    headers = [
        "rank", "gtype", "run_time_ms",
        "peak_mem_bytes", "active_mem_bytes",
        "to_offload", "to_delete", "to_prefetch", "to_recompute"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for rank, name in enumerate(prof.forward_node_names):
            if rank == 0:  # baseline row (placeholder)
                continue
            writer.writerow({
                "rank": rank - 1,                  # skip baseline for numbering
                "gtype": "forward",
                "run_time_ms": f"{prof.node_times[rank - 1]:.3f}",
                "peak_mem_bytes": prof.node_peak_mem[rank - 1],
                "active_mem_bytes": prof.node_active_mem[rank - 1],
                "to_offload": "",
                "to_delete": "",
                "to_prefetch": "",
                "to_recompute": ""
            })


def plot_peak_memory_vs_batch(
        model_name: str,
        batch_sizes: list[int],
        image_size: int,
        device: torch.device,
        patch_residual: bool,
        no_inplace_relu: bool,
        output_file: str,
    ):
    """Profile *peak* GPU memory for several batch sizes and create a bar plot.

    Args:
        model_name: Name of the torchvision ResNet variant.
        batch_sizes: Batch sizes to evaluate (e.g., [8, 16, 32]).
        image_size: Input resolution for the synthetic images.
        device: CUDA device for profiling.
        patch_residual: Whether to replace in-place residual adds.
        no_inplace_relu: Whether to make all ReLUs out-of-place.
        output_file: File name for the generated PNG figure.
    """
    # Ensure consistent visual style
    plt.style.use("seaborn-v0_8-whitegrid")

    peak_mb: list[float] = []
    for bs in batch_sizes:
        # Fresh model instance each run to avoid state carry-over
        model = getattr(torchvision.models, model_name)()
        if patch_residual:
            monkey_patch_residual_blocks()
        if no_inplace_relu:
            disable_inplace_relu(model)

        # Reset CUDA state for a clean measurement
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        _ = profile_resnet_memory(model, batch_size=bs, image_size=image_size, device=device)
        peak = torch.cuda.max_memory_allocated(device) / 2**20  # MB
        peak_mb.append(peak)

    # --- Plot -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    x_labels = [str(b) for b in batch_sizes]
    bars = ax.bar(x_labels, peak_mb, color="#69b3a2", edgecolor="black")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title(f"Peak Memory vs. Batch Size — {model_name}")

    # Annotate bars with exact values
    for rect, value in zip(bars, peak_mb):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, height + 1, f"{value:.0f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────
#  Main CLI
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FX-based GPU memory profiler for ResNet")
    parser.add_argument("--model", default="resnet152", help="torchvision ResNet variant")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-residual", action="store_true", help="Replace in-place additions")
    parser.add_argument("--no-inplace-relu", action="store_true", help="Make all ReLUs out-of-place")
    parser.add_argument("--plot", default="memory_profile.png")
    parser.add_argument("--csv", default="dynamic_profiling.csv")
    parser.add_argument("--peak-batch-sizes", default="", help="Comma-separated batch sizes for peak memory bar chart (e.g., 8,16,32)")
    parser.add_argument("--peak-plot", default="peak_vs_batch.png", help="Filename for the peak memory bar figure")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return
    device = torch.device("cuda")

    # Build model
    if not hasattr(torchvision.models, args.model):
        raise ValueError(f"Model {args.model} not in torchvision.models")
    model = getattr(torchvision.models, args.model)()

    if args.patch_residual:
        monkey_patch_residual_blocks()
    if args.no_inplace_relu:
        disable_inplace_relu(model)

    prof = profile_resnet_memory(
        model,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=device
    )

    # Outputs
    plot_memory_profile(prof, args.plot)
    write_dynamic_csv(prof, args.csv)

    peak_total_MB = torch.cuda.max_memory_allocated(device) / 2**20
    print(f"Peak GPU memory: {peak_total_MB:.1f} MB")
    print(f"Plot saved: {args.plot}")
    print(f"CSV  saved: {args.csv}")

    if args.peak_batch_sizes:
        sizes = [int(s) for s in args.peak_batch_sizes.split(',') if s.strip()]
    else:
        sizes = [8, 16, 32]

    # Always generate the peak-vs-batch plot using computed sizes
    plot_peak_memory_vs_batch(
        model_name=args.model,
        batch_sizes=sizes,
        image_size=args.image_size,
        device=device,
        patch_residual=args.patch_residual,
        no_inplace_relu=args.no_inplace_relu,
        output_file=args.peak_plot,
    )
    print(f"Peak-vs-batch plot saved: {args.peak_plot} (batch sizes: {sizes})")


if __name__ == "__main__":
    main()
