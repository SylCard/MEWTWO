#!/usr/bin/env python3
"""
BERT GPU Memory & Dynamic-Profiling Script
==========================================

•   Traces a Hugging Face BERT model with **torch.fx**  
•   Collects per-op CUDA run-time + memory statistics  
•   Outputs:
      – `bert_memory_profile.png` — activation / gradient / weight timeline  
      – `bert_dynamic_profiling.csv` — Table-A attributes for μ-TWO paper
"""

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.fx import Interpreter, symbolic_trace
from transformers import BertModel, BertConfig


# ──────────────────────────────────────────────────────────────────────────
#  FX Interpreter with memory + time instrumentation
# ──────────────────────────────────────────────────────────────────────────
class MemoryProfiler(Interpreter):
    def __init__(self, module: torch.nn.Module, device: torch.device):
        # Try to trace with transformers' symbolic tracer
        try:
            gm = symbolic_trace(module)
        except Exception as e:
            print(f"Symbolic trace failed with: {e}")
            print("Using concrete tracing for BERT...")
            # BERT models have control flow that requires concrete values
            dummy_input = torch.zeros((1, 128), dtype=torch.long, device=device)
            gm = torch.fx.Tracer().trace(module, concrete_args={"input_ids": dummy_input})
            print("Concrete tracing succeeded")
            
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
        elif hasattr(result, "last_hidden_state"):  # BERT output
            tensors = [result.last_hidden_state]
            
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
        if isinstance(output_tensor, torch.Tensor):
            loss = output_tensor.sum()
        else:  # BERT output
            loss = output_tensor.last_hidden_state.sum()
            
        loss.backward()
        torch.cuda.synchronize(self.device)
        self.backward_mem.append(("end", torch.cuda.memory_allocated(self.device)))


# ──────────────────────────────────────────────────────────────────────────
#  Profiling + plotting + CSV
# ──────────────────────────────────────────────────────────────────────────
def profile_bert_memory(model, seq_length, batch_size, device):
    model.to(device).train()
    # Create dummy inputs for BERT
    input_ids = torch.randint(0, 30522, (batch_size, seq_length), device=device)
    attention_mask = torch.ones((batch_size, seq_length), device=device)
    
    profiler = MemoryProfiler(model, device)
    try:
        out = profiler.run(input_ids=input_ids, attention_mask=attention_mask)
        profiler.run_backward(out)
    except Exception as e:
        print(f"Error during profiling: {e}")
        # Fallback to direct model run with hooks
        print("Using fallback profiling method...")
        # Reset stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        # Simple memory tracking for fallback
        memory_points = []
        def track_memory(name):
            def hook(*args):
                torch.cuda.synchronize(device)
                mem = torch.cuda.memory_allocated(device)
                memory_points.append((name, mem))
                return None
            return hook
            
        # Add hooks to layers
        handles = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules
                handles.append(module.register_forward_hook(track_memory(f"fwd:{name}")))
                handles.append(module.register_backward_hook(track_memory(f"bwd:{name}")))
        
        # Run model
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        out.last_hidden_state.sum().backward()
        
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        # Create simplified profiler data
        profiler.forward_node_names = [name for name, _ in memory_points]
        profiler.forward_mem = [mem for _, mem in memory_points]
        profiler.output_sizes = [0] * len(memory_points)
        profiler.node_times = [0.0] * len(memory_points)  # No timing in fallback
        profiler.node_peak_mem = profiler.forward_mem
        profiler.node_active_mem = profiler.forward_mem
    
    # Helper returns profiler so caller can access raw data
    return profiler


def plot_memory_profile(prof: MemoryProfiler, output_file: str):
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
    plt.title("BERT GPU Memory Timeline")
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
                "run_time_ms": f"{prof.node_times[rank - 1]:.3f}" if rank <= len(prof.node_times) else "0.0",
                "peak_mem_bytes": prof.node_peak_mem[rank - 1] if rank <= len(prof.node_peak_mem) else 0,
                "active_mem_bytes": prof.node_active_mem[rank - 1] if rank <= len(prof.node_active_mem) else 0,
                "to_offload": "",
                "to_delete": "",
                "to_prefetch": "",
                "to_recompute": ""
            })


def plot_peak_memory_vs_batch(
        config: BertConfig,
        batch_sizes: list[int],
        seq_length: int,
        device: torch.device,
        output_file: str,
    ):
    """Profile *peak* GPU memory for several batch sizes and create a bar plot."""
    # Ensure consistent visual style
    plt.style.use("seaborn-v0_8-whitegrid")

    peak_mb: list[float] = []
    for bs in batch_sizes:
        # Fresh model instance each run to avoid state carry-over
        model = BertModel(config)
        
        # Reset CUDA state for a clean measurement
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # Run BERT with these inputs
        model.to(device).train()
        input_ids = torch.randint(0, 30522, (bs, seq_length), device=device)
        attention_mask = torch.ones((bs, seq_length), device=device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        out.last_hidden_state.sum().backward()
        
        peak = torch.cuda.max_memory_allocated(device) / 2**20  # MB
        peak_mb.append(peak)

    # --- Plot -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    x_labels = [str(b) for b in batch_sizes]
    bars = ax.bar(x_labels, peak_mb, color="#69b3a2", edgecolor="black")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title(f"BERT Peak Memory vs. Batch Size")

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
    parser = argparse.ArgumentParser(description="FX-based GPU memory profiler for BERT")
    parser.add_argument("--model", default="bert-base-uncased", help="BERT model size")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=128, help="Input sequence length")
    parser.add_argument("--plot", default="bert_memory_profile.png")
    parser.add_argument("--csv", default="bert_dynamic_profiling.csv")
    parser.add_argument("--peak-batch-sizes", default="", help="Comma-separated batch sizes for peak memory bar chart")
    parser.add_argument("--peak-plot", default="bert_peak_vs_batch.png", help="Filename for the peak memory bar figure")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return
    device = torch.device("cuda")
    
    # Check model name and create appropriate configuration
    if args.model == "bert-base-uncased":
        config = BertConfig.from_pretrained("bert-base-uncased")
        model = BertModel(config)
    elif args.model == "bert-large-uncased":
        config = BertConfig.from_pretrained("bert-large-uncased")
        model = BertModel(config)
    else:
        # Custom size
        if args.model == "bert-tiny":
            config = BertConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=512)
        elif args.model == "bert-mini":
            config = BertConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024)
        elif args.model == "bert-small":
            config = BertConfig(hidden_size=512, num_hidden_layers=4, num_attention_heads=8, intermediate_size=2048)
        elif args.model == "bert-medium":
            config = BertConfig(hidden_size=512, num_hidden_layers=8, num_attention_heads=8, intermediate_size=2048)
        else:
            # Default to base config
            config = BertConfig()
        model = BertModel(config)

    # Run profiling
    try:
        prof = profile_bert_memory(
            model,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            device=device
        )

        # Outputs
        plot_memory_profile(prof, args.plot)
        write_dynamic_csv(prof, args.csv)

        peak_total_MB = torch.cuda.max_memory_allocated(device) / 2**20
        print(f"Peak GPU memory: {peak_total_MB:.1f} MB")
        print(f"Plot saved: {args.plot}")
        print(f"CSV saved: {args.csv}")
    except Exception as e:
        print(f"Error in main profiling: {e}")
        
    # Optional peak memory vs batch size analysis
    if args.peak_batch_sizes:
        sizes = [int(s) for s in args.peak_batch_sizes.split(',') if s.strip()]
    else:
        sizes = [1, 2, 4, 8]  # Default batch sizes for BERT (larger models)

    # Produce peak vs batch size plot
    plot_peak_memory_vs_batch(
        config=config,
        batch_sizes=sizes,
        seq_length=args.seq_length,
        device=device,
        output_file=args.peak_plot,
    )
    print(f"Peak-vs-batch plot saved: {args.peak_plot} (batch sizes: {sizes})")


if __name__ == "__main__":
    main() 