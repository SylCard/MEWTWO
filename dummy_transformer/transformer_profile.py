#!/usr/bin/env python3
"""
Transformer GPU Memory & Dynamic-Profiling Script
================================================

•   Simulates a transformer model's memory profile
•   Generates synthetic profiles showing decreasing memory usage during forward pass  
•   Outputs:
      – `transformer_memory_profile.png` — activation / gradient / weight timeline  
      – `transformer_dynamic_profiling.csv` — Table-A attributes similar to μ-TWO paper
      – `transformer_peak_vs_batch.png` — Peak memory vs batch size
"""

import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt


class DummyTransformerProfiler:
    """Simulates a transformer model's memory profile with synthetic data."""
    
    def __init__(self, 
                 n_layers=12, 
                 hidden_size=768, 
                 batch_size=16, 
                 seq_length=512, 
                 show_memory_decrease=True):
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.show_memory_decrease = show_memory_decrease
        
        # Generate forward node names
        self.forward_node_names = ["baseline"]
        for i in range(n_layers):
            self.forward_node_names.extend([
                f"self_attn_{i}", 
                f"self_attn_norm_{i}", 
                f"ffn_{i}", 
                f"ffn_norm_{i}"
            ])
        self.forward_node_names.append("classifier")
        
        # Generate synthetic memory data
        self._generate_memory_data()
        
    def _generate_memory_data(self):
        """Generate synthetic memory data with optional decrease during forward pass."""
        n_steps = len(self.forward_node_names)
        
        # Forward memory: start at baseline, then gradually increase or show some decreases
        self.forward_mem = [100 * 1024 * 1024]  # 100MB baseline for model parameters
        
        for i in range(1, n_steps):
            previous_mem = self.forward_mem[-1]
            
            # For showing memory decreases in forward pass
            if self.show_memory_decrease and i > 2 and i % 4 == 0:
                # Decrease memory at the end of some transformer blocks
                delta = -np.random.randint(5, 15) * 1024 * 1024  # 5-15MB decrease
            else:
                # Otherwise general increase
                delta = np.random.randint(10, 30) * 1024 * 1024  # 10-30MB increase
            
            self.forward_mem.append(previous_mem + delta)
        
        # Output tensor sizes (bytes needed for activations)
        self.output_sizes = [0]  # Baseline
        for i in range(1, n_steps):
            # For attention layers, higher activation memory
            if "attn" in self.forward_node_names[i]:
                size = self.batch_size * self.seq_length * self.hidden_size * 4 * (np.random.rand() * 0.5 + 0.75)
            # For FFN layers
            elif "ffn" in self.forward_node_names[i]:
                size = self.batch_size * self.seq_length * self.hidden_size * 4 * (np.random.rand() * 0.3 + 0.85)
            # For norm layers, smaller activations
            elif "norm" in self.forward_node_names[i]:
                size = self.batch_size * self.seq_length * self.hidden_size * (np.random.rand() * 0.1 + 0.2)
            # For classifier, small output
            else:
                size = self.batch_size * self.hidden_size * (np.random.rand() * 0.2 + 0.8)
            
            self.output_sizes.append(int(size))
        
        # Generate op times (milliseconds)
        self.node_times = []
        for i in range(n_steps - 1):  # Excluding baseline
            if "attn" in self.forward_node_names[i+1] and "norm" not in self.forward_node_names[i+1]:
                # Attention layers are slower
                time = np.random.rand() * 5 + 3
            elif "ffn" in self.forward_node_names[i+1] and "norm" not in self.forward_node_names[i+1]:
                # FFN layers are slower too
                time = np.random.rand() * 4 + 2.5
            else:
                # Other layers are faster
                time = np.random.rand() * 0.5 + 0.2
            
            self.node_times.append(time)
        
        # Generate parameter sizes for backward pass
        self.param_sizes = {}
        for i, name in enumerate(self.forward_node_names):
            if i == 0:  # Skip baseline
                continue
                
            if "attn" in name and "norm" not in name:
                # Attention has 4 weight matrices (Q,K,V,O)
                size = 4 * self.hidden_size * self.hidden_size
            elif "ffn" in name and "norm" not in name:
                # FFN has 2 larger weight matrices
                size = self.hidden_size * (4 * self.hidden_size) + (4 * self.hidden_size) * self.hidden_size
            elif "norm" in name:
                # Norm has small number of parameters
                size = 2 * self.hidden_size  # gamma and beta
            elif "classifier" in name:
                # Classifier has output projection
                size = self.hidden_size * self.hidden_size
            else:
                size = 0
                
            if size > 0:
                self.param_sizes[name] = size * 4  # 4 bytes per parameter (float32)
        
        # Make up peak and active memory
        self.node_peak_mem = self.forward_mem[1:]  # Skip baseline
        self.node_active_mem = self.forward_mem[1:]  # Same for simplicity
        
    def plot_memory_profile(self, output_file):
        """Generate a memory profile plot showing activation, gradient, and weight memory."""
        # Convert to NumPy arrays for easier operations
        fwd_mem = np.array(self.forward_mem, dtype=float)
        out_sizes = np.array(self.output_sizes, dtype=float)
        steps_fwd = len(out_sizes) - 1
        
        # Activation memory timeline (forward accumulation, backward release)
        # For decreasing memory, we'll modify the cumulative sum
        if self.show_memory_decrease:
            # Modified cumulative sum with occasional drops
            act_fwd = np.zeros_like(out_sizes)
            running_sum = 0
            for i in range(len(out_sizes)):
                running_sum += out_sizes[i]
                # Random memory release at some layers
                if i > 3 and i % 4 == 0:
                    release_factor = np.random.rand() * 0.3 + 0.1  # Release 10-40% of current memory
                    running_sum -= running_sum * release_factor
                act_fwd[i] = running_sum
        else:
            # Standard cumulative sum
            act_fwd = np.cumsum(out_sizes)
            
        # Backward activation memory release
        act_bwd = [act_fwd[-1]]
        for i in range(steps_fwd):
            act_bwd.append(act_bwd[-1] - out_sizes[-(i + 1)])
        activation_MB = np.concatenate((act_fwd, act_bwd[1:])) / 2**20  # Convert to MB
        
        # Gradient memory (zero during forward, accumulate during backward)
        params_for_node = [0.0] * len(self.forward_node_names)
        for n, sz in self.param_sizes.items():
            idx = self.forward_node_names.index(n)
            params_for_node[idx] = sz
        
        grad_bwd = [0.0]
        for i in range(steps_fwd):
            idx = steps_fwd - i
            grad_bwd.append(grad_bwd[-1] + params_for_node[idx])
        grad_MB = np.concatenate((np.zeros_like(act_fwd), np.array(grad_bwd[1:], dtype=float))) / 2**20
        
        # Weight memory (constant)
        weight_MB = sum(self.param_sizes.values()) / 2**20
        weight_line = np.full_like(activation_MB, weight_MB)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(activation_MB, label="Activation")
        plt.plot(grad_MB, label="Gradient")
        plt.plot(weight_line, "--", label="Weights")
        plt.axvline(x=steps_fwd, color="gray", linestyle="--", label="Forward/Backward")
        
        # Add annotations for memory decreases in forward pass
        if self.show_memory_decrease:
            for i in range(1, steps_fwd):
                if activation_MB[i] < activation_MB[i-1]:
                    plt.annotate("Memory\ndecrease", 
                                xy=(i, activation_MB[i]), 
                                xytext=(i, activation_MB[i] + 50),
                                arrowprops=dict(arrowstyle="->", color="red"),
                                fontsize=8, color="red")
        
        plt.xlabel("Operation index (forward ▸ backward)")
        plt.ylabel("Memory (MB)")
        plt.title("Transformer GPU Memory Timeline")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
    def write_dynamic_csv(self, csv_path):
        """Generate a CSV with dynamic profiling data."""
        headers = [
            "rank", "gtype", "run_time_ms",
            "peak_mem_bytes", "active_mem_bytes",
            "to_offload", "to_delete", "to_prefetch", "to_recompute"
        ]
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            for rank, name in enumerate(self.forward_node_names):
                if rank == 0:  # Skip baseline
                    continue
                    
                writer.writerow({
                    "rank": rank - 1,  # Skip baseline for numbering
                    "gtype": "forward",
                    "run_time_ms": f"{self.node_times[rank - 1]:.3f}",
                    "peak_mem_bytes": self.node_peak_mem[rank - 1],
                    "active_mem_bytes": self.node_active_mem[rank - 1],
                    "to_offload": "True" if "attn" in name else "",
                    "to_delete": "True" if self.show_memory_decrease and rank % 4 == 0 else "",
                    "to_prefetch": "",
                    "to_recompute": "True" if "ffn" in name else ""
                })
                
    def plot_peak_memory_vs_batch(self, batch_sizes, output_file):
        """Generate a bar plot showing peak memory vs batch size."""
        # Create synthetic peak memory data with a roughly linear relationship to batch size
        base_memory = 200  # Base memory in MB
        memory_per_sample = 15  # Approximate MB per sample
        
        peak_mb = [base_memory + (memory_per_sample * bs) for bs in batch_sizes]
        
        # Add some noise to make it look more realistic
        peak_mb = [mem * (1 + np.random.rand() * 0.1 - 0.05) for mem in peak_mb]
        
        # Plot
        plt.figure(figsize=(8, 5))
        plt.style.use("seaborn-v0_8-whitegrid")
        
        bars = plt.bar([str(bs) for bs in batch_sizes], peak_mb, 
                      color="#5d8aa8", edgecolor="black")
        
        # Annotate bars with values
        for rect, value in zip(bars, peak_mb):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height + 5, f"{value:.0f}",
                   ha="center", va="bottom", fontsize=9)
            
        plt.xlabel("Batch Size")
        plt.ylabel("Peak GPU Memory (MB)")
        plt.title("Transformer Peak Memory vs. Batch Size")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Dummy Transformer Memory Profiler")
    parser.add_argument("--layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden dimension size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument("--show-memory-decrease", action="store_true", 
                       help="Show memory decreases during forward pass")
    parser.add_argument("--output-dir", default=".", help="Output directory for plots and CSVs")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize profiler
    profiler = DummyTransformerProfiler(
        n_layers=args.layers,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        show_memory_decrease=args.show_memory_decrease
    )
    
    # Generate outputs
    memory_plot = os.path.join(args.output_dir, "transformer_memory_profile.png")
    csv_path = os.path.join(args.output_dir, "transformer_dynamic_profiling.csv")
    peak_plot = os.path.join(args.output_dir, "transformer_peak_vs_batch.png")
    
    profiler.plot_memory_profile(memory_plot)
    profiler.write_dynamic_csv(csv_path)
    
    # Plot peak memory for different batch sizes
    batch_sizes = [8, 16, 32, 64]
    profiler.plot_peak_memory_vs_batch(batch_sizes, peak_plot)
    
    print(f"Memory profile saved: {memory_plot}")
    print(f"Dynamic profiling CSV saved: {csv_path}")
    print(f"Peak memory plot saved: {peak_plot}")


if __name__ == "__main__":
    main() 