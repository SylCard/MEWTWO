#!/usr/bin/env python3
"""
ResNet GPU Memory Profiler with Decreasing Memory Usage
======================================================

•   Simulates a ResNet model with decreasing memory usage during forward pass
•   Generates synthetic memory profiles for visualization purposes
•   Outputs:
      – `resnet_memory_decrease.png` — activation / gradient / weight timeline with decreases  
      – `resnet_dynamic_profiling.csv` — Table-A attributes (modified for memory decreases)
      – `resnet_peak_vs_batch.png` — Peak memory vs batch size
"""

import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt


class DummyResNetProfiler:
    """Generates synthetic ResNet memory profile data with decreasing memory usage."""
    
    def __init__(self, 
                 num_layers=50,  # Simulate a ResNet-50
                 batch_size=16, 
                 image_size=224):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.image_size = image_size
        
        # Generate fake layer structure
        self.forward_node_names = self._create_resnet_layer_names()
        
        # Generate synthetic memory data
        self._generate_memory_data()
        
    def _create_resnet_layer_names(self):
        """Create a structure that mimics ResNet layer names."""
        names = ["baseline", "conv1", "bn1", "relu", "maxpool"]
        
        # Simplified - create layer names similar to ResNet structure
        layer_sizes = [3, 4, 6, 3]  # Similar to ResNet-50
        for i, blocks in enumerate(layer_sizes):
            for j in range(blocks):
                # Each block has conv, bn, relu components
                block_prefix = f"layer{i+1}.{j}"
                names.extend([
                    f"{block_prefix}.conv1",
                    f"{block_prefix}.bn1",
                    f"{block_prefix}.relu1",
                    f"{block_prefix}.conv2",
                    f"{block_prefix}.bn2",
                    f"{block_prefix}.relu2",
                    f"{block_prefix}.conv3",
                    f"{block_prefix}.bn3",
                    # After each block, add shortcut and activation
                    f"{block_prefix}.add",
                    f"{block_prefix}.relu3"
                ])
                
        names.append("avgpool")
        names.append("fc")
        
        return names
        
    def _generate_memory_data(self):
        """Generate synthetic memory data with decreasing memory during forward pass."""
        n_steps = len(self.forward_node_names)
        
        # Forward memory: start at baseline, then show appropriate pattern
        self.forward_mem = [100 * 1024 * 1024]  # 100MB baseline for model parameters
        
        # Generate forward memory pattern with decreases
        for i in range(1, n_steps):
            previous_mem = self.forward_mem[-1]
            node_name = self.forward_node_names[i]
            
            # Memory decreases after certain operations
            if "add" in node_name:
                # After residual add, we can decrease memory (simulate freeing inputs)
                delta = -np.random.randint(20, 40) * 1024 * 1024  # 20-40MB decrease
            elif "relu3" in node_name:
                # After final activation in block, decrease memory (simulate garbage collection)
                delta = -np.random.randint(10, 30) * 1024 * 1024  # 10-30MB decrease
            elif "maxpool" in node_name:
                # After pooling, memory decreases (reduced spatial dimensions)
                delta = -np.random.randint(15, 35) * 1024 * 1024  # 15-35MB decrease
            else:
                # Most operations increase memory
                if "conv" in node_name:
                    # Convolutions create larger feature maps
                    delta = np.random.randint(20, 50) * 1024 * 1024  # 20-50MB increase
                elif "bn" in node_name or "relu" in node_name:
                    # BN and ReLU have smaller memory impact
                    delta = np.random.randint(1, 5) * 1024 * 1024  # 1-5MB increase
                else:
                    # Other operations have variable impact
                    delta = np.random.randint(5, 15) * 1024 * 1024  # 5-15MB increase
            
            # Ensure memory doesn't go below a minimum baseline
            new_mem = max(previous_mem + delta, 80 * 1024 * 1024)  # Minimum 80MB
            self.forward_mem.append(new_mem)
        
        # Output tensor sizes (theoretical activations that need gradients)
        self.output_sizes = [0]  # Baseline has no activation output
        
        for i in range(1, n_steps):
            name = self.forward_node_names[i]
            
            # Size depends on layer type and stage in network
            if "conv" in name:
                # Convolutional outputs are larger
                layer_num = int(name.split('.')[0].replace('layer', '') if 'layer' in name else 1)
                channels = 64 * (2 ** (layer_num - 1)) if layer_num > 0 else 64
                # Progressive decrease in spatial size
                spatial_size = self.image_size // (2 ** max(1, layer_num))
                size = self.batch_size * channels * spatial_size * spatial_size * 4  # NCHW * 4 bytes
            elif "fc" in name:
                # Fully connected output is smaller
                size = self.batch_size * 1000 * 4  # 1000 classes * 4 bytes
            elif "pool" in name:
                # Pooling reduces spatial dimensions
                layer_num = 4  # Assume near end of network
                channels = 64 * (2 ** (layer_num - 1))
                spatial_size = self.image_size // (2 ** max(1, layer_num))
                size = self.batch_size * channels * spatial_size * spatial_size * 4
            else:
                # Other layers typically maintain size
                if i > 1:
                    size = self.output_sizes[-1]
                else:
                    size = self.batch_size * 64 * (self.image_size // 2) * (self.image_size // 2) * 4
            
            # Add some noise
            size = int(size * (0.95 + np.random.rand() * 0.1))
            self.output_sizes.append(size)
            
        # Generate operation times (milliseconds)
        self.node_times = []
        for i in range(n_steps - 1):  # Excluding baseline
            name = self.forward_node_names[i+1]
            
            if "conv" in name:
                # Convolutions are slower
                time = np.random.rand() * 2 + 1
            elif "fc" in name:
                # FC can be slow too
                time = np.random.rand() * 1.5 + 1
            else:
                # Other operations are faster
                time = np.random.rand() * 0.3 + 0.1
                
            self.node_times.append(time)
            
        # Generate parameter sizes for backward pass
        self.param_sizes = {}
        for i, name in enumerate(self.forward_node_names):
            if i == 0:  # Skip baseline
                continue
                
            if "conv" in name:
                # Convolution parameters - size depends on stage
                layer_num = int(name.split('.')[0].replace('layer', '') if 'layer' in name else 1)
                in_channels = 64 * (2 ** (layer_num - 1)) if layer_num > 0 else 3
                out_channels = 64 * (2 ** (layer_num - 1)) if layer_num > 0 else 64
                kernel_size = 3
                size = in_channels * out_channels * kernel_size * kernel_size
            elif "fc" in name:
                # Fully connected parameters
                size = 512 * 1000  # typical final FC layer
            elif "bn" in name:
                # BatchNorm parameters (gamma, beta)
                layer_num = int(name.split('.')[0].replace('layer', '') if 'layer' in name else 1)
                channels = 64 * (2 ** (layer_num - 1)) if layer_num > 0 else 64
                size = channels * 2
            else:
                # Other layers might not have parameters
                size = 0
                
            if size > 0:
                self.param_sizes[name] = size * 4  # 4 bytes per parameter (float32)
        
        # Peak and active memory for CSV
        self.node_peak_mem = self.forward_mem[1:]  # Skip baseline
        self.node_active_mem = self.forward_mem[1:]  # Same for simplicity
        
    def plot_memory_profile(self, output_file):
        """Generate a memory profile plot showing activation, gradient, and weight memory."""
        # Convert to NumPy arrays for easier operations
        fwd_mem = np.array(self.forward_mem, dtype=float)
        out_sizes = np.array(self.output_sizes, dtype=float)
        steps_fwd = len(out_sizes) - 1
        
        # Activation memory timeline
        # Modified to show decreases during forward pass
        act_fwd = np.zeros_like(out_sizes)
        running_sum = 0
        for i in range(len(out_sizes)):
            node_name = self.forward_node_names[i] if i < len(self.forward_node_names) else ""
            running_sum += out_sizes[i]
            
            # Memory decreases after certain operations
            if i > 0 and ("add" in node_name or "relu3" in node_name or "maxpool" in node_name):
                release_factor = np.random.rand() * 0.4 + 0.2  # Release 20-60% of current memory
                running_sum -= running_sum * release_factor
                
            act_fwd[i] = running_sum
        
        # Backward activation memory release (standard pattern)
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
        for i in range(1, steps_fwd):
            if activation_MB[i] < activation_MB[i-1]:
                plt.annotate("Memory\ndecrease", 
                            xy=(i, activation_MB[i]), 
                            xytext=(i, activation_MB[i] + 50),
                            arrowprops=dict(arrowstyle="->", color="red"),
                            fontsize=8, color="red")
        
        plt.xlabel("Operation index (forward ▸ backward)")
        plt.ylabel("Memory (MB)")
        plt.title("ResNet GPU Memory Timeline with Memory Optimizations")
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
                    
                # Determine if this operation is a candidate for memory optimizations
                is_delete_candidate = "add" in name or "relu3" in name or "maxpool" in name
                is_offload_candidate = "conv" in name and not any(x in name for x in ["conv1", "1.0.conv"])
                is_recompute_candidate = "relu" in name and not "relu3" in name
                
                writer.writerow({
                    "rank": rank - 1,  # Skip baseline for numbering
                    "gtype": "forward",
                    "run_time_ms": f"{self.node_times[rank - 1]:.3f}",
                    "peak_mem_bytes": self.node_peak_mem[rank - 1],
                    "active_mem_bytes": self.node_active_mem[rank - 1],
                    "to_offload": "True" if is_offload_candidate else "",
                    "to_delete": "True" if is_delete_candidate else "",
                    "to_prefetch": "",
                    "to_recompute": "True" if is_recompute_candidate else ""
                })
                
    def plot_peak_memory_vs_batch(self, batch_sizes, output_file):
        """Generate a bar plot showing peak memory vs batch size."""
        # Create synthetic peak memory data
        base_memory = 150  # Base memory in MB
        memory_per_sample = 8  # Approximate MB per sample for ResNet
        
        # Create a slightly sublinear curve (memory efficiency at scale)
        peak_mb = [base_memory + (memory_per_sample * bs * (0.95 + 0.05 * np.log(bs/8)/np.log(4))) 
                  for bs in batch_sizes]
        
        # Add some noise to make it look realistic
        peak_mb = [mem * (1 + np.random.rand() * 0.05 - 0.025) for mem in peak_mb]
        
        # Plot
        plt.figure(figsize=(8, 5))
        plt.style.use("seaborn-v0_8-whitegrid")
        
        bars = plt.bar([str(bs) for bs in batch_sizes], peak_mb, 
                      color="#69b3a2", edgecolor="black")
        
        # Annotate bars with values
        for rect, value in zip(bars, peak_mb):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height + 5, f"{value:.0f}",
                   ha="center", va="bottom", fontsize=9)
            
        plt.xlabel("Batch Size")
        plt.ylabel("Peak GPU Memory (MB)")
        plt.title("ResNet Peak Memory vs. Batch Size")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="ResNet Memory Decrease Profiler")
    parser.add_argument("--layers", type=int, default=50, help="Number of ResNet layers")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--output-dir", default=".", help="Output directory for plots and CSVs")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize profiler
    profiler = DummyResNetProfiler(
        num_layers=args.layers,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    # Generate outputs
    memory_plot = os.path.join(args.output_dir, "resnet_memory_decrease.png")
    csv_path = os.path.join(args.output_dir, "resnet_dynamic_profiling.csv")
    peak_plot = os.path.join(args.output_dir, "resnet_peak_vs_batch.png")
    
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