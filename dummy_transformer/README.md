# Dummy Transformer and ResNet Memory Profiling

This folder contains scripts that generate synthetic memory profiles for transformer and ResNet models, specifically designed to show decreasing memory usage during the forward pass.

## Scripts

1. `transformer_profile.py` - Generates synthetic memory profiles for a transformer model
2. `resnet_memory_decrease.py` - Generates synthetic memory profiles for a ResNet model with memory decreases in forward pass

## Usage

### Transformer Profiler

```bash
# Basic usage (displays memory decreases by default)
python transformer_profile.py

# With custom parameters
python transformer_profile.py --layers 24 --hidden-size 1024 --batch-size 32 --seq-length 1024 --output-dir ./transformer_results
```

### ResNet Memory Decrease Profiler

```bash
# Basic usage
python resnet_memory_decrease.py

# With custom parameters
python resnet_memory_decrease.py --layers 101 --batch-size 32 --image-size 256 --output-dir ./resnet_results
```

## Generated Outputs

Each script generates:

1. A memory timeline plot showing activation, gradient, and weight memory usage
2. A CSV file with per-operation profiling data (similar to μ-TWO paper's Table-A)
3. A peak memory vs batch size bar chart

The plots highlight where memory decreases occur during the forward pass, which is the focus of these visualization tools.

## Notes

- These are synthetic visualizations that don't require CUDA-capable hardware
- All memory values and timings are artificially generated to mimic realistic patterns
- Memory decreases are inserted at specific operations (like after residual adds) 