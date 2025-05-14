# ResNet Memory Profiler

This script profiles the GPU memory usage of a ResNet-152 model during a single forward and backward pass. It tracks and visualizes memory usage of:
- Feature maps (activations)
- Weights (model parameters)
- Gradients

The script produces a visualization similar to the example image, showing memory usage across operations with a clear forward/backward boundary.

## Requirements

- Python 3.6+
- PyTorch (with CUDA support)
- torchvision
- matplotlib
- numpy

## Usage

### Basic Usage

To run with default settings (batch size = 16):

```bash
python resnet_profile.py
```

This will generate results in the `resnet_results` directory:
- `memory_profile.png`: Memory usage visualization
- `memory_summary_bs16.txt`: Text summary of memory usage

### Advanced Usage

You can customize the batch size and output file:

```bash
python resnet_profile.py --batch_size 32 --output custom_results/memory_profile_bs32.png
```

### Command Line Arguments

- `--batch_size`: Batch size for profiling (default: 16)
- `--output`: Output image filename (default: 'resnet_results/memory_profile.png')
- `--no_title`: Disable the plot title (flag, no value needed)

## Output Files

The script generates the following outputs:

1. **Memory usage plot** (`memory_profile.png`) showing:
   - Blue: Weights (model parameters)
   - Orange: Gradients
   - Green: Feature maps (activations)
   - Vertical dashed line: Boundary between forward and backward passes

2. **Memory summary text file** with peak memory usage for each component:
   - Max Weights Memory (GB)
   - Max Gradients Memory (GB)
   - Max Feature Maps Memory (GB)
   - Total Peak Memory (GB)

## How It Works

The script works by:
1. Loading a ResNet-152 model
2. Registering hooks on each module to track memory usage during forward and backward passes
3. Running a single forward and backward pass with random input data
4. Calculating memory usage for weights, gradients, and feature maps at each step
5. Generating a visualization of the memory usage and a text summary

The memory tracking distinguishes between:
- Weights: Model parameters (constant during inference)
- Gradients: Parameter gradients accumulated during backpropagation
- Feature maps: Intermediate activations generated during forward pass

## Customizing

To profile a different model or architecture, modify the `profile_resnet152` function to use your desired model. 