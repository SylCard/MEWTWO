#!/usr/bin/env python3
"""
BERT Activation Checkpointing Analyzer
======================================

Reads dynamic profiling CSV and outputs a checkpoint plan, identifying
operations with high memory-to-compute ratio for memory optimization.
"""

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def read_dynamic_profile(csv_path):
    """Read dynamic profiling data from CSV file."""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def calculate_recompute_ratio(profile_data):
    """
    Calculate the recompute ratio for each operation:
    ratio = memory_size_bytes / run_time_ms
    Higher ratio = better candidate for checkpointing
    """
    for row in profile_data:
        # Parse memory size from active memory
        memory_size = int(row.get('active_mem_bytes', 0)) if row.get('active_mem_bytes', '0') else 0
        
        # Get run time, default to a small value to avoid division by zero
        run_time = float(row.get('run_time_ms', 0.1)) if row.get('run_time_ms', '0.1') else 0.1
        if run_time < 0.001:  # Minimum to avoid extreme ratios
            run_time = 0.001
            
        # Calculate ratio
        ratio = memory_size / run_time if memory_size > 0 else 0
        
        # Add to row data
        row['memory_size_bytes'] = memory_size
        row['recompute_ratio'] = ratio
    
    return profile_data


def calculate_memory_diff(profile_data):
    """Calculate memory difference between consecutive operations."""
    memory_diffs = []
    last_mem = 0
    
    for row in profile_data:
        curr_mem = int(row.get('active_mem_bytes', 0)) if row.get('active_mem_bytes', '0') else 0
        diff = curr_mem - last_mem
        memory_diffs.append(diff)
        last_mem = curr_mem
        
    return memory_diffs


def find_recompute_candidates(profile_data, target_reduction_percent=30):
    """
    Find operations to recompute based on memory-to-compute ratio.
    
    The goal is to identify operations to checkpoint until we hit the target memory reduction.
    Prioritize operations with high memory/compute ratio (most memory saved per compute cost).
    """
    # Filter only operations with positive memory impact
    forward_ops = [row for row in profile_data if row.get('gtype', '') == 'forward']
    
    # Calculate total memory used by all operations
    total_memory = sum(int(row.get('memory_size_bytes', 0)) for row in forward_ops)
    
    # Target memory reduction
    target_reduction = (target_reduction_percent / 100.0) * total_memory
    
    # Sort by recompute ratio (highest first)
    candidates = sorted(
        forward_ops, 
        key=lambda x: float(x.get('recompute_ratio', 0)),
        reverse=True
    )
    
    # Select operations until we hit the target
    to_recompute = []
    current_reduction = 0
    
    for op in candidates:
        memory_size = int(op.get('memory_size_bytes', 0))
        
        # Skip operations with no memory impact
        if memory_size <= 0:
            continue
            
        # Check if adding this operation exceeds our target
        if current_reduction + memory_size <= target_reduction:
            to_recompute.append(op)
            current_reduction += memory_size
            
            # BERT-specific: also include the other attention components if this is part of attention
            if 'attention' in op.get('rank', ''):
                related_ops = [c for c in candidates 
                               if c.get('rank', '') != op.get('rank', '') and
                               'attention' in c.get('rank', '')]
                for related in related_ops:
                    if related not in to_recompute:
                        to_recompute.append(related)
                        current_reduction += int(related.get('memory_size_bytes', 0))
        
        # Exit if we've hit our target
        if current_reduction >= target_reduction:
            break
    
    # Analyze results
    actual_reduction_percent = (current_reduction / total_memory) * 100 if total_memory > 0 else 0
    recompute_ranks = [int(op.get('rank')) for op in to_recompute]
    
    print(f"Target memory reduction: {target_reduction_percent:.1f}%")
    print(f"Actual reduction: {actual_reduction_percent:.1f}%")
    print(f"Memory savings: {current_reduction / (1024*1024):.1f} MB")
    print(f"Operations to recompute: {len(recompute_ranks)}")
    
    return to_recompute, recompute_ranks


def write_checkpoint_plan(profile_data, output_path, recompute_ranks):
    """Write the checkpoint plan to a CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=profile_data[0].keys())
        writer.writeheader()
        
        for row in profile_data:
            if int(row.get('rank', -1)) in recompute_ranks:
                row['to_recompute'] = 'yes'
            writer.writerow(row)
    
    print(f"Checkpoint plan written to {output_path}")


def plot_memory_impact(profile_data, recompute_ranks, output_path):
    """Plot the memory impact of checkpointing."""
    # Extract memory before checkpointing
    memory_before = [int(row.get('active_mem_bytes', 0)) for row in profile_data]
    
    # Calculate memory after checkpointing (subtract memory from recomputed ops)
    memory_after = memory_before.copy()
    for i, row in enumerate(profile_data):
        if int(row.get('rank', -1)) in recompute_ranks:
            memory_saved = int(row.get('memory_size_bytes', 0))
            for j in range(i, len(memory_after)):
                memory_after[j] -= memory_saved
    
    # Convert to MB
    memory_before_mb = [m / (1024*1024) for m in memory_before]
    memory_after_mb = [m / (1024*1024) for m in memory_after]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(memory_before))
    ax.plot(x, memory_before_mb, label='Before Checkpointing', color='tab:blue')
    ax.plot(x, memory_after_mb, label='After Checkpointing', color='tab:orange')
    
    # Fill area between curves
    ax.fill_between(x, memory_before_mb, memory_after_mb, alpha=0.3, color='tab:green')
    
    # Add labels and title
    ax.set_xlabel('Operation Index')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('BERT Memory Reduction with Activation Checkpointing')
    ax.legend()
    
    # Add annotations
    peak_before = max(memory_before_mb)
    peak_after = max(memory_after_mb)
    reduction = ((peak_before - peak_after) / peak_before) * 100
    
    ax.annotate(f"Peak before: {peak_before:.1f} MB", 
                xy=(0.02, 0.98), xycoords='axes fraction',
                verticalalignment='top')
    ax.annotate(f"Peak after: {peak_after:.1f} MB", 
                xy=(0.02, 0.93), xycoords='axes fraction',
                verticalalignment='top')
    ax.annotate(f"Reduction: {reduction:.1f}%", 
                xy=(0.02, 0.88), xycoords='axes fraction',
                verticalalignment='top')
    
    # Highlight operations to recompute
    for i, row in enumerate(profile_data):
        if int(row.get('rank', -1)) in recompute_ranks:
            ax.axvspan(i-0.5, i+0.5, alpha=0.2, color='tab:red')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Memory impact plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze BERT dynamic profiling data and create checkpointing plan")
    parser.add_argument("--input", default="bert_dynamic_profiling.csv", help="Input CSV from bert_profile.py")
    parser.add_argument("--output", default="bert_checkpoint_plan.csv", help="Output checkpoint plan CSV")
    parser.add_argument("--plot", default="bert_memory_reduction.png", help="Memory reduction plot")
    parser.add_argument("--target", type=float, default=35.0, help="Target memory reduction percentage")
    args = parser.parse_args()
    
    # Load and process data
    profile_data = read_dynamic_profile(args.input)
    profile_data = calculate_recompute_ratio(profile_data)
    
    # Find operations to recompute
    to_recompute, recompute_ranks = find_recompute_candidates(
        profile_data, 
        target_reduction_percent=args.target
    )
    
    # Write checkpoint plan
    write_checkpoint_plan(profile_data, args.output, recompute_ranks)
    
    # Plot memory impact
    plot_memory_impact(profile_data, recompute_ranks, args.plot)


if __name__ == "__main__":
    main() 