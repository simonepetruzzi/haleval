import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from torch import Tensor

from utils.activation_hooks import calculate_attention_activation_statistics, calculate_mlp_activation_statistics

def layer_name_to_index(name):
    """Extract layer number from name like 'model.layers.11.mlp'"""
    try:
        # Find the number between 'layers.' and the next dot
        layer_num = int(name.split('layers.')[1].split('.')[0])
        return layer_num
    except:
        return float('inf')


def plot_mlp_activation_statistics(activations: Dict[str, List[Tensor]], save_path: str = None):
    """
    Plot mean activation magnitudes for MLP layers.

    Args:
        activations: Dictionary containing activations for each layer.
        save_path: Optional path to save the plot.
    """
    # Filter MLP activations
    mlp_activations = {k: v for k, v in activations.items() if "mlp" in k}

    # Calculate statistics
    mlp_stats = calculate_mlp_activation_statistics(mlp_activations)

    # Sort by layer order
    #sorted_mlp_stats = {k: mlp_stats[k] for k in sorted(mlp_stats.keys())}
    sorted_mlp_stats = {k: mlp_stats[k] for k in sorted(mlp_stats.keys(), 
                                                   key=layer_name_to_index)}
    # Prepare x and y for plotting
    x = list(range(len(sorted_mlp_stats)))  # Numerical indices for layers
    y = list(sorted_mlp_stats.values())    # Mean magnitudes
    labels = list(sorted_mlp_stats.keys()) # Layer names

    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(x, y, marker="o", linestyle="-", label="MLP Mean Activation Magnitude", color="salmon")
    plt.xticks(ticks=x, labels=labels, rotation=45, ha="right", fontsize=8)
    plt.xlabel("MLP Layers", fontsize=12)
    plt.ylabel("Mean Activation Magnitude", fontsize=12)
    plt.title("Mean Activation Magnitude for MLP Layers", fontsize=14)
    plt.tight_layout()
    plt.legend()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_attention_activation_statistics(activations: Dict[str, List[Tensor]], save_path: str = None):
    """
    Plot mean activation magnitudes for attention layers.

    Args:
        activations: Dictionary containing activations for each layer.
        save_path: Optional path to save the plot.
    """
    # Filter attention activations
    attention_activations = {k: v for k, v in activations.items() if "self_attn" in k}

    # Calculate statistics
    attention_stats = calculate_attention_activation_statistics(attention_activations)

    # Sort by layer order
    #sorted_attention_stats = {k: attention_stats[k] for k in sorted(attention_stats.keys())}
    sorted_attention_stats = {k: attention_stats[k] for k in sorted(attention_stats.keys(),
                                                              key=layer_name_to_index)}
    # Prepare x and y for plotting
    x = list(range(len(sorted_attention_stats)))  # Numerical indices for layers
    y = list(sorted_attention_stats.values())    # Mean magnitudes
    labels = list(sorted_attention_stats.keys()) # Layer names

    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(x, y, marker="o", linestyle="-", label="Attention Mean Activation Magnitude", color="skyblue")
    plt.xticks(ticks=x, labels=labels, rotation=45, ha="right", fontsize=8)
    plt.xlabel("Attention Layers", fontsize=12)
    plt.ylabel("Mean Activation Magnitude", fontsize=12)
    plt.title("Mean Activation Magnitude for Attention Layers", fontsize=14)
    plt.tight_layout()
    plt.legend()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

