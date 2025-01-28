from collections import defaultdict
from typing import Dict, List, Callable
import torch
from torch import Tensor


def get_activation_hook(layer_name: str, activations: Dict[str, List[Tensor]]) -> Callable:
    """
    Creates a forward hook for collecting layer activations.
    
    Args:
        layer_name: Name identifier for the layer
        activations: Dictionary to store activations
        
    Returns:
        Hook function to be registered
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            activations[layer_name].append(output[0].detach())
        else:
            activations[layer_name].append(output.detach())
    return hook

def register_activation_hooks(model: torch.nn.Module) -> tuple[Dict[str, List[torch.Tensor]], List[torch.utils.hooks.RemovableHandle]]:
    """
    Registers forward hooks to collect activations after MLP or self-attention layers in transformer blocks.

    Args:
        model: PyTorch model to analyze.

    Returns:
        Tuple containing:
        - Dictionary mapping layer names to their activations.
        - List of hook handles for removal.
    """
    activations = defaultdict(list)
    handles = []

    # Loop through all named modules
    for name, module in model.named_modules():
        # Register hooks only on the 'mlp' modules in transformer blocks
        if name.endswith(".mlp") or name.endswith(".self_attn"):
            handle = module.register_forward_hook(get_activation_hook(name, activations))
            handles.append(handle)

    return activations, handles


def calculate_mlp_activation_statistics(activations: Dict[str, List[Tensor]]) -> Dict[str, float]:
    mlp_stats = {}
    for layer_name, tensors in activations.items():
        if "mlp" in layer_name:
            try:
                # Take mean across sequence length dimension for each tensor
                means = [t.mean(dim=1) for t in tensors]  # Results in shape [1, 4096] for each tensor
                # Stack the means
                stacked_means = torch.stack(means, dim=0)  # Shape: [num_steps, 1, 4096]
                mean_magnitude = stacked_means.abs().mean().item()
                mlp_stats[layer_name] = mean_magnitude
            except RuntimeError as e:
                print(f"Error processing MLP layer {layer_name}: {e}")
    return mlp_stats

def calculate_attention_activation_statistics(activations: Dict[str, List[Tensor]]) -> Dict[str, float]:
    attention_stats = {}
    for layer_name, tensors in activations.items():
        if "self_attn" in layer_name:
            try:
                # Take mean across sequence length dimension for each tensor 
                means = [t.mean(dim=1) for t in tensors]  # Results in shape [1, 4096] for each tensor
                # Stack the means
                stacked_means = torch.stack(means, dim=0)  # Shape: [num_steps, 1, 4096]
                mean_magnitude = stacked_means.abs().mean().item()
                attention_stats[layer_name] = mean_magnitude
            except RuntimeError as e:
                print(f"Error processing attention layer {layer_name}: {e}")
    return attention_stats

