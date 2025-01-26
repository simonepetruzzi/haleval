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

def register_activation_hooks(model: torch.nn.Module) -> tuple[Dict[str, List[Tensor]], List[torch.utils.hooks.RemovableHandle]]:
    """
    Registers forward hooks on all layers of the model to collect activations.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Tuple containing:
        - Dictionary mapping layer names to their activations
        - List of hook handles for removal
    """
    activations = defaultdict(list)
    handles = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            handle = module.register_forward_hook(
                get_activation_hook(name, activations)
            )
            handles.append(handle)
            
    return activations, handles

