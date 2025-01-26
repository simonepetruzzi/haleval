import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_activations(activations, layer_name=None):
    """
    Visualize activations from a specific layer or all layers.
    
    Args:
        activations: Dictionary of layer activations from generate_text()
        layer_name: Optional specific layer to visualize
    """
    if layer_name is not None and layer_name in activations:
        # Visualize specific layer
        layer_data = activations[layer_name][0].cpu().numpy()
        plt.figure(figsize=(12, 6))
        sns.heatmap(layer_data, cmap='viridis')
        plt.title(f'Activations for layer: {layer_name}')
        plt.xlabel('Neuron Index')
        plt.ylabel('Sequence Position')
        plt.tight_layout()
        plt.savefig(f'activations_{layer_name}.png', dpi=300, bbox_inches='tight')
    else:
        # Visualize average activation magnitude across all layers
        layer_means = []
        layer_names = []
        for name, acts in activations.items():
            layer_means.append(acts[0].abs().mean().cpu().item())
            layer_names.append(name)
        
        plt.figure(figsize=(15, 5))
        plt.bar(range(len(layer_means)), layer_means)
        plt.xticks(range(len(layer_means)), layer_names, rotation=45, ha='right')
        plt.title('Average Activation Magnitude by Layer')
        plt.xlabel('Layer Name')
        plt.ylabel('Mean Activation')
        plt.tight_layout()
        plt.savefig('average_activation_magnitude.png', dpi=300, bbox_inches='tight')
        