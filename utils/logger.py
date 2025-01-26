import wandb
import numpy as np
import matplotlib.pyplot as plt

def log_activations_wandb(activations, step=0):
    """Log activation visualizations to W&B"""
    # Log per-layer statistics
    for layer_name, acts in activations.items():
        act_tensor = acts[0].cpu()
        wandb.log({
            f"{layer_name}/mean": act_tensor.mean(),
            f"{layer_name}/std": act_tensor.std(),
            f"{layer_name}/histogram": wandb.Histogram(act_tensor.numpy().flatten()),
            f"{layer_name}/heatmap": wandb.Image(
                plt.figure(figsize=(10,10), dpi=100),
                caption=f"Activation heatmap for {layer_name}"
            )
        }, step=step)