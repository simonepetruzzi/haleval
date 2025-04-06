import numpy as np
import torch
import torch.nn as nn
from sae_lens import SAE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP

# Load SAE and move it to CUDA
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="llama_scope_lxm_8x",
    sae_id="l10m_8x",
)
print(sae.cfg)
sae.eval()
sae = sae.to('cuda')  

# Load activations
activations_path = "../false_activations_layer13_mlp.pt"
activations = torch.load(activations_path).to('cuda')
# use to project gemma activations to 4096
# project_4096 = nn.Linear(3584, 4096).to('cuda')
# project_4096.eval()

print("Activations shape:", activations.shape) 

with torch.no_grad():
    # if you want to restrict to first 2048 dimensions to match SAE input dim
    #activations = activations[:, :2048]
    # If you want to project to 4096, uncomment the following line
    #activations = project_4096(activations)
    latent_representations = []
    for i in range(activations.shape[0]):
        single_activation = activations[i].unsqueeze(0)  # Shape: [1, 2048]
        latent = sae.encode(single_activation)
        # Ensure latent has a batch dimension
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        latent_representations.append(latent)

# Stack into a single 2D tensor [N, latent_dim]
latent_matrix = torch.stack([x.squeeze(0) for x in latent_representations], dim=0)
print("Latent representations shape:", latent_matrix.shape)

# Quick stats check to ensure you’re not getting all zeros
print("Mean of latent_matrix:", latent_matrix.mean().item())
print("Std of latent_matrix:", latent_matrix.std().item())

# ---- Activation Analysis Per Sample ----
top_k = 5
top_values, top_indices = torch.topk(latent_matrix, k=top_k, dim=1)
top_indices_np = top_indices.cpu().numpy()

latent_dim = latent_matrix.shape[1]
top_unit_counts = np.zeros(latent_dim, dtype=int)

for indices in top_indices_np:
    for idx in indices:
        top_unit_counts[idx] += 1

# 1) Sort units by frequency descending and pick top 50
sorted_indices = np.argsort(-top_unit_counts)  # negative for descending order
top_n = 50
top_units = sorted_indices[:top_n]

# --- Visualization 1: Bar Plot of the Top 50 Units ---
plt.figure(figsize=(12, 6))
plt.bar(np.arange(top_n), top_unit_counts[top_units], width=0.8)
plt.xticks(np.arange(top_n), top_units, rotation=90)
plt.xlabel("Latent Unit Index (Top 50 by Frequency)")
plt.ylabel(f"Frequency in Top {top_k}")
plt.title("Top 50 Most Frequently Activated Latent Units")
plt.tight_layout()
plt.savefig("top_unit_frequencies.png")


# --- Visualization 2: Heatmap (Top 50 Units x up to 100 Samples) ---
# Create a binary matrix: 1 if unit is in top_k, else 0
binary_top_matrix = np.zeros((latent_matrix.shape[0], latent_dim))
for i, indices in enumerate(top_indices_np):
    binary_top_matrix[i, indices] = 1

# Subset rows (samples) and columns (units)
num_samples_to_show = min(100, latent_matrix.shape[0])
binary_top_subset = binary_top_matrix[:num_samples_to_show, top_units]

plt.figure(figsize=(12, 8))
plt.imshow(binary_top_subset, aspect='auto', cmap='viridis')
plt.xticks(np.arange(top_n), top_units, rotation=90)
plt.xlabel("Latent Unit Index (Top 50 by Frequency)")
plt.ylabel("Sample Index")
plt.title(f"Heatmap of Top {top_k} Activated Latent Units (per sample) - First {num_samples_to_show} Samples")
plt.colorbar(label="Activation (1 = top, 0 = otherwise)")
plt.tight_layout()
plt.savefig("top_activation_heatmap.png")


# --- Visualization 3: PCA of Latent Representations ---
# Color each sample by the single most activated latent unit
most_activated_unit = torch.argmax(latent_matrix, dim=1).cpu().numpy()

pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent_matrix.cpu().numpy())

plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                      c=most_activated_unit, cmap='tab20', alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Latent Representations Colored by Most Activated Unit")
cbar = plt.colorbar(scatter, label="Latent Unit Index")
plt.savefig("pca_latent_representations.png")


# --- Visualization 4: t-SNE of Latent Representations ---
# Color each sample by the single most activated latent unit
most_activated_unit = torch.argmax(latent_matrix, dim=1).cpu().numpy()

# Create a t-SNE instance. Adjust parameters like perplexity if needed.
tsne = TSNE(n_components=2, random_state=42)
latent_2d = tsne.fit_transform(latent_matrix.cpu().numpy())

plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                      c=most_activated_unit, cmap='tab20', alpha=0.7)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("t-SNE of Latent Representations Colored by Most Activated Unit")
cbar = plt.colorbar(scatter, label="Latent Unit Index")
plt.savefig("tsne_latent_representations.png")

# --- Visualization 5: UMAP of Latent Representations ---
# Color each sample by the single most activated latent unit
most_activated_unit = torch.argmax(latent_matrix, dim=1).cpu().numpy()

# Create a UMAP instance. Adjust parameters like n_neighbors or min_dist as needed.
umap_embedder = UMAP(n_components=2, random_state=42)
latent_2d = umap_embedder.fit_transform(latent_matrix.cpu().numpy())

plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                      c=most_activated_unit, cmap='tab20', alpha=0.7)
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("UMAP of Latent Representations Colored by Most Activated Unit")
cbar = plt.colorbar(scatter, label="Latent Unit Index")
plt.savefig("umap_latent_representations.png")


# import numpy as np
# import torch
# from sae_lens import SAE
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# # Load SAE and move it to CUDA
# sae, cfg_dict, sparsity = SAE.from_pretrained(
#     release="gemma-scope-2b-pt-att-canonical",
#     sae_id="layer_13/width_16k/canonical",
# )
# print(sae.cfg)
# sae.eval()
# sae = sae.to('cuda')  

# # Load activations
# activations_path = "../true_activations_layer13_attention.pt"
# activations = torch.load(activations_path)
# print("Activations shape:", activations.shape) 

# with torch.no_grad():
#     # Restrict to first 2048 dimensions to match SAE input dim
#     activations = activations[:, :2048].to('cuda')
#     latent_representations = []
#     for i in range(activations.shape[0]):
#         single_activation = activations[i].unsqueeze(0)  # Shape: [1, 2048]
#         latent = sae.encode(single_activation)
#         # Check latent dimensions. If SAE.encode returns a 1D tensor, add a batch dim.
#         if latent.dim() == 1:
#             latent = latent.unsqueeze(0)
#         latent_representations.append(latent)

# # Use torch.stack instead of torch.cat to preserve the sample dimension.
# # Here, we also remove the extra batch dimension if present.
# latent_matrix = torch.stack([latent.squeeze(0) for latent in latent_representations], dim=0)
# print("Latent representations shape:", latent_matrix.shape)

# # ---- Activation Analysis Per Sample ----

# # Set the number of top activated units to consider per sample
# top_k = 15

# # For each sample, get the top k activated latent units and their values
# top_values, top_indices = torch.topk(latent_matrix, k=top_k, dim=1)
# # Convert indices to numpy for easier processing
# top_indices_np = top_indices.cpu().numpy()

# # Count frequency of each latent unit appearing among the top activations
# latent_dim = latent_matrix.shape[1]
# top_unit_counts = np.zeros(latent_dim, dtype=int)
# for indices in top_indices_np:
#     for idx in indices:
#         top_unit_counts[idx] += 1

# # --- Visualization 1: Bar Plot of Top Unit Frequencies ---
# plt.figure(figsize=(12, 6))
# plt.bar(np.arange(latent_dim), top_unit_counts)
# plt.xlabel("Latent Unit Index")
# plt.ylabel(f"Frequency in Top {top_k}")
# plt.title("Frequency of Latent Units as Top Activated Across Samples")
# plt.savefig("top_unit_frequencies.png")

# # --- Visualization 2: Heatmap of Top Activation per Sample ---
# # Create a binary matrix: For each sample, mark top activated units as 1
# binary_top_matrix = np.zeros((latent_matrix.shape[0], latent_dim))
# for i, indices in enumerate(top_indices_np):
#     binary_top_matrix[i, indices] = 1

# # Visualize a subset (e.g., up to 100 samples) for clarity
# num_samples_to_show = min(100, latent_matrix.shape[0])
# plt.figure(figsize=(12, 8))
# plt.imshow(binary_top_matrix[:num_samples_to_show], aspect='auto', cmap='viridis')
# plt.xlabel("Latent Unit Index")
# plt.ylabel("Sample Index")
# plt.title(f"Heatmap of Top {top_k} Activated Latent Units (per sample)")
# plt.colorbar(label="Activation (1 = top, 0 = otherwise)")
# plt.savefig("top_activation_heatmap.png")

# # --- Visualization 3: PCA of Latent Representations ---
# # For each sample, determine the single most activated unit (for coloring)
# most_activated_unit = torch.argmax(latent_matrix, dim=1).cpu().numpy()

# # Reduce latent dimensions to 2D with PCA for visualization
# pca = PCA(n_components=2)
# latent_2d = pca.fit_transform(latent_matrix.cpu().numpy())

# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=most_activated_unit, cmap='tab20', alpha=0.7)
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("PCA of Latent Representations Colored by Most Activated Unit")
# plt.colorbar(scatter, label="Latent Unit Index")
# plt.savefig("pca_latent_representations.png")
