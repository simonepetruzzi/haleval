import numpy as np
import torch
from sae_lens import SAE


sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gemma-scope-2b-pt-att-canonical",
    sae_id = "layer_13/width_16k/canonical",
    )

sae.eval()

activations_path = "./outputs/2025-03-21/13-04-51/cached_data/gemma-2-2b-it/popQA/activation_attn/conflict/layer13_activations.pt"

activations = torch.load(activations_path)

print("Activations shape:", activations.shape)

# path_to_params = hf_hub_download(
#     repo_id="google/gemma-scope-2b-pt-att",
#     filename="layer_20/width_16k/canonical/params.npz",
#     force_download=False,
# )

# params = np.load(path_to_params)
# pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
# {k:v.shape for k, v in pt_params.items()} 