import os
import torch
from pathlib import Path
from tqdm import tqdm
from utils.patch_utils import InspectOutput  # Your hook class and helper functions

def save_activations_to_files(model, dataloader, target_layers, save_base_dir,
                              activation_type='hallucination', device='cuda'):
    """
    Capture and save activations from a model during inference.
    
    For each example (batch size assumed to be 1), the function captures
    activations from the specified modules (hidden, self-attn, and mlp) via hooks.
    Each activation is saved in a file named "layer{layer_idx}-id{batch_index}.pt" 
    inside a subdirectory that depends on the module type.
    
    Args:
        model (torch.nn.Module): The model to inspect.
        dataloader (DataLoader): DataLoader yielding batches from the dataset.
        target_layers (List[int]): List of layer indices to capture.
        save_base_dir (str or Path): Base directory to store activation files.
        activation_type (str): Identifier for the activation condition (e.g. 'conflict' or 'none_conflict').
        device (str): Device on which the model is running.
    """
    save_base_dir = Path(save_base_dir)
    # Define directories for each activation type
    hidden_save_dir = save_base_dir / "activation_hidden" / activation_type
    mlp_save_dir    = save_base_dir / "activation_mlp"    / activation_type
    attn_save_dir   = save_base_dir / "activation_attn"   / activation_type

    for d in [hidden_save_dir, mlp_save_dir, attn_save_dir]:
        os.makedirs(d, exist_ok=True)

    # Build list of module names to hook.
    # For each target layer, we hook the base layer, self-attention, and mlp sub-modules.
    module_names = []
    for idx in target_layers:
        module_names.append(f"model.layers.{idx}")
        module_names.append(f"model.layers.{idx}.self_attn")
        module_names.append(f"model.layers.{idx}.mlp")

    batch_index = 0
    # Loop over the dataset (batch size assumed to be 1)
    tqdm_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Saving activations", leave=True)
    for bid, batch in tqdm_bar:
        tqdm_bar.set_description(f"Processing batch {bid}")
        # Use your hook context to capture outputs from the desired modules.
        with InspectOutput(model, module_names, move_to_cpu=True, last_position=True) as inspector:
            # Assuming your batch contains an input key named "input_ids"
            inputs = batch["input_ids"].to(device)
            # Forward pass (modify keyword arguments if needed)
            model(input_ids=inputs, use_cache=False, return_dict=True)
        
        # Iterate over the captured outputs.
        for module_name, ac in inspector.catcher.items():
            # ac is expected to be a tensor of shape [batch_size, hidden_dim] (since last_position=True)
            # With batch_size==1, select the first element.
            ac_last = ac[0].float()
            try:
                # Parse the layer index from the module name.
                layer_idx = int(module_name.split(".")[2])
            except Exception:
                continue
            # Construct the filename: "layer{layer_idx}-id{batch_index}.pt"
            save_name = f"layer{layer_idx}-id{bid}.pt"
            if "mlp" in module_name:
                torch.save(ac_last, mlp_save_dir / save_name)
            elif "self_attn" in module_name:
                torch.save(ac_last, attn_save_dir / save_name)
            else:
                torch.save(ac_last, hidden_save_dir / save_name)
        batch_index += 1

    # Once all batches are processed, combine individual files into a single tensor per layer.
    combine_activations(save_base_dir, target_layers, activation_type=activation_type)

def parse_layer_id_and_instance_id(filename: str):
    """
    Parse the layer and instance index from a filename formatted as "layer{layer_idx}-id{instance_idx}.pt"
    """
    try:
        layer_part, id_part = filename.split("-")
        layer_idx = int(layer_part[len("layer"):])
        instance_idx = int(id_part[len("id"):-len(".pt")])
        return layer_idx, instance_idx
    except Exception as e:
        print("Error parsing filename:", filename)
        return None, None

def combine_activations(save_base_dir, target_layers, activation_type='conflict', analyse_activation_list=["mlp", "attn", "hidden"]):
    """
    Combine the individual activation files for each layer and module type into a single tensor,
    then delete the individual files.
    
    For each module type (mlp, attn, hidden), the function:
      - Lists the files in the corresponding directory.
      - Groups the files by layer (using the filename).
      - Sorts the files for each layer by their instance index.
      - Loads and stacks the tensors to form a single tensor.
      - Saves the combined tensor as "layer{layer_idx}_activations.pt".
      - Removes the individual activation files.
    """
    save_base_dir = Path(save_base_dir)
    for aa in analyse_activation_list:
        act_dir = save_base_dir / f"activation_{aa}" / activation_type
        if not act_dir.exists():
            continue
        act_files = [f for f in os.listdir(act_dir) if len(f.split("-")) == 2 and f.endswith(".pt")]
        # Group files by layer index
        layer_group_files = {lid: [] for lid in target_layers}
        for f in act_files:
            layer_idx, instance_idx = parse_layer_id_and_instance_id(f)
            if layer_idx is not None:
                layer_group_files[layer_idx].append((f, instance_idx))
        # For each layer, sort and stack the files.
        for lid in target_layers:
            files = layer_group_files[lid]
            if not files:
                continue
            files_sorted = sorted(files, key=lambda x: x[1])
            acts = []
            loaded_paths = []
            for idx, (fname, instance_idx) in enumerate(files_sorted):
                # Optionally, verify the ordering:
                assert idx == instance_idx, f"Mismatch in ordering for layer {lid}: expected {idx} but got {instance_idx}"
                act = torch.load(act_dir / fname)
                acts.append(act)
                loaded_paths.append(act_dir / fname)
            acts_tensor = torch.stack(acts)
            print(f"{act_dir.parent.name} {activation_type} {aa} layer {lid} combined shape: {acts_tensor.shape}")
            save_path = act_dir / f"layer{lid}_activations.pt"
            torch.save(acts_tensor, save_path)
            # Remove individual files.
            for p in loaded_paths:
                os.remove(p)
