import os
import torch
from pathlib import Path
import argparse

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
    Combine the individual activation files for each layer and module type into a single tensor.
    If an activation tensor does not have the expected shape (i.e. its first dimension is not 1),
    adjust it by computing the mean across the rows.
    After combining, the individual files are deleted.
    """
    save_base_dir = Path(save_base_dir)
    for aa in analyse_activation_list:
        act_dir = save_base_dir / f"activation_{aa}" / activation_type
        if not act_dir.exists():
            continue
        act_files = [f for f in os.listdir(act_dir) if len(f.split("-")) == 2 and f.endswith(".pt")]
        layer_group_files = {lid: [] for lid in target_layers}
        for f in act_files:
            layer_idx, instance_idx = parse_layer_id_and_instance_id(f)
            if layer_idx is not None:
                layer_group_files[layer_idx].append((f, instance_idx))
        for lid in target_layers:
            files = layer_group_files[lid]
            if not files:
                continue
            files_sorted = sorted(files, key=lambda x: x[1])
            acts = []
            loaded_paths = []
            for idx, (fname, instance_idx) in enumerate(files_sorted):
                # Check that file order matches instance index
                assert idx == instance_idx, f"Mismatch in ordering for layer {lid}: expected {idx} but got {instance_idx}"
                act = torch.load(act_dir / fname)
                # If the tensor's first dimension is not 1, adjust by taking the mean over that dimension.
                if act.dim() > 0 and act.shape[0] != 1:
                    act = act.mean(dim=0, keepdim=True)
                acts.append(act)
                loaded_paths.append(act_dir / fname)
            acts_tensor = torch.stack(acts)
            print(f"{act_dir.parent.name} {activation_type} {aa} layer {lid} combined shape: {acts_tensor.shape}")
            save_path = act_dir / f"layer{lid}_activations.pt"
            torch.save(acts_tensor, save_path)
            for p in loaded_paths:
                os.remove(p)

def main():
    parser = argparse.ArgumentParser(description="Combine activation files into single tensors per layer.")
    parser.add_argument(
        "--save_base_dir",
        type=str,
        required=True,
        help="Base directory where activation files are stored."
    )
    parser.add_argument(
        "--activation_type",
        type=str,
        default="conflict",
        help="Activation type identifier (default: 'conflict')."
    )
    parser.add_argument(
        "--modules",
        type=str,
        nargs="+",
        default=["mlp", "attn", "hidden"],
        help="List of module types to analyze (default: mlp attn hidden)."
    )
    args = parser.parse_args()

    combine_activations(
        save_base_dir=args.save_base_dir,
        target_layers=list(range(32)),  # Layers 0 to 31
        activation_type=args.activation_type,
        analyse_activation_list=args.modules
    )

if __name__ == "__main__":
    main()
