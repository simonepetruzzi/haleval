import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch                      
from torch.utils.data import DataLoader
from utils.tools import load_model                               
from datasets.truthfulQAmq import TruthfulQAMultiChoiceDataset   
from utils.patch_utils import InspectOutput, parse_layer_idx                      

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):

    # Experiment setup

    # Logger setup

    # Load the model and tokenizer
    model, tokenizer = load_model(cfg.model.model_name, use_flash_attention=cfg.model.use_flash_attention, device=cfg.device)

    # Load the dataset and data loader
    dataset = TruthfulQAMultiChoiceDataset(tokenizer, split=cfg.dataset.split, max_length=cfg.dataset.max_length)

    # Create a DataLoader
    eval_dataloader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=cfg.dataset.shuffle, num_workers=cfg.dataset.num_workers)

    # Set up directory to save data
    model_name = model_name.split("/")[-1]
    data_name = cfg.dataset.dataset_name
    save_dir =  "cached_data" / model_name / data_name
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set up the file path to save the activation features
    feature_records = []
    output_path = save_dir / "activation_features.json"

    if os.path.exists(output_path):
        print(f"{output_path} exists")  

    
    num_layer = cfg.model.layers
    module_names = []
    for idx in range(num_layer):
        module_names.append(f"model.layers.{idx}")
        module_names.append(f"model.layers.{idx}.self_attn")
        module_names.append(f"model.layers.{idx}.mlp")
    
    # Evaluation loop
    model.eval()

    move_to_cpu = cfg.get("move_to_cpu", True)
    last_position = cfg.get("last_position", True)

    for batch_idx, batch in enumerate(eval_dataloader):
        
        input_ids = batch["input_ids"].to(cfg.device)

        with torch.no_grad():
            with InspectOutput(model, module_names, move_to_cpu=move_to_cpu, last_position=last_position) as inspector:
                _ = model(input_ids=input_ids)

        # Process captured activations from each module
        for module, activation in inspector.catcher.items():
            # If last_position is True, activation shape is likely [batch_size, hidden_dim]
            # We extract the activation for the first example in the batch as an example
            activation_extracted = activation[0].cpu() 
            # Parse the layer index for clarity in naming/logging
            try:
                layer_idx = parse_layer_idx(module)
            except ValueError:
                layer_idx = "unknown"

            # Construct a filename encoding the module name and batch index
            filename = f"{module.replace('.', '_')}_batch{batch_idx}.pt"
            file_path = os.path.join(save_dir, filename)
            torch.save(activation_extracted, file_path)
            print(f"Saved activation for module '{module}' (layer {layer_idx}) from batch {batch_idx} to {file_path}")


    