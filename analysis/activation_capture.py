import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch                      
from torch.utils.data import DataLoader
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.tools import load_model, generate_text_batch, save_answers_csv                                 
from dataset.popQA import PopQADataset
from utils.patch_utils import InspectOutput, parse_layer_idx      
from utils.tools import generate_responses_for_popqa_batch                  


@torch.inference_mode()
@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    # Experiment setup

    # Logger setup

    # Load the model and tokenizer
    model, tokenizer = load_model(cfg.model.model_name, use_flash_attention=cfg.model.use_flash_attention, device=cfg.device)

    # Load the dataset and data loader
    if cfg.dataset.dataset_name == "popQA":
        dataset = PopQADataset(tokenizer, split=cfg.dataset.split, max_length=cfg.dataset.max_length)

    dataloader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, num_workers=cfg.dataset.num_workers)
    
    # Set up directory to save data
    model_name = cfg.model.model_name.split("/")[-1]
    data_name = cfg.dataset.dataset_name
    save_dir = Path("cached_data") / model_name / data_name
    
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

    model.eval()
    for batch in dataloader:
        # Extract the list of prompt strings from the batch
        prompts = batch["prompt"]
        metadata = batch["metadata"]
        model_answers = generate_text_batch(model, tokenizer, prompts, max_length=30) 
        save_answers_csv(metadata, model_answers, "gemma-2-2b-it_responses.csv")
        print(model_answers)
        
    
    
if __name__ == '__main__':
    main()