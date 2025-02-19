import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch                      
from torch.utils.data import DataLoader
from utils.tools import load_model                               
from datasets.truthfulQAmq import TruthfulQAMultiChoiceDataset   
from utils.patch_utils import InspectOutput                      

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
    save_dir =  "cached_data" / model_name
    data_name = cfg.dataset.dataset_name
    if data_name != "truthfulQAmq":
        save_dir = save_dir / data_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set up the file path to save the activation features
    feature_records = []
    output_path = save_dir / "activation_features.json"

    if os.path.exists(output_path):
        print(f"{output_path} exists")  

    
    num_layer = cfg.model.layers
    
    # Evaluation loop


    