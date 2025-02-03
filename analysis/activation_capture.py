import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from utils.tools import load_model  # Your load_model function
from datasets.truthfulQAmq import TruthfulQAMultiChoiceDataset   # Function to load your dataset (e.g., from a JSON file)
from utils.patch_utils import InspectOutput  # Your InspectOutput class

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):

    # Experiment setup

    # Logger setup

    # Load the model and tokenizer
    model, tokenizer = load_model(cfg.model_name, use_flash_attention=cfg.use_flash_attention, device=cfg.device)

    # Load the dataset and data loader
    dataset = TruthfulQAMultiChoiceDataset(tokenizer, split=cfg.split, max_length=cfg.max_length)

    # Create a DataLoader
    eval_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers)

    # Evaluation loop
    