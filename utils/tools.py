import os
import json
import torch
import logging
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from pathlib import Path

def load_model(model_name: str, 
               use_flash_attention: bool = False, 
               device: str = "cuda", 
               torch_dtype=torch.float16) -> (torch.nn.Module, AutoTokenizer):
    """
    Loads a Hugging Face language model and tokenizer, and optionally enables Flash Attention.
    
    Args:
        model_name (str): The name or path of the pretrained model.
        use_flash_attention (bool): Whether to enable flash attention if available.
        device (str): The device to map the model to (e.g., 'cuda' or 'cpu').
        torch_dtype: The torch data type to use (e.g., torch.float16 for FP16).
    
    Returns:
        model (torch.nn.Module): The loaded language model.
        tokenizer (AutoTokenizer): The corresponding tokenizer.
    """
    # Load the model configuration (useful for any adjustments if needed)
    config = AutoConfig.from_pretrained(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the model with the specified dtype; use device_map="auto" if you want automatic multi-GPU mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        config=config,
        torch_dtype=torch_dtype,
        # Uncomment the following line if you want to let Hugging Face handle device placement automatically
        # device_map="auto"
    )
    
    # Move the model to the target device (if not using device_map)
    model.to(device)
    
    # Optionally enable Flash Attention if the model supports it and the user requested it.
    if use_flash_attention:
        if hasattr(model, "enable_flash_attention"):
            model.enable_flash_attention()
            print("Flash Attention enabled.")
        else:
            print("Flash Attention not supported for this model.")
    
    return model, tokenizer
