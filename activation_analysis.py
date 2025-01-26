import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import wandb

from utils.activation_hooks import register_activation_hooks
from utils.logger import log_activations_wandb


wandb.init(project="haleval")

# Set device to GPU if available
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Load the model and tokenizer from Hugging Face
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure 8-bit quantization
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the model with quantization
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     quantization_config=quantization_config, 
#     device_map="auto"
# )

# Load the model with 16bit quantization (original is 32bit)
model = AutoModelForCausalLM.from_pretrained(model_name).half().to(device)


# Function to generate text based on a prompt
def generate_text(prompt, max_length=500):

    activations, handles = register_activation_hooks(model)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generation_config = {
        "max_length": max_length,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
    }
    outputs = model.generate(**inputs, **generation_config)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Log activations to W&B
    log_activations_wandb(activations)

    for handle in handles:
        handle.remove()

    return generated_text, activations

if __name__ == "__main__":
    prompt = "Once upon a time"
    generated_text, activations = generate_text(prompt)
    print(generated_text)
    wandb.finish()