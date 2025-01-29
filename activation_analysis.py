import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import wandb
import csv
from datasets import load_dataset

from utils.activation_hooks import register_activation_hooks
from utils.logger import log_activations_wandb
from utils.visualize import plot_mlp_activation_statistics, plot_attention_activation_statistics


# Initialize W&B
wandb.init(project="haleval")

# Set device to GPU if available
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Load the model and tokenizer from Hugging Face
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with original FP32 weights
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Load the model with 16bit quantization (original is 32bit)
#model = AutoModelForCausalLM.from_pretrained(model_name).half().to(device)

# Configure 8-bit quantization
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the model with quantization
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     quantization_config=quantization_config, 
#     device_map="auto"
# )

# Import dataset
def generate_responses_for_truthfulqa(model, tokenizer, device, output_csv="truthfulqa_responses.csv", split="validation"):
    """
    Load the TruthfulQA dataset, iterate over each question in the specified split,
    generate model outputs, and save the results to a CSV file.

    Args:
        model: The transformer model to use for generation (already loaded).
        tokenizer: The tokenizer corresponding to the model.
        device: The torch.device to run inference on (CPU or GPU).
        output_csv (str): Path of the CSV file where results will be saved.
        split (str): Which dataset split to use (e.g. "train", "validation", "test").
    """
    # 1) Load the dataset
    #    - We'll use the 'generation' configuration of TruthfulQA,
    #      which includes the free-form questions.
    ds_split = load_dataset("truthful_qa", "generation")
    
    # 2) Select the split (e.g. validation)
    questions = ds_split[split]

    # 3) Prepare CSV writer
    fieldnames = ["idx", "question", "model_response"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # 4) Loop over the dataset examples
        for idx, question in enumerate(questions[:10]):
            
            # 5) Generate response using your existing function
            #    (make sure generate_text is already defined and uses the correct model, tokenizer, device)
            model_answer = generate_text(question)

            # 6) Save to CSV
            writer.writerow({
                "idx": idx,
                "question": question,
                "model_response": model_answer
            })

    print(f"Finished generating responses. Results saved to {output_csv}.")

# Function to generate text based on a prompt
def generate_text(prompt, max_length=500):

    activations, handles = register_activation_hooks(model)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generation_config = {
        "max_length": max_length,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
    }
    outputs = model.generate(**inputs, **generation_config)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Log activations to W&B
    # log_activations_wandb(activations)

    # Visualize activations
    plot_mlp_activation_statistics(activations, './images/mlp_activations')
    plot_attention_activation_statistics(activations, './images/attention_activations')
    
    for handle in handles:
        handle.remove()

    return generated_text

if __name__ == "__main__":
    #prompt = "Who actually said, 'Let them eat cake' ?"
    #generated_text = generate_text(prompt)
    #print(generated_text)
    generate_responses_for_truthfulqa(model, tokenizer, device, output_csv="truthfulqa_responses.csv", split="validation") 
    wandb.finish()