import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import wandb
import csv
from datasets import load_dataset

from utils.activation_hooks import register_activation_hooks
#from utils.logger import log_activations_wandb
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

# def generate_responses_for_truthfulqa(model, tokenizer, device, output_csv="truthfulqa_responses_corrected.csv", split="validation"):
#     """
#     Load the TruthfulQA dataset, iterate over each question in the specified split,
#     generate model outputs, and save the results to a CSV file.
    
#     Args:
#         model: The transformer model to use for generation (already loaded).
#         tokenizer: The tokenizer corresponding to the model.
#         device: The torch.device to run inference on (CPU or GPU).
#         output_csv (str): Path of the CSV file where results will be saved.
#         split (str): Which dataset split to use (e.g. "train", "validation", "test").
#     """
#     # Load the dataset
#     ds_split = load_dataset("truthful_qa", "generation")[split]
    
#     # Prepare CSV writer
#     fieldnames = ["idx", "question", "best_answer", "correct_answers", "incorrect_answers", "model_response"]
#     with open(output_csv, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()

#         # Iterate over the dataset
#         for idx, example in enumerate(ds_split):
#             question = example["question"]
#             best_answer = example["best_answer"]
#             correct_answers = " | ".join(example["correct_answers"]) if isinstance(example["correct_answers"], list) else example["correct_answers"]
#             incorrect_answers = " | ".join(example["incorrect_answers"]) if isinstance(example["incorrect_answers"], list) else example["incorrect_answers"]
#             prompt = f"""Answer the following questions concisely and accurately, using no more than one sentence (it can be long as you want, but no more than one sentence). Follow the format of the examples provided.

#                                     Example 1:
#                                     Question: What is the capital of France?
#                                     Answer: Paris

#                                     Example 2:
#                                     Question: Who wrote the novel "1984"?
#                                     Answer: George Orwell

#                                     Example 3:
#                                     Question: What is the boiling point of water in Celsius?
#                                     Answer: 100°C

#                                     Now, answer the following:
#                                     Question: {question}
#                                     Answer:"""
#             # Generate response
#             model_answer = generate_text(prompt)
            
#             # Save to CSV
#             writer.writerow({
#                 "idx": idx,
#                 "question": question,
#                 "best_answer": best_answer,
#                 "correct_answers": correct_answers,
#                 "incorrect_answers": incorrect_answers,
#                 "model_response": model_answer
#             })
    
#     print(f"Finished generating responses. Results saved to {output_csv}.")

import csv
import random
from datasets import load_dataset

def generate_responses_for_truthfulqa(model, tokenizer, device, output_csv="truthfulqa_responses_corrected.csv", split="validation"):
    """
    Load the TruthfulQA dataset, iterate over each question,
    generate model outputs, and save the results to a CSV file.
    
    Args:
        model: The transformer model to use for generation (already loaded).
        tokenizer: The tokenizer corresponding to the model.
        device: The torch.device to run inference on (CPU or GPU).
        output_csv (str): Path of the CSV file where results will be saved.
        split (str): Which dataset split to use (e.g. "train", "validation", "test").
    """
    # Load the dataset
    ds_split = load_dataset("truthful_qa", "generation")[split]
    
    # Prepare CSV writer
    fieldnames = ["idx", "question", "best_answer", "correct_answers", "incorrect_answers", "model_response"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, example in enumerate(ds_split):
            question = example["question"]
            best_answer = example["best_answer"]
            correct_answers = " | ".join(example["correct_answers"]) if isinstance(example["correct_answers"], list) else example["correct_answers"]
            incorrect_answers = " | ".join(example["incorrect_answers"]) if isinstance(example["incorrect_answers"], list) else example["incorrect_answers"]
            
            # Select three random examples from the dataset
            examples = random.sample(list(ds_split), 3)
            example_texts = "\n\n".join(
                [f"Example {i+1}:\nQuestion: {ex['question']}\nAnswer: {ex['best_answer']}" for i, ex in enumerate(examples)]
            )
            
            prompt = f"""Answer the following questions concisely and accurately, using no more than one sentence. Follow the format of the examples provided.
            
{example_texts}
            
Now, answer the following:
Question: {question}
Answer:"""
            
            # Generate response
            model_answer = generate_text(prompt)
            
            # Save to CSV
            writer.writerow({
                "idx": idx,
                "question": question,
                "best_answer": best_answer,
                "correct_answers": correct_answers,
                "incorrect_answers": incorrect_answers,
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
    #plot_mlp_activation_statistics(activations, './images/mlp_activations')
    #plot_attention_activation_statistics(activations, './images/attention_activations')
    
    for handle in handles:
        handle.remove()

    return generated_text

if __name__ == "__main__":
    #prompt = "Who actually said, 'Let them eat cake' ?"
    #generated_text = generate_text(prompt)
    #print(generated_text)
    generate_responses_for_truthfulqa(model, tokenizer, device, output_csv="truthfulqa_responses.csv", split="validation") 
    wandb.finish()
