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


def generate_responses_for_popqa(model, tokenizer, device, output_csv="gemma2b_popqa_responses.csv"):
    """
    Load the PopQA dataset, iterate over each question,
    generate model outputs, and save the results to a CSV file.
    
    Args:
        model: The transformer model to use for generation (already loaded).
        tokenizer: The tokenizer corresponding to the model.
        device: The torch.device to run inference on (CPU or GPU).
        output_csv (str): Path of the CSV file where results will be saved.
        split (str): Which dataset split to use (e.g. "train", "validation", "test").
    """
    # Load the dataset
    ds_split = load_dataset("akariasai/PopQA")['test'] 

    # Prepare CSV writer
    fieldnames = ["idx", "question", "possible_answers", "model_response"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        prompts = []
        for idx, example in enumerate(ds_split):
            question = example["question"]
            possible_answers = " | ".join(example["possible_answers"]) if isinstance(example["possible_answers"], list) else example["possible_answers"]
            # Select three random examples from the dataset
            examples = random.sample(list(ds_split), 10)
            example_texts = "\n\n".join(
                [
                    f"Example {i+1}:\nQuestion: {ex['question']}\nAnswer: {ex['possible_answers'].split(' | ')[0]}"
                    if isinstance(ex["possible_answers"], str)
                    else f"Example {i+1}:\nQuestion: {ex['question']}\nAnswer: {ex['possible_answers'][0]}"
                    for i, ex in enumerate(examples)
                ]
            )   
            
            prompt = f"""Answer the following questions concisely and accurately, using no more than one sentence. Follow the format of the examples provided.
            
            {example_texts}
                        
            Now, answer the following:
            Question: {question}
            Answer:"""
            prompts.append(prompt)
            
            # Generate response
            model_answer = generate_text(prompt)
            
            # Save to CSV
            writer.writerow({
                "idx": idx,
                "question": question,
                "possible_answers": possible_answers,
                "model_response": model_answer
            })
    
    print(f"Finished generating responses. Results saved to {output_csv}.")

def generate_responses_for_popqa_batch(model, tokenizer, device, output_csv="gemma2b_popqa_responses.csv"):
    """
    Load the PopQA dataset, iterate over each question,
    generate model outputs, and save the results to a CSV file.
    
    Args:
        model: The transformer model to use for generation (already loaded).
        tokenizer: The tokenizer corresponding to the model.
        device: The torch.device to run inference on (CPU or GPU).
        output_csv (str): Path of the CSV file where results will be saved.
        split (str): Which dataset split to use (e.g. "train", "validation", "test").
    """
    # Load the dataset
    ds_split = load_dataset("akariasai/PopQA")['test'] 

    # Prepare CSV writer
    fieldnames = ["idx", "question", "possible_answers", "model_response"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        prompts = []
        metadata = []
    for idx, example in enumerate(ds_split):
        question = example["question"]
        possible_answers = (
            " | ".join(example["possible_answers"]) 
            if isinstance(example["possible_answers"], list) 
            else example["possible_answers"]
        )
        # Select 10 random examples from the dataset for context
        examples = random.sample(list(ds_split), 10)
        example_texts = "\n\n".join(
            [
                f"Example {i+1}:\nQuestion: {ex['question']}\nAnswer: {ex['possible_answers'].split(' | ')[0]}"
                if isinstance(ex["possible_answers"], str)
                else f"Example {i+1}:\nQuestion: {ex['question']}\nAnswer: {ex['possible_answers'][0]}"
                for i, ex in enumerate(examples)
            ]
        )

        prompt = f"""Answer the following questions concisely and accurately, using no more than one sentence. Follow the format of the examples provided.

    {example_texts}

    Now, answer the following:
    Question: {question}
    Answer:"""
        prompts.append((idx, question, possible_answers, prompt))
        metadata.append((idx, question, possible_answers))

    # Step 2: Batch generate responses
    # Assuming generate_text_batch is a function that accepts a list of prompts and returns a list of responses
    model_answers = generate_text([p[3] for p in prompts])

    # Step 3: Save results to CSV
    for (idx, question, possible_answers, _), model_answer in zip(prompts, model_answers):
        writer.writerow({
            "idx": idx,
            "question": question,
            "possible_answers": possible_answers,
            "model_response": model_answer
        })

def generate_text_batch(prompts, max_length=10000):
    # Register hooks (if needed) for the batch generation
    activations, handles = register_activation_hooks(model)

    # Tokenize prompts as a batch with padding
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    # Set up generation configuration
    generation_config = {
        "max_length": max_length,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": True,
        "return_dict_in_generate": True,
        "output_scores": True
    }

    # Generate output sequences for the entire batch
    output_sequences = model.generate(**inputs, **generation_config)
    batch_generated_texts = []

    # Iterate over each prompt to remove its original input text from the output
    for i, prompt in enumerate(prompts):
        # Tokenize each prompt separately to get its actual length (without padding)
        prompt_tokens = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = prompt_tokens.input_ids.shape[1]

        generated_ids = output_sequences.sequences[i]
        # Decode the generated text, excluding the original prompt tokens
        generated_text = tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True)
        batch_generated_texts.append(generated_text)

    # Remove hooks after generation
    for handle in handles:
        handle.remove()

    return batch_generated_texts

# Function to generate text based on a prompt
def generate_text(prompt, max_length=10000):
    activations, handles = register_activation_hooks(model)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generation_config = {
        "max_length": max_length,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": True,
        "return_dict_in_generate": True,
        "output_scores": True
    }

    output_sequences = model.generate(**inputs, **generation_config)
    generated_ids = output_sequences.sequences[0]

    # Identify the input token length to remove the prompt from the output
    input_length = inputs.input_ids.shape[1]
    generated_text = tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True)

    for handle in handles:
        handle.remove()

    return generated_text


if __name__ == "__main__":
    #prompt = "Who actually said, 'Let them eat cake' ?"
    #generated_text = generate_text(prompt)
    #print(generated_text)
    #generate_responses_for_truthfulqa(model, tokenizer, device, output_csv="gemma2_truthfulqa_responses.csv", split="validation") 
    generate_responses_for_popqa(model, tokenizer, device, output_csv="gemma2b_popqa_responses.csv")
    wandb.finish()
