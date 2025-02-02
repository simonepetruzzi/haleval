import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import wandb
import csv
from datasets import load_dataset

from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

generation_config = {
    "max_new_tokens": 500,  # Ensure that new tokens are generated (instead of just truncating at max_length)
    "temperature": 0.7,     # Enable sampling
    "top_p": 0.9,          # Nucleus sampling
    "top_k": 50,           # Top-k sampling
    "do_sample": True,     # Ensure sampling is enabled
}

def compute_embeddings(prompt, tokenizer, model, device, original_prompt_len=None):
    with t.no_grad():
        # Tokenize the entire prompt for the model, take the last hidden states and sum them
        inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]  # shape: (batch_size, seq_len, hidden_dim)

        # If original_prompt_len is given, sum only over the original prompt portion
        if original_prompt_len is not None:
            seq_len = last_hidden_states.shape[1]
            truncated_len = min(original_prompt_len, seq_len)
            sentence_embedding = last_hidden_states[:, :truncated_len, :].sum(dim=1).squeeze()
        else:
            # Use full sentence embedding
            sentence_embedding = last_hidden_states.sum(dim=1).squeeze()

        return sentence_embedding.cpu()

def compute_embedding_trajectories(prompts, tokenizer, model, device, max_tokens_gen=500, generation_kwargs={}):
    all_embeddings = []
    all_generated_texts = []
    all_distances_llama = []
    all_word_counts = []
    all_partial_embeddings = []  # To store the partial embeddings at each time step

    for prompt in prompts:
        # Compute number of tokens in the original prompt
        original_input = tokenizer(prompt, return_tensors='pt', padding=True)
        original_prompt_len = original_input.input_ids.shape[1]

        # Initialize lists for this prompt's trajectory
        embeddings = []
        distances_llama = []
        partial_embeddings = []
        word_counts = [0]

        # Compute initial full embedding
        full_sentence_embedding = compute_embeddings(prompt, tokenizer, model, device)
        embeddings.append(full_sentence_embedding)

        # Compute initial partial embedding (just the original prompt portion)
        initial_partial_embedding = compute_embeddings(prompt, tokenizer, model, device, original_prompt_len=original_prompt_len)
        partial_embeddings.append(initial_partial_embedding)

        # Distances from initial full embedding
        distances_llama.append(0)

        # Initial input for generation
        inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)

        for i in range(max_tokens_gen):
            if i % 100 == 0:
                print(i)

            with t.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    pad_token_id=tokenizer.eos_token_id,
                    **generation_kwargs
                )

            if generated_ids[0][-1] == tokenizer.eos_token_id:
                 # Stop if EOS token is generated
                break

             # Decode generated textates tok
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            generated_text = generated_text.replace("<|begin_of_text|>", "")

            # Update inputs with the newly generated text
            new_token_id = generated_ids[:, -1].unsqueeze(-1)  # Ensure correct shape
            inputs['input_ids'] = t.cat([inputs['input_ids'], new_token_id], dim=-1)         
            inputs['attention_mask'] = t.cat([inputs['attention_mask'], t.ones_like(new_token_id)], dim=-1)  # Update attention mask


            # Compute embeddings for the newly generated sequence (full)
            sentence_embedding = compute_embeddings(generated_text, tokenizer, model, device)
            embeddings.append(sentence_embedding)

            # Compute partial embedding (original prompt portion only)
            partial_embedding = compute_embeddings(generated_text, tokenizer, model, device, original_prompt_len=original_prompt_len)
            partial_embeddings.append(partial_embedding)

            # Distances from initial embedding (full)
            distance_llama = cosine_distances([embeddings[0].numpy()], [sentence_embedding.numpy()])[0][0]
            distances_llama.append(distance_llama)

            # Count words and store count difference
            word_count = len(generated_text.split()) - len(prompt.split())
            word_counts.append(word_count)

        # Store results for this prompt
        all_embeddings.append(embeddings)
        all_distances_llama.append(distances_llama)
        all_word_counts.append(word_counts)
        all_partial_embeddings.append(partial_embeddings)

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        print(generated_text)
        all_generated_texts.append(generated_text)

    return {
        "embeddings": all_embeddings,
        "distances_llama": all_distances_llama,
        "word_counts": all_word_counts,
        "partial_embeddings": all_partial_embeddings,
        "generated_texts": all_generated_texts
    }

prompts = [
    "Who actually said, 'Let them eat cake' ?",
    "Who discovered penicillin?", 
]

trajectories = compute_embedding_trajectories(prompts, tokenizer, model, device, max_tokens_gen=500, generation_kwargs=generation_config)

# Access trajectories for further processing or plotting
embeddings_trajectories = trajectories["embeddings"]
distances_llama_trajectories = trajectories["distances_llama"]
word_counts_trajectories = trajectories["word_counts"]
generated_texts = trajectories["generated_texts"]


# Assuming `trajectories_data` contains the embeddings trajectories for all prompts,
# as generated from `compute_embedding_trajectories` function
# Each trajectory in `trajectories_data["embeddings"]` is a list of embeddings over time

# Convert each prompt’s embeddings list to numpy arrays for easier manipulation
all_embeddings = [np.array([emb.cpu().numpy() for emb in trajectory]) for trajectory in trajectories["embeddings"]]

# Stack all trajectories into one combined matrix for PCA (shape: (P * T, D))
# where P is the number of prompts, T is the number of timesteps, D is the embedding dimension
combined_trajectories = np.vstack(all_embeddings)

# Apply PCA on the combined trajectory data
pca = PCA()
pca.fit(combined_trajectories)

# Get the explained variance ratios for each component
explained_variance_ratios = pca.explained_variance_ratio_

# Determine the number of components needed to reach the desired explained variance (e.g., 95%)
cumulative_variance = np.cumsum(explained_variance_ratios)
#n_components = np.argmax(cumulative_variance >= 0.95) + 1  # For 95% variance
n_components = max(2, np.argmax(cumulative_variance >= 0.95) + 1)
#n_components = 100
print(f"Number of effective dimensions for 95% variance: {n_components}")

# Plot 2: Evolution of the trajectories in 2D using the first two principal components
# Transform all embeddings to the PCA space (using first two components for 2D visualization)
trajectories_pca = pca.transform(combined_trajectories)

# Separate and plot each prompt's trajectory in 2D
plt.figure(figsize=(10, 6))
for i, trajectory in enumerate(all_embeddings):
    
    start_idx = i * len(trajectory)  # Starting index for the i-th trajectory in combined array
    end_idx = start_idx + len(trajectory)

    trajectory_2d = trajectories_pca[start_idx:end_idx, :n_components]  # First two components for 2D
    print(f"Trajectory {i+1} shape: {trajectory_2d.shape}")
    plt.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], label=f'Trajectory {i+1}', marker='o', alpha=0.6)

# Adding labels and title
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA Projection of Trajectories (First Two Principal Components)")
plt.legend()
plt.show()
plt.savefig("images/embedding_trajectories", dpi=300, bbox_inches="tight")
