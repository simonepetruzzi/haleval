import argparse
import glob
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score  # Import f1_score
import numpy as np
import wandb  # Import wandb

# Define the MLP classifier with best practices (hidden layers, ReLU, dropout)
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32, 16, 8], dropout_rate=0.5):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        # Final layer outputs a single logit for binary classification
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_mlp_model(X_train, y_train, X_test, y_test, input_dim, batch_size=64, epochs=50, lr=0.001):
    """
    Trains an MLP model and returns training and test accuracy along with their F1 scores.
    """
    # Create DataLoader objects for batching
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize the model, loss function, and optimizer
    model = MLPClassifier(input_dim).to('cuda')
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to('cuda'), y_batch.to('cuda')
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Evaluate on training data
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to('cuda'), y_batch.to('cuda')
            outputs = model(X_batch)
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (predictions == y_batch).sum().item()
            train_total += y_batch.size(0)
            train_preds.append(predictions.cpu())
            train_labels.append(y_batch.cpu())
        train_accuracy = train_correct / train_total
        train_preds = torch.cat(train_preds).numpy()
        train_labels = torch.cat(train_labels).numpy()
        train_f1 = f1_score(train_labels, train_preds)
        
        # Evaluate on test data
        test_correct = 0
        test_total = 0
        test_preds = []
        test_labels = []
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to('cuda'), y_batch.to('cuda')
            outputs = model(X_batch)
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            test_correct += (predictions == y_batch).sum().item()
            test_total += y_batch.size(0)
            test_preds.append(predictions.cpu())
            test_labels.append(y_batch.cpu())
        test_accuracy = test_correct / test_total
        test_preds = torch.cat(test_preds).numpy()
        test_labels = torch.cat(test_labels).numpy()
        test_f1 = f1_score(test_labels, test_preds)
    
    return train_accuracy, test_accuracy, train_f1, test_f1

def evaluate_activation_files(csv_file_path, pt_files, category, batch_size=64, epochs=50, lr=0.001):
    """
    For each activation file:
      - Loads activations and labels from the CSV,
      - Balances the dataset via undersampling,
      - Standardizes features,
      - Splits data into train and test sets,
      - Trains an MLP classifier,
      - Records training/test accuracy and F1 scores.
    Additionally, extracts the activation vectors corresponding to hallucinated (true) samples
    after the dataset has been balanced.
    Logs the metrics to wandb for every layer and returns layer names, metrics, and the true activations.
    """
    # Load CSV and convert the "hallucinated" column to binary labels (1 for 'true', 0 for 'false')
    df = pd.read_csv(csv_file_path)
    df['label'] = df['hallucinated'].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)
    
    # Indices for the two classes
    true_indices = df.index[df['label'] == 1].tolist()
    false_indices = df.index[df['label'] == 0].tolist()
    
    # Compute the minimum number of examples between the two classes
    min_count = min(len(true_indices), len(false_indices))
    
    # Set seed for reproducibility
    np.random.seed(42)
    true_indices_sampled = np.random.choice(true_indices, min_count, replace=False)
    false_indices_sampled = np.random.choice(false_indices, min_count, replace=False)
    
    # Combine and sort the indices to maintain the original order
    balanced_indices = sorted(np.concatenate((true_indices_sampled, false_indices_sampled)))
    
    # Filter the balanced DataFrame and reset the index
    df_balanced = df.iloc[balanced_indices].reset_index(drop=True)
    # Extract labels from the balanced DataFrame
    y_np = df_balanced['label'].values
    
    layers = []
    train_accuracies = []
    test_accuracies = []
    train_f1_scores = []
    test_f1_scores = []
    # List to hold true activations for this category (as 2D tensors)
    true_activations_all = []
    
    for pt_file in pt_files:
        print(f"Processing file: {pt_file}")
        # Load the activations tensor and convert it to NumPy
        activations = torch.load(pt_file)
        X_np = activations.numpy()
        # If the activation has more than 2 dimensions, reshape it so that each sample is a flat vector.
        if X_np.ndim > 2:
            X_np = X_np.reshape(X_np.shape[0], -1)
        
        expected_dim = 4096
        if X_np.shape[1] != expected_dim:
            raise ValueError(f"Expected activation dimension {expected_dim}, but got {X_np.shape[1]} in file {pt_file}!")
        
        # Verify that the number of samples matches that of the original CSV
        if X_np.shape[0] != len(df):
            raise ValueError(f"Mismatch between number of samples in {pt_file} and CSV!")
        
        # Use the same indices to balance the activations
        X_balanced = X_np[balanced_indices]
        
        # --- NEW CODE: Extract and store true and false activations AFTER balancing ---
        balanced_true_mask = (df_balanced['label'].values == 1)
        file_true_activations = X_balanced[balanced_true_mask]
        file_false_activations = X_balanced[~balanced_true_mask]
        file_true_tensor = torch.tensor(file_true_activations, dtype=torch.float32)
        file_false_tensor = torch.tensor(file_false_activations, dtype=torch.float32)
        print(f"File {os.path.basename(pt_file)} true activations shape: {file_true_tensor.shape}")
        print(f"File {os.path.basename(pt_file)} false activations shape: {file_false_tensor.shape}")
        true_activations_all.append(file_true_tensor)
        # --- END NEW CODE ---
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_balanced)
        
        # Split the balanced dataset into training and testing sets
        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
            X_scaled, y_np, test_size=0.2, random_state=42, stratify=y_np
        )
        
        # Convert the arrays into PyTorch tensors
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)
        
        input_dim = X_train.shape[1]
        # Train the MLP on the balanced activations
        train_acc, test_acc, train_f1, test_f1 = train_mlp_model(
            X_train, y_train, X_test, y_test, input_dim,
            batch_size=batch_size, epochs=epochs, lr=lr
        )
        
        layers.append(os.path.basename(pt_file))
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_f1_scores.append(train_f1)
        test_f1_scores.append(test_f1)
        print(f"{layers[-1]} -> Train acc: {train_acc * 100:.2f}%, Test acc: {test_acc * 100:.2f}% | Train F1: {train_f1:.2f}, Test F1: {test_f1:.2f}")
        
        # Log the metrics for this layer to wandb
        wandb.log({
            "category": category,
            "layer": os.path.basename(pt_file),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_f1": train_f1,
            "test_f1": test_f1,
        })
        
    # Concatenate all true activations from the different files along axis 0.
    if true_activations_all:
        true_activations_all = torch.cat(true_activations_all, dim=0)
        print("Final true activations shape:", true_activations_all.shape)
    else:
        true_activations_all = torch.empty((0, expected_dim))
    
    return layers, train_accuracies, test_accuracies, train_f1_scores, test_f1_scores, true_activations_all


def process_single_activation_file(csv_file_path, pt_file, expected_dim=4096):
    """
    Loads the activations from a single file (expected to be from layer 13),
    applies dataset balancing (undersampling) based on the CSV labels,
    and extracts both hallucinated (true) and non-hallucinated (false) activations.
    
    Returns a tuple of two tensors:
      - true_activations_tensor of shape [number_of_true_activations_after_balancing, expected_dim],
      - false_activations_tensor of shape [number_of_false_activations_after_balancing, expected_dim].
    """
    # Load CSV and create binary label (1 for hallucinated true, 0 for false)
    df = pd.read_csv(csv_file_path)
    df['label'] = df['hallucinated'].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)
    
    # Get indices for true and false samples; then undersample to balance the dataset.
    true_indices = df.index[df['label'] == 1].tolist()
    false_indices = df.index[df['label'] == 0].tolist()
    min_count = min(len(true_indices), len(false_indices))
    np.random.seed(42)
    true_indices_sampled = np.random.choice(true_indices, min_count, replace=False)
    false_indices_sampled = np.random.choice(false_indices, min_count, replace=False)
    balanced_indices = sorted(np.concatenate((true_indices_sampled, false_indices_sampled)))
    
    # Filter the balanced DataFrame
    df_balanced = df.iloc[balanced_indices].reset_index(drop=True)
    
    # Load the activations from the pt file.
    activations = torch.load(pt_file)
    X_np = activations.numpy()
    if X_np.ndim > 2:
        X_np = X_np.reshape(X_np.shape[0], -1)
    
    # Check dimensions.
    if X_np.shape[1] != expected_dim:
        raise ValueError(f"Expected activation dimension {expected_dim}, but got {X_np.shape[1]} in {pt_file}!")
    if X_np.shape[0] != len(df):
        raise ValueError(f"Mismatch between number of samples in {pt_file} and CSV!")
    
    # Apply the same balanced indices.
    X_balanced = X_np[balanced_indices]
    
    # Create masks for true and false samples in the balanced set.
    balanced_true_mask = (df_balanced['label'].values == 1)
    true_activations = X_balanced[balanced_true_mask]
    false_activations = X_balanced[~balanced_true_mask]
    
    # Convert to tensors.
    true_activations_tensor = torch.tensor(true_activations, dtype=torch.float32)
    false_activations_tensor = torch.tensor(false_activations, dtype=torch.float32)
    print(f"Processed file {os.path.basename(pt_file)}: true activations shape = {true_activations_tensor.shape}, false activations shape = {false_activations_tensor.shape}")
    return true_activations_tensor, false_activations_tensor

def extract_model_name_from_pattern(pattern):
    """
    Extracts the model name from a glob pattern.
    For example, if the pattern contains:
      .../cached_data/gemma-3-27b-it/...
    returns 'gemma-3-27b-it'.
    """
    base_path = pattern.split('*')[0].rstrip('/')
    parts = base_path.split('/')
    if 'cached_data' in parts:
        idx = parts.index('cached_data')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None

def main(csv_file_path, mlp_pattern, attn_pattern, hidden_pattern):
    # For each category, we assume the glob pattern returns at least one file.
    # And we process the first file that includes "layer13" in its name.
    model = extract_model_name_from_pattern(mlp_pattern)

    # Process MLP
    mlp_files = sorted(glob.glob(mlp_pattern))
    mlp_file = next((f for f in mlp_files if "layer13" in os.path.basename(f)), None)
    if mlp_file is None:
        raise ValueError("No layer 13 file found for MLP.")
    wandb.init(project="haleval", name=f"{model}_mlp")
    mlp_true_acts, mlp_false_acts = process_single_activation_file(csv_file_path, mlp_file, expected_dim=4096)
    wandb.finish()
    mlp_true_output_file = "true_activations_layer13_mlp.pt"
    mlp_false_output_file = "false_activations_layer13_mlp.pt"
    torch.save(mlp_true_acts, mlp_true_output_file)
    torch.save(mlp_false_acts, mlp_false_output_file)
    print(f"Saved MLP hallucinated activations: {mlp_true_acts.shape} to '{mlp_true_output_file}'")
    print(f"Saved MLP false activations: {mlp_false_acts.shape} to '{mlp_false_output_file}'")
    
    # Process Attention
    attn_files = sorted(glob.glob(attn_pattern))
    attn_file = next((f for f in attn_files if "layer13" in os.path.basename(f)), None)
    if attn_file is None:
        raise ValueError("No layer 13 file found for Attention.")
    wandb.init(project="haleval", name=f"{model}_attention")
    attn_true_acts, attn_false_acts = process_single_activation_file(csv_file_path, attn_file, expected_dim=4096)
    wandb.finish()
    attn_true_output_file = "true_activations_layer13_attention.pt"
    attn_false_output_file = "false_activations_layer13_attention.pt"
    torch.save(attn_true_acts, attn_true_output_file)
    torch.save(attn_false_acts, attn_false_output_file)
    print(f"Saved Attention hallucinated activations: {attn_true_acts.shape} to '{attn_true_output_file}'")
    print(f"Saved Attention false activations: {attn_false_acts.shape} to '{attn_false_output_file}'")
    
    # Process Hidden
    hidden_files = sorted(glob.glob(hidden_pattern))
    hidden_file = next((f for f in hidden_files if "layer13" in os.path.basename(f)), None)
    if hidden_file is None:
        raise ValueError("No layer 13 file found for Hidden.")
    wandb.init(project="haleval", name=f"{model}_hidden")
    hidden_true_acts, hidden_false_acts = process_single_activation_file(csv_file_path, hidden_file, expected_dim=4096)
    wandb.finish()
    hidden_true_output_file = "true_activations_layer13_hidden.pt"
    hidden_false_output_file = "false_activations_layer13_hidden.pt"
    torch.save(hidden_true_acts, hidden_true_output_file)
    torch.save(hidden_false_acts, hidden_false_output_file)
    print(f"Saved Hidden hallucinated activations: {hidden_true_acts.shape} to '{hidden_true_output_file}'")
    print(f"Saved Hidden false activations: {hidden_false_acts.shape} to '{hidden_false_output_file}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract and save hallucinated layer 13 activations for MLP, Attention, and Hidden categories."
    )
    parser.add_argument('--csv_file_path', type=str, required=True,
                        help='Path to the CSV file with queries and hallucinated labels.')
    parser.add_argument('--mlp_pattern', type=str, required=False,
                        help='Glob pattern for MLP pt files (e.g., "./mlp/layer*.pt").')
    parser.add_argument('--attn_pattern', type=str, required=False,
                        help='Glob pattern for Attention pt files (e.g., "./attention/layer*.pt").')
    parser.add_argument('--hidden_pattern', type=str, required=False,
                        help='Glob pattern for Hidden pt files (e.g., "./hidden/layer*.pt").')
    
    args = parser.parse_args()
    main(args.csv_file_path, args.mlp_pattern, args.attn_pattern, args.hidden_pattern)
