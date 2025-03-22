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
    Logs the metrics to wandb for every layer and returns layer names and corresponding metrics.
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
    
    for pt_file in pt_files:
        print(f"Processing file: {pt_file}")
        # Load the activations tensor and convert it to NumPy
        activations = torch.load(pt_file)
        X_np = activations.numpy()

        if X_np.ndim > 2:
            X_np = X_np.reshape(X_np.shape[0], -1)

        # Verify that the number of samples matches that of the original CSV
        if X_np.shape[0] != len(df):
            raise ValueError(f"Mismatch between number of samples in {pt_file} and CSV!")
        
        # Use the same indices to balance the activations
        X_balanced = X_np[balanced_indices]
        
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
        
    return layers, train_accuracies, test_accuracies, train_f1_scores, test_f1_scores

def main(csv_file_path, mlp_pattern, attn_pattern, hidden_pattern,
         batch_size=32, epochs=50, lr=0.001):
    model = extract_model_name_from_pattern(mlp_pattern)
    # 1) Run for MLP
    wandb.init(project="haleval", name=f"{model}_mlp")
    mlp_files = sorted(glob.glob(mlp_pattern))
    mlp_layers, mlp_train_acc, mlp_test_acc, mlp_train_f1, mlp_test_f1 = \
        evaluate_activation_files(csv_file_path, mlp_files, "MLP",
                                  batch_size=batch_size, epochs=epochs, lr=lr)
    # produce & log your MLP-only plots here if desired
    wandb.finish()
    
    # 2) Run for Attention
    wandb.init(project="haleval", name=f"{model}_attention")
    attn_files = sorted(glob.glob(attn_pattern))
    attn_layers, attn_train_acc, attn_test_acc, attn_train_f1, attn_test_f1 = \
        evaluate_activation_files(csv_file_path, attn_files, "Attention",
                                  batch_size=batch_size, epochs=epochs, lr=lr)
    # produce & log your Attention-only plots
    wandb.finish()
    
    # 3) Run for Hidden
    wandb.init(project="haleval", name=f"{model}_hidden")
    hidden_files = sorted(glob.glob(hidden_pattern))
    hidden_layers, hidden_train_acc, hidden_test_acc, hidden_train_f1, hidden_test_f1 = \
        evaluate_activation_files(csv_file_path, hidden_files, "Hidden",
                                  batch_size=batch_size, epochs=epochs, lr=lr)
    # produce & log your Hidden-only plots
    wandb.finish()

def extract_model_name_from_pattern(pattern):
    """
    Given a glob pattern like:
      analysis/outputs/2025-03-20/10-40-48/cached_data/gemma-3-27b-it/popQA/activation_hidden/conflict/layer*.pt
    This function finds the directory immediately after 'cached_data' and returns it,
    e.g. 'gemma-3-27b-it'.
    """
    # Remove any trailing wildcard (e.g. "layer*.pt") by splitting on '*'
    base_path = pattern.split('*')[0].rstrip('/')
    
    # Split into components
    parts = base_path.split('/')
    
    # Look for 'cached_data' in the path
    if 'cached_data' in parts:
        idx = parts.index('cached_data')
        # The model name is assumed to be the folder right after 'cached_data'
        if idx + 1 < len(parts):
            return parts[idx + 1]  # e.g. 'gemma-3-27b-it'
    
    # If not found, return a default or None
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train an MLP classifier on activation data from multiple layers and categories (MLP, Attention, Hidden) with hallucination labels, and plot test accuracy and F1 score."
    )
    parser.add_argument('--csv_file_path', type=str, required=True,
                        help='Path to the CSV file containing the questions and hallucinated labels.')
    parser.add_argument('--mlp_pattern', type=str, required=True,
                        help='Glob pattern for the MLP pt files (e.g., "./mlp/layer*.pt").')
    parser.add_argument('--attn_pattern', type=str, required=True,
                        help='Glob pattern for the Attention pt files (e.g., "./attention/layer*.pt").')
    parser.add_argument('--hidden_pattern', type=str, required=True,
                        help='Glob pattern for the Hidden Layer pt files (e.g., "./hidden/layer*.pt").')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    
    args = parser.parse_args()
    main(args.csv_file_path, args.mlp_pattern, args.attn_pattern, args.hidden_pattern,
         args.batch_size, args.epochs, args.lr)
