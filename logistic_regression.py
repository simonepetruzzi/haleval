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

# Define the MLP classifier with best practices (hidden layers, ReLU, dropout)
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.5):
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

def train_mlp_model(X_train, y_train, X_test, y_test, input_dim, batch_size=32, epochs=50, lr=0.001):
    """
    Trains an MLP model on the provided training data and returns training and test accuracy.
    """
    # Create DataLoader objects for batching
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize the model, loss function, and optimizer
    model = MLPClassifier(input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        # Uncomment the following line to print per-epoch loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    # Evaluate on training data
    model.eval()
    with torch.no_grad():
        train_correct = 0
        train_total = 0
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (predictions == y_batch).sum().item()
            train_total += y_batch.size(0)
        train_accuracy = train_correct / train_total
        
        # Evaluate on test data
        test_correct = 0
        test_total = 0
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            test_correct += (predictions == y_batch).sum().item()
            test_total += y_batch.size(0)
        test_accuracy = test_correct / test_total
    
    return train_accuracy, test_accuracy

def evaluate_activation_files(csv_file_path, pt_files, batch_size=32, epochs=50, lr=0.001):
    """
    For each activation file in pt_files:
      - Loads activations and labels,
      - Standardizes the features,
      - Splits data into train and test sets,
      - Trains an MLP classifier,
      - Records training and test accuracy.
    Returns layer names and corresponding accuracies.
    """
    # Load CSV and convert hallucination labels to binary (1 for 'true', 0 otherwise)
    df = pd.read_csv(csv_file_path)
    df['label'] = df['hallucinated'].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)
    y_np = df['label'].values
    
    layers = []
    train_accuracies = []
    test_accuracies = []
    
    for pt_file in pt_files:
        print(f"Processing file: {pt_file}")
        # Load activation tensor and convert to NumPy
        activations = torch.load(pt_file)
        X_np = activations.numpy()

        if X_np.ndim > 2:
            X_np = X_np.reshape(X_np.shape[0], -1)

        # Check that the number of samples matches
        if X_np.shape[0] != len(y_np):
            raise ValueError(f"Mismatch between number of samples in {pt_file} and CSV!")
        
        # Standardize features (zero mean, unit variance)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_np)
        
        # Split data into training and testing sets
        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_scaled, y_np, test_size=0.2, random_state=42)
        
        # Convert arrays to PyTorch tensors
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)
        
        input_dim = X_train.shape[1]
        # Train the MLP model on the current activations
        train_acc, test_acc = train_mlp_model(X_train, y_train, X_test, y_test, input_dim,
                                                batch_size=batch_size, epochs=epochs, lr=lr)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        layer_name = os.path.basename(pt_file)
        layers.append(layer_name)
        print(f"{layer_name} -> Train acc: {train_acc * 100:.2f}%, Test acc: {test_acc * 100:.2f}%")
        
    return layers, train_accuracies, test_accuracies

def main(csv_file_path, mlp_pattern, attn_pattern, hidden_pattern, batch_size=32, epochs=50, lr=0.001):
    # Get file lists for each category using glob patterns
    mlp_files = sorted(glob.glob(mlp_pattern))
    attn_files = sorted(glob.glob(attn_pattern))
    hidden_files = sorted(glob.glob(hidden_pattern))
    
    if not mlp_files:
        raise ValueError("No MLP pt files found!")
    if not attn_files:
        raise ValueError("No Attention pt files found!")
    if not hidden_files:
        raise ValueError("No Hidden pt files found!")
    
    # Evaluate activations for each category using the MLP classifier
    mlp_layers, mlp_train_acc, mlp_test_acc = evaluate_activation_files(csv_file_path, mlp_files, batch_size, epochs, lr)
    attn_layers, attn_train_acc, attn_test_acc = evaluate_activation_files(csv_file_path, attn_files, batch_size, epochs, lr)
    hidden_layers, hidden_train_acc, hidden_test_acc = evaluate_activation_files(csv_file_path, hidden_files, batch_size, epochs, lr)
    
    # Plot Test Accuracy across layers for each activation category
    plt.figure(figsize=(12, 8))
    
    # For plotting, we assume that files in each category are ordered by layer index.
    x_mlp = range(len(mlp_layers))
    x_attn = range(len(attn_layers))
    x_hidden = range(len(hidden_layers))
    
    plt.plot(x_mlp, [acc * 100 for acc in mlp_test_acc], marker='o', label='MLP - Test Accuracy')
    plt.plot(x_attn, [acc * 100 for acc in attn_test_acc], marker='o', label='Attention - Test Accuracy')
    plt.plot(x_hidden, [acc * 100 for acc in hidden_test_acc], marker='o', label='Hidden Layer - Test Accuracy')
    
    plt.xlabel('Layer Index')
    plt.ylabel('Test Accuracy (%)')
    plt.title('MLP Test Accuracy Across Layers and Activation Categories')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train an MLP classifier on activation data from multiple layers and categories (MLP, Attention, Hidden) with hallucination labels, and plot test accuracy."
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
