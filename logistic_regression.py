import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the MLP model with best practices (hidden layers, dropout)
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.5):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        # Build hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        # Final layer for binary classification (output is logit)
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def main(csv_file_path, pt_file_path, batch_size=32, epochs=50, lr=0.001):
    # 1. Load the CSV file and prepare labels
    df = pd.read_csv(csv_file_path)
    df['label'] = df['hallucinated'].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)
    print("Number of samples in CSV:", len(df))
    
    # 2. Load the activation tensor from the .pt file
    activations = torch.load(pt_file_path)
    print("Activation tensor shape:", activations.shape)
    
    # 3. Convert activations to NumPy array and extract labels
    X = activations.numpy()
    y = df['label'].values
    assert X.shape[0] == y.shape[0], "Mismatch between number of activation samples and labels!"
    
    # 4. Standardize features (zero mean, unit variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1)
    
    # 5. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    # Create TensorDatasets and DataLoaders for batch processing
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 6. Initialize the MLP model, loss function, and optimizer
    input_dim = X_tensor.shape[1]
    model = MLPClassifier(input_dim).to('cuda')
    criterion = nn.BCEWithLogitsLoss()  # more numerically stable than separate sigmoid + BCE
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 7. Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    # 8. Evaluation on test data
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
    
    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train an MLP classifier on activation data with hallucination labels."
    )
    parser.add_argument('--csv_file_path', type=str, required=True, help='Path to the CSV file.')
    parser.add_argument('--pt_file_path', type=str, required=True, help='Path to the .pt file.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    
    args = parser.parse_args()
    main(args.csv_file_path, args.pt_file_path, args.batch_size, args.epochs, args.lr)
