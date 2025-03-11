import argparse
import glob
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def evaluate_activation_files(csv_file_path, pt_files):
    # Load the CSV file and convert hallucination labels to binary
    df = pd.read_csv(csv_file_path)
    df['label'] = df['hallucinated'].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)
    y = df['label'].values
    
    layers = []
    test_accuracies = []
    train_accuracies = []
    
    for pt_file in pt_files:
        print(f"Processing file: {pt_file}")
        # Load activation tensor
        activations = torch.load(pt_file)
        X = activations.numpy()
        
        # Sanity check: number of samples must match
        if X.shape[0] != len(y):
            raise ValueError(f"Mismatch between number of samples in {pt_file} and CSV!")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train logistic regression classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Use the file name as the layer identifier
        layer_name = os.path.basename(pt_file)
        layers.append(layer_name)
        
        print(f"{layer_name} -> Train acc: {train_acc * 100:.2f}%, Test acc: {test_acc * 100:.2f}%")
    
    return layers, train_accuracies, test_accuracies

def main(csv_file_path, mlp_pattern, attn_pattern, hidden_pattern):
    # Get file lists for each category
    mlp_files = sorted(glob.glob(mlp_pattern))
    attn_files = sorted(glob.glob(attn_pattern))
    hidden_files = sorted(glob.glob(hidden_pattern))
    
    if not mlp_files:
        raise ValueError("No MLP pt files found!")
    if not attn_files:
        raise ValueError("No attention pt files found!")
    if not hidden_files:
        raise ValueError("No hidden layer pt files found!")
    
    # Evaluate each category of activations
    mlp_layers, mlp_train_acc, mlp_test_acc = evaluate_activation_files(csv_file_path, mlp_files)
    attn_layers, attn_train_acc, attn_test_acc = evaluate_activation_files(csv_file_path, attn_files)
    hidden_layers, hidden_train_acc, hidden_test_acc = evaluate_activation_files(csv_file_path, hidden_files)
    
    # For plotting, we assume that the pt files in each category are ordered by layer.
    # We use the index (layer number) for the x-axis.
    plt.figure(figsize=(12, 8))
    
    # Plot Test Accuracy lines for each category
    x_mlp = range(len(mlp_layers))
    x_attn = range(len(attn_layers))
    x_hidden = range(len(hidden_layers))
    
    plt.plot(x_mlp, [acc * 100 for acc in mlp_test_acc], marker='o', label='MLP - Test Accuracy')
    plt.plot(x_attn, [acc * 100 for acc in attn_test_acc], marker='o', label='Attention - Test Accuracy')
    plt.plot(x_hidden, [acc * 100 for acc in hidden_test_acc], marker='o', label='Hidden Layer - Test Accuracy')
    
    plt.xlabel('Layer index')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Logistic Regression Test Accuracy Across Layers and Activation Categories')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train logistic regression on activation data from multiple layers and categories with hallucination labels."
    )
    parser.add_argument(
        '--csv_file_path', type=str, required=True,
        help='Path to the CSV file containing the questions and hallucinated labels.'
    )
    parser.add_argument(
        '--mlp_pattern', type=str, required=True,
        help='Glob pattern for the MLP pt files (e.g., "./mlp/layer*.pt").'
    )
    parser.add_argument(
        '--attn_pattern', type=str, required=True,
        help='Glob pattern for the Attention pt files (e.g., "./attention/layer*.pt").'
    )
    parser.add_argument(
        '--hidden_pattern', type=str, required=True,
        help='Glob pattern for the Hidden Layer pt files (e.g., "./hidden/layer*.pt").'
    )
    
    args = parser.parse_args()
    main(args.csv_file_path, args.mlp_pattern, args.attn_pattern, args.hidden_pattern)
