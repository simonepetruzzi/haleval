import argparse
import glob
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_and_evaluate(csv_file_path, pt_files):
    # 1. Load the CSV file containing the questions and the hallucination labels
    df = pd.read_csv(csv_file_path)
    # Convert the "hallucinated" column to binary labels: 1 for "True", 0 for any other value
    df['label'] = df['hallucinated'].apply(lambda x: 1 if str(x).strip().lower() == 'true' else 0)
    print("Number of samples in CSV:", len(df))
    
    # Extract the label vector from the dataframe
    y = df['label'].values
    
    # Prepare lists to store results
    train_accuracies = []
    test_accuracies = []
    layer_names = []

    # 2. Loop over each pt file
    for pt_file in pt_files:
        print(f"\nProcessing file: {pt_file}")
        # Load the activation tensor
        activations = torch.load(pt_file)
        print("Activation tensor shape:", activations.shape)
        
        # Convert the activation tensor to a NumPy array
        X = activations.numpy()
        
        # Sanity check: Ensure the number of samples in X and y match
        if X.shape[0] != len(y):
            raise ValueError(f"Mismatch between number of activation samples and labels for {pt_file}!")
        
        # 3. Preprocess: Standardize the features to have zero mean and unit variance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 4. Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # 5. Train a logistic regression classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        
        # Evaluate model performance on training and testing sets
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Use the file name (or part of it) as the layer identifier
        layer_name = os.path.basename(pt_file)
        layer_names.append(layer_name)
        
        print(f"Training accuracy for {layer_name}: {train_acc * 100:.2f}%")
        print(f"Test accuracy for {layer_name}: {test_acc * 100:.2f}%")
    
    # 6. Plot the evaluation results
    plt.figure(figsize=(10, 6))
    plt.plot(layer_names, [acc * 100 for acc in test_accuracies], marker='o', label='Test Accuracy')
    plt.plot(layer_names, [acc * 100 for acc in train_accuracies], marker='o', label='Train Accuracy')
    plt.xlabel('Layer (pt file)')
    plt.ylabel('Accuracy (%)')
    plt.title('Logistic Regression Accuracy Across Different Layers')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train logistic regression on activation data from multiple layers with hallucination labels."
    )
    parser.add_argument(
        '--csv_file_path', type=str, required=True,
        help='Path to the CSV file containing the questions and hallucinated labels.'
    )
    parser.add_argument(
        '--pt_files_pattern', type=str, required=True,
        help='Glob pattern for the .pt files (e.g., "./activations/layer*.pt").'
    )
    
    args = parser.parse_args()
    
    # Retrieve the list of .pt files using the provided glob pattern
    pt_files = sorted(glob.glob(args.pt_files_pattern))
    if not pt_files:
        raise ValueError("No .pt files found matching the given pattern!")
    
    train_and_evaluate(args.csv_file_path, pt_files)
