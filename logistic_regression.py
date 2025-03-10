import argparse
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def main(csv_file_path, pt_file_path):
    # 1. Load the CSV file containing the questions and the hallucination labels
    df = pd.read_csv(csv_file_path)
    
    # Convert the "hallucinated" column to binary labels: 1 for "Yes" and 0 for "No"
    df['label'] = df['hallucinated'].apply(lambda x: 1 if x.strip().lower() == 'yes' else 0)
    print("Number of samples in CSV:", len(df))
    
    # 2. Load the activation tensor from the .pt file for a specific layer
    activations = torch.load(pt_file_path)  
    print("Activation tensor shape:", activations.shape)
    
    # 3. Convert the activation tensor to a NumPy array for scikit-learn
    X = activations.numpy()  
    
    # 4. Extract the label vector from the dataframe
    y = df['label'].values  
    
    # Sanity check: Ensure the number of samples in X and y match
    assert X.shape[0] == y.shape[0], "Mismatch between number of activation samples and labels!"
    
    # 5. Preprocess: Standardize the features to have zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 6. Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 7. Train a logistic regression classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate the model performance on training and testing sets
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train logistic regression on activation data with hallucination labels.")
    parser.add_argument('--csv_file_path', type=str, required=True, help='Path to the CSV file containing the questions and hallucinated labels.')
    parser.add_argument('--pt_file_path', type=str, required=True, help='Path to the .pt file containing the activation tensor.')
    
    args = parser.parse_args()
    
    main(args.csv_file_path, args.pt_file_path)
