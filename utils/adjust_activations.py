import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description="Modify activation tensor dimensionality")
    parser.add_argument('--input_file', type=str, required=True,
                        help="Path to the input activation file (.pt)")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Path where the modified activation file will be saved")
    parser.add_argument('--method', type=str, choices=['first', 'mean'], default='first',
                        help="Method to reduce tensor shape: 'first' to select the first row, 'mean' to compute the mean across rows")
    
    args = parser.parse_args()

    # Load the activation tensor from the file
    tensor = torch.load(args.input_file)
    print(f"Original tensor shape: {tensor.shape}")

    # Modify the tensor based on the selected method
    if args.method == 'first':
        # Select the first row and keep dimensions as [1, 3584]
        tensor_modified = tensor[0:1, :]
    elif args.method == 'mean':
        # Compute the mean across rows while preserving the dimensions
        tensor_modified = tensor.mean(dim=0, keepdim=True)
    
    print(f"Modified tensor shape: {tensor_modified.shape}")

    # Save the modified tensor to the output file
    torch.save(tensor_modified, args.output_file)
    print(f"Modified tensor saved to {args.output_file}")

if __name__ == "__main__":
    main()
