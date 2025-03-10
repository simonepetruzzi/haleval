import pandas as pd
import re
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Clean bracketed responses in a CSV file.")
parser.add_argument("input_file", help="Path to the input CSV file")
parser.add_argument("output_file", help="Path to save the cleaned CSV file")
args = parser.parse_args()

# Load the CSV file
df = pd.read_csv(args.input_file)

# Define a function to extract text inside brackets and ensure closing bracket
def clean_response(response):
    matches = re.findall(r'\[.*?\]', response)  # Find all bracketed texts
    if matches:
        return ' '.join(matches)  # Keep only bracketed content
    elif '[' in response:  # If an opening bracket is present but no closing
        return response + ']'  # Add closing bracket
    return ''  # If no brackets are found, return empty string

# Apply the cleaning function to the model's response column
column_name = 'model_response'  # Adjust if needed
df[column_name] = df[column_name].astype(str).apply(clean_response)

# Save the cleaned data back to a CSV file
df.to_csv(args.output_file, index=False)

print(f'Cleaned CSV file saved to {args.output_file}')
