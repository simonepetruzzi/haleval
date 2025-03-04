import pandas as pd
import re

# Load the CSV file
file_path = 'gemma-2-2b-it_responses.csv'
df = pd.read_csv(file_path)

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
cleaned_file_path = 'gemma-2-2b-it_responses_cleaned.csv'
df.to_csv(cleaned_file_path, index=False)

print(f'Cleaned CSV file saved to {cleaned_file_path}')
