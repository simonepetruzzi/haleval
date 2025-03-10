import string
import csv
import re
import difflib
import ast
import argparse

def normalize_text(s: str) -> str:
    # Strip text
    s = s.strip()
    
    # Lowercase text
    s = s.lower()
    
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', '', s)
    
    # Fix extra whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s

def extract_entities(response: str) -> list[str]:
    """
    Extracts a list of entities from the model response.
    
    The function first tries to see if the response is a Python list (e.g. "['entity1', 'entity2']").
    If not, it looks for a leading "Entities:" marker and uses the text after that,
    otherwise it assumes the entire response is a comma-separated list.
    """
    response = response.strip()
    
    # Try to parse a Python list
    if response.startswith("[") and response.endswith("]"):
        try:
            entities = ast.literal_eval(response)
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if str(e).strip()]
        except Exception:
            pass

    # Look for a marker "Entities:" and use the text after it.
    pattern = re.compile(r'Entities:\s*(.*)', re.IGNORECASE)
    match = pattern.search(response)
    if match:
        entities_text = match.group(1)
    else:
        entities_text = response

    # Split by commas (you can adjust the delimiter if needed)
    entities = [ent.strip() for ent in entities_text.split(",") if ent.strip()]
    return entities

def evaluate_csv(
    input_file_path: str, 
    output_file_path: str, 
    append_mode=False
):
    """
    Reads a CSV file (input_file_path) with columns: 
       [question, possible_answer, model_response].
    - `possible_answer` may contain multiple entities separated by the "|" character.
    Extracts entities from the model_response, normalizes them along with the possible_answer entities,
    and then checks for an exact match between any pair.
    Writes the results to an output CSV file (output_file_path) with columns:
       [question, possible_answer, model_entities, hallucinated].
    """
    
    mode = 'a' if append_mode else 'w'
    
    with open(input_file_path, mode='r', encoding='utf-8') as infile, \
         open(output_file_path, mode=mode, newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = ['question', 'possible_answer', 'model_entities', 'hallucinated']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        # Write header if not appending
        if not append_mode:
            writer.writeheader()

        for row in reader:
            question = row.get("question", "").strip()
            raw_model_response = row.get("model_response", "").strip()
            # Split the possible answers on "|" to get a list
            possible_answers_list = row.get("possible_answer", "").strip().split("|")
            
            # Extract entities from the model response
            extracted_entities = extract_entities(raw_model_response)
            
            # Normalize the entities and the possible answers
            extracted_entities = [normalize_text(entity) for entity in extracted_entities]
            possible_answers_list = [normalize_text(ans) for ans in possible_answers_list]
            
            # Check for an exact match between any extracted entity and any possible answer
            match_found = any(entity == possible for entity in extracted_entities for possible in possible_answers_list)
            
            hallucinated = "false" if match_found else "true"

            writer.writerow({
                'question': question,
                'possible_answer': "|".join(possible_answers_list),
                'model_entities': "|".join(extracted_entities),
                'hallucinated': hallucinated
            })
            
    print(f"Evaluation complete. Results saved to {output_file_path}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Evaluate a CSV file and save the output.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_file", help="Path to save the evaluated CSV file")
    parser.add_argument("--append", action="store_true", help="Enable append mode (default is overwrite)")

    args = parser.parse_args()

    # Call the evaluation function with dynamic file paths
    evaluate_csv(args.input_file, args.output_file, append_mode=args.append)
