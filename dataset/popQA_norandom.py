import datasets
from torch.utils.data import Dataset
from transformers import LlamaTokenizer

class PopQADataset(Dataset):
    def __init__(self, tokenizer, split="test", max_length=512):
        """
        Args:
            tokenizer: A Hugging Face tokenizer.
            split (str): Which split to load (e.g., "test", "validation", "train").
            max_length (int): Maximum token length for the prompt.
        """
        self.tokenizer = tokenizer
        self.data = datasets.load_dataset("akariasai/PopQA")[split]
        self.max_length = max_length
        # Convert dataset to a list for efficient access
        self.data_list = list(self.data)
        # Use a fixed context of 7 examples (here, simply using the first 7 examples)
        self.fixed_context = self.data_list[:7]
        
    def __len__(self):
        return len(self.data_list)

    def __get_current_prompt__(self, idx):
        # Get the current example for which we need a prompt
        example = self.data_list[idx]
        question = example["question"]
        
        # Build the fixed context prompt using the same 7 examples each time
        example_texts = "\n\n".join(
            [
                f"Example {i+1}:\nQuestion: {ex['question']}\nAnswer: {ex['possible_answers'].split(' | ')[0]}"
                if isinstance(ex["possible_answers"], str)
                else f"Example {i+1}:\nQuestion: {ex['question']}\nAnswer: {ex['possible_answers'][0]}"
                for i, ex in enumerate(self.fixed_context)
            ]
        )
        
        prompt = f"""Below are some examples that show the correct response format. Each example consists of a question followed by an answer that is a list of entities. Your task is to answer the final question by listing only the entities that directly answer the question. The answer must:

• Include only the entities (no extra words or explanation).
• Be enclosed in square brackets [ ].
• List the entities separated by commas.

Examples:
{example_texts}

Now, answer the following question following the same format:
Question: {question}
Answer:
"""
        
        return prompt

    def __getitem__(self, idx):
        # Extract metadata from the example
        example = self.data_list[idx]
        question = example["question"]
        possible_answers = (
            " | ".join(example["possible_answers"]) 
            if isinstance(example["possible_answers"], list) 
            else example["possible_answers"]
        )
        
        # Build the prompt using the fixed context
        prompt = self.__get_current_prompt__(idx)
        
        # Tokenize the prompt
        tokenized = self.tokenizer(
            prompt, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        # Squeeze the tensors to remove the extra batch dimension
        input_ids = tokenized["input_ids"].squeeze()
        attention_mask = tokenized["attention_mask"].squeeze()
        
        # Compute the length of the prompt (number of non-padding tokens)
        prompt_length = attention_mask.sum().item()  # Count of actual tokens
        last_prompt_token_position = prompt_length - 1  # Position of the last token
        
        # Return tokenized inputs along with the prompt and metadata
        return {
            "prompt": prompt, 
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "prompt_last_token_position": last_prompt_token_position,
            "metadata": {
                "idx": idx,
                "question": question,
                "possible_answers": possible_answers
            }
        }
