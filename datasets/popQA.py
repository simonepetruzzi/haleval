import numpy as np
import datasets
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer
import random


class PopQAMultiChoiceDataset(Dataset):

    def __init__(self, tokenizer, split="test", max_length=512):
        """
        Args:
            tokenizer: A Hugging Face tokenizer.
            split (str): Which split to load (e.g., "test", "validation", "train").
            max_length (int): Maximum token length for the prompt.
        """
        self.tokenizer = tokenizer
        self.data = datasets.load_dataset("akariasai/PopQA")['test']
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __get_current_prompt__(self, idx):
        # Get the raw example
        example = self.data[idx]

        # Assume the following keys exist; adjust if necessary.
        question = example["question"]
        examples = random.sample(list(self.data), 10)
        example_texts = "\n\n".join(
            [
                f"Example {i+1}:\nQuestion: {ex['question']}\nAnswer: {ex['possible_answers'].split(' | ')[0]}"
                if isinstance(ex["possible_answers"], str)
                else f"Example {i+1}:\nQuestion: {ex['question']}\nAnswer: {ex['possible_answers'][0]}"
                for i, ex in enumerate(examples)
            ]
        )   
        
        prompt = f"""Answer the following questions concisely and accurately, using no more than one sentence. Follow the format of the examples provided.
        
        {example_texts}
                    
        Now, answer the following:
        Question: {question}
        Answer:"""

        return prompt
