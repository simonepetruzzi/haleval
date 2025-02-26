import numpy as np
import datasets
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer
import copy
import logging


class TruthfulQAMultiChoiceDataset(Dataset):

    def __init__(self, tokenizer, split="test", max_length=512):
        """
        Args:
            tokenizer: A Hugging Face tokenizer.
            split (str): Which split to load (e.g., "test", "validation", "train").
            max_length (int): Maximum token length for the prompt.
        """
        self.tokenizer = tokenizer
        self.data = datasets.load_dataset("truthful_qa", "multiple_choice", split=split)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the raw example
        example = self.data[idx]

        # Assume the following keys exist; adjust if necessary.
        question = example["question"]
        choices = example["choices"]  # A list of answer options
        correct_answer = example["answer_key"]  # The correct answer (could be an index or a string)

        # Build a prompt string that includes the question and all answer choices.
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"Choice {i+1}: {choice}\n"
        prompt += "Answer:"  # Leave the answer blank for the model to generate or evaluate.

        return {"prompt": prompt, "correct_answer": correct_answer}

