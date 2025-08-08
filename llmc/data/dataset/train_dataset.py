import torch
import numpy as np
from loguru import logger
import copy


class TrainJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, block_size) -> None:
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized_datasets = []
        self.data = []
        for d in raw_data:
            if "text" in d:
                tokenized_datasets.append(self.tokenize_function(d))
            else:
                d.update(labels=d['input_ids'].detach())
                self.data.append({k: (v.squeeze(0) if isinstance(v, torch.Tensor) else v) for k, v in d.items()})

        if tokenized_datasets:
            grouped_dataset = self.group_texts(tokenized_datasets)
            self.data.extend({
                'input_ids': grouped_dataset['input_ids'][i], 
                'labels': grouped_dataset['labels'][i]} for i in range(len(grouped_dataset['input_ids'])))
        keys = set()
        for d in self.data:
            keys.update(d.keys())
        for d in self.data:
            for k in keys:
                if k not in d:
                    d[k] = None
        np.random.shuffle(self.data)  # Shuffle the dataset
        # self.data = [
        #     dict(input_ids=self.input_ids[i], labels=self.labels[i],
        #          attention_mask=self.attention_mask[i])
        #     for i in range(len(self.input_ids))
        # ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'])

    def group_texts(self, examples):
        # Concatenate all texts.
        # Initialize an empty dictionary
        concatenated_examples = {}

        # Loop through the list of dictionaries
        for d in examples:
            # Loop through the keys in each dictionary
            for key in d.keys():
                # If the key is not already a key in the dict_of_lists, create a new list
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                # Append the value to the list associated with the key in dict_of_lists
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples['input_ids'])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result['labels'] = result['input_ids'].copy()
        return result