'''
We will use the IMDb Dataset available on Kaggle for binary sentiment classification
(https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
'''

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class IMDbDataset(Dataset):
    def __init__(self, path_to_data, max_length=256):
        self.data = pd.read_csv(path_to_data)

        self.positive = 0
        self.negative = 1

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        review = self.data.loc[index,"review"]
        sentiment = self.data.loc[index,"sentiment"]
        token_ids = self.tokenizer(
            review,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]
        label = self.positive if sentiment == "positive" else self.negative
        return token_ids, label
    

if __name__ == "__main__":
    dataset = IMDbDataset("IMDbData.csv")
    a, b, c = dataset.evaluate_batch(torch.tensor([[1,0],[1,0],[1,0],[0,1]]), torch.tensor([0,0,1,1]))
    print(b)
    print(c)