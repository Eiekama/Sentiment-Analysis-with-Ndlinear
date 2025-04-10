'''
We will use the IMDb Dataset available on Kaggle for binary sentiment classification
(https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
'''

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torcheval.metrics.functional import binary_confusion_matrix

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
    
    @staticmethod
    def evaluate_batch(pred_logits, targ_labels):
        pred_labels = torch.argmax(pred_logits, dim=-1)
        num_correct = torch.sum(pred_labels == targ_labels).item()
        confusion = binary_confusion_matrix(pred_labels, targ_labels)
        return len(targ_labels), num_correct, confusion
    
    def evaluate(self, model, batch_size=128):
        dataloader = DataLoader(self, batch_size=batch_size, shuffle=True)
        n = 0
        accuracy = 0
        confusion_matrix = torch.zeros(2,2)
        for input_data, targ_labels in self.dataloader:
            pred_logits = model(input_data)
            b, correct, confusion = self.evaluate_batch(pred_logits, targ_labels)
            n += b
            accuracy += correct
            confusion_matrix += confusion
        return {
            "Total Number" : n,
            "Accuracy" : accuracy/n,
            "TP" : confusion_matrix[1,1],
            "FP" : confusion_matrix[0,1],
            "FN" : confusion_matrix[1,0],
            "TN" : confusion_matrix[0,0],
        }

if __name__ == "__main__":
    dataset = IMDbDataset("IMDbData.csv")
    a, b, c = dataset.evaluate_batch(torch.tensor([[1,0],[1,0],[1,0],[0,1]]), torch.tensor([0,0,1,1]))
    print(b)
    print(c)