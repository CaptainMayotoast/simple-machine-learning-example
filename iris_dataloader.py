import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from ucimlrepo import fetch_ucirepo 
from sklearn import model_selection

class IrisDataset(Dataset):
    def __init__(self, train=True):
        self.iris = fetch_ucirepo(id=53)
        
        self.x = self.iris.data.features 
        self.y = self.iris.data.targets 

        if train:
            # 20% of data is typically test
            self.x, _, self.y, _ = model_selection.train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        else:
            _, self.x, _, self.y = model_selection.train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        
        self.normalized_x = (self.x - self.x.mean()) / self.x.std() 

        self.y = self.y.astype('category')
        self.y = self.y['class'].cat.codes

        # print(self.normalized_x)
        # print(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        input_row = torch.tensor(self.normalized_x.iloc[idx, :], dtype=torch.float)
        label = torch.tensor(self.y.iloc[idx],  dtype=torch.long)
        # print(f"input row: {input_row}, label: {label}")
        return input_row, label

if __name__ == "__main__":
    id = IrisDataset(True)
