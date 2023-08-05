import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from butterflydata import ButterflyDataset
import os
import numpy as np
import pandas as pd

train_dataset = ButterflyDataset(
    labels="data/Training_set.csv",
    img_dir="data/test/",
    transform=None,
    target_transform=None
)
test_dataset = ButterflyDataset(
    labels="data/Testing_set.csv",
    img_dir="data/test/",
    transform=None,
    target_transform=None
)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
x, y = next(iter(train_dataset))
print(x)