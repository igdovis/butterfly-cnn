import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from butterflydata import ButterflyDataset
import os
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from model import CNN as CNN
norm_transform = transforms.Compose(
    [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = ButterflyDataset(
    labels="data/Training_set.csv",
    img_dir="data/test/",
    transform=norm_transform,
    target_transform=None
)
test_dataset = ButterflyDataset(
    labels="data/Testing_set.csv",
    img_dir="data/test/",
    transform=norm_transform,
    target_transform=None
)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
x, y = next(iter(train_dataloader))
print(x.shape)
cnn = CNN()
x = x.reshape(-1)
print(x.shape)
output = cnn(x)
print(output.shape)