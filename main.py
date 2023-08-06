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
from train import train

device = torch.device("cuda")
trainingdata = pd.read_csv("data/Training_set.csv")
labelset = set()
for i in range(len(trainingdata)):
    labelset.add(trainingdata.iloc[i, 1])
labelset = np.array(list(labelset))
lti = {}
for i, entry in enumerate(labelset):
    lti[entry] = i
itl = {value:key for (key, value) in lti.items()}


norm_transform = transforms.Compose(
    [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = ButterflyDataset(
    labels="data/Training_set.csv",
    img_dir="data/train/",
    transform=norm_transform,
    target_transform=None
)
test_dataset = ButterflyDataset(
    labels="data/Testing_set.csv",
    img_dir="data/test/",
    transform=norm_transform,
    target_transform=None
)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


test_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
cnn = CNN().to(device)
epochs = 100
optimizer = "SGD"
loss = "cross_entropy"
train(loss, optimizer, cnn, train_dataloader, test_dataloader, epochs, device, lti, itl)
