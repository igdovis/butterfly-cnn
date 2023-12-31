import torch
from torch.utils.data import DataLoader
from butterflydata import ButterflyDataset
import os
import pandas as pd
import torchvision.transforms as transforms
from model import CNN as CNN
from train import train, evaluate
import streamlit as st
from streamlitapp import streamlit_app
device = torch.device("cpu")
trainingdata = pd.read_csv("data/Training_set.csv")
labelset = sorted(trainingdata['label'].unique().tolist())
lti = {}
for i, entry in enumerate(labelset):
    lti[entry] = i
itl = {value:key for (key, value) in lti.items()}

norm_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
target_transform = transforms.Compose([
    transforms.Lambda(lambda x: lti[x]),
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))
])

train_dataset = ButterflyDataset(
    labels="data/Training_set.csv",
    img_dir="data/train/",
    transform=norm_transform,
    target_transform=target_transform
)
test_dataset = ButterflyDataset(
    labels="data/Testing_set.csv",
    img_dir="data/test/",
    transform=norm_transform,
    target_transform=target_transform
)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
#cnn = CNN().to(device)
#cnn = CNN()
epochs = 22
optimizer = "SGD"
loss = "cross_entropy"
#train(loss, optimizer, cnn, train_dataloader, test_dataloader, epochs, device, lti, itl)
#cnn.load_state_dict(torch.load('./butterlynet.pth', map_location=torch.device("cpu")))
#evaluate(cnn, test_dataloader, itl, lti, device)

streamlit_app(device, lti, itl)
