import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np


def train(loss_name, optimizer_name, model, train_data, test_data, epoch, device, lti, itl):
    loss = None
    optimizer = None
    if loss_name == "cross_entropy":
        loss = nn.CrossEntropyLoss()
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.008)
    loss.to(device)
    for e in range(epoch):
        n = 0
        running_loss = 0.0
        for i, data in enumerate(train_data, 0):
            n += 1
            inputs, labels = data
            inputs = inputs.to(device)
            labels = [lti[word] for word in labels]
            labels_oh = torch.tensor(labels, dtype=torch.long).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            l = loss(outputs, labels_oh)
            l.backward()
            optimizer.step()
            # print statistics
            running_loss += l.item()
        print("episode", e, ", Running loss", running_loss/n)

    print('Finished Training')