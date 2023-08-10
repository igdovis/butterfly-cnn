import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

def train(loss_name, optimizer_name, model, train_data, test_data, epoch, device, lti, itl):
    running_losses = []
    model.train()
    loss = None
    optimizer = None
    if loss_name == "cross_entropy":
        loss = nn.CrossEntropyLoss()
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss.to(device)
    for e in tqdm(range(epoch)):
        n = 0
        running_loss = 0.0
        for i, data in enumerate(train_data, 0):
            n += 1
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
            running_loss += l.item()
        running_losses.append(running_loss)
        print("episode", e, ", Running loss", running_loss/n)
    PATH = './butterflynet.pth'
    torch.save(model.state_dict(), PATH)
    print('Finished Training')
    evaluate(model, test_data, itl,lti,  device)

def evaluate(model, test_data, itl, lti, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_data:
            images, ground_truth = data
            images = images.to(device)
            ground_truth = ground_truth.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            top3probs = torch.topk(probs[0], 3)
            print("GROUND TRUTH", itl[ground_truth[0].item()])
            _, predicted = torch.max(outputs.data, 1)
            print("PREDICT", itl[predicted[0].item()])
            print("TOP 3 PROBS", top3probs)
            total += ground_truth.size()[0]
            correct += (predicted == ground_truth).sum().item()

    print(f"Accuracy of test dataset is {correct/total * 100}%")

