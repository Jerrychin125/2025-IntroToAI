import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import pandas as pd

class CNN(nn.Module):
    def __init__(self, num_classes=5):
        # (TODO) Design your CNN, it can only be less than 3 convolution layers
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, num_classes)
        # raise NotImplementedError

    def forward(self, x):
        # (TODO) Forward the model
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # raise NotImplementedError
        return x

def _model_loop(model, loader, criterion, device, optimiser=None):
    is_train = optimiser is not None
    mode = "train" if is_train else "eval"
    if is_train:
        model.train()
    else:
        model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(loader, desc=mode, leave=False)
    with torch.set_grad_enabled(is_train):
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if is_train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    avg_loss = running_loss / total
    accuracy = correct / total if not is_train else None
    return avg_loss, accuracy
                

def train(model: CNN, train_loader: DataLoader, criterion, optimizer, device)->float:
    # (TODO) Train the model and return the average loss of the data, we suggest use tqdm to know the progress
    avg_loss, _ = _model_loop(model, train_loader, criterion, device, optimiser=optimizer)
    # raise NotImplementedError
    return avg_loss


def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    # (TODO) Validate the model and return the average loss and accuracy of the data, we suggest use tqdm to know the progress
    avg_loss, accuracy = _model_loop(model, val_loader, criterion, device)
    # raise NotImplementedError
    return avg_loss, accuracy

def test(model: CNN, test_loader: DataLoader, criterion, device):
    # (TODO) Test the model on testing dataset and write the result to 'CNN.csv'
    model.eval()
    predictions = []
    image_ids = []
    with torch.no_grad():
        for imgs, names in tqdm(test_loader, desc="test", leave=False):
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().tolist())
            image_ids.extend(names)
    df = pd.DataFrame({"id": image_ids, "prediction": predictions})
    df.to_csv("CNN.csv", index=False)
    # raise NotImplementedError
    print(f"Predictions saved to 'CNN.csv'")
    return