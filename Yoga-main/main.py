import numpy as np
import os
from glob import glob

import cv2 as cv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
from utils.dataProcess import getData
from models.MLP import MLP

def main():
    BATCH_SIZE = 4
    EPOCH = 100

    train_loader = DataLoader(getData(folder='C:/Users/LENOVO/Downloads/archive/DATASET/TRAIN'), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(getData(folder='C:/Users/LENOVO/Downloads/archive/DATASET/TEST'), batch_size=BATCH_SIZE, shuffle=False)

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCH):
        train_loss = 0
        model.train()
        for batch, (feature, label) in enumerate(train_loader):
            feature = feature.view(feature.size(0), -1)  # Flatten the input
            pred = model(feature)
            loss = criterion(pred, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        print(f"EPOCH = {epoch+1}, train loss = {train_loss / len(train_loader):.4f}")

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch, (feature, label) in enumerate(test_loader):
            feature = feature.view(feature.size(0), -1)  # Flatten the input
            pred = model(feature)
            _, predicted = torch.max(pred.data, 1)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
        # Convert y_true from one-hot encoded to class integers
    y_true = np.argmax(y_true, axis=1)
    # Ensure y_true and y_pred are in the correct format
    print(f'y_true: {y_true[:10]}')
    print(f'y_pred: {y_pred[:10]}')
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted' ,zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted',zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted',zero_division=1)

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

if __name__ == '__main__':
    main()
