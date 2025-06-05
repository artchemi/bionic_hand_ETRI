import os
import sys

from model import FullModel
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import (SurfaceEMGDataset, EMGTransform, extract_emg_labels, make_weighted_sampler, 
                     compute_stats, save_stats, load_stats, GlobalStandardizer, gestures)

from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    # Импортируем корневую директорию
from config import SUBJECTS, WINDOW_SIZE, GLOBAL_SEED, TRAIN_SIZE, BATCH_SIZE, GESTURE_INDEXES, LEARNING_RATE, EPOCHS

import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(GLOBAL_SEED)


class Trainer:
    def __init__(self, model: nn.Module, device=torch.device('cuda'), lr=1e-3, weights=None):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self, dataloader: DataLoader) -> list:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs, _ = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return [avg_loss, accuracy]

    def evaluate(self, dataloader: DataLoader) -> list:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return [avg_loss, accuracy]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = FullModel()
    
    emg, labels = extract_emg_labels(subjects=SUBJECTS, window_size=WINDOW_SIZE, onehot=False)
    # emg, labels = relax_shrink(emg, labels, shrink_size=80000, seed=GLOBAL_SEED)
    emg, labels = gestures(
        emg,
        labels,
        targets=[0,1,2,3],    # или любые твои метки
        relax_shrink=80000,
        rand_seed=GLOBAL_SEED
    )

    emg_train, emg_test, labels_train, labels_test = train_test_split(emg, labels, train_size=TRAIN_SIZE, random_state=GLOBAL_SEED)

    # NOTE: Сохраняем среднее и стандартное отклонение в json, можно закомментировать, если они уже есть
    means, stds = compute_stats(emg_train)      # emg_train: np.ndarray [N, window, C]
    save_stats(means, stds, 'emg_stats.json')

    transform = EMGTransform(normalize=True)

    means, stds = load_stats('emg_stats.json')
    standardizer = GlobalStandardizer(means, stds)

    train_dataset = SurfaceEMGDataset(emg_train, labels_train, gestures=GESTURE_INDEXES, transform=standardizer)
    test_dataset = SurfaceEMGDataset(emg_test, labels_test, gestures=GESTURE_INDEXES, transform=standardizer)

    train_sampler = make_weighted_sampler(train_dataset)

    # ys = torch.tensor([y[1] for y in train_dataset])
    # print(ys.shape)
    # print(torch.where(ys == 0)[0].shape)
    # print(torch.where(ys == [1, 2])[0])
    # print(torch.where(ys == [1, 2])[0].shape)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)    # shuffle=True
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    trainer = Trainer(model, device=device, lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        train_loss, train_acc = trainer.train_epoch(train_dataloader)
        test_loss,  test_acc = trainer.evaluate(test_dataloader)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # for X, y in train_dataloader:
    #     # print(X.shape)
    #     print('Размерность основного выхода', model(X)[0])``
    #     print('Размерность после сверток', model(X)[1].shape)
    #     break


if __name__ == "__main__":
    main()

