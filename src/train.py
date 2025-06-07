import os
import sys

from model import FullModel
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import (SurfaceEMGDataset, EMGTransform, extract_emg_labels, make_weighted_sampler, 
                     compute_stats, save_stats, load_stats, GlobalStandardizer, gestures)

from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    # Импортируем корневую директорию
from config import SUBJECTS, WINDOW_SIZE, GLOBAL_SEED, TRAIN_SIZE, BATCH_SIZE, GESTURE_INDEXES, LEARNING_RATE, EPOCHS, RUN_NAME, VALID_SIZE

import random
import numpy as np
import pandas as pd

from tqdm import tqdm 

from collections import defaultdict
import logging


def set_seed(seed=42):    # Фиксируем сиды
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(GLOBAL_SEED)

# NOTE: Логгер 
logging.basicConfig(filename=f'{RUN_NAME}.log',  # Имя файла
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s'
                    )
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# NOTE: Словарь для хранения метрики в процессе обучения 
data = defaultdict(list)    # data['accuracy'].append(...)    data['F1'].append(...)


class Trainer:
    def __init__(self, model: nn.Module, device=torch.device('cuda'), lr=1e-3, weights=None):
        """_summary_

        Args:
            model (nn.Module): _description_
            device (_type_, optional): _description_. Defaults to torch.device('cuda').
            lr (_type_, optional): _description_. Defaults to 1e-3.
            weights (_type_, optional): _description_. Defaults to None.
        """
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
    emg, labels = gestures(
        emg,
        labels,
        targets=[0,1,2,3],    # или любые твои метки
        relax_shrink=80000,
        rand_seed=GLOBAL_SEED
    )

    # NOTE: Разделение данных на обучающую, валидационную и тестовую выборки
    emg_train, emg_tmp, labels_train, labels_tmp = train_test_split(emg, labels, train_size=TRAIN_SIZE, random_state=GLOBAL_SEED)
    emg_valid, emg_test, labels_valid, labels_test = train_test_split(emg_tmp, labels_tmp, train_size=VALID_SIZE, random_state=GLOBAL_SEED)

    # NOTE: Сохраняем среднее и стандартное отклонение в json, можно закомментировать, если они уже есть
    means, stds = compute_stats(emg_train)      # emg_train: np.ndarray [N, window, C]
    save_stats(means, stds, 'emg_stats.json')
    means, stds = load_stats('emg_stats.json')    # Импортируем stats и делаем стандартизатор
    standardizer = GlobalStandardizer(means, stds)

    # transform = EMGTransform(normalize=True)

    train_dataset = SurfaceEMGDataset(emg_train, labels_train, gestures=GESTURE_INDEXES, transform=standardizer)
    valid_dataset = SurfaceEMGDataset(emg_valid, labels_valid, gestures=GESTURE_INDEXES, transform=standardizer)
    test_dataset = SurfaceEMGDataset(emg_test, labels_test, gestures=GESTURE_INDEXES, transform=standardizer)

    train_sampler = make_weighted_sampler(train_dataset)    # NOTE: Сэмплер для несбалансированных классов

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)    # shuffle=True
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    trainer = Trainer(model, device=device, lr=LEARNING_RATE)
    for epoch in tqdm(range(EPOCHS)):
        train_loss, train_acc = trainer.train_epoch(train_dataloader)
        test_loss,  test_acc = trainer.evaluate(test_dataloader)

        info_str = f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {test_loss:.4f}, Acc: {test_acc:.4f}"
        logger.info(info_str)
        

if __name__ == "__main__":
    main()

