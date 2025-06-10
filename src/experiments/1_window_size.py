import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from model import FullModel

import torch
from torch import nn
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info    # Для оценки нагруженности модели

from dataset import (SurfaceEMGDataset, EMGTransform, extract_emg_labels, make_weighted_sampler, 
                     compute_stats, save_stats, load_stats, GlobalStandardizer, gestures)

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    # Импортируем корневую директорию
from config import (SUBJECTS, WINDOW_SIZE, GLOBAL_SEED, TRAIN_SIZE, BATCH_SIZE, GESTURE_INDEXES, 
                    LEARNING_RATE, EPOCHS, RUN_NAME, VALID_SIZE, EARLY_STOP_THRS, INPUT_DIM_CNN)

import random
import numpy as np
import pandas as pd

from tqdm import tqdm 

from collections import defaultdict
import logging
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--window_size', type=int, help='Размер окна')
args = parser.parse_args()


def set_seed(seed=42):    # Фиксируем сиды
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(GLOBAL_SEED)

# NOTE: Создаем папку для хранения логов
logs_path = f'logs/model_{args.window_size}'
log_file = logs_path+'/'+'run.log'
os.makedirs(logs_path, exist_ok=True)

# NOTE: Логгер 
if os.path.exists(log_file):    # Удаляет старый лог
    os.remove(log_file)
logging.basicConfig(filename=log_file,  # Имя файла
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class CSVLogger:
    def __init__(self) -> None:
        self.data = defaultdict(list)
        self.set_types = ['train', 'valid', 'test']

    def add_metrics(self, loss: list, accuracy: list, f1: list=None, precision: list=None, recall: list=None) -> None:
        """_summary_

        Args:
            loss (list): _description_
            accuracy (list): _description_
            f1 (list, optional): _description_. Defaults to None.
            precision (list, optional): _description_. Defaults to None.
            recall (list, optional): _description_. Defaults to None.
        """
        assert len(loss) == len(accuracy), "Количество метрик должно совпадать"

        for set, loss_set, accuracy_set, f1_set, precision_set, recall_set in zip(self.set_types, loss, accuracy, f1, precision, recall):
            self.data[f'{set}_loss'].append(loss_set)
            self.data[f'{set}_accuracy'].append(accuracy_set)
            self.data[f'{set}_f1'].append(f1_set)
            self.data[f'{set}_precision'].append(precision_set)
            self.data[f'{set}_recall'].append(recall_set)
    
    def save_csv(self, path: str) -> None:
        df = pd.DataFrame(self.data)
        df.to_csv(path, index=False)


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

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / total
        accuracy = correct / total
        
        # NOTE: Расчет метрик классификации
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')    # FIXME: Подумать, что будет лучше: micro, macro или weighted 

        return [avg_loss, accuracy, f1, precision, recall]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    emg, labels = extract_emg_labels(subjects=SUBJECTS, window_size=args.window_size, onehot=False)
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

    first_batch = next(iter(train_dataloader))
    x_sample, _ = first_batch  # x_sample: [batch_size, height, width]
    input_shape = x_sample.shape[1:]  # -> (height, width)
    model = FullModel(input_shape=input_shape)

    trainer = Trainer(model, device=device, lr=LEARNING_RATE)
    logger_csv = CSVLogger()    # Сохраняет метрики после обучения в .csv

    # input_shape = (INPUT_DIM_CNN[0], INPUT_DIM_CNN[1])
    input_shape = (input_shape[0], input_shape[1])
    macs, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=True, verbose=True)
    flops = (float(macs.split(' ')[0]) * 2) / 1e6
    # print(f'MACs: {macs}')
    # print(f'Params: {params}')
    # print(f'MFlops: {flops}')

    logger.info(f'MACs: {macs}')
    logger.info(f'Parameters: {params}')
    logger.info(f'FLops: {flops}')

    early_stopping_dict = {'min_loss_valid': 0, 'best_epoch': 0, 'counter': 0}
    for epoch in tqdm(range(EPOCHS)):
        train_loss, train_acc = trainer.train_epoch(train_dataloader)    # NOTE: Обучение на одной эпохе

        # NOTE: Тестирование
        train_loss, train_acc, f1_train, precision_train, recall_train = trainer.evaluate(train_dataloader) 
        valid_loss, valid_acc, f1_valid, precision_valid, recall_valid = trainer.evaluate(valid_dataloader)
        test_loss, test_acc, f1_test, precision_test, recall_test = trainer.evaluate(test_dataloader)

        logger_csv.add_metrics(loss=[train_loss, valid_loss, test_loss], accuracy=[train_acc, valid_acc, test_acc], 
                               f1=[f1_train, f1_valid, f1_test], precision=[precision_train, precision_valid, precision_test], 
                               recall=[recall_train, recall_valid, recall_test])

        info_str = f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {valid_loss:.4f}, Acc: {valid_acc:.4f}"
        logger.info(info_str)

        if epoch == 0:
            early_stopping_dict['min_loss_valid'] = valid_loss
            early_stopping_dict['best_epoch'] = epoch
        elif valid_loss < early_stopping_dict['min_loss_valid']:    # Сброс счетчика и обновление оптимальных метрик
            early_stopping_dict['min_loss_valid'] = valid_loss
            early_stopping_dict['best_epoch'] = epoch
            early_stopping_dict['counter'] = 0
        else:
            early_stopping_dict['counter'] += 1
        
        if early_stopping_dict['counter'] > EARLY_STOP_THRS:
            logger.info(f"Best valid loss: {early_stopping_dict['min_loss_valid']}")
            logger.info(f"Best epoch: {early_stopping_dict['best_epoch']}")
            break

    logger_csv.save_csv(path=logs_path+'/'+'metrics.csv')    # NOTE: Сохранение .csv    
    torch.save(model.state_dict(), logs_path+f"/best_model_{early_stopping_dict['best_epoch']}.pth")    # Сохранение весов


if __name__ == "__main__":
    main()

