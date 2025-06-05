import os
import sys
import json

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from scipy.io import loadmat
import numpy as np
import random

import utils
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    # Импортируем корневую директорию
from config import SUBJECTS, WINDOW_SIZE, GLOBAL_SEED, STEP_SIZE


def extract_emg_labels(subjects, exercise=2, window_size=WINDOW_SIZE, onehot=False) -> list:
    emgs = []
    labels = []

    for subject in tqdm(subjects): 
        file_path = f'data/s{subject}/S{subject}_E{exercise}_A1.mat'
        data = loadmat(file_path)

        emg_raw = data['emg'][:, 0:8] 
        labels_raw = data['stimulus']       

        emg_windows = utils.split_into_batches(emg_raw, window_size, STEP_SIZE)
        label_windows = utils.split_into_batches(labels_raw, window_size, STEP_SIZE)

        for emg_win, label_win in zip(emg_windows, label_windows):
            # 💥 Добавим фильтрацию по размеру окна (должно быть ровно [window_size, 8])
            if emg_win.shape != (window_size, 8):
                continue

            labels_unique, freqs = np.unique(label_win, return_counts=True)

            if freqs.shape == (2,):
                idx_max = np.where(freqs == freqs.max())
                if len(idx_max[0]) == 2:
                    continue
                else:
                    emgs.append(emg_win)
                    labels.append(int(labels_unique[idx_max[0][0]]))
            else:
                emgs.append(emg_win)
                labels.append(int(labels_unique[0]))

    if onehot:
        enc = OneHotEncoder(sparse_output=False)
        labels = enc.fit_transform(np.array(labels).reshape(-1, 1))      

    return np.asarray(emgs), np.asarray(labels)


class EMGTransform:
    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, x):
        if self.normalize:
            # Нормализация по каналам (ось 0 — временная, ось 1 — каналы)
            x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        return x
    

class SurfaceEMGDataset(Dataset):
    def __init__(self, emg: np.ndarray, labels:np.ndarray, gestures: list, transform=None):
        self.emg = emg
        self.labels = labels

        self.X = []
        self.y = []

        self.transform = transform

        # print(self.labels.shape)
        # print(type(self.emg))
        for gesture in gestures:
            idxs = np.where(self.labels == gesture)[0]

            # print(idxs)

            self.X.extend(self.emg[idxs])
            self.y.extend(self.labels[idxs])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)    # NOTE: Для BCE нужен тип Long 

        if self.transform:
            X = self.transform(X)

        return X, y
    

def make_weighted_sampler(dataset):
    """
    Принимает датасет, у которого в dataset.y лежат метки (list или np.ndarray),
    и возвращает WeightedRandomSampler.
    """
    # Собираем все метки
    labels = torch.tensor(dataset.y, dtype=torch.long)
    num_classes = int(labels.max().item()) + 1

    # Считаем количество примеров каждого класса
    class_counts = torch.bincount(labels, minlength=num_classes).float()

    # Веса для каждого класса: total_samples / (num_classes * count_i)
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)
    # Вес каждого примера = вес его класса
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True
    )
    return sampler

def compute_stats(emg_array):
    """
    Вычислить среднее и σ по каждому каналу для всего датасета.
    emg_array: np.ndarray или torch.Tensor формы [N, window, C]
    Возвращает:
        means: torch.Tensor формы [C]
        stds:  torch.Tensor формы [C]
    """
    # Приводим к тензору
    if not isinstance(emg_array, torch.Tensor):
        emg = torch.tensor(emg_array, dtype=torch.float32)
    else:
        emg = emg_array.float()
    # Перемещаем каналы на вторую ось: [N, C, window]
    emg = emg.permute(0, 2, 1)
    means = emg.mean(dim=(0, 2))
    stds  = emg.std(dim=(0, 2))
    return means, stds

def save_stats(means, stds, path):
    """Сохранить means и stds в JSON."""
    stats = {'mean': means.tolist(), 'std': stds.tolist()}
    with open(path, 'w') as f:
        json.dump(stats, f)

def load_stats(path):
    """Загрузить means и stds из JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    return torch.tensor(data['mean']), torch.tensor(data['std'])

class GlobalStandardizer:
    def __init__(self, means, stds):
        """
        means, stds: torch.Tensor формы [C]
        """
        self.means = means
        self.stds  = stds
        
    def __call__(self, x):
        """
        x: torch.Tensor формы [window, C] или [C, window]
        Возвращает: torch.Tensor формы [C, window]
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Если первый размер не равен числу каналов, транспонируем
        C = self.means.shape[0]
        if x.ndim == 2 and x.shape[0] != C:
            # предполагаем, что x.shape == [window, C]
            x = x.permute(1, 0)
        
        # Теперь x: [C, window]
        return (x - self.means.unsqueeze(1)) / (self.stds.unsqueeze(1))
    

def gestures(emg: np.ndarray,
             labels: np.ndarray,
             targets: list = [0, 1, 3, 6],
             relax_shrink: int = 80000,
             rand_seed: int = 2022):
    """
    Аналог TensorFlow-функции gestures(): 
      - Разбивает сэмплы по классам (targets).
      - Урезает класс 0 до relax_shrink.
      - Склеивает обратно списки в два массива.
    
    Args:
        emg (np.ndarray): shape [N, window, C]
        labels (np.ndarray): shape [N]
        targets (list): список меток, которые нас интересуют
        relax_shrink (int): количество образцов класса 0, которое оставляем
        rand_seed (int): сид для reproducibility
    
    Returns:
        emg_out (np.ndarray): undersampled EMG, shape [M, window, C]
        labels_out (np.ndarray): метки, shape [M]
    """
    # 1) Собираем по классам
    class_dict = {t: [] for t in targets}
    for x, y in zip(emg, labels):
        if y in class_dict:
            class_dict[y].append(x)
    
    # 2) Урезаем класс 0 (relax), если требуется
    random.seed(rand_seed)
    if 0 in class_dict and relax_shrink is not None:
        relax_list = class_dict[0]
        if len(relax_list) > relax_shrink:
            class_dict[0] = random.sample(relax_list, relax_shrink)
        # иначе оставляем все
    
    # 3) Собираем обратно списки в плоские массивы
    emg_out = []
    labels_out = []
    for label, samples in class_dict.items():
        emg_out.extend(samples)
        labels_out.extend([label] * len(samples))
    
    # 4) Перемешиваем общий порядок
    combined = list(zip(emg_out, labels_out))
    random.seed(rand_seed)
    random.shuffle(combined)
    emg_out, labels_out = zip(*combined)
    
    return np.stack(emg_out), np.array(labels_out, dtype=np.int64)

    


def main():
    # means, stds = compute_stats(emg_train)      # emg_train: np.ndarray [N, window, C]
    # save_stats(means, stds, 'emg_stats.json')
    pass


if __name__ == "__main__":
    main()