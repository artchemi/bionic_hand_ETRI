import os
import sys

import torch
from torch.utils.data import Dataset

from scipy.io import loadmat
import numpy as np

import utils
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    # Импортируем корневую директорию
from config import SUBJECTS, WINDOW_SIZE, GLOBAL_SEED


def extract_emg_labels(subjects, exercise=2, window_size=WINDOW_SIZE) -> list:
    emgs = []
    labels = []

    for subject in tqdm(subjects): 
        file_path = f'data/s{subject}/S{subject}_E{exercise}_A1.mat'
        data = loadmat(file_path)

        # NOTE: Добавляются первые 8 колонок, которые равномерно расположены вокруг 
        # предплечья на высоте лучезапястного сустава
        emg_raw = data['emg'][:,0:8] 
        labels_raw = data['stimulus']       

        emg_windows = utils.split_into_batches(emg_raw, window_size)
        label_windows = utils.split_into_batches(labels_raw, window_size)

        for emg_win, label_win in zip(emg_windows, label_windows):
            labels_unique, freqs = np.unique(label_win, return_counts=True)

            if freqs.shape == (2,):
                idx_max = np.where(freqs == freqs.max())
                if len(idx_max[0]) == 2:  # равные частоты
                    continue
                else:
                    emgs.append(emg_win)
                    labels.append(int(labels_unique[idx_max[0][0]]))
            else:
                emgs.append(emg_win)
                labels.append(int(labels_unique[0]))

    enc = OneHotEncoder(sparse_output=False)
    labels_encoded = enc.fit_transform(np.array(labels).reshape(-1, 1))            
    # TODO: Добавить one-hot кодирование 
    return emgs, labels_encoded


class EMGTransform:
    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if self.normalize:
            # Нормализация по каналам (ось 0 — временная, ось 1 — каналы)
            x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        return x
    

class SurfaceEMGDataset(Dataset):
    def __init__(self, emg: np.ndarray, labels_encoded:np.ndarray, transform=None):
        self.emg = emg
        self.labels_encoded = labels_encoded

        self.transform = transform

    def __len__(self):
        return len(self.emg)

    def __getitem__(self, idx):
        X = torch.tensor(self.emg[idx], dtype=torch.float32)
        y = torch.tensor(self.labels_encoded[idx], dtype=torch.float32)

        if self.transform:
            X = self.transform(X)

        return X, y
        

def main():
    # TODO: Написать тест для этого датасета
    emg, labels = extract_emg_labels(SUBJECTS, exercise=2, window_size=WINDOW_SIZE)
    emg_train, emg_test, labels_train, labels_test = train_test_split(emg, labels, train_size=0.7, random_state=GLOBAL_SEED)

    transform = EMGTransform(normalize=True)
    train_dataset = SurfaceEMGDataset(emg=emg_train, labels_encoded=labels_train, transform=transform)
    test_dataset = SurfaceEMGDataset(emg=emg_test, labels_encoded=labels_test, transform=transform)
    
    print(len(train_dataset))
    print(len(test_dataset))

if __name__ == "__main__":
    main()