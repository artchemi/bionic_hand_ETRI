import os
import sys

import torch
from torch.utils.data import Dataset

from scipy.io import loadmat
import numpy as np

import utils
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    # Импортируем корневую директорию
from config import SUBJECTS, WINDOW_SIZE


def extract_emg_labels(subjects, exercise=2, window_size=WINDOW_SIZE) -> list:
    emgs = []
    labels = []

    for subject in tqdm(subjects):    # TODO: Добавить выбор пользователей через аргументы 
        file_path = f'data/s{subject}/S{subject}_E{exercise}_A1.mat'
        data = loadmat(file_path)

        # NOTE: Добавляются первые 8 колонок, которые равномерно расположены вокруг 
        # предплечья на высоте лучезапястного сустава
        emg_raw = data['emg'][:,0:8] 
        labels_raw = data['stimulus']       

        emgs.extend(utils.split_into_batches(emg_raw, window_size))
        labels_tmp = utils.split_into_batches(labels_raw, window_size)

        for label_window in labels_tmp:
            labels_unique, freqs = np.unique(label_window, return_counts=True) 

            if freqs.shape == (2,):    # Если в частотах больше одного значения
                idx_max = np.where(freqs == freqs.max())

                # print(idx_max[0])
                if len(idx_max[0].tolist()) == 2:    # Могут быть равные количества, скипаем это
                    continue
                else:
                    # labels.append(labels_unique[idx_max])
                    # labels.append(labels_unique[idx_max[0][0]])
                    labels.append(int(labels_unique[idx_max[0][0]]))
            else:
                # labels.append(labels_unique)
                labels.append(int(labels_unique[0]))

    enc = OneHotEncoder(sparse_output=False)
    labels_encoded = enc.fit_transform(np.array(labels).reshape(-1, 1))            
    # TODO: Добавить one-hot кодирование 
    return emgs, labels_encoded



class SurfaceEMGDataset(Dataset):
    def __init__(self,  subjects_lst=SUBJECTS, exercise=2, transform=None, window_size=32):
        self.transform = transform

        self.emg, self.labels = extract_emg_labels(subjects_lst, exercise=exercise, window_size=window_size)

    def __len__(self):
        return len(self.emg)

    def __getitem__(self, idx):
        X = torch.tensor(self.emg[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            X = self.transform(X)

        return X, y
        

def main():
    dataset = SurfaceEMGDataset(window_size=30)

    print(dataset[0])
    print(len(dataset))

if __name__ == "__main__":
    main()