import os
import sys
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
import utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    # Импортируем корневую директорию
from config import SUBJECTS

class SurfaceEMGDataset(Dataset):
    def __init__(self,  subjects_lst=SUBJECTS, data_dir='data/', exercise=2, transform=None, window_size=32):
        self.data_dir = data_dir
        self.transform = transform
        self.exercise = exercise
        self.window_size = window_size

        self.folder_list = [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        self.folder_list = sorted(self.folder_list)

        self.emg = []



    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        folder_path = os.path.join(self.data_dir, self.folder_list[idx])
        
        for i in range(1, 11):
            filename = f'S1_E{self.exercise}_A1.mat'
            file_path = os.path.join(folder_path, filename)

            data = loadmat(file_path)


        image = Image.open(file_path + f'S1_E{self.exercise}_A1.mat')

        if self.transform:
            image = self.transform(image)

        label = self._get_label(file_path)  # свой способ получения метки
        return image, label

    def _get_label(self, file_path):
        # Например, извлекаем метку из имени файла
        # 'cat_01.jpg' -> метка 0, 'dog_01.jpg' -> метка 1
        if "cat" in file_path:
            return 0
        elif "dog" in file_path:
            return 1
        else:
            return -1  # или обработка ошибки
        

def main():
    emgs = []
    labels = []

    for i in range(1, 3):
        file_path = f'data/s{i}/S{i}_E2_A1.mat'
        data = loadmat(file_path)

        emg_raw = data['emg'][:,0:8]    # TODO: Проверить какие датчики добавляются, т.к. в массиве 16 сигналов
        labels_raw = data['stimulus']       

        emgs.extend(utils.split_into_batches(emg_raw, 32))
        labels_tmp = utils.split_into_batches(labels_raw, 32)

        for label_window in labels_tmp:
            _, freqs = np.unique(label_window, return_counts=True) 

            if freqs.shape != (1,):
                print(np.where(freqs == freqs.max()))
                print(freqs)

    print(len(emgs))
    print(emgs[0].shape)
    print(emgs[0])

if __name__ == "__main__":
    main()