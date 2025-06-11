import torch
from torch import nn
from torch.utils.data import DataLoader

import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    # Импортируем корневую директорию
import config


class ConvEncoder(nn.Module):
    def __init__(self, filters=config.FILTERS, dropout=config.DROPOUT2D, kernel_size=config.KERNEL_SIZE, 
                 input_shape=(1, config.INPUT_DIM_CNN[0], config.INPUT_DIM_CNN[1]), pool_size=config.POOL_SIZE) -> None:
        """Конструктор класса сверточного энкодера

        Args:
            filters (list, optional): Количество фильтров. Defaults to [32, 64].
            dropout (float, optional): Вероятность выключения нейронов в сверточных слоях. Defaults to 0.1.
            kernel_size (tuple, optional): Размерность ядра. Defaults to (5, 3).
            input_shape (tuple, optional): Размерность входных данных без батчей. Defaults to (1, 8, 52).
            pool_size (tuple, optional): Размер матрицы для пуллинга. Defaults to (3, 1).
        """
        super(ConvEncoder, self).__init__()

        self.conv_1 = nn.Sequential(*[
            nn.Conv2d(in_channels=1, out_channels=filters[0], kernel_size=kernel_size, padding='same'),    # padding='same'
            nn.BatchNorm2d(filters[0]), nn.PReLU(), nn.Dropout2d(dropout), 
            nn.MaxPool2d(kernel_size=pool_size)
        ])

        self.conv_2 = nn.Sequential(*[
            nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(filters[1]), nn.PReLU(), nn.Dropout2d(dropout), 
            nn.MaxPool2d(pool_size),
        ])

        self.flatten = nn.Flatten()

    def get_output_dim(self, input_shape):
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)  # input_shape = (8, 52)
            x = x.unsqueeze(1)  # manually match actual forward()
            x = self.conv_1(x)
            x = self.conv_2(x)
            x = self.flatten(x)
            return x.shape[1]

    def forward(self, x):
        x = x.unsqueeze(1)    # [B, 1, H, W]
        x = self.conv_1(x)
        x_channels = self.conv_2(x)
        x = self.flatten(x_channels)

        return x, x_channels
    

class FFClassifier(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=config.HIDDEN_DIM, output_dim=config.OUTPUT_DIM) -> None:
        super(FFClassifier, self).__init__()
        if input_dim is None:
            raise ValueError("FFClassifier: input_dim не задан (None).")

        layers = []
        layers.extend([nn.Linear(input_dim, hidden_dim[0]), nn.PReLU()])

        for i in range(len(hidden_dim)):
            if i + 1 == len(hidden_dim):
                layers.extend([nn.Linear(hidden_dim[i], output_dim)])    # nn.Softmax(dim=1)ё
            else:
                layers.extend([nn.Linear(hidden_dim[i], hidden_dim[i+1]), nn.PReLU()])

        self.ff_classifier = nn.Sequential(*layers)

    def forward(self, x):
        out = self.ff_classifier(x)

        return out
    

class FullModel(nn.Module):
    def __init__(self, input_shape):
        super(FullModel, self).__init__()

        self.conv_encoder = ConvEncoder()    # [B, C, H, W]
        self.output_dim = self.conv_encoder.get_output_dim((config.INPUT_DIM_CNN[0], config.INPUT_DIM_CNN[1]))    # Автоматически подбирает размерность
        self.output_dim = self.conv_encoder.get_output_dim(input_shape)
        self.ff_classifier = FFClassifier(input_dim=self.output_dim)

    def forward(self, x):
        x, x_channels = self.conv_encoder(x)
        out = self.ff_classifier(x)

        return out, x_channels
