# NOTE: Жесты
GESTURE_INDEXES = []

# NOTE: Параметры сверточных слоев
FILTERS = [32, 64]
KERNEL_SIZE = (5, 3)
POOL_SIZE = (3, 1)
DROPOUT2D = 0.2
INPUT_DIM_CNN = (8, 52)

# NOTE: Параметры классификатора
INPUT_DIM_CLASSIFIER = FILTERS[-1] * INPUT_DIM_CNN[0] * INPUT_DIM_CNN[-1]
HIDDEN_DIM = [512, 128]
OUTPUT_DIM = len(GESTURE_INDEXES)

# NOTE: Список субъектов
SUBJECTS = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']