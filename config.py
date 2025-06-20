# TODO: 
RUN_NAME = 'model_10'

GLOBAL_SEED = 42

# NOTE: Жесты
GESTURE_INDEXES = [0, 1, 3, 5, 6, 7, 13, 14]    # [0, 1, 2, 6, 7] + добавить один жест из другого упражнения

# NOTE: Параметры сигналов
WINDOW_SIZE = 10
N_CHANNELS = 8
STEP_SIZE = 10

# NOTE: Параметры сверточных слоев
FILTERS = [32, 64]
KERNEL_SIZE = (3, 5)
POOL_SIZE = (1, 3)
DROPOUT2D = 0.2
INPUT_DIM_CNN = (N_CHANNELS, WINDOW_SIZE)
# INPUT_DIM_CNN = (WINDOW_SIZE, N_CHANNELS)

# NOTE: Параметры классификатора
INPUT_DIM_CLASSIFIER = FILTERS[-1] * INPUT_DIM_CNN[0] * INPUT_DIM_CNN[-1]
HIDDEN_DIM = [512, 256, 128, 64]
OUTPUT_DIM = 17    # len(GESTURE_INDEXES)

# NOTE: Список субъектов
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SUBJECT_TEST = 10

# NOTE: Параметры датасета
TRAIN_SIZE = 0.7
VALID_SIZE = 0.5    # Относительно тестовой выборки
BATCH_SIZE = 2**9

# NOTE: Параметры оптимизатора
LEARNING_RATE = 1e-4

# NOTE: Параметры обучения 
EPOCHS = 10000
EARLY_STOP_THRS = 250