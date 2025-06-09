# Description

MAIN

## Project structure

```markdown
basic/
│
├── data
│ ├── s1
│ ├── s2
│ ├── ...
│ └── s_10
│
├── notebooks
│ ├── 1_data.ipynb
│ ├── 2_conv_exampole.ipynb
│
├── src
│ ├── model.py    # Main model
│ ├── preprocessing.py    # Download data 
│ ├── train.py    # Run training
│ └── utils.py    # Additional functions: data parsing, metrics and etc.
│
├── config.py    # Model hyperparameters
└── README.md
```

## Dependecies

- Python 3.10.12
- PyTorch 2.7.1+cu128
- Numpy 2.2.6    # FIXME
- Optuna ?
- MLFlow ?

Installation:

```python
pip install -r requirements.txt
```

## Dataset description

DB5 - this Ninapro dataset includes sEMG and kinematic data from 10 intact subjects while repeating 52 hand movements plus the rest position.
The dataset is described in detail in the following scientific paper:

[Pizzolato et al., Comparison of six electromyography acquisition setups on hand movement classification tasks, PLOS One, 2017](https://pubmed.ncbi.nlm.nih.gov/29023548/)

There are 7 gestures and 10 subjects in dataset for training.

## Model

### Structure

- **Input**: Window of sEMG signals with shape `(N_CHANNELS, WINDOW_SIZE)`. Check `config,py` to change it.

- **Output**: Gestures probability.

- **Hidden layers**: Convolutional block `ConvBlock` consists of `Conv2d(...) →  BatchNorm2d(...) → PReLU → Dropout2d(...) → MaxPool2d(...)`.

Gerenal structure:

```markdown

sEMG (8 channels) window → ConvBlock1 → ConvBlock2 → Classifier → Gestures probabilities 

FullModel(
  (conv_encoder): ConvEncoder(
    (conv_1): Sequential(
      (0): Conv2d(1, 32, kernel_size=(3, 5), stride=(1, 1), padding=same)
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): PReLU(num_parameters=1)
      (3): Dropout2d(p=0.2, inplace=False)
      (4): MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=0, dilation=1, ceil_mode=False)
    )
    (conv_2): Sequential(
      (0): Conv2d(32, 64, kernel_size=(3, 5), stride=(1, 1), padding=same)
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): PReLU(num_parameters=1)
      (3): Dropout2d(p=0.2, inplace=False)
      (4): MaxPool2d(kernel_size=(1, 3), stride=(1, 3), padding=0, dilation=1, ceil_mode=False)
    )
    (flatten): Flatten(start_dim=1, end_dim=-1)
  )
  (ff_classifier): FFClassifier(
    (ff_classifier): Sequential(
      (0): Linear(in_features=1536, out_features=512, bias=True)
      (1): PReLU(num_parameters=1)
      (2): Linear(in_features=512, out_features=256, bias=True)
      (3): PReLU(num_parameters=1)
      (4): Linear(in_features=256, out_features=128, bias=True)
      (5): PReLU(num_parameters=1)
      (6): Linear(in_features=128, out_features=64, bias=True)
      (7): PReLU(num_parameters=1)
      (8): Linear(in_features=64, out_features=17, bias=True)
    )
  )
)

```

### Learning parameters

- **Optimizer**: Adam(lr=1e-4)

- **Loss**: Weighted CrossEntropyLoss

- **Metrics**: Accuracy, F1, Precision, Recall

- **Batch size**: 512

- **Epochs**: 10000

- **Early stopping threshold**: 500 epochs

## Run

If you want to set up specific parameters for training `config.py` should be changed. After setting up run `train.py`.

## Commits description

```markdown

ADD    # new features

FIX    # bugs

UPDATE    # update current features

REMOVE    # delete code or features

REFACTOR    #change code without functional behavour

DOCS    # change documentation

```
