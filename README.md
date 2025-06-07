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
