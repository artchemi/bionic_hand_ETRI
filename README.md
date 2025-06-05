# Project structure

```markdown
basic/
│
├── src
│ ├── model.py    # Main model
│ ├── preprocessing.py    # Download data 
│ ├── utils.py    # Additional functions: data parsing, metrics and etc.
│ └── train-labels.idx1-ubyte
│
├── config.py    # Model hyperparameters
└── README.md
```

# Description

Main.

## Dataset description

DB5 - this Ninapro dataset includes sEMG and kinematic data from 10 intact subjects while repeating 52 hand movements plus the rest position.
The dataset is described in detail in the following scientific paper:

[Pizzolato et al., Comparison of six electromyography acquisition setups on hand movement classification tasks, PLOS One, 2017](https://pubmed.ncbi.nlm.nih.gov/29023548/)

There are 7 gestures and 10 subjects in dataset for training.

## Commits description

Add - new features

Fix - bugs

Update - update current features

Remove - delete code or features

Refactor - change code without functional behavour

Docs - change documentation
