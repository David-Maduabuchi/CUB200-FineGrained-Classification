# CUB-200-2011 Fine-Grained Classification with Swin-B

PyTorch code that hits ~84.5% top-1 on the official test set (single model, no tricks).

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## Results

| Split | Top-1  | Top-5  |
| ----- | ------ | ------ |
| Test  | 84.48% | 98.38% |

Backbone: Swin-B (pretrained on ImageNet-22k → 1k)  
Training: freeze backbone first 5 epochs → unfreeze last two stages  
Input: 224×224 + bbox crop + normal augments

## Get the data (must do this)

Dataset is ~1.2 GB — do NOT commit images.

Fast ways:

- Kaggle: https://www.kaggle.com/datasets/wenewone/cub2002011  
  Download → unzip to `data/raw/CUB_200_2011/`

- Hugging Face: https://huggingface.co/datasets/tonyassi/cub-200-2011  
  `huggingface-cli download tonyassi/cub-200-2011 --local-dir data/raw/CUB_200_2011`

After unzip you need:

````bash
data/raw/CUB_200_2011/
├── images/
├── images.txt
├── image_class_labels.txt
├── bounding_boxes.txt
└── train_test_split.txt
````

## Install & Run

```bash
pip install -r requirements.txt

# Train (uses configs/base.yaml + configs/swin_b.yaml)
python scripts/train.py
```

## Project Structure

````bash
CUB200-FineGrained-Classification/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
|   ├── raw/                # unzip dataset here
│   ├── metadata/           # annotation txt files (committed)
│   └── processed/          # empty for now
│
├── notebooks/
│   └── 01_cub200_swin_baseline.ipynb
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── vitmix_swin.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── validate.py
│   │   └── metrics.py
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       ├── logging.py
│       └── unfreeze.py
│
├── configs/
│   ├── base.yaml
│   └── swin_b.yaml
│
└── scripts/
    ├── train.py
    └── evaluate.py
````
