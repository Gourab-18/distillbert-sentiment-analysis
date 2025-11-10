# Data Directory

This directory contains the datasets used for the sentiment analysis project.

## Structure

```
data/
├── raw/              # Raw dataset files
├── processed/        # Processed and cleaned datasets
└── README.md         # This file
```

## Dataset Information

### IMDb Movie Reviews

- **Source**: Hugging Face Datasets (`imdb`)
- **Task**: Binary sentiment classification
- **Size**: 50,000 movie reviews
  - Training: 25,000 reviews
  - Test: 25,000 reviews
- **Classes**:
  - 0: Negative sentiment
  - 1: Positive sentiment
- **Balance**: Perfectly balanced (50/50 split)

## Data Split Strategy

For model training, we use the following split:

- **Training**: 20,000 samples (40% of total, 80% of original train)
- **Validation**: 5,000 samples (10% of total, 20% of original train)
- **Test**: 25,000 samples (50% of total, kept separate)

This provides an approximate **70/15/15** split across train/validation/test.

## Data Characteristics

- **Average review length**: ~230 words (~1,300 characters)
- **Length range**: Highly variable (from very short to very long reviews)
- **Vocabulary**: Rich and diverse across both sentiment classes
- **Special considerations**:
  - Some reviews exceed 512 tokens (model max sequence length)
  - May contain HTML tags and special characters
  - Truncation will be applied during preprocessing

## Loading the Dataset

The dataset is automatically downloaded from Hugging Face when running the notebooks or training scripts:

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
train_data = dataset['train']
test_data = dataset['test']
```

## Notes

- Raw data files are not committed to the repository (see `.gitignore`)
- The dataset will be automatically downloaded during first run
- Processed data will be saved in the `processed/` directory
