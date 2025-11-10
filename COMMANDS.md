# Command Reference Guide

This document contains all the commands needed to run the scripts in this project.

## Table of Contents
- [Setup](#setup)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Baseline Model Training](#baseline-model-training)
- [DistilBERT Training](#distilbert-training-coming-soon)
- [Model Evaluation](#model-evaluation)
- [Inference](#inference)

---

## Setup

### Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import torch; import transformers; print('Setup successful!')"
```

### Setup Weights & Biases (Optional)
```bash
# Install wandb if not already installed
pip install wandb

# Login to W&B
wandb login
```

---

## Exploratory Data Analysis

### Option 1: Run EDA Python Script (Recommended - Faster)
```bash
# Navigate to scripts directory
cd scripts

# Run EDA script
python run_eda.py

# Or run from project root
python scripts/run_eda.py
```

**Output:**
- `reports/class_distribution.png`
- `reports/text_length_analysis.png`
- `reports/wordclouds.png`
- `reports/top_words.png`
- `reports/data_split.png`
- `reports/data_statistics_summary.json`

### Option 2: Run EDA Jupyter Notebook (Interactive)
```bash
# From project root
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb

# Or open in VS Code and run all cells
```

---

## Data Preprocessing & Tokenization

### Preprocess and Tokenize for DistilBERT

#### Basic Usage
```bash
# Navigate to scripts directory
cd scripts

# Process all splits (train, validation, test)
python preprocess.py --split all

# Or from project root
python scripts/preprocess.py --split all
```

#### Process Individual Splits
```bash
# Process only training set
python preprocess.py --split train

# Process only validation set
python preprocess.py --split validation

# Process only test set
python preprocess.py --split test
```

#### With Custom Parameters
```bash
# Full customization
python preprocess.py \
    --split all \
    --train_size 20000 \
    --val_size 5000 \
    --max_length 512 \
    --model_name distilbert-base-uncased \
    --padding max_length \
    --seed 42

# Example: Use shorter sequences
python preprocess.py --split all --max_length 256

# Example: Disable HTML removal
python preprocess.py --split all --no-remove-html

# Example: Disable lowercasing
python preprocess.py --split all --no-lowercase
```

#### Available Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--split` | all | Which split to process (all/train/validation/test) |
| `--train_size` | 20000 | Training set size |
| `--val_size` | 5000 | Validation set size |
| `--max_length` | 512 | Maximum sequence length (DistilBERT limit) |
| `--model_name` | distilbert-base-uncased | DistilBERT model name |
| `--padding` | max_length | Padding strategy |
| `--lowercase` | True | Convert to lowercase |
| `--remove_html` | True | Remove HTML tags |
| `--remove_urls` | True | Remove URLs |
| `--remove_special_chars` | False | Remove special characters |
| `--seed` | 42 | Random seed |

**Output:**
- Tokenized datasets: `data/processed/tokenized_distilbert-base-uncased/`
  - `train/`: 20,000 samples (~49 MB)
  - `validation/`: 5,000 samples (~12 MB)
  - `test/`: 25,000 samples (~61 MB)
- Config file: `data/processed/preprocessing_config.json`
- Token distribution plot: `reports/token_length_distribution.png`

**Token Statistics:**
- Training: Mean=265 tokens, Max=512
- Validation: Mean=263 tokens, Max=512
- Test: Mean=261 tokens, Max=512
- All sequences within 512 token limit ✓

---

## Baseline Model Training

### Train TF-IDF + Logistic Regression Baseline

#### Basic Usage
```bash
# Navigate to scripts directory
cd scripts

# Run with default parameters
python train_baseline.py

# Or from project root
python scripts/train_baseline.py
```

#### With Custom Parameters
```bash
# Full customization
python train_baseline.py \
    --train_size 20000 \
    --val_size 5000 \
    --max_features 10000 \
    --ngram_range 1 2 \
    --max_iter 1000 \
    --seed 42

# Example: Use more TF-IDF features
python train_baseline.py --max_features 15000

# Example: Use trigrams
python train_baseline.py --ngram_range 1 3

# Example: Smaller training set for quick testing
python train_baseline.py --train_size 5000 --val_size 1000
```

#### Available Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--train_size` | 20000 | Number of samples for training |
| `--val_size` | 5000 | Number of samples for validation |
| `--max_features` | 10000 | Maximum number of TF-IDF features |
| `--ngram_range` | 1 2 | N-gram range (unigrams and bigrams) |
| `--max_iter` | 1000 | Maximum iterations for Logistic Regression |
| `--seed` | 42 | Random seed for reproducibility |

**Output:**
- `models/baseline_tfidf_logreg.joblib` - Trained model
- `reports/baseline_metrics.json` - Performance metrics
- `reports/baseline_confusion_matrix_val.png` - Validation confusion matrix
- `reports/baseline_confusion_matrix_test.png` - Test confusion matrix

**Expected Performance:**
- Validation Accuracy: ~89%
- Test Accuracy: ~88%
- Training Time: ~10 seconds

---

## DistilBERT Training (Coming Soon)

This section will be updated in Task 4.

```bash
# Placeholder for DistilBERT training commands
# python train_distilbert.py
```

---

## Model Evaluation

### Load and Evaluate Saved Baseline Model

```python
# Python script to load and use the baseline model
import joblib
from datasets import load_dataset

# Load model
model = joblib.load('models/baseline_tfidf_logreg.joblib')

# Load test data
dataset = load_dataset("imdb")
test_texts = dataset['test']['text'][:10]

# Make predictions
predictions = model.predict(test_texts)
print(predictions)  # 0 = Negative, 1 = Positive
```

### View Saved Metrics
```bash
# View baseline metrics
cat reports/baseline_metrics.json | python -m json.tool

# Or use jq if installed
cat reports/baseline_metrics.json | jq '.'
```

---

## Inference

### Baseline Model Inference (Python)

```python
import joblib

# Load trained baseline model
model = joblib.load('models/baseline_tfidf_logreg.joblib')

# Sample reviews
reviews = [
    "This movie was absolutely terrible! Waste of time.",
    "Amazing film! Best movie I've seen this year!",
    "It was okay, nothing special but not bad either."
]

# Predict sentiment
predictions = model.predict(reviews)
probabilities = model.predict_proba(reviews)

# Display results
for review, pred, prob in zip(reviews, predictions, probabilities):
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = prob[pred] * 100
    print(f"Review: {review[:50]}...")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f}%)")
    print("-" * 60)
```

### Quick Command-Line Inference
```bash
# Create a simple inference script
python -c "
import joblib
model = joblib.load('models/baseline_tfidf_logreg.joblib')
review = 'This movie was amazing!'
prediction = model.predict([review])[0]
print(f'Sentiment: {\"Positive\" if prediction == 1 else \"Negative\"}')
"
```

---

## Gradio UI (Coming Soon)

This section will be updated in Task 6.

```bash
# Placeholder for Gradio demo
# python app.py
```

---

## Common Issues and Solutions

### Issue: Module not found
```bash
# Make sure you're in the correct directory
pwd

# Install missing dependencies
pip install -r requirements.txt
```

### Issue: Dataset download fails
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/datasets/imdb
python scripts/train_baseline.py
```

### Issue: Out of memory during training
```bash
# Reduce training size
python scripts/train_baseline.py --train_size 10000 --val_size 2000
```

### Issue: pyarrow version conflict
```bash
# Reinstall compatible versions
pip install --upgrade 'pyarrow>=21.0.0' 'datasets>=4.4.0' --force-reinstall
```

---

## Quick Reference: All Commands

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Run EDA
python scripts/run_eda.py

# 3. Preprocess and tokenize data
python scripts/preprocess.py --split all

# 4. Train baseline model
python scripts/train_baseline.py

# 5. View results
cat reports/baseline_metrics.json | python -m json.tool

# 6. View preprocessing config
cat data/processed/preprocessing_config.json | python -m json.tool

# 7. Check model size
ls -lh models/

# 8. View all reports
ls -lh reports/
```

---

## Project Structure Reference

```
distillbert_fineTune/
├── scripts/
│   ├── run_eda.py              # EDA automation script
│   ├── preprocess.py           # Data preprocessing & tokenization
│   └── train_baseline.py       # Baseline model training
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── models/
│   └── baseline_tfidf_logreg.joblib
├── reports/
│   ├── baseline_metrics.json
│   ├── baseline_confusion_matrix_val.png
│   ├── baseline_confusion_matrix_test.png
│   ├── class_distribution.png
│   ├── text_length_analysis.png
│   ├── token_length_distribution.png
│   ├── wordclouds.png
│   ├── top_words.png
│   └── data_split.png
├── data/
│   ├── raw/                    # Raw data (auto-downloaded)
│   └── processed/              # Processed & tokenized data
│       ├── tokenized_distilbert-base-uncased/
│       └── preprocessing_config.json
├── requirements.txt
├── COMMANDS.md
└── README.md
```

---

## Performance Benchmarks

### Current Results (Task 4 Complete)

| Model | Validation Acc | Test Acc | Training Time | Model Size |
|-------|---------------|----------|---------------|------------|
| TF-IDF + LogReg | 89.02% | 87.78% | 9.6s | 446 KB |
| DistilBERT | TBD | TBD | TBD | TBD |

### Data Processing Statistics

| Split | Samples | Mean Tokens | Max Tokens | Size |
|-------|---------|-------------|------------|------|
| Train | 20,000 | 265 | 512 | 49 MB |
| Validation | 5,000 | 263 | 512 | 12 MB |
| Test | 25,000 | 261 | 512 | 61 MB |

---

## Next Steps

- [x] Task 1: Project Setup & Environment Configuration
- [x] Task 2: Dataset Acquisition & Exploratory Analysis
- [x] Task 3: Classical Baseline Implementation
- [x] Task 4: Data Preprocessing & Tokenization Pipeline
- [ ] Task 5: DistilBERT Fine-tuning
- [ ] Task 6: Model Interpretability (SHAP)
- [ ] Task 7: Gradio Deployment

---

**Last Updated:** Task 4 Complete
**Project:** DistilBERT Fine-Tuning for Sentiment Analysis
