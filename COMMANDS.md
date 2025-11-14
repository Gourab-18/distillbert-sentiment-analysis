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

### Option 1: Install Dependencies (Direct Installation)
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python3 -c "import torch; import transformers; print('Setup successful!')"
```

### Option 2: Virtual Environment Setup (Recommended for Stability)
```bash
# Create virtual environment (Python3 3.9.6 recommended)
python33 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install specific PyTorch version (for compatibility)
pip install torch==2.3.1

# Install all other dependencies
pip install -r requirements.txt

# Verify installation
python33 -c "import torch; import transformers; print('Setup successful!')"
```

**Why use virtual environment?**
- Avoids PyTorch compatibility issues (2.8.0 had hanging issues)
- Ensures consistent package versions
- Isolates project dependencies

### Check Hardware Device
```bash
# Navigate to scripts directory
cd scripts

# Check available device (GPU/CPU/MPS)
python33 check_device.py
```

**Output:** Shows available hardware and estimated training time

### Setup Weights & Biases (Optional)
```bash
# Install wandb if not already installed
pip install wandb

# Login to W&B
wandb login
```

---

## Exploratory Data Analysis

### Option 1: Run EDA Python3 Script (Recommended - Faster)
```bash
# Navigate to scripts directory
cd scripts

# Run EDA script
python33 run_eda.py

# Or run from project root
python33 scripts/run_eda.py
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
python33 preprocess.py --split all

# Or from project root
python33 scripts/preprocess.py --split all
```

#### Process Individual Splits
```bash
# Process only training set
python3 preprocess.py --split train

# Process only validation set
python33 preprocess.py --split validation

# Process only test set
python3 preprocess.py --split test
```

#### With Custom Parameters
```bash
# Full customization
python3 preprocess.py \
    --split all \
    --train_size 20000 \
    --val_size 5000 \
    --max_length 512 \
    --model_name distilbert-base-uncased \
    --padding max_length \
    --seed 42

# Example: Use shorter sequences
python3 preprocess.py --split all --max_length 256

# Example: Disable HTML removal
python3 preprocess.py --split all --no-remove-html

# Example: Disable lowercasing
python33 preprocess.py --split all --no-lowercase
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
python3 train_baseline.py

# Or from project root
python3 scripts/train_baseline.py
```

#### With Custom Parameters
```bash
# Full customization
python3 train_baseline.py \
    --train_size 20000 \
    --val_size 5000 \
    --max_features 10000 \
    --ngram_range 1 2 \
    --max_iter 1000 \
    --seed 42

# Example: Use more TF-IDF features
python3 train_baseline.py --max_features 15000

# Example: Use trigrams
python3 train_baseline.py --ngram_range 1 3

# Example: Smaller training set for quick testing
python3 train_baseline.py --train_size 5000 --val_size 1000
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

## DistilBERT Training

### Train DistilBERT for Sentiment Classification

#### Basic Usage (with Virtual Environment - Recommended)
```bash
# Navigate to scripts directory
cd scripts

# Activate virtual environment
source ../venv/bin/activate

# Train with default configuration (without W&B)
python3 train_distilbert.py --config ../configs/distilbert_config.yaml --no-wandb

# Or with W&B tracking
python3 train_distilbert.py --config ../configs/distilbert_config.yaml
```

#### Basic Usage (without Virtual Environment)
```bash
# Navigate to scripts directory
cd scripts

# Train with default configuration
python3 train_distilbert.py --config ../configs/distilbert_config.yaml --no-wandb
```

#### Quick Test Training (for debugging)
```bash
# Use test configuration with smaller dataset
python3 train_distilbert.py --config ../configs/distilbert_config_test.yaml --no-wandb
```

#### Disable Weights & Biases
```bash
# Train without W&B logging
python3 train_distilbert.py --no-wandb
```

#### Available Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | ../configs/distilbert_config.yaml | Path to configuration file |
| `--no-wandb` | False | Disable Weights & Biases logging |

#### Configuration File (`configs/distilbert_config.yaml`)

**Model Settings:**
```yaml
model:
  name: distilbert-base-uncased
  num_labels: 2
  dropout: 0.1
```

**Training Hyperparameters:**
```yaml
training:
  num_epochs: 3
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_steps: 500
  train_batch_size: 16
  eval_batch_size: 32
```

**Optimizer:**
- Type: AdamW
- Betas: [0.9, 0.999]
- Epsilon: 1e-8

**Scheduler:**
- Type: Linear warmup and decay
- Warmup ratio: 0.1

**Output:**
- Best model: `models/distilbert-sentiment/best_model/`
- Checkpoints: `models/distilbert-sentiment/checkpoint-epoch{N}-step{M}/`
- Training results: `models/distilbert-sentiment/training_results.json`
- W&B dashboard: Real-time metrics tracking

**Expected Performance:**
- Target Validation Accuracy: ~92%
- Training Time: ~30-45 minutes (GPU) or ~3-4 hours (CPU)
- Model Size: ~256 MB

**Features:**
- ✓ Automatic GPU/CPU/MPS detection
- ✓ Mixed precision training support (FP16)
- ✓ Gradient accumulation
- ✓ Learning rate warmup and decay
- ✓ Checkpoint saving (best + periodic)
- ✓ Early stopping based on validation accuracy
- ✓ Weights & Biases experiment tracking

---

## Monitor Training Progress

### Option 1: View Epoch & Accuracy Summary (Recommended)
```bash
# Navigate to scripts directory
cd scripts

# View training progress with epochs and accuracies
python3 view_training_output.py

# View all available monitoring commands
python3 view_training_output.py --commands
```

**Output:**
- ✅ Training status (Running/Completed)
- 📊 Epoch-by-epoch summary with accuracies
- 🎯 Target achievement status
- ⏱️  Estimated time remaining
- 💾 Saved model location

### Option 2: Quick Status Check (Bash Script)
```bash
# Run quick status check
bash check_training.sh

# Or with detailed model info
bash view_training_progress.sh
```

**Output:**
- Training process status (running/not running)
- Process details (PID, CPU, memory usage, runtime)
- Model checkpoint status
- Training results (if completed)

### Option 3: Detailed Monitoring (Python3 Script)
```bash
# Single status check
python3 monitor_training.py

# Continuous monitoring (refreshes every 30 seconds)
python3 monitor_training.py --watch
```

**Output:**
- Complete training status
- Process resource usage
- Model directory contents
- Checkpoints saved
- Training results with metrics

### Option 4: Check Training Output Directly
```bash
# Check if training process is running
ps aux | grep train_distilbert.py

# Monitor training log (if redirected to file)
tail -f training.log
```

### Manual Status Verification
```bash
# Check if best model exists
ls -lh ../models/distilbert-sentiment/best_model/

# Check training results
cat ../models/distilbert-sentiment/training_results.json | python3 -m json.tool

# Check all checkpoints
ls -lh ../models/distilbert-sentiment/
```

---

## Model Evaluation

### Load and Evaluate Saved Baseline Model

```python3
# Python3 script to load and use the baseline model
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
cat reports/baseline_metrics.json | python3 -m json.tool

# Or use jq if installed
cat reports/baseline_metrics.json | jq '.'
```

---

## Inference

### Baseline Model Inference (Python3)

```python3
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
python3 -c "
import joblib
model = joblib.load('models/baseline_tfidf_logreg.joblib')
review = 'This movie was amazing!'
prediction = model.predict([review])[0]
print(f'Sentiment: {\"Positive\" if prediction == 1 else \"Negative\"}')
"
```

---

## Task 9: Interactive Demo with Streamlit

### Launch Streamlit App

```bash
# Navigate to project root
cd /Users/gourabnanda/Desktop/ML/distillbert_fineTune

# Activate virtual environment
source venv/bin/activate

# Launch Streamlit app
streamlit run app.py

# Or with custom port
streamlit run app.py --server.port 8501
```

**Access the app:**
- **Local URL:** http://localhost:8501
- **Network URL:** Available on your local network
- **External URL:** Available if firewall allows

### Features

The Streamlit app provides:
- ✅ Real-time sentiment prediction with confidence scores
- ✅ Token-level SHAP explanations showing word importance
- ✅ Interactive visualization of model decisions
- ✅ 5 example inputs for quick testing
- ✅ Clean, professional UI with sidebar
- ✅ Model caching for fast performance
- ✅ Probability distribution display
- ✅ Color-coded SHAP visualizations (Red = Positive, Blue = Negative)

### How to Use

1. Open the app in your browser (automatically opens)
2. Enter text in the text area or click an example from the sidebar
3. Click "🔍 Analyze Sentiment" button
4. View the sentiment prediction, confidence score, and SHAP explanation

### Outputs

The app displays:
- **Sentiment Label:** Positive 😊 or Negative 😞
- **Confidence Score:** Percentage confidence of prediction
- **Probability Distribution:** Visual bars showing positive/negative probabilities
- **SHAP Explanation:** Color-coded token importance visualization
  - Red tokens = push toward Positive sentiment
  - Blue tokens = push toward Negative sentiment
  - Intensity = strength of influence

### Example Inputs Available

1. Positive movie review with praise
2. Negative service complaint
3. Neutral product review
4. Enthusiastic recommendation
5. Strong disappointment

### Performance

- **Model Loading:** ~5-10 seconds (cached after first run)
- **Prediction Time:** <1 second per analysis
- **SHAP Generation:** 1-2 seconds per explanation
- **Total Latency:** ~2-3 seconds for complete analysis

### Stopping the App

```bash
# Press Ctrl+C in the terminal to stop the Streamlit server
```

### Task 9 Results (COMPLETED ✅)

- **Framework:** Streamlit 1.50.0
- **Model:** DistilBERT (92.65% test accuracy)
- **SHAP Integration:** Token-level interpretability
- **Interface:** Clean, responsive web UI
- **Examples:** 5 pre-loaded examples
- **Caching:** Enabled for fast performance

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
python3 scripts/train_baseline.py
```

### Issue: Out of memory during training
```bash
# Reduce training size
python3 scripts/train_baseline.py --train_size 10000 --val_size 2000
```

### Issue: pyarrow version conflict
```bash
# Reinstall compatible versions
pip install --upgrade 'pyarrow>=21.0.0' 'datasets>=4.4.0' --force-reinstall
```

### Issue: CUDA out of memory (DistilBERT training)
```bash
# Reduce batch size in config file
# Edit configs/distilbert_config.yaml:
#   train_batch_size: 8  (reduce from 16)
#   eval_batch_size: 16  (reduce from 32)

# Or enable gradient accumulation
#   gradient_accumulation_steps: 2
```

### Issue: W&B login required
```bash
# Login to Weights & Biases
wandb login

# Or disable W&B
python3 train_distilbert.py --no-wandb
```

---

## Quick Reference: All Commands

```bash
# 1. Setup (with virtual environment - recommended)
python33 -m venv venv
source venv/bin/activate
pip install torch==2.3.1
pip install -r requirements.txt

# 2. Check hardware device
cd scripts
python3 check_device.py

# 3. Run EDA
python3 run_eda.py

# 4. Preprocess and tokenize data
python3 preprocess.py --split all

# 5. Train baseline model
python3 train_baseline.py

# 6. Train DistilBERT
source ../venv/bin/activate
python3 train_distilbert.py --config ../configs/distilbert_config.yaml --no-wandb

# 7. Monitor training progress (choose one)
python3 view_training_output.py         # Best: Shows epochs & accuracies
bash check_training.sh                  # Quick status check
python3 monitor_training.py --watch      # Continuous monitoring

# 8. View results
cat ../reports/baseline_metrics.json | python3 -m json.tool
cat ../models/distilbert-sentiment/training_results.json | python3 -m json.tool

# 9. View preprocessing config
cat ../data/processed/preprocessing_config.json | python3 -m json.tool

# 10. Check model sizes
ls -lh ../models/

# 11. View all reports
ls -lh ../reports/

# 12. Launch Streamlit demo app
streamlit run app.py
```

---

## Project Structure Reference

```
distillbert_fineTune/
├── venv/                       # Virtual environment (recommended)
├── scripts/
│   ├── run_eda.py                 # EDA automation script
│   ├── preprocess.py              # Data preprocessing & tokenization
│   ├── train_baseline.py          # Baseline model training
│   ├── train_distilbert.py        # DistilBERT training
│   ├── check_device.py            # Hardware device checker
│   ├── view_training_output.py    # View epoch & accuracy summary (Recommended)
│   ├── monitor_training.py        # Training progress monitor (Python3)
│   ├── check_training.sh          # Quick status check (Bash)
│   └── view_training_progress.sh  # Detailed status check (Bash)
├── configs/
│   ├── distilbert_config.yaml      # Full training configuration
│   └── distilbert_config_test.yaml # Quick test configuration
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── models/
│   ├── baseline_tfidf_logreg.joblib
│   └── distilbert-sentiment/   # DistilBERT checkpoints
│       ├── best_model/         # Best model saved
│       ├── checkpoint-*/       # Training checkpoints
│       └── training_results.json
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

## Task 7: Comprehensive Model Evaluation & Performance Comparison

### Run Full Evaluation on Test Set

```bash
cd scripts
source ../venv/bin/activate

# Run comprehensive evaluation (takes ~20-30 minutes for 25K samples)
python evaluate.py
```

**What it does:**
- Evaluates both Baseline and DistilBERT on full test set (25,000 samples)
- Generates comprehensive metrics (Accuracy, Precision, Recall, F1, ROC AUC)
- Creates confusion matrix comparison
- Performs detailed error analysis
- Generates 3 visualization plots

**Outputs:**
- `reports/evaluation_metrics.json` - Complete metrics data
- `reports/evaluation_summary.md` - Human-readable summary
- `reports/confusion_matrix_comparison.png` - Side-by-side confusion matrices
- `reports/performance_comparison.png` - Bar chart of all metrics
- `reports/per_class_metrics.png` - Per-class precision/recall/F1

### View Evaluation Results

```bash
# View summary report
cat ../reports/evaluation_summary.md

# View JSON results
python -m json.tool ../reports/evaluation_metrics.json | less

# Open visualizations
open ../reports/confusion_matrix_comparison.png
open ../reports/performance_comparison.png
open ../reports/per_class_metrics.png
```

### Task 7 Results (COMPLETED ✅)

- **DistilBERT Test Accuracy:** 92.65% 🎉 (Target ≥92% ACHIEVED!)
- **Baseline Test Accuracy:** 87.78%
- **Improvement:** +4.87 percentage points
- **Error Reduction:** 1,217 samples (39.9% fewer errors)
- **DistilBERT Errors:** 1,837 (7.35%)
- **Baseline Errors:** 3,054 (12.22%)

**Report:** See `reports/evaluation_summary.md` for full analysis

---

## Task 8: SHAP-Based Model Interpretability

### Generate SHAP Explanations

```bash
cd scripts
source ../venv/bin/activate

# Generate explanations for default 30 examples
python explain.py

# Generate for fewer examples (faster)
python explain.py --num-examples 5

# Generate for more examples
python explain.py --num-examples 50

# Explain a specific text
python explain.py --text "This movie was absolutely amazing! Great acting and wonderful plot."
```

**What it does:**
- Generates SHAP (SHapley Additive exPlanations) for model predictions
- Shows token-level importance scores
- Creates 3 types of visualizations per example
- Provides interpretability for why the model made its prediction

**Outputs:**
- `reports/explanations/force_plots/` - Interactive HTML force plots
- `reports/explanations/waterfall_plots/` - Waterfall PNG charts
- `reports/explanations/text_plots/` - Color-coded text PNG images
- `reports/explanations/explanations_metadata.json` - All explanation data
- `reports/explanations/explanations_summary.md` - Human-readable summary

### View SHAP Explanations

```bash
cd reports/explanations

# Open folder in Finder
open .

# Open specific force plot (interactive HTML)
open force_plots/example_000_idx23921.html

# Open waterfall plot (PNG)
open waterfall_plots/example_000_idx23921.png

# Open text plot (PNG)
open text_plots/example_000_idx23921.png

# View summary
cat explanations_summary.md

# View summary plots
open category_distribution.png
open confidence_distribution.png
```

### Use Interactive Viewer Script

```bash
cd scripts

# Run interactive menu to view explanations
./view_explanations.sh
```

**Menu options:**
1. Open explanations folder in Finder
2. Open first 5 force plots in browser
3. Open first 5 waterfall plots
4. Open first 5 text plots
5. Open all summary plots
6. Open summary report
7. List all files
8. Open specific example by number

### Programmatic Usage (Python API)

```python
from explain import SentimentExplainer

# Initialize explainer
explainer = SentimentExplainer()

# Explain a new text
result = explainer.predict_with_explanation("This movie was fantastic!")

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Access SHAP values
shap_values = result['shap_values']

# Save visualizations
explainer.save_force_plot(shap_values, "my_force_plot.html")
explainer.save_waterfall_plot(shap_values, "my_waterfall.png")
```

### Understanding SHAP Visualizations

**Force Plots (HTML):**
- Red text = pushes toward Positive sentiment
- Blue text = pushes toward Negative sentiment
- Hover over words to see exact SHAP values

**Waterfall Plots (PNG):**
- Shows cumulative effect of top tokens
- Bars show individual token contributions
- Easy to see which words had biggest impact

**Text Plots (PNG):**
- Color-coded inline text
- Red/pink = positive contribution
- Blue/cyan = negative contribution
- Intensity = strength of contribution

### Task 8 Results (COMPLETED ✅)

- **Examples Explained:** 5 diverse examples
- **Force Plots:** 5 interactive HTML visualizations
- **Waterfall Plots:** 5 PNG charts
- **Text Plots:** 5 PNG color-coded visualizations
- **Summary Plots:** 3 aggregate visualizations
- **API:** Programmatic interface available for new inputs

**Documentation:** See `reports/task8_overview.md` for detailed guide

---

## Performance Benchmarks

### Current Results (Task 6 COMPLETED ✅)

| Model | Validation Acc | Test Acc | Training Time | Model Size |
|-------|---------------|----------|---------------|------------|
| TF-IDF + LogReg | 89.02% | 87.78% | 9.6s | 446 KB |
| **DistilBERT** | **92.72%** 🎉 | **92.78%** 🎉 | ~2.5 hours | 268 MB |

**DistilBERT Final Results:**
- ✅ **Test Accuracy: 92.78%** (Target: 92%+ ACHIEVED!)
- ✅ **Validation Accuracy: 92.72%**
- ✅ **Test Loss: 0.2732**
- ✅ **Improvement over Baseline: +5.0 percentage points**
- ✅ All 3 epochs completed successfully
- Hardware: MPS (Apple Silicon GPU)
- Speed: ~2.5 seconds/batch
- Total Training Time: ~2.5 hours (3 epochs)

### Data Processing Statistics

| Split | Samples | Mean Tokens | Max Tokens | Size |
|-------|---------|-------------|------------|------|
| Train | 20,000 | 265 | 512 | 49 MB |
| Validation | 5,000 | 263 | 512 | 12 MB |
| Test | 25,000 | 261 | 512 | 61 MB |

### Hardware Tested

| Device | Status | Training Speed | Notes |
|--------|--------|----------------|-------|
| MPS (Apple Silicon) | ✅ Working | ~2.5s/batch | Recommended for Mac M1/M2/M3 |
| CPU | ✅ Working | ~5-10s/batch | Slower but stable |
| CUDA GPU | Not Tested | ~1-2s/batch | Expected performance |

---

## Next Steps

- [x] Task 1: Project Setup & Environment Configuration
- [x] Task 2: Dataset Acquisition & Exploratory Analysis
- [x] Task 3: Classical Baseline Implementation
- [x] Task 4: Data Preprocessing & Tokenization Pipeline
- [x] Task 5: DistilBERT Model Architecture & Training Setup
- [x] **Task 6: Model Training Execution & Evaluation** ✅ **COMPLETED!**
  - Test Accuracy: **92.78%** 🎉
  - Validation Accuracy: **92.72%** 🎉
  - Report: `reports/task6_completion_report.md`
- [x] **Task 7: Comprehensive Model Evaluation & Performance Comparison** ✅ **COMPLETED!**
  - Test Accuracy: **92.65%** 🎉 (Target ≥92% ACHIEVED!)
  - Improvement over Baseline: **+4.87 pp**
  - Error Reduction: **1,217 samples** (39.9% fewer errors)
  - Report: `reports/evaluation_summary.md`
- [x] **Task 8: SHAP-Based Model Interpretability** ✅ **COMPLETED!**
  - **5 diverse examples** explained with token-level importance
  - **15 visualizations** (force, waterfall, text plots)
  - **Programmatic API** available for new inputs
  - Documentation: `reports/task8_overview.md`
- [x] **Task 9: Interactive Demo with Streamlit** ✅ **COMPLETED!**
  - **Streamlit web app** with real-time predictions
  - **SHAP visualizations** integrated into UI
  - **5 example inputs** for quick testing
  - **Model caching** for fast performance
  - Launch with: `streamlit run app.py`

---

## Troubleshooting Guide

### PyTorch Compatibility Issues

**Problem:** Training hangs during model initialization or import takes forever

**Solution:**
```bash
# Use virtual environment with PyTorch 2.3.1
python33 -m venv venv
source venv/bin/activate
pip install torch==2.3.1
pip install -r requirements.txt
```

### Training Not Running

**Problem:** Training script appears stuck or not producing output

**Check:**
```bash
# Verify training process is running
ps aux | grep train_distilbert.py

# Check CPU/Memory usage
top -pid <PID>

# Use monitoring scripts
bash check_training.sh
```

### MPS (Apple Silicon) Not Detected

**Problem:** Training uses CPU instead of MPS GPU

**Check:**
```bash
# Verify MPS availability
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Make sure config has auto_detect: true
cat ../configs/distilbert_config.yaml | grep auto_detect
```

---

**Last Updated:** Tasks 6, 7, 8, 9 COMPLETED ✅
**Project:** DistilBERT Fine-Tuning for Sentiment Analysis

**Key Results:**
- **Task 6 (Training):** 92.78% test accuracy (Target: 92%+ ACHIEVED!)
- **Task 7 (Evaluation):** Comprehensive analysis with 92.65% test accuracy, +4.87pp improvement over baseline
- **Task 8 (Interpretability):** SHAP explanations with 5 examples, 15 visualizations, programmatic API
- **Task 9 (Interactive Demo):** Streamlit web app with real-time predictions and SHAP visualizations

**Reports:**
- Task 6: `reports/task6_completion_report.md`
- Task 7: `reports/evaluation_summary.md`
- Task 8: `reports/task8_overview.md`
- Task 9: Launch with `streamlit run app.py`
