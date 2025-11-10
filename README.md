# DistilBERT Fine-Tuning for Sentiment Analysis

A comprehensive sentiment analysis project that fine-tunes DistilBERT achieving 92% accuracy, with classical baseline comparison, SHAP-based interpretability, and Gradio UI deployment.

## Features

- Fine-tuned DistilBERT model achieving 92% accuracy
- TF-IDF baseline for performance comparison
- SHAP-based model interpretability
- Interactive Gradio UI for demo deployment
- Weights & Biases integration for experiment tracking
- Comprehensive training notebooks with detailed explanations

## Tech Stack

- **Transformers**: Hugging Face Transformers library
- **PyTorch**: Deep learning framework
- **Weights & Biases**: Experiment tracking and visualization
- **Gradio**: Interactive UI deployment
- **SHAP**: Model interpretability
- **Scikit-learn**: Classical ML baselines

## Project Structure

```
distillbert_fineTune/
├── data/               # Dataset storage
├── models/             # Trained model checkpoints
├── scripts/            # Python scripts for training and inference
├── notebooks/          # Jupyter notebooks for exploration and training
├── reports/            # Generated analysis and reports
├── requirements.txt    # Project dependencies
├── .gitignore         # Git ignore rules
└── README.md          # Project documentation
```

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- GPU access (optional, but recommended for training)
- pip or poetry for package management

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd distillbert_fineTune
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python -c "import torch; import transformers; print('Setup successful!')"
```

### Optional: Weights & Biases Setup

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Login to W&B:
```bash
wandb login
```

## Dataset

- **Name**: IMDb Movie Reviews
- **Source**: Hugging Face Datasets
- **Size**: 50,000 movie reviews
- **Classes**: Binary (Positive/Negative)
- **Split**: 70/15/15 (Train/Validation/Test)

See `notebooks/01_exploratory_data_analysis.ipynb` for detailed EDA.

## Usage

### Exploratory Data Analysis

Run the EDA notebook to understand the dataset:

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

### Training Baseline Model

Train the TF-IDF + Logistic Regression baseline:

```bash
cd scripts
python train_baseline.py
```

Optional arguments:
- `--train_size`: Training set size (default: 20000)
- `--val_size`: Validation set size (default: 5000)
- `--max_features`: Max TF-IDF features (default: 10000)
- `--ngram_range`: N-gram range (default: 1 2)

### Training DistilBERT

DistilBERT training scripts will be added in Task 4.

### Inference

Inference scripts and Gradio demo will be added in subsequent tasks.

## Model Performance

### Baseline Model (TF-IDF + Logistic Regression)
- **Validation Accuracy**: 89.02%
- **Test Accuracy**: 87.78%
- **Precision**: 87.63%
- **Recall**: 87.99%
- **F1 Score**: 87.81%
- **Training Time**: 9.6 seconds

### DistilBERT (Target)
- **Target Accuracy**: 92%
- Training to be completed in Task 4

## Project Status

- [x] Task 1: Project Setup & Environment Configuration
- [x] Task 2: Dataset Acquisition & Exploratory Analysis
- [x] Task 3: Classical Baseline Implementation (TF-IDF + Logistic Regression)
- [ ] Task 4: DistilBERT Fine-tuning
- [ ] Task 5: Model Interpretability
- [ ] Task 6: Gradio Deployment
- [ ] Task 7: Documentation & Reporting

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Hugging Face for the Transformers library
- DistilBERT model by Victor Sanh et al.
- Weights & Biases for experiment tracking tools
