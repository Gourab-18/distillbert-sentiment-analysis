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

## Usage

### Training

Training scripts and notebooks will be added in subsequent tasks.

### Inference

Inference scripts and Gradio demo will be added in subsequent tasks.

## Model Performance

- **DistilBERT**: 92% accuracy
- **TF-IDF Baseline**: TBD

## Project Status

- [x] Task 1: Project Setup & Environment Configuration
- [ ] Task 2: Data Collection & Preprocessing
- [ ] Task 3: TF-IDF Baseline Implementation
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
