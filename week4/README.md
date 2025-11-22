# Week 4: Advanced Topics & Real-World Applications

A comprehensive Deep Learning and Machine Learning curriculum covering advanced NLP, generative models, deployment, and best practices.

## 📚 Contents

### Day 22: Natural Language Processing (NLP) Basics
**Location:** `day22_nlp_basics/`

- **Text Preprocessing**: Tokenization, stemming, lemmatization
- **Word Embeddings**: Word2Vec, GloVe implementation
- **Sequence Encoding**: Padding and encoding for deep learning
- **Assignment**: News article multi-class classifier

```bash
cd day22_nlp_basics
python news_classifier.py
```

### Day 23: Advanced NLP - Transformers
**Location:** `day23_transformers/`

- **Attention Mechanism**: Self-attention, multi-head attention theory
- **Transformer Architecture**: Encoder-decoder overview
- **Hugging Face**: Using pre-trained models (BERT, GPT)
- **Assignment**: Fine-tune BERT for question answering

```bash
cd day23_transformers
python bert_qa_finetuning.py
```

### Day 24: Autoencoders & Dimensionality Reduction
**Location:** `day24_autoencoders/`

- **PCA**: Principal Component Analysis for dimensionality reduction
- **Autoencoders**: Basic, denoising, and convolutional
- **VAE**: Variational Autoencoders introduction
- **Assignment**: Image compression and denoising

```bash
cd day24_autoencoders
python autoencoder_image.py
```

### Day 25: Generative Adversarial Networks (GAN)
**Location:** `day25_gans/`

- **GAN Theory**: Generator vs Discriminator
- **Training Dynamics**: Loss functions, mode collapse
- **Implementations**: Basic GAN, DCGAN, Conditional GAN
- **Assignment**: Generate handwritten digits

```bash
cd day25_gans
python mnist_gan.py
```

### Day 26: Model Deployment & MLOps
**Location:** `day26_deployment/`

- **Model Saving**: pickle, joblib, SavedModel, ONNX
- **REST API**: FastAPI implementation
- **Docker**: Containerization basics
- **Assignment**: Deploy model as REST API

```bash
cd day26_deployment
python fastapi_app.py  # Demo mode
python fastapi_app.py serve  # Run server
```

### Day 27: Hyperparameter Tuning & Optimization
**Location:** `day27_hyperparameter_tuning/`

- **Search Methods**: Grid search, random search
- **Bayesian Optimization**: Optuna implementation
- **Learning Curves**: Analysis and diagnosis
- **Assignment**: Optimize model, improve by 5%+

```bash
cd day27_hyperparameter_tuning
python hyperparameter_optimization.py
```

### Day 28: Ethics, Bias, and Explainability
**Location:** `day28_ethics_explainability/`

- **Responsible AI**: Fairness, transparency, accountability
- **Bias Detection**: Disparate impact, equalized odds
- **Explainability**: SHAP, LIME implementations
- **Assignment**: Bias analysis and explainability report

```bash
cd day28_ethics_explainability
python bias_explainability_report.py
```

### Day 29: Capstone Project Planning
**Location:** `day29_capstone_planning/`

- **Project Proposal Template**: Comprehensive planning tool
- **Sample Projects**: NLP and Computer Vision examples
- **Assignment**: Submit detailed project proposal

```bash
cd day29_capstone_planning
python capstone_proposal_template.py
```

### Day 30: Capstone Project Execution
**Location:** `day30_capstone_execution/`

- **End-to-End Pipeline**: Complete sentiment analysis system
- **Model Training**: Baseline and advanced models
- **Evaluation**: Comprehensive metrics and visualization
- **Deployment**: API-ready model

```bash
cd day30_capstone_execution
python capstone_sentiment_analysis.py
```

## 🛠️ Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📦 Requirements

See `requirements.txt` for full list. Key dependencies:
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- scikit-learn >= 1.3.0
- SHAP >= 0.42.0
- LIME >= 0.2.0
- FastAPI >= 0.100.0
- Optuna >= 3.0.0

## 🎯 Learning Objectives

By the end of Week 4, you will be able to:

1. **NLP**: Build text classification pipelines with traditional and transformer-based models
2. **Generative Models**: Understand and implement autoencoders and GANs
3. **Deployment**: Deploy ML models as REST APIs with Docker
4. **Optimization**: Use advanced hyperparameter tuning techniques
5. **Responsible AI**: Analyze bias and create explainable models
6. **End-to-End**: Execute complete ML projects from planning to deployment

## 📊 Output Files

Each day's assignment generates visualizations and reports:
- Confusion matrices
- Training curves
- Feature importance plots
- SHAP/LIME explanations
- Model comparison charts

## 🚀 Quick Start

```python
# Example: Run the capstone project
from day30_capstone_execution.capstone_sentiment_analysis import SentimentAnalysisPipeline

pipeline = SentimentAnalysisPipeline()
models, results = pipeline.run()
```

## 📝 Project Structure

```
week4/
├── README.md
├── requirements.txt
├── day22_nlp_basics/
│   ├── src/
│   │   ├── text_preprocessing.py
│   │   ├── word_embeddings.py
│   │   └── sequence_encoding.py
│   └── news_classifier.py
├── day23_transformers/
│   ├── src/
│   │   └── attention_mechanism.py
│   └── bert_qa_finetuning.py
├── day24_autoencoders/
│   └── autoencoder_image.py
├── day25_gans/
│   └── mnist_gan.py
├── day26_deployment/
│   ├── src/
│   │   └── model_saving.py
│   ├── fastapi_app.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── day27_hyperparameter_tuning/
│   └── hyperparameter_optimization.py
├── day28_ethics_explainability/
│   └── bias_explainability_report.py
├── day29_capstone_planning/
│   └── capstone_proposal_template.py
└── day30_capstone_execution/
    └── capstone_sentiment_analysis.py
```

## 📜 License

This educational material is provided for learning purposes.

## 🤝 Contributing

Feel free to submit issues and pull requests for improvements.
