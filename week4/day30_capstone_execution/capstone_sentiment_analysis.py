"""
Day 30: Capstone Project Execution
Complete End-to-End Sentiment Analysis System
Combining all skills from Week 4
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import re

# NLP
import nltk
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Deep Learning
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Transformers
try:
    from transformers import (
        DistilBertTokenizer, DistilBertForSequenceClassification,
        Trainer, TrainingArguments, pipeline
    )
    from datasets import Dataset as HFDataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False


# ==================== DATA MODULE ====================

class SentimentDataGenerator:
    """
    Generate or load sentiment analysis dataset
    """

    @staticmethod
    def generate_synthetic_data(n_samples=5000):
        """Generate synthetic sentiment data for demonstration"""
        np.random.seed(42)

        positive_templates = [
            "I absolutely love this product! It's amazing.",
            "Great experience, would highly recommend.",
            "This exceeded my expectations. Very satisfied!",
            "Best purchase I've ever made. Fantastic quality.",
            "Wonderful service and excellent results.",
            "I'm so happy with this. Works perfectly!",
            "Outstanding performance. Totally worth it.",
            "Incredible value for money. Love it!",
            "This made my day. Absolutely brilliant!",
            "Five stars! Couldn't be happier."
        ]

        negative_templates = [
            "Terrible product, complete waste of money.",
            "Very disappointed with the quality.",
            "Does not work as advertised. Avoid!",
            "Worst purchase ever. Returning immediately.",
            "Poor customer service, never buying again.",
            "Broke after one day. Total garbage.",
            "Not worth the price at all.",
            "Extremely frustrated with this purchase.",
            "Save your money, this is awful.",
            "One star is too generous for this junk."
        ]

        neutral_templates = [
            "It's okay, nothing special.",
            "Average product, does the job.",
            "Neither good nor bad. It's fine.",
            "Meets basic expectations.",
            "Standard quality, as expected.",
            "It works but nothing impressive.",
            "Decent product for the price.",
            "No complaints but no praise either.",
            "Middle of the road experience.",
            "It's acceptable, I suppose."
        ]

        modifiers = [
            "", "Really ", "Very ", "Quite ", "Somewhat ", "Definitely ",
            "Absolutely ", "Totally ", "Honestly, ", "In my opinion, "
        ]

        data = []

        # Generate positive samples
        for _ in range(n_samples // 3):
            template = np.random.choice(positive_templates)
            modifier = np.random.choice(modifiers)
            text = modifier + template
            if np.random.random() > 0.7:
                text += " " + np.random.choice(positive_templates).split('.')[-2] + "."
            data.append({"text": text, "sentiment": "positive"})

        # Generate negative samples
        for _ in range(n_samples // 3):
            template = np.random.choice(negative_templates)
            modifier = np.random.choice(modifiers)
            text = modifier + template
            if np.random.random() > 0.7:
                text += " " + np.random.choice(negative_templates).split('.')[-2] + "."
            data.append({"text": text, "sentiment": "negative"})

        # Generate neutral samples
        for _ in range(n_samples - 2 * (n_samples // 3)):
            template = np.random.choice(neutral_templates)
            modifier = np.random.choice(modifiers)
            text = modifier + template
            data.append({"text": text, "sentiment": "neutral"})

        df = pd.DataFrame(data)
        return df.sample(frac=1, random_state=42).reset_index(drop=True)


class TextPreprocessor:
    """
    Text preprocessing pipeline
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()

    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()

        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        return ' '.join(tokens)

    def preprocess_batch(self, texts):
        """Preprocess batch of texts"""
        return [self.clean_text(text) for text in texts]


# ==================== MODEL MODULES ====================

class BaselineModel:
    """
    TF-IDF + Logistic Regression baseline
    """

    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2
        )
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.label_encoder = LabelEncoder()

    def fit(self, X_train, y_train):
        """Train the model"""
        X_vec = self.vectorizer.fit_transform(X_train)
        y_enc = self.label_encoder.fit_transform(y_train)
        self.classifier.fit(X_vec, y_enc)
        return self

    def predict(self, X):
        """Make predictions"""
        X_vec = self.vectorizer.transform(X)
        y_pred = self.classifier.predict(X_vec)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_vec = self.vectorizer.transform(X)
        return self.classifier.predict_proba(X_vec)

    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        y_pred = self.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'classification_report': classification_report(y_test, y_pred)
        }

    def save(self, path):
        """Save model"""
        joblib.dump({
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder
        }, path)

    def load(self, path):
        """Load model"""
        data = joblib.load(path)
        self.vectorizer = data['vectorizer']
        self.classifier = data['classifier']
        self.label_encoder = data['label_encoder']


if TORCH_AVAILABLE:
    class LSTMSentimentModel(nn.Module):
        """
        LSTM-based sentiment classifier
        """

        def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                     num_classes=3, dropout=0.3):
            super(LSTMSentimentModel, self).__init__()

            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                                bidirectional=True, dropout=dropout)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, (hidden, _) = self.lstm(embedded)

            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            hidden = self.dropout(hidden)
            output = self.fc(hidden)

            return output


class TransformerModel:
    """
    DistilBERT-based sentiment classifier
    """

    def __init__(self, model_name='distilbert-base-uncased', num_labels=3):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required")

        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_pretrained(self):
        """Load pre-trained model"""
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        self.model.to(self.device)

    def prepare_dataset(self, texts, labels=None, max_length=128):
        """Prepare dataset for training/inference"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )

        if labels is not None:
            return HFDataset.from_dict({
                'input_ids': encodings['input_ids'].tolist(),
                'attention_mask': encodings['attention_mask'].tolist(),
                'labels': labels
            })
        return encodings

    def train(self, train_texts, train_labels, val_texts=None, val_labels=None,
              epochs=3, batch_size=16, output_dir='./transformer_model'):
        """Train the model"""
        self.load_pretrained()

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels)

        eval_dataset = None
        if val_texts is not None and val_labels is not None:
            eval_dataset = self.prepare_dataset(val_texts, val_labels)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        print("Training transformer model...")
        trainer.train()

        return trainer

    def predict(self, texts):
        """Make predictions"""
        self.model.eval()
        encodings = self.prepare_dataset(texts)

        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in encodings.items()}
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        return predictions.cpu().numpy()


# ==================== EVALUATION MODULE ====================

class ModelEvaluator:
    """
    Comprehensive model evaluation
    """

    def __init__(self, model, model_name, label_names=None):
        self.model = model
        self.model_name = model_name
        self.label_names = label_names or ['negative', 'neutral', 'positive']

    def evaluate(self, X_test, y_test):
        """Full evaluation"""
        y_pred = self.model.predict(X_test)

        results = {
            'model_name': self.model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted')
        }

        return results

    def plot_confusion_matrix(self, X_test, y_test, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_names, yticklabels=self.label_names)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

    def generate_report(self, X_test, y_test):
        """Generate evaluation report"""
        y_pred = self.model.predict(X_test)

        print("\n" + "=" * 50)
        print(f"EVALUATION REPORT: {self.model_name}")
        print("=" * 50)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_names))


# ==================== EXPLAINABILITY MODULE ====================

class ModelExplainer:
    """
    Model explainability with SHAP
    """

    def __init__(self, model, vectorizer=None, feature_names=None):
        self.model = model
        self.vectorizer = vectorizer
        self.feature_names = feature_names

    def explain_prediction(self, text, preprocessed_text):
        """Explain a single prediction"""
        prediction = self.model.predict([preprocessed_text])[0]
        proba = self.model.predict_proba([preprocessed_text])[0]

        print(f"\nText: {text[:100]}...")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {max(proba):.2%}")
        print(f"Probabilities: {dict(zip(self.model.label_encoder.classes_, proba))}")

        # Get feature importance for this prediction
        if self.vectorizer:
            X_vec = self.vectorizer.transform([preprocessed_text])
            feature_names = self.vectorizer.get_feature_names_out()

            # Get non-zero features
            nonzero_idx = X_vec.nonzero()[1]
            feature_weights = []

            if hasattr(self.model.classifier, 'coef_'):
                for idx in nonzero_idx:
                    word = feature_names[idx]
                    tfidf_value = X_vec[0, idx]
                    # Get coefficient for predicted class
                    pred_idx = list(self.model.label_encoder.classes_).index(prediction)
                    if len(self.model.classifier.coef_.shape) > 1:
                        coef = self.model.classifier.coef_[pred_idx, idx]
                    else:
                        coef = self.model.classifier.coef_[idx]
                    weight = tfidf_value * coef
                    feature_weights.append((word, weight))

                # Sort by absolute weight
                feature_weights.sort(key=lambda x: abs(x[1]), reverse=True)

                print("\nTop contributing words:")
                for word, weight in feature_weights[:10]:
                    direction = "+" if weight > 0 else "-"
                    print(f"  {word}: {direction}{abs(weight):.4f}")

        return prediction, proba


# ==================== DEPLOYMENT MODULE ====================

class SentimentAPIHandler:
    """
    API handler for model deployment
    """

    def __init__(self, model_path='models/sentiment_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.preprocessor = TextPreprocessor()
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            self.model = BaselineModel()
            self.model.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"Model not found at {self.model_path}")

    def predict(self, text: str) -> Dict:
        """Make prediction for single text"""
        if self.model is None:
            return {"error": "Model not loaded"}

        # Preprocess
        processed_text = self.preprocessor.clean_text(text)

        # Predict
        prediction = self.model.predict([processed_text])[0]
        probabilities = self.model.predict_proba([processed_text])[0]

        return {
            "text": text,
            "sentiment": prediction,
            "confidence": float(max(probabilities)),
            "probabilities": {
                label: float(prob)
                for label, prob in zip(self.model.label_encoder.classes_, probabilities)
            },
            "timestamp": datetime.now().isoformat()
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Make predictions for multiple texts"""
        return [self.predict(text) for text in texts]


# ==================== MAIN PIPELINE ====================

class SentimentAnalysisPipeline:
    """
    Complete end-to-end pipeline
    """

    def __init__(self, output_dir='capstone_outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)
        os.makedirs(f"{output_dir}/reports", exist_ok=True)

        self.preprocessor = TextPreprocessor()
        self.models = {}
        self.results = {}

    def run(self):
        """Execute complete pipeline"""
        print("=" * 70)
        print("CAPSTONE PROJECT: SENTIMENT ANALYSIS SYSTEM")
        print("=" * 70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Data Loading
        print("\n" + "=" * 50)
        print("STEP 1: DATA LOADING AND EXPLORATION")
        print("=" * 50)

        df = SentimentDataGenerator.generate_synthetic_data(n_samples=5000)
        print(f"Dataset shape: {df.shape}")
        print(f"\nSentiment distribution:")
        print(df['sentiment'].value_counts())

        # Save sample data
        df.head(100).to_csv(f"{self.output_dir}/sample_data.csv", index=False)

        # Step 2: Preprocessing
        print("\n" + "=" * 50)
        print("STEP 2: TEXT PREPROCESSING")
        print("=" * 50)

        df['processed_text'] = self.preprocessor.preprocess_batch(df['text'].tolist())

        print("Sample preprocessing:")
        for i in range(3):
            print(f"\nOriginal: {df.iloc[i]['text'][:80]}...")
            print(f"Processed: {df.iloc[i]['processed_text'][:80]}...")

        # Step 3: Train/Test Split
        print("\n" + "=" * 50)
        print("STEP 3: DATA SPLITTING")
        print("=" * 50)

        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'].values,
            df['sentiment'].values,
            test_size=0.2,
            random_state=42,
            stratify=df['sentiment'].values
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")

        # Step 4: Train Baseline Model
        print("\n" + "=" * 50)
        print("STEP 4: BASELINE MODEL (TF-IDF + Logistic Regression)")
        print("=" * 50)

        baseline = BaselineModel(max_features=5000)
        baseline.fit(X_train, y_train)

        evaluator = ModelEvaluator(baseline, "Baseline (TF-IDF + LR)")
        baseline_results = evaluator.evaluate(X_test, y_test)
        evaluator.generate_report(X_test, y_test)
        evaluator.plot_confusion_matrix(
            X_test, y_test,
            f"{self.output_dir}/figures/baseline_confusion.png"
        )

        self.models['baseline'] = baseline
        self.results['baseline'] = baseline_results

        # Save baseline model
        baseline.save(f"{self.output_dir}/models/baseline_model.joblib")

        # Step 5: Random Forest Model
        print("\n" + "=" * 50)
        print("STEP 5: RANDOM FOREST MODEL")
        print("=" * 50)

        rf_model = BaselineModel(max_features=5000)
        rf_model.classifier = RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42
        )
        rf_model.fit(X_train, y_train)

        rf_evaluator = ModelEvaluator(rf_model, "Random Forest")
        rf_results = rf_evaluator.evaluate(X_test, y_test)
        rf_evaluator.generate_report(X_test, y_test)
        rf_evaluator.plot_confusion_matrix(
            X_test, y_test,
            f"{self.output_dir}/figures/rf_confusion.png"
        )

        self.models['random_forest'] = rf_model
        self.results['random_forest'] = rf_results

        # Step 6: Transformer Model (if available)
        if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
            print("\n" + "=" * 50)
            print("STEP 6: TRANSFORMER MODEL (DistilBERT)")
            print("=" * 50)

            # Note: Full training would take longer
            print("Note: Transformer training requires GPU and takes longer.")
            print("Using pre-trained pipeline for demo...")

            try:
                sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                print("Loaded pre-trained sentiment pipeline")
            except:
                print("Could not load transformer model")
        else:
            print("\n[Skipping Transformer model - requires GPU/transformers library]")

        # Step 7: Model Comparison
        print("\n" + "=" * 50)
        print("STEP 7: MODEL COMPARISON")
        print("=" * 50)

        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.drop('model_name', axis=1)
        print("\nModel Comparison:")
        print(comparison_df.to_string())

        # Save comparison
        comparison_df.to_csv(f"{self.output_dir}/reports/model_comparison.csv")

        # Plot comparison
        plt.figure(figsize=(10, 6))
        metrics = ['accuracy', 'f1_weighted', 'f1_macro']
        x = np.arange(len(metrics))
        width = 0.35

        for i, (model_name, results) in enumerate(self.results.items()):
            values = [results[m] for m in metrics]
            plt.bar(x + i * width, values, width, label=model_name)

        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Model Comparison')
        plt.xticks(x + width / 2, metrics)
        plt.legend()
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/model_comparison.png", dpi=150)
        plt.close()
        print(f"\nSaved: {self.output_dir}/figures/model_comparison.png")

        # Step 8: Explainability
        print("\n" + "=" * 50)
        print("STEP 8: MODEL EXPLAINABILITY")
        print("=" * 50)

        explainer = ModelExplainer(baseline, baseline.vectorizer)

        test_texts = [
            "This is absolutely wonderful, I love it!",
            "Terrible experience, very disappointed.",
            "It's okay, nothing special."
        ]

        for text in test_texts:
            processed = self.preprocessor.clean_text(text)
            explainer.explain_prediction(text, processed)

        # Step 9: Save Best Model for Deployment
        print("\n" + "=" * 50)
        print("STEP 9: SAVING BEST MODEL FOR DEPLOYMENT")
        print("=" * 50)

        best_model_name = max(self.results, key=lambda k: self.results[k]['f1_weighted'])
        best_model = self.models[best_model_name]

        print(f"Best model: {best_model_name}")
        print(f"F1 Score: {self.results[best_model_name]['f1_weighted']:.4f}")

        # Save for deployment
        deployment_path = f"{self.output_dir}/models/sentiment_model.joblib"
        best_model.save(deployment_path)
        print(f"Model saved to: {deployment_path}")

        # Step 10: Test Deployment Handler
        print("\n" + "=" * 50)
        print("STEP 10: TESTING DEPLOYMENT")
        print("=" * 50)

        handler = SentimentAPIHandler(deployment_path)

        test_inputs = [
            "This product is amazing! Best purchase ever!",
            "Terrible quality, complete waste of money.",
            "It works fine, nothing special."
        ]

        print("\nAPI Test Results:")
        for text in test_inputs:
            result = handler.predict(text)
            print(f"\nInput: {text}")
            print(f"Sentiment: {result['sentiment']} ({result['confidence']:.2%})")

        # Final Summary
        print("\n" + "=" * 70)
        print("CAPSTONE PROJECT COMPLETE")
        print("=" * 70)
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nOutputs saved to: {self.output_dir}/")
        print(f"  - models/: Trained models")
        print(f"  - figures/: Visualizations")
        print(f"  - reports/: Evaluation reports")

        print(f"\nBest Model Performance:")
        print(f"  Model: {best_model_name}")
        print(f"  Accuracy: {self.results[best_model_name]['accuracy']:.4f}")
        print(f"  F1 Score: {self.results[best_model_name]['f1_weighted']:.4f}")

        return self.models, self.results


if __name__ == "__main__":
    pipeline = SentimentAnalysisPipeline()
    models, results = pipeline.run()

    print("\n" + "=" * 70)
    print("TO DEPLOY THE MODEL:")
    print("=" * 70)
    print("""
    1. Use FastAPI deployment from Day 26:
       - Copy sentiment_model.joblib to deployment directory
       - Update FastAPI app to use SentimentAPIHandler

    2. Run with Docker:
       docker build -t sentiment-api .
       docker run -p 8000:8000 sentiment-api

    3. Access API:
       - Docs: http://localhost:8000/docs
       - Predict: POST /predict {"text": "Your text here"}
    """)
