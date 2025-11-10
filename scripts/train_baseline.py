
# First here we compare with easier models like logistic regression and then 
# compare with more complex models like DistilBERT.

# TF - IDF converts words into vectors based on their frequency in the document
# less frequent words are given more importance

import os
import sys
import json
import time
import joblib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_and_split_data(train_size=20000, val_size=5000, seed=42):

# loads data, splits it and converts to dictionary of dataframes
    print("=" * 80)
    print("LOADING IMDB DATASET")
    print("=" * 80)

    # Load dataset
    print("\nLoading IMDb dataset from Hugging Face...")
    dataset = load_dataset("imdb")

    # Convert to pandas for easier manipulation
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    print(f"Original train size: {len(train_df):,}")
    print(f"Original test size: {len(test_df):,}")

    # Shuffle and split training set into train/validation
    np.random.seed(seed)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Create splits
    train_split = train_df.iloc[:train_size]
    val_split = train_df.iloc[train_size:train_size + val_size]
    test_split = test_df

    print(f"\nFinal splits:")
    print(f"  Training:   {len(train_split):,} samples")
    print(f"  Validation: {len(val_split):,} samples")
    print(f"  Test:       {len(test_split):,} samples")

    # Check class balance
    print(f"\nClass distribution in training set:")
    print(f"  Negative (0): {(train_split['label'] == 0).sum():,} ({(train_split['label'] == 0).sum()/len(train_split)*100:.1f}%)")
    print(f"  Positive (1): {(train_split['label'] == 1).sum():,} ({(train_split['label'] == 1).sum()/len(train_split)*100:.1f}%)")

    return {
        'train': train_split,
        'val': val_split,
        'test': test_split
    }


def build_pipeline(max_features=10000, ngram_range=(1, 2), max_iter=1000):
  
#   creates scikit-learn pipeline with TF-IDF and Logistic Regression
    print("\n" + "=" * 80)
    print("BUILDING BASELINE PIPELINE")
    print("=" * 80)

    print(f"\nPipeline configuration:")
    print(f"  TF-IDF max_features: {max_features:,}")
    print(f"  TF-IDF ngram_range: {ngram_range}")
    print(f"  Logistic Regression max_iter: {max_iter}")
    print(f"  Logistic Regression solver: lbfgs")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            min_df=2,  # Ignore terms that appear in fewer than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )),
        ('classifier', LogisticRegression(
            max_iter=max_iter,
            solver='lbfgs',
            random_state=42,
            n_jobs=-1,
            verbose=1
        ))
    ])

    return pipeline


def train_model(pipeline, X_train, y_train):

# training the pipeline
    print("\n" + "=" * 80)
    print("TRAINING BASELINE MODEL")
    print("=" * 80)

    print(f"\nTraining on {len(X_train):,} samples...")
    start_time = time.time()

    pipeline.fit(X_train, y_train)

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")

    # Display feature information
    tfidf = pipeline.named_steps['tfidf']
    print(f"\nTF-IDF vocabulary size: {len(tfidf.vocabulary_):,} features")

    return pipeline, train_time


def evaluate_model(pipeline, X, y, split_name="Test"):

# prediction
    print(f"\nEvaluating on {split_name} set ({len(X):,} samples)...")

    # Make predictions
    y_pred = pipeline.predict(X)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='binary')
    recall = recall_score(y, y_pred, average='binary')
    f1 = f1_score(y, y_pred, average='binary')

    # Get per-class metrics
    precision_per_class = precision_score(y, y_pred, average=None)
    recall_per_class = recall_score(y, y_pred, average=None)
    f1_per_class = f1_score(y, y_pred, average=None)

    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'precision_per_class': {
            'negative': float(precision_per_class[0]),
            'positive': float(precision_per_class[1])
        },
        'recall_per_class': {
            'negative': float(recall_per_class[0]),
            'positive': float(recall_per_class[1])
        },
        'f1_per_class': {
            'negative': float(f1_per_class[0]),
            'positive': float(f1_per_class[1])
        },
        'confusion_matrix': cm.tolist(),
        'num_samples': len(X)
    }

    # Print metrics
    print(f"\n{split_name} Set Results:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    print(f"\nPer-class metrics:")
    print(f"  Negative - Precision: {precision_per_class[0]:.4f}, Recall: {recall_per_class[0]:.4f}, F1: {f1_per_class[0]:.4f}")
    print(f"  Positive - Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}")

    # Print classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y, y_pred, target_names=['Negative', 'Positive']))

    return metrics, y_pred


def plot_confusion_matrix(cm, split_name, save_path):


# creates confusion matrix plot
    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'})

    plt.title(f'Confusion Matrix - TF-IDF Baseline ({split_name} Set)',
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')

    # Add accuracy text
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)',
             ha='center', transform=plt.gca().transAxes,
             fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to: {save_path}")
    plt.close()


def save_model(pipeline, save_path):


    joblib.dump(pipeline, save_path)
    file_size = os.path.getsize(save_path) / (1024 * 1024)  # Convert to MB
    print(f"\n✓ Model saved to: {save_path}")
    print(f"  File size: {file_size:.2f} MB")


def save_metrics(metrics, save_path):
    """
    Save metrics to JSON file.

    Args:
        metrics: Dictionary with all metrics
        save_path: Path to save the JSON file
    """
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {save_path}")


def main(args):
    """Main training function."""

    # Create directories
    Path('../models').mkdir(exist_ok=True)
    Path('../reports').mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("TF-IDF + LOGISTIC REGRESSION BASELINE")
    print("Sentiment Analysis on IMDb Dataset")
    print("=" * 80)

    # Load and split data
    data_splits = load_and_split_data(
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed
    )

    X_train = data_splits['train']['text'].values
    y_train = data_splits['train']['label'].values

    X_val = data_splits['val']['text'].values
    y_val = data_splits['val']['label'].values

    X_test = data_splits['test']['text'].values
    y_test = data_splits['test']['label'].values

    # Build pipeline
    pipeline = build_pipeline(
        max_features=args.max_features,
        ngram_range=tuple(args.ngram_range),
        max_iter=args.max_iter
    )

    # Train model
    pipeline, train_time = train_model(pipeline, X_train, y_train)

    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("VALIDATION SET EVALUATION")
    print("=" * 80)
    val_metrics, val_pred = evaluate_model(pipeline, X_val, y_val, "Validation")

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    test_metrics, test_pred = evaluate_model(pipeline, X_test, y_test, "Test")

    # Plot confusion matrices
    plot_confusion_matrix(
        np.array(val_metrics['confusion_matrix']),
        "Validation",
        "../reports/baseline_confusion_matrix_val.png"
    )

    plot_confusion_matrix(
        np.array(test_metrics['confusion_matrix']),
        "Test",
        "../reports/baseline_confusion_matrix_test.png"
    )

    # Save model
    model_path = f"../models/baseline_tfidf_logreg.joblib"
    save_model(pipeline, model_path)

    # Compile all metrics
    all_metrics = {
        "model_type": "TF-IDF + Logistic Regression",
        "dataset": "IMDb Movie Reviews",
        "training_config": {
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": len(X_test),
            "max_features": args.max_features,
            "ngram_range": args.ngram_range,
            "max_iter": args.max_iter,
            "random_seed": args.seed
        },
        "training_time_seconds": train_time,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics
    }

    # Save metrics
    save_metrics(all_metrics, "../reports/baseline_metrics.json")

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"\n📊 Model Performance:")
    print(f"  Validation Accuracy: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
    print(f"  Test Accuracy:       {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"\n⏱️  Training Time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    print(f"\n💾 Saved Artifacts:")
    print(f"  Model:              {model_path}")
    print(f"  Metrics:            ../reports/baseline_metrics.json")
    print(f"  Confusion Matrix:   ../reports/baseline_confusion_matrix_val.png")
    print(f"                      ../reports/baseline_confusion_matrix_test.png")

    print("\n" + "=" * 80)
    print("✅ Baseline training completed successfully!")
    print("=" * 80)

    return all_metrics


if __name__ == "__main__":
    # custom parameters can be set here
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression baseline")
    parser.add_argument("--train_size", type=int, default=20000, help="Training set size")
    parser.add_argument("--val_size", type=int, default=5000, help="Validation set size")
    parser.add_argument("--max_features", type=int, default=10000, help="Max features for TF-IDF")
    parser.add_argument("--ngram_range", type=int, nargs=2, default=[1, 2], help="N-gram range")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for LogReg")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    main(args)
