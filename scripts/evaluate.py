#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script

This script evaluates both the baseline (TF-IDF + LogReg) and DistilBERT models
on the test set, performs detailed comparison, and generates visualizations.

Usage:
    python evaluate.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from datasets import load_dataset, load_from_disk
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_test_data():
    """Load test dataset."""
    print("\n" + "="*80)
    print("LOADING TEST DATA")
    print("="*80 + "\n")

    # Load IMDb dataset
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb", split="test")

    print(f"✅ Test set loaded: {len(dataset)} samples\n")
    return dataset


def evaluate_baseline(test_dataset):
    """Evaluate baseline TF-IDF + Logistic Regression model."""
    print("\n" + "="*80)
    print("EVALUATING BASELINE MODEL (TF-IDF + Logistic Regression)")
    print("="*80 + "\n")

    # Load baseline model
    baseline_path = "../models/baseline_tfidf_logreg.joblib"
    print(f"Loading baseline model from: {baseline_path}")

    if not os.path.exists(baseline_path):
        print("❌ Baseline model not found!")
        return None

    baseline_model = joblib.load(baseline_path)
    print("✅ Baseline model loaded\n")

    # Get predictions
    print("Getting predictions on test set...")
    texts = test_dataset['text']
    true_labels = np.array(test_dataset['label'])

    predictions = baseline_model.predict(texts)
    probabilities = baseline_model.predict_proba(texts)

    # Calculate metrics
    print("Calculating metrics...\n")

    metrics = {
        'model_name': 'TF-IDF + Logistic Regression',
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, average='weighted'),
        'recall': recall_score(true_labels, predictions, average='weighted'),
        'f1_weighted': f1_score(true_labels, predictions, average='weighted'),
        'precision_per_class': precision_score(true_labels, predictions, average=None).tolist(),
        'recall_per_class': recall_score(true_labels, predictions, average=None).tolist(),
        'f1_per_class': f1_score(true_labels, predictions, average=None).tolist(),
        'confusion_matrix': confusion_matrix(true_labels, predictions).tolist(),
        'roc_auc': roc_auc_score(true_labels, probabilities[:, 1])
    }

    # Print results
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_weighted']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

    return {
        'metrics': metrics,
        'predictions': predictions,
        'probabilities': probabilities,
        'true_labels': true_labels
    }


def evaluate_distilbert(test_dataset):
    """Evaluate fine-tuned DistilBERT model."""
    print("\n" + "="*80)
    print("EVALUATING DISTILBERT MODEL")
    print("="*80 + "\n")

    # Load model and tokenizer
    model_path = "../models/distilbert-sentiment/best_model"
    print(f"Loading DistilBERT model from: {model_path}")

    if not os.path.exists(model_path):
        print("❌ DistilBERT model not found!")
        return None

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model.to(device)
    model.eval()

    print("✅ DistilBERT model loaded\n")

    # Get predictions
    print("Getting predictions on test set...")
    texts = test_dataset['text']
    true_labels = np.array(test_dataset['label'])

    predictions = []
    probabilities = []

    batch_size = 32
    num_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Evaluating"):
            batch_texts = texts[i:i+batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get predictions
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)

    # Calculate metrics
    print("\nCalculating metrics...\n")

    metrics = {
        'model_name': 'DistilBERT (fine-tuned)',
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, average='weighted'),
        'recall': recall_score(true_labels, predictions, average='weighted'),
        'f1_weighted': f1_score(true_labels, predictions, average='weighted'),
        'precision_per_class': precision_score(true_labels, predictions, average=None).tolist(),
        'recall_per_class': recall_score(true_labels, predictions, average=None).tolist(),
        'f1_per_class': f1_score(true_labels, predictions, average=None).tolist(),
        'confusion_matrix': confusion_matrix(true_labels, predictions).tolist(),
        'roc_auc': roc_auc_score(true_labels, probabilities[:, 1])
    }

    # Print results
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_weighted']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

    return {
        'metrics': metrics,
        'predictions': predictions,
        'probabilities': probabilities,
        'true_labels': true_labels
    }


def plot_confusion_matrices(baseline_results, distilbert_results):
    """Plot side-by-side confusion matrices."""
    print("\n" + "="*80)
    print("GENERATING CONFUSION MATRIX COMPARISON")
    print("="*80 + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Baseline confusion matrix
    cm_baseline = np.array(baseline_results['metrics']['confusion_matrix'])
    sns.heatmap(
        cm_baseline,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=axes[0],
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        cbar_kws={'label': 'Count'}
    )
    axes[0].set_title(f"Baseline Model\nAccuracy: {baseline_results['metrics']['accuracy']:.2%}",
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)

    # DistilBERT confusion matrix
    cm_distilbert = np.array(distilbert_results['metrics']['confusion_matrix'])
    sns.heatmap(
        cm_distilbert,
        annot=True,
        fmt='d',
        cmap='Greens',
        ax=axes[1],
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        cbar_kws={'label': 'Count'}
    )
    axes[1].set_title(f"DistilBERT Model\nAccuracy: {distilbert_results['metrics']['accuracy']:.2%}",
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)

    plt.tight_layout()
    output_path = "../reports/confusion_matrix_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved confusion matrix comparison to: {output_path}")
    plt.close()


def plot_performance_comparison(baseline_results, distilbert_results):
    """Plot performance metrics comparison."""
    print("\n" + "="*80)
    print("GENERATING PERFORMANCE COMPARISON CHART")
    print("="*80 + "\n")

    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    baseline_scores = [
        baseline_results['metrics']['accuracy'],
        baseline_results['metrics']['precision'],
        baseline_results['metrics']['recall'],
        baseline_results['metrics']['f1_weighted'],
        baseline_results['metrics']['roc_auc']
    ]
    distilbert_scores = [
        distilbert_results['metrics']['accuracy'],
        distilbert_results['metrics']['precision'],
        distilbert_results['metrics']['recall'],
        distilbert_results['metrics']['f1_weighted'],
        distilbert_results['metrics']['roc_auc']
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline (TF-IDF)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, distilbert_scores, width, label='DistilBERT', color='forestgreen', alpha=0.8)

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    add_labels(bars1)
    add_labels(bars2)

    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison\nBaseline vs DistilBERT',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([0.8, 1.0])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = "../reports/performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved performance comparison to: {output_path}")
    plt.close()


def plot_per_class_metrics(baseline_results, distilbert_results):
    """Plot per-class precision, recall, and F1 scores."""
    print("\n" + "="*80)
    print("GENERATING PER-CLASS METRICS COMPARISON")
    print("="*80 + "\n")

    classes = ['Negative', 'Positive']
    metrics = ['Precision', 'Recall', 'F1 Score']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, metric in enumerate(metrics):
        if metric == 'Precision':
            baseline_values = baseline_results['metrics']['precision_per_class']
            distilbert_values = distilbert_results['metrics']['precision_per_class']
        elif metric == 'Recall':
            baseline_values = baseline_results['metrics']['recall_per_class']
            distilbert_values = distilbert_results['metrics']['recall_per_class']
        else:  # F1 Score
            baseline_values = baseline_results['metrics']['f1_per_class']
            distilbert_values = distilbert_results['metrics']['f1_per_class']

        x = np.arange(len(classes))
        width = 0.35

        bars1 = axes[idx].bar(x - width/2, baseline_values, width,
                              label='Baseline', color='steelblue', alpha=0.8)
        bars2 = axes[idx].bar(x + width/2, distilbert_values, width,
                              label='DistilBERT', color='forestgreen', alpha=0.8)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.3f}',
                             ha='center', va='bottom', fontsize=9, fontweight='bold')

        axes[idx].set_ylabel('Score', fontsize=11, fontweight='bold')
        axes[idx].set_title(metric, fontsize=12, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(classes)
        axes[idx].legend(fontsize=9)
        axes[idx].set_ylim([0.8, 1.0])
        axes[idx].grid(axis='y', alpha=0.3)

    plt.suptitle('Per-Class Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = "../reports/per_class_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved per-class metrics to: {output_path}")
    plt.close()


def analyze_errors(test_dataset, baseline_results, distilbert_results, num_examples=10):
    """Analyze misclassified examples."""
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80 + "\n")

    true_labels = baseline_results['true_labels']
    baseline_preds = baseline_results['predictions']
    distilbert_preds = distilbert_results['predictions']

    # Find different error types
    baseline_errors = (baseline_preds != true_labels)
    distilbert_errors = (distilbert_preds != true_labels)

    # Errors only in baseline
    baseline_only_errors = baseline_errors & ~distilbert_errors
    # Errors only in DistilBERT
    distilbert_only_errors = distilbert_errors & ~baseline_errors
    # Errors in both
    both_errors = baseline_errors & distilbert_errors

    error_analysis = {
        'total_samples': len(true_labels),
        'baseline_errors': int(baseline_errors.sum()),
        'distilbert_errors': int(distilbert_errors.sum()),
        'baseline_only_errors': int(baseline_only_errors.sum()),
        'distilbert_only_errors': int(distilbert_only_errors.sum()),
        'both_errors': int(both_errors.sum()),
        'error_reduction': int(baseline_errors.sum() - distilbert_errors.sum())
    }

    print(f"Total test samples: {error_analysis['total_samples']}")
    print(f"\nBaseline errors: {error_analysis['baseline_errors']} ({error_analysis['baseline_errors']/error_analysis['total_samples']*100:.2f}%)")
    print(f"DistilBERT errors: {error_analysis['distilbert_errors']} ({error_analysis['distilbert_errors']/error_analysis['total_samples']*100:.2f}%)")
    print(f"\nErrors only in Baseline: {error_analysis['baseline_only_errors']}")
    print(f"Errors only in DistilBERT: {error_analysis['distilbert_only_errors']}")
    print(f"Errors in both models: {error_analysis['both_errors']}")
    print(f"\n✅ Error reduction: {error_analysis['error_reduction']} samples")

    # Sample misclassified examples
    print(f"\n" + "-"*80)
    print(f"SAMPLE MISCLASSIFIED EXAMPLES")
    print("-"*80 + "\n")

    examples = {
        'baseline_fixed_by_distilbert': [],
        'distilbert_new_errors': [],
        'both_models_wrong': []
    }

    # Get indices for each category
    baseline_fixed_indices = np.where(baseline_only_errors)[0][:num_examples]
    distilbert_new_indices = np.where(distilbert_only_errors)[0][:num_examples]
    both_wrong_indices = np.where(both_errors)[0][:num_examples]

    label_names = ['Negative', 'Positive']

    # Baseline errors fixed by DistilBERT
    if len(baseline_fixed_indices) > 0:
        print("1. Errors Fixed by DistilBERT (Baseline wrong, DistilBERT correct):\n")
        for idx in baseline_fixed_indices:
            example = {
                'text': test_dataset[int(idx)]['text'][:200] + "...",
                'true_label': label_names[int(true_labels[idx])],
                'baseline_pred': label_names[int(baseline_preds[idx])],
                'distilbert_pred': label_names[int(distilbert_preds[idx])]
            }
            examples['baseline_fixed_by_distilbert'].append(example)
            print(f"   True: {example['true_label']}, Baseline: {example['baseline_pred']}, DistilBERT: {example['distilbert_pred']}")
            print(f"   Text: {example['text']}\n")

    # New errors in DistilBERT
    if len(distilbert_new_indices) > 0:
        print("\n2. New Errors in DistilBERT (Baseline correct, DistilBERT wrong):\n")
        for idx in distilbert_new_indices:
            example = {
                'text': test_dataset[int(idx)]['text'][:200] + "...",
                'true_label': label_names[int(true_labels[idx])],
                'baseline_pred': label_names[int(baseline_preds[idx])],
                'distilbert_pred': label_names[int(distilbert_preds[idx])]
            }
            examples['distilbert_new_errors'].append(example)
            print(f"   True: {example['true_label']}, Baseline: {example['baseline_pred']}, DistilBERT: {example['distilbert_pred']}")
            print(f"   Text: {example['text']}\n")

    # Errors in both models
    if len(both_wrong_indices) > 0:
        print("\n3. Errors in Both Models:\n")
        for idx in both_wrong_indices:
            example = {
                'text': test_dataset[int(idx)]['text'][:200] + "...",
                'true_label': label_names[int(true_labels[idx])],
                'baseline_pred': label_names[int(baseline_preds[idx])],
                'distilbert_pred': label_names[int(distilbert_preds[idx])]
            }
            examples['both_models_wrong'].append(example)
            print(f"   True: {example['true_label']}, Baseline: {example['baseline_pred']}, DistilBERT: {example['distilbert_pred']}")
            print(f"   Text: {example['text']}\n")

    return error_analysis, examples


def save_results(baseline_results, distilbert_results, error_analysis, examples):
    """Save comprehensive evaluation results to JSON."""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80 + "\n")

    results = {
        'evaluation_date': datetime.now().isoformat(),
        'dataset': 'IMDb Movie Reviews (Test Set)',
        'num_samples': int(len(baseline_results['true_labels'])),
        'models': {
            'baseline': baseline_results['metrics'],
            'distilbert': distilbert_results['metrics']
        },
        'comparison': {
            'accuracy_improvement': float(distilbert_results['metrics']['accuracy'] - baseline_results['metrics']['accuracy']),
            'precision_improvement': float(distilbert_results['metrics']['precision'] - baseline_results['metrics']['precision']),
            'recall_improvement': float(distilbert_results['metrics']['recall'] - baseline_results['metrics']['recall']),
            'f1_improvement': float(distilbert_results['metrics']['f1_weighted'] - baseline_results['metrics']['f1_weighted']),
            'roc_auc_improvement': float(distilbert_results['metrics']['roc_auc'] - baseline_results['metrics']['roc_auc'])
        },
        'error_analysis': error_analysis,
        'misclassified_examples': examples
    }

    output_path = "../reports/evaluation_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved evaluation metrics to: {output_path}")

    # Also create a summary markdown report
    create_summary_report(results)


def create_summary_report(results):
    """Create a markdown summary report."""
    report_path = "../reports/evaluation_summary.md"

    with open(report_path, 'w') as f:
        f.write("# Model Evaluation Summary Report\n\n")
        f.write(f"**Date:** {results['evaluation_date']}\n\n")
        f.write(f"**Dataset:** {results['dataset']}\n\n")
        f.write(f"**Test Samples:** {results['num_samples']:,}\n\n")

        f.write("---\n\n")
        f.write("## Performance Comparison\n\n")

        f.write("| Metric | Baseline | DistilBERT | Improvement |\n")
        f.write("|--------|----------|------------|-------------|\n")

        baseline = results['models']['baseline']
        distilbert = results['models']['distilbert']
        comp = results['comparison']

        f.write(f"| **Accuracy** | {baseline['accuracy']:.4f} | {distilbert['accuracy']:.4f} | +{comp['accuracy_improvement']:.4f} |\n")
        f.write(f"| **Precision** | {baseline['precision']:.4f} | {distilbert['precision']:.4f} | +{comp['precision_improvement']:.4f} |\n")
        f.write(f"| **Recall** | {baseline['recall']:.4f} | {distilbert['recall']:.4f} | +{comp['recall_improvement']:.4f} |\n")
        f.write(f"| **F1 Score** | {baseline['f1_weighted']:.4f} | {distilbert['f1_weighted']:.4f} | +{comp['f1_improvement']:.4f} |\n")
        f.write(f"| **ROC AUC** | {baseline['roc_auc']:.4f} | {distilbert['roc_auc']:.4f} | +{comp['roc_auc_improvement']:.4f} |\n\n")

        f.write("---\n\n")
        f.write("## Error Analysis\n\n")

        ea = results['error_analysis']
        f.write(f"- **Baseline Errors:** {ea['baseline_errors']} ({ea['baseline_errors']/ea['total_samples']*100:.2f}%)\n")
        f.write(f"- **DistilBERT Errors:** {ea['distilbert_errors']} ({ea['distilbert_errors']/ea['total_samples']*100:.2f}%)\n")
        f.write(f"- **Error Reduction:** {ea['error_reduction']} samples\n\n")

        f.write(f"- Errors fixed by DistilBERT: {ea['baseline_only_errors']}\n")
        f.write(f"- New errors in DistilBERT: {ea['distilbert_only_errors']}\n")
        f.write(f"- Errors in both models: {ea['both_errors']}\n\n")

        f.write("---\n\n")
        f.write("## Visualizations\n\n")
        f.write("- `confusion_matrix_comparison.png` - Side-by-side confusion matrices\n")
        f.write("- `performance_comparison.png` - Overall performance metrics comparison\n")
        f.write("- `per_class_metrics.png` - Per-class precision, recall, and F1 scores\n\n")

        f.write("---\n\n")
        f.write("## Conclusion\n\n")

        if distilbert['accuracy'] >= 0.92:
            f.write(f"✅ **Test accuracy target (≥92%) ACHIEVED: {distilbert['accuracy']:.2%}**\n\n")

        f.write(f"DistilBERT shows clear performance improvement over the baseline:\n")
        f.write(f"- Accuracy improved by {comp['accuracy_improvement']*100:.2f} percentage points\n")
        f.write(f"- Reduced errors by {ea['error_reduction']} samples\n")
        f.write(f"- Achieved {distilbert['roc_auc']:.4f} ROC AUC score\n")

    print(f"✅ Saved evaluation summary to: {report_path}")


def main():
    """Main evaluation pipeline."""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)

    # Load test data
    test_dataset = load_test_data()

    # Evaluate both models
    baseline_results = evaluate_baseline(test_dataset)
    distilbert_results = evaluate_distilbert(test_dataset)

    if baseline_results is None or distilbert_results is None:
        print("\n❌ Evaluation failed - missing models")
        return

    # Generate visualizations
    plot_confusion_matrices(baseline_results, distilbert_results)
    plot_performance_comparison(baseline_results, distilbert_results)
    plot_per_class_metrics(baseline_results, distilbert_results)

    # Analyze errors
    error_analysis, examples = analyze_errors(test_dataset, baseline_results, distilbert_results)

    # Save results
    save_results(baseline_results, distilbert_results, error_analysis, examples)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print("\n📊 Results saved to:")
    print("   - reports/evaluation_metrics.json")
    print("   - reports/evaluation_summary.md")
    print("\n📈 Visualizations saved to:")
    print("   - reports/confusion_matrix_comparison.png")
    print("   - reports/performance_comparison.png")
    print("   - reports/per_class_metrics.png")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
