#!/usr/bin/env python3

# SHAP (SHapley Additive exPlanations)
# Html plots
# Other plats

import os
import sys
import json
import argparse
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SentimentExplainer:
# shap explainer for DistilBERT sentiment model

    def __init__(self, model_path: str = "../models/distilbert-sentiment/best_model"):

        print(f"\nInitializing SentimentExplainer...")
        print(f"Model path: {model_path}")

        # Load model and tokenizer
        self.device = self._get_device()
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Create pipeline for easier prediction
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            return_all_scores=True,
            truncation=True,
            max_length=512
        )

        # Initialize SHAP explainer
        print("\nInitializing SHAP explainer (this may take a moment)...")
        # Use pipeline directly with SHAP
        self.explainer = shap.Explainer(self.pipeline)

        # Label mapping
        self.labels = ["Negative", "Positive"]

        print("✅ SentimentExplainer initialized successfully!\n")

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _predict_proba(self, texts) -> np.ndarray:

        # Convert to list of strings if needed
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            # SHAP might pass numpy arrays
            if texts.dtype == object:
                texts = texts.tolist()
            else:
                # Handle masked arrays - convert to strings
                texts = [str(t) if t is not np.ma.masked else "" for t in texts.flatten()]
        elif not isinstance(texts, list):
            texts = list(texts)

        # Ensure all elements are strings
        texts = [str(t) if not isinstance(t, str) else t for t in texts]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        return probs.cpu().numpy()

    def explain_text(self, text: str, show_plot: bool = False) -> shap.Explanation:
        # Generate SHAP explanation for a single text.


        # Truncate text if too long (simple word-level truncation)
        # Tokenizer will handle proper truncation, but this prevents excessively long texts
        words = text.split()
        if len(words) > 400:  # Rough estimate to stay well under 512 tokens
            text = ' '.join(words[:400])

        # Generate SHAP values
        shap_values = self.explainer([text])

        if show_plot:
            shap.plots.text(shap_values)

        return shap_values

    def predict_with_explanation(self, text: str) -> Dict:
        
        # Get prediction
        outputs = self.pipeline(text)

        # Handle different output formats
        if isinstance(outputs[0], list):
            probs = {item['label']: item['score'] for item in outputs[0]}
        else:
            probs = {outputs[0]['label']: outputs[0]['score']}

        # Determine predicted label
        label_scores = {}
        for key, val in probs.items():
            if 'LABEL_1' in key or 'POSITIVE' in key.upper() or key == '1':
                label_scores['Positive'] = val
            elif 'LABEL_0' in key or 'NEGATIVE' in key.upper() or key == '0':
                label_scores['Negative'] = val

        if 'Positive' not in label_scores:
            label_scores = {'Negative': 1 - list(probs.values())[0], 'Positive': list(probs.values())[0]}

        predicted_label = "Positive" if label_scores['Positive'] > 0.5 else "Negative"
        confidence = max(label_scores.values())

        # Get SHAP explanation
        shap_values = self.explain_text(text)

        return {
            'text': text,
            'prediction': predicted_label,
            'confidence': confidence,
            'probabilities': label_scores,
            'shap_values': shap_values
        }

    def save_force_plot(self, shap_values: shap.Explanation, output_path: str, index: int = 0):
#   save as html
        # Generate force plot
        force_plot = shap.plots.force(
            shap_values[index, :, 1],  # SHAP values for positive class
            matplotlib=False
        )

        # Save as HTML
        shap.save_html(output_path, force_plot)

    def save_waterfall_plot(self, shap_values: shap.Explanation, output_path: str, index: int = 0):

        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_values[index, :, 1], show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_text_plot(self, shap_values: shap.Explanation, output_path: str, index: int = 0):

        plt.figure(figsize=(16, 3))
        shap.plots.text(shap_values[index], display=False)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def select_diverse_examples(
    dataset,
    evaluation_results_path: str,
    num_examples: int = 30
) -> List[Dict]:

    # get diverse examples from test set
    print(f"\nSelecting {num_examples} diverse examples from test set...")

    # Load evaluation results if available
    if os.path.exists(evaluation_results_path):
        with open(evaluation_results_path, 'r') as f:
            eval_results = json.load(f)

        # Check if we have per-sample results
        if 'distilbert' in eval_results and 'predictions' in eval_results['distilbert']:
            predictions = np.array(eval_results['distilbert']['predictions'])
            probabilities = np.array(eval_results['distilbert']['probabilities'])
            true_labels = np.array(eval_results['distilbert']['true_labels'])
        else:
            # Generate predictions if not available
            print("Evaluation results don't contain predictions. Generating...")
            return select_examples_without_eval(dataset, num_examples)
    else:
        # Generate predictions if evaluation file doesn't exist
        print("Evaluation results not found. Generating predictions...")
        return select_examples_without_eval(dataset, num_examples)

    # Categorize examples
    tp_indices = np.where((true_labels == 1) & (predictions == 1))[0]  # True positives
    tn_indices = np.where((true_labels == 0) & (predictions == 0))[0]  # True negatives
    fp_indices = np.where((true_labels == 0) & (predictions == 1))[0]  # False positives
    fn_indices = np.where((true_labels == 1) & (predictions == 0))[0]  # False negatives

    # Calculate how many from each category
    examples_per_category = num_examples // 4

    selected_examples = []

    # Helper function to select examples with diverse confidence
    def select_from_category(indices, label, category_name, n):
        if len(indices) == 0:
            return []

        # Get probabilities for these indices
        probs = probabilities[indices]
        confidences = np.max(probs, axis=1)

        # Sort by confidence
        sorted_idx = np.argsort(confidences)

        # Select diverse examples: high, medium, low confidence
        selected_idx = []
        if len(indices) >= n:
            # Spread selection across confidence levels
            step = len(sorted_idx) // n
            for i in range(n):
                idx = sorted_idx[min(i * step, len(sorted_idx) - 1)]
                selected_idx.append(indices[idx])
        else:
            # Take all available
            selected_idx = indices[:n]

        examples = []
        for idx in selected_idx:
            examples.append({
                'index': int(idx),
                'text': dataset[int(idx)]['text'],
                'true_label': int(true_labels[idx]),
                'predicted_label': int(predictions[idx]),
                'probabilities': probabilities[idx].tolist(),
                'confidence': float(np.max(probabilities[idx])),
                'category': category_name
            })

        return examples

    # Select from each category
    selected_examples.extend(select_from_category(tp_indices, 1, "True Positive", examples_per_category))
    selected_examples.extend(select_from_category(tn_indices, 0, "True Negative", examples_per_category))
    selected_examples.extend(select_from_category(fp_indices, 0, "False Positive", examples_per_category))
    selected_examples.extend(select_from_category(fn_indices, 1, "False Negative", examples_per_category))

    # If we need more examples, add from true positives and negatives
    remaining = num_examples - len(selected_examples)
    if remaining > 0:
        all_correct = np.concatenate([tp_indices, tn_indices])
        if len(all_correct) > remaining:
            extra_indices = np.random.choice(all_correct, remaining, replace=False)
            for idx in extra_indices:
                selected_examples.append({
                    'index': int(idx),
                    'text': dataset[int(idx)]['text'],
                    'true_label': int(true_labels[idx]),
                    'predicted_label': int(predictions[idx]),
                    'probabilities': probabilities[idx].tolist(),
                    'confidence': float(np.max(probabilities[idx])),
                    'category': "Additional Correct"
                })

    print(f"✅ Selected {len(selected_examples)} examples:")
    print(f"   - True Positives: {sum(1 for e in selected_examples if e['category'] == 'True Positive')}")
    print(f"   - True Negatives: {sum(1 for e in selected_examples if e['category'] == 'True Negative')}")
    print(f"   - False Positives: {sum(1 for e in selected_examples if e['category'] == 'False Positive')}")
    print(f"   - False Negatives: {sum(1 for e in selected_examples if e['category'] == 'False Negative')}")

    return selected_examples


def select_examples_without_eval(dataset, num_examples: int) -> List[Dict]:

    # Select mix of positive and negative examples
    positive_indices = [i for i, item in enumerate(dataset) if item['label'] == 1]
    negative_indices = [i for i, item in enumerate(dataset) if item['label'] == 0]

    n_pos = num_examples // 2
    n_neg = num_examples - n_pos

    selected_pos = np.random.choice(positive_indices, min(n_pos, len(positive_indices)), replace=False)
    selected_neg = np.random.choice(negative_indices, min(n_neg, len(negative_indices)), replace=False)

    selected_examples = []
    for idx in selected_pos:
        selected_examples.append({
            'index': int(idx),
            'text': dataset[int(idx)]['text'],
            'true_label': 1,
            'category': "Positive (Random)"
        })
    for idx in selected_neg:
        selected_examples.append({
            'index': int(idx),
            'text': dataset[int(idx)]['text'],
            'true_label': 0,
            'category': "Negative (Random)"
        })

    return selected_examples


def generate_explanations(
    explainer: SentimentExplainer,
    examples: List[Dict],
    output_dir: str
):
# get SHAP explanations for examples
    print(f"\n{'='*80}")
    print("GENERATING SHAP EXPLANATIONS")
    print(f"{'='*80}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/force_plots", exist_ok=True)
    os.makedirs(f"{output_dir}/waterfall_plots", exist_ok=True)
    os.makedirs(f"{output_dir}/text_plots", exist_ok=True)

    all_explanations = []
    all_shap_values = []

    print(f"Generating explanations for {len(examples)} examples...")
    print("This may take several minutes...\n")

    for i, example in enumerate(tqdm(examples, desc="Explaining")):
        text = example['text']

        # Truncate very long texts for visualization
        if len(text) > 1000:
            text_display = text[:1000] + "..."
        else:
            text_display = text

        # Get prediction and explanation
        result = explainer.predict_with_explanation(text)
        shap_values = result['shap_values']

        # Save metadata
        explanation_data = {
            'index': example['index'],
            'true_label': example['true_label'],
            'predicted_label': result['prediction'],
            'confidence': result['confidence'],
            'category': example['category'],
            'text': text,
            'text_length': len(text)
        }

        if 'probabilities' in example:
            explanation_data['stored_probabilities'] = example['probabilities']

        all_explanations.append(explanation_data)
        all_shap_values.append(shap_values)

        # Save visualizations for this example
        example_id = f"example_{i:03d}_idx{example['index']}"

        try:
            # Force plot (HTML)
            force_path = f"{output_dir}/force_plots/{example_id}.html"
            explainer.save_force_plot(shap_values, force_path, index=0)

            # Waterfall plot (image)
            waterfall_path = f"{output_dir}/waterfall_plots/{example_id}.png"
            explainer.save_waterfall_plot(shap_values, waterfall_path, index=0)

            # Text plot (image) - shows colored tokens
            text_path = f"{output_dir}/text_plots/{example_id}.png"
            explainer.save_text_plot(shap_values, text_path, index=0)

        except Exception as e:
            print(f"\nWarning: Could not create all visualizations for example {i}: {e}")

    # Save all explanations metadata
    with open(f"{output_dir}/explanations_metadata.json", 'w') as f:
        json.dump(all_explanations, f, indent=2)

    print(f"\n✅ Generated explanations for {len(examples)} examples")
    print(f"\n📁 Outputs saved to: {output_dir}/")
    print(f"   - Force plots (HTML): {output_dir}/force_plots/")
    print(f"   - Waterfall plots (PNG): {output_dir}/waterfall_plots/")
    print(f"   - Text plots (PNG): {output_dir}/text_plots/")
    print(f"   - Metadata (JSON): {output_dir}/explanations_metadata.json")

    # Generate summary visualization
    print(f"\nGenerating summary visualizations...")
    generate_summary_plots(all_explanations, output_dir)

    return all_explanations


def generate_summary_plots(explanations: List[Dict], output_dir: str):

    # Summary statistics
    categories = {}
    for exp in explanations:
        cat = exp['category']
        if cat not in categories:
            categories[cat] = {'count': 0, 'confidences': []}
        categories[cat]['count'] += 1
        categories[cat]['confidences'].append(exp['confidence'])

    # Plot 1: Category distribution
    plt.figure(figsize=(10, 6))
    cats = list(categories.keys())
    counts = [categories[cat]['count'] for cat in cats]
    plt.bar(cats, counts, color=['green', 'blue', 'orange', 'red'][:len(cats)])
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Distribution of Explained Examples by Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_distribution.png", dpi=300)
    plt.close()

    # Plot 2: Confidence distribution
    plt.figure(figsize=(12, 6))
    for i, cat in enumerate(cats):
        confidences = categories[cat]['confidences']
        plt.subplot(1, len(cats), i+1)
        plt.hist(confidences, bins=10, alpha=0.7, color=['green', 'blue', 'orange', 'red'][i % 4])
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title(f'{cat}\n(n={len(confidences)})')
        plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confidence_distribution.png", dpi=300)
    plt.close()

    # Plot 3: Text length distribution
    plt.figure(figsize=(10, 6))
    text_lengths = [exp['text_length'] for exp in explanations]
    plt.hist(text_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Count')
    plt.title('Distribution of Text Lengths in Explained Examples')
    plt.axvline(np.mean(text_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(text_lengths):.0f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/text_length_distribution.png", dpi=300)
    plt.close()

    print(f"✅ Summary plots saved to {output_dir}/")


def create_summary_report(explanations: List[Dict], output_path: str):

    with open(output_path, 'w') as f:
        f.write("# Task 8: SHAP Model Interpretability - Summary Report\n\n")
        f.write(f"**Generated:** {Path(output_path).stat().st_mtime}\n\n")
        f.write("---\n\n")

        f.write("## Overview\n\n")
        f.write(f"Total examples explained: **{len(explanations)}**\n\n")

        # Category breakdown
        categories = {}
        for exp in explanations:
            cat = exp['category']
            categories[cat] = categories.get(cat, 0) + 1

        f.write("### Breakdown by Category\n\n")
        f.write("| Category | Count | Percentage |\n")
        f.write("|----------|-------|------------|\n")
        for cat, count in sorted(categories.items()):
            pct = (count / len(explanations)) * 100
            f.write(f"| {cat} | {count} | {pct:.1f}% |\n")

        f.write("\n---\n\n")

        f.write("## Explanation Files\n\n")
        f.write("### Visualization Types\n\n")
        f.write("1. **Force Plots** (`force_plots/*.html`)\n")
        f.write("   - Interactive HTML visualizations\n")
        f.write("   - Shows how each token pushes prediction toward positive or negative\n")
        f.write("   - Red tokens push toward positive, blue toward negative\n\n")

        f.write("2. **Waterfall Plots** (`waterfall_plots/*.png`)\n")
        f.write("   - Shows cumulative effect of tokens\n")
        f.write("   - Displays top contributing tokens\n")
        f.write("   - Easy to see which words had the biggest impact\n\n")

        f.write("3. **Text Plots** (`text_plots/*.png`)\n")
        f.write("   - Color-coded text visualization\n")
        f.write("   - Inline display of token importance\n")
        f.write("   - Quick visual scan of important words\n\n")

        f.write("---\n\n")

        f.write("## Sample Examples\n\n")

        # Show a few interesting examples
        for i, exp in enumerate(explanations[:5]):
            f.write(f"### Example {i+1}: {exp['category']}\n\n")
            f.write(f"**True Label:** {['Negative', 'Positive'][exp['true_label']]}\n\n")
            f.write(f"**Predicted:** {exp['predicted_label']} (Confidence: {exp['confidence']:.2%})\n\n")
            f.write(f"**Text:** {exp['text'][:200]}{'...' if len(exp['text']) > 200 else ''}\n\n")
            f.write(f"**Visualizations:**\n")
            f.write(f"- [Force Plot](force_plots/example_{i:03d}_idx{exp['index']}.html)\n")
            f.write(f"- [Waterfall Plot](waterfall_plots/example_{i:03d}_idx{exp['index']}.png)\n")
            f.write(f"- [Text Plot](text_plots/example_{i:03d}_idx{exp['index']}.png)\n\n")
            f.write("---\n\n")

        f.write("## Programmatic Usage\n\n")
        f.write("```python\n")
        f.write("from explain import SentimentExplainer\n\n")
        f.write("# Initialize explainer\n")
        f.write("explainer = SentimentExplainer()\n\n")
        f.write("# Explain a new text\n")
        f.write('result = explainer.predict_with_explanation("This movie was fantastic!")\n\n')
        f.write("print(f'Prediction: {result[\"prediction\"]}')\n")
        f.write("print(f'Confidence: {result[\"confidence\"]:.2%}')\n")
        f.write("```\n\n")

        f.write("---\n\n")
        f.write("## Key Findings\n\n")
        f.write("1. **Token Importance:** SHAP values reveal which specific words drive sentiment predictions\n")
        f.write("2. **Interpretability:** Force plots show clear positive/negative contributions\n")
        f.write("3. **Error Analysis:** False positive/negative examples show where model struggles\n")
        f.write("4. **Confidence Patterns:** Higher confidence predictions typically have stronger token attributions\n\n")


def main():
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for DistilBERT sentiment model")
    parser.add_argument('--text', type=str, help='Explain a specific text')
    parser.add_argument('--num-examples', type=int, default=30, help='Number of examples to explain')
    parser.add_argument('--model-path', type=str, default='../models/distilbert-sentiment/best_model',
                        help='Path to trained model')
    parser.add_argument('--output-dir', type=str, default='../reports/explanations',
                        help='Directory to save explanations')
    parser.add_argument('--eval-results', type=str, default='../reports/evaluation_metrics.json',
                        help='Path to evaluation results JSON')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("TASK 8: SHAP-BASED MODEL INTERPRETABILITY")
    print(f"{'='*80}\n")

    # Initialize explainer
    explainer = SentimentExplainer(model_path=args.model_path)

    # Single text explanation
    if args.text:
        print(f"\nExplaining text: {args.text[:100]}...")
        result = explainer.predict_with_explanation(args.text)

        print(f"\n{'='*80}")
        print("PREDICTION RESULTS")
        print(f"{'='*80}\n")
        print(f"Text: {args.text}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nSHAP values computed successfully!")

        # Display force plot
        shap.plots.text(result['shap_values'])
        return

    # Batch explanation on test set
    print("Loading IMDb test dataset...")
    dataset = load_dataset("imdb", split="test")
    print(f"✅ Loaded {len(dataset)} test samples\n")

    # Select diverse examples
    examples = select_diverse_examples(
        dataset,
        args.eval_results,
        num_examples=args.num_examples
    )

    # Generate explanations
    explanations = generate_explanations(explainer, examples, args.output_dir)

    # Create summary report
    print(f"\nGenerating summary report...")
    report_path = f"{args.output_dir}/explanations_summary.md"
    create_summary_report(explanations, report_path)
    print(f"✅ Summary report saved to: {report_path}")

    print(f"\n{'='*80}")
    print("TASK 8 COMPLETE!")
    print(f"{'='*80}\n")
    print(f"📁 All explanations saved to: {args.output_dir}/")
    print(f"\nTo explain a new text, run:")
    print(f'    python explain.py --text "Your text here"')
    print()


if __name__ == "__main__":
    main()
