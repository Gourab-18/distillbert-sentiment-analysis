# raw data taken and tokenized for DistilBERT
# here we are using imdb dataset from huggingface  

import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset, Dataset, DatasetDict
from transformers import DistilBertTokenizer
from tqdm import tqdm

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TextPreprocessor:

# cleaning raw test
    def __init__(self, lowercase=True, remove_html=True, remove_urls=True,
                 remove_special_chars=False):

# remove all useless things sent in parameters
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_special_chars = remove_special_chars

    def clean_html(self, text):
        if not self.remove_html:
            return text

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Remove HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text)

        # remove <br /> ( many present in imdb reviews )
        text = re.sub(r'<br\s*/?\s*>', ' ', text)

        return text

    def clean_urls(self, text):
        # remove urls
        if not self.remove_urls:
            return text

        # Remove http/https URLs
        text = re.sub(r'http\S+|www\.\S+', ' ', text)

        return text

    def clean_special_chars(self, text):
        if not self.remove_special_chars:

            text = re.sub(r'\s+', ' ', text)
            return text

        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text

    def preprocess(self, text):
# calling all above functions
        if not isinstance(text, str):
            return ""

        # Clean HTML
        text = self.clean_html(text)

        # Clean URLs
        text = self.clean_urls(text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Clean special characters
        text = self.clean_special_chars(text)

        # Strip whitespace
        text = text.strip()

        return text


class DatasetTokenizer:
    """Tokenizer for preparing datasets for DistilBERT."""
    # 

    def __init__(self, model_name='distilbert-base-uncased', max_length=512,
                 padding='max_length', truncation=True):
        
        # we get from Hugging Face

        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        # Load tokenizer
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        # Store config
        self.config = {
            'model_name': model_name,
            'max_length': max_length,
            'padding': padding,
            'truncation': truncation,
            'vocab_size': self.tokenizer.vocab_size,
            'model_max_length': self.tokenizer.model_max_length
        }

    def tokenize_function(self, examples):
        """
        Tokenize a batch of examples.

        Args:
            examples: Dictionary with 'text' key

        Returns:
            Dictionary with tokenized outputs
        """
        return self.tokenizer(
            examples['text'],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors=None  # Return lists for datasets library
        )

    def get_config(self):
        """Return tokenization configuration."""
        return self.config


def load_and_split_data(train_size=20000, val_size=5000, seed=42):

# loads imdb dataset from hugging face and split into train, val, test
    print("=" * 80)
    print("LOADING IMDB DATASET")
    print("=" * 80)

    # Load dataset
    print("\nLoading IMDb dataset from Hugging Face...")
    dataset = load_dataset("imdb")

    # Convert to pandas for splitting
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    print(f"Original train size: {len(train_df):,}")
    print(f"Original test size: {len(test_df):,}")

    # Shuffle train set
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

    # Convert back to Dataset objects
    datasets = DatasetDict({
        'train': Dataset.from_pandas(train_split, preserve_index=False),
        'validation': Dataset.from_pandas(val_split, preserve_index=False),
        'test': Dataset.from_pandas(test_split, preserve_index=False)
    })

    return datasets


def preprocess_dataset(dataset, preprocessor, split_name):

# get dataset from hugging face and process
    print(f"\n[{split_name}] Preprocessing {len(dataset):,} samples...")

    def preprocess_batch(examples):
        """Preprocess a batch of examples."""
        cleaned_texts = [preprocessor.preprocess(text) for text in examples['text']]
        return {'text': cleaned_texts}

    # Apply preprocessing
    processed_dataset = dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=1000,
        desc=f"Preprocessing {split_name}"
    )

    # Show example
    print(f"\nExample from {split_name} set:")
    print(f"  Original: {dataset[0]['text'][:100]}...")
    print(f"  Cleaned:  {processed_dataset[0]['text'][:100]}...")

    return processed_dataset


def tokenize_dataset(dataset, tokenizer_obj, split_name):
    """
    Tokenize a dataset split.

    Args:
        dataset: HuggingFace Dataset
        tokenizer_obj: DatasetTokenizer instance
        split_name: Name of the split

    Returns:
        Tokenized dataset
    """
    print(f"\n[{split_name}] Tokenizing {len(dataset):,} samples...")

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenizer_obj.tokenize_function,
        batched=True,
        batch_size=1000,
        desc=f"Tokenizing {split_name}",
        remove_columns=['text']  # Remove text column after tokenization
    )

    # Show example
    print(f"\nTokenized example from {split_name} set:")
    print(f"  Input IDs length: {len(tokenized_dataset[0]['input_ids'])}")
    print(f"  First 10 tokens: {tokenized_dataset[0]['input_ids'][:10]}")
    print(f"  Attention mask: {tokenized_dataset[0]['attention_mask'][:10]}")

    return tokenized_dataset


def validate_token_lengths(dataset, split_name, max_length=512):
    """
    Validate token length distributions.

    Args:
        dataset: Tokenized dataset
        split_name: Name of the split
        max_length: Maximum allowed token length

    Returns:
        Dictionary with statistics
    """
    print(f"\n[{split_name}] Validating token lengths...")

    # Get all token lengths
    token_lengths = [sum(ex['attention_mask']) for ex in dataset]

    # Calculate statistics
    stats = {
        'split': split_name,
        'num_samples': len(token_lengths),
        'mean_length': float(np.mean(token_lengths)),
        'median_length': float(np.median(token_lengths)),
        'min_length': int(np.min(token_lengths)),
        'max_length': int(np.max(token_lengths)),
        'std_length': float(np.std(token_lengths)),
        'exceeds_limit': int(np.sum(np.array(token_lengths) > max_length)),
        'percentile_95': float(np.percentile(token_lengths, 95)),
        'percentile_99': float(np.percentile(token_lengths, 99))
    }

    # Print statistics
    print(f"\nToken Length Statistics for {split_name}:")
    print(f"  Mean:   {stats['mean_length']:.2f}")
    print(f"  Median: {stats['median_length']:.2f}")
    print(f"  Min:    {stats['min_length']}")
    print(f"  Max:    {stats['max_length']}")
    print(f"  Std:    {stats['std_length']:.2f}")
    print(f"  95th percentile: {stats['percentile_95']:.2f}")
    print(f"  99th percentile: {stats['percentile_99']:.2f}")
    print(f"  Samples exceeding {max_length}: {stats['exceeds_limit']}")

    # Validation check
    if stats['max_length'] > max_length:
        print(f"  ⚠️  WARNING: Some sequences exceed max_length of {max_length}!")
    else:
        print(f"  ✓ All sequences within {max_length} token limit")

    return stats, token_lengths


def plot_token_length_distribution(all_stats, all_lengths, save_path):
    """
    Plot token length distributions for all splits.

    Args:
        all_stats: Dictionary of statistics for each split
        all_lengths: Dictionary of token lengths for each split
        save_path: Path to save the plot
    """
    print("\nGenerating token length distribution plot...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    splits = ['train', 'validation', 'test']
    colors = ['#3498db', '#f39c12', '#e74c3c']

    # Plot 1: Histograms
    for split, color in zip(splits, colors):
        if split in all_lengths:
            axes[0, 0].hist(all_lengths[split], bins=50, alpha=0.6,
                          label=split.capitalize(), color=color)

    axes[0, 0].set_xlabel('Token Length', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Token Length Distribution by Split', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].axvline(x=512, color='red', linestyle='--', label='Max Length (512)')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Box plots
    box_data = [all_lengths[split] for split in splits if split in all_lengths]
    box_labels = [split.capitalize() for split in splits if split in all_lengths]

    bp = axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[0, 1].set_ylabel('Token Length', fontsize=12)
    axes[0, 1].set_title('Token Length Box Plot', fontsize=14, fontweight='bold')
    axes[0, 1].axhline(y=512, color='red', linestyle='--', label='Max Length (512)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Statistics comparison
    splits_present = [s for s in splits if s in all_stats]
    metrics = ['mean_length', 'median_length', 'percentile_95']
    x = np.arange(len(splits_present))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [all_stats[split][metric] for split in splits_present]
        axes[1, 0].bar(x + i*width, values, width, label=metric.replace('_', ' ').title())

    axes[1, 0].set_xlabel('Split', fontsize=12)
    axes[1, 0].set_ylabel('Token Length', fontsize=12)
    axes[1, 0].set_title('Token Length Statistics Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels([s.capitalize() for s in splits_present])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Cumulative distribution
    for split, color in zip(splits, colors):
        if split in all_lengths:
            sorted_lengths = np.sort(all_lengths[split])
            cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
            axes[1, 1].plot(sorted_lengths, cumulative, label=split.capitalize(),
                          color=color, linewidth=2)

    axes[1, 1].set_xlabel('Token Length', fontsize=12)
    axes[1, 1].set_ylabel('Cumulative Percentage (%)', fontsize=12)
    axes[1, 1].set_title('Cumulative Token Length Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].axvline(x=512, color='red', linestyle='--', label='Max Length (512)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Token length distribution plot saved to: {save_path}")
    plt.close()


def save_datasets(datasets, output_dir):
    """
    Save processed datasets to disk.

    Args:
        datasets: DatasetDict with all splits
        output_dir: Output directory
    """
    print(f"\nSaving processed datasets to {output_dir}...")

    # Create directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save as HuggingFace datasets format
    datasets.save_to_disk(output_dir)

    # Calculate and show sizes
    total_size = 0
    for split in datasets.keys():
        split_path = Path(output_dir) / split
        if split_path.exists():
            size = sum(f.stat().st_size for f in split_path.glob('**/*') if f.is_file())
            total_size += size
            print(f"  {split}: {size / (1024*1024):.2f} MB")

    print(f"  Total size: {total_size / (1024*1024):.2f} MB")
    print(f"✓ Datasets saved successfully!")


def main(args):
    """Main preprocessing pipeline."""

    print("\n" + "=" * 80)
    print("DATA PREPROCESSING & TOKENIZATION PIPELINE")
    print("DistilBERT Fine-Tuning for Sentiment Analysis")
    print("=" * 80)

    # Create directories for storing processed data and reports
    Path('../data/processed').mkdir(parents=True, exist_ok=True)
    Path('../reports').mkdir(parents=True, exist_ok=True)

    # Initialized preprocessor
    print("\n" + "=" * 80)
    print("INITIALIZING PREPROCESSOR")
    print("=" * 80)

    preprocessor = TextPreprocessor(
        lowercase=args.lowercase,
        remove_html=args.remove_html,
        remove_urls=args.remove_urls,
        remove_special_chars=args.remove_special_chars
    )

    print(f"\nPreprocessing configuration:")
    print(f"  Lowercase: {args.lowercase}")
    print(f"  Remove HTML: {args.remove_html}")
    print(f"  Remove URLs: {args.remove_urls}")
    print(f"  Remove special chars: {args.remove_special_chars}")

    # Initialize tokenizer
    print("\n" + "=" * 80)
    print("INITIALIZING TOKENIZER")
    print("=" * 80)

    tokenizer_obj = DatasetTokenizer(
        model_name=args.model_name,
        max_length=args.max_length,
        padding=args.padding,
        truncation=args.truncation
    )

    print(f"\nTokenization configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Max length: {args.max_length}")
    print(f"  Padding: {args.padding}")
    print(f"  Truncation: {args.truncation}")
    print(f"  Vocab size: {tokenizer_obj.config['vocab_size']}")

    # Load and split data
    if args.split == 'all':
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        datasets = load_and_split_data(
            train_size=args.train_size,
            val_size=args.val_size,
            seed=args.seed
        )

        splits_to_process = ['train', 'validation', 'test']
    else:
        print(f"\n" + "=" * 80)
        print(f"LOADING {args.split.upper()} SPLIT ONLY")
        print("=" * 80)

        full_datasets = load_and_split_data(
            train_size=args.train_size,
            val_size=args.val_size,
            seed=args.seed
        )

        datasets = DatasetDict({args.split: full_datasets[args.split]})
        splits_to_process = [args.split]

    # Preprocess text
    print("\n" + "=" * 80)
    print("TEXT PREPROCESSING")
    print("=" * 80)

    for split in splits_to_process:
        datasets[split] = preprocess_dataset(datasets[split], preprocessor, split)

    # Tokenize
    print("\n" + "=" * 80)
    print("TOKENIZATION")
    print("=" * 80)

    for split in splits_to_process:
        datasets[split] = tokenize_dataset(datasets[split], tokenizer_obj, split)

    # Validate token lengths
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    all_stats = {}
    all_lengths = {}

    for split in splits_to_process:
        stats, lengths = validate_token_lengths(datasets[split], split, args.max_length)
        all_stats[split] = stats
        all_lengths[split] = lengths

    # Plot token distributions
    if args.split == 'all':
        plot_token_length_distribution(
            all_stats,
            all_lengths,
            '../reports/token_length_distribution.png'
        )

    # Save datasets
    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATA")
    print("=" * 80)

    output_dir = f'../data/processed/tokenized_{args.model_name.replace("/", "_")}'
    save_datasets(datasets, output_dir)

    # Save preprocessing config
    config = {
        'preprocessing': {
            'lowercase': args.lowercase,
            'remove_html': args.remove_html,
            'remove_urls': args.remove_urls,
            'remove_special_chars': args.remove_special_chars
        },
        'tokenization': tokenizer_obj.get_config(),
        'data_splits': {
            'train_size': args.train_size,
            'val_size': args.val_size,
            'seed': args.seed
        },
        'token_statistics': all_stats,
        'output_directory': output_dir
    }

    config_path = '../data/processed/preprocessing_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Configuration saved to: {config_path}")

    # Summary
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE - SUMMARY")
    print("=" * 80)

    print(f"\n📊 Processed Splits:")
    for split in splits_to_process:
        print(f"  {split.capitalize()}: {len(datasets[split]):,} samples")

    print(f"\n📏 Token Length Statistics:")
    for split in splits_to_process:
        print(f"  {split.capitalize()}: Mean={all_stats[split]['mean_length']:.1f}, Max={all_stats[split]['max_length']}")

    print(f"\n💾 Saved Artifacts:")
    print(f"  Tokenized datasets: {output_dir}")
    print(f"  Configuration: {config_path}")
    if args.split == 'all':
        print(f"  Token distribution plot: ../reports/token_length_distribution.png")

    print("\n✅ Preprocessing pipeline completed successfully!")
    print("=" * 80)

    return datasets, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess and tokenize data for DistilBERT fine-tuning"
    )

    # Data arguments
    parser.add_argument(
        '--split',
        type=str,
        default='all',
        choices=['all', 'train', 'validation', 'test'],
        help='Which split to process (default: all)'
    )
    parser.add_argument('--train_size', type=int, default=20000, help='Training set size')
    parser.add_argument('--val_size', type=int, default=5000, help='Validation set size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Preprocessing arguments
    parser.add_argument('--lowercase', action='store_true', default=True, help='Convert to lowercase')
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_false')
    parser.add_argument('--remove_html', action='store_true', default=True, help='Remove HTML tags')
    parser.add_argument('--no-remove-html', dest='remove_html', action='store_false')
    parser.add_argument('--remove_urls', action='store_true', default=True, help='Remove URLs')
    parser.add_argument('--no-remove-urls', dest='remove_urls', action='store_false')
    parser.add_argument('--remove_special_chars', action='store_true', default=False,
                       help='Remove special characters (not recommended for sentiment)')

    # Tokenization arguments
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                       help='DistilBERT model name')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length (DistilBERT limit: 512)')
    parser.add_argument('--padding', type=str, default='max_length',
                       choices=['max_length', 'longest', 'do_not_pad'],
                       help='Padding strategy')
    parser.add_argument('--truncation', action='store_true', default=True,
                       help='Truncate sequences exceeding max_length')

    args = parser.parse_args()

    main(args)
