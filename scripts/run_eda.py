
# EDA Script for IMDb Sentiment Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from collections import Counter
import warnings
import json
from pathlib import Path
from wordcloud import WordCloud, STOPWORDS

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - IMDb Sentiment Dataset")
print("=" * 80)

# path for reports
Path('../reports').mkdir(parents=True, exist_ok=True)

# load dataset
print("\n[1/9] Loading IMDb dataset...")
dataset = load_dataset("imdb")


print("✓ Dataset loaded successfully!")
print(f"\nDataset structure: {dataset}")

# converting to pandas DataFrame for easier analysis
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])
print(train_df.head())

print(f"\nTraining set size: {len(train_df):,}")
print(f"Test set size: {len(test_df):,}")
print(f"Total samples: {len(train_df) + len(test_df):,}")

# show first few samples
print("\nFirst 3 samples from training set:")
print(train_df.head(3)[['text', 'label']].to_string())


# analyzing how many are positive and negative reviews ( 0 negative, 1 positive )
print("\n[2/9] Analyzing class distribution...")

train_class_dist = train_df['label'].value_counts().sort_index()
test_class_dist = test_df['label'].value_counts().sort_index()

print("\nTraining Set Class Distribution:")
print(f"  Negative (0): {train_class_dist[0]:,} ({train_class_dist[0]/len(train_df)*100:.2f}%)")
print(f"  Positive (1): {train_class_dist[1]:,} ({train_class_dist[1]/len(train_df)*100:.2f}%)")

print("\nTest Set Class Distribution:")
print(f"  Negative (0): {test_class_dist[0]:,} ({test_class_dist[0]/len(test_df)*100:.2f}%)")
print(f"  Positive (1): {test_class_dist[1]:,} ({test_class_dist[1]/len(test_df)*100:.2f}%)")

# From above we see distibution is perfectly balanced (50/50) in both train and test sets

# Plotting and saving it in reports folder
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training set
axes[0].bar(['Negative', 'Positive'], train_class_dist.values, color=['#e74c3c', '#2ecc71'])
axes[0].set_title('Training Set - Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_ylim(0, max(train_class_dist.values) * 1.1)
for i, v in enumerate(train_class_dist.values):
    axes[0].text(i, v + 500, f'{v:,}\n({v/len(train_df)*100:.1f}%)', ha='center', fontsize=11)

# Test set
axes[1].bar(['Negative', 'Positive'], test_class_dist.values, color=['#e74c3c', '#2ecc71'])
axes[1].set_title('Test Set - Class Distribution', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_ylim(0, max(test_class_dist.values) * 1.1)
for i, v in enumerate(test_class_dist.values):
    axes[1].text(i, v + 500, f'{v:,}\n({v/len(test_df)*100:.1f}%)', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('../reports/class_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: reports/class_distribution.png")
plt.close()



# Analyzing stats like mean, median, min, max etc.
print("\n[3/9] Analyzing text statistics...")

# Calculate text statistics
train_df['text_length'] = train_df['text'].apply(len)
train_df['word_count'] = train_df['text'].apply(lambda x: len(x.split()))

test_df['text_length'] = test_df['text'].apply(len)
test_df['word_count'] = test_df['text'].apply(lambda x: len(x.split()))


print("\nTraining Set - Text Statistics:")
print(f"  Average character length: {train_df['text_length'].mean():.2f}")
print(f"  Median character length: {train_df['text_length'].median():.2f}")
print(f"  Min character length: {train_df['text_length'].min()}")
print(f"  Max character length: {train_df['text_length'].max()}")
print(f"\n  Average word count: {train_df['word_count'].mean():.2f}")
print(f"  Median word count: {train_df['word_count'].median():.2f}")
print(f"  Min word count: {train_df['word_count'].min()}")
print(f"  Max word count: {train_df['word_count'].max()}")

# Text length distribution by sentiment
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Character length distribution
axes[0, 0].hist([train_df[train_df['label']==0]['text_length'],
                 train_df[train_df['label']==1]['text_length']],
                bins=50, label=['Negative', 'Positive'], color=['#e74c3c', '#2ecc71'], alpha=0.7)
axes[0, 0].set_title('Character Length Distribution by Sentiment', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Character Length', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 3000)

# Word count distribution
axes[0, 1].hist([train_df[train_df['label']==0]['word_count'],
                 train_df[train_df['label']==1]['word_count']],
                bins=50, label=['Negative', 'Positive'], color=['#e74c3c', '#2ecc71'], alpha=0.7)
axes[0, 1].set_title('Word Count Distribution by Sentiment', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Word Count', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].legend()
axes[0, 1].set_xlim(0, 600)

# Box plot for character length
train_df.boxplot(column='text_length', by='label', ax=axes[1, 0])
axes[1, 0].set_title('Character Length by Sentiment', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Sentiment (0=Negative, 1=Positive)', fontsize=12)
axes[1, 0].set_ylabel('Character Length', fontsize=12)
axes[1, 0].get_figure().suptitle('')

# Box plot for word count
train_df.boxplot(column='word_count', by='label', ax=axes[1, 1])
axes[1, 1].set_title('Word Count by Sentiment', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Sentiment (0=Negative, 1=Positive)', fontsize=12)
axes[1, 1].set_ylabel('Word Count', fontsize=12)
axes[1, 1].get_figure().suptitle('')

plt.tight_layout()
plt.savefig('../reports/text_length_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: reports/text_length_analysis.png")
plt.close()


# seeing  sample reviews
print("\n[4/9] Displaying sample reviews...")

print("\n" + "=" * 80)
print("SAMPLE NEGATIVE REVIEWS")
print("=" * 80)

negative_samples = train_df[train_df['label'] == 0].sample(3, random_state=42)
for idx, (i, row) in enumerate(negative_samples.iterrows(), 1):
    print(f"\nNegative Review {idx}:")
    print(f"Length: {row['word_count']} words")
    print(f"Text: {row['text'][:200]}...")
    print("-" * 80)

print("\n" + "=" * 80)
print("SAMPLE POSITIVE REVIEWS")
print("=" * 80)

positive_samples = train_df[train_df['label'] == 1].sample(3, random_state=42)
for idx, (i, row) in enumerate(positive_samples.iterrows(), 1):
    print(f"\nPositive Review {idx}:")
    print(f"Length: {row['word_count']} words")
    print(f"Text: {row['text'][:200]}...")
    print("-" * 80)

# 
print("\n[5/9] Generating word clouds...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# here more frequent worsds will appear larger

# STOPWORDS removes useless words like 'the', 'is', 'and' etc.
# Negative reviews word cloud
negative_text = ' '.join(train_df[train_df['label'] == 0]['text'].sample(1000, random_state=42))
wordcloud_neg = WordCloud(width=800, height=400,
                          background_color='white',
                          stopwords=STOPWORDS,
                          colormap='Reds',
                          max_words=100).generate(negative_text)

axes[0].imshow(wordcloud_neg, interpolation='bilinear')
axes[0].set_title('Negative Reviews - Word Cloud', fontsize=16, fontweight='bold')
axes[0].axis('off')

# Positive reviews word cloud
positive_text = ' '.join(train_df[train_df['label'] == 1]['text'].sample(1000, random_state=42))
wordcloud_pos = WordCloud(width=800, height=400,
                          background_color='white',
                          stopwords=STOPWORDS,
                          colormap='Greens',
                          max_words=100).generate(positive_text)

axes[1].imshow(wordcloud_pos, interpolation='bilinear')
axes[1].set_title('Positive Reviews - Word Cloud', fontsize=16, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('../reports/wordclouds.png', dpi=300, bbox_inches='tight')
print("✓ Saved: reports/wordclouds.png")
plt.close()


# analyzing top words
print("\n[6/9] Analyzing top words...")

def get_top_words(text_series, n=20):
    """Get top n words from text series"""
    words = ' '.join(text_series).lower().split()
    stopwords = set(STOPWORDS)
    words = [word for word in words if word not in stopwords and len(word) > 3]
    return Counter(words).most_common(n)

negative_words = get_top_words(train_df[train_df['label'] == 0]['text'].sample(5000, random_state=42))
positive_words = get_top_words(train_df[train_df['label'] == 1]['text'].sample(5000, random_state=42))

print("\nTop 10 words in Negative reviews:")
for word, count in negative_words[:10]:
    print(f"  {word}: {count}")

print("\nTop 10 words in Positive reviews:")
for word, count in positive_words[:10]:
    print(f"  {word}: {count}")

# Visualize top words
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Negative words
words_neg, counts_neg = zip(*negative_words[:15])
axes[0].barh(words_neg, counts_neg, color='#e74c3c')
axes[0].set_title('Top 15 Words in Negative Reviews', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Frequency', fontsize=12)
axes[0].invert_yaxis()

# Positive words
words_pos, counts_pos = zip(*positive_words[:15])
axes[1].barh(words_pos, counts_pos, color='#2ecc71')
axes[1].set_title('Top 15 Words in Positive Reviews', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Frequency', fontsize=12)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('../reports/top_words.png', dpi=300, bbox_inches='tight')
print("✓ Saved: reports/top_words.png")
plt.close()



print("\n[7/9] Documenting data split strategy...")

total_samples = len(train_df) + len(test_df)
proposed_train = 20000
proposed_val = 5000
proposed_test = len(test_df)

print("\nProposed Data Split:")
print(f"  Total samples: {total_samples:,}")
print(f"\n  Training: {proposed_train:,} ({proposed_train/total_samples*100:.1f}%)")
print(f"  Validation: {proposed_val:,} ({proposed_val/total_samples*100:.1f}%)")
print(f"  Test: {proposed_test:,} ({proposed_test/total_samples*100:.1f}%)")

# Visualize split
fig, ax = plt.subplots(figsize=(10, 6))
splits = ['Train', 'Validation', 'Test']
sizes = [proposed_train, proposed_val, proposed_test]
colors = ['#3498db', '#f39c12', '#e74c3c']
explode = (0.05, 0.05, 0.05)

ax.pie(sizes, explode=explode, labels=splits, colors=colors, autopct='%1.1f%%',
       shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax.set_title('Proposed Train/Validation/Test Split', fontsize=16, fontweight='bold')

plt.savefig('../reports/data_split.png', dpi=300, bbox_inches='tight')
print("✓ Saved: reports/data_split.png")
plt.close()


# 8. DATA STATISTICS SUMMARY
print("\n[8/9] Generating data statistics summary...")

stats_summary = {
    "dataset_name": "IMDb Movie Reviews",
    "dataset_source": "Hugging Face Datasets",
    "task": "Binary Sentiment Classification",
    "total_samples": int(total_samples),
    "original_split": {
        "train": int(len(train_df)),
        "test": int(len(test_df))
    },
    "proposed_split": {
        "train": proposed_train,
        "validation": proposed_val,
        "test": proposed_test,
        "train_percentage": round(proposed_train/total_samples*100, 2),
        "validation_percentage": round(proposed_val/total_samples*100, 2),
        "test_percentage": round(proposed_test/total_samples*100, 2)
    },
    "class_distribution": {
        "train": {
            "negative": int(train_class_dist[0]),
            "positive": int(train_class_dist[1]),
            "balance": "Perfectly balanced (50/50)"
        },
        "test": {
            "negative": int(test_class_dist[0]),
            "positive": int(test_class_dist[1]),
            "balance": "Perfectly balanced (50/50)"
        }
    },
    "text_statistics": {
        "character_length": {
            "mean": round(train_df['text_length'].mean(), 2),
            "median": round(train_df['text_length'].median(), 2),
            "min": int(train_df['text_length'].min()),
            "max": int(train_df['text_length'].max()),
            "std": round(train_df['text_length'].std(), 2)
        },
        "word_count": {
            "mean": round(train_df['word_count'].mean(), 2),
            "median": round(train_df['word_count'].median(), 2),
            "min": int(train_df['word_count'].min()),
            "max": int(train_df['word_count'].max()),
            "std": round(train_df['word_count'].std(), 2)
        }
    },
    "potential_challenges": [
        "Variable text lengths - need to handle padding/truncation",
        "Long reviews may exceed model max sequence length (512 tokens for BERT models)",
        "HTML tags and special characters may need cleaning",
        "Some reviews contain spoilers which may affect sentiment"
    ],
    "recommendations": [
        "Use max_length=512 for tokenization (DistilBERT limit)",
        "Apply truncation for longer reviews",
        "Consider data augmentation if needed",
        "Monitor for class imbalance in validation split"
    ]
}

# Save to JSON
with open('../reports/data_statistics_summary.json', 'w') as f:
    json.dump(stats_summary, f, indent=2)

print("✓ Saved: reports/data_statistics_summary.json")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n[9/9] Generating final summary...")

print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS COMPLETE")
print("=" * 80)

print("\n📊 Generated Reports:")
print("  ✓ reports/class_distribution.png")
print("  ✓ reports/text_length_analysis.png")
print("  ✓ reports/wordclouds.png")
print("  ✓ reports/top_words.png")
print("  ✓ reports/data_split.png")
print("  ✓ reports/data_statistics_summary.json")

print("\n📈 Key Findings:")
print(f"  • Dataset: IMDb Movie Reviews ({total_samples:,} samples)")
print(f"  • Class Balance: Perfect 50/50 split")
print(f"  • Average Review Length: {train_df['word_count'].mean():.0f} words")
print(f"  • Proposed Split: {proposed_train:,} train / {proposed_val:,} val / {proposed_test:,} test")


