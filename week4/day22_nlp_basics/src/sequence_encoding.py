"""
Day 22: NLP Basics - Sequence Padding and Encoding
Topics: Text encoding, padding, and sequence preparation for deep learning
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional
import json
import pickle


class TextEncoder:
    """
    Encode text into numerical sequences for deep learning models
    """

    def __init__(self, num_words: int = None, oov_token: str = '<OOV>',
                 lower: bool = True):
        """
        Initialize text encoder

        Args:
            num_words: Maximum number of words to keep (vocabulary size)
            oov_token: Token for out-of-vocabulary words
            lower: Convert text to lowercase
        """
        self.num_words = num_words
        self.oov_token = oov_token
        self.lower = lower

        # Special tokens
        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'

        # Indices for special tokens
        self.word_index = {}
        self.index_word = {}
        self.word_counts = Counter()
        self.document_count = 0

    def fit(self, texts: List[str]):
        """
        Build vocabulary from texts

        Args:
            texts: List of text strings
        """
        self.document_count = len(texts)

        # Count words
        for text in texts:
            if self.lower:
                text = text.lower()
            words = text.split()
            self.word_counts.update(words)

        # Create word index
        # Reserve indices for special tokens
        self.word_index = {
            self.pad_token: 0,
            self.oov_token: 1,
            self.start_token: 2,
            self.end_token: 3
        }

        # Add words by frequency
        sorted_words = self.word_counts.most_common(self.num_words)
        for word, count in sorted_words:
            if word not in self.word_index:
                self.word_index[word] = len(self.word_index)

        # Create reverse index
        self.index_word = {i: w for w, i in self.word_index.items()}

        print(f"Vocabulary size: {len(self.word_index)}")
        print(f"Total documents: {self.document_count}")

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Convert texts to sequences of integers

        Args:
            texts: List of text strings

        Returns:
            List of integer sequences
        """
        sequences = []
        oov_idx = self.word_index.get(self.oov_token, 1)

        for text in texts:
            if self.lower:
                text = text.lower()
            words = text.split()
            sequence = [
                self.word_index.get(word, oov_idx)
                for word in words
            ]
            sequences.append(sequence)

        return sequences

    def sequences_to_texts(self, sequences: List[List[int]]) -> List[str]:
        """
        Convert sequences back to text

        Args:
            sequences: List of integer sequences

        Returns:
            List of text strings
        """
        texts = []
        for sequence in sequences:
            words = [
                self.index_word.get(idx, self.oov_token)
                for idx in sequence
                if idx != 0  # Skip padding
            ]
            texts.append(' '.join(words))
        return texts

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.word_index)

    def save(self, path: str):
        """Save encoder to file"""
        data = {
            'word_index': self.word_index,
            'num_words': self.num_words,
            'oov_token': self.oov_token,
            'lower': self.lower
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load encoder from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.word_index = data['word_index']
        self.index_word = {int(i): w for w, i in self.word_index.items()}
        self.num_words = data.get('num_words')
        self.oov_token = data.get('oov_token', '<OOV>')
        self.lower = data.get('lower', True)


class SequencePadder:
    """
    Pad sequences to uniform length
    """

    @staticmethod
    def pad_sequences(sequences: List[List[int]],
                      maxlen: int = None,
                      padding: str = 'pre',
                      truncating: str = 'pre',
                      value: int = 0) -> np.ndarray:
        """
        Pad sequences to the same length

        Args:
            sequences: List of sequences (lists of integers)
            maxlen: Maximum length (if None, uses the longest sequence)
            padding: 'pre' or 'post' - where to add padding
            truncating: 'pre' or 'post' - where to truncate
            value: Padding value (usually 0)

        Returns:
            2D numpy array of padded sequences
        """
        if maxlen is None:
            maxlen = max(len(seq) for seq in sequences)

        # Initialize output array
        result = np.full((len(sequences), maxlen), value, dtype=np.int32)

        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                continue

            # Truncate if necessary
            if len(seq) > maxlen:
                if truncating == 'pre':
                    trunc = seq[-maxlen:]
                else:
                    trunc = seq[:maxlen]
            else:
                trunc = seq

            # Pad and insert
            if padding == 'pre':
                result[i, -len(trunc):] = trunc
            else:
                result[i, :len(trunc)] = trunc

        return result

    @staticmethod
    def get_sequence_lengths(sequences: List[List[int]]) -> np.ndarray:
        """Get lengths of all sequences"""
        return np.array([len(seq) for seq in sequences])

    @staticmethod
    def analyze_sequence_lengths(sequences: List[List[int]]) -> Dict:
        """Analyze sequence length distribution"""
        lengths = [len(seq) for seq in sequences]
        return {
            'min': min(lengths),
            'max': max(lengths),
            'mean': np.mean(lengths),
            'median': np.median(lengths),
            'std': np.std(lengths),
            'percentile_95': np.percentile(lengths, 95),
            'percentile_99': np.percentile(lengths, 99)
        }


class LabelEncoder:
    """
    Encode categorical labels
    """

    def __init__(self):
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}

    def fit(self, labels: List[str]):
        """Fit encoder on labels"""
        self.classes = sorted(set(labels))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        print(f"Classes: {self.classes}")

    def transform(self, labels: List[str]) -> np.ndarray:
        """Transform labels to indices"""
        return np.array([self.class_to_idx[label] for label in labels])

    def inverse_transform(self, indices: np.ndarray) -> List[str]:
        """Transform indices back to labels"""
        return [self.idx_to_class[int(idx)] for idx in indices]

    def fit_transform(self, labels: List[str]) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(labels)
        return self.transform(labels)

    def get_num_classes(self) -> int:
        """Get number of classes"""
        return len(self.classes)


class OneHotEncoder:
    """
    One-hot encode labels
    """

    @staticmethod
    def encode(labels: np.ndarray, num_classes: int = None) -> np.ndarray:
        """
        Convert integer labels to one-hot encoding

        Args:
            labels: 1D array of integer labels
            num_classes: Number of classes (if None, inferred from data)

        Returns:
            2D one-hot encoded array
        """
        if num_classes is None:
            num_classes = np.max(labels) + 1

        one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
        one_hot[np.arange(len(labels)), labels] = 1

        return one_hot

    @staticmethod
    def decode(one_hot: np.ndarray) -> np.ndarray:
        """Convert one-hot encoding back to labels"""
        return np.argmax(one_hot, axis=1)


class CharacterEncoder:
    """
    Character-level encoding for text
    """

    def __init__(self, max_chars: int = None):
        self.max_chars = max_chars
        self.char_index = {}
        self.index_char = {}

    def fit(self, texts: List[str]):
        """Build character vocabulary"""
        chars = set()
        for text in texts:
            chars.update(text)

        # Sort for consistency
        chars = sorted(chars)

        # Reserve 0 for padding
        self.char_index = {char: i + 1 for i, char in enumerate(chars)}
        self.char_index['<PAD>'] = 0
        self.index_char = {i: c for c, i in self.char_index.items()}

        if self.max_chars:
            # Keep only most common characters
            self.char_index = dict(list(self.char_index.items())[:self.max_chars])
            self.index_char = {i: c for c, i in self.char_index.items()}

        print(f"Character vocabulary size: {len(self.char_index)}")

    def encode(self, text: str) -> List[int]:
        """Encode text to character indices"""
        return [self.char_index.get(c, 0) for c in text]

    def decode(self, indices: List[int]) -> str:
        """Decode character indices to text"""
        return ''.join(self.index_char.get(i, '') for i in indices if i != 0)


def demonstrate_encoding():
    """Demonstrate sequence encoding and padding"""

    print("=" * 60)
    print("SEQUENCE ENCODING AND PADDING DEMONSTRATION")
    print("=" * 60)

    # Sample texts
    texts = [
        "I love machine learning",
        "Deep learning is amazing",
        "Natural language processing is fascinating",
        "Neural networks can learn complex patterns",
        "Word embeddings capture semantic meaning",
        "This is a short sentence",
        "A"
    ]

    labels = ['positive', 'positive', 'positive', 'positive', 'positive', 'neutral', 'neutral']

    # === TEXT ENCODING ===
    print("\n1. TEXT ENCODING:")
    print("-" * 40)

    encoder = TextEncoder(num_words=1000)
    encoder.fit(texts)

    print(f"\nWord Index (first 20):")
    for word, idx in list(encoder.word_index.items())[:20]:
        print(f"  '{word}': {idx}")

    sequences = encoder.texts_to_sequences(texts)

    print(f"\nOriginal texts and their sequences:")
    for text, seq in zip(texts, sequences):
        print(f"  Text: '{text}'")
        print(f"  Sequence: {seq}")
        print()

    # === SEQUENCE PADDING ===
    print("\n2. SEQUENCE PADDING:")
    print("-" * 40)

    # Analyze lengths
    length_stats = SequencePadder.analyze_sequence_lengths(sequences)
    print(f"\nSequence length statistics:")
    for key, value in length_stats.items():
        print(f"  {key}: {value:.2f}")

    # Pad with different strategies
    maxlen = 6

    # Pre-padding (default)
    padded_pre = SequencePadder.pad_sequences(sequences, maxlen=maxlen, padding='pre')
    print(f"\nPre-padding (maxlen={maxlen}):")
    print(padded_pre)

    # Post-padding
    padded_post = SequencePadder.pad_sequences(sequences, maxlen=maxlen, padding='post')
    print(f"\nPost-padding (maxlen={maxlen}):")
    print(padded_post)

    # Pre-truncating
    padded_trunc_pre = SequencePadder.pad_sequences(sequences, maxlen=4, truncating='pre')
    print(f"\nPre-truncating (maxlen=4):")
    print(padded_trunc_pre)

    # Post-truncating
    padded_trunc_post = SequencePadder.pad_sequences(sequences, maxlen=4, truncating='post')
    print(f"\nPost-truncating (maxlen=4):")
    print(padded_trunc_post)

    # === LABEL ENCODING ===
    print("\n3. LABEL ENCODING:")
    print("-" * 40)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    print(f"\nOriginal labels: {labels}")
    print(f"Encoded labels: {encoded_labels}")

    # One-hot encoding
    one_hot_labels = OneHotEncoder.encode(encoded_labels)
    print(f"\nOne-hot encoded labels:")
    print(one_hot_labels)

    # === CHARACTER ENCODING ===
    print("\n4. CHARACTER ENCODING:")
    print("-" * 40)

    char_encoder = CharacterEncoder()
    char_encoder.fit(texts)

    sample_text = "Hello ML"
    char_encoded = char_encoder.encode(sample_text)
    print(f"\nText: '{sample_text}'")
    print(f"Character encoded: {char_encoded}")

    decoded = char_encoder.decode(char_encoded)
    print(f"Decoded: '{decoded}'")

    # === CONVERT BACK TO TEXT ===
    print("\n5. SEQUENCE TO TEXT CONVERSION:")
    print("-" * 40)

    # Decode padded sequences
    decoded_texts = encoder.sequences_to_texts(padded_post.tolist())
    print("\nDecoded from post-padded sequences:")
    for original, decoded in zip(texts, decoded_texts):
        print(f"  Original: '{original}'")
        print(f"  Decoded:  '{decoded}'")
        print()

    return encoder, label_encoder


def create_data_pipeline(texts: List[str], labels: List[str],
                         max_features: int = 10000,
                         maxlen: int = 100,
                         padding: str = 'post') -> Tuple:
    """
    Complete data pipeline for text classification

    Args:
        texts: List of text strings
        labels: List of label strings
        max_features: Maximum vocabulary size
        maxlen: Maximum sequence length
        padding: Padding strategy ('pre' or 'post')

    Returns:
        Tuple of (X_padded, y_encoded, text_encoder, label_encoder)
    """
    # Text encoding
    text_encoder = TextEncoder(num_words=max_features)
    text_encoder.fit(texts)
    sequences = text_encoder.texts_to_sequences(texts)

    # Padding
    X_padded = SequencePadder.pad_sequences(sequences, maxlen=maxlen, padding=padding)

    # Label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    print(f"\nData Pipeline Summary:")
    print(f"  Vocabulary size: {text_encoder.get_vocab_size()}")
    print(f"  Sequence shape: {X_padded.shape}")
    print(f"  Number of classes: {label_encoder.get_num_classes()}")

    return X_padded, y_encoded, text_encoder, label_encoder


if __name__ == "__main__":
    # Run demonstration
    encoder, label_encoder = demonstrate_encoding()

    # Test complete pipeline
    print("\n" + "=" * 60)
    print("COMPLETE DATA PIPELINE EXAMPLE")
    print("=" * 60)

    sample_texts = [
        "This movie is excellent and very entertaining",
        "Terrible film, waste of time",
        "Average movie, nothing special",
        "Absolutely loved it, great acting",
        "Boring and predictable plot"
    ]
    sample_labels = ['positive', 'negative', 'neutral', 'positive', 'negative']

    X, y, text_enc, label_enc = create_data_pipeline(
        sample_texts, sample_labels,
        max_features=1000,
        maxlen=10
    )

    print(f"\nProcessed data:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  X sample:\n{X}")
    print(f"  y sample: {y}")
