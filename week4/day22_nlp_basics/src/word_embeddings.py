"""
Day 22: NLP Basics - Word Embeddings
Topics: Word2Vec, GloVe, FastText
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import os
import urllib.request
import zipfile
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Gensim for Word2Vec
try:
    from gensim.models import Word2Vec, KeyedVectors
    from gensim.models.fasttext import FastText
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Gensim not installed. Install with: pip install gensim")


class Word2VecTrainer:
    """
    Train and use Word2Vec embeddings
    """

    def __init__(self, vector_size=100, window=5, min_count=1, workers=4, sg=0):
        """
        Initialize Word2Vec trainer

        Args:
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Ignores words with frequency lower than this
            workers: Number of worker threads
            sg: Training algorithm: 0 for CBOW, 1 for Skip-gram
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.model = None

    def train(self, sentences: List[List[str]], epochs=10):
        """
        Train Word2Vec model on tokenized sentences

        Args:
            sentences: List of tokenized sentences (list of lists)
            epochs: Number of training epochs
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is required for Word2Vec training")

        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=epochs
        )
        print(f"Word2Vec model trained with vocabulary size: {len(self.model.wv)}")

    def get_vector(self, word: str) -> np.ndarray:
        """Get vector for a word"""
        if self.model and word in self.model.wv:
            return self.model.wv[word]
        return None

    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """Find most similar words"""
        if self.model and word in self.model.wv:
            return self.model.wv.most_similar(word, topn=topn)
        return []

    def similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words"""
        if self.model and word1 in self.model.wv and word2 in self.model.wv:
            return self.model.wv.similarity(word1, word2)
        return 0.0

    def analogy(self, positive: List[str], negative: List[str], topn: int = 5):
        """
        Word analogy: king - man + woman = queen
        positive = ['king', 'woman'], negative = ['man']
        """
        if self.model:
            return self.model.wv.most_similar(positive=positive, negative=negative, topn=topn)
        return []

    def save(self, path: str):
        """Save the model"""
        if self.model:
            self.model.save(path)

    def load(self, path: str):
        """Load a saved model"""
        self.model = Word2Vec.load(path)


class GloVeLoader:
    """
    Load and use pre-trained GloVe embeddings
    """

    def __init__(self, embedding_dim: int = 100):
        """
        Initialize GloVe loader

        Args:
            embedding_dim: Dimension of embeddings (50, 100, 200, or 300)
        """
        self.embedding_dim = embedding_dim
        self.embeddings = {}
        self.vocab = set()

    def load_glove(self, glove_path: str):
        """
        Load GloVe embeddings from file

        Args:
            glove_path: Path to GloVe file (e.g., glove.6B.100d.txt)
        """
        print(f"Loading GloVe embeddings from {glove_path}...")

        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.embeddings[word] = vector
                self.vocab.add(word)

        print(f"Loaded {len(self.embeddings)} word vectors")

    def create_synthetic_embeddings(self, vocab_size: int = 1000):
        """
        Create synthetic embeddings for demonstration
        (Use when GloVe files are not available)
        """
        np.random.seed(42)

        # Common words for demonstration
        common_words = [
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'need',
            'king', 'queen', 'man', 'woman', 'prince', 'princess',
            'boy', 'girl', 'father', 'mother', 'son', 'daughter',
            'good', 'bad', 'happy', 'sad', 'big', 'small', 'fast', 'slow',
            'computer', 'machine', 'learning', 'deep', 'neural', 'network',
            'data', 'science', 'python', 'code', 'algorithm', 'model',
            'cat', 'dog', 'bird', 'fish', 'animal', 'pet', 'food', 'water',
            'house', 'home', 'car', 'road', 'city', 'country', 'world'
        ]

        # Create embeddings with some semantic structure
        for i, word in enumerate(common_words):
            # Base random vector
            vector = np.random.randn(self.embedding_dim).astype('float32')

            # Add some structure for related words
            if word in ['king', 'queen', 'prince', 'princess']:
                vector[:10] += 1.0  # Royalty cluster
            if word in ['man', 'boy', 'father', 'son']:
                vector[10:20] += 1.0  # Male cluster
            if word in ['woman', 'girl', 'mother', 'daughter']:
                vector[10:20] -= 1.0  # Female cluster (opposite direction)
            if word in ['computer', 'machine', 'neural', 'network', 'algorithm']:
                vector[20:30] += 1.0  # Tech cluster

            # Normalize
            vector = vector / np.linalg.norm(vector)
            self.embeddings[word] = vector
            self.vocab.add(word)

        print(f"Created {len(self.embeddings)} synthetic word vectors")

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get vector for a word"""
        return self.embeddings.get(word.lower())

    def similarity(self, word1: str, word2: str) -> float:
        """Calculate cosine similarity between two words"""
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)

        if vec1 is not None and vec2 is not None:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return 0.0

    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """Find most similar words using cosine similarity"""
        vec = self.get_vector(word)
        if vec is None:
            return []

        similarities = []
        for w, v in self.embeddings.items():
            if w != word.lower():
                sim = np.dot(vec, v) / (np.linalg.norm(vec) * np.linalg.norm(v))
                similarities.append((w, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

    def analogy(self, word1: str, word2: str, word3: str, topn: int = 5):
        """
        Word analogy: word1 - word2 + word3 = ?
        Example: king - man + woman = queen
        """
        v1 = self.get_vector(word1)
        v2 = self.get_vector(word2)
        v3 = self.get_vector(word3)

        if v1 is None or v2 is None or v3 is None:
            return []

        # Compute analogy vector
        result_vec = v1 - v2 + v3

        # Find most similar words
        similarities = []
        exclude = {word1.lower(), word2.lower(), word3.lower()}

        for w, v in self.embeddings.items():
            if w not in exclude:
                sim = np.dot(result_vec, v) / (np.linalg.norm(result_vec) * np.linalg.norm(v))
                similarities.append((w, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]


class EmbeddingMatrix:
    """
    Create embedding matrices for neural network models
    """

    def __init__(self, embeddings: Dict[str, np.ndarray], embedding_dim: int):
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim

    def create_embedding_matrix(self, word_index: Dict[str, int],
                                max_features: int = None) -> np.ndarray:
        """
        Create embedding matrix for Keras Embedding layer

        Args:
            word_index: Dictionary mapping words to indices
            max_features: Maximum vocabulary size

        Returns:
            Embedding matrix of shape (vocab_size, embedding_dim)
        """
        if max_features is None:
            max_features = len(word_index) + 1

        embedding_matrix = np.zeros((max_features, self.embedding_dim))
        found_count = 0

        for word, i in word_index.items():
            if i >= max_features:
                continue

            embedding_vector = self.embeddings.get(word.lower())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                found_count += 1
            else:
                # Random initialization for OOV words
                embedding_matrix[i] = np.random.normal(0, 0.1, self.embedding_dim)

        coverage = found_count / min(len(word_index), max_features)
        print(f"Embedding coverage: {coverage:.2%} ({found_count}/{min(len(word_index), max_features)})")

        return embedding_matrix


def demonstrate_word2vec():
    """Demonstrate Word2Vec training and usage"""

    print("=" * 60)
    print("WORD2VEC DEMONSTRATION")
    print("=" * 60)

    if not GENSIM_AVAILABLE:
        print("Gensim not available. Skipping Word2Vec demo.")
        return

    # Sample corpus
    corpus = [
        ['machine', 'learning', 'is', 'subset', 'of', 'artificial', 'intelligence'],
        ['deep', 'learning', 'uses', 'neural', 'networks'],
        ['neural', 'networks', 'are', 'inspired', 'by', 'human', 'brain'],
        ['natural', 'language', 'processing', 'deals', 'with', 'text', 'data'],
        ['word', 'embeddings', 'capture', 'semantic', 'meaning'],
        ['word2vec', 'creates', 'dense', 'word', 'vectors'],
        ['skip', 'gram', 'predicts', 'context', 'words'],
        ['cbow', 'predicts', 'target', 'word', 'from', 'context'],
        ['transfer', 'learning', 'uses', 'pretrained', 'models'],
        ['bert', 'is', 'a', 'transformer', 'based', 'model'],
        ['transformers', 'use', 'attention', 'mechanism'],
        ['attention', 'is', 'all', 'you', 'need'],
        ['king', 'is', 'to', 'man', 'as', 'queen', 'is', 'to', 'woman'],
        ['the', 'king', 'rules', 'the', 'kingdom'],
        ['the', 'queen', 'rules', 'alongside', 'the', 'king']
    ]

    # Train CBOW model
    print("\n1. Training CBOW Model:")
    print("-" * 40)
    cbow_trainer = Word2VecTrainer(vector_size=50, window=3, sg=0)
    cbow_trainer.train(corpus, epochs=100)

    # Train Skip-gram model
    print("\n2. Training Skip-gram Model:")
    print("-" * 40)
    skipgram_trainer = Word2VecTrainer(vector_size=50, window=3, sg=1)
    skipgram_trainer.train(corpus, epochs=100)

    # Test similarity
    print("\n3. Word Similarities (Skip-gram):")
    print("-" * 40)
    test_pairs = [
        ('learning', 'deep'),
        ('neural', 'networks'),
        ('word', 'embeddings')
    ]
    for w1, w2 in test_pairs:
        sim = skipgram_trainer.similarity(w1, w2)
        print(f"  similarity('{w1}', '{w2}'): {sim:.4f}")

    # Most similar words
    print("\n4. Most Similar Words:")
    print("-" * 40)
    for word in ['learning', 'neural', 'word']:
        if word in skipgram_trainer.model.wv:
            similar = skipgram_trainer.most_similar(word, topn=3)
            print(f"  Similar to '{word}': {similar}")

    return skipgram_trainer


def demonstrate_glove():
    """Demonstrate GloVe embeddings"""

    print("\n" + "=" * 60)
    print("GLOVE EMBEDDINGS DEMONSTRATION")
    print("=" * 60)

    # Create synthetic embeddings for demo
    glove = GloVeLoader(embedding_dim=100)
    glove.create_synthetic_embeddings()

    # Test similarity
    print("\n1. Word Similarities:")
    print("-" * 40)
    test_pairs = [
        ('king', 'queen'),
        ('man', 'woman'),
        ('computer', 'machine'),
        ('cat', 'dog')
    ]
    for w1, w2 in test_pairs:
        sim = glove.similarity(w1, w2)
        print(f"  similarity('{w1}', '{w2}'): {sim:.4f}")

    # Most similar words
    print("\n2. Most Similar Words:")
    print("-" * 40)
    for word in ['king', 'computer', 'happy']:
        similar = glove.most_similar(word, topn=5)
        if similar:
            print(f"  Similar to '{word}':")
            for w, s in similar:
                print(f"    {w}: {s:.4f}")

    # Word analogies
    print("\n3. Word Analogies (king - man + woman = ?):")
    print("-" * 40)
    analogy = glove.analogy('king', 'man', 'woman', topn=5)
    if analogy:
        for w, s in analogy:
            print(f"  {w}: {s:.4f}")

    return glove


def visualize_embeddings(embeddings_dict: Dict[str, np.ndarray], words: List[str] = None):
    """
    Visualize word embeddings using t-SNE or PCA
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
    except ImportError:
        print("sklearn and matplotlib required for visualization")
        return

    if words is None:
        words = list(embeddings_dict.keys())[:50]

    # Get vectors
    vectors = []
    valid_words = []
    for word in words:
        if word in embeddings_dict:
            vectors.append(embeddings_dict[word])
            valid_words.append(word)

    if len(vectors) < 2:
        print("Not enough vectors to visualize")
        return

    vectors = np.array(vectors)

    # Use PCA if vectors are high-dimensional
    if vectors.shape[1] > 50:
        pca = PCA(n_components=50)
        vectors = pca.fit_transform(vectors)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(vectors) - 1))
    vectors_2d = tsne.fit_transform(vectors)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6)

    for i, word in enumerate(valid_words):
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
                     fontsize=9, alpha=0.8)

    plt.title("Word Embeddings Visualization (t-SNE)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig('word_embeddings_visualization.png', dpi=150)
    plt.close()
    print("Visualization saved to 'word_embeddings_visualization.png'")


if __name__ == "__main__":
    # Demonstrate Word2Vec
    w2v_model = demonstrate_word2vec()

    # Demonstrate GloVe
    glove_model = demonstrate_glove()

    # Visualize if possible
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    visualize_embeddings(glove_model.embeddings)
