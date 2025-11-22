"""
Day 23: Advanced NLP - Attention Mechanism Theory
Understanding the attention mechanism in transformers
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
import math

# Optional: PyTorch for neural network implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AttentionFromScratch:
    """
    Implement attention mechanism from scratch using NumPy
    """

    @staticmethod
    def softmax(x, axis=-1):
        """Compute softmax values"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    @staticmethod
    def scaled_dot_product_attention(Q, K, V, mask=None):
        """
        Scaled Dot-Product Attention

        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

        Args:
            Q: Query matrix [batch, seq_len, d_k]
            K: Key matrix [batch, seq_len, d_k]
            V: Value matrix [batch, seq_len, d_v]
            mask: Optional mask for padding/causality

        Returns:
            attention_output: Weighted sum of values
            attention_weights: Attention probability distribution
        """
        d_k = K.shape[-1]

        # Compute attention scores: QK^T
        scores = np.matmul(Q, K.transpose(0, 2, 1))

        # Scale by sqrt(d_k)
        scores = scores / np.sqrt(d_k)

        # Apply mask (if provided)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Apply softmax to get attention weights
        attention_weights = AttentionFromScratch.softmax(scores, axis=-1)

        # Compute weighted sum of values
        output = np.matmul(attention_weights, V)

        return output, attention_weights

    @staticmethod
    def create_causal_mask(seq_len):
        """
        Create causal mask for autoregressive attention
        (used in decoder to prevent attending to future tokens)
        """
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mask == 0  # True where attention is allowed


class MultiHeadAttention:
    """
    Multi-Head Attention implementation using NumPy
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize Multi-Head Attention

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Initialize weights (random initialization for demo)
        np.random.seed(42)
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

    def split_heads(self, x):
        """Split the last dimension into (num_heads, d_k)"""
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # [batch, heads, seq_len, d_k]

    def combine_heads(self, x):
        """Combine heads back to original shape"""
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)  # [batch, seq_len, heads, d_k]
        return x.reshape(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        Multi-Head Attention forward pass

        Args:
            query: Query tensor [batch, seq_len, d_model]
            key: Key tensor [batch, seq_len, d_model]
            value: Value tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            output: Attention output
            attention_weights: Attention weights from all heads
        """
        batch_size = query.shape[0]

        # Linear projections
        Q = np.matmul(query, self.W_q)
        K = np.matmul(key, self.W_k)
        V = np.matmul(value, self.W_v)

        # Split into heads
        Q = self.split_heads(Q)  # [batch, heads, seq_len, d_k]
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Apply attention for each head
        d_k = self.d_k
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        attention_weights = AttentionFromScratch.softmax(scores, axis=-1)
        attention_output = np.matmul(attention_weights, V)

        # Combine heads
        concat = self.combine_heads(attention_output)

        # Final linear projection
        output = np.matmul(concat, self.W_o)

        return output, attention_weights


class SelfAttentionDemo:
    """
    Demonstrate self-attention with simple examples
    """

    @staticmethod
    def visualize_attention(attention_weights, tokens, title="Attention Weights"):
        """
        Visualize attention weights as heatmap

        Args:
            attention_weights: 2D attention matrix
            tokens: List of tokens
            title: Plot title
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights, annot=True, fmt='.2f',
                    xticklabels=tokens, yticklabels=tokens,
                    cmap='Blues', vmin=0, vmax=1)
        plt.title(title)
        plt.xlabel('Keys')
        plt.ylabel('Queries')
        plt.tight_layout()
        plt.savefig(f'attention_visualization.png', dpi=150)
        plt.close()
        print(f"Attention visualization saved to 'attention_visualization.png'")

    @staticmethod
    def demonstrate_basic_attention():
        """
        Demonstrate basic attention with a simple example
        """
        print("=" * 60)
        print("BASIC ATTENTION DEMONSTRATION")
        print("=" * 60)

        # Simple example: 3 tokens with 4-dimensional embeddings
        # Sentence: "The cat sat"
        tokens = ["The", "cat", "sat"]

        # Simulated embeddings (random for demo)
        np.random.seed(42)
        seq_len = 3
        d_model = 4

        # Create embeddings with some structure
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.5],  # "The" - article
            [0.0, 1.0, 0.5, 0.0],  # "cat" - noun
            [0.0, 0.5, 1.0, 0.0],  # "sat" - verb
        ])

        embeddings = embeddings.reshape(1, seq_len, d_model)  # Add batch dimension

        print(f"\n1. Input embeddings shape: {embeddings.shape}")
        print(f"   Tokens: {tokens}")
        print(f"   Embeddings:\n{embeddings[0]}")

        # For self-attention: Q = K = V = embeddings
        Q = K = V = embeddings

        # Apply scaled dot-product attention
        output, attention_weights = AttentionFromScratch.scaled_dot_product_attention(Q, K, V)

        print(f"\n2. Attention scores (before softmax):")
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_model)
        print(scores[0])

        print(f"\n3. Attention weights (after softmax):")
        print(attention_weights[0])

        print(f"\n4. Output (weighted sum of values):")
        print(output[0])

        # Interpretation
        print("\n5. Interpretation:")
        for i, token in enumerate(tokens):
            print(f"\n   '{token}' attends to:")
            for j, other in enumerate(tokens):
                weight = attention_weights[0, i, j]
                print(f"     - '{other}': {weight:.4f}")

        # Visualize
        SelfAttentionDemo.visualize_attention(attention_weights[0], tokens)

        return attention_weights

    @staticmethod
    def demonstrate_multi_head_attention():
        """
        Demonstrate multi-head attention
        """
        print("\n" + "=" * 60)
        print("MULTI-HEAD ATTENTION DEMONSTRATION")
        print("=" * 60)

        # Parameters
        batch_size = 1
        seq_len = 4
        d_model = 8
        num_heads = 2

        tokens = ["I", "love", "machine", "learning"]

        # Random input
        np.random.seed(42)
        x = np.random.randn(batch_size, seq_len, d_model)

        print(f"\nInput shape: {x.shape}")
        print(f"Model dimension: {d_model}")
        print(f"Number of heads: {num_heads}")
        print(f"Dimension per head: {d_model // num_heads}")

        # Apply multi-head attention
        mha = MultiHeadAttention(d_model, num_heads)
        output, attention_weights = mha.forward(x, x, x)

        print(f"\nOutput shape: {output.shape}")
        print(f"Attention weights shape: {attention_weights.shape}")

        # Visualize attention from each head
        for head in range(num_heads):
            print(f"\n   Head {head + 1} attention weights:")
            print(attention_weights[0, head])

        return output, attention_weights


def demonstrate_attention_types():
    """
    Demonstrate different types of attention
    """
    print("\n" + "=" * 60)
    print("TYPES OF ATTENTION")
    print("=" * 60)

    # 1. Self-Attention
    print("\n1. SELF-ATTENTION:")
    print("   Q = K = V (same input)")
    print("   Used in encoder to relate positions within sequence")

    # 2. Cross-Attention
    print("\n2. CROSS-ATTENTION:")
    print("   Q from decoder, K and V from encoder")
    print("   Used in decoder to attend to encoder outputs")

    # 3. Causal/Masked Attention
    print("\n3. CAUSAL ATTENTION:")
    print("   Masks future positions in sequence")
    print("   Used in decoder for autoregressive generation")

    # Demonstrate causal mask
    seq_len = 5
    causal_mask = AttentionFromScratch.create_causal_mask(seq_len)
    print(f"\n   Causal mask for seq_len={seq_len}:")
    print(causal_mask.astype(int))
    print("   (1 = can attend, 0 = cannot attend)")


if TORCH_AVAILABLE:
    class PyTorchMultiHeadAttention(nn.Module):
        """
        Multi-Head Attention using PyTorch
        """

        def __init__(self, d_model, num_heads, dropout=0.1):
            super().__init__()

            assert d_model % num_heads == 0

            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

            self.dropout = nn.Dropout(dropout)

        def forward(self, query, key, value, mask=None):
            batch_size = query.size(0)

            # Linear projections
            Q = self.W_q(query)
            K = self.W_k(key)
            V = self.W_v(value)

            # Split into heads
            Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

            # Attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            # Apply attention to values
            output = torch.matmul(attention_weights, V)

            # Combine heads
            output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

            # Final projection
            output = self.W_o(output)

            return output, attention_weights


def main():
    """Main demonstration function"""

    # Basic attention
    SelfAttentionDemo.demonstrate_basic_attention()

    # Multi-head attention
    SelfAttentionDemo.demonstrate_multi_head_attention()

    # Attention types
    demonstrate_attention_types()

    # PyTorch implementation (if available)
    if TORCH_AVAILABLE:
        print("\n" + "=" * 60)
        print("PYTORCH IMPLEMENTATION")
        print("=" * 60)

        mha = PyTorchMultiHeadAttention(d_model=64, num_heads=8)
        x = torch.randn(2, 10, 64)  # [batch, seq_len, d_model]
        output, attn = mha(x, x, x)
        print(f"\nPyTorch MHA output shape: {output.shape}")
        print(f"Attention weights shape: {attn.shape}")


if __name__ == "__main__":
    main()
