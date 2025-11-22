"""
Day 20: Recurrent Neural Networks (RNN)
Assignment: Build sentiment analysis model for movie reviews (IMDB dataset)

This module covers:
- Sequential data and time series
- RNN architecture and vanishing gradient problem
- LSTM and GRU units
- Bidirectional RNNs
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")


# =============================================================================
# PART 1: RNN CONCEPTS EXPLANATION
# =============================================================================

def explain_rnn_concepts():
    """Explain RNN concepts in detail."""
    explanation = """
    ===================================================================
    RECURRENT NEURAL NETWORKS (RNN) CONCEPTS
    ===================================================================

    1. WHY RNNs FOR SEQUENTIAL DATA?
    --------------------------------
    - Standard neural networks treat each input independently
    - Sequential data has temporal dependencies
    - Examples: text, time series, audio, video

    In text: "The movie was not good" - 'not' changes meaning of 'good'
    RNNs can capture such dependencies through recurrence.

    2. BASIC RNN ARCHITECTURE
    -------------------------
    At each time step t:
        h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)
        y_t = W_hy * h_t + b_y

    Where:
    - x_t: input at time t
    - h_t: hidden state at time t (memory)
    - h_{t-1}: previous hidden state
    - W_xh, W_hh, W_hy: weight matrices

    The hidden state h_t carries information from previous time steps.

    3. VANISHING/EXPLODING GRADIENT PROBLEM
    ---------------------------------------
    During backpropagation through time (BPTT):
    - Gradients are multiplied at each time step
    - Many multiplications of values < 1 → vanishing
    - Many multiplications of values > 1 → exploding

    Result: RNNs struggle with long-term dependencies

    Example: In "I grew up in France... I speak fluent ___"
    - The word "France" is important for predicting "French"
    - But it's many steps away → gradient vanishes

    4. LSTM (Long Short-Term Memory)
    --------------------------------
    Introduced by Hochreiter & Schmidhuber (1997)
    Solves vanishing gradient with gating mechanisms.

    LSTM has 3 gates:
    a) Forget Gate (f_t): What to forget from cell state
       f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)

    b) Input Gate (i_t): What new info to store
       i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
       C̃_t = tanh(W_C * [h_{t-1}, x_t] + b_C)

    c) Output Gate (o_t): What to output
       o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
       h_t = o_t * tanh(C_t)

    Cell State Update:
       C_t = f_t * C_{t-1} + i_t * C̃_t

    Key: The cell state C_t can flow through time with minimal changes,
         allowing gradients to flow better.

    5. GRU (Gated Recurrent Unit)
    -----------------------------
    Introduced by Cho et al. (2014)
    Simplified version of LSTM with 2 gates.

    a) Reset Gate (r_t): How much past info to forget
       r_t = sigmoid(W_r * [h_{t-1}, x_t])

    b) Update Gate (z_t): How much to update hidden state
       z_t = sigmoid(W_z * [h_{t-1}, x_t])

    Candidate:
       h̃_t = tanh(W * [r_t * h_{t-1}, x_t])

    Output:
       h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t

    GRU vs LSTM:
    - GRU: Fewer parameters, faster training
    - LSTM: More expressive, better for complex patterns
    - Performance often similar; try both!

    6. BIDIRECTIONAL RNNs
    ---------------------
    Process sequence in both directions:
    - Forward RNN: left to right
    - Backward RNN: right to left
    - Concatenate both hidden states

    Why? Context from both sides is useful.
    Example: "The apple was ___ and delicious"
    - Forward: "The apple was" → could be many things
    - Backward: "and delicious" → suggests positive adjective

    7. SEQUENCE CLASSIFICATION
    --------------------------
    For sentiment analysis:
    - Many-to-One: sequence → single output
    - Use final hidden state or pooling over all states

    Options:
    - return_sequences=False: only final hidden state
    - return_sequences=True: all hidden states
    - GlobalMaxPooling1D/GlobalAveragePooling1D on all states

    ===================================================================
    """
    print(explanation)


# =============================================================================
# PART 2: LOAD AND PREPROCESS IMDB DATA
# =============================================================================

def load_imdb_data(max_features=10000, max_len=200):
    """
    Load IMDB movie reviews dataset.

    IMDB Dataset:
    - 25,000 training reviews
    - 25,000 testing reviews
    - Binary sentiment: positive (1) or negative (0)
    """
    print("Loading IMDB dataset...")

    # Load data
    (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(
        num_words=max_features
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Vocabulary size: {max_features}")

    # Analyze sequence lengths
    train_lengths = [len(seq) for seq in X_train]
    print(f"\nReview length statistics:")
    print(f"  Mean: {np.mean(train_lengths):.1f}")
    print(f"  Max: {np.max(train_lengths)}")
    print(f"  Min: {np.min(train_lengths)}")

    # Pad sequences
    print(f"\nPadding sequences to length {max_len}...")
    X_train = pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=max_len, padding='post', truncating='post')

    # Create validation split
    val_size = 5000
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]

    print(f"\nFinal shapes:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    # Get word index for decoding
    word_index = keras.datasets.imdb.get_word_index()

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), word_index


def decode_review(encoded_review, word_index):
    """Decode an encoded review back to text."""
    # Reverse word index
    reverse_word_index = {v: k for k, v in word_index.items()}

    # Decode (indices are offset by 3)
    decoded = ' '.join([
        reverse_word_index.get(i - 3, '?') for i in encoded_review if i > 0
    ])

    return decoded


def visualize_sequence_lengths(X_train_original):
    """Visualize distribution of sequence lengths."""
    lengths = [len(seq) for seq in X_train_original]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(lengths, bins=50, color='steelblue', edgecolor='white')
    axes[0].axvline(x=200, color='red', linestyle='--', label='Max length (200)')
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Review Lengths')
    axes[0].legend()

    # Box plot
    axes[1].boxplot(lengths)
    axes[1].set_ylabel('Sequence Length')
    axes[1].set_title('Review Length Box Plot')

    plt.tight_layout()
    return fig


# =============================================================================
# PART 3: RNN MODELS
# =============================================================================

def build_simple_rnn(max_features=10000, max_len=200, embedding_dim=128):
    """
    Simple RNN model (for demonstration of vanishing gradient problem).
    Not recommended for real use.
    """
    model = keras.Sequential([
        layers.Embedding(max_features, embedding_dim, input_length=max_len),
        layers.SimpleRNN(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='Simple_RNN')

    return model


def build_lstm_model(max_features=10000, max_len=200, embedding_dim=128):
    """
    LSTM model for sentiment analysis.
    """
    model = keras.Sequential([
        layers.Embedding(max_features, embedding_dim, input_length=max_len),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='LSTM_Model')

    return model


def build_gru_model(max_features=10000, max_len=200, embedding_dim=128):
    """
    GRU model for sentiment analysis.
    """
    model = keras.Sequential([
        layers.Embedding(max_features, embedding_dim, input_length=max_len),
        layers.GRU(64, return_sequences=True),
        layers.GRU(32, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='GRU_Model')

    return model


def build_bidirectional_lstm(max_features=10000, max_len=200, embedding_dim=128):
    """
    Bidirectional LSTM model.
    """
    model = keras.Sequential([
        layers.Embedding(max_features, embedding_dim, input_length=max_len),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='Bidirectional_LSTM')

    return model


def build_lstm_with_attention(max_features=10000, max_len=200, embedding_dim=128):
    """
    LSTM with simple attention mechanism.
    """
    # Input
    inputs = keras.Input(shape=(max_len,))

    # Embedding
    x = layers.Embedding(max_features, embedding_dim)(inputs)

    # Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    # Simple attention: learn importance of each time step
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(128)(attention)  # 64*2 for bidirectional
    attention = layers.Permute([2, 1])(attention)

    # Apply attention
    x = layers.Multiply()([x, attention])
    x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)

    # Classification
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs, name='LSTM_with_Attention')
    return model


def build_conv_lstm_hybrid(max_features=10000, max_len=200, embedding_dim=128):
    """
    CNN + LSTM hybrid model.
    CNN extracts local features, LSTM captures sequential patterns.
    """
    model = keras.Sequential([
        layers.Embedding(max_features, embedding_dim, input_length=max_len),

        # 1D CNN for local feature extraction
        layers.Conv1D(64, 5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),

        # LSTM for sequential modeling
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),

        # Classification
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='CNN_LSTM_Hybrid')

    return model


# =============================================================================
# PART 4: TRAINING
# =============================================================================

def get_callbacks(model_name):
    """Get training callbacks."""
    return [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=f'{model_name}_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=10, batch_size=64):
    """Train model and return history."""
    print(f"\n{'='*60}")
    print(f"Training: {model.name}")
    print(f"{'='*60}")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(model.name),
        verbose=1
    )

    return history


# =============================================================================
# PART 5: VISUALIZATION
# =============================================================================

def plot_training_history(history, model_name):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], 'b-', label='Training', linewidth=2)
    axes[0].plot(history.history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name}: Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history['accuracy'], 'b-', label='Training', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name}: Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_model_comparison(histories, names, accuracies):
    """Compare multiple models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Training curves
    for history, name in zip(histories, names):
        axes[0].plot(history.history['val_accuracy'], label=name, linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Accuracy')
    axes[0].set_title('Validation Accuracy Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Final accuracies bar chart
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(names)))
    bars = axes[1].barh(names, [a * 100 for a in accuracies], color=colors)
    axes[1].set_xlabel('Test Accuracy (%)')
    axes[1].set_title('Model Comparison')
    for bar, acc in zip(bars, accuracies):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{acc*100:.1f}%', va='center')

    # Validation loss comparison
    for history, name in zip(histories, names):
        axes[2].plot(history.history['val_loss'], label=name, linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Validation Loss')
    axes[2].set_title('Validation Loss Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_predictions(model, X_test, y_test, word_index, num_samples=5):
    """Visualize model predictions with decoded reviews."""
    predictions = model.predict(X_test[:num_samples], verbose=0)
    pred_classes = (predictions > 0.5).astype(int).flatten()

    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    for i in range(num_samples):
        print(f"\n--- Review {i+1} ---")
        decoded = decode_review(X_test[i], word_index)
        print(f"Review: {decoded[:200]}...")
        print(f"True Sentiment: {'Positive' if y_test[i] == 1 else 'Negative'}")
        print(f"Predicted: {'Positive' if pred_classes[i] == 1 else 'Negative'}")
        print(f"Confidence: {predictions[i][0]:.4f}")
        print(f"Correct: {'Yes' if pred_classes[i] == y_test[i] else 'No'}")


def visualize_rnn_architecture():
    """Create visual explanation of RNN architectures."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Simple RNN
    ax = axes[0, 0]
    ax.text(0.5, 0.9, 'Simple RNN', fontsize=14, fontweight='bold',
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.7, 'h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b)', fontsize=10,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.5, 'Problem: Vanishing Gradients', fontsize=10,
            ha='center', transform=ax.transAxes, color='red')
    ax.text(0.5, 0.3, 'Cannot learn long-term dependencies', fontsize=9,
            ha='center', transform=ax.transAxes)
    ax.axis('off')

    # LSTM
    ax = axes[0, 1]
    ax.text(0.5, 0.9, 'LSTM (Long Short-Term Memory)', fontsize=14, fontweight='bold',
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.7, 'Gates: Forget, Input, Output', fontsize=10,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.55, 'Cell State: C_t (long-term memory)', fontsize=10,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.4, 'Hidden State: h_t (short-term memory)', fontsize=10,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.25, 'Solves vanishing gradient!', fontsize=10,
            ha='center', transform=ax.transAxes, color='green')
    ax.axis('off')

    # GRU
    ax = axes[1, 0]
    ax.text(0.5, 0.9, 'GRU (Gated Recurrent Unit)', fontsize=14, fontweight='bold',
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.7, 'Gates: Reset, Update', fontsize=10,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.55, 'No separate cell state', fontsize=10,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.4, 'Fewer parameters than LSTM', fontsize=10,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.25, 'Often similar performance to LSTM', fontsize=10,
            ha='center', transform=ax.transAxes, color='blue')
    ax.axis('off')

    # Bidirectional
    ax = axes[1, 1]
    ax.text(0.5, 0.9, 'Bidirectional RNN', fontsize=14, fontweight='bold',
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.7, 'Forward: x_1 → x_2 → ... → x_T', fontsize=10,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.55, 'Backward: x_T → ... → x_2 → x_1', fontsize=10,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.4, 'Output: [h_forward; h_backward]', fontsize=10,
            ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.25, 'Context from both directions', fontsize=10,
            ha='center', transform=ax.transAxes, color='purple')
    ax.axis('off')

    plt.suptitle('RNN Architecture Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# PART 6: MAIN EXECUTION
# =============================================================================

def main():
    """Main function for RNN sentiment analysis."""
    print("=" * 70)
    print("DAY 20: RECURRENT NEURAL NETWORKS")
    print("Sentiment Analysis on IMDB Movie Reviews")
    print("=" * 70)

    # Explain concepts
    explain_rnn_concepts()

    # Visualize architecture
    fig_arch = visualize_rnn_architecture()
    fig_arch.savefig('rnn_architecture_overview.png', dpi=150, bbox_inches='tight')
    print("Saved: rnn_architecture_overview.png")

    # Parameters
    MAX_FEATURES = 10000  # Vocabulary size
    MAX_LEN = 200         # Max sequence length
    EMBEDDING_DIM = 128   # Embedding dimension

    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), word_index = load_imdb_data(
        max_features=MAX_FEATURES, max_len=MAX_LEN
    )

    # Show sample review
    print("\n--- Sample Review (decoded) ---")
    print(decode_review(X_train[0], word_index)[:500])
    print(f"Sentiment: {'Positive' if y_train[0] == 1 else 'Negative'}")

    # Build models
    models_to_train = [
        build_lstm_model(MAX_FEATURES, MAX_LEN, EMBEDDING_DIM),
        build_gru_model(MAX_FEATURES, MAX_LEN, EMBEDDING_DIM),
        build_bidirectional_lstm(MAX_FEATURES, MAX_LEN, EMBEDDING_DIM),
        build_conv_lstm_hybrid(MAX_FEATURES, MAX_LEN, EMBEDDING_DIM)
    ]

    # Train and evaluate
    histories = []
    accuracies = []
    names = []

    for model in models_to_train:
        model.summary()

        history = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=10, batch_size=64
        )

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")

        histories.append(history)
        accuracies.append(test_acc)
        names.append(model.name)

    # Plot comparison
    fig_comparison = plot_model_comparison(histories, names, accuracies)
    fig_comparison.savefig('rnn_model_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: rnn_model_comparison.png")

    # Best model
    best_idx = np.argmax(accuracies)
    best_model = models_to_train[best_idx]
    print(f"\nBest Model: {names[best_idx]} with {accuracies[best_idx]*100:.2f}% accuracy")

    # Plot best model history
    fig_history = plot_training_history(histories[best_idx], names[best_idx])
    fig_history.savefig('best_rnn_training.png', dpi=150, bbox_inches='tight')
    print("Saved: best_rnn_training.png")

    # Visualize predictions
    visualize_predictions(best_model, X_test, y_test, word_index)

    # Save best model
    best_model.save('imdb_sentiment_model.keras')
    print("\nSaved: imdb_sentiment_model.keras")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nModel Results:")
    for name, acc in zip(names, accuracies):
        print(f"  {name}: {acc * 100:.2f}%")

    print(f"""
    Key Learnings:

    1. RNN TYPES:
       - Simple RNN: Vanishing gradient, can't learn long dependencies
       - LSTM: Gates control information flow, solves vanishing gradient
       - GRU: Simplified LSTM, fewer parameters, similar performance
       - Bidirectional: Context from both directions

    2. FOR SENTIMENT ANALYSIS:
       - Word embeddings capture semantic meaning
       - RNNs capture sequential dependencies
       - Bidirectional helps capture full context
       - CNN+RNN hybrid can capture local and global patterns

    3. BEST PRACTICES:
       - Use LSTM/GRU instead of Simple RNN
       - Try Bidirectional for better context
       - Use dropout for regularization
       - Pad sequences to same length
       - Use pre-trained embeddings (Word2Vec, GloVe) for better results

    4. LIMITATIONS:
       - RNNs are sequential → can't parallelize
       - Modern alternative: Transformers (BERT, etc.)
       - For production: Consider using DistilBERT or similar
    """)

    plt.close('all')
    print("\nAll visualizations saved successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
