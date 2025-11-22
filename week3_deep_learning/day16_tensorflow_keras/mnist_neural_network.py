"""
Day 16: Introduction to TensorFlow/Keras
Assignment: Build 3-layer neural network for MNIST digit classification

This module covers:
- TensorFlow and Keras API
- Sequential model vs Functional API
- Layers: Dense, Dropout, BatchNormalization
- Model compilation: optimizers, loss, metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from tensorflow.keras.utils import plot_model
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")


# =============================================================================
# PART 1: LOAD AND PREPROCESS MNIST DATA
# =============================================================================

def load_and_preprocess_mnist():
    """
    Load MNIST dataset and preprocess for neural network.

    MNIST: 70,000 grayscale images of handwritten digits (0-9)
    - Training: 60,000 images
    - Testing: 10,000 images
    - Image size: 28x28 pixels
    """
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Image shape: {X_train.shape[1:]}")

    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Flatten images for Dense layers (28*28 = 784)
    X_train_flat = X_train.reshape(-1, 784)
    X_test_flat = X_test.reshape(-1, 784)

    print(f"Flattened shape: {X_train_flat.shape}")

    # Create validation split
    val_size = 10000
    X_val = X_train_flat[-val_size:]
    y_val = y_train[-val_size:]
    X_train_flat = X_train_flat[:-val_size]
    y_train_subset = y_train[:-val_size]

    return (X_train_flat, y_train_subset), (X_val, y_val), (X_test_flat, y_test), (X_train, X_test)


def visualize_mnist_samples(X, y, num_samples=10):
    """Visualize sample MNIST digits."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i in range(num_samples):
        idx = np.random.randint(len(X))
        axes[i].imshow(X[idx], cmap='gray')
        axes[i].set_title(f'Label: {y[idx]}')
        axes[i].axis('off')

    plt.suptitle('Sample MNIST Digits', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# PART 2: SEQUENTIAL MODEL API
# =============================================================================

def build_sequential_model():
    """
    Build a 3-layer neural network using the Sequential API.

    Sequential API: Linear stack of layers
    - Simple and clean for straightforward architectures
    - Layers are added in order
    - Best for: single input, single output, layer-by-layer architecture
    """
    model = keras.Sequential([
        # Input layer is implicit, but we can specify input_shape

        # First Dense layer (Hidden layer 1)
        layers.Dense(256, activation='relu', input_shape=(784,),
                    kernel_initializer='he_normal',
                    name='hidden_layer_1'),

        # Batch Normalization: Normalizes activations, speeds up training
        layers.BatchNormalization(name='batch_norm_1'),

        # Dropout: Randomly sets inputs to 0 during training (regularization)
        layers.Dropout(0.3, name='dropout_1'),

        # Second Dense layer (Hidden layer 2)
        layers.Dense(128, activation='relu',
                    kernel_initializer='he_normal',
                    name='hidden_layer_2'),
        layers.BatchNormalization(name='batch_norm_2'),
        layers.Dropout(0.2, name='dropout_2'),

        # Third Dense layer (Hidden layer 3)
        layers.Dense(64, activation='relu',
                    kernel_initializer='he_normal',
                    name='hidden_layer_3'),
        layers.BatchNormalization(name='batch_norm_3'),

        # Output layer: 10 classes (digits 0-9)
        layers.Dense(10, activation='softmax', name='output')
    ], name='MNIST_Sequential_Model')

    return model


# =============================================================================
# PART 3: FUNCTIONAL API
# =============================================================================

def build_functional_model():
    """
    Build a 3-layer neural network using the Functional API.

    Functional API: Defines a graph of layers
    - More flexible than Sequential
    - Supports multiple inputs/outputs
    - Supports layer sharing
    - Best for: complex architectures, multi-input/output models
    """
    # Define input
    inputs = keras.Input(shape=(784,), name='input')

    # First hidden layer
    x = layers.Dense(256, activation='relu',
                    kernel_initializer='he_normal',
                    name='hidden_layer_1')(inputs)
    x = layers.BatchNormalization(name='batch_norm_1')(x)
    x = layers.Dropout(0.3, name='dropout_1')(x)

    # Second hidden layer
    x = layers.Dense(128, activation='relu',
                    kernel_initializer='he_normal',
                    name='hidden_layer_2')(x)
    x = layers.BatchNormalization(name='batch_norm_2')(x)
    x = layers.Dropout(0.2, name='dropout_2')(x)

    # Third hidden layer
    x = layers.Dense(64, activation='relu',
                    kernel_initializer='he_normal',
                    name='hidden_layer_3')(x)
    x = layers.BatchNormalization(name='batch_norm_3')(x)

    # Output layer
    outputs = layers.Dense(10, activation='softmax', name='output')(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='MNIST_Functional_Model')

    return model


# =============================================================================
# PART 4: MODEL COMPILATION
# =============================================================================

def compile_model(model, optimizer='adam', learning_rate=0.001):
    """
    Compile model with optimizer, loss function, and metrics.

    Optimizers:
    - SGD: Stochastic Gradient Descent
    - Adam: Adaptive Moment Estimation (most popular)
    - RMSprop: Root Mean Square Propagation
    - Adagrad: Adaptive Gradient Algorithm

    Loss Functions:
    - sparse_categorical_crossentropy: For integer labels
    - categorical_crossentropy: For one-hot encoded labels
    - binary_crossentropy: For binary classification

    Metrics:
    - accuracy: Classification accuracy
    - precision, recall, f1: For imbalanced datasets
    """
    # Define optimizers
    optimizers_dict = {
        'sgd': optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        'adam': optimizers.Adam(learning_rate=learning_rate),
        'rmsprop': optimizers.RMSprop(learning_rate=learning_rate),
        'adagrad': optimizers.Adagrad(learning_rate=learning_rate)
    }

    opt = optimizers_dict.get(optimizer.lower(), optimizers.Adam(learning_rate=learning_rate))

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',  # Use sparse because labels are integers
        metrics=['accuracy']
    )

    return model


# =============================================================================
# PART 5: TRAINING WITH CALLBACKS
# =============================================================================

def get_callbacks(model_name='mnist_model'):
    """
    Define training callbacks.

    Callbacks:
    - EarlyStopping: Stop training when validation loss stops improving
    - ModelCheckpoint: Save the best model during training
    - ReduceLROnPlateau: Reduce learning rate when loss plateaus
    - TensorBoard: Log training for visualization
    """
    callback_list = [
        # Early stopping: prevent overfitting
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate when loss plateaus
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),

        # Model checkpoint: save best model
        callbacks.ModelCheckpoint(
            filepath=f'{model_name}_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    return callback_list


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=30, batch_size=128):
    """Train the model with callbacks."""
    print(f"\nTraining {model.name}...")
    print("-" * 50)

    callback_list = get_callbacks(model.name)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1
    )

    return history


# =============================================================================
# PART 6: VISUALIZATION
# =============================================================================

def plot_training_history(history, model_name='Model'):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Loss
    axes[0].plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name}: Training vs Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot Accuracy
    axes[1].plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name}: Training vs Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_predictions(model, X_test, y_test, X_test_2d, num_samples=10):
    """Visualize model predictions on test samples."""
    # Get predictions
    predictions = model.predict(X_test[:num_samples], verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(num_samples):
        axes[i].imshow(X_test_2d[i], cmap='gray')
        color = 'green' if predicted_classes[i] == y_test[i] else 'red'
        axes[i].set_title(f'True: {y_test[i]}\nPred: {predicted_classes[i]}\nConf: {predictions[i].max():.2f}',
                         color=color, fontsize=10)
        axes[i].axis('off')

    plt.suptitle('Model Predictions on Test Samples\n(Green=Correct, Red=Incorrect)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_confusion_matrix(model, X_test, y_test):
    """Plot confusion matrix for predictions."""
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=range(10), yticklabels=range(10))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return fig


# =============================================================================
# PART 7: LAYER EXPLANATION
# =============================================================================

def explain_layers():
    """Explain the key layers used in neural networks."""
    explanation = """
    ===================================================================
    KERAS LAYERS EXPLAINED
    ===================================================================

    1. DENSE (Fully Connected) Layer
    --------------------------------
    - Every neuron connected to every neuron in previous layer
    - Parameters: units (number of neurons), activation function
    - Example: Dense(128, activation='relu')
    - Total parameters = (input_dim + 1) * units (weights + biases)

    2. DROPOUT Layer
    ----------------
    - Regularization technique to prevent overfitting
    - Randomly sets fraction of inputs to 0 during training
    - Example: Dropout(0.3) -> 30% of inputs set to 0
    - Not active during inference/prediction

    3. BATCH NORMALIZATION Layer
    ----------------------------
    - Normalizes layer inputs (mean=0, variance=1)
    - Speeds up training, allows higher learning rates
    - Reduces internal covariate shift
    - Has learnable parameters: gamma (scale) and beta (shift)

    4. ACTIVATION Functions (built into Dense or separate)
    ------------------------------------------------------
    - ReLU: f(x) = max(0, x) - Most common for hidden layers
    - Sigmoid: f(x) = 1/(1+e^-x) - Binary output
    - Softmax: e^xi / Σe^xj - Multi-class probability output
    - Tanh: f(x) = (e^x - e^-x)/(e^x + e^-x) - Range (-1, 1)

    ===================================================================
    OPTIMIZERS EXPLAINED
    ===================================================================

    1. SGD (Stochastic Gradient Descent)
    ------------------------------------
    - Simple: w = w - lr * gradient
    - Can add momentum to accelerate convergence
    - Requires careful learning rate tuning

    2. Adam (Adaptive Moment Estimation)
    ------------------------------------
    - Combines momentum and RMSprop
    - Adaptive learning rates for each parameter
    - Generally works well out of the box
    - Most popular choice

    3. RMSprop
    ----------
    - Adapts learning rate based on recent gradients
    - Good for recurrent neural networks

    ===================================================================
    LOSS FUNCTIONS
    ===================================================================

    1. Sparse Categorical Cross-Entropy
    -----------------------------------
    - For multi-class classification with integer labels
    - Loss = -Σ y_true * log(y_pred)

    2. Categorical Cross-Entropy
    ----------------------------
    - Same as above but expects one-hot encoded labels

    3. Binary Cross-Entropy
    -----------------------
    - For binary classification
    - Loss = -[y*log(p) + (1-y)*log(1-p)]

    4. MSE (Mean Squared Error)
    ---------------------------
    - For regression problems
    - Loss = Σ(y_true - y_pred)²

    ===================================================================
    """
    print(explanation)


# =============================================================================
# PART 8: MAIN EXECUTION
# =============================================================================

def main():
    """Main function demonstrating TensorFlow/Keras for MNIST."""
    print("=" * 70)
    print("DAY 16: INTRODUCTION TO TENSORFLOW/KERAS")
    print("3-Layer Neural Network for MNIST Digit Classification")
    print("=" * 70)

    # Explain layers
    explain_layers()

    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_train_2d, X_test_2d) = load_and_preprocess_mnist()

    # Visualize samples
    fig_samples = visualize_mnist_samples(X_train_2d, y_train)
    fig_samples.savefig('mnist_samples.png', dpi=150, bbox_inches='tight')
    print("Saved: mnist_samples.png")

    # Build and display Sequential model
    print("\n" + "=" * 70)
    print("BUILDING SEQUENTIAL MODEL")
    print("=" * 70)
    sequential_model = build_sequential_model()
    sequential_model.summary()

    # Compile model
    sequential_model = compile_model(sequential_model, optimizer='adam', learning_rate=0.001)

    # Train model
    history = train_model(sequential_model, X_train, y_train, X_val, y_val,
                         epochs=20, batch_size=128)

    # Plot training history
    fig_history = plot_training_history(history, 'Sequential Model')
    fig_history.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Saved: training_history.png")

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)
    test_loss, test_accuracy = sequential_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Visualize predictions
    fig_preds = visualize_predictions(sequential_model, X_test, y_test, X_test_2d)
    fig_preds.savefig('predictions.png', dpi=150, bbox_inches='tight')
    print("Saved: predictions.png")

    # Confusion matrix
    try:
        fig_cm = plot_confusion_matrix(sequential_model, X_test, y_test)
        fig_cm.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("Saved: confusion_matrix.png")
    except ImportError:
        print("Note: Install seaborn for confusion matrix visualization")

    # Build and display Functional model (for comparison)
    print("\n" + "=" * 70)
    print("FUNCTIONAL API MODEL (for reference)")
    print("=" * 70)
    functional_model = build_functional_model()
    functional_model.summary()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Results:
    - Model: 3-Layer Neural Network
    - Architecture: 784 → 256 → 128 → 64 → 10
    - Test Accuracy: {test_accuracy * 100:.2f}%

    Key Learnings:
    1. Sequential API: Simple, linear stack of layers
    2. Functional API: More flexible, supports complex architectures
    3. Dense layers: Fully connected layers
    4. Dropout: Regularization to prevent overfitting
    5. BatchNormalization: Speeds up training
    6. Adam optimizer: Adaptive learning rates
    7. Sparse Categorical Cross-Entropy: Multi-class classification loss
    """)

    plt.close('all')
    print("\nAll visualizations saved successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
