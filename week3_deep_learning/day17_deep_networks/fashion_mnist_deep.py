"""
Day 17: Deep Neural Networks
Assignment: Build deep network (5+ layers) for Fashion-MNIST, achieve >90% accuracy

This module covers:
- Architecture design: input → hidden layers → output
- Regularization: L1, L2, dropout
- Learning rate schedules
- Early stopping and model checkpointing
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")


# =============================================================================
# PART 1: FASHION-MNIST DATASET
# =============================================================================

# Fashion-MNIST class names
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def load_fashion_mnist():
    """
    Load and preprocess Fashion-MNIST dataset.

    Fashion-MNIST: 70,000 grayscale images of clothing items
    - 10 categories
    - 28x28 pixels
    - More challenging than MNIST digits
    """
    print("Loading Fashion-MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Image shape: {X_train.shape[1:]}")

    # Normalize to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Keep 2D for visualization
    X_train_2d = X_train.copy()
    X_test_2d = X_test.copy()

    # Flatten for Dense layers
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)

    # Validation split
    val_size = 10000
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    X_train_2d = X_train_2d[:-val_size]

    print(f"\nData shapes after preprocessing:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_train_2d, X_test_2d)


def visualize_samples(X, y, class_names, num_samples=15):
    """Visualize sample images from the dataset."""
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()

    for i in range(num_samples):
        idx = np.random.randint(len(X))
        axes[i].imshow(X[idx], cmap='gray')
        axes[i].set_title(f'{class_names[y[idx]]}')
        axes[i].axis('off')

    plt.suptitle('Fashion-MNIST Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# PART 2: REGULARIZATION TECHNIQUES
# =============================================================================

def explain_regularization():
    """Explain regularization techniques."""
    explanation = """
    ===================================================================
    REGULARIZATION TECHNIQUES
    ===================================================================

    1. L1 REGULARIZATION (Lasso)
    ----------------------------
    - Adds sum of absolute weights to loss: λ * Σ|w|
    - Encourages sparse weights (some weights become exactly 0)
    - Feature selection effect
    - Use: kernel_regularizer=regularizers.l1(0.01)

    2. L2 REGULARIZATION (Ridge/Weight Decay)
    -----------------------------------------
    - Adds sum of squared weights to loss: λ * Σw²
    - Encourages small weights, doesn't force them to 0
    - More common in deep learning
    - Use: kernel_regularizer=regularizers.l2(0.01)

    3. ELASTIC NET (L1 + L2)
    ------------------------
    - Combination of L1 and L2: λ1 * Σ|w| + λ2 * Σw²
    - Use: kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)

    4. DROPOUT
    ----------
    - Randomly zeroes neurons during training
    - Prevents co-adaptation of neurons
    - Acts like ensemble of networks
    - Use: layers.Dropout(rate=0.5)

    5. BATCH NORMALIZATION
    ----------------------
    - Normalizes layer inputs
    - Has slight regularization effect
    - Reduces internal covariate shift

    6. EARLY STOPPING
    -----------------
    - Stop training when validation loss stops improving
    - Prevents overfitting by limiting training time
    - Use: callbacks.EarlyStopping(patience=5)

    7. DATA AUGMENTATION
    --------------------
    - Artificially increase training data
    - Rotations, flips, crops, etc.
    - Creates invariance to transformations

    ===================================================================
    """
    print(explanation)


# =============================================================================
# PART 3: DEEP NEURAL NETWORK ARCHITECTURES
# =============================================================================

def build_deep_model_v1(input_shape=784, num_classes=10):
    """
    Deep Neural Network V1: Basic 5-layer architecture with dropout.
    """
    model = keras.Sequential([
        # Layer 1
        layers.Dense(512, activation='relu', input_shape=(input_shape,),
                    kernel_initializer='he_normal', name='dense_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(0.3, name='dropout_1'),

        # Layer 2
        layers.Dense(256, activation='relu',
                    kernel_initializer='he_normal', name='dense_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(0.3, name='dropout_2'),

        # Layer 3
        layers.Dense(128, activation='relu',
                    kernel_initializer='he_normal', name='dense_3'),
        layers.BatchNormalization(name='bn_3'),
        layers.Dropout(0.2, name='dropout_3'),

        # Layer 4
        layers.Dense(64, activation='relu',
                    kernel_initializer='he_normal', name='dense_4'),
        layers.BatchNormalization(name='bn_4'),
        layers.Dropout(0.2, name='dropout_4'),

        # Layer 5
        layers.Dense(32, activation='relu',
                    kernel_initializer='he_normal', name='dense_5'),
        layers.BatchNormalization(name='bn_5'),

        # Output
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='DeepNet_V1_Dropout')

    return model


def build_deep_model_v2(input_shape=784, num_classes=10, l2_rate=0.001):
    """
    Deep Neural Network V2: 6-layer architecture with L2 regularization.
    """
    model = keras.Sequential([
        # Layer 1
        layers.Dense(1024, activation='relu', input_shape=(input_shape,),
                    kernel_regularizer=regularizers.l2(l2_rate),
                    kernel_initializer='he_normal', name='dense_1'),
        layers.BatchNormalization(name='bn_1'),

        # Layer 2
        layers.Dense(512, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_rate),
                    kernel_initializer='he_normal', name='dense_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(0.4, name='dropout_2'),

        # Layer 3
        layers.Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_rate),
                    kernel_initializer='he_normal', name='dense_3'),
        layers.BatchNormalization(name='bn_3'),

        # Layer 4
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_rate),
                    kernel_initializer='he_normal', name='dense_4'),
        layers.BatchNormalization(name='bn_4'),
        layers.Dropout(0.3, name='dropout_4'),

        # Layer 5
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_rate),
                    kernel_initializer='he_normal', name='dense_5'),
        layers.BatchNormalization(name='bn_5'),

        # Layer 6
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_rate),
                    kernel_initializer='he_normal', name='dense_6'),

        # Output
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='DeepNet_V2_L2_Regularization')

    return model


def build_deep_model_v3(input_shape=784, num_classes=10):
    """
    Deep Neural Network V3: 7-layer architecture with residual-like connections.
    Using Functional API for skip connections.
    """
    inputs = keras.Input(shape=(input_shape,), name='input')

    # Block 1
    x = layers.Dense(512, activation='relu', kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Block 2 with skip
    x1 = layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Add()([x, x1])  # Skip connection
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Dropout(0.3)(x1)

    # Block 3
    x2 = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.3)(x2)

    # Block 4 with skip
    x3 = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Add()([x2, x3])  # Skip connection
    x3 = layers.Activation('relu')(x3)
    x3 = layers.Dropout(0.2)(x3)

    # Block 5
    x4 = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x3)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Dropout(0.2)(x4)

    # Block 6
    x5 = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x4)
    x5 = layers.BatchNormalization()(x5)

    # Block 7
    x6 = layers.Dense(32, activation='relu', kernel_initializer='he_normal')(x5)

    # Output
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x6)

    model = keras.Model(inputs=inputs, outputs=outputs, name='DeepNet_V3_ResidualLike')
    return model


# =============================================================================
# PART 4: LEARNING RATE SCHEDULES
# =============================================================================

def get_lr_schedule(schedule_type='exponential', initial_lr=0.001):
    """
    Get different learning rate schedules.

    Types:
    - constant: Fixed learning rate
    - step_decay: Reduce by factor every N epochs
    - exponential: Exponential decay
    - cosine: Cosine annealing
    - warmup: Gradual warmup then decay
    """
    if schedule_type == 'exponential':
        # lr = initial_lr * decay_rate ^ (step / decay_steps)
        return optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )

    elif schedule_type == 'cosine':
        # Cosine annealing
        return optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=10000
        )

    elif schedule_type == 'polynomial':
        # Polynomial decay
        return optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=10000,
            end_learning_rate=1e-6,
            power=2
        )

    else:  # constant
        return initial_lr


def visualize_lr_schedules():
    """Visualize different learning rate schedules."""
    initial_lr = 0.01
    steps = np.arange(0, 10000)

    schedules = {
        'Exponential': optimizers.schedules.ExponentialDecay(initial_lr, 1000, 0.9),
        'Cosine': optimizers.schedules.CosineDecay(initial_lr, 10000),
        'Polynomial': optimizers.schedules.PolynomialDecay(initial_lr, 10000, 1e-5, 2),
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, schedule in schedules.items():
        lrs = [schedule(step) for step in steps]
        ax.plot(steps, lrs, label=name, linewidth=2)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedules Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, initial_lr * 1.1])

    plt.tight_layout()
    return fig


# =============================================================================
# PART 5: CALLBACKS AND TRAINING
# =============================================================================

def get_training_callbacks(model_name, patience=10):
    """Get comprehensive training callbacks."""
    callback_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),

        # Model checkpoint
        callbacks.ModelCheckpoint(
            filepath=f'{model_name}_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),

        # Reduce LR on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),

        # Learning rate logger
        callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(f"  LR: {keras.backend.get_value(model.optimizer.learning_rate):.6f}")
            if epoch % 5 == 0 else None
        )
    ]

    return callback_list


def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test,
                       epochs=50, batch_size=128, lr=0.001):
    """Train model and return results."""
    print(f"\n{'='*60}")
    print(f"Training: {model.name}")
    print(f"{'='*60}")

    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Get callbacks
    callback_list = get_training_callbacks(model.name)

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy * 100:.2f}%")

    return history, test_accuracy


# =============================================================================
# PART 6: VISUALIZATION
# =============================================================================

def plot_training_comparison(histories, names):
    """Compare training histories of multiple models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training Loss
    for history, name in zip(histories, names):
        axes[0, 0].plot(history.history['loss'], label=name, linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Validation Loss
    for history, name in zip(histories, names):
        axes[0, 1].plot(history.history['val_loss'], label=name, linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Training Accuracy
    for history, name in zip(histories, names):
        axes[1, 0].plot(history.history['accuracy'], label=name, linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Validation Accuracy
    for history, name in zip(histories, names):
        axes[1, 1].plot(history.history['val_accuracy'], label=name, linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Model Training Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_confusion_matrix_detailed(model, X_test, y_test, class_names):
    """Plot detailed confusion matrix with per-class metrics."""
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # Per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    axes[1].barh(class_names, class_accuracy, color='steelblue')
    axes[1].set_xlabel('Accuracy')
    axes[1].set_title('Per-Class Accuracy')
    axes[1].set_xlim([0, 1])
    for i, v in enumerate(class_accuracy):
        axes[1].text(v + 0.01, i, f'{v:.2%}', va='center')

    plt.tight_layout()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return fig


def visualize_misclassified(model, X_test, y_test, X_test_2d, class_names, num_samples=10):
    """Visualize misclassified examples."""
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    # Find misclassified indices
    misclassified_idx = np.where(y_pred != y_test)[0]

    if len(misclassified_idx) == 0:
        print("No misclassified samples!")
        return None

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(min(num_samples, len(misclassified_idx))):
        idx = misclassified_idx[i]
        axes[i].imshow(X_test_2d[idx], cmap='gray')
        axes[i].set_title(f'True: {class_names[y_test[idx]]}\n'
                         f'Pred: {class_names[y_pred[idx]]}\n'
                         f'Conf: {predictions[idx].max():.2f}',
                         color='red', fontsize=9)
        axes[i].axis('off')

    plt.suptitle('Misclassified Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# PART 7: MAIN EXECUTION
# =============================================================================

def main():
    """Main function for deep neural networks on Fashion-MNIST."""
    print("=" * 70)
    print("DAY 17: DEEP NEURAL NETWORKS")
    print("Fashion-MNIST Classification with 5+ Layer Networks")
    print("=" * 70)

    # Explain regularization
    explain_regularization()

    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_train_2d, X_test_2d) = load_fashion_mnist()

    # Visualize samples
    fig_samples = visualize_samples(X_train_2d, y_train, CLASS_NAMES)
    fig_samples.savefig('fashion_mnist_samples.png', dpi=150, bbox_inches='tight')
    print("Saved: fashion_mnist_samples.png")

    # Visualize LR schedules
    fig_lr = visualize_lr_schedules()
    fig_lr.savefig('learning_rate_schedules.png', dpi=150, bbox_inches='tight')
    print("Saved: learning_rate_schedules.png")

    # Build models
    models_list = [
        build_deep_model_v1(),
        build_deep_model_v2(l2_rate=0.0005),
        build_deep_model_v3()
    ]

    # Print model summaries
    for model in models_list:
        print(f"\n{'='*60}")
        print(f"Model: {model.name}")
        print(f"{'='*60}")
        model.summary()

    # Train all models
    histories = []
    accuracies = []

    for model in models_list:
        history, accuracy = train_and_evaluate(
            model, X_train, y_train, X_val, y_val, X_test, y_test,
            epochs=30, batch_size=128, lr=0.001
        )
        histories.append(history)
        accuracies.append(accuracy)

    # Compare models
    names = [m.name for m in models_list]
    fig_comparison = plot_training_comparison(histories, names)
    fig_comparison.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: model_comparison.png")

    # Best model analysis
    best_idx = np.argmax(accuracies)
    best_model = models_list[best_idx]
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model.name}")
    print(f"Test Accuracy: {accuracies[best_idx] * 100:.2f}%")
    print(f"{'='*60}")

    # Detailed evaluation of best model
    fig_cm = plot_confusion_matrix_detailed(best_model, X_test, y_test, CLASS_NAMES)
    fig_cm.savefig('confusion_matrix_fashion.png', dpi=150, bbox_inches='tight')
    print("Saved: confusion_matrix_fashion.png")

    # Misclassified examples
    fig_misc = visualize_misclassified(best_model, X_test, y_test, X_test_2d, CLASS_NAMES)
    if fig_misc:
        fig_misc.savefig('misclassified_examples.png', dpi=150, bbox_inches='tight')
        print("Saved: misclassified_examples.png")

    # Save best model
    best_model.save('fashion_mnist_best_model.keras')
    print(f"Saved: fashion_mnist_best_model.keras")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nModel Results:")
    for name, acc in zip(names, accuracies):
        status = "ACHIEVED >90%!" if acc > 0.90 else "Below 90%"
        print(f"  {name}: {acc * 100:.2f}% - {status}")

    print(f"""
    Key Learnings:
    1. Deep networks (5+ layers) can achieve >90% on Fashion-MNIST
    2. Regularization (L2, Dropout) prevents overfitting
    3. Batch Normalization speeds up training
    4. Learning rate schedules improve convergence
    5. Early stopping prevents overfitting
    6. Skip/residual connections help gradient flow

    Architecture Tips:
    - Start wide, get narrower (pyramid structure)
    - Use BatchNorm after Dense layers
    - Apply Dropout between layers (0.2-0.5)
    - Use 'he_normal' initialization for ReLU
    - Apply L2 regularization (0.0001-0.01)
    """)

    plt.close('all')
    print("\nAll visualizations saved successfully!")
    print("=" * 70)


if __name__ == "__main__":
    # Make global reference to model for LR callback
    model = None
    main()
