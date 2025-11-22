"""
Day 18: Convolutional Neural Networks (CNN) - Part 1
Assignment: Build basic CNN for CIFAR-10 classification

This module covers:
- Convolution operation: filters, kernels, feature maps
- Pooling layers: max pooling, average pooling
- CNN architecture patterns: Conv → Pool → Conv → Pool → Dense
- Understanding why CNNs are effective for images
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")


# =============================================================================
# CIFAR-10 CLASS NAMES
# =============================================================================

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# =============================================================================
# PART 1: CNN CONCEPTS EXPLANATION
# =============================================================================

def explain_cnn_concepts():
    """Explain CNN concepts in detail."""
    explanation = """
    ===================================================================
    CONVOLUTIONAL NEURAL NETWORKS (CNN) CONCEPTS
    ===================================================================

    1. CONVOLUTION OPERATION
    ------------------------
    - A convolution applies a small filter (kernel) across the input image
    - The filter slides across the image, computing dot products
    - Output is called a "feature map" or "activation map"

    Formula: output[i,j] = Σ Σ input[i+m, j+n] * kernel[m,n]

    Key terms:
    - Filter/Kernel: Small matrix of learnable weights (e.g., 3x3, 5x5)
    - Stride: Step size for moving the filter (usually 1 or 2)
    - Padding: Adding zeros around input to control output size
      - 'valid': No padding (output smaller than input)
      - 'same': Padding to keep output same size as input

    Example: 3x3 filter on 28x28 image with valid padding = 26x26 output

    2. WHY CONVOLUTIONS FOR IMAGES?
    -------------------------------
    a) Parameter Sharing: Same filter applied across entire image
       - 3x3 filter = only 9 parameters (vs millions in Dense layer)

    b) Translation Invariance: Can detect features anywhere in image
       - Cat detector works whether cat is top-left or bottom-right

    c) Spatial Hierarchy: Early layers detect edges, later detect objects
       - Layer 1: Edges, corners
       - Layer 2: Textures, patterns
       - Layer 3: Object parts (eyes, ears)
       - Layer 4: Full objects

    3. POOLING LAYERS
    -----------------
    - Reduce spatial dimensions while retaining important features
    - Makes model more robust to small translations

    Types:
    a) Max Pooling: Takes maximum value in each window
       - Most common, preserves strongest activations
       - Example: 2x2 max pooling reduces 4x4 → 2x2

    b) Average Pooling: Takes average of values in window
       - Smoother downsampling
       - Sometimes used at end before Dense layers

    c) Global Average Pooling: Average over entire feature map
       - Reduces feature map to single value per channel
       - Alternative to Flatten → Dense at network end

    4. CNN ARCHITECTURE PATTERN
    ---------------------------
    Input → [Conv → ReLU → Pool] × N → Flatten → Dense → Output

    Typical pattern:
    - Start with fewer filters, increase as you go deeper
    - Filter sizes: 3x3 most common (or 5x5)
    - Pool after every 1-2 Conv layers
    - Flatten before Dense layers
    - End with Dense layer with softmax for classification

    5. FEATURE MAPS
    ---------------
    - Each filter produces one feature map
    - Filter count determines "depth" of output
    - Example: 32 3x3 filters on RGB image: (H,W,3) → (H',W',32)

    6. RECEPTIVE FIELD
    ------------------
    - Region of input that affects a particular output neuron
    - Grows larger in deeper layers
    - Stacking 3x3 convs: two 3x3 = 5x5 receptive field

    ===================================================================
    """
    print(explanation)


# =============================================================================
# PART 2: LOAD AND PREPROCESS CIFAR-10
# =============================================================================

def load_cifar10():
    """
    Load and preprocess CIFAR-10 dataset.

    CIFAR-10: 60,000 32x32 color images in 10 classes
    - Training: 50,000 images
    - Testing: 10,000 images
    """
    print("Loading CIFAR-10 dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Squeeze labels
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Image shape: {X_train.shape[1:]}")

    # Normalize to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Validation split
    val_size = 5000
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]

    print(f"\nData shapes after preprocessing:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def visualize_cifar_samples(X, y, class_names, num_samples=15):
    """Visualize sample images."""
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()

    for i in range(num_samples):
        idx = np.random.randint(len(X))
        axes[i].imshow(X[idx])
        axes[i].set_title(f'{class_names[y[idx]]}')
        axes[i].axis('off')

    plt.suptitle('CIFAR-10 Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# PART 3: VISUALIZE CONVOLUTION OPERATION
# =============================================================================

def visualize_convolution_operation():
    """Visualize how convolution works step by step."""
    # Create a simple 5x5 image
    image = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=float)

    # Different kernels
    kernels = {
        'Edge Detection (Vertical)': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'Edge Detection (Horizontal)': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'Blur': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image (5x5)')
    axes[0, 0].axis('off')

    # Show each kernel and its result
    for idx, (name, kernel) in enumerate(kernels.items()):
        # Manual convolution (valid padding)
        output = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                output[i, j] = np.sum(image[i:i+3, j:j+3] * kernel)

        row = (idx + 1) // 3
        col = (idx + 1) % 3

        if idx < 2:
            ax = axes[0, idx + 1]
        else:
            ax = axes[1, idx - 2]

        ax.imshow(output, cmap='gray')
        ax.set_title(f'{name}\nKernel:\n{kernel}')
        ax.axis('off')

    # Show pooling example
    pooling_input = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    axes[1, 2].imshow(pooling_input, cmap='viridis', vmin=0, vmax=16)
    axes[1, 2].set_title('2x2 Max Pooling Example\nInput (4x4) → Output (2x2)\n[[6,8],[14,16]]')
    for i in range(4):
        for j in range(4):
            axes[1, 2].text(j, i, str(pooling_input[i, j]),
                          ha='center', va='center', color='white', fontsize=12)
    axes[1, 2].axis('off')

    plt.suptitle('Convolution and Pooling Operations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# PART 4: CNN MODELS
# =============================================================================

def build_basic_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Basic CNN: Simple architecture for learning.

    Architecture:
    Conv(32) → Pool → Conv(64) → Pool → Flatten → Dense(64) → Output
    """
    model = keras.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=input_shape, name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),

        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),

        # Third Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'),

        # Classifier
        layers.Flatten(name='flatten'),
        layers.Dense(64, activation='relu', name='dense1'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='Basic_CNN')

    return model


def build_improved_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Improved CNN: With BatchNorm and Dropout.

    Architecture follows the pattern: Conv → BN → ReLU → Conv → BN → ReLU → Pool → Dropout
    """
    model = keras.Sequential([
        # Block 1: 32 filters
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape, name='conv1a'),
        layers.BatchNormalization(name='bn1a'),
        layers.Activation('relu', name='relu1a'),
        layers.Conv2D(32, (3, 3), padding='same', name='conv1b'),
        layers.BatchNormalization(name='bn1b'),
        layers.Activation('relu', name='relu1b'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.25, name='dropout1'),

        # Block 2: 64 filters
        layers.Conv2D(64, (3, 3), padding='same', name='conv2a'),
        layers.BatchNormalization(name='bn2a'),
        layers.Activation('relu', name='relu2a'),
        layers.Conv2D(64, (3, 3), padding='same', name='conv2b'),
        layers.BatchNormalization(name='bn2b'),
        layers.Activation('relu', name='relu2b'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.25, name='dropout2'),

        # Block 3: 128 filters
        layers.Conv2D(128, (3, 3), padding='same', name='conv3a'),
        layers.BatchNormalization(name='bn3a'),
        layers.Activation('relu', name='relu3a'),
        layers.Conv2D(128, (3, 3), padding='same', name='conv3b'),
        layers.BatchNormalization(name='bn3b'),
        layers.Activation('relu', name='relu3b'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(0.25, name='dropout3'),

        # Classifier
        layers.Flatten(name='flatten'),
        layers.Dense(256, name='dense1'),
        layers.BatchNormalization(name='bn_dense'),
        layers.Activation('relu', name='relu_dense'),
        layers.Dropout(0.5, name='dropout_dense'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='Improved_CNN')

    return model


def build_vgg_style_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    VGG-style CNN: Following VGG architectural principles.

    VGG principles:
    - Only 3x3 convolutions
    - Double filters after each pooling
    - Multiple conv layers before pooling
    """
    model = keras.Sequential([
        # Block 1: 64 filters
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     input_shape=input_shape, name='conv1_1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'),
        layers.MaxPooling2D((2, 2), strides=2, name='pool1'),
        layers.Dropout(0.2, name='dropout1'),

        # Block 2: 128 filters
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'),
        layers.MaxPooling2D((2, 2), strides=2, name='pool2'),
        layers.Dropout(0.3, name='dropout2'),

        # Block 3: 256 filters
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3'),
        layers.MaxPooling2D((2, 2), strides=2, name='pool3'),
        layers.Dropout(0.4, name='dropout3'),

        # Classifier
        layers.Flatten(name='flatten'),
        layers.Dense(512, activation='relu', name='fc1'),
        layers.Dropout(0.5, name='dropout_fc1'),
        layers.Dense(256, activation='relu', name='fc2'),
        layers.Dropout(0.5, name='dropout_fc2'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='VGG_Style_CNN')

    return model


# =============================================================================
# PART 5: DATA AUGMENTATION
# =============================================================================

def create_data_augmentation():
    """Create data augmentation layer."""
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name='data_augmentation')

    return data_augmentation


def build_cnn_with_augmentation(input_shape=(32, 32, 3), num_classes=10):
    """CNN with built-in data augmentation."""
    data_augmentation = create_data_augmentation()

    model = keras.Sequential([
        # Data augmentation (only active during training)
        data_augmentation,

        # Conv blocks
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Global Average Pooling instead of Flatten
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='CNN_with_Augmentation')

    return model


# =============================================================================
# PART 6: TRAINING
# =============================================================================

def get_callbacks(model_name):
    """Get training callbacks."""
    return [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
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
                epochs=50, batch_size=64, lr=0.001):
    """Train a model and return history."""
    print(f"\n{'='*60}")
    print(f"Training: {model.name}")
    print(f"{'='*60}")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
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
# PART 7: VISUALIZATION
# =============================================================================

def visualize_filters(model, layer_name='conv1'):
    """Visualize convolutional filters."""
    # Get the layer
    layer = None
    for l in model.layers:
        if l.name == layer_name:
            layer = l
            break

    if layer is None:
        print(f"Layer {layer_name} not found")
        return None

    filters, biases = layer.get_weights()
    print(f"Filter shape: {filters.shape}")

    # Normalize filters for visualization
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    n_filters = min(filters.shape[3], 32)
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 1.5))
    axes = axes.flatten()

    for i in range(n_filters):
        f = filters[:, :, :, i]
        # For RGB filters, show as RGB image
        if f.shape[2] == 3:
            axes[i].imshow(f)
        else:
            axes[i].imshow(f[:, :, 0], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'F{i+1}', fontsize=8)

    # Hide unused axes
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Learned Filters from {layer_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_feature_maps(model, image, layer_names=None):
    """Visualize feature maps for a single image."""
    if layer_names is None:
        # Get all conv layer names
        layer_names = [l.name for l in model.layers if 'conv' in l.name][:3]

    # Create a model that outputs feature maps
    outputs = [model.get_layer(name).output for name in layer_names]
    feature_model = keras.Model(inputs=model.input, outputs=outputs)

    # Get feature maps
    image_batch = np.expand_dims(image, axis=0)
    feature_maps = feature_model.predict(image_batch, verbose=0)

    fig, axes = plt.subplots(len(layer_names), 8, figsize=(16, len(layer_names) * 2))

    for row, (name, fmaps) in enumerate(zip(layer_names, feature_maps)):
        for col in range(8):
            if col < fmaps.shape[3]:
                ax = axes[row, col] if len(layer_names) > 1 else axes[col]
                ax.imshow(fmaps[0, :, :, col], cmap='viridis')
                ax.axis('off')
                if col == 0:
                    ax.set_ylabel(name, fontsize=10)

    plt.suptitle('Feature Maps at Different Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


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


def plot_confusion_matrix_cifar(model, X_test, y_test, class_names):
    """Plot confusion matrix."""
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix - CIFAR-10')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return fig


# =============================================================================
# PART 8: MAIN EXECUTION
# =============================================================================

def main():
    """Main function for CNN on CIFAR-10."""
    print("=" * 70)
    print("DAY 18: CONVOLUTIONAL NEURAL NETWORKS - PART 1")
    print("CNN for CIFAR-10 Classification")
    print("=" * 70)

    # Explain CNN concepts
    explain_cnn_concepts()

    # Visualize convolution operation
    fig_conv = visualize_convolution_operation()
    fig_conv.savefig('convolution_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved: convolution_visualization.png")

    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_cifar10()

    # Visualize samples
    fig_samples = visualize_cifar_samples(X_train, y_train, CLASS_NAMES)
    fig_samples.savefig('cifar10_samples.png', dpi=150, bbox_inches='tight')
    print("Saved: cifar10_samples.png")

    # Build models
    print("\n" + "=" * 60)
    print("BUILDING CNN MODELS")
    print("=" * 60)

    # We'll train the improved CNN for best results
    model = build_improved_cnn()
    model.summary()

    # Train
    history = train_model(model, X_train, y_train, X_val, y_val,
                         epochs=30, batch_size=64, lr=0.001)

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Plot training history
    fig_history = plot_training_history(history, model.name)
    fig_history.savefig('cnn_training_history.png', dpi=150, bbox_inches='tight')
    print("Saved: cnn_training_history.png")

    # Visualize filters
    fig_filters = visualize_filters(model, 'conv1a')
    if fig_filters:
        fig_filters.savefig('cnn_filters.png', dpi=150, bbox_inches='tight')
        print("Saved: cnn_filters.png")

    # Visualize feature maps
    sample_image = X_test[0]
    fig_fmaps = visualize_feature_maps(model, sample_image)
    fig_fmaps.savefig('cnn_feature_maps.png', dpi=150, bbox_inches='tight')
    print("Saved: cnn_feature_maps.png")

    # Confusion matrix
    fig_cm = plot_confusion_matrix_cifar(model, X_test, y_test, CLASS_NAMES)
    fig_cm.savefig('cifar10_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Saved: cifar10_confusion_matrix.png")

    # Save model
    model.save('cifar10_cnn_model.keras')
    print("Saved: cifar10_cnn_model.keras")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Results:
    - Model: {model.name}
    - Test Accuracy: {test_accuracy * 100:.2f}%
    - Total Parameters: {model.count_params():,}

    CNN Architecture Pattern:
    Conv → BN → ReLU → Conv → BN → ReLU → Pool → Dropout (repeated)
    → Flatten → Dense → Output

    Key Learnings:
    1. Convolution: Applies filters to detect local features
    2. Pooling: Reduces spatial dimensions, adds translation invariance
    3. BatchNorm: Normalizes activations, speeds training
    4. Dropout: Regularization to prevent overfitting
    5. Filter hierarchy: Low-level → High-level features

    Architecture Tips:
    - Use 3x3 filters (small but effective)
    - Double filters after each pooling
    - Use padding='same' to control output size
    - Add BatchNorm after Conv layers
    - Apply Dropout between blocks
    """)

    plt.close('all')
    print("\nAll visualizations saved successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
