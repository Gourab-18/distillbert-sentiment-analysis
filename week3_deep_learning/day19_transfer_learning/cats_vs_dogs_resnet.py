"""
Day 19: CNNs - Part 2 & Transfer Learning
Assignment: Use pre-trained ResNet50 for custom image classification (cats vs dogs)

This module covers:
- Famous architectures: LeNet, AlexNet, VGG, ResNet
- Transfer learning concept
- Fine-tuning pre-trained models
- Data augmentation techniques
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import (
    ResNet50, VGG16, MobileNetV2, EfficientNetB0
)
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")


# =============================================================================
# PART 1: FAMOUS CNN ARCHITECTURES EXPLANATION
# =============================================================================

def explain_famous_architectures():
    """Explain famous CNN architectures."""
    explanation = """
    ===================================================================
    FAMOUS CNN ARCHITECTURES
    ===================================================================

    1. LeNet-5 (1998) - Yann LeCun
    ------------------------------
    - First successful CNN for digit recognition
    - Architecture: Conv → Pool → Conv → Pool → FC → FC → Output
    - 60K parameters
    - Input: 32x32 grayscale
    - Key innovation: Demonstrated CNNs work for image classification

    2. AlexNet (2012) - Alex Krizhevsky
    -----------------------------------
    - Won ImageNet 2012, started the deep learning revolution
    - Architecture: 5 Conv layers + 3 FC layers
    - 60M parameters
    - Key innovations:
      • ReLU activation (instead of tanh/sigmoid)
      • Dropout for regularization
      • Data augmentation
      • GPU training

    3. VGGNet (2014) - Oxford Visual Geometry Group
    -----------------------------------------------
    - Deep but simple architecture
    - VGG16: 16 layers, VGG19: 19 layers
    - 138M parameters (VGG16)
    - Key innovations:
      • Only 3x3 convolutions
      • Two 3x3 = one 5x5 receptive field, fewer params
      • Stack of small filters instead of large ones

    4. GoogLeNet/Inception (2014) - Google
    --------------------------------------
    - Introduced "Inception module"
    - 22 layers but only 5M parameters
    - Key innovations:
      • Inception module: parallel conv with different kernel sizes
      • 1x1 convolutions for dimensionality reduction
      • Global Average Pooling instead of FC layers

    5. ResNet (2015) - Microsoft Research
    -------------------------------------
    - Introduced residual connections (skip connections)
    - Enabled training of 152+ layer networks
    - Won ImageNet 2015
    - Key innovations:
      • Skip connections: y = F(x) + x
      • Solves vanishing gradient in very deep networks
      • Identity mapping allows gradients to flow directly

    6. DenseNet (2017)
    ------------------
    - Each layer connected to all subsequent layers
    - Feature reuse, fewer parameters
    - Key innovation: Dense connections

    7. EfficientNet (2019) - Google
    -------------------------------
    - Compound scaling of depth, width, resolution
    - State-of-the-art efficiency
    - EfficientNetB0 to B7

    8. Vision Transformer (ViT) (2020)
    ----------------------------------
    - Applies Transformer architecture to images
    - Patches images, treats them as sequences
    - Competitive with CNNs, especially with large data

    ===================================================================
    TRANSFER LEARNING
    ===================================================================

    What is Transfer Learning?
    - Use knowledge from one task to help with another task
    - Pre-train on large dataset (ImageNet: 14M images, 1000 classes)
    - Transfer learned features to new, smaller dataset

    Why Transfer Learning?
    - Saves training time
    - Requires less data
    - Better performance on small datasets
    - Lower layers learn general features (edges, textures)

    Two Approaches:

    1. FEATURE EXTRACTION
    ---------------------
    - Freeze pre-trained layers (don't update weights)
    - Only train new classifier layers
    - Fast, works with small datasets
    - Use when: new dataset is small, similar to original

    2. FINE-TUNING
    --------------
    - Unfreeze some/all pre-trained layers
    - Train entire network with low learning rate
    - Better performance, more data needed
    - Use when: new dataset is larger, different from original

    ===================================================================
    """
    print(explanation)


# =============================================================================
# PART 2: DATA LOADING AND AUGMENTATION
# =============================================================================

def create_synthetic_cats_dogs_data(num_samples=2000, img_size=(224, 224)):
    """
    Create synthetic data for demonstration.
    In practice, you would load the actual Cats vs Dogs dataset.
    """
    print("Creating synthetic cats vs dogs data for demonstration...")

    # Create random images with different patterns
    np.random.seed(42)

    X_cats = []
    X_dogs = []

    for _ in range(num_samples // 2):
        # "Cat" pattern: more horizontal features, orange-ish tint
        cat_img = np.random.rand(img_size[0], img_size[1], 3) * 0.5
        cat_img[:, :, 0] += 0.3  # More red
        cat_img[:, :, 1] += 0.2  # Some green
        # Add horizontal lines (whisker-like)
        for y in range(0, img_size[0], 20):
            cat_img[y:y+2, :, :] = 0.8
        X_cats.append(cat_img)

        # "Dog" pattern: more vertical features, brown tint
        dog_img = np.random.rand(img_size[0], img_size[1], 3) * 0.5
        dog_img[:, :, 0] += 0.4  # More red
        dog_img[:, :, 1] += 0.3  # Green (brown = red + green)
        # Add vertical lines (ear-like)
        for x in range(0, img_size[1], 20):
            dog_img[:, x:x+2, :] = 0.8
        X_dogs.append(dog_img)

    X_cats = np.array(X_cats).astype('float32')
    X_dogs = np.array(X_dogs).astype('float32')

    # Clip values to [0, 1]
    X_cats = np.clip(X_cats, 0, 1)
    X_dogs = np.clip(X_dogs, 0, 1)

    # Combine and create labels
    X = np.concatenate([X_cats, X_dogs], axis=0)
    y = np.array([0] * len(X_cats) + [1] * len(X_dogs))

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_data_augmentation_layer():
    """Create comprehensive data augmentation."""
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
    ], name='data_augmentation')

    return data_augmentation


def visualize_augmentation(X, data_augmentation, num_variations=8):
    """Visualize data augmentation effects."""
    fig, axes = plt.subplots(2, num_variations, figsize=(16, 4))

    # Original image
    sample_image = X[0:1]

    # Show original
    axes[0, 0].imshow(sample_image[0])
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    # Show augmented versions
    for i in range(1, num_variations):
        augmented = data_augmentation(sample_image, training=True)
        axes[0, i].imshow(augmented[0])
        axes[0, i].set_title(f'Aug {i}')
        axes[0, i].axis('off')

    # Another image
    sample_image2 = X[1:2]
    axes[1, 0].imshow(sample_image2[0])
    axes[1, 0].set_title('Original')
    axes[1, 0].axis('off')

    for i in range(1, num_variations):
        augmented = data_augmentation(sample_image2, training=True)
        axes[1, i].imshow(augmented[0])
        axes[1, i].set_title(f'Aug {i}')
        axes[1, i].axis('off')

    plt.suptitle('Data Augmentation Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# PART 3: TRANSFER LEARNING MODELS
# =============================================================================

def build_feature_extraction_model(base_model_name='resnet50', img_size=(224, 224),
                                   num_classes=2):
    """
    Build model using pre-trained network as feature extractor.
    Freeze all base model layers.
    """
    # Select base model
    base_models = {
        'resnet50': ResNet50,
        'vgg16': VGG16,
        'mobilenet': MobileNetV2,
        'efficientnet': EfficientNetB0
    }

    base_model_class = base_models.get(base_model_name.lower(), ResNet50)

    # Load pre-trained model without top layers
    base_model = base_model_class(
        weights='imagenet',  # Pre-trained on ImageNet
        include_top=False,   # Remove classification layers
        input_shape=(*img_size, 3)
    )

    # Freeze base model
    base_model.trainable = False

    # Build new model
    inputs = keras.Input(shape=(*img_size, 3))

    # Data augmentation
    x = create_data_augmentation_layer()(inputs)

    # Preprocess for specific model
    if base_model_name.lower() == 'resnet50':
        x = keras.applications.resnet50.preprocess_input(x)
    elif base_model_name.lower() == 'vgg16':
        x = keras.applications.vgg16.preprocess_input(x)
    elif base_model_name.lower() == 'mobilenet':
        x = keras.applications.mobilenet_v2.preprocess_input(x)
    else:
        x = keras.applications.efficientnet.preprocess_input(x)

    # Base model
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'

    model = keras.Model(inputs, outputs, name=f'{base_model_name}_FeatureExtraction')

    return model, base_model, loss


def build_fine_tuning_model(base_model_name='resnet50', img_size=(224, 224),
                            num_classes=2, fine_tune_at=100):
    """
    Build model for fine-tuning.
    Unfreeze top layers of base model.
    """
    # First, get the feature extraction model
    model, base_model, loss = build_feature_extraction_model(
        base_model_name, img_size, num_classes
    )

    # Unfreeze the base model from a certain layer
    base_model.trainable = True

    # Freeze all layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"\nFine-tuning from layer {fine_tune_at}")
    print(f"Trainable layers: {len([l for l in base_model.layers if l.trainable])}")
    print(f"Frozen layers: {len([l for l in base_model.layers if not l.trainable])}")

    model._name = f'{base_model_name}_FineTuning'

    return model, base_model, loss


def build_simple_cnn(img_size=(224, 224), num_classes=2):
    """Build a simple CNN from scratch for comparison."""
    model = keras.Sequential([
        create_data_augmentation_layer(),

        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*img_size, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='Simple_CNN_FromScratch')

    return model, None, 'binary_crossentropy'


# =============================================================================
# PART 4: TRAINING
# =============================================================================

def get_callbacks(model_name):
    """Get training callbacks."""
    return [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=f'{model_name}_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]


def train_model(model, X_train, y_train, X_val, y_val, loss,
                epochs=20, batch_size=32, lr=0.001, model_name='model'):
    """Train model and return history."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(model_name),
        verbose=1
    )

    return history


def fine_tune_training(model, X_train, y_train, X_val, y_val,
                       epochs=10, batch_size=32):
    """Fine-tuning training with low learning rate."""
    print("\nStarting fine-tuning with low learning rate...")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),  # Very low LR for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(model.name + '_finetuned'),
        verbose=1
    )

    return history


# =============================================================================
# PART 5: VISUALIZATION
# =============================================================================

def plot_training_comparison(histories, names):
    """Compare training histories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for history, name in zip(histories, names):
        axes[0].plot(history.history['loss'], label=f'{name} (train)', linestyle='-')
        axes[0].plot(history.history['val_loss'], label=f'{name} (val)', linestyle='--')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for history, name in zip(histories, names):
        axes[1].plot(history.history['accuracy'], label=f'{name} (train)', linestyle='-')
        axes[1].plot(history.history['val_accuracy'], label=f'{name} (val)', linestyle='--')

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_predictions(model, X_test, y_test, class_names=['Cat', 'Dog'],
                         num_samples=10):
    """Visualize model predictions."""
    predictions = model.predict(X_test[:num_samples], verbose=0)

    if predictions.shape[1] == 1:
        pred_classes = (predictions > 0.5).astype(int).flatten()
        pred_probs = predictions.flatten()
    else:
        pred_classes = np.argmax(predictions, axis=1)
        pred_probs = predictions.max(axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(num_samples):
        axes[i].imshow(X_test[i])
        true_label = class_names[y_test[i]]
        pred_label = class_names[pred_classes[i]]
        color = 'green' if pred_classes[i] == y_test[i] else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\n'
                         f'Conf: {pred_probs[i]:.2f}',
                         color=color, fontsize=9)
        axes[i].axis('off')

    plt.suptitle('Model Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_model_architecture():
    """Create a visual comparison of architectures."""
    fig, ax = plt.subplots(figsize=(14, 8))

    architectures = {
        'LeNet (1998)': {'layers': 5, 'params': 0.06, 'top1': 99.0},
        'AlexNet (2012)': {'layers': 8, 'params': 60, 'top1': 63.3},
        'VGG16 (2014)': {'layers': 16, 'params': 138, 'top1': 71.3},
        'GoogLeNet (2014)': {'layers': 22, 'params': 5, 'top1': 74.8},
        'ResNet50 (2015)': {'layers': 50, 'params': 25, 'top1': 76.0},
        'ResNet152 (2015)': {'layers': 152, 'params': 60, 'top1': 77.8},
        'DenseNet121 (2017)': {'layers': 121, 'params': 8, 'top1': 74.9},
        'EfficientNetB0 (2019)': {'layers': 82, 'params': 5.3, 'top1': 77.1},
    }

    names = list(architectures.keys())
    layers_count = [architectures[n]['layers'] for n in names]
    params = [architectures[n]['params'] for n in names]
    accuracy = [architectures[n]['top1'] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, layers_count, width, label='Layers', color='steelblue')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, accuracy, width, label='ImageNet Top-1 %', color='coral')

    ax.set_xlabel('Architecture')
    ax.set_ylabel('Number of Layers', color='steelblue')
    ax2.set_ylabel('ImageNet Top-1 Accuracy %', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('CNN Architecture Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# PART 6: MAIN EXECUTION
# =============================================================================

def main():
    """Main function demonstrating transfer learning."""
    print("=" * 70)
    print("DAY 19: CNNs PART 2 & TRANSFER LEARNING")
    print("Pre-trained ResNet50 for Cats vs Dogs Classification")
    print("=" * 70)

    # Explain architectures
    explain_famous_architectures()

    # Visualize architecture comparison
    fig_arch = visualize_model_architecture()
    fig_arch.savefig('architecture_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: architecture_comparison.png")

    # Create synthetic data (in practice, load real cats vs dogs dataset)
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_synthetic_cats_dogs_data(
        num_samples=1000, img_size=(224, 224)
    )

    # Visualize augmentation
    data_aug = create_data_augmentation_layer()
    fig_aug = visualize_augmentation(X_train, data_aug)
    fig_aug.savefig('data_augmentation.png', dpi=150, bbox_inches='tight')
    print("Saved: data_augmentation.png")

    # Store results
    histories = []
    names = []
    accuracies = []

    # 1. Simple CNN from scratch
    print("\n" + "=" * 60)
    print("1. TRAINING SIMPLE CNN FROM SCRATCH")
    print("=" * 60)

    simple_model, _, simple_loss = build_simple_cnn()
    simple_model.summary()

    history_simple = train_model(
        simple_model, X_train, y_train, X_val, y_val,
        loss=simple_loss, epochs=15, batch_size=32, lr=0.001,
        model_name='simple_cnn'
    )

    simple_acc = simple_model.evaluate(X_test, y_test, verbose=0)[1]
    histories.append(history_simple)
    names.append('Simple CNN')
    accuracies.append(simple_acc)

    # 2. Feature Extraction with ResNet50
    print("\n" + "=" * 60)
    print("2. FEATURE EXTRACTION WITH RESNET50")
    print("=" * 60)

    fe_model, base_model, fe_loss = build_feature_extraction_model('resnet50')
    fe_model.summary()

    history_fe = train_model(
        fe_model, X_train, y_train, X_val, y_val,
        loss=fe_loss, epochs=10, batch_size=32, lr=0.001,
        model_name='resnet50_feature_extraction'
    )

    fe_acc = fe_model.evaluate(X_test, y_test, verbose=0)[1]
    histories.append(history_fe)
    names.append('ResNet50 Feature Extraction')
    accuracies.append(fe_acc)

    # 3. Fine-tuning (build new model and do fine-tuning)
    print("\n" + "=" * 60)
    print("3. FINE-TUNING RESNET50")
    print("=" * 60)

    ft_model, ft_base, ft_loss = build_fine_tuning_model('resnet50', fine_tune_at=100)

    # First train with frozen base
    history_ft1 = train_model(
        ft_model, X_train, y_train, X_val, y_val,
        loss=ft_loss, epochs=5, batch_size=32, lr=0.001,
        model_name='resnet50_finetune_phase1'
    )

    # Then fine-tune
    history_ft2 = fine_tune_training(
        ft_model, X_train, y_train, X_val, y_val,
        epochs=5, batch_size=32
    )

    ft_acc = ft_model.evaluate(X_test, y_test, verbose=0)[1]
    histories.append(history_ft2)
    names.append('ResNet50 Fine-tuning')
    accuracies.append(ft_acc)

    # Compare results
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    fig_comparison = plot_training_comparison(histories, names)
    fig_comparison.savefig('transfer_learning_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: transfer_learning_comparison.png")

    # Visualize predictions from best model
    best_idx = np.argmax(accuracies)
    best_model = [simple_model, fe_model, ft_model][best_idx]

    fig_preds = visualize_predictions(best_model, X_test, y_test)
    fig_preds.savefig('predictions_cats_dogs.png', dpi=150, bbox_inches='tight')
    print("Saved: predictions_cats_dogs.png")

    # Save best model
    best_model.save('cats_dogs_best_model.keras')
    print("Saved: cats_dogs_best_model.keras")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nModel Test Accuracies:")
    for name, acc in zip(names, accuracies):
        print(f"  {name}: {acc * 100:.2f}%")

    print(f"""
    Key Learnings:

    1. TRANSFER LEARNING APPROACHES:
       - Feature Extraction: Freeze pre-trained weights, train only classifier
       - Fine-tuning: Unfreeze some layers, train with low learning rate

    2. WHY TRANSFER LEARNING WORKS:
       - Lower layers learn general features (edges, textures)
       - Higher layers learn task-specific features
       - Pre-trained features transfer well to similar tasks

    3. BEST PRACTICES:
       - Start with feature extraction, then fine-tune if needed
       - Use very low learning rate for fine-tuning (1e-5 to 1e-4)
       - Unfreeze from top layers (task-specific) first
       - Use data augmentation to prevent overfitting
       - BatchNorm layers: keep in inference mode during fine-tuning

    4. WHEN TO USE EACH APPROACH:
       - Feature Extraction: Small dataset, similar to ImageNet
       - Fine-tuning: Larger dataset, different from ImageNet
       - From Scratch: Very large dataset, very different domain

    5. PRE-TRAINED MODELS:
       - ResNet50: Good balance of accuracy and speed
       - EfficientNet: Best accuracy-efficiency trade-off
       - MobileNet: For mobile/edge deployment
       - VGG: Simple but large, good for feature extraction
    """)

    plt.close('all')
    print("\nAll visualizations saved successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
