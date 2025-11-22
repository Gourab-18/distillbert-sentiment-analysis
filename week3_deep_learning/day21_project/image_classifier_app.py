"""
Day 21: Week 3 Project
Assignment: Complete image classifier web app with UI

This module covers:
- Computer vision project: build image classifier for 10 classes
- Implement from scratch and with transfer learning
- Compare performance and training time
- Deploy model using Streamlit

Run with: streamlit run image_classifier_app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
import time
from PIL import Image
import io
import os

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


# =============================================================================
# CONSTANTS
# =============================================================================

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Model save paths
SCRATCH_MODEL_PATH = 'cifar10_scratch_model.keras'
TRANSFER_MODEL_PATH = 'cifar10_transfer_model.keras'


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

def build_cnn_from_scratch(input_shape=(32, 32, 3), num_classes=10):
    """Build CNN from scratch for CIFAR-10."""
    model = keras.Sequential([
        # Data augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='CNN_From_Scratch')

    return model


def build_transfer_learning_model(input_shape=(32, 32, 3), num_classes=10):
    """Build transfer learning model using MobileNetV2."""
    # MobileNetV2 expects at least 32x32, but works better with larger
    # We'll resize CIFAR-10 images to 96x96

    inputs = keras.Input(shape=input_shape)

    # Data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    # Resize to larger size for better transfer learning
    x = layers.Resizing(96, 96)(x)

    # Preprocess for MobileNetV2
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    # Base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(96, 96, 3)
    )
    base_model.trainable = False

    x = base_model(x, training=False)

    # Classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='Transfer_Learning_MobileNetV2')

    return model, base_model


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

@st.cache_data
def load_cifar10():
    """Load and preprocess CIFAR-10 dataset."""
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Squeeze labels
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # Normalize
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Validation split
    val_size = 5000
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_model_with_progress(model, X_train, y_train, X_val, y_val,
                               epochs=20, batch_size=64, model_name='model'):
    """Train model with Streamlit progress bar."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Custom callback for Streamlit progress
    class StreamlitCallback(callbacks.Callback):
        def __init__(self, epochs, progress_bar, status_text):
            self.epochs = epochs
            self.progress_bar = progress_bar
            self.status_text = status_text
            self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / self.epochs
            self.progress_bar.progress(progress)
            self.status_text.text(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Loss: {logs['loss']:.4f} - Acc: {logs['accuracy']:.4f} - "
                f"Val Loss: {logs['val_loss']:.4f} - Val Acc: {logs['val_accuracy']:.4f}"
            )
            for key in self.history:
                self.history[key].append(logs[key])

    progress_bar = st.progress(0)
    status_text = st.empty()

    streamlit_callback = StreamlitCallback(epochs, progress_bar, status_text)

    start_time = time.time()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            streamlit_callback,
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ],
        verbose=0
    )

    training_time = time.time() - start_time

    return streamlit_callback.history, training_time


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_training_history(history, model_name):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['loss'], 'b-', label='Training')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name}: Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['accuracy'], 'b-', label='Training')
    axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name}: Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison(scratch_history, transfer_history, scratch_time, transfer_time,
                   scratch_acc, transfer_acc):
    """Plot comparison between scratch and transfer learning."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Validation accuracy comparison
    axes[0, 0].plot(scratch_history['val_accuracy'], 'b-', label='From Scratch', linewidth=2)
    axes[0, 0].plot(transfer_history['val_accuracy'], 'r-', label='Transfer Learning', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Validation Accuracy')
    axes[0, 0].set_title('Validation Accuracy Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Validation loss comparison
    axes[0, 1].plot(scratch_history['val_loss'], 'b-', label='From Scratch', linewidth=2)
    axes[0, 1].plot(transfer_history['val_loss'], 'r-', label='Transfer Learning', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss')
    axes[0, 1].set_title('Validation Loss Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Training time comparison
    times = [scratch_time, transfer_time]
    names = ['From Scratch', 'Transfer Learning']
    colors = ['steelblue', 'coral']
    bars = axes[1, 0].bar(names, times, color=colors)
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Training Time Comparison')
    for bar, t in zip(bars, times):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{t:.1f}s', ha='center')

    # Final accuracy comparison
    accs = [scratch_acc * 100, transfer_acc * 100]
    bars = axes[1, 1].bar(names, accs, color=colors)
    axes[1, 1].set_ylabel('Test Accuracy (%)')
    axes[1, 1].set_title('Final Test Accuracy Comparison')
    axes[1, 1].set_ylim([0, 100])
    for bar, a in zip(bars, accs):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{a:.1f}%', ha='center')

    plt.tight_layout()
    return fig


def plot_predictions(model, images, true_labels, class_names):
    """Plot predictions for sample images."""
    predictions = model.predict(images, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)

    n_images = len(images)
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_images):
        row, col = i // n_cols, i % n_cols
        axes[row, col].imshow(images[i])
        true_label = class_names[true_labels[i]]
        pred_label = class_names[pred_classes[i]]
        confidence = predictions[i].max() * 100
        color = 'green' if pred_classes[i] == true_labels[i] else 'red'
        axes[row, col].set_title(f'True: {true_label}\nPred: {pred_label}\n{confidence:.1f}%',
                                 color=color, fontsize=9)
        axes[row, col].axis('off')

    # Hide unused axes
    for i in range(n_images, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    return fig


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Week 3 Project: Image Classifier",
        page_icon="",
        layout="wide"
    )

    st.title(" Week 3 Project: Image Classifier")
    st.markdown("""
    This project demonstrates building an image classifier for CIFAR-10 (10 classes):
    - **From Scratch**: Custom CNN architecture
    - **Transfer Learning**: Pre-trained MobileNetV2

    Compare performance and training time between both approaches!
    """)

    # Sidebar
    st.sidebar.header("Settings")

    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Train Models", "Test Classifier", "View Comparison", "About"]
    )

    if mode == "Train Models":
        train_models_page()
    elif mode == "Test Classifier":
        test_classifier_page()
    elif mode == "View Comparison":
        view_comparison_page()
    else:
        about_page()


def train_models_page():
    """Page for training models."""
    st.header("Train Image Classifiers")

    # Load data
    st.subheader("1. Load CIFAR-10 Dataset")
    if st.button("Load Dataset"):
        with st.spinner("Loading CIFAR-10..."):
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_cifar10()
            st.session_state['data'] = {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            }
            st.success(f"Loaded! Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

            # Show sample images
            fig, axes = plt.subplots(2, 5, figsize=(12, 5))
            for i, ax in enumerate(axes.flatten()):
                idx = np.random.randint(len(X_train))
                ax.imshow(X_train[idx])
                ax.set_title(CIFAR10_CLASSES[y_train[idx]])
                ax.axis('off')
            st.pyplot(fig)
            plt.close()

    # Training parameters
    st.subheader("2. Training Parameters")
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Epochs", 5, 50, 15)
    with col2:
        batch_size = st.selectbox("Batch Size", [32, 64, 128], index=1)

    # Train from scratch
    st.subheader("3. Train CNN from Scratch")
    if st.button("Train From Scratch"):
        if 'data' not in st.session_state:
            st.error("Please load the dataset first!")
        else:
            data = st.session_state['data']
            model_scratch = build_cnn_from_scratch()

            st.write(f"Model Parameters: {model_scratch.count_params():,}")

            history_scratch, time_scratch = train_model_with_progress(
                model_scratch,
                data['X_train'], data['y_train'],
                data['X_val'], data['y_val'],
                epochs=epochs, batch_size=batch_size,
                model_name='CNN_From_Scratch'
            )

            # Evaluate
            test_loss, test_acc = model_scratch.evaluate(
                data['X_test'], data['y_test'], verbose=0
            )

            st.success(f"Training Complete! Time: {time_scratch:.1f}s, Test Accuracy: {test_acc*100:.2f}%")

            # Save results
            st.session_state['scratch_results'] = {
                'history': history_scratch,
                'time': time_scratch,
                'accuracy': test_acc,
                'model': model_scratch
            }

            # Plot history
            fig = plot_training_history(history_scratch, "CNN From Scratch")
            st.pyplot(fig)
            plt.close()

    # Train with transfer learning
    st.subheader("4. Train with Transfer Learning")
    if st.button("Train Transfer Learning"):
        if 'data' not in st.session_state:
            st.error("Please load the dataset first!")
        else:
            data = st.session_state['data']
            model_transfer, base_model = build_transfer_learning_model()

            st.write(f"Model Parameters: {model_transfer.count_params():,}")
            st.write(f"Trainable Parameters: {sum(np.prod(w.shape) for w in model_transfer.trainable_weights):,}")

            history_transfer, time_transfer = train_model_with_progress(
                model_transfer,
                data['X_train'], data['y_train'],
                data['X_val'], data['y_val'],
                epochs=epochs, batch_size=batch_size,
                model_name='Transfer_Learning'
            )

            # Evaluate
            test_loss, test_acc = model_transfer.evaluate(
                data['X_test'], data['y_test'], verbose=0
            )

            st.success(f"Training Complete! Time: {time_transfer:.1f}s, Test Accuracy: {test_acc*100:.2f}%")

            # Save results
            st.session_state['transfer_results'] = {
                'history': history_transfer,
                'time': time_transfer,
                'accuracy': test_acc,
                'model': model_transfer
            }

            # Plot history
            fig = plot_training_history(history_transfer, "Transfer Learning")
            st.pyplot(fig)
            plt.close()


def test_classifier_page():
    """Page for testing the classifier."""
    st.header("Test Image Classifier")

    # Check if models are trained
    if 'scratch_results' not in st.session_state and 'transfer_results' not in st.session_state:
        st.warning("No trained models found. Please train models first!")
        return

    # Model selection
    available_models = []
    if 'scratch_results' in st.session_state:
        available_models.append("CNN From Scratch")
    if 'transfer_results' in st.session_state:
        available_models.append("Transfer Learning")

    selected_model = st.selectbox("Select Model", available_models)

    if selected_model == "CNN From Scratch":
        model = st.session_state['scratch_results']['model']
    else:
        model = st.session_state['transfer_results']['model']

    # Test on random samples
    st.subheader("Test on CIFAR-10 Samples")
    if st.button("Show Random Predictions"):
        if 'data' not in st.session_state:
            st.error("Please load the dataset first!")
        else:
            data = st.session_state['data']
            indices = np.random.choice(len(data['X_test']), 10, replace=False)
            images = data['X_test'][indices]
            labels = data['y_test'][indices]

            fig = plot_predictions(model, images, labels, CIFAR10_CLASSES)
            st.pyplot(fig)
            plt.close()

    # Upload custom image
    st.subheader("Upload Your Own Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=200)

        # Preprocess
        image = image.resize((32, 32))
        image_array = np.array(image) / 255.0

        # Handle grayscale
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]

        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        if st.button("Classify Image"):
            predictions = model.predict(image_array, verbose=0)
            pred_class = np.argmax(predictions[0])
            confidence = predictions[0].max() * 100

            st.success(f"Prediction: **{CIFAR10_CLASSES[pred_class]}** ({confidence:.1f}% confidence)")

            # Show top 5 predictions
            top5_idx = np.argsort(predictions[0])[::-1][:5]
            st.write("Top 5 Predictions:")
            for idx in top5_idx:
                st.write(f"- {CIFAR10_CLASSES[idx]}: {predictions[0][idx]*100:.1f}%")


def view_comparison_page():
    """Page for viewing comparison between models."""
    st.header("Model Comparison")

    if 'scratch_results' not in st.session_state or 'transfer_results' not in st.session_state:
        st.warning("Please train both models first to see the comparison!")
        return

    scratch = st.session_state['scratch_results']
    transfer = st.session_state['transfer_results']

    # Comparison metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("CNN From Scratch Accuracy", f"{scratch['accuracy']*100:.2f}%")
    with col2:
        st.metric("Transfer Learning Accuracy", f"{transfer['accuracy']*100:.2f}%")
    with col3:
        diff = (transfer['accuracy'] - scratch['accuracy']) * 100
        st.metric("Accuracy Difference", f"{diff:+.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("From Scratch Time", f"{scratch['time']:.1f}s")
    with col2:
        st.metric("Transfer Learning Time", f"{transfer['time']:.1f}s")

    # Detailed comparison plot
    st.subheader("Detailed Comparison")
    fig = plot_comparison(
        scratch['history'], transfer['history'],
        scratch['time'], transfer['time'],
        scratch['accuracy'], transfer['accuracy']
    )
    st.pyplot(fig)
    plt.close()

    # Analysis
    st.subheader("Analysis")
    st.markdown(f"""
    **Findings:**

    1. **Accuracy**: Transfer Learning achieved {transfer['accuracy']*100:.2f}% vs
       From Scratch {scratch['accuracy']*100:.2f}%

    2. **Training Time**: Transfer Learning took {transfer['time']:.1f}s vs
       From Scratch {scratch['time']:.1f}s

    3. **Convergence**: Check the training curves above to see which model
       converged faster.

    **Key Takeaways:**
    - Transfer learning leverages pre-trained features from ImageNet
    - Even with frozen base model, transfer learning often performs better
    - From-scratch models need more data and longer training for best results
    - The choice depends on your specific use case and constraints
    """)


def about_page():
    """About page with project information."""
    st.header("About This Project")

    st.markdown("""
    ## Week 3 Project: Deep Learning Image Classifier

    This project is the culmination of Week 3's deep learning curriculum, demonstrating:

    ### Concepts Covered

    **Day 15**: Neural Networks Theory
    - Perceptron, activation functions
    - Forward/backward propagation
    - Loss functions

    **Day 16**: TensorFlow/Keras
    - Sequential and Functional APIs
    - Dense, Dropout, BatchNorm layers

    **Day 17**: Deep Neural Networks
    - Regularization (L1, L2, Dropout)
    - Learning rate schedules
    - Early stopping

    **Day 18**: CNNs Part 1
    - Convolution operation
    - Pooling layers
    - CNN architectures

    **Day 19**: CNNs Part 2 & Transfer Learning
    - Famous architectures (VGG, ResNet)
    - Pre-trained models
    - Fine-tuning

    **Day 20**: RNNs
    - LSTM, GRU
    - Sequence modeling
    - Sentiment analysis

    **Day 21**: This Project!
    - Complete image classifier
    - Comparison of approaches
    - Web deployment with Streamlit

    ### Technologies Used
    - TensorFlow/Keras
    - Streamlit
    - NumPy, Matplotlib
    - MobileNetV2 (pre-trained)

    ### CIFAR-10 Classes
    """)

    # Show class names with icons
    cols = st.columns(5)
    icons = ['', '', '', '', '', '', '', '', '', '']
    for i, (name, icon) in enumerate(zip(CIFAR10_CLASSES, icons)):
        cols[i % 5].write(f"{icon} {name}")

    st.markdown("""
    ---
    Built as part of the Deep Learning Foundations curriculum.
    """)


if __name__ == "__main__":
    main()
