# Week 3: Deep Learning Foundations

A comprehensive curriculum covering neural networks, CNNs, transfer learning, and RNNs.

## Overview

This week covers the essential foundations of deep learning:

| Day | Topic | Assignment |
|-----|-------|------------|
| 15 | Neural Networks Theory | Perceptron for logic gates |
| 16 | TensorFlow/Keras | 3-layer NN for MNIST |
| 17 | Deep Neural Networks | Fashion-MNIST >90% accuracy |
| 18 | CNN Part 1 | CIFAR-10 classification |
| 19 | Transfer Learning | Cats vs Dogs with ResNet50 |
| 20 | RNNs | IMDB Sentiment Analysis |
| 21 | Week Project | Image Classifier Web App |

## Directory Structure

```
week3_deep_learning/
├── day15_neural_networks/
│   └── perceptron_from_scratch.py    # Perceptron, activations, logic gates
├── day16_tensorflow_keras/
│   └── mnist_neural_network.py       # MNIST with Sequential & Functional API
├── day17_deep_networks/
│   └── fashion_mnist_deep.py         # 5+ layer networks, regularization
├── day18_cnn_part1/
│   └── cifar10_cnn.py                # CNN architecture, convolution ops
├── day19_transfer_learning/
│   └── cats_vs_dogs_resnet.py        # Transfer learning, fine-tuning
├── day20_rnn/
│   └── imdb_sentiment_rnn.py         # LSTM, GRU, BiLSTM sentiment analysis
├── day21_project/
│   └── image_classifier_app.py       # Complete Streamlit web app
├── app_week3_download.py             # Download portal app
├── requirements_week3.txt            # Python dependencies
└── README_week3.md                   # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements_week3.txt
```

### 2. Run Individual Days

```bash
# Day 15: Perceptron
python day15_neural_networks/perceptron_from_scratch.py

# Day 16: MNIST Neural Network
python day16_tensorflow_keras/mnist_neural_network.py

# Day 17: Fashion-MNIST Deep Network
python day17_deep_networks/fashion_mnist_deep.py

# Day 18: CIFAR-10 CNN
python day18_cnn_part1/cifar10_cnn.py

# Day 19: Transfer Learning
python day19_transfer_learning/cats_vs_dogs_resnet.py

# Day 20: RNN Sentiment Analysis
python day20_rnn/imdb_sentiment_rnn.py
```

### 3. Run Web Apps

```bash
# Day 21 Project: Image Classifier
streamlit run day21_project/image_classifier_app.py

# Download Portal
streamlit run app_week3_download.py
```

## Day-by-Day Content

### Day 15: Neural Networks Theory

**Concepts:**
- Perceptron architecture
- Activation functions: sigmoid, tanh, ReLU, leaky ReLU
- Forward propagation mathematics
- Backpropagation and chain rule
- Gradient descent optimization
- Loss functions: MSE, binary cross-entropy

**Key Code:**
```python
# Single-layer perceptron
class Perceptron:
    def forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)

    def backward(self, X, y, y_pred):
        dw = np.dot(X.T, (y_pred - y)) / len(y)
        db = np.mean(y_pred - y)
        return dw, db
```

### Day 16: TensorFlow/Keras

**Concepts:**
- TensorFlow ecosystem
- Sequential API vs Functional API
- Layer types: Dense, Dropout, BatchNormalization
- Model compilation: optimizers, loss, metrics
- Training callbacks

**Key Code:**
```python
# Sequential API
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# Functional API
inputs = keras.Input(shape=(784,))
x = layers.Dense(256, activation='relu')(inputs)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs, outputs)
```

### Day 17: Deep Neural Networks

**Concepts:**
- Deep architecture design
- L1/L2 regularization
- Dropout for regularization
- Learning rate schedules
- Early stopping
- Model checkpointing

**Key Code:**
```python
# Deep network with regularization
layers.Dense(256, activation='relu',
             kernel_regularizer=regularizers.l2(0.001))

# Learning rate schedule
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.96
)
```

### Day 18: CNN Part 1

**Concepts:**
- Convolution operation
- Filters and feature maps
- Pooling: max and average
- CNN architecture patterns
- Spatial hierarchies

**Key Code:**
```python
# CNN Block
layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
layers.BatchNormalization(),
layers.MaxPooling2D((2, 2)),
layers.Dropout(0.25)
```

### Day 19: Transfer Learning

**Concepts:**
- Famous architectures: VGG, ResNet, EfficientNet
- Feature extraction
- Fine-tuning
- Data augmentation

**Key Code:**
```python
# Transfer learning with ResNet50
base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Feature extraction

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False
```

### Day 20: RNNs

**Concepts:**
- Sequential data modeling
- Vanishing gradient problem
- LSTM and GRU units
- Bidirectional RNNs
- Sentiment analysis

**Key Code:**
```python
# Bidirectional LSTM
model = keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(1, activation='sigmoid')
])
```

### Day 21: Week Project

**Features:**
- Complete image classifier
- Scratch vs transfer learning comparison
- Interactive Streamlit UI
- Model training with progress
- Image upload and prediction

## Learning Outcomes

By completing this week, you will:

1. **Understand neural network fundamentals**
   - How perceptrons work
   - Activation functions and their derivatives
   - Forward and backward propagation

2. **Master TensorFlow/Keras**
   - Build models with Sequential and Functional APIs
   - Use various layer types effectively
   - Configure training with callbacks

3. **Build effective CNNs**
   - Understand convolution and pooling
   - Design CNN architectures
   - Apply data augmentation

4. **Apply transfer learning**
   - Use pre-trained models
   - Feature extraction vs fine-tuning
   - Choose the right approach for your data

5. **Work with sequential data**
   - Understand RNN limitations
   - Implement LSTM and GRU models
   - Build sentiment analysis systems

6. **Deploy ML models**
   - Create web interfaces with Streamlit
   - Compare model performance
   - Present results effectively

## Tips for Success

1. **Run each day's code** before moving to the next
2. **Experiment** with hyperparameters
3. **Read the comments** - they contain important explanations
4. **Check the visualizations** - they help understand concepts
5. **Use GPU** if available for faster training
6. **Take notes** on what works and what doesn't

## Troubleshooting

### TensorFlow Installation Issues

```bash
# For Mac M1/M2
pip install tensorflow-macos tensorflow-metal

# For GPU support
pip install tensorflow[and-cuda]
```

### Memory Issues

- Reduce batch size
- Use smaller models
- Enable memory growth:
```python
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Streamlit Issues

```bash
# Clear cache
streamlit cache clear

# Run with specific port
streamlit run app.py --server.port 8502
```

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Deep Learning Book](https://www.deeplearningbook.org/)

## License

This educational content is provided for learning purposes.

---

Happy Learning!
