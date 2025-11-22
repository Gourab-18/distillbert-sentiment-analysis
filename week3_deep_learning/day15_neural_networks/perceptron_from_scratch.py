"""
Day 15: Neural Networks Theory
Assignment: Implement single-layer perceptron from scratch for AND/OR gates

This module covers:
- Perceptron and activation functions (sigmoid, tanh, ReLU)
- Forward propagation mathematics
- Backpropagation and gradient descent
- Loss functions: MSE, cross-entropy
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PART 1: ACTIVATION FUNCTIONS
# =============================================================================

class ActivationFunctions:
    """Collection of activation functions and their derivatives."""

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))

        Properties:
        - Output range: (0, 1)
        - Smooth and differentiable
        - Used for binary classification output
        - Problem: Vanishing gradient for very large/small inputs
        """
        # Clip to avoid overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))"""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """
        Hyperbolic tangent: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

        Properties:
        - Output range: (-1, 1)
        - Zero-centered (unlike sigmoid)
        - Still has vanishing gradient problem
        """
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh: tanh'(x) = 1 - tanh²(x)"""
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """
        Rectified Linear Unit: ReLU(x) = max(0, x)

        Properties:
        - Output range: [0, ∞)
        - Computationally efficient
        - No vanishing gradient for positive values
        - Problem: "Dying ReLU" - neurons can become inactive
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU: 1 if x > 0, else 0"""
        return (x > 0).astype(float)

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        Leaky ReLU: f(x) = x if x > 0, else alpha * x

        Addresses the dying ReLU problem by having small gradient for negative inputs.
        """
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU"""
        return np.where(x > 0, 1, alpha)

    @staticmethod
    def step(x: np.ndarray) -> np.ndarray:
        """
        Step function (Heaviside): f(x) = 1 if x >= 0, else 0

        Used in original perceptron but not differentiable.
        """
        return (x >= 0).astype(float)


# =============================================================================
# PART 2: LOSS FUNCTIONS
# =============================================================================

class LossFunctions:
    """Collection of loss functions and their derivatives."""

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error: MSE = (1/n) * Σ(y_true - y_pred)²

        Used for regression problems.
        """
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Derivative of MSE with respect to y_pred: -2/n * (y_true - y_pred)"""
        n = len(y_true)
        return -2/n * (y_true - y_pred)

    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray,
                             epsilon: float = 1e-15) -> float:
        """
        Binary Cross-Entropy: BCE = -1/n * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]

        Used for binary classification problems.
        Measures the performance of a classification model whose output is a probability.
        """
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def binary_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray,
                                        epsilon: float = 1e-15) -> np.ndarray:
        """Derivative of BCE: (ŷ - y) / (ŷ * (1 - ŷ))"""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))


# =============================================================================
# PART 3: PERCEPTRON IMPLEMENTATION
# =============================================================================

class Perceptron:
    """
    Single-layer Perceptron implementation.

    The perceptron is the simplest neural network - a single artificial neuron.

    Mathematical formulation:
    1. Weighted sum: z = Σ(w_i * x_i) + b = W·X + b
    2. Activation: a = f(z)
    3. Output: ŷ = a

    Learning process:
    1. Forward pass: compute prediction
    2. Compute loss
    3. Backward pass: compute gradients
    4. Update weights: w = w - lr * gradient
    """

    def __init__(self, input_size: int, activation: str = 'sigmoid',
                 learning_rate: float = 0.1, random_state: int = 42):
        """
        Initialize the perceptron.

        Args:
            input_size: Number of input features
            activation: Activation function ('sigmoid', 'tanh', 'relu', 'step')
            learning_rate: Learning rate for gradient descent
            random_state: Random seed for reproducibility
        """
        np.random.seed(random_state)

        # Initialize weights randomly (small values)
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate

        # Set activation function and its derivative
        self._set_activation(activation)

        # Training history
        self.loss_history = []
        self.accuracy_history = []

    def _set_activation(self, activation: str):
        """Set activation function and its derivative."""
        activations = {
            'sigmoid': (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
            'tanh': (ActivationFunctions.tanh, ActivationFunctions.tanh_derivative),
            'relu': (ActivationFunctions.relu, ActivationFunctions.relu_derivative),
            'step': (ActivationFunctions.step, lambda x: np.ones_like(x))
        }

        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")

        self.activation_name = activation
        self.activation, self.activation_derivative = activations[activation]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation.

        z = W·X + b (weighted sum)
        a = f(z) (activation)

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Activated output of shape (n_samples,)
        """
        # Linear transformation
        self.z = np.dot(X, self.weights) + self.bias
        # Activation
        self.a = self.activation(self.z)
        return self.a

    def backward(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Backward propagation (gradient computation).

        Using chain rule:
        dL/dw = dL/da * da/dz * dz/dw
        dL/db = dL/da * da/dz * dz/db

        For binary cross-entropy with sigmoid:
        dL/dw = (a - y) * x
        dL/db = (a - y)

        Args:
            X: Input data
            y: True labels
            y_pred: Predicted values

        Returns:
            Tuple of (weight gradients, bias gradient)
        """
        n_samples = len(y)

        # Error signal: (prediction - true) * activation_derivative
        # For sigmoid + BCE, this simplifies to (a - y)
        if self.activation_name == 'sigmoid':
            error = y_pred - y
        else:
            # General case: dL/da * da/dz
            dL_da = LossFunctions.mse_derivative(y, y_pred)
            da_dz = self.activation_derivative(self.z)
            error = dL_da * da_dz

        # Gradients
        dw = np.dot(X.T, error) / n_samples
        db = np.mean(error)

        return dw, db

    def update_weights(self, dw: np.ndarray, db: float):
        """
        Update weights using gradient descent.

        w = w - learning_rate * dw
        b = b - learning_rate * db
        """
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100,
            verbose: bool = True) -> 'Perceptron':
        """
        Train the perceptron.

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Labels of shape (n_samples,)
            epochs: Number of training iterations
            verbose: Whether to print progress

        Returns:
            self
        """
        self.loss_history = []
        self.accuracy_history = []

        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Compute loss
            loss = LossFunctions.binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            # Compute accuracy
            predictions = (y_pred >= 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            self.accuracy_history.append(accuracy)

            # Backward pass
            dw, db = self.backward(X, y, y_pred)

            # Update weights
            self.update_weights(dw, db)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

        return self

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make predictions."""
        probabilities = self.forward(X)
        return (probabilities >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        return self.forward(X)


# =============================================================================
# PART 4: LOGIC GATES IMPLEMENTATION
# =============================================================================

def create_logic_gate_data(gate: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training data for logic gates.

    Args:
        gate: 'AND', 'OR', 'NAND', 'NOR', 'XOR'

    Returns:
        Tuple of (X, y)
    """
    # Input combinations for 2-input gate
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    gates = {
        'AND': np.array([0, 0, 0, 1]),   # Both inputs must be 1
        'OR': np.array([0, 1, 1, 1]),    # At least one input must be 1
        'NAND': np.array([1, 1, 1, 0]),  # NOT AND
        'NOR': np.array([1, 0, 0, 0]),   # NOT OR
        'XOR': np.array([0, 1, 1, 0])    # Exclusive OR (not linearly separable!)
    }

    if gate.upper() not in gates:
        raise ValueError(f"Unknown gate: {gate}")

    return X, gates[gate.upper()]


def visualize_decision_boundary(perceptron: Perceptron, X: np.ndarray, y: np.ndarray,
                                title: str = "Decision Boundary"):
    """Visualize the decision boundary of the perceptron."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Decision boundary
    ax1 = axes[0]

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Get predictions for mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

    # Plot data points
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                         edgecolors='black', s=200, linewidths=2)

    ax1.set_xlabel('Input 1')
    ax1.set_ylabel('Input 2')
    ax1.set_title(f'{title}\nWeights: [{perceptron.weights[0]:.2f}, {perceptron.weights[1]:.2f}], Bias: {perceptron.bias:.2f}')

    # Plot 2: Loss history
    ax2 = axes[1]
    ax2.plot(perceptron.loss_history, 'b-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Accuracy history
    ax3 = axes[2]
    ax3.plot(perceptron.accuracy_history, 'g-', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Training Accuracy')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.1])

    plt.tight_layout()
    return fig


def demonstrate_activation_functions():
    """Visualize all activation functions and their derivatives."""
    x = np.linspace(-5, 5, 100)

    activations = [
        ('Sigmoid', ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
        ('Tanh', ActivationFunctions.tanh, ActivationFunctions.tanh_derivative),
        ('ReLU', ActivationFunctions.relu, ActivationFunctions.relu_derivative),
        ('Leaky ReLU', ActivationFunctions.leaky_relu, ActivationFunctions.leaky_relu_derivative)
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, (name, func, deriv) in enumerate(activations):
        # Function
        axes[0, i].plot(x, func(x), 'b-', linewidth=2)
        axes[0, i].set_title(f'{name}')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('f(x)')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].axhline(y=0, color='k', linewidth=0.5)
        axes[0, i].axvline(x=0, color='k', linewidth=0.5)

        # Derivative
        axes[1, i].plot(x, deriv(x), 'r-', linewidth=2)
        axes[1, i].set_title(f'{name} Derivative')
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel("f'(x)")
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].axhline(y=0, color='k', linewidth=0.5)
        axes[1, i].axvline(x=0, color='k', linewidth=0.5)

    plt.suptitle('Activation Functions and Their Derivatives', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# PART 5: MAIN EXECUTION
# =============================================================================

def main():
    """Main function demonstrating perceptron for logic gates."""
    print("=" * 70)
    print("DAY 15: NEURAL NETWORKS THEORY")
    print("Single-Layer Perceptron for Logic Gates")
    print("=" * 70)

    # 1. Demonstrate activation functions
    print("\n1. ACTIVATION FUNCTIONS")
    print("-" * 40)
    fig_activations = demonstrate_activation_functions()
    fig_activations.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
    print("   Saved: activation_functions.png")

    # 2. Train perceptron for AND gate
    print("\n2. AND GATE")
    print("-" * 40)
    X_and, y_and = create_logic_gate_data('AND')

    print("   Truth Table:")
    print("   X1  X2  |  Y")
    for i in range(len(X_and)):
        print(f"   {X_and[i, 0]}   {X_and[i, 1]}   |  {y_and[i]}")

    perceptron_and = Perceptron(input_size=2, activation='sigmoid', learning_rate=1.0)
    perceptron_and.fit(X_and, y_and, epochs=100, verbose=False)

    predictions_and = perceptron_and.predict(X_and)
    print(f"\n   Final Accuracy: {np.mean(predictions_and == y_and) * 100:.1f}%")
    print(f"   Weights: {perceptron_and.weights}")
    print(f"   Bias: {perceptron_and.bias:.4f}")
    print(f"   Predictions: {predictions_and}")

    fig_and = visualize_decision_boundary(perceptron_and, X_and, y_and, "AND Gate")
    fig_and.savefig('and_gate_perceptron.png', dpi=150, bbox_inches='tight')
    print("   Saved: and_gate_perceptron.png")

    # 3. Train perceptron for OR gate
    print("\n3. OR GATE")
    print("-" * 40)
    X_or, y_or = create_logic_gate_data('OR')

    print("   Truth Table:")
    print("   X1  X2  |  Y")
    for i in range(len(X_or)):
        print(f"   {X_or[i, 0]}   {X_or[i, 1]}   |  {y_or[i]}")

    perceptron_or = Perceptron(input_size=2, activation='sigmoid', learning_rate=1.0)
    perceptron_or.fit(X_or, y_or, epochs=100, verbose=False)

    predictions_or = perceptron_or.predict(X_or)
    print(f"\n   Final Accuracy: {np.mean(predictions_or == y_or) * 100:.1f}%")
    print(f"   Weights: {perceptron_or.weights}")
    print(f"   Bias: {perceptron_or.bias:.4f}")
    print(f"   Predictions: {predictions_or}")

    fig_or = visualize_decision_boundary(perceptron_or, X_or, y_or, "OR Gate")
    fig_or.savefig('or_gate_perceptron.png', dpi=150, bbox_inches='tight')
    print("   Saved: or_gate_perceptron.png")

    # 4. Demonstrate XOR limitation (not linearly separable)
    print("\n4. XOR GATE (Demonstration of Limitation)")
    print("-" * 40)
    X_xor, y_xor = create_logic_gate_data('XOR')

    print("   Truth Table:")
    print("   X1  X2  |  Y")
    for i in range(len(X_xor)):
        print(f"   {X_xor[i, 0]}   {X_xor[i, 1]}   |  {y_xor[i]}")

    perceptron_xor = Perceptron(input_size=2, activation='sigmoid', learning_rate=1.0)
    perceptron_xor.fit(X_xor, y_xor, epochs=1000, verbose=False)

    predictions_xor = perceptron_xor.predict(X_xor)
    print(f"\n   Final Accuracy: {np.mean(predictions_xor == y_xor) * 100:.1f}%")
    print(f"   Predictions: {predictions_xor}")
    print("\n   NOTE: XOR is NOT linearly separable!")
    print("   A single-layer perceptron CANNOT learn XOR.")
    print("   This was the key insight that led to multi-layer networks.")

    fig_xor = visualize_decision_boundary(perceptron_xor, X_xor, y_xor, "XOR Gate (Fails!)")
    fig_xor.savefig('xor_gate_perceptron.png', dpi=150, bbox_inches='tight')
    print("   Saved: xor_gate_perceptron.png")

    # 5. NAND and NOR gates
    print("\n5. NAND GATE")
    print("-" * 40)
    X_nand, y_nand = create_logic_gate_data('NAND')
    perceptron_nand = Perceptron(input_size=2, activation='sigmoid', learning_rate=1.0)
    perceptron_nand.fit(X_nand, y_nand, epochs=100, verbose=False)
    predictions_nand = perceptron_nand.predict(X_nand)
    print(f"   Final Accuracy: {np.mean(predictions_nand == y_nand) * 100:.1f}%")

    fig_nand = visualize_decision_boundary(perceptron_nand, X_nand, y_nand, "NAND Gate")
    fig_nand.savefig('nand_gate_perceptron.png', dpi=150, bbox_inches='tight')
    print("   Saved: nand_gate_perceptron.png")

    print("\n6. NOR GATE")
    print("-" * 40)
    X_nor, y_nor = create_logic_gate_data('NOR')
    perceptron_nor = Perceptron(input_size=2, activation='sigmoid', learning_rate=1.0)
    perceptron_nor.fit(X_nor, y_nor, epochs=100, verbose=False)
    predictions_nor = perceptron_nor.predict(X_nor)
    print(f"   Final Accuracy: {np.mean(predictions_nor == y_nor) * 100:.1f}%")

    fig_nor = visualize_decision_boundary(perceptron_nor, X_nor, y_nor, "NOR Gate")
    fig_nor.savefig('nor_gate_perceptron.png', dpi=150, bbox_inches='tight')
    print("   Saved: nor_gate_perceptron.png")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    Key Concepts Covered:

    1. PERCEPTRON: The simplest neural network unit
       - Weighted sum: z = Σ(w_i * x_i) + b
       - Activation: a = f(z)

    2. ACTIVATION FUNCTIONS:
       - Sigmoid: Output (0,1), good for probability
       - Tanh: Output (-1,1), zero-centered
       - ReLU: Output [0,∞), computationally efficient
       - Step: Binary output, original perceptron

    3. FORWARD PROPAGATION:
       - Input → Weighted Sum → Activation → Output

    4. BACKPROPAGATION:
       - Compute gradient of loss with respect to weights
       - Use chain rule: dL/dw = dL/da * da/dz * dz/dw

    5. GRADIENT DESCENT:
       - Update rule: w = w - learning_rate * gradient

    6. LOSS FUNCTIONS:
       - MSE: For regression
       - Binary Cross-Entropy: For binary classification

    7. LIMITATION:
       - Single-layer perceptron cannot solve XOR
       - Non-linearly separable problems require multi-layer networks
    """)

    plt.close('all')
    print("\nAll visualizations saved successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
