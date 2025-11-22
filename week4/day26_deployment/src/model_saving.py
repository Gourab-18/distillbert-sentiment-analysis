"""
Day 26: Model Deployment - Save and Load Models
Using pickle, joblib, and framework-specific methods
"""

import os
import pickle
import joblib
import json
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class ModelSaver:
    """
    Comprehensive model saving and loading utilities
    """

    @staticmethod
    def save_pickle(model, filepath):
        """
        Save model using pickle

        Pros: Simple, works with any Python object
        Cons: Security risk with untrusted files, not compression
        """
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {filepath} (pickle)")

    @staticmethod
    def load_pickle(filepath):
        """Load model from pickle file"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath} (pickle)")
        return model

    @staticmethod
    def save_joblib(model, filepath, compress=3):
        """
        Save model using joblib

        Pros: Efficient for numpy arrays, supports compression
        Cons: Same security concerns as pickle
        """
        joblib.dump(model, filepath, compress=compress)
        print(f"Model saved to {filepath} (joblib, compress={compress})")

    @staticmethod
    def load_joblib(filepath):
        """Load model from joblib file"""
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath} (joblib)")
        return model

    @staticmethod
    def save_sklearn_pipeline(pipeline, filepath, metadata=None):
        """
        Save sklearn pipeline with metadata

        Args:
            pipeline: sklearn Pipeline object
            filepath: Path to save
            metadata: Optional dict with model info
        """
        save_dict = {
            'pipeline': pipeline,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'sklearn_version': __import__('sklearn').__version__
        }
        joblib.dump(save_dict, filepath)
        print(f"Pipeline saved with metadata to {filepath}")

    @staticmethod
    def load_sklearn_pipeline(filepath):
        """Load sklearn pipeline with metadata"""
        save_dict = joblib.load(filepath)
        print(f"Pipeline loaded from {filepath}")
        print(f"  Saved at: {save_dict.get('timestamp', 'unknown')}")
        print(f"  Metadata: {save_dict.get('metadata', {})}")
        return save_dict['pipeline'], save_dict.get('metadata', {})


class PyTorchModelSaver:
    """
    PyTorch model saving utilities
    """

    @staticmethod
    def save_state_dict(model, filepath, optimizer=None, metadata=None):
        """
        Save PyTorch model state dict (recommended method)

        Args:
            model: PyTorch model
            filepath: Path to save
            optimizer: Optional optimizer to save
            metadata: Optional metadata dict
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'timestamp': datetime.now().isoformat()
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if metadata:
            checkpoint['metadata'] = metadata

        torch.save(checkpoint, filepath)
        print(f"PyTorch model saved to {filepath}")

    @staticmethod
    def load_state_dict(model_class, filepath, device='cpu'):
        """
        Load PyTorch model from state dict

        Args:
            model_class: Model class to instantiate
            filepath: Path to checkpoint
            device: Device to load model on
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        checkpoint = torch.load(filepath, map_location=device)

        # Instantiate model
        model = model_class()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        print(f"PyTorch model loaded from {filepath}")
        print(f"  Model class: {checkpoint.get('model_class', 'unknown')}")
        print(f"  Saved at: {checkpoint.get('timestamp', 'unknown')}")

        return model, checkpoint.get('metadata', {})

    @staticmethod
    def save_entire_model(model, filepath):
        """
        Save entire PyTorch model (includes architecture)
        Not recommended for production - use state_dict instead
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        torch.save(model, filepath)
        print(f"Entire PyTorch model saved to {filepath}")

    @staticmethod
    def save_torchscript(model, example_input, filepath):
        """
        Save model as TorchScript for deployment

        TorchScript advantages:
        - Can be loaded without Python
        - Optimized for inference
        - Works with C++/mobile
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        model.eval()
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(filepath)
        print(f"TorchScript model saved to {filepath}")

    @staticmethod
    def save_onnx(model, example_input, filepath, input_names=None, output_names=None):
        """
        Export PyTorch model to ONNX format

        ONNX advantages:
        - Framework agnostic
        - Can run on various inference engines
        - Wide hardware support
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        model.eval()
        torch.onnx.export(
            model,
            example_input,
            filepath,
            input_names=input_names or ['input'],
            output_names=output_names or ['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"ONNX model saved to {filepath}")


class TensorFlowModelSaver:
    """
    TensorFlow/Keras model saving utilities
    """

    @staticmethod
    def save_h5(model, filepath):
        """
        Save Keras model in HDF5 format

        Includes: architecture, weights, optimizer state
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")

        model.save(filepath)
        print(f"Keras model saved to {filepath} (HDF5)")

    @staticmethod
    def load_h5(filepath):
        """Load Keras model from HDF5 file"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")

        model = keras.models.load_model(filepath)
        print(f"Keras model loaded from {filepath}")
        return model

    @staticmethod
    def save_savedmodel(model, dirpath):
        """
        Save in TensorFlow SavedModel format (recommended)

        SavedModel advantages:
        - Complete serialization
        - Works with TensorFlow Serving
        - Language agnostic (can load in C++, Java, etc.)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")

        model.save(dirpath)
        print(f"SavedModel saved to {dirpath}")

    @staticmethod
    def load_savedmodel(dirpath):
        """Load TensorFlow SavedModel"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")

        model = keras.models.load_model(dirpath)
        print(f"SavedModel loaded from {dirpath}")
        return model

    @staticmethod
    def save_weights_only(model, filepath):
        """Save only model weights"""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")

        model.save_weights(filepath)
        print(f"Model weights saved to {filepath}")

    @staticmethod
    def convert_to_tflite(model, filepath, quantize=False):
        """
        Convert to TensorFlow Lite for mobile/edge deployment

        Args:
            model: Keras model
            filepath: Output path
            quantize: Whether to apply quantization
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")

        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        with open(filepath, 'wb') as f:
            f.write(tflite_model)

        print(f"TFLite model saved to {filepath}")


class ModelVersioning:
    """
    Simple model versioning system
    """

    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.registry_file = os.path.join(model_dir, 'model_registry.json')

    def _load_registry(self):
        """Load model registry"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {'models': {}}

    def _save_registry(self, registry):
        """Save model registry"""
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2)

    def register_model(self, model_name, model, metrics=None, tags=None):
        """
        Register a new model version

        Args:
            model_name: Name of the model
            model: Model object to save
            metrics: Dict of model metrics
            tags: List of tags
        """
        registry = self._load_registry()

        if model_name not in registry['models']:
            registry['models'][model_name] = {'versions': []}

        # Determine version number
        versions = registry['models'][model_name]['versions']
        version = len(versions) + 1

        # Save model
        filename = f"{model_name}_v{version}.joblib"
        filepath = os.path.join(self.model_dir, filename)
        joblib.dump(model, filepath)

        # Register version
        version_info = {
            'version': version,
            'filepath': filepath,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
            'tags': tags or []
        }
        versions.append(version_info)

        self._save_registry(registry)

        print(f"Registered {model_name} v{version}")
        print(f"  Path: {filepath}")
        print(f"  Metrics: {metrics}")

        return version

    def get_model(self, model_name, version='latest'):
        """
        Get a specific model version

        Args:
            model_name: Name of the model
            version: Version number or 'latest'
        """
        registry = self._load_registry()

        if model_name not in registry['models']:
            raise ValueError(f"Model {model_name} not found")

        versions = registry['models'][model_name]['versions']

        if version == 'latest':
            version_info = versions[-1]
        else:
            version_info = next(
                (v for v in versions if v['version'] == version),
                None
            )

        if not version_info:
            raise ValueError(f"Version {version} not found for {model_name}")

        model = joblib.load(version_info['filepath'])
        return model, version_info

    def list_models(self):
        """List all registered models"""
        registry = self._load_registry()

        print("\n=== Model Registry ===")
        for name, info in registry['models'].items():
            print(f"\n{name}:")
            for v in info['versions']:
                print(f"  v{v['version']}: {v['timestamp']}")
                if v.get('metrics'):
                    for metric, value in v['metrics'].items():
                        print(f"    {metric}: {value}")


def demonstrate_model_saving():
    """Demonstrate all model saving methods"""

    print("=" * 60)
    print("MODEL SAVING AND LOADING DEMONSTRATION")
    print("=" * 60)

    # Create output directory
    os.makedirs('saved_models', exist_ok=True)

    # Create sample data and model
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train a sklearn pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    print(f"\nTrained sklearn pipeline with accuracy: {accuracy:.4f}")

    # === Pickle ===
    print("\n" + "-" * 40)
    print("1. PICKLE")
    print("-" * 40)
    ModelSaver.save_pickle(pipeline, 'saved_models/model_pickle.pkl')
    loaded = ModelSaver.load_pickle('saved_models/model_pickle.pkl')
    print(f"   Verification accuracy: {loaded.score(X_test, y_test):.4f}")

    # === Joblib ===
    print("\n" + "-" * 40)
    print("2. JOBLIB")
    print("-" * 40)
    ModelSaver.save_joblib(pipeline, 'saved_models/model_joblib.joblib')
    loaded = ModelSaver.load_joblib('saved_models/model_joblib.joblib')
    print(f"   Verification accuracy: {loaded.score(X_test, y_test):.4f}")

    # === Pipeline with metadata ===
    print("\n" + "-" * 40)
    print("3. PIPELINE WITH METADATA")
    print("-" * 40)
    metadata = {
        'accuracy': accuracy,
        'features': 20,
        'training_samples': len(X_train)
    }
    ModelSaver.save_sklearn_pipeline(pipeline, 'saved_models/pipeline_meta.joblib', metadata)
    loaded, meta = ModelSaver.load_sklearn_pipeline('saved_models/pipeline_meta.joblib')

    # === Model Versioning ===
    print("\n" + "-" * 40)
    print("4. MODEL VERSIONING")
    print("-" * 40)
    versioner = ModelVersioning('saved_models/versioned')

    # Register multiple versions
    versioner.register_model('classifier', pipeline, {'accuracy': 0.85}, ['baseline'])

    # Train improved model
    pipeline2 = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    pipeline2.fit(X_train, y_train)
    accuracy2 = pipeline2.score(X_test, y_test)

    versioner.register_model('classifier', pipeline2, {'accuracy': accuracy2}, ['improved'])
    versioner.list_models()

    # === PyTorch ===
    if TORCH_AVAILABLE:
        print("\n" + "-" * 40)
        print("5. PYTORCH MODEL SAVING")
        print("-" * 40)

        # Simple PyTorch model
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(20, 64)
                self.fc2 = nn.Linear(64, 2)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        model = SimpleNet()

        # Save state dict
        PyTorchModelSaver.save_state_dict(
            model,
            'saved_models/pytorch_model.pt',
            metadata={'architecture': 'SimpleNet'}
        )

        # Save TorchScript
        example_input = torch.randn(1, 20)
        PyTorchModelSaver.save_torchscript(
            model,
            example_input,
            'saved_models/pytorch_model_scripted.pt'
        )

        # Save ONNX
        PyTorchModelSaver.save_onnx(
            model,
            example_input,
            'saved_models/pytorch_model.onnx'
        )

    print("\n" + "=" * 60)
    print("Files saved in 'saved_models/' directory")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_model_saving()
