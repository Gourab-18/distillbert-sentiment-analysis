"""
Day 24: Assignment - Autoencoder for Image Compression and Denoising
Complete implementation with PCA comparison, denoising autoencoder, and VAE
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ==================== PCA DIMENSIONALITY REDUCTION ====================

class PCADimensionalityReduction:
    """
    PCA for dimensionality reduction and comparison with autoencoders
    """

    def __init__(self, n_components=32):
        """
        Initialize PCA

        Args:
            n_components: Number of principal components to keep
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()

    def fit_transform(self, X):
        """Fit PCA and transform data"""
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        return X_pca

    def inverse_transform(self, X_pca):
        """Reconstruct from PCA components"""
        X_reconstructed = self.pca.inverse_transform(X_pca)
        X_original_scale = self.scaler.inverse_transform(X_reconstructed)
        return X_original_scale

    def get_reconstruction_error(self, X):
        """Calculate reconstruction error"""
        X_pca = self.fit_transform(X)
        X_reconstructed = self.inverse_transform(X_pca)
        mse = np.mean((X - X_reconstructed) ** 2)
        return mse

    def explained_variance_ratio(self):
        """Get explained variance ratio"""
        return self.pca.explained_variance_ratio_

    @staticmethod
    def plot_variance_explained(X, max_components=100):
        """Plot cumulative variance explained"""
        pca_full = PCA(n_components=min(max_components, X.shape[1]))
        pca_full.fit(X)

        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA - Explained Variance vs Components')
        plt.grid(True)
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        plt.legend()
        plt.tight_layout()
        plt.savefig('pca_variance_explained.png', dpi=150)
        plt.close()
        print("Saved: pca_variance_explained.png")

        # Find components for 95% variance
        n_95 = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"Components needed for 95% variance: {n_95}")

        return cumulative_variance


# ==================== AUTOENCODER (PyTorch) ====================

if TORCH_AVAILABLE:
    class Autoencoder(nn.Module):
        """
        Basic Autoencoder architecture in PyTorch
        """

        def __init__(self, input_dim, encoding_dim=32):
            """
            Initialize Autoencoder

            Args:
                input_dim: Input dimension (e.g., 784 for 28x28 images)
                encoding_dim: Dimension of the encoding (latent space)
            """
            super(Autoencoder, self).__init__()

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, encoding_dim),
                nn.ReLU()
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim),
                nn.Sigmoid()  # Output between 0 and 1
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            return self.decoder(z)


    class DenoisingAutoencoder(nn.Module):
        """
        Denoising Autoencoder - learns to remove noise from inputs
        """

        def __init__(self, input_dim, encoding_dim=32, dropout_rate=0.3):
            super(DenoisingAutoencoder, self).__init__()

            # Encoder with dropout for denoising
            self.encoder = nn.Sequential(
                nn.Dropout(dropout_rate),  # Add noise during training
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, encoding_dim),
                nn.ReLU()
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim),
                nn.Sigmoid()
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded


    class ConvolutionalAutoencoder(nn.Module):
        """
        Convolutional Autoencoder for images
        """

        def __init__(self):
            super(ConvolutionalAutoencoder, self).__init__()

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=7),  # 7x7 -> 1x1
                nn.ReLU()
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=7),  # 1x1 -> 7x7
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
                nn.Sigmoid()
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded


    class VariationalAutoencoder(nn.Module):
        """
        Variational Autoencoder (VAE)
        """

        def __init__(self, input_dim, latent_dim=20):
            super(VariationalAutoencoder, self).__init__()

            self.latent_dim = latent_dim

            # Encoder
            self.encoder_layers = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU()
            )

            # Latent space parameters
            self.fc_mu = nn.Linear(128, latent_dim)
            self.fc_logvar = nn.Linear(128, latent_dim)

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim),
                nn.Sigmoid()
            )

        def encode(self, x):
            h = self.encoder_layers(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

        def reparameterize(self, mu, logvar):
            """Reparameterization trick"""
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

        @staticmethod
        def loss_function(recon_x, x, mu, logvar):
            """VAE loss = reconstruction loss + KL divergence"""
            # Reconstruction loss
            BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

            # KL divergence
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            return BCE + KLD


# ==================== TRAINING FUNCTIONS ====================

class AutoencoderTrainer:
    """
    Train autoencoders on image data
    """

    def __init__(self, device=None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def load_mnist_data(self, n_samples=10000):
        """Load and preprocess MNIST data"""
        print("Loading MNIST data...")

        try:
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
            X = mnist.data[:n_samples].astype('float32') / 255.0
            y = mnist.target[:n_samples].astype('int')
        except Exception as e:
            print(f"Could not load MNIST: {e}")
            print("Generating synthetic data instead...")
            np.random.seed(42)
            X = np.random.rand(n_samples, 784).astype('float32')
            y = np.random.randint(0, 10, n_samples)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def train_autoencoder(self, model, X_train, X_test,
                          epochs=20, batch_size=128, learning_rate=1e-3):
        """
        Train a basic autoencoder

        Args:
            model: Autoencoder model
            X_train: Training data
            X_test: Test data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        model.to(self.device)

        # Create data loaders
        train_tensor = torch.FloatTensor(X_train)
        test_tensor = torch.FloatTensor(X_test)

        train_loader = DataLoader(
            TensorDataset(train_tensor, train_tensor),
            batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(test_tensor, test_tensor),
            batch_size=batch_size, shuffle=False
        )

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        train_losses = []
        test_losses = []

        print(f"\nTraining for {epochs} epochs...")

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_x, _ in test_loader:
                    batch_x = batch_x.to(self.device)
                    output = model(batch_x)
                    test_loss += criterion(output, batch_x).item()

            train_losses.append(train_loss / len(train_loader))
            test_losses.append(test_loss / len(test_loader))

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {train_losses[-1]:.6f} - "
                      f"Test Loss: {test_losses[-1]:.6f}")

        return train_losses, test_losses

    def train_vae(self, vae, X_train, X_test,
                  epochs=20, batch_size=128, learning_rate=1e-3):
        """Train a Variational Autoencoder"""
        vae.to(self.device)

        train_tensor = torch.FloatTensor(X_train)
        test_tensor = torch.FloatTensor(X_test)

        train_loader = DataLoader(
            TensorDataset(train_tensor, train_tensor),
            batch_size=batch_size, shuffle=True
        )

        optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

        train_losses = []

        print(f"\nTraining VAE for {epochs} epochs...")

        for epoch in range(epochs):
            vae.train()
            train_loss = 0

            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()
                recon_x, mu, logvar = vae(batch_x)
                loss = VariationalAutoencoder.loss_function(recon_x, batch_x, mu, logvar)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_losses.append(train_loss / len(train_loader))

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_losses[-1]:.2f}")

        return train_losses


# ==================== VISUALIZATION ====================

def visualize_reconstructions(original, reconstructed, n=10, title="Reconstructions"):
    """Visualize original vs reconstructed images"""
    fig, axes = plt.subplots(2, n, figsize=(15, 3))

    for i in range(n):
        # Original
        axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')

        # Reconstructed
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()
    print(f"Saved: {title.lower().replace(' ', '_')}.png")


def visualize_denoising(original, noisy, denoised, n=10):
    """Visualize denoising results"""
    fig, axes = plt.subplots(3, n, figsize=(15, 5))

    for i in range(n):
        axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')

        axes[1, i].imshow(noisy[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Noisy')

        axes[2, i].imshow(denoised[i].reshape(28, 28), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Denoised')

    plt.tight_layout()
    plt.savefig('denoising_results.png', dpi=150)
    plt.close()
    print("Saved: denoising_results.png")


def visualize_latent_space(encoder, X, y, device, n_samples=1000):
    """Visualize 2D latent space (requires 2D encoding)"""
    from sklearn.manifold import TSNE

    encoder.eval()
    X_tensor = torch.FloatTensor(X[:n_samples]).to(device)

    with torch.no_grad():
        encoded = encoder(X_tensor).cpu().numpy()

    # Apply t-SNE if encoding dim > 2
    if encoded.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        encoded_2d = tsne.fit_transform(encoded)
    else:
        encoded_2d = encoded

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(encoded_2d[:, 0], encoded_2d[:, 1],
                          c=y[:n_samples], cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Digit')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Latent Space Visualization')
    plt.tight_layout()
    plt.savefig('latent_space.png', dpi=150)
    plt.close()
    print("Saved: latent_space.png")


def plot_training_curves(train_losses, test_losses, title="Training Curves"):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.close()
    print("Saved: training_curves.png")


# ==================== MAIN ====================

def run_autoencoder_demo():
    """Run complete autoencoder demonstration"""

    print("=" * 60)
    print("AUTOENCODER FOR IMAGE COMPRESSION AND DENOISING")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("\nPyTorch not available. Running PCA-only demo.")

    # Initialize trainer
    trainer = AutoencoderTrainer() if TORCH_AVAILABLE else None

    # Load data
    if trainer:
        X_train, X_test, y_train, y_test = trainer.load_mnist_data(n_samples=10000)
    else:
        print("Generating synthetic data for PCA demo...")
        np.random.seed(42)
        X_train = np.random.rand(8000, 784).astype('float32')
        X_test = np.random.rand(2000, 784).astype('float32')
        y_train = np.random.randint(0, 10, 8000)
        y_test = np.random.randint(0, 10, 2000)

    # ==================== PCA ====================
    print("\n" + "=" * 60)
    print("1. PCA DIMENSIONALITY REDUCTION")
    print("=" * 60)

    # Analyze variance
    PCADimensionalityReduction.plot_variance_explained(X_train)

    # Compare different compression levels
    for n_comp in [16, 32, 64, 128]:
        pca = PCADimensionalityReduction(n_components=n_comp)
        mse = pca.get_reconstruction_error(X_train)
        print(f"PCA with {n_comp} components - MSE: {mse:.6f}")

    # Visualize PCA reconstruction
    pca = PCADimensionalityReduction(n_components=32)
    X_pca = pca.fit_transform(X_test)
    X_recon_pca = pca.inverse_transform(X_pca)
    X_recon_pca = np.clip(X_recon_pca, 0, 1)
    visualize_reconstructions(X_test, X_recon_pca, title="PCA Reconstructions (32 components)")

    if not TORCH_AVAILABLE:
        print("\nPyTorch not available. Skipping neural autoencoder demos.")
        return

    # ==================== BASIC AUTOENCODER ====================
    print("\n" + "=" * 60)
    print("2. BASIC AUTOENCODER")
    print("=" * 60)

    ae = Autoencoder(input_dim=784, encoding_dim=32)
    print(f"Model parameters: {sum(p.numel() for p in ae.parameters()):,}")

    train_losses, test_losses = trainer.train_autoencoder(
        ae, X_train, X_test, epochs=20, batch_size=128
    )
    plot_training_curves(train_losses, test_losses, "Autoencoder Training")

    # Reconstruct test images
    ae.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(trainer.device)
        X_recon_ae = ae(X_test_tensor).cpu().numpy()

    visualize_reconstructions(X_test, X_recon_ae, title="Autoencoder Reconstructions")

    # ==================== DENOISING AUTOENCODER ====================
    print("\n" + "=" * 60)
    print("3. DENOISING AUTOENCODER")
    print("=" * 60)

    # Add noise to data
    noise_factor = 0.3
    X_train_noisy = X_train + noise_factor * np.random.randn(*X_train.shape).astype('float32')
    X_test_noisy = X_test + noise_factor * np.random.randn(*X_test.shape).astype('float32')
    X_train_noisy = np.clip(X_train_noisy, 0, 1)
    X_test_noisy = np.clip(X_test_noisy, 0, 1)

    dae = DenoisingAutoencoder(input_dim=784, encoding_dim=32)

    # Train on noisy data to reconstruct clean data
    train_tensor = torch.FloatTensor(X_train_noisy)
    target_tensor = torch.FloatTensor(X_train)
    train_loader = DataLoader(
        TensorDataset(train_tensor, target_tensor),
        batch_size=128, shuffle=True
    )

    dae.to(trainer.device)
    optimizer = optim.Adam(dae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("Training denoising autoencoder...")
    for epoch in range(20):
        dae.train()
        for noisy_batch, clean_batch in train_loader:
            noisy_batch = noisy_batch.to(trainer.device)
            clean_batch = clean_batch.to(trainer.device)

            optimizer.zero_grad()
            output = dae(noisy_batch)
            loss = criterion(output, clean_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/20 - Loss: {loss.item():.6f}")

    # Denoise test images
    dae.eval()
    with torch.no_grad():
        X_test_noisy_tensor = torch.FloatTensor(X_test_noisy).to(trainer.device)
        X_denoised = dae(X_test_noisy_tensor).cpu().numpy()

    visualize_denoising(X_test, X_test_noisy, X_denoised)

    # ==================== VAE ====================
    print("\n" + "=" * 60)
    print("4. VARIATIONAL AUTOENCODER (VAE)")
    print("=" * 60)

    vae = VariationalAutoencoder(input_dim=784, latent_dim=20)
    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")

    vae_losses = trainer.train_vae(vae, X_train, X_test, epochs=20)

    # Generate new samples
    vae.eval()
    with torch.no_grad():
        # Sample from latent space
        z = torch.randn(10, 20).to(trainer.device)
        generated = vae.decode(z).cpu().numpy()

    # Visualize generated samples
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i, ax in enumerate(axes):
        ax.imshow(generated[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    plt.suptitle('VAE Generated Samples')
    plt.tight_layout()
    plt.savefig('vae_generated.png', dpi=150)
    plt.close()
    print("Saved: vae_generated.png")

    # Visualize latent space
    visualize_latent_space(vae.encoder_layers, X_test, y_test, trainer.device)

    # ==================== COMPARISON ====================
    print("\n" + "=" * 60)
    print("5. COMPRESSION COMPARISON")
    print("=" * 60)

    # Calculate reconstruction errors
    ae.eval()
    vae.eval()

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(trainer.device)

        # Autoencoder
        ae_recon = ae(X_test_tensor).cpu().numpy()
        ae_mse = np.mean((X_test - ae_recon) ** 2)

        # VAE
        vae_recon, _, _ = vae(X_test_tensor)
        vae_recon = vae_recon.cpu().numpy()
        vae_mse = np.mean((X_test - vae_recon) ** 2)

    # PCA
    pca_mse = pca.get_reconstruction_error(X_test)

    print("\nReconstruction Error Comparison (32-dim latent space):")
    print(f"  PCA:         MSE = {pca_mse:.6f}")
    print(f"  Autoencoder: MSE = {ae_mse:.6f}")
    print(f"  VAE:         MSE = {vae_mse:.6f}")

    # Compression ratio
    original_size = 784  # 28x28
    compressed_size = 32
    compression_ratio = original_size / compressed_size
    print(f"\nCompression ratio: {compression_ratio:.1f}x ({original_size} -> {compressed_size})")

    return ae, vae, pca


if __name__ == "__main__":
    models = run_autoencoder_demo()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    This module demonstrated:
    1. PCA for dimensionality reduction
    2. Basic autoencoder for image compression
    3. Denoising autoencoder for noise removal
    4. Variational Autoencoder (VAE) for generative modeling

    Key concepts:
    - Encoder-decoder architecture
    - Latent space representation
    - Reconstruction loss minimization
    - VAE: Reparameterization trick and KL divergence

    Output files:
    - pca_variance_explained.png
    - pca_reconstructions.png
    - autoencoder_reconstructions.png
    - denoising_results.png
    - vae_generated.png
    - latent_space.png
    - training_curves.png
    """)
