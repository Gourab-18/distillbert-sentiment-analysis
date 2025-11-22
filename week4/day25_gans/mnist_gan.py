"""
Day 25: Assignment - Generate Handwritten Digits using GAN
Complete implementation of GAN, DCGAN, and conditional GAN
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch torchvision")


# ==================== GAN THEORY ====================

class GANTheory:
    """
    GAN Theory and Concepts
    """

    @staticmethod
    def explain():
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║               GENERATIVE ADVERSARIAL NETWORKS (GANs)                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Two-player minimax game:                                             ║
║                                                                       ║
║    min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]            ║
║                                                                       ║
║  ┌─────────────────┐          ┌─────────────────┐                    ║
║  │   GENERATOR     │   fake   │  DISCRIMINATOR  │                    ║
║  │                 │ ───────> │                 │──> Real/Fake       ║
║  │  z (noise) ──>  │          │    <───────     │                    ║
║  │  Fake images    │          │   Real images   │                    ║
║  └─────────────────┘          └─────────────────┘                    ║
║                                                                       ║
║  Training Process:                                                    ║
║  1. Train D: Maximize log(D(real)) + log(1-D(G(z)))                  ║
║  2. Train G: Minimize log(1-D(G(z))) or Maximize log(D(G(z)))        ║
║                                                                       ║
║  Mode Collapse: Generator produces limited variety                    ║
║  Solutions: Feature matching, mini-batch discrimination,              ║
║             unrolled GANs, Wasserstein GAN                           ║
╚══════════════════════════════════════════════════════════════════════╝
        """)


# ==================== BASIC GAN ====================

if TORCH_AVAILABLE:
    class Generator(nn.Module):
        """
        Generator network for basic GAN
        Takes random noise and generates images
        """

        def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
            super(Generator, self).__init__()

            self.img_shape = img_shape
            self.img_size = int(np.prod(img_shape))

            self.model = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256),

                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(512),

                nn.Linear(512, 1024),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024),

                nn.Linear(1024, self.img_size),
                nn.Tanh()  # Output between -1 and 1
            )

        def forward(self, z):
            img = self.model(z)
            return img.view(img.size(0), *self.img_shape)


    class Discriminator(nn.Module):
        """
        Discriminator network for basic GAN
        Takes images and classifies as real/fake
        """

        def __init__(self, img_shape=(1, 28, 28)):
            super(Discriminator, self).__init__()

            self.img_size = int(np.prod(img_shape))

            self.model = nn.Sequential(
                nn.Linear(self.img_size, 512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),

                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),

                nn.Linear(256, 1),
                nn.Sigmoid()  # Output probability
            )

        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            validity = self.model(img_flat)
            return validity


    # ==================== DCGAN ====================

    class DCGANGenerator(nn.Module):
        """
        Deep Convolutional GAN Generator
        Uses transposed convolutions for upsampling
        """

        def __init__(self, latent_dim=100, channels=1):
            super(DCGANGenerator, self).__init__()

            self.init_size = 7  # Initial size before upsampling
            self.l1 = nn.Sequential(
                nn.Linear(latent_dim, 128 * self.init_size ** 2)
            )

            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),  # 7 -> 14

                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Upsample(scale_factor=2),  # 14 -> 28

                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(64, channels, 3, stride=1, padding=1),
                nn.Tanh()
            )

        def forward(self, z):
            out = self.l1(z)
            out = out.view(out.size(0), 128, self.init_size, self.init_size)
            img = self.conv_blocks(out)
            return img


    class DCGANDiscriminator(nn.Module):
        """
        Deep Convolutional GAN Discriminator
        Uses strided convolutions for downsampling
        """

        def __init__(self, channels=1):
            super(DCGANDiscriminator, self).__init__()

            def discriminator_block(in_filters, out_filters, bn=True):
                block = [
                    nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25)
                ]
                if bn:
                    block.append(nn.BatchNorm2d(out_filters))
                return block

            self.model = nn.Sequential(
                *discriminator_block(channels, 16, bn=False),  # 28 -> 14
                *discriminator_block(16, 32),  # 14 -> 7
                *discriminator_block(32, 64),  # 7 -> 4
                *discriminator_block(64, 128),  # 4 -> 2
            )

            self.adv_layer = nn.Sequential(
                nn.Linear(128 * 2 * 2, 1),
                nn.Sigmoid()
            )

        def forward(self, img):
            out = self.model(img)
            out = out.view(out.size(0), -1)
            validity = self.adv_layer(out)
            return validity


    # ==================== CONDITIONAL GAN ====================

    class ConditionalGenerator(nn.Module):
        """
        Conditional Generator - generates images conditioned on class labels
        """

        def __init__(self, latent_dim=100, n_classes=10, img_shape=(1, 28, 28)):
            super(ConditionalGenerator, self).__init__()

            self.img_shape = img_shape
            self.img_size = int(np.prod(img_shape))

            self.label_embedding = nn.Embedding(n_classes, n_classes)

            self.model = nn.Sequential(
                nn.Linear(latent_dim + n_classes, 256),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256),

                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(512),

                nn.Linear(512, 1024),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024),

                nn.Linear(1024, self.img_size),
                nn.Tanh()
            )

        def forward(self, z, labels):
            # Embed labels
            label_input = self.label_embedding(labels)
            # Concatenate noise and label
            gen_input = torch.cat((z, label_input), -1)
            img = self.model(gen_input)
            return img.view(img.size(0), *self.img_shape)


    class ConditionalDiscriminator(nn.Module):
        """
        Conditional Discriminator - classifies images given class labels
        """

        def __init__(self, n_classes=10, img_shape=(1, 28, 28)):
            super(ConditionalDiscriminator, self).__init__()

            self.img_shape = img_shape
            self.img_size = int(np.prod(img_shape))

            self.label_embedding = nn.Embedding(n_classes, n_classes)

            self.model = nn.Sequential(
                nn.Linear(self.img_size + n_classes, 512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),

                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),

                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        def forward(self, img, labels):
            img_flat = img.view(img.size(0), -1)
            label_input = self.label_embedding(labels)
            d_input = torch.cat((img_flat, label_input), -1)
            validity = self.model(d_input)
            return validity


# ==================== GAN TRAINER ====================

class GANTrainer:
    """
    Train GANs on MNIST dataset
    """

    def __init__(self, device=None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.latent_dim = 100

    def load_mnist(self, n_samples=60000):
        """Load MNIST data"""
        print("Loading MNIST data...")

        try:
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
            X = mnist.data[:n_samples].astype('float32')
            y = mnist.target[:n_samples].astype('int')

            # Normalize to [-1, 1] for tanh activation
            X = (X / 255.0) * 2 - 1

        except Exception as e:
            print(f"Could not load MNIST: {e}")
            print("Generating synthetic data...")
            np.random.seed(42)
            X = np.random.randn(n_samples, 784).astype('float32') * 0.5
            y = np.random.randint(0, 10, n_samples)

        X = X.reshape(-1, 1, 28, 28)

        print(f"Data shape: {X.shape}")
        print(f"Data range: [{X.min():.2f}, {X.max():.2f}]")

        return X, y

    def train_basic_gan(self, n_epochs=50, batch_size=64, sample_interval=500):
        """
        Train basic GAN
        """
        print("\n" + "=" * 60)
        print("TRAINING BASIC GAN")
        print("=" * 60)

        # Load data
        X, y = self.load_mnist()

        # Initialize models
        generator = Generator(latent_dim=self.latent_dim).to(self.device)
        discriminator = Discriminator().to(self.device)

        # Optimizers
        optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Loss
        adversarial_loss = nn.BCELoss()

        # Data loader
        dataset = TensorDataset(torch.FloatTensor(X))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training history
        g_losses = []
        d_losses = []

        print(f"\nGenerator params: {sum(p.numel() for p in generator.parameters()):,}")
        print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
        print(f"\nTraining for {n_epochs} epochs...")

        for epoch in range(n_epochs):
            for i, (imgs,) in enumerate(dataloader):
                batch_size_current = imgs.size(0)

                # Ground truths
                valid = torch.ones(batch_size_current, 1).to(self.device)
                fake = torch.zeros(batch_size_current, 1).to(self.device)

                real_imgs = imgs.to(self.device)

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                # Sample noise
                z = torch.randn(batch_size_current, self.latent_dim).to(self.device)

                # Generate images
                gen_imgs = generator(z)

                # Generator loss
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()

                # Real loss
                real_loss = adversarial_loss(discriminator(real_imgs), valid)

                # Fake loss
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

                # Total discriminator loss
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs} - "
                      f"D Loss: {d_loss.item():.4f} - "
                      f"G Loss: {g_loss.item():.4f}")

            # Save sample images
            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.save_samples(generator, epoch + 1, prefix="basic_gan")

        # Plot training curves
        self.plot_training_curves(g_losses, d_losses, "Basic GAN Training")

        # Final samples
        self.save_samples(generator, n_epochs, prefix="basic_gan_final", n=25)

        return generator, discriminator

    def train_dcgan(self, n_epochs=50, batch_size=64):
        """
        Train Deep Convolutional GAN
        """
        print("\n" + "=" * 60)
        print("TRAINING DCGAN")
        print("=" * 60)

        X, y = self.load_mnist()

        generator = DCGANGenerator(latent_dim=self.latent_dim).to(self.device)
        discriminator = DCGANDiscriminator().to(self.device)

        optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        adversarial_loss = nn.BCELoss()

        dataset = TensorDataset(torch.FloatTensor(X))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print(f"\nDCGAN Generator params: {sum(p.numel() for p in generator.parameters()):,}")
        print(f"DCGAN Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
        print(f"\nTraining for {n_epochs} epochs...")

        for epoch in range(n_epochs):
            for i, (imgs,) in enumerate(dataloader):
                batch_size_current = imgs.size(0)

                valid = torch.ones(batch_size_current, 1).to(self.device)
                fake = torch.zeros(batch_size_current, 1).to(self.device)

                real_imgs = imgs.to(self.device)

                # Train Generator
                optimizer_G.zero_grad()
                z = torch.randn(batch_size_current, self.latent_dim).to(self.device)
                gen_imgs = generator(z)
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                g_loss.backward()
                optimizer_G.step()

                # Train Discriminator
                optimizer_D.zero_grad()
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs} - "
                      f"D Loss: {d_loss.item():.4f} - "
                      f"G Loss: {g_loss.item():.4f}")

            if (epoch + 1) % 10 == 0:
                self.save_samples(generator, epoch + 1, prefix="dcgan")

        self.save_samples(generator, n_epochs, prefix="dcgan_final", n=25)

        return generator, discriminator

    def train_conditional_gan(self, n_epochs=50, batch_size=64):
        """
        Train Conditional GAN
        """
        print("\n" + "=" * 60)
        print("TRAINING CONDITIONAL GAN")
        print("=" * 60)

        X, y = self.load_mnist()

        generator = ConditionalGenerator(latent_dim=self.latent_dim).to(self.device)
        discriminator = ConditionalDiscriminator().to(self.device)

        optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        adversarial_loss = nn.BCELoss()

        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print(f"\nConditional GAN training for {n_epochs} epochs...")

        for epoch in range(n_epochs):
            for i, (imgs, labels) in enumerate(dataloader):
                batch_size_current = imgs.size(0)

                valid = torch.ones(batch_size_current, 1).to(self.device)
                fake = torch.zeros(batch_size_current, 1).to(self.device)

                real_imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # Train Generator
                optimizer_G.zero_grad()
                z = torch.randn(batch_size_current, self.latent_dim).to(self.device)
                gen_labels = torch.randint(0, 10, (batch_size_current,)).to(self.device)
                gen_imgs = generator(z, gen_labels)
                g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
                g_loss.backward()
                optimizer_G.step()

                # Train Discriminator
                optimizer_D.zero_grad()
                real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs} - "
                      f"D Loss: {d_loss.item():.4f} - "
                      f"G Loss: {g_loss.item():.4f}")

            if (epoch + 1) % 10 == 0:
                self.save_conditional_samples(generator, epoch + 1)

        self.save_conditional_samples(generator, n_epochs, prefix="cgan_final")

        return generator, discriminator

    def save_samples(self, generator, epoch, prefix="gan", n=16):
        """Save generated samples"""
        generator.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim).to(self.device)
            gen_imgs = generator(z).cpu().numpy()

        # Denormalize
        gen_imgs = (gen_imgs + 1) / 2

        rows = int(np.sqrt(n))
        fig, axes = plt.subplots(rows, rows, figsize=(8, 8))

        idx = 0
        for i in range(rows):
            for j in range(rows):
                if idx < n:
                    axes[i, j].imshow(gen_imgs[idx, 0], cmap='gray')
                    axes[i, j].axis('off')
                idx += 1

        plt.suptitle(f'{prefix} - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f'{prefix}_epoch_{epoch}.png', dpi=150)
        plt.close()
        print(f"Saved: {prefix}_epoch_{epoch}.png")

        generator.train()

    def save_conditional_samples(self, generator, epoch, prefix="cgan"):
        """Save conditional GAN samples - one row per digit"""
        generator.eval()
        with torch.no_grad():
            # Generate 10 samples for each digit (0-9)
            samples_per_class = 10
            imgs = []

            for digit in range(10):
                z = torch.randn(samples_per_class, self.latent_dim).to(self.device)
                labels = torch.full((samples_per_class,), digit, dtype=torch.long).to(self.device)
                gen_imgs = generator(z, labels).cpu().numpy()
                imgs.append(gen_imgs)

        imgs = np.concatenate(imgs, axis=0)
        imgs = (imgs + 1) / 2

        fig, axes = plt.subplots(10, 10, figsize=(10, 10))

        for i in range(10):
            for j in range(10):
                axes[i, j].imshow(imgs[i * 10 + j, 0], cmap='gray')
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_ylabel(str(i))

        plt.suptitle(f'{prefix} - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f'{prefix}_epoch_{epoch}.png', dpi=150)
        plt.close()
        print(f"Saved: {prefix}_epoch_{epoch}.png")

        generator.train()

    def plot_training_curves(self, g_losses, d_losses, title="GAN Training"):
        """Plot training curves"""
        plt.figure(figsize=(12, 5))

        # Smooth the curves
        window = 100
        g_smooth = np.convolve(g_losses, np.ones(window)/window, mode='valid')
        d_smooth = np.convolve(d_losses, np.ones(window)/window, mode='valid')

        plt.subplot(1, 2, 1)
        plt.plot(g_losses, alpha=0.3, label='Generator (raw)')
        plt.plot(range(window-1, len(g_losses)), g_smooth, label='Generator (smoothed)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Generator Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(d_losses, alpha=0.3, label='Discriminator (raw)')
        plt.plot(range(window-1, len(d_losses)), d_smooth, label='Discriminator (smoothed)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Discriminator Loss')
        plt.legend()

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig('gan_training_curves.png', dpi=150)
        plt.close()
        print("Saved: gan_training_curves.png")


# ==================== MODE COLLAPSE SOLUTIONS ====================

def demonstrate_mode_collapse_solutions():
    """
    Explain solutions to mode collapse
    """
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                  MODE COLLAPSE SOLUTIONS                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  1. FEATURE MATCHING                                                  ║
║     - Train G to match expected features of D's intermediate layer    ║
║     - Prevents G from over-optimizing for current D                   ║
║                                                                       ║
║  2. MINI-BATCH DISCRIMINATION                                         ║
║     - D looks at multiple samples together                            ║
║     - Can detect if G produces similar outputs                        ║
║                                                                       ║
║  3. UNROLLED GANs                                                     ║
║     - G is trained against future versions of D                       ║
║     - Prevents G from exploiting current D weaknesses                 ║
║                                                                       ║
║  4. WASSERSTEIN GAN (WGAN)                                            ║
║     - Uses Wasserstein distance instead of JS divergence              ║
║     - More stable training, better gradients                          ║
║     - Critic (D) is trained to Lipschitz continuity                   ║
║                                                                       ║
║  5. SPECTRAL NORMALIZATION                                            ║
║     - Normalize weights by spectral norm                              ║
║     - Controls Lipschitz constant of D                                ║
║                                                                       ║
║  6. PROGRESSIVE GROWING                                               ║
║     - Start with low resolution, gradually increase                   ║
║     - More stable for high-resolution generation                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


# ==================== MAIN ====================

def run_gan_demo():
    """Run complete GAN demonstration"""

    # Explain GAN theory
    GANTheory.explain()

    if not TORCH_AVAILABLE:
        print("\nPyTorch not available. Cannot run GAN training.")
        return

    trainer = GANTrainer()

    # Train Basic GAN
    basic_gen, basic_disc = trainer.train_basic_gan(n_epochs=30)

    # Train DCGAN
    dc_gen, dc_disc = trainer.train_dcgan(n_epochs=30)

    # Train Conditional GAN
    cond_gen, cond_disc = trainer.train_conditional_gan(n_epochs=30)

    # Explain mode collapse solutions
    demonstrate_mode_collapse_solutions()

    # Generate final showcase
    print("\n" + "=" * 60)
    print("FINAL GENERATION SHOWCASE")
    print("=" * 60)

    # Generate using all models
    with torch.no_grad():
        z = torch.randn(10, 100).to(trainer.device)

        basic_samples = basic_gen(z).cpu().numpy()
        dc_samples = dc_gen(z).cpu().numpy()

        labels = torch.arange(10).to(trainer.device)
        cond_samples = cond_gen(z, labels).cpu().numpy()

    # Comparison plot
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))

    for i in range(10):
        axes[0, i].imshow((basic_samples[i, 0] + 1) / 2, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Basic GAN')

        axes[1, i].imshow((dc_samples[i, 0] + 1) / 2, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('DCGAN')

        axes[2, i].imshow((cond_samples[i, 0] + 1) / 2, cmap='gray')
        axes[2, i].axis('off')
        axes[2, i].set_xlabel(str(i))
        if i == 0:
            axes[2, i].set_ylabel('Conditional')

    plt.suptitle('GAN Comparison: Basic vs DCGAN vs Conditional')
    plt.tight_layout()
    plt.savefig('gan_comparison.png', dpi=150)
    plt.close()
    print("Saved: gan_comparison.png")

    return basic_gen, dc_gen, cond_gen


if __name__ == "__main__":
    models = run_gan_demo()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    This module demonstrated:
    1. Basic GAN architecture (Generator + Discriminator)
    2. Deep Convolutional GAN (DCGAN) with conv layers
    3. Conditional GAN for class-specific generation
    4. Mode collapse problem and solutions

    Output files:
    - basic_gan_epoch_*.png: Basic GAN samples
    - dcgan_epoch_*.png: DCGAN samples
    - cgan_epoch_*.png: Conditional GAN samples (digits 0-9)
    - gan_training_curves.png: Training loss curves
    - gan_comparison.png: Final comparison of all models
    """)
