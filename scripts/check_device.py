"""
Device Check Script

This script checks your hardware setup and estimates training time.
"""

import torch
import sys

print("=" * 80)
print("DEVICE & TRAINING TIME CHECK")
print("=" * 80)

# PyTorch version
print(f"\nPyTorch version: {torch.__version__}")

# Check CUDA (NVIDIA GPU)
cuda_available = torch.cuda.is_available()
print(f"\n🔍 CUDA (NVIDIA GPU) available: {cuda_available}")

if cuda_available:
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   CUDA Version: {torch.version.cuda}")
    print("\n✅ RECOMMENDED DEVICE: GPU")
    print("   Estimated training time: 30-60 minutes")
    print("   Recommendation: Use full training (20K samples, 3 epochs)")
else:
    print("   No NVIDIA GPU detected")

# Check MPS (Apple Silicon)
mps_available = torch.backends.mps.is_available()
print(f"\n🔍 MPS (Apple Silicon) available: {mps_available}")

if mps_available:
    print("   Apple Silicon GPU detected (M1/M2/M3)")
    print("\n✅ RECOMMENDED DEVICE: MPS")
    print("   Estimated training time: 1.5-2.5 hours")
    print("   Recommendation: Use full training OR test config for faster results")
else:
    print("   Not running on Apple Silicon")

# CPU info
print(f"\n🔍 CPU available: True (always available)")
import platform
print(f"   Platform: {platform.system()} {platform.release()}")
print(f"   Processor: {platform.processor()}")

# Final recommendation
print("\n" + "=" * 80)
print("FINAL RECOMMENDATION")
print("=" * 80)

if cuda_available:
    device = "cuda"
    time_estimate = "30-60 minutes"
    config_rec = "Full config (configs/distilbert_config.yaml)"
elif mps_available:
    device = "mps (Apple Silicon)"
    time_estimate = "1.5-2.5 hours"
    config_rec = "Full config OR Quick test (configs/distilbert_config_test.yaml)"
else:
    device = "cpu"
    time_estimate = "3-6 hours"
    config_rec = "Quick test config (configs/distilbert_config_test.yaml)"

print(f"\n🎯 Best device: {device}")
print(f"⏱️  Estimated time (full training): {time_estimate}")
print(f"📝 Recommended config: {config_rec}")

# Test tensor creation
print("\n" + "=" * 80)
print("DEVICE TEST")
print("=" * 80)

try:
    if cuda_available:
        device_test = torch.device("cuda:0")
        x = torch.randn(100, 100).to(device_test)
        print("✅ GPU tensor creation successful!")
    elif mps_available:
        device_test = torch.device("mps")
        x = torch.randn(100, 100).to(device_test)
        print("✅ MPS tensor creation successful!")
    else:
        device_test = torch.device("cpu")
        x = torch.randn(100, 100).to(device_test)
        print("✅ CPU tensor creation successful!")

    # Simple operation
    y = x @ x.T
    print(f"✅ Matrix multiplication successful on {device_test}")
    print(f"   Tensor shape: {y.shape}")

except Exception as e:
    print(f"❌ Error: {e}")
    print("   There might be an issue with your PyTorch installation")

# Command recommendations
print("\n" + "=" * 80)
print("QUICK START COMMANDS")
print("=" * 80)

if cuda_available or mps_available:
    print("\n1. Quick test (5-15 minutes):")
    print("   python train_distilbert.py --config ../configs/distilbert_config_test.yaml")
    print("\n2. Full training (recommended):")
    print("   python train_distilbert.py --config ../configs/distilbert_config.yaml")
else:
    print("\n⚠️  CPU training will be slow! Recommendations:")
    print("\n1. Quick test FIRST (45-90 minutes):")
    print("   python train_distilbert.py --config ../configs/distilbert_config_test.yaml")
    print("\n2. If test works, consider:")
    print("   - Using a machine with GPU")
    print("   - Running overnight (3-6 hours)")
    print("   - Using cloud GPU (Colab, Kaggle, etc.)")

print("\n" + "=" * 80)
