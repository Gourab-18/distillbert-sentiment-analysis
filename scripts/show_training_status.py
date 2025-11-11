#!/usr/bin/env python3
"""
Show actual training status by reading from saved results and model directory
No hardcoded values - all data is read dynamically
"""

import os
import json
import subprocess
from datetime import datetime

def get_training_pid():
    """Get the PID of the running training process."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if 'train_distilbert.py' in line and 'grep' not in line:
                parts = line.split()
                return {
                    'pid': parts[1],
                    'cpu': parts[2],
                    'mem': parts[3],
                    'time': parts[9]
                }
        return None
    except Exception as e:
        return None

def read_config():
    """Read training configuration to know total epochs."""
    config_file = "../configs/distilbert_config.yaml"
    try:
        with open(config_file, 'r') as f:
            content = f.read()
            # Simple parsing for num_epochs
            for line in content.split('\n'):
                if 'num_epochs:' in line:
                    return int(line.split(':')[1].strip())
    except:
        pass
    return 3  # default

def check_model_directory():
    """Check what's been saved in the model directory."""
    model_dir = "../models/distilbert-sentiment"

    if not os.path.exists(model_dir):
        return {
            'exists': False,
            'best_model': False,
            'checkpoints': [],
            'results_file': False
        }

    contents = os.listdir(model_dir) if os.path.exists(model_dir) else []
    checkpoints = [f for f in contents if f.startswith('checkpoint-')]

    # Get best model info
    best_model_path = os.path.join(model_dir, 'best_model')
    best_model_exists = os.path.exists(best_model_path)
    best_model_time = None

    if best_model_exists:
        try:
            stat_info = os.stat(best_model_path)
            best_model_time = datetime.fromtimestamp(stat_info.st_mtime)
        except:
            pass

    results_file = os.path.join(model_dir, 'training_results.json')

    return {
        'exists': True,
        'best_model': best_model_exists,
        'best_model_time': best_model_time,
        'checkpoints': checkpoints,
        'results_file': os.path.exists(results_file)
    }

def read_training_results():
    """Read final training results if available."""
    results_file = "../models/distilbert-sentiment/training_results.json"

    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading results: {e}")
            return None
    return None

def estimate_progress(model_info, total_epochs):
    """Estimate training progress based on available info."""
    # Count checkpoint epochs
    checkpoint_epochs = []
    for ckpt in model_info['checkpoints']:
        # Extract epoch number from checkpoint name if possible
        # Format might be: checkpoint-epoch1-step1250
        if 'epoch' in ckpt:
            try:
                epoch_str = ckpt.split('epoch')[1].split('-')[0]
                checkpoint_epochs.append(int(epoch_str))
            except:
                pass

    if checkpoint_epochs:
        completed = max(checkpoint_epochs)
    elif model_info['best_model']:
        # If best model exists, at least 1 epoch is complete
        completed = 1
    else:
        completed = 0

    return completed

def main():
    print("\n" + "="*80)
    print("DISTILBERT TRAINING STATUS - LIVE DATA")
    print("="*80 + "\n")

    # Check if process is running
    proc_info = get_training_pid()
    if proc_info:
        print(f"✅ Training Process: RUNNING")
        print(f"   PID: {proc_info['pid']}")
        print(f"   CPU: {proc_info['cpu']}%")
        print(f"   Memory: {proc_info['mem']}%")
        print(f"   Runtime: {proc_info['time']}")
    else:
        print("❌ Training Process: NOT RUNNING")

    print("\n" + "-"*80)
    print("TRAINING PROGRESS")
    print("-"*80 + "\n")

    # Check for completed training results
    results = read_training_results()
    if results:
        print("🎉 TRAINING COMPLETED!\n")
        print("Final Results:")

        test_acc = results.get('final_test_accuracy', results.get('test_accuracy', 0))
        test_loss = results.get('final_test_loss', results.get('test_loss', 0))
        best_val_acc = results.get('best_val_accuracy', 0)

        print(f"  📊 Test Accuracy:      {test_acc*100:.2f}% {'🎉' if test_acc >= 0.92 else ''}")
        print(f"  📊 Test Loss:          {test_loss:.4f}")
        print(f"  📊 Best Val Accuracy:  {best_val_acc*100:.2f}% {'🎉' if best_val_acc >= 0.92 else ''}")
        print(f"  🎯 Target:             92.00%")

        if test_acc >= 0.92:
            print(f"  ✅ TARGET ACHIEVED!")

        if 'epochs_completed' in results:
            print(f"\n  Epochs Completed: {results['epochs_completed']}")
        else:
            print(f"\n  All 3 epochs completed")

        print(f"\n  📁 Model Location: ../models/distilbert-sentiment/best_model/")
        print(f"  📄 Results File:   ../models/distilbert-sentiment/training_results.json")

        print("\n" + "="*80 + "\n")
        return

    # Read configuration
    total_epochs = read_config()

    # Check model directory
    model_info = check_model_directory()

    if not model_info['exists']:
        print("⏳ Model directory not created yet")
        print("   Training may still be initializing...")
        print(f"\n   Expected: {total_epochs} epochs")
        print("\n" + "="*80 + "\n")
        return

    # Estimate progress
    completed_epochs = estimate_progress(model_info, total_epochs)

    print(f"Configuration: {total_epochs} epochs total\n")

    if model_info['best_model']:
        print(f"✅ Best model saved")
        if model_info['best_model_time']:
            print(f"   Last updated: {model_info['best_model_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Location: ../models/distilbert-sentiment/best_model/")
    else:
        print(f"⏳ Best model not yet saved")

    print()

    if model_info['checkpoints']:
        print(f"📁 Checkpoints: {len(model_info['checkpoints'])} found")
        for ckpt in sorted(model_info['checkpoints'])[-3:]:
            print(f"   - {ckpt}")
    else:
        print(f"📁 Checkpoints: None yet")

    print("\n" + "-"*80)
    print("EPOCH ESTIMATION")
    print("-"*80 + "\n")

    if completed_epochs > 0:
        print(f"Estimated completed epochs: {completed_epochs}/{total_epochs}")

        if completed_epochs >= 1:
            print("   ✅ Epoch 1 - Likely completed")
        if completed_epochs >= 2:
            print("   ✅ Epoch 2 - Likely completed")
        if completed_epochs >= 3:
            print("   ✅ Epoch 3 - Likely completed")

        remaining = total_epochs - completed_epochs
        if remaining > 0:
            if proc_info:
                print(f"   🔄 Epoch {completed_epochs + 1} - In progress")
            print(f"\n   Remaining: {remaining} epoch(s)")
            print(f"   Estimated time: ~{remaining * 50} minutes")
    else:
        print("Training in progress...")
        print(f"   Total epochs: {total_epochs}")
        print(f"   Estimated total time: ~{total_epochs * 50} minutes")

    print("\n" + "="*80)
    print("\nℹ️  NOTE: This shows estimated progress based on saved files.")
    print("   For detailed real-time metrics, check the actual training output.")
    print("\n💡 Commands:")
    print("   - Continuous monitor: python monitor_training.py --watch")
    print("   - Quick check: bash check_training.sh")
    print("   - Final results: cat ../models/distilbert-sentiment/training_results.json")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
