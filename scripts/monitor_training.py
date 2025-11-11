#!/usr/bin/env python3
"""
Training Monitor Script

This script monitors the DistilBERT training progress in real-time.
Run this while training is in progress to see live updates.

Usage:
    python monitor_training.py
"""

import os
import sys
import time
import subprocess
from datetime import datetime, timedelta

def check_training_process():
    """Check if training process is running."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        lines = result.stdout.split('\n')
        for line in lines:
            if 'train_distilbert.py' in line and 'grep' not in line:
                parts = line.split()
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                time_running = parts[9]
                return {
                    'running': True,
                    'pid': pid,
                    'cpu': cpu,
                    'mem': mem,
                    'time': time_running
                }
        return {'running': False}
    except Exception as e:
        print(f"Error checking process: {e}")
        return {'running': False}

def check_model_directory():
    """Check what's been saved in the model directory."""
    model_dir = "../models/distilbert-sentiment"
    if not os.path.exists(model_dir):
        return None

    contents = os.listdir(model_dir)
    info = {
        'best_model': 'best_model' in contents,
        'checkpoints': [f for f in contents if f.startswith('checkpoint-')],
        'results_file': 'training_results.json' in contents
    }
    return info

def get_file_modification_time(filepath):
    """Get last modification time of a file."""
    try:
        if os.path.exists(filepath):
            mtime = os.path.getmtime(filepath)
            return datetime.fromtimestamp(mtime)
    except:
        pass
    return None

def display_status():
    """Display current training status."""
    print("\n" + "="*80)
    print("DISTILBERT TRAINING MONITOR")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Check process
    proc_info = check_training_process()

    if proc_info['running']:
        print("\n✅ TRAINING STATUS: RUNNING")
        print(f"   PID: {proc_info['pid']}")
        print(f"   CPU Usage: {proc_info['cpu']}%")
        print(f"   Memory Usage: {proc_info['mem']}%")
        print(f"   Running Time: {proc_info['time']}")
    else:
        print("\n❌ TRAINING STATUS: NOT RUNNING")
        print("   No training process detected.")

    # Check model directory
    print("\n" + "-"*80)
    print("MODEL DIRECTORY STATUS")
    print("-"*80)

    model_info = check_model_directory()
    if model_info:
        if model_info['best_model']:
            print("✅ Best model saved")
            best_model_path = "../models/distilbert-sentiment/best_model"
            if os.path.exists(best_model_path):
                mtime = get_file_modification_time(best_model_path)
                if mtime:
                    print(f"   Last updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("⏳ Best model not yet saved (training in progress)")

        if model_info['checkpoints']:
            print(f"📁 Checkpoints: {len(model_info['checkpoints'])} found")
            for ckpt in sorted(model_info['checkpoints'])[-3:]:  # Show last 3
                print(f"   - {ckpt}")

        if model_info['results_file']:
            print("✅ Training results file exists")
        else:
            print("⏳ Training results not yet saved")
    else:
        print("⏳ Model directory not yet created")

    # Check for training results
    results_file = "../models/distilbert-sentiment/training_results.json"
    if os.path.exists(results_file):
        print("\n" + "-"*80)
        print("TRAINING RESULTS")
        print("-"*80)
        try:
            import json
            with open(results_file, 'r') as f:
                results = json.load(f)

            print(f"✅ Training completed!")
            if 'test_accuracy' in results:
                print(f"   Test Accuracy: {results['test_accuracy']:.4f}")
            if 'test_loss' in results:
                print(f"   Test Loss: {results['test_loss']:.4f}")
            if 'best_val_accuracy' in results:
                print(f"   Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
        except Exception as e:
            print(f"   Error reading results: {e}")

    print("\n" + "="*80)
    if proc_info['running']:
        print("Training in progress... Run this script again to check status.")
        print("\nEstimated total time: ~2.5 hours (3 epochs)")
        print("Time per epoch: ~50 minutes")
    else:
        print("Training not running. Check if it completed or encountered an error.")
    print("="*80 + "\n")

def monitor_continuous():
    """Continuously monitor training progress."""
    print("\n🔍 CONTINUOUS TRAINING MONITOR")
    print("Press Ctrl+C to stop monitoring\n")

    try:
        while True:
            display_status()
            time.sleep(30)  # Update every 30 seconds
            print("\n" + "↻ Refreshing in 30 seconds..." + "\n")
    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped by user.\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        monitor_continuous()
    else:
        display_status()
        print("\nTip: Use 'python monitor_training.py --watch' for continuous monitoring\n")
