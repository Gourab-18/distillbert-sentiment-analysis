#!/usr/bin/env python3
"""
View Training Output - Shows epoch and accuracy information
Dynamically extracts data from training logs
"""

import subprocess
import re
import sys
import os
import json

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
                return parts[1]
        return None
    except Exception as e:
        print(f"Error finding training process: {e}")
        return None

def parse_training_log():
    """Parse training log to extract epoch and accuracy information."""
    # Try to read from a log file if it exists
    log_file = "../logs/training.log"

    if not os.path.exists(log_file):
        # If no log file, try to use lsof to find output
        return None

    epochs_data = []
    current_epoch = None

    try:
        with open(log_file, 'r') as f:
            content = f.read()

            # Parse epoch headers
            epoch_pattern = r'EPOCH (\d+)/(\d+)'
            # Parse training results
            train_acc_pattern = r'Training results:.*?Accuracy: ([\d.]+)'
            train_loss_pattern = r'Training results:.*?Loss: ([\d.]+)'
            # Parse validation results
            val_acc_pattern = r'Validation results:.*?Accuracy: ([\d.]+)'
            val_loss_pattern = r'Validation results:.*?Loss: ([\d.]+)'

            # Extract all epochs
            for match in re.finditer(epoch_pattern, content):
                epoch_num = int(match.group(1))
                total_epochs = int(match.group(2))

                # Find training and validation results after this epoch
                pos = match.end()
                next_content = content[pos:pos+500]

                train_loss_match = re.search(train_loss_pattern, next_content, re.DOTALL)
                train_acc_match = re.search(train_acc_pattern, next_content, re.DOTALL)
                val_loss_match = re.search(val_loss_pattern, next_content, re.DOTALL)
                val_acc_match = re.search(val_acc_pattern, next_content, re.DOTALL)

                epoch_info = {
                    'epoch': epoch_num,
                    'total': total_epochs,
                    'train_loss': float(train_loss_match.group(1)) if train_loss_match else None,
                    'train_acc': float(train_acc_match.group(1)) if train_acc_match else None,
                    'val_loss': float(val_loss_match.group(1)) if val_loss_match else None,
                    'val_acc': float(val_acc_match.group(1)) if val_acc_match else None,
                }

                # Only add if we have at least accuracy data
                if epoch_info['train_acc'] is not None or epoch_info['val_acc'] is not None:
                    epochs_data.append(epoch_info)

            return {
                'epochs': epochs_data,
                'total_epochs': total_epochs if epochs_data else 3
            }
    except Exception as e:
        print(f"Error parsing log file: {e}")
        return None

def get_training_results():
    """Get training results if training is complete."""
    results_file = "../models/distilbert-sentiment/training_results.json"

    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def extract_training_summary_dynamic():
    """Extract epoch and accuracy information dynamically."""
    print("\n" + "="*80)
    print("DISTILBERT TRAINING - EPOCH & ACCURACY SUMMARY")
    print("="*80 + "\n")

    # Check if training is running
    pid = get_training_pid()
    if pid:
        print(f"✅ Training Process: Running (PID: {pid})")
    else:
        print("❌ Training Process: Not Running")

    print("\n" + "-"*80)
    print("TRAINING PROGRESS")
    print("-"*80 + "\n")

    # Try to get results from completed training first
    results = get_training_results()
    if results:
        print("🎉 TRAINING COMPLETED!")
        print("")
        print(f"📊 Final Test Accuracy:  {results.get('test_accuracy', 0)*100:.2f}%")
        print(f"📊 Final Test Loss:      {results.get('test_loss', 0):.4f}")
        print(f"📊 Best Val Accuracy:    {results.get('best_val_accuracy', 0)*100:.2f}%")
        print("")
        print(json.dumps(results, indent=2))
        return

    # Try to parse log file
    log_data = parse_training_log()

    if log_data and log_data['epochs']:
        total_epochs = log_data['total_epochs']
        completed_epochs = log_data['epochs']
        best_val_acc = max([e['val_acc'] for e in completed_epochs if e['val_acc']], default=0)

        # Show completed epochs
        for epoch_info in completed_epochs:
            epoch_num = epoch_info['epoch']
            status = "✅ COMPLETED"

            print(f"📊 EPOCH {epoch_num}/{total_epochs} - {status}")
            if epoch_info['train_loss'] is not None:
                print(f"   Training Loss:       {epoch_info['train_loss']:.4f}")
            if epoch_info['train_acc'] is not None:
                print(f"   Training Accuracy:   {epoch_info['train_acc']*100:.2f}%")
            if epoch_info['val_loss'] is not None:
                print(f"   Validation Loss:     {epoch_info['val_loss']:.4f}")
            if epoch_info['val_acc'] is not None:
                acc_str = f"{epoch_info['val_acc']*100:.2f}%"
                if epoch_info['val_acc'] >= 0.92:
                    acc_str += " 🎉"
                print(f"   Validation Accuracy: {acc_str}")
                if epoch_info['val_acc'] == best_val_acc:
                    print("   Status: Best validation accuracy!")
            print("")

        # Show pending epochs
        current_epoch = len(completed_epochs)
        if current_epoch < total_epochs:
            next_epoch = current_epoch + 1
            print(f"📊 EPOCH {next_epoch}/{total_epochs} - 🔄 IN PROGRESS")
            print("   Status: Currently training...")
            print("")

            for ep in range(next_epoch + 1, total_epochs + 1):
                print(f"📊 EPOCH {ep}/{total_epochs} - ⏳ PENDING")
                print("   Status: Waiting to start")
                print("")

        print("-"*80)
        print("KEY METRICS")
        print("-"*80 + "\n")

        print(f"✨ Best Validation Accuracy:  {best_val_acc*100:.2f}%")
        print(f"🎯 Target Accuracy:           92.00%")
        if best_val_acc >= 0.92:
            print(f"📈 Status:                    TARGET ACHIEVED! ✅")
        else:
            print(f"📈 Status:                    {(best_val_acc/0.92)*100:.1f}% of target")
        print("")

        epochs_left = total_epochs - current_epoch
        if epochs_left > 0:
            time_est = epochs_left * 50
            print(f"⏱️  Estimated Time Remaining: ~{time_est} minutes ({epochs_left} epoch{'s' if epochs_left > 1 else ''} left)")
        else:
            print("⏱️  Training completed!")

        print("💾 Model Saved:               ../models/distilbert-sentiment/best_model/")
        print("")
    else:
        # Fallback: Show message about log file
        print("⚠️  No log file found. Training output may not be captured.")
        print("")
        print("To view training progress:")
        print("1. Check if training is running: ps aux | grep train_distilbert.py")
        print("2. Use continuous monitor: python monitor_training.py --watch")
        print("3. Check model directory: ls -lh ../models/distilbert-sentiment/")
        print("")

        # Try to show checkpoint info
        if os.path.exists("../models/distilbert-sentiment/best_model"):
            print("✅ Best model has been saved")
            print("📁 Location: ../models/distilbert-sentiment/best_model/")
        else:
            print("⏳ Waiting for training to save first checkpoint...")
        print("")

    print("="*80)
    print("")

    print("💡 TIP: For continuous monitoring with live updates:")
    print("   python monitor_training.py --watch")
    print("")

def show_detailed_command_options():
    """Show command options for viewing training details."""
    print("\n" + "="*80)
    print("COMMANDS TO VIEW TRAINING DETAILS")
    print("="*80 + "\n")

    print("1. Quick Status Check:")
    print("   bash check_training.sh")
    print("")

    print("2. Detailed Monitoring (updates every 30s):")
    print("   python monitor_training.py --watch")
    print("")

    print("3. View this summary:")
    print("   python view_training_output.py")
    print("")

    print("4. Check if process is running:")
    print("   ps aux | grep train_distilbert.py")
    print("")

    print("5. View saved model:")
    print("   ls -lh ../models/distilbert-sentiment/best_model/")
    print("")

    print("6. View results (after training completes):")
    print("   cat ../models/distilbert-sentiment/training_results.json | python -m json.tool")
    print("")

    print("="*80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--commands":
        show_detailed_command_options()
    else:
        extract_training_summary_dynamic()
        print("Run with '--commands' flag to see all monitoring options")
        print("Example: python view_training_output.py --commands\n")
