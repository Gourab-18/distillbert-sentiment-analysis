"""
DistilBERT Training Script for Sentiment Analysis

This script fine-tunes DistilBERT on the IMDb sentiment dataset with:
- AdamW optimizer with weight decay
- Linear warmup and decay scheduler
- Weights & Biases experiment tracking
- Automatic GPU/CPU detection
- Checkpoint saving and evaluation
"""

import os
import sys
import yaml
import json
import argparse
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from datasets import load_from_disk
from tqdm import tqdm
import wandb


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def detect_device(config):
    """Detect and configure device (GPU/CPU/MPS)."""
    if config['device']['auto_detect']:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{config['device']['gpu_id']}")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            print("GPU not available, using CPU")
    else:
        device = torch.device("cpu")
        print("Using CPU (auto_detect=False)")

    return device


def load_datasets(config):
    """Load preprocessed datasets."""
    print("\n" + "=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)

    data_dir = config['paths']['data_dir']
    print(f"Loading from: {data_dir}")

    # Load datasets
    datasets = load_from_disk(data_dir)

    print(f"\nDataset splits:")
    print(f"  Train: {len(datasets['train']):,} samples")
    print(f"  Validation: {len(datasets['validation']):,} samples")
    print(f"  Test: {len(datasets['test']):,} samples")

    return datasets


def create_dataloaders(datasets, config):
    """Create PyTorch DataLoaders."""
    print("\n" + "=" * 80)
    print("CREATING DATALOADERS")
    print("=" * 80)

    # Set format to PyTorch tensors
    datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Create dataloaders
    train_dataloader = DataLoader(
        datasets['train'],
        batch_size=config['data']['train_batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=torch.cuda.is_available()
    )

    val_dataloader = DataLoader(
        datasets['validation'],
        batch_size=config['data']['eval_batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=torch.cuda.is_available()
    )

    test_dataloader = DataLoader(
        datasets['test'],
        batch_size=config['data']['eval_batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=torch.cuda.is_available()
    )

    print(f"\nDataLoader configuration:")
    print(f"  Train batch size: {config['data']['train_batch_size']}")
    print(f"  Eval batch size: {config['data']['eval_batch_size']}")
    print(f"  Num workers: {config['data']['num_workers']}")
    print(f"  Train batches: {len(train_dataloader)}")
    print(f"  Val batches: {len(val_dataloader)}")
    print(f"  Test batches: {len(test_dataloader)}")

    return train_dataloader, val_dataloader, test_dataloader


def initialize_model(config, device):
    """Initialize DistilBERT model."""
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)

    model_name = config['model']['name']
    num_labels = config['model']['num_labels']

    print(f"Loading model: {model_name}")

    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        dropout=config['model']['dropout'],
        attention_dropout=config['model']['attention_dropout']
    )

    # Move to device
    model = model.to(device)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel information:")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")
    print(f"  Model size: {num_params * 4 / 1e6:.2f} MB (fp32)")

    return model


def setup_optimizer_and_scheduler(model, train_dataloader, config):
    """Setup optimizer and learning rate scheduler."""
    print("\n" + "=" * 80)
    print("CONFIGURING OPTIMIZER & SCHEDULER")
    print("=" * 80)

    # Prepare optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': config['training']['weight_decay']
        },
        {
            'params': [p for n, p in model.named_parameters()
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config['training']['learning_rate'],
        betas=config['optimizer']['betas'],
        eps=config['optimizer']['eps']
    )

    # Calculate total training steps
    num_epochs = config['training']['num_epochs']
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    warmup_steps = config['training']['warmup_steps']

    # Create scheduler
    if config['scheduler']['type'] == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    elif config['scheduler']['type'] == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config['scheduler']['type']}")

    print(f"\nOptimizer configuration:")
    print(f"  Type: {config['optimizer']['type']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Weight decay: {config['training']['weight_decay']}")
    print(f"  Betas: {config['optimizer']['betas']}")

    print(f"\nScheduler configuration:")
    print(f"  Type: {config['scheduler']['type']}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Epochs: {num_epochs}")

    return optimizer, scheduler, total_steps


def initialize_wandb(config):
    """Initialize Weights & Biases tracking."""
    if not config['wandb']['enabled']:
        print("\nWeights & Biases tracking disabled")
        return None

    print("\n" + "=" * 80)
    print("INITIALIZING WEIGHTS & BIASES")
    print("=" * 80)

    # Generate run name if not provided
    run_name = config['wandb']['run_name']
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"distilbert_{timestamp}"

    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        name=run_name,
        tags=config['wandb']['tags'],
        notes=config['wandb']['notes'],
        config={
            'model': config['model'],
            'training': config['training'],
            'optimizer': config['optimizer'],
            'data': config['data']
        }
    )

    print(f"Run name: {run_name}")
    print(f"Project: {config['wandb']['project']}")

    return wandb


def evaluate(model, dataloader, device, desc="Evaluating"):
    """Evaluate model on a dataset."""
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels
    }


def train_epoch(model, train_dataloader, optimizer, scheduler, device, config, epoch, global_step):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['training']['max_grad_norm']
        )

        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        total_loss += loss.item()
        global_step += 1

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{correct/total:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })

        # Log to wandb
        if config['wandb']['enabled'] and global_step % config['evaluation']['logging_steps'] == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/accuracy': correct / total,
                'train/learning_rate': scheduler.get_last_lr()[0],
                'train/step': global_step
            })

    avg_loss = total_loss / len(train_dataloader)
    accuracy = correct / total

    return avg_loss, accuracy, global_step


def save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, global_step,
                   val_metrics, config, is_best=False):
    """Save model checkpoint."""
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_best:
        checkpoint_dir = output_dir / "best_model"
    else:
        checkpoint_dir = output_dir / f"checkpoint-epoch{epoch}-step{global_step}"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model and tokenizer
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # Save training state
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_metrics': val_metrics
    }, checkpoint_dir / 'training_state.pt')

    # Save config
    with open(checkpoint_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    print(f"\n{'Best model' if is_best else 'Checkpoint'} saved to: {checkpoint_dir}")

    return checkpoint_dir


def train(config):
    """Main training function."""

    print("\n" + "=" * 80)
    print("DISTILBERT TRAINING - SENTIMENT ANALYSIS")
    print("=" * 80)

    # Set seed for reproducibility
    set_seed(config['seed'])
    print(f"\nSeed set to: {config['seed']}")

    # Detect device
    device = detect_device(config)

    # Load datasets
    datasets = load_datasets(config)

    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(datasets, config)

    # Initialize model
    model = initialize_model(config, device)

    # Load tokenizer (for saving checkpoints)
    tokenizer = DistilBertTokenizer.from_pretrained(config['model']['name'])

    # Setup optimizer and scheduler
    optimizer, scheduler, total_steps = setup_optimizer_and_scheduler(
        model, train_dataloader, config
    )

    # Initialize wandb
    wb = initialize_wandb(config)

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    best_val_accuracy = 0.0
    global_step = 0

    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{config['training']['num_epochs']}")
        print(f"{'='*80}")

        # Train
        train_loss, train_acc, global_step = train_epoch(
            model, train_dataloader, optimizer, scheduler,
            device, config, epoch, global_step
        )

        print(f"\nTraining results:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Accuracy: {train_acc:.4f}")

        # Evaluate on validation set
        print(f"\nEvaluating on validation set...")
        val_metrics = evaluate(model, val_dataloader, device, desc="Validation")

        print(f"\nValidation results:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")

        # Log to wandb
        if config['wandb']['enabled']:
            wandb.log({
                'epoch': epoch,
                'train/epoch_loss': train_loss,
                'train/epoch_accuracy': train_acc,
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy']
            })

        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_val_accuracy
        if is_best:
            best_val_accuracy = val_metrics['accuracy']
            print(f"\n🎉 New best validation accuracy: {best_val_accuracy:.4f}")

        save_checkpoint(
            model, tokenizer, optimizer, scheduler,
            epoch, global_step, val_metrics, config, is_best=is_best
        )

    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)

    test_metrics = evaluate(model, test_dataloader, device, desc="Test")

    print(f"\nTest results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")

    # Log final results
    if config['wandb']['enabled']:
        wandb.log({
            'test/loss': test_metrics['loss'],
            'test/accuracy': test_metrics['accuracy']
        })
        wandb.finish()

    # Save final results
    results = {
        'best_val_accuracy': best_val_accuracy,
        'final_test_accuracy': test_metrics['accuracy'],
        'final_test_loss': test_metrics['loss'],
        'config': config
    }

    results_path = Path(config['paths']['output_dir']) / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Training completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train DistilBERT for sentiment analysis")
    parser.add_argument(
        '--config',
        type=str,
        default='../configs/distilbert_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override wandb setting if specified
    if args.no_wandb:
        config['wandb']['enabled'] = False

    # Train model
    results = train(config)

    return results


if __name__ == "__main__":
    main()
