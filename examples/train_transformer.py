"""
Complete Transformer Training Script
Trains a Transformer model for English-German translation using the Multi30k dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import sys
import json
from typing import Dict, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.transformer import Transformer
from datasets.translation_dataset import TranslationDataset, create_dataloader
from trainers.transformer_trainer import TransformerTrainer
from utils.loss_functions import create_loss_function


def create_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 512,
    n_heads: int = 8,
    n_encoder_layers: int = 6,
    n_decoder_layers: int = 6,
    d_ff: int = 2048,
    max_len: int = 5000,
    dropout: float = 0.1
) -> Transformer:
    """
    Create a Transformer model with specified parameters.
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension
        n_heads: Number of attention heads
        n_encoder_layers: Number of encoder layers
        n_decoder_layers: Number of decoder layers
        d_ff: Feed-forward dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
        
    Returns:
        Transformer model instance
    """
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    )
    
    # Initialize parameters
    model.init_parameters()
    
    return model


def create_datasets(
    data_path: str = "data/train.json",
    max_length: int = 20,
    train_split: float = 0.8
) -> tuple[TranslationDataset, TranslationDataset]:
    """
    Create training and validation datasets.
    
    Args:
        data_path: Path to training data
        max_length: Maximum sequence length
        train_split: Fraction of data to use for training
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Load full dataset
    full_dataset = TranslationDataset(
        json_file=data_path,
        max_length=max_length
    )
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Dataset split: {train_size} train, {val_size} validation")
    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: TranslationDataset,
    val_dataset: TranslationDataset,
    batch_size: int = 32,
    num_workers: int = 0
) -> tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_dataloader, val_dataloader


def save_config(config: Dict[str, Any], save_dir: str):
    """Save training configuration to JSON file."""
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Transformer for English-German Translation')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/train.json', help='Path to training data')
    parser.add_argument('--max_length', type=int, default=20, help='Maximum sequence length')
    parser.add_argument('--train_split', type=float, default=0.8, help='Fraction of data for training')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_encoder_layers', type=int, default=6, help='Number of encoder layers')
    parser.add_argument('--n_decoder_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0, help='Gradient clipping norm')
    
    # Training strategy arguments
    parser.add_argument('--scheduler_type', type=str, default='none', choices=['none', 'step', 'cosine'], help='Learning rate scheduler')
    parser.add_argument('--loss_type', type=str, default='masked', choices=['masked', 'label_smoothing'], help='Loss function type')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker processes')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['device'] = str(device)
    save_config(config, args.save_dir)
    
    print("=" * 60)
    print("TRANSFORMER TRAINING")
    print("=" * 60)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset = create_datasets(
        data_path=args.data_path,
        max_length=args.max_length,
        train_split=args.train_split
    )
    
    # Get vocabulary sizes
    src_vocab_size, tgt_vocab_size = train_dataset.dataset.get_vocab_sizes()
    print(f"Vocabulary sizes: Source={src_vocab_size}, Target={tgt_vocab_size}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_ff=args.d_ff,
        max_len=5000,
        dropout=args.dropout
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # Create trainer
    print("Creating trainer...")
    trainer = TransformerTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler_type,
        loss_type=args.loss_type,
        ignore_index=0,  # PAD token
        gradient_clip_norm=args.gradient_clip_norm,
        save_dir=args.save_dir
    )
    
    # Set evaluator
    trainer.set_evaluator(train_dataset.dataset.tgt_vocab, train_dataset.dataset.tgt_idx2word)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_model(args.resume)
    
    # Start training
    print("Starting training...")
    metrics_history = trainer.train(
        num_epochs=args.epochs,
        save_path=os.path.join(args.save_dir, "final_model.pt"),
        early_stopping_patience=args.early_stopping_patience,
        save_best_only=True
    )
    
    # Print final results
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Final training loss: {metrics_history[-1]['train_loss']:.4f}")
    if 'val_loss' in metrics_history[-1]:
        print(f"Final validation loss: {metrics_history[-1]['val_loss']:.4f}")
    print(f"Total epochs: {len(metrics_history)}")
    print(f"Model saved to: {args.save_dir}")
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    evaluation_metrics = trainer.evaluate_model(val_dataloader, max_samples=50)
    
    if evaluation_metrics:
        print("Evaluation Results:")
        for metric, value in evaluation_metrics.items():
            print(f"  {metric}: {value:.4f}")
    else:
        print("Evaluation not available (evaluator not set)")


if __name__ == "__main__":
    main()
