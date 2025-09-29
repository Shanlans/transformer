"""
Main Training Script
Trains Transformer model using JSON configuration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import time
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.transformer import Transformer
from datasets.translation_dataset import TranslationDataset, create_dataloader
from trainers.transformer_trainer import TransformerTrainer
from utils.config_manager import load_config
from utils.config_version_manager import ConfigVersionManager


def setup_device(config):
    """Setup training device."""
    if config.system.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.system.device)
    
    print(f"Using device: {device}")
    return device


def create_datasets(config):
    """Create training and validation datasets."""
    print("Creating datasets...")
    
    # Load full dataset
    full_dataset = TranslationDataset(
        json_file=config.data.train_data_path,
        max_length=config.data.max_length
    )
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    train_size = int(config.data.train_split * dataset_size)
    val_size = dataset_size - train_size
    
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Dataset split: {train_size} train, {val_size} validation")
    print(f"Source vocabulary size: {full_dataset.get_vocab_sizes()[0]}")
    print(f"Target vocabulary size: {full_dataset.get_vocab_sizes()[1]}")
    
    return train_dataset, val_dataset, full_dataset


def create_dataloaders(config, train_dataset, val_dataset):
    """Create data loaders."""
    print("Creating data loaders...")
    
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.data.num_workers
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")
    
    return train_dataloader, val_dataloader


def create_model(config, src_vocab_size, tgt_vocab_size):
    """Create Transformer model."""
    print("Creating model...")
    
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_encoder_layers=config.model.n_encoder_layers,
        n_decoder_layers=config.model.n_decoder_layers,
        d_ff=config.model.d_ff,
        max_len=config.model.max_len,
        dropout=config.model.dropout
    )
    
    # Initialize parameters
    model.init_parameters()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 / 1024
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {model_size_mb:.2f} MB (FP32)")
    print(f"  Encoder layers: {config.model.n_encoder_layers}")
    print(f"  Decoder layers: {config.model.n_decoder_layers}")
    print(f"  Attention heads: {config.model.n_heads}")
    print(f"  Model dimension: {config.model.d_model}")
    
    return model


def create_trainer(config, model, train_dataloader, val_dataloader, device):
    """Create trainer."""
    print("Creating trainer...")
    
    trainer = TransformerTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler_type=config.scheduler.type,
        loss_type=config.loss.type,
        ignore_index=config.loss.ignore_index,
        gradient_clip_norm=config.training.gradient_clip_norm,
        save_dir=config.system.save_dir
    )
    
    # Set experiment timestamp if available
    if hasattr(config.experiment, 'timestamp_id') and config.experiment.timestamp_id:
        trainer.set_experiment_timestamp(config.experiment.timestamp_id)
    
    # Set experiment timestamp from environment variable if available (for default config)
    import os
    experiment_name = os.environ.get('EXPERIMENT_NAME')
    if experiment_name:
        print(f"🔗 Linking to experiment: {experiment_name}")
        # Load experiment to get timestamp_id
        try:
            from src.utils.experiment_manager import ExperimentManager
            exp_manager = ExperimentManager()
            exp_config_manager = exp_manager.load_experiment(experiment_name)
            exp_config = exp_config_manager.get_config()
            if hasattr(exp_config.experiment, 'timestamp_id') and exp_config.experiment.timestamp_id:
                trainer.set_experiment_timestamp(exp_config.experiment.timestamp_id)
                print(f"✅ Experiment timestamp set: {exp_config.experiment.timestamp_id}")
        except Exception as e:
            print(f"⚠️  Could not link to experiment {experiment_name}: {e}")
    
    print(f"Trainer created:")
    print(f"  Optimizer: {config.optimizer.type}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Weight decay: {config.training.weight_decay}")
    print(f"  Scheduler: {config.scheduler.type}")
    print(f"  Loss function: {config.loss.type}")
    print(f"  Gradient clipping: {config.training.gradient_clip_norm}")
    
    return trainer


def print_training_summary(config):
    """Print training summary."""
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Experiment: {config.experiment.name}")
    print(f"Description: {config.experiment.description}")
    print(f"Version: {config.experiment.version}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nModel Configuration:")
    print(f"  Architecture: Transformer")
    print(f"  Model dimension: {config.model.d_model}")
    print(f"  Attention heads: {config.model.n_heads}")
    print(f"  Encoder layers: {config.model.n_encoder_layers}")
    print(f"  Decoder layers: {config.model.n_decoder_layers}")
    print(f"  Feed-forward dimension: {config.model.d_ff}")
    print(f"  Dropout: {config.model.dropout}")
    print("\nTraining Configuration:")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Early stopping patience: {config.training.early_stopping_patience}")
    print(f"  Max sequence length: {config.data.max_length}")
    print("\nFeatures:")
    print(f"  Visualization: {'Enabled' if config.visualization.enabled else 'Disabled'}")
    print(f"  Evaluation: {'Enabled' if config.evaluation.enabled else 'Disabled'}")
    print(f"  Checkpoint management: Enabled")
    print("=" * 80)


def main():
    """Main training function."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Transformer model")
    parser.add_argument('--config', default='training_config.json', 
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Check for temporary config file (used by experiment manager)
    temp_config_path = 'temp_training_config.json'
    if os.path.exists(temp_config_path):
        config_path = temp_config_path
        print(f"Using temporary config file: {config_path}")
    else:
        config_path = args.config
    config_manager = load_config(config_path)
    config = config_manager.get_config()
    
    # Print configuration
    config_manager.print_config()
    
    # Setup device
    device = setup_device(config)
    
    # Create datasets
    train_dataset, val_dataset, full_dataset = create_datasets(config)
    
    # Get vocabulary sizes
    src_vocab_size, tgt_vocab_size = full_dataset.get_vocab_sizes()
    
    # Create data loaders
    train_dataloader, val_dataloader = create_dataloaders(config, train_dataset, val_dataset)
    
    # Create model
    model = create_model(config, src_vocab_size, tgt_vocab_size)
    
    # Create trainer
    trainer = create_trainer(config, model, train_dataloader, val_dataloader, device)
    
    # Set evaluator if evaluation is enabled
    if config.evaluation.enabled:
        trainer.set_evaluator(full_dataset.tgt_vocab, full_dataset.tgt_idx2word)
    
    # Check for resume configuration
    if hasattr(config, 'resume') and config.resume:
        print(f"\n🔄 RESUME TRAINING DETECTED")
        print(f"{'='*60}")
        print(f"Resuming from checkpoint: {config.resume.checkpoint_path}")
        print(f"Original experiment: {config.resume.original_experiment}")
        print(f"Resume timestamp: {config.resume.resume_timestamp}")
        
        # Load model and previous training history
        trainer.load_model(config.resume.checkpoint_path, load_training_history=True)
        
        # Show validation result if available
        validation_result = config.resume.validation_result
        print(f"\n📊 Validation Results:")
        print(f"  Status: {'✅ PASSED' if validation_result['is_valid'] else '❌ FAILED'}")
        if validation_result['non_structural_changes']:
            print(f"  Non-structural changes: {len(validation_result['non_structural_changes'])}")
            for change in validation_result['non_structural_changes']:
                print(f"    - {change['section']}.{change['parameter']}: {change['original']} → {change['new']}")
        print(f"{'='*60}")
    
    # Print training summary
    print_training_summary(config)
    
    # Record configuration version before training
    config_version_manager = ConfigVersionManager()
    config_version_id = config_version_manager.auto_save_config_version(
        config_path=config_path,
        description=f"Training run starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        tags=["training", "manual_run"],
        preserve_experiment_info=True
    )
    print(f"\nConfiguration version recorded: {config_version_id}")
    
    # Start training
    print("\nStarting training...")
    start_time = time.time()
    
    metrics_history = trainer.train(
        num_epochs=config.training.epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        save_best_only=config.training.save_best_only
    )
    
    training_time = time.time() - start_time
    
    # Link configuration version with checkpoint run
    config_version_manager.link_config_with_checkpoint(
        config_version_id=config_version_id,
        checkpoint_run_dir=trainer.run_dir,
        link_type="training"
    )
    print(f"Configuration version linked with checkpoint run: {trainer.run_dir}")
    
    # Print final results
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Final training loss: {metrics_history[-1]['train_loss']:.4f}")
    if 'val_loss' in metrics_history[-1]:
        print(f"Final validation loss: {metrics_history[-1]['val_loss']:.4f}")
    print(f"Total epochs: {len(metrics_history)}")
    print(f"Average time per epoch: {training_time/len(metrics_history):.2f} seconds")
    
    # Evaluate model if enabled
    if config.evaluation.enabled:
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        evaluation_metrics = trainer.evaluate_model(
            val_dataloader, 
            max_samples=config.evaluation.max_samples
        )
        
        if evaluation_metrics:
            print("Evaluation Results:")
            for metric, value in evaluation_metrics.items():
                if metric == "Perplexity":
                    print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value:.4f}")
        else:
            print("Evaluation not available")
    
    # Print visualization info
    if config.visualization.enabled:
        print(f"\nVisualizations saved to: {trainer.visualizer.save_dir}")
        print("Available visualizations:")
        print("  - Training metrics plots")
        print("  - Gradient flow analysis")
        print("  - Evaluation metrics")
        print("  - Metrics summary JSON")
    
    print(f"\nModel checkpoints saved to: {config.system.save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
