"""
Transformer Trainer Implementation
Complete training system for Transformer models with comprehensive features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import time
import os
import json
from tqdm import tqdm
import sys
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.loss_functions import create_loss_function
from utils.checkpoint_manager import CheckpointManager
from utils.training_visualizer import TrainingVisualizer
from utils.evaluation_metrics import TranslationEvaluator
from utils.config_version_manager import ConfigVersionManager


class TransformerTrainer:
    """
    Comprehensive Transformer Trainer
    
    This trainer provides a complete training system with:
    - Training and validation loops
    - Loss computation and optimization
    - Learning rate scheduling
    - Model checkpointing and resuming
    - Training metrics and logging
    - Early stopping and best model tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cpu",
        learning_rate: float = 0.0001,
        weight_decay: float = 0.01,
        scheduler_type: str = "none",
        loss_type: str = "masked",
        ignore_index: int = 0,
        gradient_clip_norm: float = 1.0,
        save_dir: str = "checkpoints"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Transformer model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            optimizer: Optimizer (if None, will create AdamW)
            criterion: Loss function (if None, will create based on loss_type)
            device: Device to use for training
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            scheduler_type: Type of learning rate scheduler ("none", "step", "cosine")
            loss_type: Type of loss function ("masked", "label_smoothing")
            ignore_index: Index to ignore in loss calculation (PAD token)
            gradient_clip_norm: Gradient clipping norm
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device(device)
        self.gradient_clip_norm = gradient_clip_norm
        self.save_dir = save_dir
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            base_dir=save_dir,
            max_checkpoints=10  # Keep last 10 checkpoints per run
        )
        
        # Create run directory
        self.run_dir = self.checkpoint_manager.create_run_directory()
        
        # Initialize config version manager
        self.config_version_manager = ConfigVersionManager()
        
        # Store experiment timestamp for checkpoint naming
        self.experiment_timestamp = None
        
        # Initialize visualizer
        self.visualizer = TrainingVisualizer(
            save_dir=os.path.join(self.run_dir, "visualizations")
        )
        
        # Initialize evaluator (will be set when dataset is available)
        self.evaluator = None
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.98),
                eps=1e-9
            )
        else:
            self.optimizer = optimizer
        
        # Initialize loss function
        if criterion is None:
            self.criterion = create_loss_function(
                loss_type=loss_type,
                ignore_index=ignore_index
            )
        else:
            self.criterion = criterion
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler(scheduler_type, learning_rate)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Optimizer: {type(self.optimizer).__name__}")
        print(f"  Loss function: {type(self.criterion).__name__}")
        print(f"  Scheduler: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
        print(f"  Save directory: {save_dir}")
        print(f"  Visualizer: {type(self.visualizer).__name__}")
    
    def set_evaluator(self, tgt_vocab: Dict[str, int], idx2word: Dict[int, str]):
        """
        Set the evaluator for the trainer.
        
        Args:
            tgt_vocab: Target vocabulary mapping
            idx2word: Index to word mapping
        """
        self.evaluator = TranslationEvaluator(tgt_vocab, idx2word)
        print(f"Evaluator set with vocabulary size: {len(tgt_vocab)}")
    
    def set_experiment_timestamp(self, timestamp: str):
        """
        Set experiment timestamp for checkpoint directory naming.
        
        Args:
            timestamp: Experiment timestamp (format: YYYYMMDD_HHMMSS)
        """
        self.experiment_timestamp = timestamp
        
        # Also set the unified timestamp manager
        try:
            from ..utils.timestamp_manager import get_timestamp_manager
            timestamp_manager = get_timestamp_manager()
            timestamp_manager.set_timestamp(timestamp)
        except ImportError:
            pass
        
        print(f"Experiment timestamp set: {timestamp}")
        
        # Recreate run directory with unified timestamp
        if self.experiment_timestamp:
            print(f"Recreating run directory with experiment timestamp: {self.experiment_timestamp}")
            self.run_dir = self.checkpoint_manager.create_run_directory(custom_timestamp=self.experiment_timestamp)
            print(f"New run directory: {self.run_dir}")
            
            # Update visualizer to use the new run directory
            self.visualizer = TrainingVisualizer(
                save_dir=os.path.join(self.run_dir, "visualizations")
            )
    
    def _create_scheduler(self, scheduler_type: str, learning_rate: float) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if scheduler_type == "none":
            return None
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)  # type: ignore
        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)  # type: ignore
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            src = batch['src'].to(self.device)
            tgt_input = batch['tgt_input'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(src, tgt_input)
            
            # Compute loss
            loss = self.criterion(predictions, tgt_output)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return {
            'train_loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary containing validation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_dataloader, desc="Validation")
            
            for batch in progress_bar:
                # Move batch to device
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                
                # Forward pass
                predictions = self.model(src, tgt_input)
                
                # Compute loss
                loss = self.criterion(predictions, tgt_output)
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return {
            'val_loss': avg_loss
        }
    
    def train(
        self,
        num_epochs: int,
        save_path: Optional[str] = None,
        early_stopping_patience: int = 10,
        save_best_only: bool = True
    ) -> List[Dict[str, float]]:
        """
        Complete training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_path: Path to save final model
            early_stopping_patience: Number of epochs to wait before early stopping
            save_best_only: Whether to save only the best model
            
        Returns:
            List of training metrics for each epoch
        """
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING")
        print(f"{'='*60}")
        print(f"Total epochs: {num_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Training batches per epoch: {len(self.train_dataloader)}")
        if self.val_dataloader:
            print(f"Validation batches per epoch: {len(self.val_dataloader)}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*60}")
        
        # If experiment timestamp is set, recreate run directory with matching timestamp
        if self.experiment_timestamp:
            print(f"Recreating run directory with experiment timestamp: {self.experiment_timestamp}")
            self.run_dir = self.checkpoint_manager.create_run_directory(custom_timestamp=self.experiment_timestamp)
            print(f"New run directory: {self.run_dir}")
            
            # Update visualizer to use the new run directory
            self.visualizer = TrainingVisualizer(
                save_dir=os.path.join(self.run_dir, "visualizations")
            )
        
        # Record configuration version (after run directory is finalized)
        config_version_id = self.config_version_manager.auto_save_config_version(
            config_path="training_config.json",
            checkpoint_run_dir=self.run_dir,
            description=f"Training run starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            tags=["training", "auto_saved"]
        )
        print(f"Configuration version recorded: {config_version_id}")
        
        metrics_history = []
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['epoch'] = epoch + 1
            epoch_metrics['epoch_time'] = time.time() - epoch_start_time
            epoch_metrics['timestamp'] = datetime.now().isoformat()
            metrics_history.append(epoch_metrics)
            
            # Add metrics to visualizer
            self.visualizer.add_metrics(epoch + 1, epoch_metrics)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Learning Rate: {train_metrics['learning_rate']:.6f}")
            print(f"  Epoch Time: {epoch_metrics['epoch_time']:.2f}s")
            
            # Print progress
            progress = (epoch + 1) / num_epochs * 100
            print(f"  Progress: {progress:.1f}%")
            
            # Print estimated remaining time
            if epoch > 0:
                avg_epoch_time = sum(m['epoch_time'] for m in metrics_history) / len(metrics_history)
                remaining_epochs = num_epochs - (epoch + 1)
                estimated_remaining = remaining_epochs * avg_epoch_time
                print(f"  Estimated remaining time: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f}min)")
            
            # Check for best model
            if val_metrics and val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                
                if save_best_only:
                    # Save best checkpoint using checkpoint manager
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch + 1,
                        loss=val_metrics['val_loss'],
                        metrics=epoch_metrics,
                        scheduler=self.scheduler,
                        checkpoint_name=f"best_model_epoch_{epoch + 1:03d}",
                        is_best=True
                    )
                    print(f"  New best model saved! Val Loss: {self.best_val_loss:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            print("-" * 50)
        
        # Save final model
        if save_path:
            # Save final checkpoint using checkpoint manager
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=len(metrics_history),
                loss=metrics_history[-1]['train_loss'],
                metrics=metrics_history[-1],
                scheduler=self.scheduler,
                checkpoint_name="final_model",
                is_best=False
            )
            print(f"Final model saved to: {save_path}")
        
        # Save training history
        self.save_training_history(metrics_history)
        
        # Generate visualizations
        self.generate_training_visualizations()
        
        print("Training completed!")
        return metrics_history
    
    def save_model(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the model
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str, load_training_history: bool = True):
        """
        Load model checkpoint.
        
        Args:
            path: Path to load the model from
            load_training_history: Whether to load previous training history for visualization
        """
        checkpoint_data = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=str(self.device)
        )
        
        # Update trainer state
        self.current_epoch = checkpoint_data.get('epoch', 0)
        self.best_val_loss = checkpoint_data.get('loss', float('inf'))
        
        # Update loss history if available
        if 'metrics' in checkpoint_data:
            metrics = checkpoint_data['metrics']
            if 'train_loss' in metrics:
                self.train_losses.append(metrics['train_loss'])
            if 'val_loss' in metrics:
                self.val_losses.append(metrics['val_loss'])
        
        # Load previous training history for visualization
        if load_training_history:
            run_dir = os.path.dirname(path)
            training_history_path = os.path.join(run_dir, 'training_history.json')
            
            if os.path.exists(training_history_path):
                print(f"📊 Loading previous training history for visualization...")
                success = self.visualizer.load_previous_training_history(training_history_path)
                if success:
                    print(f"✅ Previous training history loaded successfully")
                else:
                    print(f"⚠️  Could not load previous training history")
            else:
                print(f"⚠️  No previous training history found at: {training_history_path}")
        
        print(f"Model loaded from {path}")
        print(f"Resumed from epoch {self.current_epoch + 1}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_training_history(self, metrics_history: List[Dict[str, float]]):
        """Save training history to JSON file."""
        history_path = os.path.join(self.run_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        print(f"Training history saved to {history_path}")
    
    def list_checkpoints(self):
        """List all checkpoints in current run."""
        return self.checkpoint_manager.list_checkpoints(self.run_dir)
    
    def list_runs(self):
        """List all runs."""
        return self.checkpoint_manager.list_runs()
    
    def get_best_checkpoint(self):
        """Get the best checkpoint in current run."""
        return self.checkpoint_manager.get_best_checkpoint(self.run_dir)
    
    def cleanup_checkpoints(self, keep_latest: int = 5):
        """Cleanup checkpoints in current run."""
        return self.checkpoint_manager.cleanup_run(self.run_dir, keep_latest)
    
    def cleanup_all_runs(self, keep_latest_runs: int = 3, keep_latest_checkpoints: int = 5):
        """Cleanup all runs and checkpoints."""
        return self.checkpoint_manager.cleanup_all_runs(keep_latest_runs, keep_latest_checkpoints)
    
    def generate_training_visualizations(self):
        """Generate comprehensive training visualizations."""
        print("Generating training visualizations...")
        
        # Plot training metrics
        self.visualizer.plot_training_metrics("training_metrics.png")
        
        # Plot gradient flow for the last epoch
        if self.current_epoch > 0:
            self.visualizer.plot_gradient_flow(self.model, self.current_epoch, "gradient_flow_final.png")
        
        # Save metrics summary
        self.visualizer.save_metrics_summary("metrics_summary.json")
        
        print("Training visualizations generated!")
    
    def evaluate_model(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate model using comprehensive metrics.
        
        Args:
            dataloader: Data loader for evaluation
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.evaluator is None:
            print("Evaluator not set. Please call set_evaluator() first.")
            return {}
        
        print("Evaluating model...")
        metrics = self.evaluator.evaluate_model(
            model=self.model,
            dataloader=dataloader,
            device=str(self.device),
            max_samples=max_samples
        )
        
        # Plot evaluation metrics
        if metrics:
            self.visualizer.plot_evaluation_metrics(
                {k: [v] for k, v in metrics.items()},
                "evaluation_metrics.png"
            )
        
        print("Model evaluation completed!")
        return metrics
    
    def visualize_attention(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_vocab: Dict[str, int],
        tgt_vocab: Dict[str, int],
        epoch: int,
        layer_idx: int = 0,
        head_idx: int = 0
    ) -> str:
        """
        Visualize attention weights for a specific example.
        
        Args:
            src_tokens: Source token sequence
            tgt_tokens: Target token sequence
            src_vocab: Source vocabulary
            tgt_vocab: Target vocabulary
            epoch: Current epoch
            layer_idx: Encoder layer index
            head_idx: Attention head index
            
        Returns:
            Path to saved attention plot
        """
        return self.visualizer.plot_attention_weights(
            model=self.model,
            src_tokens=src_tokens,
            tgt_tokens=tgt_tokens,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            epoch=epoch,
            layer_idx=layer_idx,
            head_idx=head_idx
        )


if __name__ == "__main__":
    # Test the trainer with a simple model
    print("Testing TransformerTrainer...")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
        
        def forward(self, src, tgt_input):
            # Simple test forward pass
            batch_size, seq_len = src.size()
            # Use the linear layer to create trainable parameters
            dummy_input = torch.randn(batch_size, seq_len, 10, requires_grad=True)
            output = self.linear(dummy_input)
            return output
    
    # Create test data
    from torch.utils.data import Dataset, DataLoader
    
    class TestDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'src': torch.randint(0, 10, (5,)),
                'tgt_input': torch.randint(0, 5, (5,)),
                'tgt_output': torch.randint(0, 5, (5,))
            }
    
    dataset = TestDataset(100)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create trainer
    model = TestModel()
    trainer = TransformerTrainer(
        model=model,
        train_dataloader=dataloader,
        val_dataloader=dataloader,
        device="cpu",
        learning_rate=0.001,
        scheduler_type="none",
        loss_type="masked"
    )
    
    # Test training
    print("Testing training loop...")
    metrics = trainer.train(num_epochs=2, early_stopping_patience=5)
    
    print("Trainer test completed!")
    print(f"Final metrics: {metrics[-1]}")
