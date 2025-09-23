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

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.loss_functions import create_loss_function


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
    
    def _create_scheduler(self, scheduler_type: str, learning_rate: float) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if scheduler_type == "none":
            return None
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
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
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Early stopping patience: {early_stopping_patience}")
        
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
            metrics_history.append(epoch_metrics)
            
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Learning Rate: {train_metrics['learning_rate']:.6f}")
            print(f"  Epoch Time: {epoch_metrics['epoch_time']:.2f}s")
            
            # Check for best model
            if val_metrics and val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                
                if save_best_only:
                    self.save_model(os.path.join(self.save_dir, "best_model.pt"))
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
            self.save_model(save_path)
        
        # Save training history
        self.save_training_history(metrics_history)
        
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
    
    def load_model(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['current_epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Model loaded from {path}")
        print(f"Resumed from epoch {self.current_epoch + 1}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_training_history(self, metrics_history: List[Dict[str, float]]):
        """Save training history to JSON file."""
        history_path = os.path.join(self.save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        print(f"Training history saved to {history_path}")


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
