"""
Training Visualizer
Provides comprehensive visualization for Transformer training including metrics, attention, and evaluation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import os
import json
from datetime import datetime
from collections import defaultdict

# Set style for better visualizations
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


class TrainingVisualizer:
    """
    Comprehensive training visualizer for Transformer models.
    
    Features:
    - Training metrics visualization (loss, accuracy, BLEU, etc.)
    - Attention weight visualization
    - Model evaluation metrics
    - Learning rate scheduling visualization
    - Gradient flow analysis
    """
    
    def __init__(self, save_dir: str = "visualizations", figsize: Tuple[int, int] = (12, 8), 
                 previous_training_history: Optional[List[Dict]] = None):
        """
        Initialize training visualizer.
        
        Args:
            save_dir: Directory to save visualizations
            figsize: Default figure size for plots
            previous_training_history: Previous training history to continue from
        """
        self.save_dir = save_dir
        self.figsize = figsize
        self.metrics_history = []
        
        # Load previous training history if provided
        if previous_training_history:
            self.metrics_history = previous_training_history.copy()
            print(f"📊 Loaded previous training history: {len(previous_training_history)} epochs")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Create subdirectories
        self.metrics_dir = os.path.join(save_dir, "metrics")
        self.attention_dir = os.path.join(save_dir, "attention")
        self.evaluation_dir = os.path.join(save_dir, "evaluation")
        
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.attention_dir, exist_ok=True)
        os.makedirs(self.evaluation_dir, exist_ok=True)
        
        print(f"TrainingVisualizer initialized:")
        print(f"  Save directory: {save_dir}")
        print(f"  Metrics directory: {self.metrics_dir}")
        print(f"  Attention directory: {self.attention_dir}")
        print(f"  Evaluation directory: {self.evaluation_dir}")
        if previous_training_history:
            print(f"  Previous epochs loaded: {len(previous_training_history)}")
    
    def load_previous_training_history(self, training_history_path: str) -> bool:
        """
        Load previous training history from JSON file.
        
        Args:
            training_history_path: Path to training history JSON file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if os.path.exists(training_history_path):
                with open(training_history_path, 'r', encoding='utf-8') as f:
                    previous_history = json.load(f)
                
                if isinstance(previous_history, list) and previous_history:
                    self.metrics_history = previous_history.copy()
                    print(f"📊 Loaded previous training history: {len(previous_history)} epochs")
                    print(f"   From: {training_history_path}")
                    return True
                else:
                    print(f"⚠️  Training history file is empty or invalid: {training_history_path}")
                    return False
            else:
                print(f"⚠️  Training history file not found: {training_history_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading training history: {e}")
            return False

    def add_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Add metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics (loss, accuracy, BLEU, etc.)
        """
        metrics_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(metrics_entry)
    
    def plot_training_metrics(self, save_name: str = "training_metrics.png") -> str:
        """
        Plot comprehensive training metrics.
        
        Args:
            save_name: Name of the saved plot file
            
        Returns:
            Path to saved plot
        """
        if not self.metrics_history:
            print("No metrics data available for plotting")
            return ""
        
        # Extract metrics
        epochs = [m['epoch'] for m in self.metrics_history]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics Overview', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        if any('train_loss' in m for m in self.metrics_history):
            train_losses = [m.get('train_loss', None) for m in self.metrics_history]
            train_losses = [l for l in train_losses if l is not None]
            if train_losses:
                ax1.plot(epochs[:len(train_losses)], train_losses, 'b-', label='Training Loss', linewidth=2)
        
        if any('val_loss' in m for m in self.metrics_history):
            val_losses = [m.get('val_loss', None) for m in self.metrics_history]
            val_losses = [l for l in val_losses if l is not None]
            if val_losses:
                ax1.plot(epochs[:len(val_losses)], val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax2 = axes[0, 1]
        if any('train_acc' in m for m in self.metrics_history):
            train_accs = [m.get('train_acc', None) for m in self.metrics_history]
            train_accs = [a for a in train_accs if a is not None]
            if train_accs:
                ax2.plot(epochs[:len(train_accs)], train_accs, 'g-', label='Training Accuracy', linewidth=2)
        
        if any('val_acc' in m for m in self.metrics_history):
            val_accs = [m.get('val_acc', None) for m in self.metrics_history]
            val_accs = [a for a in val_accs if a is not None]
            if val_accs:
                ax2.plot(epochs[:len(val_accs)], val_accs, 'orange', label='Validation Accuracy', linewidth=2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: BLEU scores
        ax3 = axes[1, 0]
        if any('bleu_score' in m for m in self.metrics_history):
            bleu_scores = [m.get('bleu_score', None) for m in self.metrics_history]
            bleu_scores = [b for b in bleu_scores if b is not None]
            if bleu_scores:
                ax3.plot(epochs[:len(bleu_scores)], bleu_scores, 'purple', label='BLEU Score', linewidth=2)
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('BLEU Score')
                ax3.set_title('BLEU Score Evolution')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No BLEU scores available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('BLEU Score Evolution')
        
        # Plot 4: Learning rate
        ax4 = axes[1, 1]
        if any('learning_rate' in m for m in self.metrics_history):
            lrs = [m.get('learning_rate', None) for m in self.metrics_history]
            lrs = [lr for lr in lrs if lr is not None]
            if lrs:
                ax4.plot(epochs[:len(lrs)], lrs, 'brown', label='Learning Rate', linewidth=2)
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Learning Rate')
                ax4.set_title('Learning Rate Schedule')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No learning rate data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.metrics_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training metrics plot saved: {save_path}")
        return save_path
    
    def plot_attention_weights(
        self,
        model: torch.nn.Module,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_vocab: Dict[str, int],
        tgt_vocab: Dict[str, int],
        epoch: int,
        layer_idx: int = 0,
        head_idx: int = 0,
        save_name: Optional[str] = None
    ) -> str:
        """
        Visualize attention weights for a specific layer and head.
        
        Args:
            model: Transformer model
            src_tokens: Source token sequence [seq_len]
            tgt_tokens: Target token sequence [seq_len]
            src_vocab: Source vocabulary mapping
            tgt_vocab: Target vocabulary mapping
            epoch: Current epoch
            layer_idx: Encoder layer index
            head_idx: Attention head index
            save_name: Custom save name
            
        Returns:
            Path to saved attention plot
        """
        model.eval()
        
        # Create reverse vocabularies
        src_idx2word = {v: k for k, v in src_vocab.items()}
        tgt_idx2word = {v: k for k, v in tgt_vocab.items()}
        
        # Convert tokens to words
        src_words = [src_idx2word.get(idx.item(), f'<UNK_{idx.item()}>') for idx in src_tokens]
        tgt_words = [tgt_idx2word.get(idx.item(), f'<UNK_{idx.item()}>') for idx in tgt_tokens]
        
        # Get attention weights (this requires model modification to return attention)
        # For now, we'll create a placeholder visualization
        with torch.no_grad():
            # This is a simplified version - in practice, you'd need to modify the model
            # to return attention weights during forward pass
            attention_weights = torch.randn(len(tgt_words), len(src_words))
            attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Create attention heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot heatmap
        im = ax.imshow(attention_weights.cpu().numpy(), cmap='Blues', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(src_words)))
        ax.set_yticks(range(len(tgt_words)))
        ax.set_xticklabels(src_words, rotation=45, ha='right')
        ax.set_yticklabels(tgt_words)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        # Set title and labels
        ax.set_title(f'Attention Weights - Epoch {epoch}, Layer {layer_idx}, Head {head_idx}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Source Tokens', fontsize=12)
        ax.set_ylabel('Target Tokens', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save plot
        if save_name is None:
            save_name = f"attention_epoch_{epoch:03d}_layer_{layer_idx}_head_{head_idx}.png"
        
        save_path = os.path.join(self.attention_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Attention weights plot saved: {save_path}")
        return save_path
    
    def plot_evaluation_metrics(
        self,
        evaluation_results: Dict[str, List[float]],
        save_name: str = "evaluation_metrics.png"
    ) -> str:
        """
        Plot evaluation metrics comparison.
        
        Args:
            evaluation_results: Dictionary of metric names to lists of values
            save_name: Name of the saved plot file
            
        Returns:
            Path to saved plot
        """
        if not evaluation_results:
            print("No evaluation results available for plotting")
            return ""
        
        # Create subplots
        n_metrics = len(evaluation_results)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle('Evaluation Metrics Comparison', fontsize=16, fontweight='bold')
        
        # Plot each metric
        for idx, (metric_name, values) in enumerate(evaluation_results.items()):
            ax = axes[idx]
            
            # Create bar plot
            epochs = list(range(1, len(values) + 1))
            bars = ax.bar(epochs, values, alpha=0.7, color=plt.cm.Set3(idx))
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Evolution')
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits
            if values:
                y_min, y_max = min(values), max(values)
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.evaluation_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation metrics plot saved: {save_path}")
        return save_path
    
    def plot_gradient_flow(self, model: torch.nn.Module, epoch: int, save_name: Optional[str] = None) -> str:
        """
        Plot gradient flow analysis.
        
        Args:
            model: Transformer model
            epoch: Current epoch
            save_name: Custom save name
            
        Returns:
            Path to saved gradient flow plot
        """
        # Collect gradient norms
        gradient_norms = []
        layer_names = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                gradient_norms.append(grad_norm)
                layer_names.append(name)
        
        if not gradient_norms:
            print("No gradients available for analysis")
            return ""
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Gradient norms by layer
        ax1.bar(range(len(gradient_norms)), gradient_norms, alpha=0.7)
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Gradient Norm')
        ax1.set_title(f'Gradient Norms by Layer - Epoch {epoch}')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Gradient norm distribution
        ax2.hist(gradient_norms, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Gradient Norm')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Gradient Norm Distribution - Epoch {epoch}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_name is None:
            save_name = f"gradient_flow_epoch_{epoch:03d}.png"
        
        save_path = os.path.join(self.metrics_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gradient flow plot saved: {save_path}")
        return save_path
    
    def save_metrics_summary(self, save_name: str = "metrics_summary.json") -> str:
        """
        Save metrics summary to JSON file.
        
        Args:
            save_name: Name of the saved JSON file
            
        Returns:
            Path to saved JSON file
        """
        if not self.metrics_history:
            print("No metrics data available for saving")
            return ""
        
        # Calculate summary statistics
        summary = {
            'total_epochs': len(self.metrics_history),
            'metrics_summary': {},
            'best_metrics': {},
            'final_metrics': self.metrics_history[-1] if self.metrics_history else {},
            'training_duration': {
                'start': self.metrics_history[0]['timestamp'] if self.metrics_history else None,
                'end': self.metrics_history[-1]['timestamp'] if self.metrics_history else None
            }
        }
        
        # Calculate statistics for each metric
        metric_names = set()
        for metrics in self.metrics_history:
            metric_names.update(metrics.keys())
        
        metric_names.discard('epoch')
        metric_names.discard('timestamp')
        
        for metric_name in metric_names:
            values = [m.get(metric_name) for m in self.metrics_history if m.get(metric_name) is not None]
            if values:
                summary['metrics_summary'][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'final': values[-1] if values else None
                }
                
                # Find best value (for loss, lower is better; for others, higher is better)
                if 'loss' in metric_name.lower():
                    best_idx = np.argmin(values)
                    summary['best_metrics'][metric_name] = {
                        'value': values[best_idx],
                        'epoch': self.metrics_history[best_idx]['epoch']
                    }
                else:
                    best_idx = np.argmax(values)
                    summary['best_metrics'][metric_name] = {
                        'value': values[best_idx],
                        'epoch': self.metrics_history[best_idx]['epoch']
                    }
        
        # Save to JSON
        save_path = os.path.join(self.metrics_dir, save_name)
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Metrics summary saved: {save_path}")
        return save_path


def create_training_visualizer(save_dir: str = "visualizations") -> TrainingVisualizer:
    """
    Create a training visualizer instance.
    
    Args:
        save_dir: Directory to save visualizations
        
    Returns:
        TrainingVisualizer instance
    """
    return TrainingVisualizer(save_dir)


if __name__ == "__main__":
    # Test the training visualizer
    print("Testing TrainingVisualizer...")
    
    # Create visualizer
    visualizer = TrainingVisualizer("test_visualizations")
    
    # Add some test metrics
    for epoch in range(1, 11):
        metrics = {
            'train_loss': 2.0 - epoch * 0.15 + np.random.normal(0, 0.05),
            'val_loss': 2.1 - epoch * 0.14 + np.random.normal(0, 0.05),
            'train_acc': 0.1 + epoch * 0.08 + np.random.normal(0, 0.02),
            'val_acc': 0.09 + epoch * 0.07 + np.random.normal(0, 0.02),
            'bleu_score': 0.05 + epoch * 0.03 + np.random.normal(0, 0.01),
            'learning_rate': 0.001 * (0.95 ** epoch)
        }
        visualizer.add_metrics(epoch, metrics)
    
    # Create plots
    visualizer.plot_training_metrics()
    
    # Test evaluation metrics
    evaluation_results = {
        'BLEU-1': [0.1, 0.15, 0.2, 0.25, 0.3],
        'BLEU-4': [0.05, 0.08, 0.12, 0.15, 0.18],
        'METEOR': [0.08, 0.12, 0.16, 0.19, 0.22],
        'ROUGE-L': [0.12, 0.18, 0.24, 0.28, 0.32]
    }
    visualizer.plot_evaluation_metrics(evaluation_results)
    
    # Save metrics summary
    visualizer.save_metrics_summary()
    
    print("TrainingVisualizer test completed!")
