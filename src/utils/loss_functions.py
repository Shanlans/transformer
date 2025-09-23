"""
Loss Functions for Transformer Training
Implements label smoothing cross-entropy and masked cross-entropy losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross-Entropy Loss
    
    This loss function applies label smoothing to prevent overfitting
    and ignores padding tokens during loss calculation.
    
    Features:
    - Label smoothing technique to prevent overconfidence
    - Ignore padding tokens (PAD)
    - Calculate average loss over valid positions
    """
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        """
        Initialize the loss function.
        
        Args:
            smoothing: Label smoothing parameter (0.0 = no smoothing, 1.0 = uniform)
            ignore_index: Index to ignore in loss calculation (usually PAD token)
        """
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing cross-entropy loss.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target sequences [batch_size, seq_len]
        
        Returns:
            Scalar loss value
        """
        batch_size, seq_len, vocab_size = predictions.size()
        
        # Flatten predictions and targets for easier computation
        predictions = predictions.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
        targets = targets.view(-1)  # [batch_size * seq_len]
        
        # Create mask for non-padding tokens
        mask = (targets != self.ignore_index)
        
        # Apply label smoothing
        log_probs = F.log_softmax(predictions, dim=1)
        
        # Create smoothed targets
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (vocab_size - 1))  # Distribute smoothing uniformly
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)  # Set true class confidence
        
        # Compute loss
        loss = -smooth_targets * log_probs
        loss = loss.sum(dim=1)  # Sum over vocabulary dimension
        
        # Apply mask and compute mean
        loss = loss * mask.float()
        return loss.sum() / mask.sum().clamp(min=1)


class MaskedCrossEntropy(nn.Module):
    """
    Masked Cross-Entropy Loss
    
    This loss function computes cross-entropy loss while ignoring
    padding tokens using a mask.
    
    Features:
    - Ignore padding tokens using mask
    - Compute loss only on valid positions
    - Return average loss over valid tokens
    """
    
    def __init__(self, ignore_index: int = 0):
        """
        Initialize the loss function.
        
        Args:
            ignore_index: Index to ignore in loss calculation (usually PAD token)
        """
        super().__init__()
        self.ignore_index = ignore_index
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute masked cross-entropy loss.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target sequences [batch_size, seq_len]
        
        Returns:
            Scalar loss value
        """
        batch_size, seq_len, vocab_size = predictions.size()
        
        # Flatten predictions and targets
        predictions = predictions.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
        targets = targets.view(-1)  # [batch_size * seq_len]
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(predictions, targets, ignore_index=self.ignore_index, reduction='none')
        
        # Create mask for non-padding tokens
        mask = (targets != self.ignore_index)
        
        # Apply mask and compute mean
        loss = loss * mask.float()
        return loss.sum() / mask.sum().clamp(min=1)


def create_loss_function(loss_type: str = "masked", **kwargs) -> nn.Module:
    """
    Create a loss function based on the specified type.
    
    Args:
        loss_type: Type of loss function ("masked" or "label_smoothing")
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function instance
    """
    if loss_type == "masked":
        return MaskedCrossEntropy(**kwargs)
    elif loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test the loss functions
    batch_size, seq_len, vocab_size = 2, 5, 10
    predictions = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets[0, 3:] = 0  # Add some padding tokens
    
    print("Testing loss functions...")
    
    # Test masked cross-entropy
    masked_loss = MaskedCrossEntropy(ignore_index=0)
    loss1 = masked_loss(predictions, targets)
    print(f"Masked Cross-Entropy Loss: {loss1.item():.4f}")
    
    # Test label smoothing cross-entropy
    smooth_loss = LabelSmoothingCrossEntropy(smoothing=0.1, ignore_index=0)
    loss2 = smooth_loss(predictions, targets)
    print(f"Label Smoothing Cross-Entropy Loss: {loss2.item():.4f}")
    
    # Test factory function
    loss3 = create_loss_function("masked", ignore_index=0)
    loss4 = create_loss_function("label_smoothing", smoothing=0.1, ignore_index=0)
    print(f"Factory Masked Loss: {loss3(predictions, targets).item():.4f}")
    print(f"Factory Label Smoothing Loss: {loss4(predictions, targets).item():.4f}")
    
    print("Loss function tests completed!")
