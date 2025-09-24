# Transformer Training Guide

## Overview

This guide explains how to train the Transformer model for English-German translation using the implemented training system.

## Quick Start

### 1. JSON Configuration Training (Recommended)

```bash
# Train with JSON configuration
python train.py

# Edit training_config.json to customize parameters
```

### 2. Command Line Training (Legacy)

```bash
# Train with command line parameters
python examples/train_transformer.py --epochs 10 --batch_size 16 --lr 0.0001

# Train a small model for quick testing
python examples/train_transformer.py \
    --epochs 5 \
    --batch_size 8 \
    --d_model 128 \
    --n_heads 4 \
    --n_encoder_layers 2 \
    --n_decoder_layers 2 \
    --d_ff 512
```

## Configuration

### JSON Configuration (Recommended)

The project uses JSON-based configuration for flexible parameter management. Edit `training_config.json`:

```json
{
  "experiment": {
    "name": "transformer_translation",
    "description": "English-German translation using Transformer",
    "version": "1.0"
  },
  "data": {
    "train_data_path": "data/train.json",
    "max_length": 20,
    "train_split": 0.8,
    "batch_size": 32
  },
  "model": {
    "d_model": 512,
    "n_heads": 8,
    "n_encoder_layers": 6,
    "n_decoder_layers": 6,
    "d_ff": 2048,
    "dropout": 0.1
  },
  "training": {
    "epochs": 50,
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "gradient_clip_norm": 1.0,
    "early_stopping_patience": 10
  }
}
```

### Command Line Parameters (Legacy)

#### Data Parameters
- `--data_path`: Path to training data (default: `data/train.json`)
- `--max_length`: Maximum sequence length (default: 20)
- `--train_split`: Fraction of data for training (default: 0.8)

### Model Parameters
- `--d_model`: Model dimension (default: 512)
- `--n_heads`: Number of attention heads (default: 8)
- `--n_encoder_layers`: Number of encoder layers (default: 6)
- `--n_decoder_layers`: Number of decoder layers (default: 6)
- `--d_ff`: Feed-forward dimension (default: 2048)
- `--dropout`: Dropout rate (default: 0.1)

### Training Parameters
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 0.0001)
- `--weight_decay`: Weight decay (default: 0.01)
- `--gradient_clip_norm`: Gradient clipping norm (default: 1.0)

### Training Strategy
- `--scheduler_type`: Learning rate scheduler (`none`, `step`, `cosine`)
- `--loss_type`: Loss function (`masked`, `label_smoothing`)
- `--early_stopping_patience`: Early stopping patience (default: 10)

### System Parameters
- `--device`: Device to use (`cpu`, `cuda`)
- `--save_dir`: Directory to save checkpoints (default: `checkpoints`)
- `--num_workers`: Number of worker processes (default: 0)
- `--resume`: Path to checkpoint to resume from

## Training Features

### 1. Comprehensive Training System
- **Training Loop**: Complete training with progress bars
- **Validation**: Automatic validation after each epoch
- **Metrics Tracking**: Training and validation loss tracking
- **Learning Rate Scheduling**: Support for step and cosine schedulers

### 2. Loss Functions
- **Masked Cross-Entropy**: Ignores padding tokens
- **Label Smoothing**: Prevents overconfidence with smoothing parameter

### 3. Optimization
- **AdamW Optimizer**: With configurable learning rate and weight decay
- **Gradient Clipping**: Prevents exploding gradients
- **Parameter Initialization**: Proper Xavier initialization

### 4. Checkpointing
- **Best Model Saving**: Automatically saves the best model
- **Training Resume**: Resume training from checkpoints
- **Configuration Saving**: Saves all training parameters

### 5. Early Stopping
- **Patience-based**: Stops training when validation loss doesn't improve
- **Best Model Tracking**: Keeps track of the best validation performance

## Output Files

After training, the following files are created in the save directory:

- `best_model.pt`: Best model checkpoint
- `final_model.pt`: Final model checkpoint
- `config.json`: Training configuration
- `training_history.json`: Training metrics history

## Example Training Sessions

### 1. Quick Test (2 epochs, small model)
```bash
python examples/train_transformer.py \
    --epochs 2 \
    --batch_size 8 \
    --d_model 128 \
    --n_heads 4 \
    --n_encoder_layers 2 \
    --n_decoder_layers 2 \
    --d_ff 512
```

### 2. Standard Training (50 epochs, full model)
```bash
python examples/train_transformer.py \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0001 \
    --scheduler_type cosine \
    --loss_type label_smoothing \
    --early_stopping_patience 15
```

### 3. Resume Training
```bash
python examples/train_transformer.py \
    --resume checkpoints/best_model.pt \
    --epochs 100
```

## Monitoring Training

### 1. Real-time Progress
- Progress bars show current batch loss and average loss
- Epoch summaries display training and validation metrics
- Learning rate tracking

### 2. Training History
- All metrics are saved to `training_history.json`
- Includes epoch time, losses, and learning rates
- Can be used for plotting training curves

### 3. Model Information
- Parameter count and model size
- Vocabulary sizes
- Dataset split information

## Tips for Better Training

### 1. Model Size
- Start with smaller models for testing
- Increase model size gradually
- Monitor GPU memory usage

### 2. Learning Rate
- Start with 0.0001 for AdamW
- Use learning rate scheduling for long training
- Monitor loss curves for optimal learning rate

### 3. Batch Size
- Larger batch sizes for better gradient estimates
- Adjust based on available memory
- Consider gradient accumulation for very large batches

### 4. Early Stopping
- Set appropriate patience based on dataset size
- Monitor validation loss for overfitting
- Use best model for inference

## Troubleshooting

### 1. Out of Memory
- Reduce batch size
- Reduce model dimensions
- Use gradient accumulation

### 2. Slow Training
- Increase batch size if memory allows
- Use multiple workers for data loading
- Consider mixed precision training

### 3. Poor Convergence
- Check learning rate
- Verify data preprocessing
- Monitor gradient norms

## Next Steps

After training, you can:
1. **Evaluate the model** using BLEU scores
2. **Generate translations** with the trained model
3. **Fine-tune** on specific domains
4. **Deploy** the model for inference

The training system provides a solid foundation for Transformer model training with comprehensive features and monitoring capabilities.
