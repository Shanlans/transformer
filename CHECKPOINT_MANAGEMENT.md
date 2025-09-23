# Checkpoint Management System

## Overview

The checkpoint management system provides a comprehensive solution for organizing, storing, and managing model checkpoints with timestamp-based organization and cleanup options.

## Features

### 🗂️ Timestamp-Based Organization
- **Automatic Run Directories**: Each training run gets a unique timestamped directory
- **Structured Storage**: `checkpoints/run_YYYYMMDD_HHMMSS/`
- **Metadata Tracking**: Each checkpoint includes epoch, loss, timestamp, and best status

### 💾 Smart Checkpoint Management
- **Automatic Naming**: Checkpoints are automatically named with epoch and status
- **Best Model Tracking**: Automatically identifies and marks best checkpoints
- **Metadata Storage**: JSON metadata files for easy inspection
- **Size Tracking**: Monitor disk usage and checkpoint sizes

### 🧹 Cleanup Options
- **Per-Run Cleanup**: Remove old checkpoints from specific runs
- **Global Cleanup**: Remove all runs and checkpoints
- **Flexible Retention**: Keep latest N checkpoints or runs
- **Size Reporting**: Track freed disk space

## Usage

### 1. Automatic Checkpoint Management

The `TransformerTrainer` automatically uses the checkpoint management system:

```python
from src.trainers.transformer_trainer import TransformerTrainer

# Trainer automatically creates timestamped run directory
trainer = TransformerTrainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    save_dir="checkpoints"  # Base directory for checkpoints
)

# Training automatically saves checkpoints with timestamps
trainer.train(num_epochs=50)
```

### 2. Manual Checkpoint Management

```python
from src.utils.checkpoint_manager import CheckpointManager

# Create checkpoint manager
manager = CheckpointManager(
    base_dir="checkpoints",
    max_checkpoints=10  # Keep last 10 checkpoints per run
)

# Create a new run
run_dir = manager.create_run_directory("my_experiment")

# Save checkpoint
checkpoint_path = manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=10,
    loss=0.5,
    is_best=True
)

# Load checkpoint
checkpoint_data = manager.load_checkpoint(
    checkpoint_path=checkpoint_path,
    model=model,
    optimizer=optimizer
)
```

### 3. Command-Line Management

Use the `manage_checkpoints.py` script for command-line operations:

```bash
# List all runs
python examples/manage_checkpoints.py list-runs

# List checkpoints in a specific run
python examples/manage_checkpoints.py list-checkpoints --run run_20250924_073606

# Show checkpoint statistics
python examples/manage_checkpoints.py stats

# Get the best checkpoint in a run
python examples/manage_checkpoints.py get-best --run run_20250924_073606

# Cleanup specific run (keep latest 5 checkpoints)
python examples/manage_checkpoints.py cleanup-run run_20250924_073606 --keep-latest 5

# Cleanup all runs (keep latest 3 runs, 5 checkpoints each)
python examples/manage_checkpoints.py cleanup-all --keep-latest-runs 3 --keep-latest-checkpoints 5

# Remove everything
python examples/manage_checkpoints.py cleanup-all
```

## Directory Structure

```
checkpoints/
├── run_20250924_073606/
│   ├── best_model_epoch_001.pt
│   ├── best_model_epoch_001_metadata.json
│   ├── best_model_epoch_002.pt
│   ├── best_model_epoch_002_metadata.json
│   └── training_history.json
├── run_20250924_080123/
│   ├── checkpoint_epoch_001.pt
│   ├── checkpoint_epoch_001_metadata.json
│   └── ...
└── run_20250924_081456/
    ├── final_model.pt
    ├── final_model_metadata.json
    └── ...
```

## Checkpoint Data Structure

Each checkpoint file contains:

```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 10,
    'loss': 0.5,
    'timestamp': '2025-09-24T07:36:07.909717',
    'is_best': True,
    'run_dir': '/path/to/run',
    'scheduler_state_dict': scheduler.state_dict(),  # Optional
    'metrics': {  # Optional
        'train_loss': 0.5,
        'val_loss': 0.4,
        'learning_rate': 0.001
    }
}
```

## Cleanup Strategies

### 1. Conservative Cleanup
Keep multiple runs and checkpoints for comparison:
```bash
python examples/manage_checkpoints.py cleanup-all --keep-latest-runs 5 --keep-latest-checkpoints 10
```

### 2. Moderate Cleanup
Keep recent runs but limit checkpoints:
```bash
python examples/manage_checkpoints.py cleanup-all --keep-latest-runs 3 --keep-latest-checkpoints 5
```

### 3. Aggressive Cleanup
Keep only the latest run and best checkpoints:
```bash
python examples/manage_checkpoints.py cleanup-all --keep-latest-runs 1 --keep-latest-checkpoints 3
```

### 4. Complete Cleanup
Remove everything:
```bash
python examples/manage_checkpoints.py cleanup-all
```

## Integration with Training

### Trainer Methods

The `TransformerTrainer` provides additional methods for checkpoint management:

```python
# List checkpoints in current run
checkpoints = trainer.list_checkpoints()

# List all runs
runs = trainer.list_runs()

# Get best checkpoint in current run
best_checkpoint = trainer.get_best_checkpoint()

# Cleanup current run (keep latest 5)
trainer.cleanup_checkpoints(keep_latest=5)

# Cleanup all runs
trainer.cleanup_all_runs(keep_latest_runs=3, keep_latest_checkpoints=5)
```

### Training Script Integration

The training script automatically uses the checkpoint management system:

```bash
# Train with automatic checkpoint management
python examples/train_transformer.py --epochs 50 --save_dir checkpoints

# Resume from specific checkpoint
python examples/train_transformer.py --resume checkpoints/run_20250924_073606/best_model_epoch_002.pt
```

## Best Practices

### 1. Naming Conventions
- Use descriptive run names: `experiment_transformer_large`
- Include hyperparameters: `lr_0.001_batch_32`
- Add experiment purpose: `baseline_comparison`

### 2. Cleanup Schedule
- **Daily**: Remove runs older than 7 days
- **Weekly**: Keep only best 3 runs per experiment
- **Monthly**: Archive important runs to external storage

### 3. Storage Management
- Monitor disk usage regularly
- Set up automatic cleanup scripts
- Use different base directories for different projects

### 4. Checkpoint Selection
- Always keep the best checkpoint from each run
- Keep final checkpoints for reproducibility
- Consider keeping intermediate checkpoints for analysis

## Troubleshooting

### Common Issues

1. **Disk Space Full**
   ```bash
   # Check current usage
   python examples/manage_checkpoints.py stats
   
   # Cleanup old runs
   python examples/manage_checkpoints.py cleanup-all --keep-latest-runs 2
   ```

2. **Too Many Checkpoints**
   ```bash
   # Cleanup specific run
   python examples/manage_checkpoints.py cleanup-run run_name --keep-latest 5
   ```

3. **Missing Checkpoints**
   ```bash
   # List all runs to find the right one
   python examples/manage_checkpoints.py list-runs
   
   # List checkpoints in specific run
   python examples/manage_checkpoints.py list-checkpoints --run run_name
   ```

### Performance Tips

1. **Use SSD Storage**: Faster checkpoint loading/saving
2. **Compress Old Checkpoints**: Use compression for archived checkpoints
3. **Monitor I/O**: Checkpoint operations can be I/O intensive
4. **Batch Operations**: Use cleanup scripts for bulk operations

## Advanced Usage

### Custom Checkpoint Manager

```python
from src.utils.checkpoint_manager import CheckpointManager

# Custom configuration
manager = CheckpointManager(
    base_dir="experiments/transformer",
    max_checkpoints=20  # Keep more checkpoints
)

# Custom run naming
run_dir = manager.create_run_directory("transformer_large_lr_0.001")

# Save with custom metadata
checkpoint_path = manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    loss=loss,
    metrics={
        'train_loss': train_loss,
        'val_loss': val_loss,
        'bleu_score': bleu_score,
        'perplexity': perplexity
    },
    checkpoint_name=f"epoch_{epoch:03d}_bleu_{bleu_score:.3f}",
    is_best=is_best
)
```

### Automated Cleanup Script

```python
#!/usr/bin/env python3
"""
Automated checkpoint cleanup script
Run this daily to manage disk space
"""

import os
import sys
sys.path.append('src')

from utils.checkpoint_manager import CheckpointManager

def daily_cleanup():
    """Daily cleanup routine."""
    manager = CheckpointManager("checkpoints")
    
    # Keep latest 5 runs, 10 checkpoints each
    stats = manager.cleanup_all_runs(
        keep_latest_runs=5,
        keep_latest_checkpoints=10
    )
    
    print(f"Daily cleanup completed:")
    print(f"  Runs removed: {stats['runs_removed']}")
    print(f"  Checkpoints removed: {stats['checkpoints_removed']}")
    print(f"  Space freed: {stats['total_size_freed_mb']:.2f} MB")

if __name__ == "__main__":
    daily_cleanup()
```

This checkpoint management system provides a robust, scalable solution for managing model checkpoints with full traceability and easy cleanup options.
