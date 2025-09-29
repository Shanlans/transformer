# Unified Training Guide

This guide covers the unified training system with a single entry point (`run.py`) that provides intelligent resource selection and comprehensive training management.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Unified Entry Point](#unified-entry-point)
3. [Resource Selection](#resource-selection)
4. [Experiment Management](#experiment-management)
5. [Training Operations](#training-operations)
6. [Resume Training](#resume-training)
7. [Checkpoint Management](#checkpoint-management)
8. [Colab Automation](#colab-automation)
9. [Advanced Usage](#advanced-usage)

## Quick Start

The unified training system provides a single entry point for all training operations:

```bash
# List all experiments
python run.py --list-experiments

# Train with default configuration
python run.py --train --use-default

# Train with specific experiment
python run.py --train --experiment my_experiment

# Create new experiment
python run.py --create-experiment --name my_exp --description "My experiment"

# Use Colab automation (recommended for cloud training)
python run.py --colab-automation
```

## Unified Entry Point

### `run.py` - Single Entry Point

The `run.py` script is the unified entry point that provides:

- **Intelligent Resource Selection**: Automatically selects the best available resource (cloud GPU → local GPU → local CPU)
- **Experiment Management**: Create, list, and manage experiments
- **Training Operations**: Start training with various configurations
- **Resume Training**: Resume from checkpoints with hyperparameter validation
- **Checkpoint Management**: List and cleanup checkpoints

### Command Structure

```bash
python run.py [ACTION] [OPTIONS]
```

**Actions:**
- `--list-experiments` - List all experiments
- `--show-experiment` - Show experiment details
- `--create-experiment` - Create new experiment
- `--train` - Start training
- `--list-checkpoints` - List checkpoints
- `--create-resume` - Create resume configuration
- `--cleanup` - Cleanup old checkpoints

## Resource Selection

### Automatic Resource Detection

The system automatically detects and prioritizes resources:

1. **Cloud GPU (ColabCode)** - Highest priority
2. **Local GPU** - Medium priority
3. **Local CPU** - Fallback option

### Resource Status Display

```bash
python run.py --train --experiment my_experiment
```

**Output:**
```
🚀 Unified Trainer initialized!
📊 Available resources:
   ✅ COLABCODE: ColabCode SSH connection available
   ❌ LOCAL_GPU: No local GPU available
   ✅ LOCAL_CPU: Local CPU: 8 cores

================================================================================
🚀 TRAINING RESOURCE STATUS
================================================================================
☁️  ENVIRONMENT: CLOUD GPU (Google Colab)
🖥️  DEVICE: CUDA
📊 RESOURCE: ColabCode SSH Connection
⚡ GPU: Free Google Colab GPU
🌐 LOCATION: Remote Cloud Server
================================================================================
✅ Training will run on: COLABCODE
================================================================================
```

### Force Resource Selection

```bash
# Force local training
python run.py --train --experiment my_experiment --force-local

# Force cloud training
python run.py --train --experiment my_experiment --force-cloud
```

## Experiment Management

### List Experiments

```bash
python run.py --list-experiments
```

**Output:**
```
================================================================================
AVAILABLE EXPERIMENTS
================================================================================
📁 test_experiment
   Description: Test experiment for unified management
   Created: 2025-09-25T07:44:00.066135
   Version: 1.0
   Timestamp ID: 20250925_074400
```

### Show Experiment Details

```bash
python run.py --show-experiment test_experiment
```

**Output:**
```
📋 Experiment Details: test_experiment
============================================================
Name: test_experiment
Description: Test experiment for unified management
Version: 1.0
Created: 2025-09-25T07:44:00.066135
Timestamp ID: 20250925_074400

Model Configuration:
  Architecture: Transformer
  Model dimension: 512
  Attention heads: 8
  Encoder layers: 6
  Decoder layers: 6

Training Configuration:
  Epochs: 2
  Batch size: 32
  Learning rate: 0.0001
  Early stopping patience: 10
```

### Create New Experiment

```bash
python run.py --create-experiment --name my_experiment --description "My custom experiment"
```

## Training Operations

### Training with Default Configuration

```bash
python run.py --train --use-default
```

### Training with Specific Experiment

```bash
python run.py --train --experiment my_experiment
```

### Training with Specific Config File

```bash
python run.py --train --config experiments/configs/my_experiment.json
```

### Training with Resume Configuration

```bash
python run.py --train --config experiments/configs/resume_my_experiment_20250925_123456.json
```

## Resume Training

### List Checkpoints

```bash
# List all checkpoints
python run.py --list-checkpoints

# List checkpoints for specific experiment
python run.py --list-checkpoints --experiment my_experiment
```

**Output:**
```
================================================================================
AVAILABLE CHECKPOINTS
================================================================================

📁 run_20250925_070047
   Path: checkpoints/run_20250925_070047
   Timestamp: unknown
   Checkpoint Files:
     - best_model_epoch_005.pt (88.87 MB)
     - best_model_epoch_001.pt (88.87 MB)
     - best_model_epoch_002.pt (88.87 MB)
     - best_model_epoch_003.pt (88.87 MB)
     - best_model_epoch_004.pt (88.87 MB)
```

### Create Resume Configuration

```bash
python run.py --create-resume \
  --experiment my_experiment \
  --checkpoint checkpoints/run_20250925_070047/best_model_epoch_005.pt \
  --overrides '{"training": {"learning_rate": 0.0002, "epochs": 10}}'
```

**Output:**
```
🔄 Resuming training from checkpoint...
   Experiment: my_experiment
   Checkpoint: checkpoints/run_20250925_070047/best_model_epoch_005.pt

================================================================================
CHECKPOINT STATUS
================================================================================
Checkpoint: checkpoints/run_20250925_070047/best_model_epoch_005.pt
📊 Current Status:
   Epoch: 5
   Loss: 1.5982661545276642
   Timestamp: 2025-09-25T07:01:03.187162
   Is Best Model: True

📈 Training History:
   Total Epochs Trained: 5
   Best Validation Loss: 1.5982661545276642
   Final Training Loss: 1.6031927168369293
   Final Validation Loss: 1.5982661545276642
   Recent Trend: 📈 Improving
================================================================================

🔧 Hyperparameter overrides: {'training': {'learning_rate': 0.0002, 'epochs': 10}}

================================================================================
HYPERPARAMETER VALIDATION RESULTS
================================================================================
✅ VALIDATION PASSED: No structural changes detected

⚠️  NON-STRUCTURAL CHANGES (ALLOWED):
Section         Parameter            Original        New            
-----------------------------------------------------------------
training        learning_rate        0.0001          0.0002         
training        epochs               2               10             

⚠️  WARNINGS:
  - Non-structural parameter changed: training.learning_rate (0.0001 → 0.0002)
  - Non-structural parameter changed: training.epochs (2 → 10)
================================================================================

✅ Resume configuration created successfully!
   Config file: experiments/configs/resume_my_experiment_20250925_123456.json
   Location: experiments/configs/
   You can now use this config to resume training:
   python run.py --train --config experiments/configs/resume_my_experiment_20250925_123456.json
```

## Checkpoint Management

### Cleanup Old Checkpoints

```bash
python run.py --cleanup --keep-runs 3 --keep-checkpoints 5
```

**Output:**
```
✅ Checkpoint cleanup completed!
   Runs cleaned: 5
   Checkpoints cleaned: 23
   Space freed: 125.67 MB
```

## Colab Automation

The unified training system includes comprehensive Google Colab automation features that allow you to easily run training on Google's free GPU resources.

### Quick Colab Setup

```bash
# Create Colab automation notebook
python run.py --create-colab-notebook

# Run full Colab automation workflow
python run.py --colab-automation
```

### Colab Automation Features

- **One-Click Setup**: Automatically creates a complete Colab notebook
- **Automatic Installation**: Installs all dependencies automatically
- **GPU Detection**: Automatically detects and uses available GPU
- **Training Execution**: Runs training with optimal settings
- **Result Download**: Automatically downloads results to local machine
- **Training Summary**: Provides comprehensive training statistics

### Generated Files

The automation creates several files:

- `colab_automation.ipynb`: Ready-to-use Colab notebook
- `upload_to_colab.sh`: Script for uploading code to Colab
- `download_from_colab.sh`: Script for downloading results
- `COLAB_AUTOMATION_GUIDE.md`: Detailed automation guide

### Usage Workflow

1. **Generate Files**: Run `python run.py --colab-automation`
2. **Open Colab**: Go to [Google Colab](https://colab.research.google.com/)
3. **Upload Notebook**: Upload `colab_automation.ipynb`
4. **Run All**: Click "Run All" to execute the entire workflow
5. **Download Results**: Results are automatically downloaded when training completes

### Benefits

- **Free GPU Access**: Use Google's free GPU resources
- **No Local Setup**: No need to install CUDA or GPU drivers locally
- **Automatic Management**: Handles all setup and cleanup automatically
- **Result Synchronization**: Automatically downloads training results
- **Reproducible**: Consistent environment across runs

### Troubleshooting

- **Session Timeout**: Colab sessions have 12-hour limits
- **GPU Availability**: Free users have limited GPU access
- **Memory Limits**: Large models may exceed Colab memory limits
- **Network Issues**: Check internet connection for downloads

For detailed information, see [Colab Automation Guide](COLAB_AUTOMATION_GUIDE.md).

## Advanced Usage

### Complete Workflow Example

```bash
# 1. List available experiments
python run.py --list-experiments

# 2. Show experiment details
python run.py --show-experiment my_experiment

# 3. Create new experiment
python run.py --create-experiment --name new_exp --description "New experiment"

# 4. Train with the new experiment
python run.py --train --experiment new_exp

# 5. List checkpoints after training
python run.py --list-checkpoints --experiment new_exp

# 6. Create resume configuration with modified hyperparameters
python run.py --create-resume \
  --experiment new_exp \
  --checkpoint checkpoints/run_20250925_123456/best_model_epoch_003.pt \
  --overrides '{"training": {"learning_rate": 0.0003, "epochs": 8}}'

# 7. Resume training with the new configuration
python run.py --train --config experiments/configs/resume_new_exp_20250925_123456.json

# 8. Cleanup old checkpoints when done
python run.py --cleanup --keep-runs 2 --keep-checkpoints 3
```

### Resource-Specific Training

```bash
# Force local GPU training
python run.py --train --experiment my_experiment --force-local

# Force cloud GPU training
python run.py --train --experiment my_experiment --force-cloud

# Use default configuration with local training
python run.py --train --use-default --force-local
```

### Hyperparameter Override Examples

```bash
# Change learning rate only
python run.py --create-resume \
  --experiment my_experiment \
  --checkpoint checkpoints/run_xxx/best_model.pt \
  --overrides '{"training": {"learning_rate": 0.0002}}'

# Change multiple training parameters
python run.py --create-resume \
  --experiment my_experiment \
  --checkpoint checkpoints/run_xxx/best_model.pt \
  --overrides '{"training": {"learning_rate": 0.0003, "epochs": 10, "early_stopping_patience": 15}}'

# Change optimizer parameters
python run.py --create-resume \
  --experiment my_experiment \
  --checkpoint checkpoints/run_xxx/best_model.pt \
  --overrides '{"optimizer": {"betas": [0.9, 0.99]}}'

# Change loss function parameters
python run.py --create-resume \
  --experiment my_experiment \
  --checkpoint checkpoints/run_xxx/best_model.pt \
  --overrides '{"loss": {"smoothing": 0.2}}'
```

## Integration with Management Tools

The unified training system integrates seamlessly with the management tools:

```bash
# Use manage.py for experiment management
python manage.py experiments list
python manage.py experiments show my_experiment
python manage.py experiments create --name my_exp --description "My experiment"

# Use run.py for training operations
python run.py --train --experiment my_experiment
python run.py --list-checkpoints
python run.py --create-resume --experiment my_experiment --checkpoint checkpoints/run_xxx/best_model.pt
```

## Error Handling

### Common Issues and Solutions

1. **No Experiments Found**
   ```
   No experiments found.
   💡 Tip: Create your first experiment with:
      python run.py --create-experiment --name my_experiment --description "My first experiment"
   ```
   **Solution:** Create your first experiment or use default configuration.

2. **Experiment Not Found**
   ```
   ❌ Error loading experiment: Experiment 'my_experiment' not found
   ```
   **Solution:** Use `python run.py --list-experiments` to see available experiments.

3. **Checkpoint Not Found**
   ```
   ❌ Error creating resume configuration: Checkpoint file not found
   ```
   **Solution:** Use `python run.py --list-checkpoints` to see available checkpoints.

4. **Invalid JSON in Overrides**
   ```
   ❌ Error creating resume configuration: Expecting ',' delimiter: line 1 column 25 (char 24)
   ```
   **Solution:** Ensure JSON is properly formatted. Use single quotes around the JSON string.

5. **Structural Parameter Changes**
   ```
   ❌ Error: Cannot resume training: structural parameters have changed
   ```
   **Solution:** Only modify non-structural parameters (training, optimizer, scheduler, loss, system, visualization, evaluation, logging).

## Best Practices

1. **Resource Selection**: Let the system auto-select resources unless you have specific requirements
2. **Experiment Naming**: Use descriptive names for experiments
3. **Checkpoint Management**: Regularly cleanup old checkpoints to manage disk space
4. **Hyperparameter Validation**: Always validate changes before creating resume configurations
5. **Backup Important Checkpoints**: Keep backups of important model checkpoints
6. **Use Default Configuration**: Start with default configuration for quick testing

## Troubleshooting

### Debug Mode

For detailed debugging information, you can modify the training script to include verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### File Locations

- **Experiments**: `experiments/configs/`
- **Checkpoints**: `checkpoints/`
- **Resume Configurations**: `experiments/configs/resume_*.json`
- **Default Configuration**: `training_config.json`

### Performance Tips

1. Use specific experiment names when listing checkpoints
2. Clean up old checkpoints regularly
3. Use absolute paths for checkpoint files
4. Monitor disk space usage
5. Use cloud resources for large models when available
