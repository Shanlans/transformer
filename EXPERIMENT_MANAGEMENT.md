# Experiment Management Guide

## Overview

The Experiment Management system allows you to create, manage, and compare different training configurations for your Transformer model. This system helps you organize experiments, track different parameter combinations, and easily reproduce results.

## Features

- ✅ **Create Experiments**: Create new training configurations with custom parameters
- ✅ **List Experiments**: View all available experiments in a formatted table
- ✅ **Load Experiments**: Load and display specific experiment configurations
- ✅ **Compare Experiments**: Side-by-side comparison of multiple experiments
- ✅ **Run Experiments**: Execute training with specific experiment configurations
- ✅ **Template System**: Pre-defined templates for common configurations

## Quick Start

### 1. List Available Experiments
```bash
python manage_experiments.py list
```

### 2. Create a New Experiment
```bash
# Small model for quick testing
python manage_experiments.py create \
    --name small_test \
    --description "Small model for quick testing" \
    --d_model 128 \
    --n_heads 4 \
    --epochs 5 \
    --batch_size 16

# Medium model for balanced performance
python manage_experiments.py create \
    --name medium_exp \
    --description "Medium model for balanced performance" \
    --d_model 256 \
    --n_heads 8 \
    --epochs 20 \
    --batch_size 32
```

### 3. Compare Experiments
```bash
python manage_experiments.py compare --names small_test medium_exp
```

### 4. Load Experiment Configuration
```bash
python manage_experiments.py load --name small_test
```

### 5. Run an Experiment
```bash
python manage_experiments.py run --name small_test
```

## Command Reference

### Create Experiment
```bash
python manage_experiments.py create [OPTIONS]

Required:
  --name NAME                    Experiment name

Optional:
  --description TEXT            Experiment description
  --base_config PATH            Base configuration file (default: training_config.json)
  
Model Parameters:
  --d_model INT                 Model dimension
  --n_heads INT                 Number of attention heads
  --n_encoder_layers INT        Number of encoder layers
  --n_decoder_layers INT        Number of decoder layers
  --d_ff INT                    Feed-forward dimension
  --dropout FLOAT               Dropout rate

Training Parameters:
  --epochs INT                   Number of epochs
  --learning_rate FLOAT         Learning rate
  --weight_decay FLOAT          Weight decay
  --gradient_clip_norm FLOAT    Gradient clipping norm

Data Parameters:
  --batch_size INT              Batch size
  --max_length INT              Maximum sequence length
  --train_split FLOAT           Training split ratio

Other Parameters:
  --optimizer_type TEXT         Optimizer type
  --scheduler_type TEXT         Scheduler type
  --loss_type TEXT              Loss function type
```

### List Experiments
```bash
python manage_experiments.py list
```
Shows all available experiments in a formatted table with:
- Experiment name
- Description
- Creation timestamp
- Model configuration (d_model/n_heads)
- Number of epochs

### Load Experiment
```bash
python manage_experiments.py load --name NAME
```
Displays the complete configuration for a specific experiment.

### Compare Experiments
```bash
python manage_experiments.py compare --names NAME1 NAME2 [NAME3 ...]
```
Shows side-by-side comparison of multiple experiments across all configuration categories.

### Run Experiment
```bash
python manage_experiments.py run --name NAME [--train_script PATH]
```
Executes training using the specified experiment configuration.

## Directory Structure

```
experiments/
├── configs/                   # Experiment configuration files
│   ├── small_test_20250924_075518.json
│   ├── medium_exp_20250924_075521.json
│   └── ...
└── results/                   # Experiment results (future use)
```

## Configuration File Format

Each experiment is saved as a JSON file with the following structure:

```json
{
  "experiment": {
    "name": "small_test",
    "description": "Small model for quick testing",
    "version": "1.0",
    "created_at": "2025-09-24T07:55:18.473343"
  },
  "model": {
    "d_model": 128,
    "n_heads": 4,
    "n_encoder_layers": 6,
    "n_decoder_layers": 6,
    "d_ff": 2048,
    "dropout": 0.1,
    "max_len": 5000
  },
  "training": {
    "epochs": 5,
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "gradient_clip_norm": 1.0,
    "early_stopping_patience": 10,
    "save_best_only": true
  },
  "data": {
    "train_data_path": "data/train.json",
    "max_length": 20,
    "train_split": 0.8,
    "batch_size": 16,
    "num_workers": 0,
    "shuffle": true
  }
  // ... other configuration sections
}
```

## Common Use Cases

### 1. Hyperparameter Tuning
Create multiple experiments with different hyperparameters:

```bash
# Test different learning rates
python manage_experiments.py create --name lr_001 --learning_rate 0.001
python manage_experiments.py create --name lr_0001 --learning_rate 0.0001
python manage_experiments.py create --name lr_00001 --learning_rate 0.00001

# Compare results
python manage_experiments.py compare --names lr_001 lr_0001 lr_00001
```

### 2. Model Size Comparison
Test different model architectures:

```bash
# Small model
python manage_experiments.py create --name small --d_model 128 --n_heads 4 --n_encoder_layers 2 --n_decoder_layers 2

# Medium model
python manage_experiments.py create --name medium --d_model 256 --n_heads 8 --n_encoder_layers 4 --n_decoder_layers 4

# Large model
python manage_experiments.py create --name large --d_model 512 --n_heads 8 --n_encoder_layers 6 --n_decoder_layers 6
```

### 3. Batch Size Optimization
Test different batch sizes:

```bash
python manage_experiments.py create --name batch_8 --batch_size 8
python manage_experiments.py create --name batch_16 --batch_size 16
python manage_experiments.py create --name batch_32 --batch_size 32
python manage_experiments.py create --name batch_64 --batch_size 64
```

### 4. Quick Testing vs Production Training
```bash
# Quick test
python manage_experiments.py create --name quick_test --epochs 2 --batch_size 8 --d_model 64

# Production training
python manage_experiments.py create --name production --epochs 100 --batch_size 32 --d_model 512
```

## Best Practices

### 1. Naming Conventions
Use descriptive names that indicate the purpose:
- `small_test`: Quick testing
- `medium_production`: Production training
- `lr_001`: Learning rate experiment
- `batch_32`: Batch size experiment

### 2. Documentation
Always provide meaningful descriptions:
```bash
python manage_experiments.py create \
    --name attention_heads_8 \
    --description "Testing 8 attention heads vs 4 heads for better parallelization"
```

### 3. Systematic Experiments
When doing hyperparameter tuning, create systematic experiments:
```bash
# Test different model dimensions
for d_model in 128 256 512; do
    python manage_experiments.py create \
        --name "d_model_${d_model}" \
        --description "Model dimension ${d_model}" \
        --d_model $d_model
done
```

### 4. Version Control
Keep experiment configurations in version control:
```bash
git add experiments/configs/
git commit -m "Add experiment configurations for hyperparameter tuning"
```

## Integration with Training

### Using Experiments in Training Scripts
The training system automatically uses experiment configurations:

```bash
# Run training with specific experiment
python manage_experiments.py run --name small_test

# Or manually specify config file
python train.py --config experiments/configs/small_test_20250924_075518.json
```

### Custom Training Scripts
You can integrate experiment management into custom training scripts:

```python
from src.utils.experiment_manager import ExperimentManager

# Load experiment
manager = ExperimentManager()
config_manager = manager.load_experiment("small_test")
config = config_manager.get_config()

# Use configuration in your training code
model = create_model(config.model)
trainer = create_trainer(config.training)
```

## Troubleshooting

### Common Issues

1. **Experiment not found**
   ```
   Error: Experiment 'experiment_name' not found
   ```
   Solution: Use `python manage_experiments.py list` to see available experiments

2. **Configuration file not found**
   ```
   FileNotFoundError: Configuration file not found
   ```
   Solution: Ensure `training_config.json` exists in the project root

3. **Invalid parameter values**
   ```
   ValueError: d_model must be positive
   ```
   Solution: Check parameter values and ensure they meet validation requirements

### Debug Mode
For debugging, you can examine the generated configuration files directly:

```bash
# View experiment configuration
cat experiments/configs/small_test_20250924_075518.json | jq .

# Compare two configurations
diff experiments/configs/small_test_*.json experiments/configs/medium_exp_*.json
```

## Future Enhancements

1. **Experiment Results Tracking**: Store training results and metrics
2. **Automated Hyperparameter Search**: Integration with optimization libraries
3. **Experiment Visualization**: Plot experiment comparisons
4. **Export/Import**: Share experiment configurations
5. **Template Library**: More pre-defined experiment templates

---

**Note**: This system is designed to make experiment management systematic and reproducible. Always document your experiments and keep configurations in version control for better collaboration and reproducibility.
