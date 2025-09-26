# Transformer Project

A comprehensive PyTorch implementation of the Transformer model for machine translation tasks with advanced experiment management, configuration versioning, and unified training system.

## 📁 Project Structure

```
transformer/
├── src/                        # Source code
│   ├── models/                 # Model implementations
│   │   └── transformer.py     # Complete Transformer model
│   ├── datasets/              # Dataset implementations
│   │   └── translation_dataset.py  # Translation dataset
│   ├── trainers/              # Training utilities
│   │   └── transformer_trainer.py  # Training manager
│   └── utils/                  # Utility modules
│       ├── config_manager.py  # Configuration management
│       ├── checkpoint_manager.py  # Checkpoint management
│       ├── config_version_manager.py  # Configuration versioning
│       ├── experiment_manager.py  # Experiment management
│       ├── loss_functions.py  # Loss functions
│       ├── training_visualizer.py  # Training visualization
│       └── evaluation_metrics.py  # Evaluation metrics
├── tests/                      # Test files
│   ├── functional/            # Functional tests
│   │   ├── test_transformer.py
│   │   ├── test_positional_encoding.py
│   │   └── test_multihead_attention.py
│   ├── visualizations/        # Test visualizations
│   └── README_testing.md     # Testing documentation
├── examples/                  # Example scripts
│   └── create_sample_dataset.py
├── docs/                      # Documentation
│   └── DEVELOPMENT_GUIDE.md
├── data/                      # Sample data
│   ├── train.json
│   ├── train.en
│   └── train.de
├── checkpoints/               # Model checkpoints
├── experiments/               # Experiment configurations
│   ├── configs/              # Experiment config files
│   └── results/              # Experiment results
├── config_history/           # Configuration version history
├── train.py                   # Main training script
├── training_config.json       # Training configuration
├── run.py                     # Unified training and management entry point
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment specification
├── install.py                 # Automated installation script
├── setup.sh                   # Shell-based installation script
├── UNIFIED_TRAINING_GUIDE.md  # Complete unified training system documentation
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Installation Options

#### Option A: Automated Installation (Recommended)
```bash
# Run the automated installation script
python install.py
```

#### Option B: Conda Environment (Recommended for ML/DL)
```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate torch2.5
```

#### Option C: Manual Installation
```bash
# Install PyTorch (choose one based on your system)
# For CUDA support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

#### Option D: Shell Script (Linux/macOS)
```bash
# Make the script executable
chmod +x setup.sh

# Run the setup script
./setup.sh

# Or with custom options
./setup.sh --env-name my_env --python-version 3.9
```

### 2. Run Training
```bash
# Train with default configuration
python run.py --train --use-default

# Train with specific experiment
python run.py --train --experiment my_experiment

# Train with specific config file
python run.py --train --config experiments/configs/my_experiment.json

# Force local training
python run.py --train --experiment my_experiment --force-local

# Force cloud training
python run.py --train --experiment my_experiment --force-cloud
```

### 3. Unified Training and Management System

#### 🎯 **Core Training Commands**
```bash
# Train with default configuration
python run.py --train --use-default

# Train with specific experiment
python run.py --train --experiment my_experiment

# Train with specific config file
python run.py --train --config experiments/configs/my_experiment.json

# Force local training (CPU/GPU)
python run.py --train --experiment my_experiment --force-local

# Force cloud training (simulated Colab environment)
python run.py --train --experiment my_experiment --force-cloud
```

#### 📋 **Experiment Management**
```bash
# List all experiments
python run.py --list-experiments

# Show experiment details
python run.py --show-experiment small_test

# Create new experiment
python run.py --create-experiment --name small_test --description "Small model for testing"

# Create derived experiment from existing one
python run.py --create-derived-experiment --source-experiment cloud_training_20250926_082552 --name local_enhanced --description "Local enhanced training derived from cloud" --new-environment local_gpu --overrides '{"training": {"epochs": 5, "learning_rate": 0.0002}}'
```

#### 💾 **Checkpoint Management**
```bash
# List all checkpoints
python run.py --list-checkpoints

# List checkpoints for specific experiment
python run.py --list-checkpoints --experiment small_test

# Create resume config with modified hyperparameters
python run.py --create-resume --experiment small_test --checkpoint checkpoints/run_20250925_070047/best_model_epoch_005.pt --overrides '{"training": {"learning_rate": 0.0002, "epochs": 10}}'

# Validate hyperparameter changes
python run.py --validate-changes --experiment small_test --overrides '{"training": {"learning_rate": 0.0002}}'

# Resume training with new config
python run.py --train --config experiments/configs/resume_small_test_20250925_073041.json
```

#### 🧹 **System Maintenance**
```bash
# List configuration versions
python run.py --list-config-versions

# Cleanup old checkpoints
python run.py --cleanup --keep-runs 3 --keep-checkpoints 5

# Clean all generated files (experiments, checkpoints, config_history)
python run.py --cleanup-all
```


## 📖 Documentation

- **[Unified Training Guide](UNIFIED_TRAINING_GUIDE.md)** - Complete unified training and management system documentation
- **[Testing Guide](tests/README_testing.md)** - Testing documentation
- **[Development Guide](docs/DEVELOPMENT_GUIDE.md)** - Development workflow

## 🔧 Features

### Model Components
- ✅ **Positional Encoding** - Sine/cosine positional encodings
- ✅ **Multi-Head Attention** - Scaled dot-product attention
- ✅ **Feed-Forward Networks** - Position-wise feed-forward layers
- ✅ **Encoder/Decoder Layers** - Complete encoder and decoder stacks
- ✅ **Full Transformer** - Complete model with embedding layers

### Training System
- ✅ **Comprehensive Trainer** - Full training loop with validation
- ✅ **Loss Functions** - Masked cross-entropy and label smoothing
- ✅ **Optimizers** - AdamW with configurable parameters
- ✅ **Schedulers** - Step and cosine annealing learning rate schedules
- ✅ **Gradient Clipping** - Prevents gradient explosion
- ✅ **Early Stopping** - Prevents overfitting

### Data Management
- ✅ **Translation Dataset** - Flexible dataset for translation tasks
- ✅ **Vocabulary Building** - Automatic vocabulary construction
- ✅ **Data Loading** - Efficient data loading with batching
- ✅ **Tokenization** - Word-level tokenization with special tokens

### Visualization & Evaluation
- ✅ **Training Metrics** - Loss curves and learning rate plots
- ✅ **Gradient Flow** - Gradient magnitude analysis
- ✅ **Attention Visualization** - Attention weight heatmaps
- ✅ **Evaluation Metrics** - BLEU, METEOR, ROUGE-L, Exact Match

### Experiment Management
- ✅ **Configuration System** - JSON-based experiment configurations
- ✅ **Experiment Creation** - Easy creation of new experiment configs
- ✅ **Derived Experiments** - Create new experiments based on existing ones
- ✅ **Source Tracking** - Record source experiment info (timestamp, description, environment)
- ✅ **Environment Support** - Local CPU, Local GPU, Cloud (simulated Colab)
- ✅ **Version Control** - Track experiment history and changes
- ✅ **Timestamp Linkage** - Link experiment configs with checkpoint timestamps
- ✅ **Integrity Validation** - Prevent manual config modification
- ✅ **Parameter Overrides** - Modify specific parameters while preserving structure

### Checkpoint Management
- ✅ **Timestamped Runs** - Organized checkpoint storage
- ✅ **Metadata Tracking** - Comprehensive checkpoint metadata
- ✅ **Cleanup Tools** - Automated checkpoint cleanup
- ✅ **Model Resuming** - Resume training from checkpoints
- ✅ **Hyperparameter Validation** - Structural vs non-structural parameter validation
- ✅ **Resume Configuration** - Generate configs for resuming with modified parameters

## 🛠️ Configuration System

The project uses a sophisticated JSON-based configuration system with advanced experiment management:

### 📋 **Configuration Structure**
```json
{
  "model": {
    "d_model": 512,
    "n_heads": 8,
    "n_encoder_layers": 6,
    "n_decoder_layers": 6,
    "d_ff": 2048,
    "dropout": 0.1,
    "max_seq_length": 100
  },
  "training": {
    "epochs": 50,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "weight_decay": 0.01,
    "gradient_clip_norm": 1.0,
    "warmup_steps": 4000,
    "label_smoothing": 0.1
  },
  "data": {
    "train_file": "data/train.json",
    "val_file": "data/val.json",
    "vocab_size": 10000,
    "min_freq": 2
  },
  "experiment": {
    "name": "default_training",
    "description": "Default training configuration",
    "created_at": "2025-01-26T08:21:36.123456",
    "timestamp_id": "20250926_082136",
    "training_environment": "local_gpu",
    "config_file": "default_training_20250926_082136_local_gpu.json",
    "experiment_type": "new"
  }
}
```

### 🔄 **Derived Experiment Configuration**
When creating derived experiments, the system automatically adds source tracking:

```json
{
  "source_experiment_info": {
    "source_experiment_name": "cloud_training_20250926_082552",
    "source_description": "Cloud training experiment",
    "source_created_at": "2025-01-26T08:25:52.123456",
    "source_timestamp_id": "20250926_082552",
    "source_training_environment": "cloud",
    "source_config_file": "cloud_training_20250926_082552_cloud.json",
    "derivation_timestamp": "2025-01-26T08:30:15.789012",
    "derivation_reason": "Derived from cloud_training_20250926_082552 experiment"
  },
  "current_experiment": {
    "name": "local_enhanced",
    "description": "Local enhanced training derived from cloud",
    "created_at": "2025-01-26T08:30:15.789012",
    "timestamp_id": "20250926_083015",
    "training_environment": "local_gpu",
    "config_file": "local_enhanced_20250926_083015_local_gpu.json",
    "experiment_type": "derived",
    "source_experiment_name": "cloud_training_20250926_082552"
  }
}
```

### 🎯 **Environment-Specific Configurations**
The system supports three training environments:

- **`local_cpu`**: Local training on CPU
- **`local_gpu`**: Local training on GPU (if available)
- **`cloud`**: Simulated Google Colab environment

Each environment generates config files with appropriate suffixes:
- `experiment_name_20250926_082136_local_cpu.json`
- `experiment_name_20250926_082136_local_gpu.json`
- `experiment_name_20250926_082136_cloud.json`

## 🚀 **Complete Usage Workflow**

### **Step 1: Environment Setup**
```bash
# Activate conda environment
conda activate torch2.5

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### **Step 2: Clean Start**
```bash
# Clean all generated files for fresh start
python run.py --cleanup-all
```

### **Step 3: Create and Run Cloud Training**
```bash
# Create a cloud experiment
python run.py --create-experiment --name cloud_training --description "Cloud training experiment" --environment cloud

# Run cloud training
python run.py --train --experiment cloud_training --force-cloud
```

### **Step 4: Create Derived Local Experiment**
```bash
# Create a derived experiment from the cloud training
python run.py --create-derived-experiment \
  --source-experiment cloud_training_20250926_082552 \
  --name local_enhanced \
  --description "Local enhanced training derived from cloud" \
  --new-environment local_gpu \
  --overrides '{"training": {"epochs": 5, "learning_rate": 0.0002}}'

# Run the derived experiment
python run.py --train --experiment local_enhanced
```

### **Step 5: Monitor and Manage**
```bash
# List all experiments
python run.py --list-experiments

# Show specific experiment details
python run.py --show-experiment local_enhanced

# List checkpoints
python run.py --list-checkpoints --experiment local_enhanced

# Resume training with modified parameters
python run.py --create-resume \
  --experiment local_enhanced \
  --checkpoint checkpoints/run_20250926_083015/best_model_epoch_003.pt \
  --overrides '{"training": {"learning_rate": 0.0001}}'
```

### **Step 6: System Maintenance**
```bash
# Cleanup old checkpoints (keep last 3 runs, 5 checkpoints each)
python run.py --cleanup --keep-runs 3 --keep-checkpoints 5

# List configuration versions
python run.py --list-config-versions
```

## 📚 **Command Reference**

### **`run.py` - Unified Training Entry Point**

The `run.py` script is the main entry point for all training operations. It provides a comprehensive interface for experiment management, training execution, and system maintenance.

#### **Training Commands**
```bash
python run.py --train [OPTIONS]

Options:
  --use-default              Use default training configuration
  --experiment NAME          Use specific experiment configuration
  --config PATH              Use specific config file path
  --force-local              Force local training (CPU/GPU)
  --force-cloud              Force cloud training (simulated Colab)
```

#### **Experiment Management Commands**
```bash
# List all experiments
python run.py --list-experiments

# Show experiment details
python run.py --show-experiment NAME

# Create new experiment
python run.py --create-experiment --name NAME --description "DESCRIPTION" [--environment ENV]

# Create derived experiment
python run.py --create-derived-experiment \
  --source-experiment SOURCE_NAME \
  --name NEW_NAME \
  --description "DESCRIPTION" \
  --new-environment ENV \
  --overrides '{"key": "value"}'
```

#### **Checkpoint Management Commands**
```bash
# List all checkpoints
python run.py --list-checkpoints [--experiment NAME]

# Create resume configuration
python run.py --create-resume \
  --experiment NAME \
  --checkpoint PATH \
  --overrides '{"training": {"learning_rate": 0.0001}}'

# Validate parameter changes
python run.py --validate-changes \
  --experiment NAME \
  --overrides '{"training": {"learning_rate": 0.0001}}'
```

#### **System Maintenance Commands**
```bash
# List configuration versions
python run.py --list-config-versions

# Cleanup old checkpoints
python run.py --cleanup --keep-runs N --keep-checkpoints M

# Clean all generated files
python run.py --cleanup-all
```

### **Parameter Override Examples**
```bash
# Override training parameters
--overrides '{"training": {"epochs": 10, "learning_rate": 0.0002}}'

# Override model parameters
--overrides '{"model": {"d_model": 256, "n_heads": 4}}'

# Override data parameters
--overrides '{"data": {"batch_size": 64, "vocab_size": 5000}}'

# Complex nested overrides
--overrides '{"training": {"epochs": 5}, "model": {"dropout": 0.2}}'
```

## 📊 Model Architecture

The Transformer model includes:
- **Embedding Layers**: Source and target embeddings with scaling
- **Positional Encoding**: Sine/cosine positional encodings
- **Encoder Stack**: Multi-head self-attention + feed-forward layers
- **Decoder Stack**: Masked self-attention + encoder-decoder attention + feed-forward
- **Output Layer**: Linear projection to target vocabulary

## 🧪 Testing

Run comprehensive tests:
```bash
# Test all components
python -m pytest tests/functional/

# Test specific components
python tests/functional/test_transformer.py
python tests/functional/test_positional_encoding.py
python tests/functional/test_multihead_attention.py
```

## 📈 Training Visualization

The training system provides comprehensive visualizations:
- **Training Metrics**: Loss curves, learning rate schedules
- **Gradient Analysis**: Gradient flow and magnitude plots
- **Attention Maps**: Attention weight visualizations
- **Evaluation Results**: Translation quality metrics

## 🔍 Evaluation Metrics

The system supports multiple evaluation metrics:
- **BLEU**: Bilingual Evaluation Understudy
- **METEOR**: Metric for Evaluation of Translation with Explicit ORdering
- **ROUGE-L**: Recall-Oriented Understudy for Gisting Evaluation
- **Exact Match**: Exact sequence matching
- **Word Accuracy**: Word-level accuracy
- **Perplexity**: Model confidence measure

## 🛠️ Technical Stack

- **Deep Learning**: PyTorch 2.2.2
- **Python**: 3.10.18
- **Environment**: Conda
- **Visualization**: Matplotlib
- **Evaluation**: NLTK, SacreBLEU
- **Configuration**: JSON

## 📝 Development

### Code Structure
- **Modular Design**: Clear separation of concerns
- **Type Hints**: Full type annotation support
- **Documentation**: Comprehensive docstrings
- **Testing**: Extensive test coverage
- **Configuration**: JSON-based configuration system

### Best Practices
- Follow PEP 8 code style
- Write comprehensive tests
- Document all functions and classes
- Use meaningful variable names
- Implement proper error handling

## 🔧 **Troubleshooting & Common Issues**

### **Environment Issues**
```bash
# Issue: ModuleNotFoundError: No module named 'torch'
# Solution: Activate conda environment
conda activate torch2.5

# Issue: CUDA not available
# Solution: Check PyTorch installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### **Configuration Issues**
```bash
# Issue: TrainingConfig.__init__() missing required arguments
# Solution: Ensure all required fields are present in overrides
# Use --validate-changes to check before training
python run.py --validate-changes --experiment NAME --overrides '{"training": {"epochs": 5}}'
```

### **File Management Issues**
```bash
# Issue: Too many checkpoint files
# Solution: Cleanup old checkpoints
python run.py --cleanup --keep-runs 3 --keep-checkpoints 5

# Issue: Corrupted experiment configs
# Solution: Clean all and start fresh
python run.py --cleanup-all
```

### **Training Issues**
```bash
# Issue: Training fails with memory error
# Solution: Reduce batch size in overrides
--overrides '{"training": {"batch_size": 16}}'

# Issue: Training too slow
# Solution: Use GPU if available
python run.py --train --experiment NAME --force-local
```

## 🎯 **Future Enhancements**

1. **Advanced Architectures**: Implement variants like BERT, GPT
2. **Optimization**: Add mixed precision training
3. **Distributed Training**: Multi-GPU training support
4. **Inference**: Beam search and sampling strategies
5. **Deployment**: Model serving and API development
6. **Real-time Monitoring**: Web-based training dashboard
7. **AutoML**: Automated hyperparameter optimization
8. **Model Compression**: Quantization and pruning support

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

**Note**: This is an educational project for understanding Transformer architecture implementation details.