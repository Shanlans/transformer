# Transformer Project

A comprehensive PyTorch implementation of the Transformer model for machine translation tasks.

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
```bash
# List all experiments
python run.py --list-experiments

# Show experiment details
python run.py --show-experiment small_test

# Create new experiment
python run.py --create-experiment --name small_test --description "Small model for testing"

# List all checkpoints
python run.py --list-checkpoints

# List checkpoints for specific experiment
python run.py --list-checkpoints --experiment small_test

# Create resume config with modified hyperparameters
python run.py --create-resume --experiment small_test --checkpoint checkpoints/run_20250925_070047/best_model_epoch_005.pt --overrides '{"training": {"learning_rate": 0.0002, "epochs": 10}}'

# Validate hyperparameter changes
python run.py --validate-changes --experiment small_test --overrides '{"training": {"learning_rate": 0.0002}}'

# List configuration versions
python run.py --list-config-versions

# Cleanup old checkpoints
python run.py --cleanup --keep-runs 3 --keep-checkpoints 5

# Resume training with new config
python run.py --train --config experiments/configs/resume_small_test_20250925_073041.json
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
- ✅ **Experiment Comparison** - Side-by-side comparison of experiments
- ✅ **Template System** - Pre-defined configuration templates
- ✅ **Version Control** - Track experiment history and changes
- ✅ **Timestamp Linkage** - Link experiment configs with checkpoint timestamps
- ✅ **Integrity Validation** - Prevent manual config modification

### Checkpoint Management
- ✅ **Timestamped Runs** - Organized checkpoint storage
- ✅ **Metadata Tracking** - Comprehensive checkpoint metadata
- ✅ **Cleanup Tools** - Automated checkpoint cleanup
- ✅ **Model Resuming** - Resume training from checkpoints
- ✅ **Hyperparameter Validation** - Structural vs non-structural parameter validation
- ✅ **Resume Configuration** - Generate configs for resuming with modified parameters

## 🛠️ Configuration

The project uses JSON-based configuration with an advanced experiment management system:

### Basic Configuration
Edit `training_config.json` to customize:

```json
{
  "model": {
    "d_model": 512,
    "n_heads": 8,
    "n_encoder_layers": 6,
    "n_decoder_layers": 6
  },
  "training": {
    "epochs": 50,
    "learning_rate": 0.0001,
    "batch_size": 32
  }
}
```

### Experiment Management
Create and manage multiple experiment configurations:

```bash
# Create experiments with different parameters
python manage_experiments.py create --name small_test --d_model 128 --epochs 5
python manage_experiments.py create --name large_exp --d_model 512 --epochs 50

# Compare experiments
python manage_experiments.py compare --names small_test large_exp

# Run specific experiment
python manage_experiments.py run --name small_test
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

## 🎯 Future Enhancements

1. **Advanced Architectures**: Implement variants like BERT, GPT
2. **Optimization**: Add mixed precision training
3. **Distributed Training**: Multi-GPU training support
4. **Inference**: Beam search and sampling strategies
5. **Deployment**: Model serving and API development

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