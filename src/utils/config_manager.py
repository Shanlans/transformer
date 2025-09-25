"""
Configuration Manager
Handles loading and validation of training configurations from JSON files
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str
    description: str
    version: str
    created_at: Optional[str] = None
    timestamp_id: Optional[str] = None
    training_environment: Optional[str] = None
    gpu_info: Optional[Dict[str, Any]] = None


@dataclass
class DataConfig:
    """Data configuration."""
    train_data_path: str
    max_length: int
    train_split: float
    batch_size: int
    num_workers: int
    shuffle: bool


@dataclass
class ModelConfig:
    """Model configuration."""
    d_model: int
    n_heads: int
    n_encoder_layers: int
    n_decoder_layers: int
    d_ff: int
    dropout: float
    max_len: int


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    early_stopping_patience: int
    save_best_only: bool


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str
    betas: list
    eps: float


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""
    type: str
    step_size: int
    gamma: float
    T_max: int


@dataclass
class LossConfig:
    """Loss function configuration."""
    type: str
    smoothing: float
    ignore_index: int


@dataclass
class SystemConfig:
    """System configuration."""
    device: str
    save_dir: str
    max_checkpoints: int
    num_workers: int


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    enabled: bool
    save_dir: str
    plot_metrics: bool
    plot_attention: bool
    plot_gradient_flow: bool
    attention_layers: list
    attention_heads: list


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    enabled: bool
    max_samples: int
    metrics: list


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    print_every_n_epochs: int
    save_metrics_every_n_epochs: int
    detailed_progress: bool


@dataclass
class ResumeConfig:
    """Resume training configuration."""
    checkpoint_path: str
    original_experiment: str
    resume_timestamp: str
    validation_result: Dict[str, Any]


@dataclass
class TrainingConfigManager:
    """Complete training configuration manager."""
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: LossConfig
    system: SystemConfig
    visualization: VisualizationConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig
    resume: Optional[ResumeConfig] = None


class ConfigManager:
    """
    Configuration manager for loading and validating training configurations.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config_path = config_path
        self.config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Validate and create configuration objects
        resume_config = None
        if 'resume' in config_dict and config_dict['resume'] is not None:
            resume_config = ResumeConfig(**config_dict['resume'])
        
        self.config = TrainingConfigManager(
            experiment=ExperimentConfig(**config_dict['experiment']),
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            optimizer=OptimizerConfig(**config_dict['optimizer']),
            scheduler=SchedulerConfig(**config_dict['scheduler']),
            loss=LossConfig(**config_dict['loss']),
            system=SystemConfig(**config_dict['system']),
            visualization=VisualizationConfig(**config_dict['visualization']),
            evaluation=EvaluationConfig(**config_dict['evaluation']),
            logging=LoggingConfig(**config_dict['logging']),
            resume=resume_config
        )
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate model parameters
        if self.config.model.d_model % self.config.model.n_heads != 0:
            raise ValueError(f"d_model ({self.config.model.d_model}) must be divisible by n_heads ({self.config.model.n_heads})")
        
        if self.config.model.d_model <= 0:
            raise ValueError("d_model must be positive")
        
        if self.config.model.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        
        if self.config.model.n_encoder_layers <= 0:
            raise ValueError("n_encoder_layers must be positive")
        
        if self.config.model.n_decoder_layers <= 0:
            raise ValueError("n_decoder_layers must be positive")
        
        # Validate training parameters
        if self.config.training.epochs <= 0:
            raise ValueError("epochs must be positive")
        
        if self.config.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.config.training.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        
        # Validate data parameters
        if self.config.data.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if not 0 < self.config.data.train_split < 1:
            raise ValueError("train_split must be between 0 and 1")
        
        if self.config.data.max_length <= 0:
            raise ValueError("max_length must be positive")
        
        # Validate system parameters
        if self.config.system.max_checkpoints < 0:
            raise ValueError("max_checkpoints must be non-negative")
        
        print(f"Configuration loaded and validated: {self.config_path}")
    
    def get_config(self) -> TrainingConfigManager:
        """Get the loaded configuration."""
        return self.config
    
    def save_config(self, save_path: str, preserve_experiment_info: bool = True):
        """
        Save current configuration to JSON file.
        
        Args:
            save_path: Path to save the configuration
            preserve_experiment_info: Whether to preserve existing experiment info
        """
        config_dict = asdict(self.config)
        
        # If preserving experiment info and file exists, merge experiment info
        if preserve_experiment_info and os.path.exists(save_path):
            try:
                with open(save_path, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)
                
                # Preserve experiment info from existing config
                if 'experiment' in existing_config:
                    existing_experiment = existing_config['experiment']
                    # Only preserve non-training-specific experiment info
                    preserved_keys = ['name', 'description', 'version', 'created_at', 'timestamp_id', 'training_environment', 'gpu_info']
                    for key in preserved_keys:
                        if key in existing_experiment:
                            config_dict['experiment'][key] = existing_experiment[key]
                    
                    print(f"ℹ️  Preserved experiment info from existing config")
            except Exception as e:
                print(f"⚠️  Could not preserve experiment info: {e}")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration saved to: {save_path}")
    
    def check_config_overwrite(self, config_path: str) -> Dict[str, Any]:
        """
        Check if overwriting a config would affect experiment info.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary with overwrite analysis
        """
        analysis = {
            'file_exists': os.path.exists(config_path),
            'has_experiment_info': False,
            'experiment_info': None,
            'would_preserve': True,
            'recommendations': []
        }
        
        if analysis['file_exists']:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)
                
                if 'experiment' in existing_config:
                    analysis['has_experiment_info'] = True
                    analysis['experiment_info'] = existing_config['experiment']
                    
                    # Check if experiment info would be preserved
                    current_config_dict = asdict(self.config)
                    if 'experiment' in current_config_dict:
                        preserved_keys = ['name', 'description', 'version', 'created_at', 'timestamp_id', 'training_environment', 'gpu_info']
                        for key in preserved_keys:
                            if key in existing_config['experiment']:
                                if key not in current_config_dict['experiment'] or current_config_dict['experiment'][key] != existing_config['experiment'][key]:
                                    analysis['would_preserve'] = False
                                    analysis['recommendations'].append(f"Key '{key}' would be overwritten")
                    
                    if analysis['would_preserve']:
                        analysis['recommendations'].append("Experiment info would be preserved")
                    else:
                        analysis['recommendations'].append("Consider using preserve_experiment_info=True")
                        
            except Exception as e:
                analysis['recommendations'].append(f"Error reading config: {e}")
        
        return analysis
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        config_dict = asdict(self.config)
        
        # Update nested dictionaries
        for key, value in updates.items():
            if isinstance(value, dict):
                if key in config_dict:
                    config_dict[key].update(value)
                else:
                    config_dict[key] = value
            else:
                config_dict[key] = value
        
        # Reload configuration
        self.config = TrainingConfigManager(
            experiment=ExperimentConfig(**config_dict['experiment']),
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            optimizer=OptimizerConfig(**config_dict['optimizer']),
            scheduler=SchedulerConfig(**config_dict['scheduler']),
            loss=LossConfig(**config_dict['loss']),
            system=SystemConfig(**config_dict['system']),
            visualization=VisualizationConfig(**config_dict['visualization']),
            evaluation=EvaluationConfig(**config_dict['evaluation']),
            logging=LoggingConfig(**config_dict['logging'])
        )
        
        # Re-validate
        self._validate_config()
    
    def print_config(self):
        """Print current configuration."""
        print("=" * 60)
        print("TRAINING CONFIGURATION")
        print("=" * 60)
        
        config_dict = asdict(self.config)
        
        for section, params in config_dict.items():
            print(f"\n{section.upper()}:")
            if params is not None:
                for key, value in params.items():
                    print(f"  {key}: {value}")
            else:
                print("  None")
        
        print("=" * 60)


def load_config(config_path: str) -> ConfigManager:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)


if __name__ == "__main__":
    # Test the configuration manager
    print("Testing ConfigManager...")
    
    # Load configuration
    config_manager = load_config("config/training_config.json")
    
    # Print configuration
    config_manager.print_config()
    
    # Test configuration access
    config = config_manager.get_config()
    print(f"\nModel dimension: {config.model.d_model}")
    print(f"Number of epochs: {config.training.epochs}")
    print(f"Batch size: {config.data.batch_size}")
    
    print("ConfigManager test completed!")
