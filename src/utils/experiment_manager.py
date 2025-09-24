"""
Experiment Manager
Manages different training configurations and experiments
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

try:
    from .config_manager import ConfigManager, TrainingConfigManager
except ImportError:
    from config_manager import ConfigManager, TrainingConfigManager


class ExperimentManager:
    """
    Manages different training experiments with their configurations.
    
    Features:
    - Create and manage multiple experiment configurations
    - Save/load experiment configurations
    - Track experiment history
    - Compare different experiments
    - Generate experiment reports
    """
    
    def __init__(self, experiments_dir: str = "experiments"):
        """
        Initialize experiment manager.
        
        Args:
            experiments_dir: Directory to store experiment configurations
        """
        self.experiments_dir = experiments_dir
        self.configs_dir = os.path.join(experiments_dir, "configs")
        self.results_dir = os.path.join(experiments_dir, "results")
        
        # Create directories
        os.makedirs(self.experiments_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"ExperimentManager initialized:")
        print(f"  Experiments directory: {self.experiments_dir}")
        print(f"  Configs directory: {self.configs_dir}")
        print(f"  Results directory: {self.results_dir}")
    
    def create_experiment(
        self,
        name: str,
        description: str = "",
        base_config_path: str = "training_config.json",
        **overrides
    ) -> Tuple[str, str]:
        """
        Create a new experiment configuration.
        
        Args:
            name: Experiment name (will be used as filename)
            description: Experiment description
            base_config_path: Path to base configuration file
            **overrides: Configuration overrides
            
        Returns:
            Tuple of (experiment_path, timestamp_id)
        """
        # Generate unique timestamp
        timestamp = datetime.now()
        timestamp_id = timestamp.strftime('%Y%m%d_%H%M%S')
        
        # Load base configuration
        base_config_manager = ConfigManager(base_config_path)
        base_config = base_config_manager.get_config()
        
        # Apply overrides
        config_dict = self._config_to_dict(base_config)
        
        # Update experiment info with timestamp
        config_dict['experiment']['name'] = name
        config_dict['experiment']['description'] = description
        config_dict['experiment']['created_at'] = timestamp.isoformat()
        config_dict['experiment']['version'] = "1.0"
        config_dict['experiment']['timestamp_id'] = timestamp_id  # Add timestamp ID
        
        # Apply user overrides
        for key, value in overrides.items():
            if '.' in key:
                # Handle nested keys like 'model.d_model'
                keys = key.split('.')
                current = config_dict
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config_dict[key] = value
        
        # Save experiment configuration with timestamp
        experiment_filename = f"{name}_{timestamp_id}.json"
        experiment_path = os.path.join(self.configs_dir, experiment_filename)
        
        with open(experiment_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Experiment created: {experiment_filename}")
        print(f"Configuration saved to: {experiment_path}")
        print(f"Timestamp ID: {timestamp_id}")
        
        return experiment_path, timestamp_id
    
    def get_experiment_timestamp(self, experiment_name: str) -> Optional[str]:
        """
        Get timestamp ID from experiment configuration.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Timestamp ID if found, None otherwise
        """
        experiments = self.list_experiments()
        
        for exp in experiments:
            if exp['name'] == experiment_name:
                return exp.get('timestamp_id')
        
        return None
    
    def validate_experiment_integrity(self, experiment_name: str) -> bool:
        """
        Validate that experiment configuration hasn't been manually modified.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            True if integrity is valid, False otherwise
        """
        experiments = self.list_experiments()
        
        for exp in experiments:
            if exp['name'] == experiment_name:
                # Check if filename matches timestamp
                expected_filename = f"{exp['name']}_{exp.get('timestamp_id', 'unknown')}.json"
                actual_filename = os.path.basename(exp['path'])
                
                if expected_filename != actual_filename:
                    print(f"WARNING: Experiment '{experiment_name}' may have been manually modified!")
                    print(f"Expected filename: {expected_filename}")
                    print(f"Actual filename: {actual_filename}")
                    return False
                
                return True
        
        print(f"ERROR: Experiment '{experiment_name}' not found!")
        return False
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all available experiments.
        
        Returns:
            List of experiment information dictionaries
        """
        experiments = []
        
        for filename in os.listdir(self.configs_dir):
            if filename.endswith('.json'):
                config_path = os.path.join(self.configs_dir, filename)
                
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                experiments.append({
                    'filename': filename,
                    'path': config_path,
                    'name': config.get('experiment', {}).get('name', 'Unknown'),
                    'description': config.get('experiment', {}).get('description', ''),
                    'created_at': config.get('experiment', {}).get('created_at', ''),
                    'version': config.get('experiment', {}).get('version', ''),
                    'timestamp_id': config.get('experiment', {}).get('timestamp_id', ''),
                    'model_config': config.get('model', {}),
                    'training_config': config.get('training', {})
                })
        
        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x['created_at'], reverse=True)
        
        return experiments
    
    def print_experiments(self):
        """Print all available experiments in a formatted table."""
        experiments = self.list_experiments()
        
        if not experiments:
            print("No experiments found.")
            return
        
        print(f"\n{'='*100}")
        print(f"AVAILABLE EXPERIMENTS")
        print(f"{'='*100}")
        print(f"{'Name':<25} {'Description':<30} {'Created':<20} {'Model':<15} {'Epochs':<8}")
        print(f"{'-'*100}")
        
        for exp in experiments:
            model_info = f"{exp['model_config'].get('d_model', 'N/A')}/{exp['model_config'].get('n_heads', 'N/A')}"
            epochs = exp['training_config'].get('epochs', 'N/A')
            created = exp['created_at'][:19] if exp['created_at'] else 'N/A'
            
            print(f"{exp['name']:<25} {exp['description'][:29]:<30} {created:<20} {model_info:<15} {epochs:<8}")
        
        print(f"{'='*100}")
    
    def load_experiment(self, experiment_name: str) -> ConfigManager:
        """
        Load an experiment configuration.
        
        Args:
            experiment_name: Name of the experiment to load
            
        Returns:
            ConfigManager instance with loaded configuration
        """
        experiments = self.list_experiments()
        
        # Find experiment by name
        experiment = None
        for exp in experiments:
            if exp['name'] == experiment_name:
                experiment = exp
                break
        
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        # Load configuration
        config_manager = ConfigManager(experiment['path'])
        
        print(f"Loaded experiment: {experiment['name']}")
        print(f"Description: {experiment['description']}")
        print(f"Created: {experiment['created_at']}")
        
        return config_manager
    
    def run_experiment(self, experiment_name: str, train_script: str = "train.py"):
        """
        Run an experiment using its configuration.
        
        Args:
            experiment_name: Name of the experiment to run
            train_script: Path to training script
        """
        # Validate experiment integrity first
        if not self.validate_experiment_integrity(experiment_name):
            print("Experiment integrity check failed. Aborting run.")
            return
        
        # Load experiment configuration
        config_manager = self.load_experiment(experiment_name)
        
        # Get timestamp ID for checkpoint directory naming
        timestamp_id = self.get_experiment_timestamp(experiment_name)
        if timestamp_id:
            print(f"Using timestamp ID for checkpoint: {timestamp_id}")
        
        # Create temporary config file for training script
        temp_config_path = "temp_training_config.json"
        config_manager.save_config(temp_config_path)
        
        # Ensure the temporary config contains the experiment timestamp
        temp_config_manager = ConfigManager(temp_config_path)
        temp_config = temp_config_manager.get_config()
        if hasattr(temp_config.experiment, 'timestamp_id') and temp_config.experiment.timestamp_id != timestamp_id:
            # Update timestamp in temporary config
            temp_config.experiment.timestamp_id = timestamp_id
            temp_config_manager.save_config(temp_config_path)
        
        # Update train.py to use temporary config
        self._update_train_script_config(train_script, temp_config_path)
        
        # Update the command line arguments for train.py
        import sys
        if '--config' not in sys.argv:
            sys.argv.extend(['--config', temp_config_path])
        else:
            # Find and replace the config argument
            for i, arg in enumerate(sys.argv):
                if arg == '--config' and i + 1 < len(sys.argv):
                    sys.argv[i + 1] = temp_config_path
                    break
        
        print(f"\nRunning experiment: {experiment_name}")
        print(f"Configuration: {config_manager.config_path}")
        print(f"Training script: {train_script}")
        print(f"Checkpoint will use timestamp: {timestamp_id}")
        
        # Run training
        import subprocess
        try:
            result = subprocess.run(['python', train_script], check=True, capture_output=True, text=True)
            print("Training completed successfully!")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Training failed with error: {e}")
            print(f"Error output: {e.stderr}")
        finally:
            # Clean up temporary config file
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
    def compare_experiments(self, experiment_names: List[str]):
        """
        Compare multiple experiments.
        
        Args:
            experiment_names: List of experiment names to compare
        """
        experiments = []
        
        for name in experiment_names:
            try:
                config_manager = self.load_experiment(name)
                experiments.append({
                    'name': name,
                    'config': config_manager.get_config()
                })
            except ValueError as e:
                print(f"Warning: {e}")
        
        if len(experiments) < 2:
            print("Need at least 2 experiments to compare.")
            return
        
        print(f"\n{'='*120}")
        print(f"EXPERIMENT COMPARISON")
        print(f"{'='*120}")
        
        # Compare key parameters
        comparison_fields = [
            ('Model', 'model', ['d_model', 'n_heads', 'n_encoder_layers', 'n_decoder_layers', 'd_ff', 'dropout']),
            ('Training', 'training', ['epochs', 'learning_rate', 'weight_decay', 'gradient_clip_norm']),
            ('Data', 'data', ['batch_size', 'max_length', 'train_split']),
            ('Optimizer', 'optimizer', ['type', 'betas', 'eps']),
            ('Scheduler', 'scheduler', ['type', 'step_size', 'gamma']),
            ('Loss', 'loss', ['type', 'smoothing', 'ignore_index'])
        ]
        
        for section_name, section_key, fields in comparison_fields:
            print(f"\n{section_name}:")
            print(f"{'Parameter':<25} {'Experiment 1':<20} {'Experiment 2':<20} {'Experiment 3':<20}")
            print(f"{'-'*85}")
            
            for field in fields:
                values = []
                for exp in experiments:
                    section_config = getattr(exp['config'], section_key)
                    value = getattr(section_config, field, 'N/A')
                    values.append(str(value))
                
                # Pad with empty strings if fewer experiments
                while len(values) < 3:
                    values.append('')
                
                print(f"{field:<25} {values[0]:<20} {values[1]:<20} {values[2]:<20}")
        
        print(f"{'='*120}")
    
    def create_experiment_template(self, template_name: str = "template"):
        """
        Create an experiment template with common configurations.
        
        Args:
            template_name: Name of the template
        """
        templates = {
            'small': {
                'description': 'Small model for quick testing',
                'model': {
                    'd_model': 128,
                    'n_heads': 4,
                    'n_encoder_layers': 2,
                    'n_decoder_layers': 2,
                    'd_ff': 512,
                    'dropout': 0.1
                },
                'training': {
                    'epochs': 10,
                    'learning_rate': 0.001,
                    'batch_size': 16
                }
            },
            'medium': {
                'description': 'Medium model for balanced performance',
                'model': {
                    'd_model': 256,
                    'n_heads': 8,
                    'n_encoder_layers': 4,
                    'n_decoder_layers': 4,
                    'd_ff': 1024,
                    'dropout': 0.1
                },
                'training': {
                    'epochs': 30,
                    'learning_rate': 0.0005,
                    'batch_size': 32
                }
            },
            'large': {
                'description': 'Large model for best performance',
                'model': {
                    'd_model': 512,
                    'n_heads': 8,
                    'n_encoder_layers': 6,
                    'n_decoder_layers': 6,
                    'd_ff': 2048,
                    'dropout': 0.1
                },
                'training': {
                    'epochs': 50,
                    'learning_rate': 0.0001,
                    'batch_size': 32
                }
            }
        }
        
        if template_name not in templates:
            print(f"Available templates: {list(templates.keys())}")
            return
        
        template_config = templates[template_name]
        
        # Create experiment with template configuration
        experiment_path = self.create_experiment(
            name=f"{template_name}_experiment",
            description=template_config['description'],
            **template_config
        )
        
        print(f"Template '{template_name}' created: {experiment_path}")
    
    def _config_to_dict(self, config: TrainingConfigManager) -> Dict[str, Any]:
        """Convert TrainingConfigManager to dictionary."""
        from dataclasses import asdict
        return asdict(config)
    
    def _update_train_script_config(self, train_script: str, config_path: str):
        """Update train script to use specific config path."""
        # This is a placeholder - in practice, you might want to modify the train script
        # or pass the config path as a command line argument
        pass


def main():
    """Test the ExperimentManager."""
    print("Testing ExperimentManager...")
    
    # Initialize manager
    manager = ExperimentManager()
    
    # Create some example experiments
    print("\nCreating example experiments...")
    
    # Small experiment
    manager.create_experiment(
        name="small_test",
        description="Small model for quick testing",
        model_d_model=128,
        model_n_heads=4,
        model_n_encoder_layers=2,
        model_n_decoder_layers=2,
        training_epochs=5,
        training_learning_rate=0.001,
        data_batch_size=16
    )
    
    # Medium experiment
    manager.create_experiment(
        name="medium_experiment",
        description="Medium model for balanced performance",
        model_d_model=256,
        model_n_heads=8,
        model_n_encoder_layers=4,
        model_n_decoder_layers=4,
        training_epochs=20,
        training_learning_rate=0.0005,
        data_batch_size=32
    )
    
    # List experiments
    print("\nListing experiments...")
    manager.print_experiments()
    
    # Compare experiments
    print("\nComparing experiments...")
    manager.compare_experiments(["small_test", "medium_experiment"])
    
    print("\nExperimentManager test completed!")


if __name__ == "__main__":
    main()
