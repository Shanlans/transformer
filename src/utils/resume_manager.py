#!/usr/bin/env python3
"""
Resume Training Manager

This module provides functionality to:
1. Load previous experiment configurations
2. Link configurations with corresponding checkpoints
3. Validate hyperparameter changes (structural vs non-structural)
4. Resume training with modified hyperparameters
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

try:
    from .config_manager import ConfigManager, TrainingConfigManager
    from .experiment_manager import ExperimentManager
    from .checkpoint_manager import CheckpointManager
except ImportError:
    from config_manager import ConfigManager, TrainingConfigManager
    from experiment_manager import ExperimentManager
    from checkpoint_manager import CheckpointManager


class ResumeManager:
    """
    Manager for resuming training from checkpoints with configuration validation.
    
    Features:
    - Load experiment configurations
    - Find corresponding checkpoints
    - Validate hyperparameter changes
    - Resume training with modified parameters
    - Structural vs non-structural parameter validation
    """
    
    def __init__(self):
        """Initialize Resume Manager."""
        self.experiment_manager = ExperimentManager()
        self.checkpoint_manager = CheckpointManager()
        
        # Define structural vs non-structural parameters
        self.structural_params = {
            'model': ['d_model', 'n_heads', 'n_encoder_layers', 'n_decoder_layers', 'd_ff', 'max_len'],
            'data': ['max_length', 'batch_size']  # These affect model architecture
        }
        
        self.non_structural_params = {
            'training': ['epochs', 'learning_rate', 'weight_decay', 'gradient_clip_norm', 'early_stopping_patience'],
            'optimizer': ['type', 'betas', 'eps'],
            'scheduler': ['type', 'step_size', 'gamma', 'T_max'],
            'loss': ['type', 'smoothing', 'ignore_index'],
            'system': ['device', 'save_dir', 'max_checkpoints', 'num_workers'],
            'visualization': ['enabled', 'save_dir', 'plot_metrics', 'plot_attention', 'plot_gradient_flow'],
            'evaluation': ['enabled', 'max_samples', 'metrics'],
            'logging': ['level', 'print_every_n_epochs', 'save_metrics_every_n_epochs', 'detailed_progress']
        }
        
        print("ResumeManager initialized!")
        print(f"Structural parameters: {list(self.structural_params.keys())}")
        print(f"Non-structural parameters: {list(self.non_structural_params.keys())}")
    
    def find_experiment_checkpoints(self, experiment_name: str) -> List[Dict[str, Any]]:
        """
        Find all checkpoints associated with an experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            List of checkpoint information
        """
        try:
            # Load experiment to get timestamp
            config_manager = self.experiment_manager.load_experiment(experiment_name)
            config = config_manager.get_config()
            
            # Get experiment timestamp
            timestamp_id = self.experiment_manager.get_experiment_timestamp(experiment_name)
            
            # Find checkpoints with matching timestamp
            checkpoints = []
            checkpoint_base_dir = "checkpoints"
            
            if os.path.exists(checkpoint_base_dir):
                for run_dir in os.listdir(checkpoint_base_dir):
                    if run_dir.startswith("run_"):
                        run_path = os.path.join(checkpoint_base_dir, run_dir)
                        
                        # Check if this run matches the experiment timestamp
                        # If no timestamp_id, try to match by creation time
                        is_match = False
                        if timestamp_id and timestamp_id in run_dir:
                            is_match = True
                        elif not timestamp_id:
                            # Try to match by creation time
                            try:
                                # Extract timestamp from run_dir (format: run_YYYYMMDD_HHMMSS)
                                run_timestamp = run_dir.replace("run_", "")
                                # Check if this matches the experiment creation time
                                exp_created = config.experiment.created_at
                                if exp_created:
                                    exp_timestamp = exp_created.replace("-", "").replace(":", "").replace("T", "_").split(".")[0]
                                    if run_timestamp in exp_timestamp or exp_timestamp in run_timestamp:
                                        is_match = True
                            except:
                                pass
                        
                        if is_match:
                            # Get checkpoint files
                            checkpoint_files = []
                            for file in os.listdir(run_path):
                                if file.endswith('.pt') and 'best_model' in file:
                                    checkpoint_files.append(file)
                            
                            if checkpoint_files:
                                checkpoints.append({
                                    'run_dir': run_path,
                                    'run_name': run_dir,
                                    'checkpoint_files': checkpoint_files,
                                    'experiment_name': experiment_name,
                                    'timestamp_id': timestamp_id
                                })
            
            # If no specific matches found, return all recent checkpoints
            if not checkpoints:
                print(f"No specific checkpoints found for experiment '{experiment_name}'")
                print("Showing all available checkpoints:")
                
                if os.path.exists(checkpoint_base_dir):
                    for run_dir in os.listdir(checkpoint_base_dir):
                        if run_dir.startswith("run_"):
                            run_path = os.path.join(checkpoint_base_dir, run_dir)
                            
                            # Get checkpoint files
                            checkpoint_files = []
                            for file in os.listdir(run_path):
                                if file.endswith('.pt') and 'best_model' in file:
                                    checkpoint_files.append(file)
                            
                            if checkpoint_files:
                                checkpoints.append({
                                    'run_dir': run_path,
                                    'run_name': run_dir,
                                    'checkpoint_files': checkpoint_files,
                                    'experiment_name': 'unknown',
                                    'timestamp_id': 'unknown'
                                })
            
            return checkpoints
            
        except Exception as e:
            print(f"Error finding checkpoints for experiment '{experiment_name}': {e}")
            return []
    
    def load_experiment_config(self, experiment_name: str) -> Tuple[ConfigManager, TrainingConfigManager]:
        """
        Load experiment configuration.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Tuple of (config_manager, config)
        """
        try:
            config_manager = self.experiment_manager.load_experiment(experiment_name)
            config = config_manager.get_config()
            
            print(f"✅ Loaded experiment configuration: {experiment_name}")
            print(f"   Description: {config.experiment.description}")
            print(f"   Created: {config.experiment.created_at}")
            if hasattr(config.experiment, 'timestamp_id'):
                print(f"   Timestamp ID: {config.experiment.timestamp_id}")
            
            return config_manager, config
            
        except Exception as e:
            print(f"❌ Error loading experiment '{experiment_name}': {e}")
            raise
    
    def validate_hyperparameter_changes(
        self, 
        original_config: TrainingConfigManager, 
        new_config: TrainingConfigManager
    ) -> Dict[str, Any]:
        """
        Validate hyperparameter changes between configurations.
        
        Args:
            original_config: Original configuration
            new_config: New configuration
            
        Returns:
            Validation results
        """
        from dataclasses import asdict
        
        original_dict = asdict(original_config)
        new_dict = asdict(new_config)
        
        validation_result = {
            'is_valid': True,
            'structural_changes': [],
            'non_structural_changes': [],
            'errors': [],
            'warnings': []
        }
        
        # Check structural parameters
        for section, params in self.structural_params.items():
            if section in original_dict and section in new_dict:
                for param in params:
                    if param in original_dict[section] and param in new_dict[section]:
                        if original_dict[section][param] != new_dict[section][param]:
                            validation_result['structural_changes'].append({
                                'section': section,
                                'parameter': param,
                                'original': original_dict[section][param],
                                'new': new_dict[section][param]
                            })
                            validation_result['errors'].append(
                                f"Structural parameter changed: {section}.{param} "
                                f"({original_dict[section][param]} → {new_dict[section][param]})"
                            )
                            validation_result['is_valid'] = False
        
        # Check non-structural parameters
        for section, params in self.non_structural_params.items():
            if section in original_dict and section in new_dict:
                for param in params:
                    if param in original_dict[section] and param in new_dict[section]:
                        if original_dict[section][param] != new_dict[section][param]:
                            validation_result['non_structural_changes'].append({
                                'section': section,
                                'parameter': param,
                                'original': original_dict[section][param],
                                'new': new_dict[section][param]
                            })
                            validation_result['warnings'].append(
                                f"Non-structural parameter changed: {section}.{param} "
                                f"({original_dict[section][param]} → {new_dict[section][param]})"
                            )
        
        return validation_result
    
    def print_validation_results(self, validation_result: Dict[str, Any]):
        """Print validation results in a formatted way."""
        print(f"\n{'='*80}")
        print(f"HYPERPARAMETER VALIDATION RESULTS")
        print(f"{'='*80}")
        
        if validation_result['is_valid']:
            print("✅ VALIDATION PASSED: No structural changes detected")
        else:
            print("❌ VALIDATION FAILED: Structural changes detected")
        
        # Print structural changes
        if validation_result['structural_changes']:
            print(f"\n🚫 STRUCTURAL CHANGES (NOT ALLOWED):")
            print(f"{'Section':<15} {'Parameter':<20} {'Original':<15} {'New':<15}")
            print(f"{'-'*65}")
            for change in validation_result['structural_changes']:
                print(f"{change['section']:<15} {change['parameter']:<20} "
                      f"{str(change['original']):<15} {str(change['new']):<15}")
        
        # Print non-structural changes
        if validation_result['non_structural_changes']:
            print(f"\n⚠️  NON-STRUCTURAL CHANGES (ALLOWED):")
            print(f"{'Section':<15} {'Parameter':<20} {'Original':<15} {'New':<15}")
            print(f"{'-'*65}")
            for change in validation_result['non_structural_changes']:
                print(f"{change['section']:<15} {change['parameter']:<20} "
                      f"{str(change['original']):<15} {str(change['new']):<15}")
        
        # Print errors
        if validation_result['errors']:
            print(f"\n❌ ERRORS:")
            for error in validation_result['errors']:
                print(f"  - {error}")
        
        # Print warnings
        if validation_result['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warning in validation_result['warnings']:
                print(f"  - {warning}")
        
        print(f"{'='*80}")
    
    def create_resume_config(
        self, 
        experiment_name: str, 
        checkpoint_path: str,
        hyperparameter_overrides: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Create a configuration for resuming training.
        
        Args:
            experiment_name: Name of the experiment
            checkpoint_path: Path to checkpoint file
            hyperparameter_overrides: Dictionary of hyperparameter overrides
            
        Returns:
            Tuple of (config_path, validation_result)
        """
        # Load original experiment configuration
        config_manager, original_config = self.load_experiment_config(experiment_name)
        
        # Create new configuration with overrides
        from dataclasses import asdict
        new_config_dict = asdict(config_manager.get_config())
        
        if hyperparameter_overrides:
            # Apply hyperparameter overrides
            for section, params in hyperparameter_overrides.items():
                if section in new_config_dict:
                    for param, value in params.items():
                        if param in new_config_dict[section]:
                            new_config_dict[section][param] = value
                        else:
                            print(f"Warning: Parameter {section}.{param} not found in configuration")
                else:
                    print(f"Warning: Section {section} not found in configuration")
        
        # Create new config manager by saving to temporary file
        import tempfile
        temp_config_path = f"temp_resume_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(new_config_dict, f, indent=2, ensure_ascii=False)
        
        new_config_manager = ConfigManager(temp_config_path)
        new_config = new_config_manager.get_config()
        
        # Clean up temporary file
        os.remove(temp_config_path)
        
        # Validate changes
        validation_result = self.validate_hyperparameter_changes(original_config, new_config)
        
        # Print validation results
        self.print_validation_results(validation_result)
        
        if not validation_result['is_valid']:
            raise ValueError("Cannot resume training: structural parameters have changed")
        
        # Create resume config file in experiments/configs directory
        resume_config_filename = f"resume_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        resume_config_path = os.path.join("experiments", "configs", resume_config_filename)
        
        # Ensure experiments/configs directory exists
        os.makedirs(os.path.dirname(resume_config_path), exist_ok=True)
        
        new_config_manager.save_config(resume_config_path)
        
        # Add checkpoint information to config
        new_config_dict['resume'] = {
            'checkpoint_path': checkpoint_path,
            'original_experiment': experiment_name,
            'resume_timestamp': datetime.now().isoformat(),
            'validation_result': validation_result
        }
        
        # Save updated config
        with open(resume_config_path, 'w', encoding='utf-8') as f:
            json.dump(new_config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Resume configuration created: {resume_config_path}")
        
        return resume_config_path, validation_result
    
    def list_experiment_checkpoints(self, experiment_name: str):
        """List all checkpoints for an experiment."""
        checkpoints = self.find_experiment_checkpoints(experiment_name)
        
        if not checkpoints:
            print(f"No checkpoints found for experiment '{experiment_name}'")
            return
        
        print(f"\n{'='*80}")
        print(f"CHECKPOINTS FOR EXPERIMENT: {experiment_name}")
        print(f"{'='*80}")
        
        for checkpoint_info in checkpoints:
            print(f"\n📁 Run Directory: {checkpoint_info['run_name']}")
            print(f"   Path: {checkpoint_info['run_dir']}")
            print(f"   Timestamp: {checkpoint_info['timestamp_id']}")
            print(f"   Checkpoint Files:")
            
            for checkpoint_file in checkpoint_info['checkpoint_files']:
                checkpoint_path = os.path.join(checkpoint_info['run_dir'], checkpoint_file)
                file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
                print(f"     - {checkpoint_file} ({file_size:.2f} MB)")
        
        print(f"{'='*80}")
    
    def get_checkpoint_metrics(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get metrics from checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary containing checkpoint metrics
        """
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            metrics = {
                'epoch': checkpoint_data.get('epoch', 'unknown'),
                'loss': checkpoint_data.get('loss', 'unknown'),
                'timestamp': checkpoint_data.get('timestamp', 'unknown'),
                'is_best': checkpoint_data.get('is_best', False)
            }
            
            # Get training history if available
            run_dir = os.path.dirname(checkpoint_path)
            training_history_path = os.path.join(run_dir, 'training_history.json')
            
            if os.path.exists(training_history_path):
                with open(training_history_path, 'r', encoding='utf-8') as f:
                    training_history = json.load(f)
                
                if training_history:
                    metrics['training_history'] = training_history
                    metrics['total_epochs_trained'] = len(training_history)
                    metrics['best_val_loss'] = min([epoch.get('val_loss', float('inf')) for epoch in training_history])
                    metrics['final_train_loss'] = training_history[-1].get('train_loss', 'unknown')
                    metrics['final_val_loss'] = training_history[-1].get('val_loss', 'unknown')
            
            return metrics
            
        except Exception as e:
            print(f"Warning: Could not load checkpoint metrics: {e}")
            return {'error': str(e)}
    
    def print_checkpoint_status(self, checkpoint_path: str):
        """Print checkpoint status and metrics."""
        print(f"\n{'='*80}")
        print(f"CHECKPOINT STATUS")
        print(f"{'='*80}")
        print(f"Checkpoint: {checkpoint_path}")
        
        metrics = self.get_checkpoint_metrics(checkpoint_path)
        
        if 'error' in metrics:
            print(f"❌ Error loading checkpoint: {metrics['error']}")
            return
        
        print(f"📊 Current Status:")
        print(f"   Epoch: {metrics.get('epoch', 'unknown')}")
        print(f"   Loss: {metrics.get('loss', 'unknown')}")
        print(f"   Timestamp: {metrics.get('timestamp', 'unknown')}")
        print(f"   Is Best Model: {metrics.get('is_best', False)}")
        
        if 'training_history' in metrics:
            print(f"\n📈 Training History:")
            print(f"   Total Epochs Trained: {metrics.get('total_epochs_trained', 'unknown')}")
            print(f"   Best Validation Loss: {metrics.get('best_val_loss', 'unknown')}")
            print(f"   Final Training Loss: {metrics.get('final_train_loss', 'unknown')}")
            print(f"   Final Validation Loss: {metrics.get('final_val_loss', 'unknown')}")
            
            # Show recent training trend
            history = metrics['training_history']
            if len(history) >= 3:
                recent_losses = [epoch.get('val_loss', float('inf')) for epoch in history[-3:]]
                if all(isinstance(x, (int, float)) for x in recent_losses):
                    trend = "📈 Improving" if recent_losses[-1] < recent_losses[0] else "📉 Degrading" if recent_losses[-1] > recent_losses[0] else "➡️ Stable"
                    print(f"   Recent Trend: {trend}")
        
        print(f"{'='*80}")

    def resume_training(
        self,
        experiment_name: str,
        checkpoint_path: str,
        hyperparameter_overrides: Optional[Dict[str, Any]] = None,
        new_epochs: Optional[int] = None
    ) -> str:
        """
        Resume training from checkpoint with modified hyperparameters.
        
        Args:
            experiment_name: Name of the experiment
            checkpoint_path: Path to checkpoint file
            hyperparameter_overrides: Dictionary of hyperparameter overrides
            new_epochs: New number of epochs (optional)
            
        Returns:
            Path to resume configuration file
        """
        print(f"🔄 Resuming training from checkpoint...")
        print(f"   Experiment: {experiment_name}")
        print(f"   Checkpoint: {checkpoint_path}")
        
        # Show checkpoint status
        self.print_checkpoint_status(checkpoint_path)
        
        if hyperparameter_overrides:
            print(f"\n🔧 Hyperparameter overrides: {hyperparameter_overrides}")
        
        # Add epochs override if specified
        if new_epochs:
            if hyperparameter_overrides is None:
                hyperparameter_overrides = {}
            if 'training' not in hyperparameter_overrides:
                hyperparameter_overrides['training'] = {}
            hyperparameter_overrides['training']['epochs'] = new_epochs
        
        # Create resume configuration
        resume_config_path, validation_result = self.create_resume_config(
            experiment_name=experiment_name,
            checkpoint_path=checkpoint_path,
            hyperparameter_overrides=hyperparameter_overrides
        )
        
        print(f"\n✅ Resume configuration ready!")
        print(f"   Config file: {resume_config_path}")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Validation: {'PASSED' if validation_result['is_valid'] else 'FAILED'}")
        
        return resume_config_path


def main():
    """Test ResumeManager functionality."""
    print("Testing ResumeManager...")
    
    manager = ResumeManager()
    
    # List available experiments
    print("\n1. Listing available experiments...")
    manager.experiment_manager.print_experiments()
    
    # Test with small_test experiment
    experiment_name = "small_test"
    
    print(f"\n2. Finding checkpoints for experiment: {experiment_name}")
    manager.list_experiment_checkpoints(experiment_name)
    
    # Test resume configuration creation
    print(f"\n3. Testing resume configuration creation...")
    try:
        # Find a checkpoint
        checkpoints = manager.find_experiment_checkpoints(experiment_name)
        if checkpoints:
            checkpoint_info = checkpoints[0]
            checkpoint_file = checkpoint_info['checkpoint_files'][0]
            checkpoint_path = os.path.join(checkpoint_info['run_dir'], checkpoint_file)
            
            print(f"Using checkpoint: {checkpoint_path}")
            
            # Test with hyperparameter overrides
            hyperparameter_overrides = {
                'training': {
                    'learning_rate': 0.0002,  # Non-structural change
                    'epochs': 10
                },
                'scheduler': {
                    'type': 'cosine'  # Non-structural change
                }
            }
            
            resume_config_path = manager.resume_training(
                experiment_name=experiment_name,
                checkpoint_path=checkpoint_path,
                hyperparameter_overrides=hyperparameter_overrides
            )
            
            print(f"✅ Resume configuration created: {resume_config_path}")
            
        else:
            print(f"No checkpoints found for experiment '{experiment_name}'")
    
    except Exception as e:
        print(f"❌ Error testing resume functionality: {e}")
    
    print("\nResumeManager test completed!")


if __name__ == "__main__":
    main()
