#!/usr/bin/env python3
"""
Unified Training Entry Point - Simplified Version

This script provides a clean, structured entry point for all training operations.
"""

import os
import sys
import json
import argparse
import subprocess
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from utils.colab_manager import ColabManager
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False

# Import unified logger
from utils.logger import get_logger, LogLevel, LogCategory, OperationContext, track_operation

from utils.config_manager import ConfigManager
from utils.experiment_manager import ExperimentManager
from utils.resume_manager import ResumeManager
from utils.checkpoint_manager import CheckpointManager
from utils.config_version_manager import ConfigVersionManager


class UnifiedTrainer:
    """
    Unified Training System with structured logging.
    
    Features:
    - Single entry point for all training operations
    - Intelligent resource selection (cloud GPU -> local GPU -> local CPU)
    - Experiment management integration
    - Resume training support
    - Configuration management
    - Checkpoint management
    """
    
    def __init__(self, verbose: bool = True, quiet: bool = False):
        """Initialize Unified Trainer."""
        self.logger = get_logger(verbose=verbose, quiet=quiet)
        
        with OperationContext("Initializing components", LogCategory.SYSTEM):
            self.experiment_manager = ExperimentManager()
            self.resume_manager = ResumeManager()
            self.checkpoint_manager = CheckpointManager()
            self.config_version_manager = ConfigVersionManager()
            
            # Resource detection
            self.resource_info = self.detect_resources()
            self.selected_resource = None
            self.colab_manager = None
            
            # Initialize ColabCode if available
            if COLAB_AVAILABLE:
                self.colab_manager = ColabManager(
                    password="transformer123",
                    port=8888
                )
        
        self.logger.success(LogCategory.SYSTEM, "Unified Trainer initialized")
        self._log_resources()
    
    def _log_resources(self):
        """Log available resources."""
        self.logger.info(LogCategory.SYSTEM, "Available resources:")
        for resource, info in self.resource_info.items():
            status = "available" if info['status'] == 'available' else "unavailable"
            self.logger.info(LogCategory.SYSTEM, f"  {resource.upper()}: {info['description']} ({status})")
    
    def detect_resources(self) -> Dict[str, Dict[str, Any]]:
        """Detect available computing resources."""
        resources = {}
        
        # Check ColabCode availability
        if COLAB_AVAILABLE:
            resources['colabcode'] = {
                'status': 'available',
                'description': 'ColabCode SSH connection available',
                'type': 'cloud_gpu',
                'priority': 1
            }
        else:
            resources['colabcode'] = {
                'status': 'unavailable',
                'description': 'ColabCode not installed',
                'type': 'cloud_gpu',
                'priority': 1
            }
        
        # Check local GPU
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            resources['local_gpu'] = {
                'status': 'available',
                'description': f'Local GPU: {gpu_name} ({gpu_count} devices, {gpu_memory:.1f}GB)',
                'type': 'local_gpu',
                'priority': 2,
                'device_count': gpu_count,
                'device_name': gpu_name,
                'memory_gb': gpu_memory
            }
        else:
            resources['local_gpu'] = {
                'status': 'unavailable',
                'description': 'No local GPU available',
                'type': 'local_gpu',
                'priority': 2
            }
        
        # Check local CPU
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        resources['local_cpu'] = {
            'status': 'available',
            'description': f'Local CPU: {cpu_count} cores',
            'type': 'local_cpu',
            'priority': 3,
            'core_count': cpu_count
        }
        
        return resources
    
    def select_best_resource(self, force_local: bool = False, force_cloud: bool = False) -> tuple:
        """Select the best available resource."""
        if force_cloud:
            if self.resource_info['colabcode']['status'] == 'available':
                return 'colabcode', self.resource_info['colabcode']
            else:
                self.logger.error(LogCategory.SYSTEM, "Cloud training requested but ColabCode not available!")
                return None, None
        
        if force_local:
            # Skip cloud, use local resources
            for resource_name, resource_info in self.resource_info.items():
                if resource_info['type'] in ['local_gpu', 'local_cpu'] and resource_info['status'] == 'available':
                    return resource_name, resource_info
        
        # Auto-select best available resource
        sorted_resources = sorted(
            self.resource_info.items(),
            key=lambda x: x[1]['priority']
        )
        
        for resource_name, resource_info in sorted_resources:
            if resource_info['status'] == 'available':
                return resource_name, resource_info
        
        return 'local_cpu', self.resource_info['local_cpu']
    
    def print_resource_status(self, resource_name: str, resource_info: Dict[str, Any]):
        """Print resource status information."""
        self.logger.print_header("TRAINING RESOURCE STATUS")
        
        if resource_info['type'] == 'cloud_gpu':
            self.logger.info(LogCategory.SYSTEM, "Environment: CLOUD GPU (Google Colab)")
            self.logger.info(LogCategory.SYSTEM, "Device: CUDA")
            self.logger.info(LogCategory.SYSTEM, "Resource: ColabCode SSH Connection")
            self.logger.info(LogCategory.SYSTEM, "GPU: Free Google Colab GPU")
            self.logger.info(LogCategory.SYSTEM, "Location: Remote Cloud Server")
        elif resource_info['type'] == 'local_gpu':
            self.logger.info(LogCategory.SYSTEM, "Environment: LOCAL")
            self.logger.info(LogCategory.SYSTEM, "Device: CUDA")
            self.logger.info(LogCategory.SYSTEM, "Resource: Local GPU")
            self.logger.info(LogCategory.SYSTEM, f"GPU: {resource_info.get('device_name', 'Unknown')}")
            self.logger.info(LogCategory.SYSTEM, f"Memory: {resource_info.get('memory_gb', 0):.1f}GB")
        else:  # local_cpu
            self.logger.info(LogCategory.SYSTEM, "Environment: LOCAL")
            self.logger.info(LogCategory.SYSTEM, "Device: CPU")
            self.logger.info(LogCategory.SYSTEM, "Resource: Local CPU")
            self.logger.info(LogCategory.SYSTEM, f"Cores: {resource_info.get('core_count', 1)} cores")
            self.logger.info(LogCategory.SYSTEM, "Memory: System RAM")
        
        self.logger.success(LogCategory.SYSTEM, f"Training will run on: {resource_name.upper()}")
    
    @track_operation("Training execution", LogCategory.TRAINING)
    def run_training(self, config_path: str, force_local: bool = False, force_cloud: bool = False, experiment_name: str = None, training_environment: str = "unknown"):
        """Run training with the specified configuration."""
        self.logger.info(LogCategory.TRAINING, f"Starting training with config: {config_path}")
        
        # Select resource
        resource_name, resource_info = self.select_best_resource(force_local, force_cloud)
        if resource_name is None:
            self.logger.error(LogCategory.TRAINING, "No suitable resource available!")
            return False
        
        self.print_resource_status(resource_name, resource_info)
        
        # Determine training method
        if resource_info['type'] == 'cloud_gpu':
            return self.run_cloud_training(config_path, experiment_name)
        else:
            return self.run_local_training(config_path, resource_info, experiment_name)
    
    def run_cloud_training(self, config_path: str, experiment_name: str = None) -> bool:
        """Run training on cloud GPU."""
        if not self.colab_manager:
            self.logger.error(LogCategory.CLOUD, "ColabCode not available for cloud training.")
            return False
        
        self.logger.info(LogCategory.CLOUD, "Setting up cloud training...")
        
        # Try to start ColabCode session
        success = self.colab_manager.start_colab_session(gpu=True)
        if not success:
            self.logger.error(LogCategory.CLOUD, "Failed to start cloud training session.")
            return False
        
        # Get and display GPU information
        self.logger.info(LogCategory.CLOUD, "Checking GPU availability...")
        gpu_info = self.colab_manager.get_gpu_info()
        if gpu_info['available']:
            self.logger.success(LogCategory.CLOUD, f"GPU Ready: {gpu_info['name']}")
            self.logger.info(LogCategory.CLOUD, f"Memory: {gpu_info['memory_free_gb']:.1f}GB free / {gpu_info['memory_total_gb']:.1f}GB total")
            self.logger.info(LogCategory.CLOUD, f"Utilization: {gpu_info['utilization_percent']}%")
        else:
            self.logger.warning(LogCategory.CLOUD, f"GPU Status: {gpu_info['message']}")
        
        self.logger.success(LogCategory.CLOUD, "Cloud training environment ready!")
        
        # Sync project to cloud
        self.logger.info(LogCategory.CLOUD, "Syncing project to cloud...")
        sync_success = self.colab_manager.sync_project_to_colab(
            local_project_dir=".",
            remote_project_dir="/content/transformer"
        )
        
        if not sync_success:
            self.logger.error(LogCategory.CLOUD, "Failed to sync project to cloud.")
            return False
        
        # Start remote training
        self.logger.info(LogCategory.CLOUD, "Starting remote training...")
        training_success = self.colab_manager.start_remote_training(
            experiment_name=experiment_name or "cloud_training",
            config_path=config_path
        )
        
        if training_success:
            self.logger.success(LogCategory.CLOUD, "Cloud training started successfully!")
            
            # Download results
            self.logger.info(LogCategory.CLOUD, "Downloading results...")
            self.colab_manager.download_results(
                experiment_name="cloud_training",
                local_results_dir="./checkpoints"
            )
            self.logger.success(LogCategory.CLOUD, "Results downloaded!")
            
            # Clean up temporary files
            self.logger.info(LogCategory.CLOUD, "Cleaning up temporary files...")
            self.colab_manager.cleanup_temp_files()
        else:
            self.logger.error(LogCategory.CLOUD, "Cloud training failed.")
        
        return training_success
    
    def run_local_training(self, config_path: str, resource_info: Dict[str, Any], experiment_name: str = None) -> bool:
        """Run training on local device."""
        # Determine device
        if resource_info['type'] == 'local_gpu':
            device = 'cuda'
            self.logger.info(LogCategory.TRAINING, f"Using local GPU: {resource_info.get('device_name', 'Unknown')}")
        else:
            device = 'cpu'
            self.logger.info(LogCategory.TRAINING, f"Using local CPU: {resource_info.get('core_count', 1)} cores")
        
        self.logger.info(LogCategory.TRAINING, f"Starting local training on {device}...")
        
        # Run training script
        try:
            # Set environment variable for experiment name if provided
            env = os.environ.copy()
            if experiment_name:
                env['EXPERIMENT_NAME'] = experiment_name
                self.logger.info(LogCategory.TRAINING, f"Linking training to experiment: {experiment_name}")
            
            result = subprocess.run([
                'python', 'train.py', '--config', config_path
            ], check=True, capture_output=True, text=True, env=env)
            
            self.logger.success(LogCategory.TRAINING, "Local training completed successfully!")
            if self.logger.verbose:
                print(result.stdout)
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(LogCategory.TRAINING, f"Local training failed: {e}")
            if self.logger.verbose:
                print(f"Error output: {e.stderr}")
            return False
    
    # Experiment management methods
    def list_experiments(self):
        """List all available experiments."""
        with OperationContext("Listing experiments", LogCategory.EXPERIMENT):
            experiments = self.experiment_manager.list_experiments()
            if not experiments:
                self.logger.info(LogCategory.EXPERIMENT, "No experiments found")
                return
            
            self.logger.print_table(
                ["Name", "Description", "Created", "Version", "Timestamp ID"],
                [[exp['name'], exp['description'], exp['created_at'][:19], exp['version'], exp['timestamp_id']] 
                 for exp in experiments],
                "AVAILABLE EXPERIMENTS"
            )
    
    def show_experiment(self, experiment_name: str):
        """Show detailed information about an experiment."""
        with OperationContext(f"Showing experiment: {experiment_name}", LogCategory.EXPERIMENT):
            try:
                experiment_info = self.experiment_manager.get_experiment_info(experiment_name)
                self.logger.print_summary(f"Experiment Details: {experiment_name}", {
                    "Name": experiment_info['name'],
                    "Description": experiment_info['description'],
                    "Version": experiment_info['version'],
                    "Created": experiment_info['created_at'],
                    "Timestamp ID": experiment_info['timestamp_id']
                })
            except Exception as e:
                self.logger.error(LogCategory.EXPERIMENT, f"Failed to show experiment: {e}")
    
    def create_experiment(self, name: str, description: str = "", **overrides):
        """Create a new experiment."""
        with OperationContext(f"Creating experiment: {name}", LogCategory.EXPERIMENT):
            try:
                experiment_path, timestamp_id = self.experiment_manager.create_experiment(
                    name=name,
                    description=description,
                    **overrides
                )
                self.logger.success(LogCategory.EXPERIMENT, f"Experiment created: {name}")
                self.logger.info(LogCategory.EXPERIMENT, f"Config file: {experiment_path}")
                self.logger.info(LogCategory.EXPERIMENT, f"Timestamp ID: {timestamp_id}")
            except Exception as e:
                self.logger.error(LogCategory.EXPERIMENT, f"Failed to create experiment: {e}")
    
    def create_derived_experiment(self, source_experiment: str, name: str, description: str,
                                new_environment: str = "local_gpu", **overrides):
        """Create a derived experiment from an existing experiment."""
        with OperationContext(f"Creating derived experiment: {name}", LogCategory.EXPERIMENT):
            try:
                experiment_path, timestamp_id = self.experiment_manager.create_derived_experiment(
                    source_experiment_name=source_experiment,
                    new_name=name,
                    new_description=description,
                    new_training_environment=new_environment,
                    **overrides
                )
                self.logger.success(LogCategory.EXPERIMENT, f"Derived experiment created: {name}")
                self.logger.info(LogCategory.EXPERIMENT, f"Source experiment: {source_experiment}")
                self.logger.info(LogCategory.EXPERIMENT, f"New environment: {new_environment}")
                self.logger.info(LogCategory.EXPERIMENT, f"Config file: {experiment_path}")
                self.logger.info(LogCategory.EXPERIMENT, f"Timestamp ID: {timestamp_id}")
            except Exception as e:
                self.logger.error(LogCategory.EXPERIMENT, f"Failed to create derived experiment: {e}")
    
    # Checkpoint management methods
    def list_checkpoints(self, experiment_name: str = None):
        """List available checkpoints."""
        with OperationContext("Listing checkpoints", LogCategory.CHECKPOINT):
            try:
                checkpoints = self.checkpoint_manager.list_checkpoints(experiment_name)
                if not checkpoints:
                    self.logger.info(LogCategory.CHECKPOINT, "No checkpoints found")
                    return
                
                # Format checkpoint data for table
                table_data = []
                for run_dir, run_info in checkpoints.items():
                    for checkpoint in run_info['checkpoints']:
                        table_data.append([
                            run_dir,
                            checkpoint['filename'],
                            str(checkpoint['epoch']),
                            f"{checkpoint['loss']:.4f}",
                            checkpoint['timestamp'][:19],
                            "Yes" if checkpoint['is_best'] else "No"
                        ])
                
                self.logger.print_table(
                    ["Run Directory", "Checkpoint", "Epoch", "Loss", "Timestamp", "Best"],
                    table_data,
                    "AVAILABLE CHECKPOINTS"
                )
            except Exception as e:
                self.logger.error(LogCategory.CHECKPOINT, f"Failed to list checkpoints: {e}")
    
    def create_resume_config(self, experiment_name: str, checkpoint_path: str, overrides: Dict[str, Any] = None):
        """Create a resume configuration."""
        with OperationContext(f"Creating resume config for: {experiment_name}", LogCategory.CHECKPOINT):
            try:
                resume_config_path, validation_result = self.resume_manager.create_resume_config(
                    experiment_name=experiment_name,
                    checkpoint_path=checkpoint_path,
                    hyperparameter_overrides=overrides
                )
                
                self.logger.success(LogCategory.CHECKPOINT, f"Resume configuration created: {resume_config_path}")
                self.logger.info(LogCategory.CHECKPOINT, f"Validation: {'PASSED' if validation_result['is_valid'] else 'FAILED'}")
                
                if validation_result['errors']:
                    for error in validation_result['errors']:
                        self.logger.error(LogCategory.VALIDATION, error)
                
                if validation_result['warnings']:
                    for warning in validation_result['warnings']:
                        self.logger.warning(LogCategory.VALIDATION, warning)
                        
            except Exception as e:
                self.logger.error(LogCategory.CHECKPOINT, f"Failed to create resume config: {e}")
    
    def validate_changes(self, experiment_name: str, overrides: Dict[str, Any]):
        """Validate hyperparameter changes."""
        with OperationContext(f"Validating changes for: {experiment_name}", LogCategory.VALIDATION):
            try:
                validation_result = self.resume_manager.validate_hyperparameter_changes(
                    experiment_name=experiment_name,
                    hyperparameter_overrides=overrides
                )
                
                if validation_result['is_valid']:
                    self.logger.success(LogCategory.VALIDATION, "Validation PASSED: No structural changes detected")
                else:
                    self.logger.error(LogCategory.VALIDATION, "Validation FAILED: Structural changes detected")
                
                if validation_result['errors']:
                    for error in validation_result['errors']:
                        self.logger.error(LogCategory.VALIDATION, error)
                
                if validation_result['warnings']:
                    for warning in validation_result['warnings']:
                        self.logger.warning(LogCategory.VALIDATION, warning)
                        
            except Exception as e:
                self.logger.error(LogCategory.VALIDATION, f"Failed to validate changes: {e}")
    
    # System maintenance methods
    def list_config_versions(self):
        """List configuration versions."""
        with OperationContext("Listing config versions", LogCategory.CONFIG):
            try:
                versions = self.config_version_manager.list_config_versions()
                if not versions:
                    self.logger.info(LogCategory.CONFIG, "No configuration versions found")
                    return
                
                table_data = [[v['version_id'], v['description'], v['created_at'][:19], ', '.join(v['tags'])] 
                             for v in versions]
                
                self.logger.print_table(
                    ["Version ID", "Description", "Created", "Tags"],
                    table_data,
                    "CONFIGURATION VERSIONS"
                )
            except Exception as e:
                self.logger.error(LogCategory.CONFIG, f"Failed to list config versions: {e}")
    
    def cleanup_checkpoints(self, keep_latest_runs: int = 3, keep_latest_checkpoints: int = 5):
        """Clean up old checkpoints."""
        with OperationContext("Cleaning up checkpoints", LogCategory.CLEANUP):
            try:
                cleanup_result = self.checkpoint_manager.cleanup_checkpoints(
                    keep_latest_runs=keep_latest_runs,
                    keep_latest_checkpoints=keep_latest_checkpoints
                )
                
                self.logger.success(LogCategory.CLEANUP, "Checkpoint cleanup completed")
                self.logger.info(LogCategory.CLEANUP, f"Runs removed: {cleanup_result.get('runs_cleaned', 0)}")
                self.logger.info(LogCategory.CLEANUP, f"Checkpoints removed: {cleanup_result.get('checkpoints_cleaned', 0)}")
                self.logger.info(LogCategory.CLEANUP, f"Space freed: {cleanup_result.get('space_freed_mb', 0):.2f} MB")
            except Exception as e:
                self.logger.error(LogCategory.CLEANUP, f"Failed to cleanup checkpoints: {e}")
    
    def cleanup_all(self):
        """Clean up all training artifacts."""
        with OperationContext("Complete cleanup", LogCategory.CLEANUP):
            self.logger.print_header("COMPLETE CLEANUP - ALL TRAINING ARTIFACTS")
            
            total_space_freed = 0
            
            # Cleanup experiments
            self.logger.info(LogCategory.CLEANUP, "Cleaning up experiments...")
            try:
                experiments_dir = "experiments"
                if os.path.exists(experiments_dir):
                    import shutil
                    # Get size before deletion
                    total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                    for dirpath, dirnames, filenames in os.walk(experiments_dir)
                                    for filename in filenames) / (1024 * 1024)  # MB
                    
                    shutil.rmtree(experiments_dir)
                    os.makedirs(experiments_dir, exist_ok=True)
                    os.makedirs(os.path.join(experiments_dir, "configs"), exist_ok=True)
                    os.makedirs(os.path.join(experiments_dir, "results"), exist_ok=True)
                    
                    self.logger.success(LogCategory.CLEANUP, "Experiments directory cleaned")
                    self.logger.info(LogCategory.CLEANUP, f"Space freed: {total_size:.2f} MB")
                    total_space_freed += total_size
                else:
                    self.logger.info(LogCategory.CLEANUP, "No experiments directory found")
            except Exception as e:
                self.logger.error(LogCategory.CLEANUP, f"Error cleaning experiments: {e}")
            
            # Cleanup checkpoints
            self.logger.info(LogCategory.CLEANUP, "Cleaning up checkpoints...")
            try:
                checkpoints_dir = "checkpoints"
                if os.path.exists(checkpoints_dir):
                    import shutil
                    # Get size before deletion
                    total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                    for dirpath, dirnames, filenames in os.walk(checkpoints_dir)
                                    for filename in filenames) / (1024 * 1024)  # MB
                    
                    shutil.rmtree(checkpoints_dir)
                    os.makedirs(checkpoints_dir, exist_ok=True)
                    
                    self.logger.success(LogCategory.CLEANUP, "Checkpoints directory cleaned")
                    self.logger.info(LogCategory.CLEANUP, f"Space freed: {total_size:.2f} MB")
                    total_space_freed += total_size
                else:
                    self.logger.info(LogCategory.CLEANUP, "No checkpoints directory found")
            except Exception as e:
                self.logger.error(LogCategory.CLEANUP, f"Error cleaning checkpoints: {e}")
            
            # Cleanup config history
            self.logger.info(LogCategory.CLEANUP, "Cleaning up config history...")
            try:
                config_history_dir = "config_history"
                if os.path.exists(config_history_dir):
                    import shutil
                    # Get size before deletion
                    total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                    for dirpath, dirnames, filenames in os.walk(config_history_dir)
                                    for filename in filenames) / (1024 * 1024)  # MB
                    
                    shutil.rmtree(config_history_dir)
                    os.makedirs(config_history_dir, exist_ok=True)
                    os.makedirs(os.path.join(config_history_dir, "versions"), exist_ok=True)
                    os.makedirs(os.path.join(config_history_dir, "links"), exist_ok=True)
                    
                    self.logger.success(LogCategory.CLEANUP, "Config history directory cleaned")
                    self.logger.info(LogCategory.CLEANUP, f"Space freed: {total_size:.2f} MB")
                    total_space_freed += total_size
                else:
                    self.logger.info(LogCategory.CLEANUP, "No config history directory found")
            except Exception as e:
                self.logger.error(LogCategory.CLEANUP, f"Error cleaning config history: {e}")
            
            # Cleanup results
            self.logger.info(LogCategory.CLEANUP, "Cleaning up results...")
            try:
                results_dir = "results"
                if os.path.exists(results_dir):
                    import shutil
                    shutil.rmtree(results_dir)
                    self.logger.success(LogCategory.CLEANUP, "Results directory cleaned")
                else:
                    self.logger.info(LogCategory.CLEANUP, "No results directory found")
            except Exception as e:
                self.logger.error(LogCategory.CLEANUP, f"Error cleaning results: {e}")
            
            self.logger.print_header("COMPLETE CLEANUP FINISHED!")
            self.logger.success(LogCategory.CLEANUP, f"Total space freed: {total_space_freed:.2f} MB")
            self.logger.success(LogCategory.CLEANUP, "All training artifacts have been removed.")
    
    # Cloud training monitoring methods
    def check_cloud_status(self):
        """Check cloud training status."""
        if not self.colab_manager:
            self.logger.error(LogCategory.CLOUD, "ColabCode not available")
            return
        
        with OperationContext("Checking cloud status", LogCategory.CLOUD):
            status = self.colab_manager.check_training_status()
            if status['status'] == 'active':
                self.logger.success(LogCategory.CLOUD, "Cloud training is active")
                self.logger.info(LogCategory.CLOUD, f"Experiment: {status.get('experiment_name', 'Unknown')}")
                self.logger.info(LogCategory.CLOUD, f"Elapsed time: {status.get('elapsed_time', 'Unknown')}")
            else:
                self.logger.info(LogCategory.CLOUD, "No active cloud training session")
    
    def get_cloud_logs(self):
        """Get cloud training logs."""
        if not self.colab_manager:
            self.logger.error(LogCategory.CLOUD, "ColabCode not available")
            return
        
        with OperationContext("Getting cloud logs", LogCategory.CLOUD):
            logs = self.colab_manager.get_training_logs()
            if logs:
                self.logger.info(LogCategory.CLOUD, "Recent training logs:")
                for log in logs[-10:]:  # Show last 10 lines
                    self.logger.info(LogCategory.CLOUD, log)
            else:
                self.logger.info(LogCategory.CLOUD, "No logs available")
    
    def stop_cloud_training(self):
        """Stop cloud training."""
        if not self.colab_manager:
            self.logger.error(LogCategory.CLOUD, "ColabCode not available")
            return
        
        with OperationContext("Stopping cloud training", LogCategory.CLOUD):
            success = self.colab_manager.stop_training()
            if success:
                self.logger.success(LogCategory.CLOUD, "Cloud training stopped")
            else:
                self.logger.error(LogCategory.CLOUD, "Failed to stop cloud training")
    
    # Colab automation methods
    def create_colab_notebook(self):
        """Create Colab automation notebook."""
        with OperationContext("Creating Colab notebook", LogCategory.CLOUD):
            try:
                import json
                
                notebook_content = {
                    "cells": [
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": [
                                "# 🤖 Transformer 训练自动化\n",
                                "\n",
                                "这个 Notebook 会自动完成以下步骤：\n",
                                "1. 📥 克隆代码仓库\n",
                                "2. 📦 安装依赖\n",
                                "3. 🏃 执行训练\n",
                                "4. 📤 下载结果\n"
                            ]
                        },
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": [
                                "# 1. 克隆代码仓库\n",
                                "!git clone https://github.com/Shanlans/transformer.git\n",
                                "!cd transformer && ls -la"
                            ]
                        },
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": [
                                "# 2. 安装依赖\n",
                                "!cd transformer && pip install -r requirements.txt\n",
                                "!cd transformer && pip install colabcode"
                            ]
                        },
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": [
                                "# 3. 检查环境\n",
                                "import torch\n",
                                "print(f\"PyTorch 版本: {torch.__version__}\")\n",
                                "print(f\"CUDA 可用: {torch.cuda.is_available()}\")\n",
                                "if torch.cuda.is_available():\n",
                                "    print(f\"GPU 设备: {torch.cuda.get_device_name(0)}\")\n",
                                "    print(f\"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB\")"
                            ]
                        },
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": [
                                "# 4. 执行训练\n",
                                "!cd transformer && python run.py --force-cloud --use-default"
                            ]
                        },
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": [
                                "# 5. 检查训练结果\n",
                                "!cd transformer && ls -la checkpoints/\n",
                                "!cd transformer && find checkpoints/ -name \"*.pt\" -o -name \"*.json\" -o -name \"*.png\""
                            ]
                        },
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": [
                                "# 6. 压缩结果\n",
                                "!cd transformer && zip -r training_results.zip checkpoints/ config_history/ experiments/\n",
                                "!cd transformer && ls -la *.zip"
                            ]
                        },
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": [
                                "# 7. 下载结果到本地\n",
                                "from google.colab import files\n",
                                "files.download('/content/transformer/training_results.zip')\n",
                                "print(\"✅ 结果已下载到本地！\")"
                            ]
                        },
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": [
                                "# 8. 显示训练摘要\n",
                                "import json\n",
                                "import glob\n",
                                "import os\n",
                                "\n",
                                "# 查找最新的训练历史\n",
                                "history_files = glob.glob('/content/transformer/checkpoints/run_*/training_history.json')\n",
                                "if history_files:\n",
                                "    latest_history = max(history_files, key=os.path.getctime)\n",
                                "    with open(latest_history, 'r') as f:\n",
                                "        history = json.load(f)\n",
                                "    \n",
                                "    print(\"📊 训练摘要:\")\n",
                                "    print(f\"   总轮数: {len(history.get('epochs', []))}\")\n",
                                "    if history.get('epochs'):\n",
                                "        print(f\"   最终损失: {history['epochs'][-1].get('train_loss', 'N/A'):.4f}\")\n",
                                "        losses = [e.get('train_loss', float('inf')) for e in history['epochs'] if 'train_loss' in e]\n",
                                "        if losses:\n",
                                "            print(f\"   最佳损失: {min(losses):.4f}\")\n",
                                "else:\n",
                                "    print(\"⚠️ 未找到训练历史文件\")"
                            ]
                        }
                    ],
                    "metadata": {
                        "colab": {
                            "provenance": [],
                            "toc_visible": True
                        },
                        "kernelspec": {
                            "display_name": "Python 3",
                            "name": "python3"
                        },
                        "language_info": {
                            "name": "python"
                        }
                    },
                    "nbformat": 4,
                    "nbformat_minor": 0
                }
                
                # 保存 Notebook
                notebook_path = "colab_automation.ipynb"
                with open(notebook_path, 'w', encoding='utf-8') as f:
                    json.dump(notebook_content, f, indent=2, ensure_ascii=False)
                
                self.logger.success(LogCategory.CLOUD, f"Colab notebook created: {notebook_path}")
                self.logger.info(LogCategory.CLOUD, "Next steps:")
                self.logger.info(LogCategory.CLOUD, "1. Open Google Colab")
                self.logger.info(LogCategory.CLOUD, "2. Upload colab_automation.ipynb")
                self.logger.info(LogCategory.CLOUD, "3. Click 'Run All'")
                self.logger.info(LogCategory.CLOUD, "4. Wait for training to complete")
                self.logger.info(LogCategory.CLOUD, "5. Results will be automatically downloaded")
                
            except Exception as e:
                self.logger.error(LogCategory.CLOUD, f"Failed to create Colab notebook: {e}")
    
    def run_colab_automation(self):
        """Run full Colab automation workflow."""
        with OperationContext("Colab automation workflow", LogCategory.CLOUD):
            self.logger.print_header("GOOGLE COLAB AUTOMATION WORKFLOW")
            
            # Create notebook
            self.create_colab_notebook()
            
            # Create additional automation files
            self._create_automation_scripts()
            
            self.logger.success(LogCategory.CLOUD, "Colab automation workflow completed!")
            self.logger.print_summary("Automation Files Created", {
                "Notebook": "colab_automation.ipynb",
                "Upload Script": "upload_to_colab.sh",
                "Download Script": "download_from_colab.sh",
                "Guide": "COLAB_AUTOMATION_GUIDE.md"
            })
    
    def _create_automation_scripts(self):
        """Create additional automation scripts."""
        try:
            # Create upload script
            upload_script = """#!/bin/bash
# 自动上传到 Google Colab 的脚本

echo "🚀 开始上传到 Google Colab..."

# 1. 创建项目压缩包
echo "📦 创建项目压缩包..."
zip -r transformer_project.zip . -x "*.git*" "checkpoints/*" "config_history/*" "experiments/results/*" "*.log" "*.tmp" "__pycache__/*" "*.pyc"

# 2. 上传到 Google Drive (需要手动操作)
echo "📤 请手动上传 transformer_project.zip 到 Google Drive"
echo "   然后运行以下命令解压："
echo "   !unzip transformer_project.zip -d /content/transformer"

# 3. 运行训练
echo "🏃 在 Colab 中运行训练："
echo "   !cd /content/transformer && python run.py --force-cloud --use-default"

echo "✅ 上传脚本完成！"
"""
            
            with open("upload_to_colab.sh", 'w') as f:
                f.write(upload_script)
            os.chmod("upload_to_colab.sh", 0o755)
            
            # Create download script
            download_script = """#!/bin/bash
# 自动下载 Colab 结果的脚本

echo "📥 开始下载 Colab 结果..."

# 1. 检查本地结果目录
if [ ! -d "colab_results" ]; then
    mkdir colab_results
fi

# 2. 下载结果 (需要手动操作)
echo "📤 请手动下载 Colab 中的结果文件到 colab_results/ 目录"
echo "   或者使用以下 Python 代码："
echo ""
echo "   from google.colab import files"
echo "   files.download('/content/transformer/training_results.zip')"

# 3. 解压结果
if [ -f "colab_results/training_results.zip" ]; then
    echo "📦 解压结果..."
    cd colab_results
    unzip training_results.zip
    cd ..
    echo "✅ 结果已解压到 colab_results/ 目录"
else
    echo "⚠️ 未找到结果文件，请手动下载"
fi

echo "✅ 下载脚本完成！"
"""
            
            with open("download_from_colab.sh", 'w') as f:
                f.write(download_script)
            os.chmod("download_from_colab.sh", 0o755)
            
            # Create automation guide
            guide_content = """# 🤖 Google Colab 自动化指南

## 📋 **快速开始**

### **方法1：使用 Colab Notebook（推荐）**

1. **打开 Colab**：访问 [Google Colab](https://colab.research.google.com/)
2. **上传 Notebook**：上传 `colab_automation.ipynb`
3. **一键运行**：点击 "运行全部" 按钮
4. **自动下载**：训练完成后自动下载结果

### **方法2：手动步骤**

1. **克隆代码**：
   ```python
   !git clone https://github.com/Shanlans/transformer.git
   ```

2. **安装依赖**：
   ```python
   !cd transformer && pip install -r requirements.txt
   !cd transformer && pip install colabcode
   ```

3. **执行训练**：
   ```python
   !cd transformer && python run.py --force-cloud --use-default
   ```

4. **下载结果**：
   ```python
   from google.colab import files
   !cd transformer && zip -r results.zip checkpoints/
   files.download('/content/transformer/results.zip')
   ```

## 🔧 **自动化脚本**

- `colab_automation.ipynb`：完整的 Colab Notebook
- `upload_to_colab.sh`：上传脚本
- `download_from_colab.sh`：下载脚本

## 📊 **预期结果**

训练完成后，你将获得：
- ✅ 训练好的模型检查点
- ✅ 训练历史记录
- ✅ 可视化图表
- ✅ 评估结果
- ✅ 配置文件

## ⚠️ **注意事项**

1. **会话时间**：Colab 会话最长 12 小时
2. **GPU 限制**：免费用户有 GPU 使用限制
3. **存储限制**：Colab 有存储空间限制
4. **网络限制**：下载大文件可能较慢

## 🆘 **故障排除**

### **常见问题**：
- **导入错误**：检查依赖是否正确安装
- **内存不足**：减少批次大小或模型大小
- **训练中断**：检查网络连接和会话状态

### **获取帮助**：
- 查看训练日志
- 检查 Colab 控制台输出
- 使用 `!nvidia-smi` 检查 GPU 状态
"""
            
            with open("COLAB_AUTOMATION_GUIDE.md", 'w', encoding='utf-8') as f:
                f.write(guide_content)
            
            self.logger.info(LogCategory.CLOUD, "Additional automation files created")
            
        except Exception as e:
            self.logger.error(LogCategory.CLOUD, f"Failed to create automation scripts: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Unified Training Entry Point")
    
    # Action arguments
    parser.add_argument('--list-experiments', action='store_true', help='List all experiments')
    parser.add_argument('--show-experiment', help='Show experiment details')
    parser.add_argument('--create-experiment', action='store_true', help='Create new experiment')
    parser.add_argument('--create-derived-experiment', action='store_true', help='Create derived experiment from existing experiment')
    parser.add_argument('--train', action='store_true', help='Start training')
    parser.add_argument('--list-checkpoints', action='store_true', help='List checkpoints')
    parser.add_argument('--create-resume', action='store_true', help='Create resume configuration')
    parser.add_argument('--validate-changes', action='store_true', help='Validate hyperparameter changes')
    parser.add_argument('--list-config-versions', action='store_true', help='List configuration versions')
    parser.add_argument('--cleanup-checkpoints', action='store_true', help='Clean up checkpoints only')
    parser.add_argument('--cleanup-all', action='store_true', help='Clean up all training artifacts')
    parser.add_argument('--cloud-status', action='store_true', help='Check cloud training status')
    parser.add_argument('--cloud-logs', action='store_true', help='Show cloud training logs')
    parser.add_argument('--cloud-stop', action='store_true', help='Stop cloud training')
    
    # Colab automation arguments
    parser.add_argument('--create-colab-notebook', action='store_true', help='Create Colab automation notebook')
    parser.add_argument('--colab-automation', action='store_true', help='Run full Colab automation workflow')
    
    # Training options
    parser.add_argument('--experiment', help='Experiment name for training')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--use-default', action='store_true', help='Use default configuration')
    parser.add_argument('--force-local', action='store_true', help='Force local training')
    parser.add_argument('--force-cloud', action='store_true', help='Force cloud training')
    
    # Experiment creation options
    parser.add_argument('--name', help='Experiment name')
    parser.add_argument('--description', help='Experiment description')
    parser.add_argument('--source-experiment', help='Source experiment name for derived experiment')
    parser.add_argument('--new-environment', help='New training environment (local_gpu, local_cpu, cloud)')
    
    # Resume options
    parser.add_argument('--checkpoint', help='Checkpoint path for resume')
    parser.add_argument('--overrides', help='Hyperparameter overrides (JSON string)')
    parser.add_argument('--epochs', type=int, help='New number of epochs')
    
    # Cleanup options
    parser.add_argument('--keep-runs', type=int, default=3, help='Keep latest N runs (for --cleanup-checkpoints)')
    parser.add_argument('--keep-checkpoints', type=int, default=5, help='Keep latest N checkpoints per run (for --cleanup-checkpoints)')
    
    # Verbosity
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    parser.add_argument('--quiet', action='store_true', help='Quiet output')
    
    args = parser.parse_args()
    
    # Set verbosity
    verbose = args.verbose and not args.quiet
    
    # Initialize unified trainer
    trainer = UnifiedTrainer(verbose=verbose, quiet=args.quiet)
    
    success = False
    
    try:
        if args.list_experiments:
            trainer.list_experiments()
            success = True
        
        elif args.show_experiment:
            if not args.show_experiment:
                parser.error("--show-experiment requires experiment name")
            trainer.show_experiment(args.show_experiment)
            success = True
        
        elif args.create_experiment:
            if not args.name:
                parser.error("--create-experiment requires --name")
            trainer.create_experiment(
                name=args.name,
                description=args.description or ""
            )
            success = True
        
        elif args.create_derived_experiment:
            if not args.name or not args.description or not args.source_experiment:
                parser.error("--create-derived-experiment requires --name, --description, and --source-experiment")
            
            overrides = {}
            if args.overrides:
                overrides = json.loads(args.overrides)
            
            trainer.create_derived_experiment(
                source_experiment=args.source_experiment,
                name=args.name,
                description=args.description,
                new_environment=args.new_environment or "local_gpu",
                **overrides
            )
            success = True
        
        elif args.train:
            # Determine config path
            if args.use_default:
                config_path = "training_config.json"
                temp_experiment_name = f"default_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                trainer.logger.info(LogCategory.TRAINING, "Using default configuration")
                trainer.logger.info(LogCategory.TRAINING, f"Creating temporary experiment: {temp_experiment_name}")
                
                # Determine training environment first
                resource_name, resource_info = trainer.select_best_resource(args.force_local, args.force_cloud)
                if resource_info['type'] == 'cloud_gpu':
                    training_environment = "cloud"
                elif resource_info['type'] == 'local_gpu':
                    training_environment = "local_gpu"
                elif resource_info['type'] == 'local_cpu':
                    training_environment = "local_cpu"
                else:
                    training_environment = "unknown"
                
                # Create temporary experiment for checkpoint management
                trainer.create_experiment(
                    name=temp_experiment_name,
                    description=f"Temporary experiment for default config training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    training_environment=training_environment
                )
                
            elif args.config:
                config_path = args.config
            elif args.experiment:
                # Load experiment configuration
                experiment_path = f"experiments/configs/{args.experiment}.json"
                if not os.path.exists(experiment_path):
                    # Try with timestamp
                    import glob
                    pattern = f"experiments/configs/{args.experiment}_*.json"
                    matches = glob.glob(pattern)
                    if matches:
                        experiment_path = matches[0]
                    else:
                        trainer.logger.error(LogCategory.TRAINING, f"Experiment not found: {args.experiment}")
                        return
                config_path = experiment_path
            else:
                parser.error("Training requires --use-default, --config, or --experiment")
                return
            
            experiment_name = None
            training_environment = "unknown"
            
            if args.use_default:
                experiment_name = temp_experiment_name
                # Determine training environment based on resource selection
                resource_name, resource_info = trainer.select_best_resource(args.force_local, args.force_cloud)
                if resource_info['type'] == 'cloud_gpu':
                    training_environment = "cloud"
                elif resource_info['type'] == 'local_gpu':
                    training_environment = "local_gpu"
                elif resource_info['type'] == 'local_cpu':
                    training_environment = "local_cpu"
            elif args.experiment:
                experiment_name = args.experiment
            
            success = trainer.run_training(
                config_path=config_path,
                force_local=args.force_local,
                force_cloud=args.force_cloud,
                experiment_name=experiment_name,
                training_environment=training_environment
            )
        
        elif args.list_checkpoints:
            trainer.list_checkpoints(args.experiment)
            success = True
        
        elif args.create_resume:
            if not args.experiment or not args.checkpoint:
                parser.error("--create-resume requires --experiment and --checkpoint")
            
            overrides = None
            if args.overrides:
                overrides = json.loads(args.overrides)
            
            trainer.create_resume_config(
                experiment_name=args.experiment,
                checkpoint_path=args.checkpoint,
                overrides=overrides
            )
            success = True
        
        elif args.validate_changes:
            if not args.experiment or not args.overrides:
                parser.error("--validate-changes requires --experiment and --overrides")
            
            overrides = json.loads(args.overrides)
            trainer.validate_changes(args.experiment, overrides)
            success = True
        
        elif args.list_config_versions:
            trainer.list_config_versions()
            success = True
        
        elif args.cleanup_checkpoints:
            trainer.cleanup_checkpoints(args.keep_runs, args.keep_checkpoints)
            success = True
        
        elif args.cleanup_all:
            trainer.cleanup_all()
            success = True
        
        elif args.cloud_status:
            trainer.check_cloud_status()
            success = True
        
        elif args.cloud_logs:
            trainer.get_cloud_logs()
            success = True
        
        elif args.cloud_stop:
            trainer.stop_cloud_training()
            success = True
        
        elif args.create_colab_notebook:
            trainer.create_colab_notebook()
            success = True
        
        elif args.colab_automation:
            trainer.run_colab_automation()
            success = True
        
        else:
            parser.print_help()
            success = True
    
    except KeyboardInterrupt:
        trainer.logger.warning(LogCategory.SYSTEM, "Operation cancelled by user")
    except Exception as e:
        trainer.logger.error(LogCategory.SYSTEM, f"Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
