#!/usr/bin/env python3
"""
Unified Training Entry Point
Provides a single entry point for all training operations with intelligent resource selection
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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Please install PyTorch.")

try:
    from utils.colab_manager import ColabManager
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False
    print("Info: ColabCode not available. Will use local resources only.")

from utils.config_manager import ConfigManager
from utils.experiment_manager import ExperimentManager
from utils.resume_manager import ResumeManager
from utils.checkpoint_manager import CheckpointManager
from utils.config_version_manager import ConfigVersionManager


class UnifiedTrainer:
    """
    Unified Training System
    
    Features:
    - Single entry point for all training operations
    - Intelligent resource selection (cloud GPU -> local GPU -> local CPU)
    - Experiment management integration
    - Resume training support
    - Configuration management
    - Checkpoint management
    """
    
    def __init__(self):
        """Initialize Unified Trainer."""
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
        
        print("🚀 Unified Trainer initialized!")
        print("📊 Available resources:")
        for resource, info in self.resource_info.items():
            status_icon = "✅" if info['status'] == 'available' else "❌"
            print(f"   {status_icon} {resource.upper()}: {info['description']}")
    
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
                print("❌ Cloud training requested but ColabCode not available!")
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
        print(f"\n{'='*80}")
        print(f"🚀 TRAINING RESOURCE STATUS")
        print(f"{'='*80}")
        
        if resource_info['type'] == 'cloud_gpu':
            print(f"☁️  ENVIRONMENT: CLOUD GPU (Google Colab)")
            print(f"🖥️  DEVICE: CUDA")
            print(f"📊 RESOURCE: ColabCode SSH Connection")
            print(f"⚡ GPU: Free Google Colab GPU")
            print(f"🌐 LOCATION: Remote Cloud Server")
        elif resource_info['type'] == 'local_gpu':
            print(f"💻 ENVIRONMENT: LOCAL")
            print(f"🖥️  DEVICE: CUDA")
            print(f"📊 RESOURCE: Local GPU")
            print(f"⚡ GPU: {resource_info.get('device_name', 'Unknown')}")
            print(f"💾 MEMORY: {resource_info.get('memory_gb', 0):.1f}GB")
            print(f"🔢 COUNT: {resource_info.get('device_count', 1)} device(s)")
        else:  # local_cpu
            print(f"💻 ENVIRONMENT: LOCAL")
            print(f"🖥️  DEVICE: CPU")
            print(f"📊 RESOURCE: Local CPU")
            print(f"⚡ CORES: {resource_info.get('core_count', 1)} cores")
            print(f"💾 MEMORY: System RAM")
        
        print(f"{'='*80}")
        print(f"✅ Training will run on: {resource_name.upper()}")
        print(f"{'='*80}\n")
    
    def list_experiments(self):
        """List all available experiments."""
        print("\n" + "="*80)
        print("AVAILABLE EXPERIMENTS")
        print("="*80)
        
        experiments = self.experiment_manager.list_experiments()
        if not experiments:
            print("No experiments found.")
            print("\n💡 Tip: Create your first experiment with:")
            print("   python run.py --create-experiment --name my_experiment --description \"My first experiment\"")
            print("\n📋 Or use the default configuration:")
            print("   python run.py --train --use-default")
            return
        
        for exp in experiments:
            print(f"📁 {exp['name']}")
            print(f"   Description: {exp['description']}")
            print(f"   Created: {exp['created_at']}")
            print(f"   Version: {exp['version']}")
            if exp.get('timestamp_id'):
                print(f"   Timestamp ID: {exp['timestamp_id']}")
            print()
    
    def show_experiment(self, experiment_name: str):
        """Show experiment details."""
        try:
            config_manager = self.experiment_manager.load_experiment(experiment_name)
            config = config_manager.get_config()
            
            print(f"\n📋 Experiment Details: {experiment_name}")
            print("="*60)
            
            # Print experiment info
            print(f"Name: {config.experiment.name}")
            print(f"Description: {config.experiment.description}")
            print(f"Version: {config.experiment.version}")
            print(f"Created: {config.experiment.created_at}")
            if config.experiment.timestamp_id:
                print(f"Timestamp ID: {config.experiment.timestamp_id}")
            
            # Print key configuration
            print(f"\nModel Configuration:")
            print(f"  Architecture: Transformer")
            print(f"  Model dimension: {config.model.d_model}")
            print(f"  Attention heads: {config.model.n_heads}")
            print(f"  Encoder layers: {config.model.n_encoder_layers}")
            print(f"  Decoder layers: {config.model.n_decoder_layers}")
            
            print(f"\nTraining Configuration:")
            print(f"  Epochs: {config.training.epochs}")
            print(f"  Batch size: {config.data.batch_size}")
            print(f"  Learning rate: {config.training.learning_rate}")
            print(f"  Early stopping patience: {config.training.early_stopping_patience}")
            
        except Exception as e:
            print(f"❌ Error loading experiment: {e}")
    
    def create_experiment(self, name: str, description: str, **kwargs):
        """Create a new experiment."""
        try:
            config_path = self.experiment_manager.create_experiment(
                name=name,
                description=description,
                **kwargs
            )
            print(f"✅ Experiment created successfully!")
            print(f"   Name: {name}")
            print(f"   Description: {description}")
            print(f"   Config file: {config_path}")
        except Exception as e:
            print(f"❌ Error creating experiment: {e}")
    
    def list_checkpoints(self, experiment_name: Optional[str] = None):
        """List checkpoints."""
        print("\n" + "="*80)
        print("AVAILABLE CHECKPOINTS")
        print("="*80)
        
        if experiment_name:
            checkpoint_infos = self.resume_manager.find_experiment_checkpoints(experiment_name)
            print(f"Checkpoints for experiment: {experiment_name}")
            
            if not checkpoint_infos:
                print("No checkpoints found.")
                return
            
            for checkpoint_info in checkpoint_infos:
                print(f"\n📁 {checkpoint_info['run_name']}")
                print(f"   Path: {checkpoint_info['run_dir']}")
                print(f"   Timestamp: {checkpoint_info['timestamp_id']}")
                print(f"   Checkpoint Files:")
                
                for checkpoint_file in checkpoint_info['checkpoint_files']:
                    checkpoint_path = os.path.join(checkpoint_info['run_dir'], checkpoint_file)
                    try:
                        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
                        print(f"     - {checkpoint_file} ({file_size:.2f} MB)")
                    except:
                        print(f"     - {checkpoint_file}")
        else:
            runs = self.checkpoint_manager.list_runs()
            if not runs:
                print("No checkpoints found.")
                return
            
            for run in runs:
                print(f"\n📁 {run['name']}")
                run_checkpoints = self.checkpoint_manager.list_checkpoints(run['path'])
                
                for checkpoint in run_checkpoints:
                    print(f"   📄 {os.path.basename(checkpoint['path'])}")
                    print(f"      Epoch: {checkpoint.get('epoch', 'unknown')}")
                    print(f"      Loss: {checkpoint.get('loss', 'unknown')}")
                    print(f"      Timestamp: {checkpoint.get('timestamp', 'unknown')}")
                    print(f"      Is Best: {checkpoint.get('is_best', False)}")
    
    def create_resume_config(self, experiment_name: str, checkpoint_path: str, 
                           overrides: Optional[Dict[str, Any]] = None, epochs: Optional[int] = None):
        """Create a resume configuration."""
        try:
            resume_config_path = self.resume_manager.resume_training(
                experiment_name=experiment_name,
                checkpoint_path=checkpoint_path,
                hyperparameter_overrides=overrides,
                new_epochs=epochs
            )
            
            print(f"\n✅ Resume configuration created successfully!")
            print(f"   Config file: {resume_config_path}")
            print(f"   Location: experiments/configs/")
            print(f"   You can now use this config to resume training:")
            print(f"   python run.py --train --config {resume_config_path}")
            
        except Exception as e:
            print(f"❌ Error creating resume configuration: {e}")
    
    def run_training(self, config_path: str, force_local: bool = False, force_cloud: bool = False):
        """Run training with the specified configuration."""
        print(f"🚀 Starting training with config: {config_path}")
        
        # Select resource
        resource_name, resource_info = self.select_best_resource(force_local, force_cloud)
        if resource_name is None:
            print("❌ No suitable resource available!")
            return False
        
        self.print_resource_status(resource_name, resource_info)
        
        # Determine training method
        if resource_info['type'] == 'cloud_gpu':
            return self.run_cloud_training(config_path)
        else:
            return self.run_local_training(config_path, resource_info)
    
    def run_cloud_training(self, config_path: str) -> bool:
        """Run training on cloud GPU."""
        if not self.colab_manager:
            print("❌ ColabCode not available for cloud training.")
            return False
        
        print("☁️ Setting up cloud training...")
        
        # Try to start ColabCode session
        success = self.colab_manager.start_colab_session(gpu=True)
        if not success:
            print("❌ Failed to start cloud training session.")
            return False
        
        print("✅ Cloud training environment ready!")
        
        # Sync project to cloud
        print("📤 Syncing project to cloud...")
        sync_success = self.colab_manager.sync_project_to_colab(
            local_project_dir=".",
            remote_project_dir="/content/transformer"
        )
        
        if not sync_success:
            print("❌ Project sync failed.")
            return False
        
        print("✅ Project sync completed!")
        
        # Start remote training
        print("🏃 Starting remote training...")
        training_success = self.colab_manager.start_remote_training(
            experiment_name="cloud_training",
            config_path=config_path
        )
        
        if training_success:
            print("✅ Cloud training started successfully!")
            
            # Download results
            print("📥 Downloading results...")
            self.colab_manager.download_results(
                experiment_name="cloud_training",
                local_results_dir="./results"
            )
            print("✅ Results downloaded!")
        else:
            print("❌ Cloud training failed.")
        
        return training_success
    
    def run_local_training(self, config_path: str, resource_info: Dict[str, Any]) -> bool:
        """Run training on local device."""
        # Determine device
        if resource_info['type'] == 'local_gpu':
            device = 'cuda'
            print(f"🖥️ Using local GPU: {resource_info.get('device_name', 'Unknown')}")
        else:
            device = 'cpu'
            print(f"🖥️ Using local CPU: {resource_info.get('core_count', 1)} cores")
        
        print(f"🏃 Starting local training on {device}...")
        
        # Run training script
        try:
            result = subprocess.run([
                'python', 'train.py', '--config', config_path
            ], check=True, capture_output=True, text=True)
            
            print("✅ Local training completed successfully!")
            print(result.stdout)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Local training failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def cleanup_checkpoints(self, keep_latest_runs: int = 3, keep_latest_checkpoints: int = 5):
        """Cleanup old checkpoints."""
        try:
            result = self.checkpoint_manager.cleanup_all_runs(
                keep_latest_runs=keep_latest_runs,
                keep_latest_checkpoints=keep_latest_checkpoints
            )
            
            print(f"✅ Checkpoint cleanup completed!")
            print(f"   Runs cleaned: {result['runs_cleaned']}")
            print(f"   Checkpoints cleaned: {result['checkpoints_cleaned']}")
            print(f"   Space freed: {result['space_freed']:.2f} MB")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
    
    def list_config_versions(self):
        """List all configuration versions."""
        print("\n" + "="*80)
        print("CONFIGURATION VERSIONS")
        print("="*80)
        
        versions = self.config_version_manager.list_config_versions()
        if not versions:
            print("No configuration versions found.")
            return
        
        for version in versions:
            print(f"📋 {version['version_id']}")
            print(f"   Description: {version['description']}")
            print(f"   Created: {version['created_at']}")
            print(f"   Tags: {', '.join(version['tags'])}")
            if 'config_file' in version:
                print(f"   Config file: {version['config_file']}")
            if version.get('linked_checkpoint'):
                print(f"   Linked checkpoint: {version['linked_checkpoint']}")
            print()
    
    def validate_changes(self, experiment_name: str, overrides: Dict[str, Any]):
        """Validate hyperparameter changes for an experiment."""
        try:
            config_manager, original_config = self.resume_manager.load_experiment_config(experiment_name)
            
            # Create new config by updating the original config with overrides
            config_manager.update_config(overrides)
            new_config = config_manager.get_config()
            
            validation_result = self.resume_manager.validate_hyperparameter_changes(
                original_config=original_config,
                new_config=new_config
            )
            
            self.resume_manager.print_validation_results(validation_result)
            
        except Exception as e:
            print(f"❌ Error validating changes: {e}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Unified Training Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python run.py --list-experiments
  
  # Show experiment details
  python run.py --show-experiment small_test
  
  # Create new experiment
  python run.py --create-experiment --name my_exp --description "My experiment"
  
  # Train with default configuration
  python run.py --train --use-default
  
  # Train with specific experiment
  python run.py --train --experiment small_test
  
  # Train with specific config file
  python run.py --train --config experiments/configs/my_experiment.json
  
  # Train with resume configuration
  python run.py --train --config experiments/configs/resume_small_test_20250925_123456.json
  
  # Force local training
  python run.py --train --experiment small_test --force-local
  
  # Force cloud training
  python run.py --train --experiment small_test --force-cloud
  
  # List checkpoints
  python run.py --list-checkpoints
  
  # List checkpoints for specific experiment
  python run.py --list-checkpoints --experiment small_test
  
  # Create resume configuration
  python run.py --create-resume --experiment small_test --checkpoint checkpoints/run_xxx/best_model.pt --overrides '{"training": {"learning_rate": 0.0002}}'
  
  # Validate hyperparameter changes
  python run.py --validate-changes --experiment small_test --overrides '{"training": {"learning_rate": 0.0002}}'
  
  # List configuration versions
  python run.py --list-config-versions
  
  # Cleanup old checkpoints
  python run.py --cleanup --keep-runs 3 --keep-checkpoints 5
        """
    )
    
    # Main actions
    parser.add_argument('--list-experiments', action='store_true', help='List all experiments')
    parser.add_argument('--show-experiment', help='Show experiment details')
    parser.add_argument('--create-experiment', action='store_true', help='Create new experiment')
    parser.add_argument('--train', action='store_true', help='Start training')
    parser.add_argument('--list-checkpoints', action='store_true', help='List checkpoints')
    parser.add_argument('--create-resume', action='store_true', help='Create resume configuration')
    parser.add_argument('--validate-changes', action='store_true', help='Validate hyperparameter changes')
    parser.add_argument('--list-config-versions', action='store_true', help='List configuration versions')
    parser.add_argument('--cleanup', action='store_true', help='Cleanup old checkpoints')
    
    # Training options
    parser.add_argument('--experiment', help='Experiment name for training')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--use-default', action='store_true', help='Use default configuration')
    parser.add_argument('--force-local', action='store_true', help='Force local training')
    parser.add_argument('--force-cloud', action='store_true', help='Force cloud training')
    
    # Experiment creation options
    parser.add_argument('--name', help='Experiment name')
    parser.add_argument('--description', help='Experiment description')
    
    # Resume options
    parser.add_argument('--checkpoint', help='Checkpoint path for resume')
    parser.add_argument('--overrides', help='Hyperparameter overrides (JSON string)')
    parser.add_argument('--epochs', type=int, help='New number of epochs')
    
    # Cleanup options
    parser.add_argument('--keep-runs', type=int, default=3, help='Keep latest N runs')
    parser.add_argument('--keep-checkpoints', type=int, default=5, help='Keep latest N checkpoints per run')
    
    args = parser.parse_args()
    
    # Initialize unified trainer
    trainer = UnifiedTrainer()
    
    # Execute actions
    success = False
    
    if args.list_experiments:
        trainer.list_experiments()
        success = True
    
    elif args.show_experiment:
        trainer.show_experiment(args.show_experiment)
        success = True
    
    elif args.create_experiment:
        if not args.name or not args.description:
            parser.error("--create-experiment requires --name and --description")
        trainer.create_experiment(args.name, args.description)
        success = True
    
    elif args.train:
        if args.use_default:
            config_path = "training_config.json"
        elif args.config:
            config_path = args.config
        elif args.experiment:
            # Load experiment configuration
            try:
                config_manager = trainer.experiment_manager.load_experiment(args.experiment)
                config_path = config_manager.config_path
            except Exception as e:
                print(f"❌ Error loading experiment '{args.experiment}': {e}")
                return
        else:
            parser.error("--train requires --use-default, --config, or --experiment")
        
        success = trainer.run_training(
            config_path=config_path,
            force_local=args.force_local,
            force_cloud=args.force_cloud
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
            overrides=overrides,
            epochs=args.epochs
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
    
    elif args.cleanup:
        trainer.cleanup_checkpoints(args.keep_runs, args.keep_checkpoints)
        success = True
    
    else:
        parser.print_help()
        success = True
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
