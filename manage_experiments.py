"""
Experiment Management CLI
Command-line interface for managing training experiments
"""

import argparse
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.experiment_manager import ExperimentManager


def create_experiment(args):
    """Create a new experiment."""
    manager = ExperimentManager()
    
    # Prepare overrides
    overrides = {}
    
    # Model parameters
    if args.d_model:
        overrides['model.d_model'] = args.d_model
    if args.n_heads:
        overrides['model.n_heads'] = args.n_heads
    if args.n_encoder_layers:
        overrides['model.n_encoder_layers'] = args.n_encoder_layers
    if args.n_decoder_layers:
        overrides['model.n_decoder_layers'] = args.n_decoder_layers
    if args.d_ff:
        overrides['model.d_ff'] = args.d_ff
    if args.dropout:
        overrides['model.dropout'] = args.dropout
    
    # Training parameters
    if args.epochs:
        overrides['training.epochs'] = args.epochs
    if args.learning_rate:
        overrides['training.learning_rate'] = args.learning_rate
    if args.weight_decay:
        overrides['training.weight_decay'] = args.weight_decay
    if args.gradient_clip_norm:
        overrides['training.gradient_clip_norm'] = args.gradient_clip_norm
    
    # Data parameters
    if args.batch_size:
        overrides['data.batch_size'] = args.batch_size
    if args.max_length:
        overrides['data.max_length'] = args.max_length
    if args.train_split:
        overrides['data.train_split'] = args.train_split
    
    # Optimizer parameters
    if args.optimizer_type:
        overrides['optimizer.type'] = args.optimizer_type
    
    # Scheduler parameters
    if args.scheduler_type:
        overrides['scheduler.type'] = args.scheduler_type
    
    # Loss parameters
    if args.loss_type:
        overrides['loss.type'] = args.loss_type
    
    # Create experiment
    experiment_path, timestamp_id = manager.create_experiment(
        name=args.name,
        description=args.description,
        base_config_path=args.base_config,
        **overrides
    )
    
    print(f"Experiment created successfully: {experiment_path}")
    print(f"Timestamp ID: {timestamp_id}")


def list_experiments(args):
    """List all experiments."""
    manager = ExperimentManager()
    manager.print_experiments()


def load_experiment(args):
    """Load and display an experiment."""
    manager = ExperimentManager()
    
    try:
        config_manager = manager.load_experiment(args.name)
        config_manager.print_config()
    except ValueError as e:
        print(f"Error: {e}")


def run_experiment(args):
    """Run an experiment."""
    manager = ExperimentManager()
    
    try:
        # Get experiment timestamp
        timestamp_id = manager.get_experiment_timestamp(args.name)
        if timestamp_id:
            print(f"Experiment timestamp: {timestamp_id}")
        
        manager.run_experiment(args.name, args.train_script)
    except ValueError as e:
        print(f"Error: {e}")


def compare_experiments(args):
    """Compare experiments."""
    manager = ExperimentManager()
    manager.compare_experiments(args.names)


def create_template(args):
    """Create experiment template."""
    manager = ExperimentManager()
    manager.create_experiment_template(args.template)


def validate_experiment(args):
    """Validate experiment integrity."""
    manager = ExperimentManager()
    
    try:
        is_valid = manager.validate_experiment_integrity(args.name)
        if is_valid:
            print(f"Experiment '{args.name}' integrity check passed.")
        else:
            print(f"Experiment '{args.name}' integrity check failed.")
    except ValueError as e:
        print(f"Error: {e}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Manage training experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python manage_experiments.py list

  # Create a small experiment
  python manage_experiments.py create --name small_test --description "Quick test" \\
    --d_model 128 --n_heads 4 --epochs 5 --batch_size 16

  # Create a medium experiment
  python manage_experiments.py create --name medium_exp --description "Balanced model" \\
    --d_model 256 --n_heads 8 --epochs 20 --batch_size 32

  # Load an experiment
  python manage_experiments.py load --name small_test

  # Run an experiment
  python manage_experiments.py run --name small_test

  # Compare experiments
  python manage_experiments.py compare --names small_test medium_exp

  # Create template
  python manage_experiments.py template --template small
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate experiment command
    validate_parser = subparsers.add_parser('validate', help='Validate experiment integrity')
    validate_parser.add_argument('--name', required=True, help='Experiment name')
    validate_parser.set_defaults(func=validate_experiment)
    
    # Create experiment command
    create_parser = subparsers.add_parser('create', help='Create a new experiment')
    create_parser.add_argument('--name', required=True, help='Experiment name')
    create_parser.add_argument('--description', default='', help='Experiment description')
    create_parser.add_argument('--base_config', default='training_config.json', help='Base configuration file')
    
    # Model parameters
    create_parser.add_argument('--d_model', type=int, help='Model dimension')
    create_parser.add_argument('--n_heads', type=int, help='Number of attention heads')
    create_parser.add_argument('--n_encoder_layers', type=int, help='Number of encoder layers')
    create_parser.add_argument('--n_decoder_layers', type=int, help='Number of decoder layers')
    create_parser.add_argument('--d_ff', type=int, help='Feed-forward dimension')
    create_parser.add_argument('--dropout', type=float, help='Dropout rate')
    
    # Training parameters
    create_parser.add_argument('--epochs', type=int, help='Number of epochs')
    create_parser.add_argument('--learning_rate', type=float, help='Learning rate')
    create_parser.add_argument('--weight_decay', type=float, help='Weight decay')
    create_parser.add_argument('--gradient_clip_norm', type=float, help='Gradient clipping norm')
    
    # Data parameters
    create_parser.add_argument('--batch_size', type=int, help='Batch size')
    create_parser.add_argument('--max_length', type=int, help='Maximum sequence length')
    create_parser.add_argument('--train_split', type=float, help='Training split ratio')
    
    # Other parameters
    create_parser.add_argument('--optimizer_type', help='Optimizer type')
    create_parser.add_argument('--scheduler_type', help='Scheduler type')
    create_parser.add_argument('--loss_type', help='Loss function type')
    
    create_parser.set_defaults(func=create_experiment)
    
    # List experiments command
    list_parser = subparsers.add_parser('list', help='List all experiments')
    list_parser.set_defaults(func=list_experiments)
    
    # Load experiment command
    load_parser = subparsers.add_parser('load', help='Load and display an experiment')
    load_parser.add_argument('--name', required=True, help='Experiment name')
    load_parser.set_defaults(func=load_experiment)
    
    # Run experiment command
    run_parser = subparsers.add_parser('run', help='Run an experiment')
    run_parser.add_argument('--name', required=True, help='Experiment name')
    run_parser.add_argument('--train_script', default='train.py', help='Training script path')
    run_parser.set_defaults(func=run_experiment)
    
    # Compare experiments command
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('--names', nargs='+', required=True, help='Experiment names to compare')
    compare_parser.set_defaults(func=compare_experiments)
    
    # Create template command
    template_parser = subparsers.add_parser('template', help='Create experiment template')
    template_parser.add_argument('--template', choices=['small', 'medium', 'large'], 
                                default='small', help='Template type')
    template_parser.set_defaults(func=create_template)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
