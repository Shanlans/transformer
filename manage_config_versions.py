"""
Configuration Version Management CLI
Command-line interface for managing configuration versions and their links with checkpoints
"""

import argparse
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.config_version_manager import ConfigVersionManager


def save_config_version(args):
    """Save a configuration version."""
    manager = ConfigVersionManager()
    
    version_id = manager.save_config_version(
        config_path=args.config,
        version_name=args.name,
        description=args.description,
        tags=args.tags.split(',') if args.tags else None
    )
    
    print(f"Configuration version saved: {version_id}")


def list_config_versions(args):
    """List all configuration versions."""
    manager = ConfigVersionManager()
    manager.print_config_versions()


def compare_config_versions(args):
    """Compare two configuration versions."""
    manager = ConfigVersionManager()
    manager.print_config_comparison(args.version1, args.version2)


def link_with_checkpoint(args):
    """Link configuration version with checkpoint."""
    manager = ConfigVersionManager()
    
    manager.link_config_with_checkpoint(
        config_version_id=args.version_id,
        checkpoint_run_dir=args.checkpoint_dir,
        link_type=args.link_type
    )
    
    print(f"Configuration version {args.version_id} linked with checkpoint {args.checkpoint_dir}")


def get_checkpoint_config_history(args):
    """Get configuration history for a checkpoint run."""
    manager = ConfigVersionManager()
    
    linked_configs = manager.get_checkpoint_config_history(args.checkpoint)
    
    if not linked_configs:
        print(f"No configuration history found for checkpoint: {args.checkpoint}")
        return
    
    print(f"\nConfiguration history for checkpoint: {args.checkpoint}")
    print("=" * 80)
    
    for config in linked_configs:
        print(f"Config Version: {config['config_version_id']}")
        print(f"Link Type: {config['link_type']}")
        print(f"Linked At: {config['linked_at']}")
        print(f"Description: {config['config_metadata']['description']}")
        print(f"Tags: {', '.join(config['config_metadata']['tags'])}")
        print("-" * 80)


def auto_save_config(args):
    """Automatically save configuration version and optionally link with checkpoint."""
    manager = ConfigVersionManager()
    
    version_id = manager.auto_save_config_version(
        config_path=args.config,
        checkpoint_run_dir=args.checkpoint_dir,
        description=args.description,
        tags=args.tags.split(',') if args.tags else None
    )
    
    print(f"Configuration version auto-saved: {version_id}")
    if args.checkpoint_dir:
        print(f"Linked with checkpoint: {args.checkpoint_dir}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Manage configuration versions and their links with checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Save a configuration version
  python manage_config_versions.py save --config training_config.json --name baseline --description "Baseline config"

  # List all configuration versions
  python manage_config_versions.py list

  # Compare two configurations
  python manage_config_versions.py compare --version1 baseline --version2 modified

  # Link configuration with checkpoint
  python manage_config_versions.py link --version baseline --checkpoint checkpoints/run_20250101_120000

  # Get checkpoint configuration history
  python manage_config_versions.py history --checkpoint checkpoints/run_20250101_120000

  # Auto-save configuration with checkpoint link
  python manage_config_versions.py auto-save --config training_config.json --checkpoint checkpoints/run_20250101_120000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Save configuration version command
    save_parser = subparsers.add_parser('save', help='Save a configuration version')
    save_parser.add_argument('--config', required=True, help='Path to configuration file')
    save_parser.add_argument('--name', help='Version name (auto-generated if not provided)')
    save_parser.add_argument('--description', default='', help='Description of this configuration version')
    save_parser.add_argument('--tags', help='Comma-separated tags')
    save_parser.set_defaults(func=save_config_version)
    
    # List configuration versions command
    list_parser = subparsers.add_parser('list', help='List all configuration versions')
    list_parser.set_defaults(func=list_config_versions)
    
    # Compare configuration versions command
    compare_parser = subparsers.add_parser('compare', help='Compare two configuration versions')
    compare_parser.add_argument('--version1', required=True, help='First version ID')
    compare_parser.add_argument('--version2', required=True, help='Second version ID')
    compare_parser.set_defaults(func=compare_config_versions)
    
    # Link with checkpoint command
    link_parser = subparsers.add_parser('link', help='Link configuration version with checkpoint')
    link_parser.add_argument('--version', required=True, help='Configuration version ID')
    link_parser.add_argument('--checkpoint', required=True, help='Checkpoint run directory')
    link_parser.add_argument('--link-type', default='training', help='Type of link')
    link_parser.set_defaults(func=link_with_checkpoint)
    
    # Get checkpoint config history command
    history_parser = subparsers.add_parser('history', help='Get configuration history for checkpoint')
    history_parser.add_argument('--checkpoint', required=True, help='Checkpoint run directory')
    history_parser.set_defaults(func=get_checkpoint_config_history)
    
    # Auto-save configuration command
    auto_save_parser = subparsers.add_parser('auto-save', help='Auto-save configuration version')
    auto_save_parser.add_argument('--config', required=True, help='Path to configuration file')
    auto_save_parser.add_argument('--checkpoint', help='Checkpoint run directory (optional)')
    auto_save_parser.add_argument('--description', default='', help='Description of this configuration version')
    auto_save_parser.add_argument('--tags', help='Comma-separated tags')
    auto_save_parser.set_defaults(func=auto_save_config)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()