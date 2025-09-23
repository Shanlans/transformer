"""
Checkpoint Management Script
Provides command-line interface for managing model checkpoints
"""

import argparse
import os
import sys
from typing import List, Dict

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.checkpoint_manager import CheckpointManager


def list_runs(manager: CheckpointManager):
    """List all runs."""
    runs = manager.list_runs()
    
    if not runs:
        print("No runs found.")
        return
    
    print(f"Found {len(runs)} runs:")
    print("-" * 80)
    print(f"{'Run Name':<30} {'Checkpoints':<12} {'Size (MB)':<12} {'Created':<20}")
    print("-" * 80)
    
    for run in runs:
        print(f"{run['name']:<30} {run['checkpoint_count']:<12} {run['size_mb']:<12.2f} {run['created']:<20}")


def list_checkpoints(manager: CheckpointManager, run_name: str = None):
    """List checkpoints in a specific run."""
    if run_name:
        # Find run directory
        runs = manager.list_runs()
        run_dir = None
        for run in runs:
            if run['name'] == run_name:
                run_dir = run['path']
                break
        
        if run_dir is None:
            print(f"Run '{run_name}' not found.")
            return
    else:
        # Use current run directory
        run_dir = manager.current_run_dir
        if run_dir is None:
            print("No current run directory. Please specify a run name.")
            return
    
    checkpoints = manager.list_checkpoints(run_dir)
    
    if not checkpoints:
        print("No checkpoints found.")
        return
    
    print(f"Checkpoints in {os.path.basename(run_dir)}:")
    print("-" * 100)
    print(f"{'Filename':<40} {'Epoch':<8} {'Loss':<12} {'Best':<6} {'Size (MB)':<12} {'Timestamp':<20}")
    print("-" * 100)
    
    for checkpoint in checkpoints:
        print(f"{checkpoint['filename']:<40} {checkpoint['epoch']:<8} {checkpoint['loss']:<12.4f} {checkpoint['is_best']:<6} {checkpoint['size_mb']:<12.2f} {checkpoint['timestamp']:<20}")


def cleanup_run(manager: CheckpointManager, run_name: str, keep_latest: int = 0):
    """Cleanup checkpoints in a specific run."""
    runs = manager.list_runs()
    run_dir = None
    
    for run in runs:
        if run['name'] == run_name:
            run_dir = run['path']
            break
    
    if run_dir is None:
        print(f"Run '{run_name}' not found.")
        return
    
    print(f"Cleaning up run '{run_name}'...")
    removed_count = manager.cleanup_run(run_dir, keep_latest)
    print(f"Removed {removed_count} checkpoints.")


def cleanup_all(manager: CheckpointManager, keep_latest_runs: int = 0, keep_latest_checkpoints: int = 0):
    """Cleanup all runs and checkpoints."""
    print("Cleaning up all runs and checkpoints...")
    stats = manager.cleanup_all_runs(keep_latest_runs, keep_latest_checkpoints)
    print(f"Cleanup completed:")
    print(f"  Runs removed: {stats['runs_removed']}")
    print(f"  Checkpoints removed: {stats['checkpoints_removed']}")
    print(f"  Total size freed: {stats['total_size_freed_mb']:.2f} MB")


def get_best_checkpoint(manager: CheckpointManager, run_name: str = None):
    """Get the best checkpoint in a specific run."""
    if run_name:
        # Find run directory
        runs = manager.list_runs()
        run_dir = None
        for run in runs:
            if run['name'] == run_name:
                run_dir = run['path']
                break
        
        if run_dir is None:
            print(f"Run '{run_name}' not found.")
            return
    else:
        # Use current run directory
        run_dir = manager.current_run_dir
        if run_dir is None:
            print("No current run directory. Please specify a run name.")
            return
    
    best_checkpoint = manager.get_best_checkpoint(run_dir)
    
    if best_checkpoint:
        print(f"Best checkpoint: {best_checkpoint}")
    else:
        print("No best checkpoint found.")


def show_stats(manager: CheckpointManager):
    """Show checkpoint statistics."""
    runs = manager.list_runs()
    
    if not runs:
        print("No runs found.")
        return
    
    total_runs = len(runs)
    total_checkpoints = sum(run['checkpoint_count'] for run in runs)
    total_size = sum(run['size_mb'] for run in runs)
    
    print("Checkpoint Statistics:")
    print("-" * 40)
    print(f"Total runs: {total_runs}")
    print(f"Total checkpoints: {total_checkpoints}")
    print(f"Total size: {total_size:.2f} MB")
    print(f"Average checkpoints per run: {total_checkpoints / total_runs:.1f}")
    print(f"Average size per run: {total_size / total_runs:.2f} MB")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Manage Transformer checkpoints')
    
    # Global arguments
    parser.add_argument('--base_dir', type=str, default='checkpoints', help='Base directory for checkpoints')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List runs command
    list_runs_parser = subparsers.add_parser('list-runs', help='List all runs')
    
    # List checkpoints command
    list_checkpoints_parser = subparsers.add_parser('list-checkpoints', help='List checkpoints in a run')
    list_checkpoints_parser.add_argument('--run', type=str, help='Run name (default: current run)')
    
    # Cleanup run command
    cleanup_run_parser = subparsers.add_parser('cleanup-run', help='Cleanup checkpoints in a specific run')
    cleanup_run_parser.add_argument('run_name', type=str, help='Run name to cleanup')
    cleanup_run_parser.add_argument('--keep-latest', type=int, default=0, help='Number of latest checkpoints to keep (0 = remove all)')
    
    # Cleanup all command
    cleanup_all_parser = subparsers.add_parser('cleanup-all', help='Cleanup all runs and checkpoints')
    cleanup_all_parser.add_argument('--keep-latest-runs', type=int, default=0, help='Number of latest runs to keep (0 = remove all)')
    cleanup_all_parser.add_argument('--keep-latest-checkpoints', type=int, default=0, help='Number of latest checkpoints to keep per run (0 = remove all)')
    
    # Get best checkpoint command
    get_best_parser = subparsers.add_parser('get-best', help='Get the best checkpoint in a run')
    get_best_parser.add_argument('--run', type=str, help='Run name (default: current run)')
    
    # Show stats command
    stats_parser = subparsers.add_parser('stats', help='Show checkpoint statistics')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Create checkpoint manager
    manager = CheckpointManager(args.base_dir)
    
    # Execute command
    if args.command == 'list-runs':
        list_runs(manager)
    elif args.command == 'list-checkpoints':
        list_checkpoints(manager, args.run)
    elif args.command == 'cleanup-run':
        cleanup_run(manager, args.run_name, args.keep_latest)
    elif args.command == 'cleanup-all':
        cleanup_all(manager, args.keep_latest_runs, args.keep_latest_checkpoints)
    elif args.command == 'get-best':
        get_best_checkpoint(manager, args.run)
    elif args.command == 'stats':
        show_stats(manager)


if __name__ == "__main__":
    main()
