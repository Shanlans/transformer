"""
Checkpoint Manager
Manages model checkpoints with timestamp-based organization and cleanup options
"""

import os
import shutil
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import torch
import glob


class CheckpointManager:
    """
    Manages model checkpoints with timestamp-based organization.
    
    Features:
    - Timestamp-based folder organization
    - Automatic checkpoint naming
    - Cleanup options (keep latest N, remove all, etc.)
    - Metadata tracking
    - Easy checkpoint loading and saving
    """
    
    def __init__(self, base_dir: str = "checkpoints", max_checkpoints: int = 10):
        """
        Initialize checkpoint manager.
        
        Args:
            base_dir: Base directory for storing checkpoints
            max_checkpoints: Maximum number of checkpoints to keep (0 = unlimited)
        """
        self.base_dir = base_dir
        self.max_checkpoints = max_checkpoints
        self.current_run_dir = None
        
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        print(f"CheckpointManager initialized:")
        print(f"  Base directory: {base_dir}")
        print(f"  Max checkpoints: {max_checkpoints if max_checkpoints > 0 else 'unlimited'}")
    
    def create_run_directory(self, run_name: Optional[str] = None, custom_timestamp: Optional[str] = None) -> str:
        """
        Create a new run directory with timestamp.
        
        Args:
            run_name: Optional name for the run (default: auto-generated)
            custom_timestamp: Custom timestamp to use (format: YYYYMMDD_HHMMSS)
            
        Returns:
            Path to the created run directory
        """
        # Use unified timestamp manager if available
        timestamp = None
        if custom_timestamp:
            timestamp = custom_timestamp
        else:
            try:
                from .timestamp_manager import get_timestamp_manager
                timestamp_manager = get_timestamp_manager()
                current_timestamp = timestamp_manager.get_current_timestamp()
                if current_timestamp:
                    timestamp = current_timestamp
            except ImportError:
                pass
            
            # Fallback to generating new timestamp
            if not timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create run name
        if run_name is None:
            run_name = f"run_{timestamp}"
        else:
            run_name = f"{run_name}_{timestamp}"
        
        # Create run directory
        run_dir = os.path.join(self.base_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        self.current_run_dir = run_dir
        
        print(f"Created run directory: {run_dir}")
        return run_dir
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        metrics: Optional[Dict] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_name: Optional[str] = None,
        is_best: bool = False
    ) -> str:
        """
        Save model checkpoint with metadata.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            loss: Current loss value
            metrics: Additional metrics to save
            scheduler: Learning rate scheduler (optional)
            checkpoint_name: Custom checkpoint name (default: auto-generated)
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            Path to saved checkpoint
        """
        if self.current_run_dir is None:
            self.create_run_directory()
        
        # Generate checkpoint name
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{epoch:03d}"
            if is_best:
                checkpoint_name += "_best"
        
        checkpoint_path = os.path.join(self.current_run_dir, f"{checkpoint_name}.pt")
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
            'is_best': is_best,
            'run_dir': self.current_run_dir
        }
        
        # Add scheduler state if provided
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add metrics if provided
        if metrics is not None:
            checkpoint_data['metrics'] = metrics
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save metadata
        self._save_metadata(checkpoint_path, checkpoint_data)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints if needed
        if self.max_checkpoints > 0:
            self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cpu"
    ) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to load checkpoint on
            
        Returns:
            Checkpoint metadata
        """
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"  Epoch: {checkpoint_data.get('epoch', 'unknown')}")
        print(f"  Loss: {checkpoint_data.get('loss', 'unknown')}")
        print(f"  Timestamp: {checkpoint_data.get('timestamp', 'unknown')}")
        
        return checkpoint_data
    
    def list_checkpoints(self, run_dir: Optional[str] = None) -> List[Dict]:
        """
        List all checkpoints in a run directory.
        
        Args:
            run_dir: Run directory to list (default: current run)
            
        Returns:
            List of checkpoint information
        """
        if run_dir is None:
            run_dir = self.current_run_dir
        
        if run_dir is None or not os.path.exists(run_dir):
            return []
        
        checkpoints = []
        checkpoint_files = glob.glob(os.path.join(run_dir, "*.pt"))
        
        for checkpoint_file in checkpoint_files:
            try:
                # Load checkpoint metadata without loading the full model
                checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
                
                checkpoint_info = {
                    'path': checkpoint_file,
                    'filename': os.path.basename(checkpoint_file),
                    'epoch': checkpoint_data.get('epoch', 'unknown'),
                    'loss': checkpoint_data.get('loss', 'unknown'),
                    'timestamp': checkpoint_data.get('timestamp', 'unknown'),
                    'is_best': checkpoint_data.get('is_best', False),
                    'size_mb': os.path.getsize(checkpoint_file) / (1024 * 1024)
                }
                checkpoints.append(checkpoint_info)
            except Exception as e:
                print(f"Warning: Could not read checkpoint {checkpoint_file}: {e}")
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x['epoch'] if isinstance(x['epoch'], int) else 0)
        
        return checkpoints
    
    def list_runs(self) -> List[Dict]:
        """
        List all run directories.
        
        Returns:
            List of run information
        """
        if not os.path.exists(self.base_dir):
            return []
        
        runs = []
        for run_name in os.listdir(self.base_dir):
            run_path = os.path.join(self.base_dir, run_name)
            if os.path.isdir(run_path):
                # Get run metadata
                metadata_file = os.path.join(run_path, "run_metadata.json")
                metadata = {}
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        print(f"Warning: Could not read metadata for {run_name}: {e}")
                
                # Count checkpoints
                checkpoint_count = len(glob.glob(os.path.join(run_path, "*.pt")))
                
                run_info = {
                    'name': run_name,
                    'path': run_path,
                    'created': metadata.get('created', 'unknown'),
                    'checkpoint_count': checkpoint_count,
                    'size_mb': self._get_directory_size(run_path) / (1024 * 1024),
                    'metadata': metadata
                }
                runs.append(run_info)
        
        # Sort by creation time (newest first)
        runs.sort(key=lambda x: x['created'], reverse=True)
        
        return runs
    
    def cleanup_run(self, run_dir: str, keep_latest: int = 0) -> int:
        """
        Cleanup checkpoints in a specific run.
        
        Args:
            run_dir: Run directory to cleanup
            keep_latest: Number of latest checkpoints to keep (0 = remove all)
            
        Returns:
            Number of checkpoints removed
        """
        if not os.path.exists(run_dir):
            print(f"Run directory does not exist: {run_dir}")
            return 0
        
        checkpoints = self.list_checkpoints(run_dir)
        
        if keep_latest > 0:
            # Keep only the latest N checkpoints
            checkpoints_to_remove = checkpoints[:-keep_latest]
        else:
            # Remove all checkpoints
            checkpoints_to_remove = checkpoints
        
        removed_count = 0
        for checkpoint in checkpoints_to_remove:
            try:
                os.remove(checkpoint['path'])
                removed_count += 1
                print(f"Removed checkpoint: {checkpoint['filename']}")
            except Exception as e:
                print(f"Error removing checkpoint {checkpoint['path']}: {e}")
        
        print(f"Cleaned up {removed_count} checkpoints from {run_dir}")
        return removed_count
    
    def cleanup_all_runs(self, keep_latest_runs: int = 0, keep_latest_checkpoints: int = 0) -> Dict[str, int]:
        """
        Cleanup all runs and checkpoints.
        
        Args:
            keep_latest_runs: Number of latest runs to keep (0 = remove all)
            keep_latest_checkpoints: Number of latest checkpoints to keep per run (0 = remove all)
            
        Returns:
            Dictionary with cleanup statistics
        """
        runs = self.list_runs()
        
        if keep_latest_runs > 0:
            runs_to_remove = runs[keep_latest_runs:]
        else:
            runs_to_remove = runs
        
        cleanup_stats = {
            'runs_removed': 0,
            'checkpoints_removed': 0,
            'total_size_freed_mb': 0
        }
        
        for run in runs_to_remove:
            run_size_mb = run['size_mb']
            
            # Cleanup checkpoints in this run
            checkpoints_removed = self.cleanup_run(run['path'], keep_latest_checkpoints)
            cleanup_stats['checkpoints_removed'] += checkpoints_removed
            
            # Remove the entire run directory if no checkpoints left
            if keep_latest_checkpoints == 0 or checkpoints_removed == run['checkpoint_count']:
                try:
                    shutil.rmtree(run['path'])
                    cleanup_stats['runs_removed'] += 1
                    cleanup_stats['total_size_freed_mb'] += run_size_mb
                    print(f"Removed run directory: {run['name']}")
                except Exception as e:
                    print(f"Error removing run directory {run['path']}: {e}")
        
        print(f"Cleanup completed:")
        print(f"  Runs removed: {cleanup_stats['runs_removed']}")
        print(f"  Checkpoints removed: {cleanup_stats['checkpoints_removed']}")
        print(f"  Total size freed: {cleanup_stats['total_size_freed_mb']:.2f} MB")
        
        return cleanup_stats
    
    def get_best_checkpoint(self, run_dir: Optional[str] = None) -> Optional[str]:
        """
        Get the best checkpoint in a run directory.
        
        Args:
            run_dir: Run directory to search (default: current run)
            
        Returns:
            Path to best checkpoint or None
        """
        checkpoints = self.list_checkpoints(run_dir)
        
        best_checkpoint = None
        best_loss = float('inf')
        
        for checkpoint in checkpoints:
            if checkpoint['is_best'] and isinstance(checkpoint['loss'], (int, float)):
                if checkpoint['loss'] < best_loss:
                    best_loss = checkpoint['loss']
                    best_checkpoint = checkpoint['path']
        
        return best_checkpoint
    
    def _save_metadata(self, checkpoint_path: str, checkpoint_data: Dict):
        """Save checkpoint metadata to JSON file."""
        metadata_path = checkpoint_path.replace('.pt', '_metadata.json')
        
        metadata = {
            'epoch': checkpoint_data.get('epoch'),
            'loss': checkpoint_data.get('loss'),
            'timestamp': checkpoint_data.get('timestamp'),
            'is_best': checkpoint_data.get('is_best', False),
            'run_dir': checkpoint_data.get('run_dir')
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _cleanup_old_checkpoints(self):
        """Cleanup old checkpoints based on max_checkpoints setting."""
        if self.current_run_dir is None:
            return
        
        checkpoints = self.list_checkpoints(self.current_run_dir)
        
        if len(checkpoints) > self.max_checkpoints:
            # Remove oldest checkpoints
            checkpoints_to_remove = checkpoints[:-self.max_checkpoints]
            
            for checkpoint in checkpoints_to_remove:
                try:
                    os.remove(checkpoint['path'])
                    # Also remove metadata file
                    metadata_path = checkpoint['path'].replace('.pt', '_metadata.json')
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                except Exception as e:
                    print(f"Error removing old checkpoint {checkpoint['path']}: {e}")
    
    def _get_directory_size(self, directory: str) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size


def create_checkpoint_manager(base_dir: str = "checkpoints", max_checkpoints: int = 10) -> CheckpointManager:
    """
    Create a checkpoint manager instance.
    
    Args:
        base_dir: Base directory for storing checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        
    Returns:
        CheckpointManager instance
    """
    return CheckpointManager(base_dir, max_checkpoints)


if __name__ == "__main__":
    # Test the checkpoint manager
    print("Testing CheckpointManager...")
    
    # Create manager
    manager = CheckpointManager("test_checkpoints", max_checkpoints=3)
    
    # Create a test run
    run_dir = manager.create_run_directory("test_run")
    
    # Create a dummy model and optimizer
    import torch.nn as nn
    import torch.optim as optim
    
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters())
    
    # Save some test checkpoints
    for epoch in range(5):
        loss = 1.0 - epoch * 0.1
        is_best = epoch == 3
        
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=loss,
            is_best=is_best
        )
    
    # List checkpoints
    print("\nCheckpoints in current run:")
    checkpoints = manager.list_checkpoints()
    for checkpoint in checkpoints:
        print(f"  {checkpoint['filename']}: epoch={checkpoint['epoch']}, loss={checkpoint['loss']:.3f}, best={checkpoint['is_best']}")
    
    # List runs
    print("\nAll runs:")
    runs = manager.list_runs()
    for run in runs:
        print(f"  {run['name']}: {run['checkpoint_count']} checkpoints, {run['size_mb']:.2f} MB")
    
    # Test cleanup
    print("\nTesting cleanup...")
    cleanup_stats = manager.cleanup_all_runs(keep_latest_runs=1, keep_latest_checkpoints=2)
    
    # Clean up test directory
    shutil.rmtree("test_checkpoints")
    print("Test completed!")
