"""
Configuration Version Manager
Tracks configuration changes and links them with checkpoints
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import hashlib

try:
    from .config_manager import ConfigManager, TrainingConfigManager
except ImportError:
    from config_manager import ConfigManager, TrainingConfigManager


class ConfigVersionManager:
    """
    Manages configuration versions and their relationship with checkpoints.
    
    Features:
    - Track configuration changes over time
    - Link configurations with training runs
    - Compare configuration differences
    - Generate configuration reports
    - Maintain configuration history
    """
    
    def __init__(self, config_history_dir: str = "config_history"):
        """
        Initialize configuration version manager.
        
        Args:
            config_history_dir: Directory to store configuration history
        """
        self.config_history_dir = config_history_dir
        self.versions_dir = os.path.join(config_history_dir, "versions")
        self.links_dir = os.path.join(config_history_dir, "links")
        
        # Create directories
        os.makedirs(self.config_history_dir, exist_ok=True)
        os.makedirs(self.versions_dir, exist_ok=True)
        os.makedirs(self.links_dir, exist_ok=True)
        
        print(f"ConfigVersionManager initialized:")
        print(f"  History directory: {self.config_history_dir}")
        print(f"  Versions directory: {self.versions_dir}")
        print(f"  Links directory: {self.links_dir}")
    
    def save_config_version(
        self,
        config_path: str,
        version_name: Optional[str] = None,
        description: str = "",
        tags: List[str] = None
    ) -> str:
        """
        Save a configuration version.
        
        Args:
            config_path: Path to configuration file
            version_name: Custom version name (if None, auto-generated)
            description: Description of this configuration version
            tags: List of tags for this configuration
            
        Returns:
            Version ID
        """
        # Load configuration
        config_manager = ConfigManager(config_path)
        config = config_manager.get_config()
        
        # Generate version ID using unified timestamp manager
        if version_name is None:
            try:
                from .timestamp_manager import get_timestamp_manager
                timestamp_manager = get_timestamp_manager()
                current_timestamp = timestamp_manager.get_current_timestamp()
                if current_timestamp:
                    version_id = f"config_{current_timestamp}"
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    version_id = f"config_{timestamp}"
            except ImportError:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                version_id = f"config_{timestamp}"
        else:
            version_id = version_name
        
        # Create version metadata
        version_metadata = {
            'version_id': version_id,
            'created_at': datetime.now().isoformat(),
            'description': description,
            'tags': tags or [],
            'source_path': config_path,
            'config_hash': self._calculate_config_hash(config)
        }
        
        # Save configuration copy
        version_config_path = os.path.join(self.versions_dir, f"{version_id}.json")
        config_manager.save_config(version_config_path)
        
        # Save metadata
        metadata_path = os.path.join(self.versions_dir, f"{version_id}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(version_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration version saved: {version_id}")
        print(f"  Description: {description}")
        print(f"  Tags: {tags or []}")
        print(f"  Config file: {version_config_path}")
        
        return version_id
    
    def link_config_with_checkpoint(
        self,
        config_version_id: str,
        checkpoint_run_dir: str,
        link_type: str = "training"
    ):
        """
        Link a configuration version with a checkpoint run.
        
        Args:
            config_version_id: Configuration version ID
            checkpoint_run_dir: Checkpoint run directory
            link_type: Type of link (training, evaluation, etc.)
        """
        link_metadata = {
            'config_version_id': config_version_id,
            'checkpoint_run_dir': checkpoint_run_dir,
            'link_type': link_type,
            'linked_at': datetime.now().isoformat()
        }
        
        # Create link file
        link_filename = f"{config_version_id}_{os.path.basename(checkpoint_run_dir)}.json"
        link_path = os.path.join(self.links_dir, link_filename)
        
        with open(link_path, 'w', encoding='utf-8') as f:
            json.dump(link_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration linked with checkpoint:")
        print(f"  Config version: {config_version_id}")
        print(f"  Checkpoint run: {checkpoint_run_dir}")
        print(f"  Link type: {link_type}")
    
    def list_config_versions(self) -> List[Dict[str, Any]]:
        """
        List all configuration versions.
        
        Returns:
            List of configuration version information
        """
        versions = []
        
        for filename in os.listdir(self.versions_dir):
            if filename.endswith('_metadata.json'):
                metadata_path = os.path.join(self.versions_dir, filename)
                
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                versions.append(metadata)
        
        # Sort by creation time (newest first)
        versions.sort(key=lambda x: x['created_at'], reverse=True)
        
        return versions
    
    def print_config_versions(self):
        """Print all configuration versions in a formatted table."""
        versions = self.list_config_versions()
        
        if not versions:
            print("No configuration versions found.")
            return
        
        print(f"\n{'='*120}")
        print(f"CONFIGURATION VERSIONS")
        print(f"{'='*120}")
        print(f"{'Version ID':<25} {'Created':<20} {'Description':<30} {'Tags':<20} {'Hash':<12}")
        print(f"{'-'*120}")
        
        for version in versions:
            created = version['created_at'][:19] if version['created_at'] else 'N/A'
            description = version['description'][:29] if version['description'] else 'N/A'
            tags = ', '.join(version['tags'][:2]) if version['tags'] else 'N/A'
            config_hash = version['config_hash'][:11] if version['config_hash'] else 'N/A'
            
            print(f"{version['version_id']:<25} {created:<20} {description:<30} {tags:<20} {config_hash:<12}")
        
        print(f"{'='*120}")
    
    def compare_config_versions(
        self,
        version_id1: str,
        version_id2: str
    ) -> Dict[str, Any]:
        """
        Compare two configuration versions.
        
        Args:
            version_id1: First version ID
            version_id2: Second version ID
            
        Returns:
            Dictionary containing comparison results
        """
        # Load configurations
        config1_path = os.path.join(self.versions_dir, f"{version_id1}.json")
        config2_path = os.path.join(self.versions_dir, f"{version_id2}.json")
        
        if not os.path.exists(config1_path) or not os.path.exists(config2_path):
            raise ValueError("One or both configuration versions not found")
        
        config1_manager = ConfigManager(config1_path)
        config2_manager = ConfigManager(config2_path)
        
        config1 = config1_manager.get_config()
        config2 = config2_manager.get_config()
        
        # Compare configurations
        differences = self._compare_configs(config1, config2)
        
        comparison = {
            'version1': version_id1,
            'version2': version_id2,
            'differences': differences,
            'total_differences': len(differences),
            'compared_at': datetime.now().isoformat()
        }
        
        return comparison
    
    def print_config_comparison(self, version_id1: str, version_id2: str):
        """Print configuration comparison in a formatted table."""
        try:
            comparison = self.compare_config_versions(version_id1, version_id2)
            
            print(f"\n{'='*100}")
            print(f"CONFIGURATION COMPARISON")
            print(f"{'='*100}")
            print(f"Version 1: {version_id1}")
            print(f"Version 2: {version_id2}")
            print(f"Total differences: {comparison['total_differences']}")
            print(f"{'='*100}")
            
            if comparison['differences']:
                print(f"{'Section':<20} {'Parameter':<25} {'Version 1':<20} {'Version 2':<20}")
                print(f"{'-'*85}")
                
                for diff in comparison['differences']:
                    section = diff['section']
                    parameter = diff['parameter']
                    value1 = str(diff['value1'])
                    value2 = str(diff['value2'])
                    
                    print(f"{section:<20} {parameter:<25} {value1:<20} {value2:<20}")
            else:
                print("No differences found between configurations.")
            
            print(f"{'='*100}")
            
        except ValueError as e:
            print(f"Error: {e}")
    
    def get_checkpoint_config_history(self, checkpoint_run_dir: str) -> List[Dict[str, Any]]:
        """
        Get configuration history for a specific checkpoint run.
        
        Args:
            checkpoint_run_dir: Checkpoint run directory
            
        Returns:
            List of linked configuration versions
        """
        linked_configs = []
        
        for filename in os.listdir(self.links_dir):
            if filename.endswith('.json'):
                link_path = os.path.join(self.links_dir, filename)
                
                with open(link_path, 'r', encoding='utf-8') as f:
                    link_data = json.load(f)
                
                if link_data['checkpoint_run_dir'] == checkpoint_run_dir:
                    # Get configuration metadata
                    config_version_id = link_data['config_version_id']
                    metadata_path = os.path.join(self.versions_dir, f"{config_version_id}_metadata.json")
                    
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            config_metadata = json.load(f)
                        
                        linked_configs.append({
                            'config_version_id': config_version_id,
                            'link_type': link_data['link_type'],
                            'linked_at': link_data['linked_at'],
                            'config_metadata': config_metadata
                        })
        
        return linked_configs
    
    def auto_save_config_version(
        self,
        config_path: str,
        checkpoint_run_dir: Optional[str] = None,
        description: str = "",
        tags: List[str] = None
    ) -> str:
        """
        Automatically save configuration version and link with checkpoint if provided.
        
        Args:
            config_path: Path to configuration file
            checkpoint_run_dir: Checkpoint run directory (optional)
            description: Description of this configuration version
            tags: List of tags for this configuration
            
        Returns:
            Version ID
        """
        # Save configuration version
        version_id = self.save_config_version(
            config_path=config_path,
            description=description,
            tags=tags
        )
        
        # Link with checkpoint if provided
        if checkpoint_run_dir:
            self.link_config_with_checkpoint(
                config_version_id=version_id,
                checkpoint_run_dir=checkpoint_run_dir
            )
        
        return version_id
    
    def _calculate_config_hash(self, config: TrainingConfigManager) -> str:
        """Calculate hash of configuration for change detection."""
        from dataclasses import asdict
        config_dict = asdict(config)
        
        # Convert to JSON string and calculate hash
        config_str = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _compare_configs(
        self,
        config1: TrainingConfigManager,
        config2: TrainingConfigManager
    ) -> List[Dict[str, Any]]:
        """Compare two configurations and return differences."""
        from dataclasses import asdict
        
        config1_dict = asdict(config1)
        config2_dict = asdict(config2)
        
        differences = []
        
        # Compare each section
        for section_name in config1_dict:
            if section_name in config2_dict:
                section1 = config1_dict[section_name]
                section2 = config2_dict[section_name]
                
                # Compare parameters within section
                for param_name in section1:
                    if param_name in section2:
                        if section1[param_name] != section2[param_name]:
                            differences.append({
                                'section': section_name,
                                'parameter': param_name,
                                'value1': section1[param_name],
                                'value2': section2[param_name]
                            })
                    else:
                        differences.append({
                            'section': section_name,
                            'parameter': param_name,
                            'value1': section1[param_name],
                            'value2': 'MISSING'
                        })
                else:
                    # Check for parameters only in section2
                    for param_name in section2:
                        if param_name not in section1:
                            differences.append({
                                'section': section_name,
                                'parameter': param_name,
                                'value1': 'MISSING',
                                'value2': section2[param_name]
                            })
            else:
                # Section missing in config2
                for param_name in config1_dict[section_name]:
                    differences.append({
                        'section': section_name,
                        'parameter': param_name,
                        'value1': config1_dict[section_name][param_name],
                        'value2': 'MISSING'
                    })
        
        return differences


def main():
    """Test the ConfigVersionManager."""
    print("Testing ConfigVersionManager...")
    
    # Initialize manager
    manager = ConfigVersionManager()
    
    # Save some configuration versions
    print("\nSaving configuration versions...")
    
    version1 = manager.save_config_version(
        config_path="training_config.json",
        version_name="baseline_config",
        description="Baseline configuration for initial training",
        tags=["baseline", "initial"]
    )
    
    # Simulate configuration change
    print("\nSimulating configuration change...")
    # In real usage, you would modify the config file here
    
    version2 = manager.save_config_version(
        config_path="training_config.json",
        version_name="modified_config",
        description="Modified configuration with different learning rate",
        tags=["modified", "lr_change"]
    )
    
    # List versions
    print("\nListing configuration versions...")
    manager.print_config_versions()
    
    # Compare versions
    print("\nComparing configurations...")
    manager.print_config_comparison(version1, version2)
    
    print("\nConfigVersionManager test completed!")


if __name__ == "__main__":
    main()
