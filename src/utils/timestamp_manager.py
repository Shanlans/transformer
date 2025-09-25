"""
统一时间戳管理器

确保所有训练相关的文件和目录使用统一的时间戳：
- Experiment创建时间戳
- Checkpoint目录时间戳  
- Config history时间戳
- Cloud training时间戳
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any


class TimestampManager:
    """统一时间戳管理器"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.timestamp_file = os.path.join(base_dir, ".experiment_timestamp")
        self._current_timestamp = None
    
    def create_experiment_timestamp(self) -> str:
        """
        创建新的实验时间戳
        
        Returns:
            str: 格式为 YYYYMMDD_HHMMSS 的时间戳
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_timestamp = timestamp
        self._save_timestamp(timestamp)
        return timestamp
    
    def get_current_timestamp(self) -> Optional[str]:
        """
        获取当前实验的时间戳
        
        Returns:
            str: 当前时间戳，如果没有则返回None
        """
        if self._current_timestamp:
            return self._current_timestamp
        
        if os.path.exists(self.timestamp_file):
            try:
                with open(self.timestamp_file, 'r') as f:
                    data = json.load(f)
                    self._current_timestamp = data.get('timestamp')
                    return self._current_timestamp
            except Exception:
                pass
        
        return None
    
    def set_timestamp(self, timestamp: str):
        """
        设置当前实验的时间戳
        
        Args:
            timestamp: 时间戳字符串
        """
        self._current_timestamp = timestamp
        self._save_timestamp(timestamp)
    
    def clear_timestamp(self):
        """清除当前时间戳"""
        self._current_timestamp = None
        if os.path.exists(self.timestamp_file):
            os.remove(self.timestamp_file)
    
    def _save_timestamp(self, timestamp: str):
        """保存时间戳到文件"""
        try:
            data = {
                'timestamp': timestamp,
                'created_at': datetime.now().isoformat(),
                'description': 'Current experiment timestamp'
            }
            with open(self.timestamp_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save timestamp: {e}")
    
    def get_run_directory_name(self) -> str:
        """
        获取统一的run目录名称
        
        Returns:
            str: run_YYYYMMDD_HHMMSS 格式的目录名
        """
        timestamp = self.get_current_timestamp()
        if not timestamp:
            raise ValueError("No experiment timestamp available. Create an experiment first.")
        return f"run_{timestamp}"
    
    def get_config_version_name(self) -> str:
        """
        获取统一的config版本名称
        
        Returns:
            str: config_YYYYMMDD_HHMMSS 格式的文件名
        """
        timestamp = self.get_current_timestamp()
        if not timestamp:
            raise ValueError("No experiment timestamp available. Create an experiment first.")
        return f"config_{timestamp}"
    
    def get_link_name(self, checkpoint_run_dir: str) -> str:
        """
        获取统一的link文件名
        
        Args:
            checkpoint_run_dir: checkpoint的run目录名
            
        Returns:
            str: config_YYYYMMDD_HHMMSS_run_YYYYMMDD_HHMMSS.json 格式的文件名
        """
        timestamp = self.get_current_timestamp()
        if not timestamp:
            raise ValueError("No experiment timestamp available. Create an experiment first.")
        return f"config_{timestamp}_{checkpoint_run_dir}.json"
    
    def validate_timestamp_consistency(self) -> Dict[str, Any]:
        """
        验证时间戳一致性
        
        Returns:
            Dict: 验证结果
        """
        current_timestamp = self.get_current_timestamp()
        if not current_timestamp:
            return {
                'consistent': False,
                'error': 'No current timestamp available'
            }
        
        # 检查各种文件的时间戳一致性
        issues = []
        
        # 检查checkpoints目录
        checkpoints_dir = os.path.join(self.base_dir, "checkpoints")
        if os.path.exists(checkpoints_dir):
            for item in os.listdir(checkpoints_dir):
                if item.startswith("run_"):
                    if item != f"run_{current_timestamp}":
                        issues.append(f"Checkpoint directory {item} doesn't match current timestamp")
        
        # 检查experiments目录
        experiments_dir = os.path.join(self.base_dir, "experiments", "configs")
        if os.path.exists(experiments_dir):
            for item in os.listdir(experiments_dir):
                if item.endswith(".json") and not item.startswith("resume_"):
                    if current_timestamp not in item:
                        issues.append(f"Experiment config {item} doesn't contain current timestamp")
        
        # 检查config history
        config_history_dir = os.path.join(self.base_dir, "config_history", "versions")
        if os.path.exists(config_history_dir):
            for item in os.listdir(config_history_dir):
                if item.startswith("config_") and item.endswith(".json"):
                    if item != f"config_{current_timestamp}.json":
                        issues.append(f"Config history {item} doesn't match current timestamp")
        
        return {
            'consistent': len(issues) == 0,
            'current_timestamp': current_timestamp,
            'issues': issues
        }
    
    def print_timestamp_status(self):
        """打印时间戳状态"""
        current_timestamp = self.get_current_timestamp()
        if current_timestamp:
            print(f"🕐 Current experiment timestamp: {current_timestamp}")
            print(f"📁 Expected run directory: run_{current_timestamp}")
            print(f"📋 Expected config version: config_{current_timestamp}.json")
        else:
            print("❌ No experiment timestamp set")
        
        # 验证一致性
        validation = self.validate_timestamp_consistency()
        if validation['consistent']:
            print("✅ All timestamps are consistent")
        else:
            print("⚠️  Timestamp inconsistencies found:")
            for issue in validation['issues']:
                print(f"   - {issue}")


# 全局时间戳管理器实例
_timestamp_manager = None

def get_timestamp_manager() -> TimestampManager:
    """获取全局时间戳管理器实例"""
    global _timestamp_manager
    if _timestamp_manager is None:
        _timestamp_manager = TimestampManager()
    return _timestamp_manager

def create_experiment_timestamp() -> str:
    """创建新的实验时间戳"""
    return get_timestamp_manager().create_experiment_timestamp()

def get_current_timestamp() -> Optional[str]:
    """获取当前实验时间戳"""
    return get_timestamp_manager().get_current_timestamp()

def set_timestamp(timestamp: str):
    """设置当前实验时间戳"""
    get_timestamp_manager().set_timestamp(timestamp)

def clear_timestamp():
    """清除当前实验时间戳"""
    get_timestamp_manager().clear_timestamp()
