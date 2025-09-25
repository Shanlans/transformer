#!/usr/bin/env python3
"""
Config Protection Demo

演示配置覆盖保护功能，包括：
1. 检查配置覆盖是否会影响实验信息
2. 保护实验信息不被意外覆盖
3. 提供配置管理建议
"""

import sys
import os
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_manager import ConfigManager

def demo_config_protection():
    """演示配置保护功能"""
    
    print("=" * 80)
    print("🔍 CONFIG PROTECTION DEMO")
    print("=" * 80)
    
    # 1. 检查experiment config
    print("\n1️⃣ 检查Experiment Config:")
    print("-" * 40)
    
    exp_config_path = "experiments/configs/default_training_20250926_073904_cloud.json"
    if os.path.exists(exp_config_path):
        config_manager = ConfigManager(exp_config_path)
        analysis = config_manager.check_config_overwrite(exp_config_path)
        
        print(f"📁 Config: {exp_config_path}")
        print(f"🔬 Has experiment info: {analysis['has_experiment_info']}")
        print(f"🛡️  Would preserve: {analysis['would_preserve']}")
        
        if analysis['has_experiment_info']:
            exp_info = analysis['experiment_info']
            print(f"📊 Experiment: {exp_info.get('name', 'N/A')}")
            print(f"⏰ Created: {exp_info.get('created_at', 'N/A')}")
            print(f"🎯 Environment: {exp_info.get('training_environment', 'N/A')}")
            print(f"🖥️  GPU: {exp_info.get('gpu_info', {}).get('type', 'N/A')}")
    
    # 2. 检查base config
    print("\n2️⃣ 检查Base Config:")
    print("-" * 40)
    
    base_config_path = "training_config.json"
    if os.path.exists(base_config_path):
        config_manager = ConfigManager(base_config_path)
        analysis = config_manager.check_config_overwrite(base_config_path)
        
        print(f"📁 Config: {base_config_path}")
        print(f"🔬 Has experiment info: {analysis['has_experiment_info']}")
        print(f"🛡️  Would preserve: {analysis['would_preserve']}")
        
        if analysis['has_experiment_info']:
            exp_info = analysis['experiment_info']
            print(f"📊 Experiment: {exp_info.get('name', 'N/A')}")
            print(f"⏰ Created: {exp_info.get('created_at', 'N/A')}")
    
    # 3. 演示配置更新保护
    print("\n3️⃣ 演示配置更新保护:")
    print("-" * 40)
    
    if os.path.exists(exp_config_path):
        print("📝 模拟配置更新...")
        
        # 加载配置
        config_manager = ConfigManager(exp_config_path)
        
        # 更新一些参数
        updates = {
            'training.learning_rate': 0.0002,
            'training.epochs': 5,
            'model.d_model': 768
        }
        
        print(f"🔄 更新参数: {updates}")
        config_manager.update_config(updates)
        
        # 检查更新后的配置
        updated_config = config_manager.get_config()
        print(f"✅ Learning rate: {updated_config.training.learning_rate}")
        print(f"✅ Epochs: {updated_config.training.epochs}")
        print(f"✅ Model dimension: {updated_config.model.d_model}")
        
        # 检查实验信息是否被保护
        print(f"🛡️  Experiment name: {updated_config.experiment.name}")
        print(f"🛡️  Timestamp ID: {updated_config.experiment.timestamp_id}")
        print(f"🛡️  Environment: {updated_config.experiment.training_environment}")
    
    # 4. 提供最佳实践建议
    print("\n4️⃣ 最佳实践建议:")
    print("-" * 40)
    
    recommendations = [
        "✅ 使用experiment config进行训练，避免直接修改base config",
        "✅ 使用 --check-config-overwrite 检查配置覆盖影响",
        "✅ 使用 preserve_experiment_info=True 保护实验信息",
        "✅ 创建新experiment而不是覆盖现有配置",
        "✅ 使用config version管理来跟踪配置变化",
        "✅ 分离实验信息和训练参数"
    ]
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print("\n💡 使用命令行工具:")
    print("   python run.py --check-config-overwrite <config_path>")
    print("   python run.py --list-config-versions")
    print("   python run.py --create-experiment")

if __name__ == "__main__":
    demo_config_protection()
