#!/usr/bin/env python3
"""
Cloud Training Monitoring Demo

演示cloud训练监控功能，包括：
1. 启动cloud训练
2. 实时监控训练状态
3. 查看训练日志
4. 停止训练
"""

import time
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.colab_manager import ColabManager

def demo_cloud_monitoring():
    """演示cloud训练监控功能"""
    
    print("=" * 80)
    print("☁️  CLOUD TRAINING MONITORING DEMO")
    print("=" * 80)
    
    # 创建ColabManager实例
    colab_manager = ColabManager()
    
    # 模拟连接状态（用于演示）
    colab_manager.is_connected = True
    
    # 模拟启动cloud训练
    print("\n🚀 启动cloud训练...")
    success = colab_manager.start_remote_training('demo_experiment', 'training_config.json')
    
    if not success:
        print("❌ 启动cloud训练失败")
        return
    
    print("✅ Cloud训练已启动!")
    
    # 模拟训练过程监控
    print("\n📊 开始监控训练过程...")
    print("=" * 60)
    
    for i in range(5):
        print(f"\n⏰ 检查 #{i+1} (等待 {i*2} 秒后):")
        
        # 检查状态
        status = colab_manager.check_training_status()
        
        if status['status'] == 'no_session':
            print("ℹ️  没有活跃的训练会话")
            break
        elif status['status'] == 'disconnected':
            print("❌ ColabCode会话未连接")
            break
        elif status['status'] == 'error':
            print(f"❌ 错误: {status['message']}")
            break
        
        # 显示状态信息
        print(f"📋 实验: {status['experiment_name']}")
        print(f"⏰ 开始时间: {status['start_time']}")
        print(f"⏱️  已运行: {status['elapsed_time']}")
        print(f"📁 配置: {status['config_path']}")
        
        # 状态特定信息
        if status['status'] == 'initializing':
            print("🔄 状态: 初始化中")
            print("📝 消息: 训练正在初始化...")
        elif status['status'] == 'training':
            print("🏃 状态: 训练进行中")
            print(f"📝 消息: {status['message']}")
            print(f"📊 进度: {status['progress']}%")
            print(f"🎯 Epoch: {status['current_epoch']}/{status['total_epochs']}")
            print(f"⏳ 预计完成: {status['estimated_completion']}")
        elif status['status'] == 'completed':
            print("✅ 状态: 已完成")
            print("📝 消息: 训练完成!")
            print(f"📊 进度: {status['progress']}%")
            print("🎯 Epoch: 2/2")
            print("⏳ 预计完成: 已完成")
            break
        
        # 显示最近的日志
        print("\n📋 最近的训练日志:")
        logs = colab_manager.get_training_logs(3)
        for log in logs:
            print(f"  {log}")
        
        # 等待下一次检查
        if i < 4:  # 最后一次不等待
            print(f"\n⏳ 等待 2 秒后进行下一次检查...")
            time.sleep(2)
    
    print("\n" + "=" * 60)
    print("📊 监控演示完成!")
    
    # 显示如何使用命令行工具
    print("\n💡 使用命令行工具监控cloud训练:")
    print("  python run.py --cloud-status    # 检查训练状态")
    print("  python run.py --cloud-logs      # 查看训练日志")
    print("  python run.py --cloud-stop      # 停止训练")
    
    print("\n💡 在真实环境中，这些命令会:")
    print("  - 通过SSH连接到ColabCode会话")
    print("  - 检查训练进程状态")
    print("  - 获取实时训练日志")
    print("  - 监控GPU使用情况")
    print("  - 显示训练进度和损失值")

if __name__ == "__main__":
    demo_cloud_monitoring()
