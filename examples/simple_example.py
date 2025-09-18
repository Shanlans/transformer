"""
简单示例
展示如何使用Transformer框架进行训练

TODO: 请根据指导逐步实现以下功能：
1. 导入必要的模块
2. 创建简单数据集
3. 创建小模型
4. 训练模型
5. 测试模型
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# TODO: 导入自定义模块
# from models.transformer import create_model
# from data.dataset import NumberSequenceDataset, create_data_loaders
# from training.trainer import TransformerTrainer
# from utils.helpers import set_seed, get_device, count_parameters


def create_simple_dataset():
    """
    创建简单数据集
    
    TODO: 实现简单数据集创建
    - 使用NumberSequenceDataset
    - 设置小规模参数
    - 返回数据加载器
    """
    # TODO: 实现数据集创建逻辑
    pass


def create_simple_model(vocab_size):
    """
    创建简单模型
    
    TODO: 实现简单模型创建
    - 使用较小的模型参数
    - 适合快速训练和测试
    - 返回模型实例
    """
    # TODO: 实现模型创建逻辑
    pass


def train_simple_model():
    """
    训练简单模型
    
    TODO: 实现简单模型训练
    - 创建数据集
    - 创建模型
    - 创建训练器
    - 开始训练
    - 测试模型
    """
    print("🚀 开始简单模型训练示例")
    
    # TODO: 设置随机种子
    # set_seed(42)
    
    # TODO: 获取设备
    # device = get_device()
    # print(f"使用设备: {device}")
    
    # TODO: 创建数据集
    # train_loader, val_loader, vocab_size = create_simple_dataset()
    # print(f"词汇表大小: {vocab_size}")
    
    # TODO: 创建模型
    # model = create_simple_model(vocab_size)
    # print(f"模型参数数量: {count_parameters(model):,}")
    
    # TODO: 创建训练器
    # trainer = TransformerTrainer(model, train_loader, val_loader, device)
    
    # TODO: 开始训练
    # print("📚 开始训练...")
    # trainer.train(num_epochs=5, save_path='simple_model.pth')
    
    # TODO: 测试模型
    # print("🧪 测试模型...")
    # test_model(model, device)
    
    print("✅ 简单模型训练完成！")


def test_model(model, device):
    """
    测试模型
    
    TODO: 实现模型测试
    - 生成测试样本
    - 显示预测结果
    - 计算准确率
    """
    # TODO: 实现模型测试逻辑
    pass


if __name__ == "__main__":
    print("🧪 简单示例框架已创建")
    print("📝 请根据TODO注释逐步实现示例功能")
    print("🔧 实现完成后，可以运行: python examples/simple_example.py")
    print("")
    print("💡 这个示例将展示:")
    print("1. 如何创建简单的数据集")
    print("2. 如何创建小规模的模型")
    print("3. 如何进行快速训练")
    print("4. 如何测试模型性能")
    print("")
    print("🎯 建议先完成这个简单示例，再尝试更复杂的任务")
