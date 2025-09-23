"""
Transformer训练器 - 基础框架
请根据以下指导实现完整的训练器类
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import time
import os


class TransformerTrainer:
    """
    Transformer训练器
    
    需要实现的功能：
    1. 模型训练循环
    2. 验证循环
    3. 损失计算
    4. 优化器管理
    5. 学习率调度
    6. 模型保存和加载
    7. 训练日志记录
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
        optimizer: optim.Optimizer = None,
        criterion: nn.Module = None,
        device: str = "cpu"
    ):
        """
        初始化训练器
        
        Args:
            model: Transformer模型
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            optimizer: 优化器
            criterion: 损失函数
            device: 设备
        """
        # TODO: 实现初始化逻辑
        pass
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        # TODO: 实现训练循环
        pass
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        # TODO: 实现验证循环
        pass
    
    def train(self, num_epochs: int, save_path: str = None) -> List[Dict[str, float]]:
        """完整训练流程"""
        # TODO: 实现完整训练流程
        pass
    
    def save_model(self, path: str):
        """保存模型"""
        # TODO: 实现模型保存
        pass
    
    def load_model(self, path: str):
        """加载模型"""
        # TODO: 实现模型加载
        pass


# 指导问题：
"""
1. 您想使用什么优化器？Adam, AdamW, SGD？
2. 您需要学习率调度吗？StepLR, CosineAnnealingLR？
3. 您希望如何计算损失？交叉熵、标签平滑？
4. 您需要早停机制吗？
5. 您希望保存哪些训练状态？模型权重、优化器状态、学习率？
"""
