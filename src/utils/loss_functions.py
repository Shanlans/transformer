"""
损失函数 - 基础框架
请根据以下指导实现适合Transformer的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失
    
    需要实现的功能：
    1. 标签平滑技术
    2. 忽略填充标记
    3. 计算平均损失
    """
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        """
        初始化损失函数
        
        Args:
            smoothing: 标签平滑参数
            ignore_index: 忽略的索引（通常是PAD标记）
        """
        # TODO: 实现初始化
        pass
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算损失
        
        Args:
            predictions: 模型预测 [batch_size, seq_len, vocab_size]
            targets: 目标序列 [batch_size, seq_len]
        
        Returns:
            损失值
        """
        # TODO: 实现标签平滑交叉熵损失
        pass


class MaskedCrossEntropy(nn.Module):
    """
    带掩码的交叉熵损失
    
    需要实现的功能：
    1. 创建掩码忽略填充位置
    2. 计算有效位置的损失
    3. 返回平均损失
    """
    
    def __init__(self, ignore_index: int = 0):
        """
        初始化损失函数
        
        Args:
            ignore_index: 忽略的索引
        """
        # TODO: 实现初始化
        pass
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算损失
        
        Args:
            predictions: 模型预测 [batch_size, seq_len, vocab_size]
            targets: 目标序列 [batch_size, seq_len]
        
        Returns:
            损失值
        """
        # TODO: 实现带掩码的交叉熵损失
        pass


# 指导问题：
"""
1. 您想使用标签平滑吗？平滑参数设为多少？
2. 您需要忽略哪些标记？PAD, UNK？
3. 您希望如何计算损失？平均、求和、还是其他？
4. 您需要支持不同的损失函数吗？
"""
