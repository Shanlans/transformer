#!/usr/bin/env python3
"""
测试脚本：验证PyTorch环境是否正确配置
"""

import sys
import torch
import torchvision
import torchaudio
import numpy as np

def test_environment():
    """测试环境配置"""
    print("=" * 50)
    print("PyTorch环境测试")
    print("=" * 50)
    
    # Python版本
    print(f"Python版本: {sys.version}")
    print()
    
    # PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Torchvision版本: {torchvision.__version__}")
    print(f"Torchaudio版本: {torchaudio.__version__}")
    print()
    
    # CUDA支持
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
    else:
        print("使用CPU进行计算")
    print()
    
    # 基本功能测试
    print("基本功能测试:")
    
    # 创建张量
    x = torch.randn(3, 4)
    print(f"创建随机张量: {x.shape}")
    
    # 矩阵运算
    y = torch.mm(x, x.t())
    print(f"矩阵乘法结果: {y.shape}")
    
    # NumPy兼容性
    np_array = np.array([1, 2, 3, 4])
    torch_tensor = torch.from_numpy(np_array)
    print(f"NumPy到PyTorch转换: {torch_tensor}")
    
    print()
    print("✅ 环境配置测试通过！")
    print("=" * 50)

if __name__ == "__main__":
    test_environment()

