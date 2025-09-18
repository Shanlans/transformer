"""
数据集框架
用于训练Transformer模型的数据处理

TODO: 请根据指导逐步实现以下组件：
1. 基础数据集类 (BaseDataset)
2. 简单翻译数据集 (SimpleTranslationDataset)
3. 数字序列数据集 (NumberSequenceDataset)
4. 数据加载器工具函数
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
import string


class BaseDataset(Dataset):
    """
    基础数据集类
    
    TODO: 实现基础数据集功能
    - 定义特殊token
    - 实现序列填充方法
    - 提供基础接口
    """
    
    def __init__(self):
        # TODO: 定义特殊token
        # self.PAD_TOKEN = ?
        # self.SOS_TOKEN = ?
        # self.EOS_TOKEN = ?
        # self.UNK_TOKEN = ?
        pass
    
    def _pad_sequence(self, seq, max_length):
        """
        填充序列到指定长度
        
        TODO: 实现序列填充逻辑
        """
        pass
    
    def __len__(self):
        # TODO: 返回数据集大小
        pass
    
    def __getitem__(self, idx):
        # TODO: 返回指定索引的数据
        pass


class SimpleTranslationDataset(BaseDataset):
    """
    简单翻译数据集
    
    TODO: 实现简单翻译数据集
    - 生成随机翻译对
    - 支持不同序列长度
    - 添加特殊token
    """
    
    def __init__(self, vocab_size=100, max_length=20, num_samples=10000):
        super().__init__()
        # TODO: 实现初始化逻辑
        pass
    
    def _generate_data(self):
        """
        生成随机翻译数据
        
        TODO: 实现数据生成逻辑
        """
        pass


class NumberSequenceDataset(BaseDataset):
    """
    数字序列数据集 - 学习数字的某种变换
    
    TODO: 实现数字序列数据集
    - 生成数字序列对
    - 实现某种数学变换
    - 支持不同长度序列
    """
    
    def __init__(self, max_number=100, max_length=10, num_samples=5000):
        super().__init__()
        # TODO: 实现初始化逻辑
        pass
    
    def _generate_data(self):
        """
        生成数字序列数据
        
        TODO: 实现数据生成逻辑
        """
        pass


class SimpleLanguageDataset(BaseDataset):
    """
    简单语言数据集 - 学习字符级别的变换
    
    TODO: 实现字符级语言数据集
    - 字符到索引映射
    - 字符串变换规则
    - 序列编码和解码
    """
    
    def __init__(self, max_length=15, num_samples=8000):
        super().__init__()
        # TODO: 实现初始化逻辑
        pass
    
    def _generate_data(self):
        """
        生成字符序列数据
        
        TODO: 实现数据生成逻辑
        """
        pass
    
    def decode_sequence(self, indices):
        """
        将索引序列解码为字符串
        
        TODO: 实现序列解码逻辑
        """
        pass


def create_data_loaders(dataset_class, batch_size=32, train_ratio=0.8, **dataset_kwargs):
    """
    创建训练和验证数据加载器
    
    TODO: 实现数据加载器创建
    - 分割训练和验证集
    - 创建DataLoader
    - 返回加载器和词汇表大小
    """
    # TODO: 实现数据加载器创建逻辑
    pass


if __name__ == "__main__":
    print("🧪 数据集框架已创建")
    print("📝 请根据TODO注释逐步实现各个数据集类")
    print("🔧 实现完成后，可以用于训练Transformer模型")
