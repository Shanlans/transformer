"""
Transformer模型框架
基于论文 "Attention Is All You Need" (Vaswani et al., 2017)

TODO: 请根据指导逐步实现以下组件：
1. 位置编码 (PositionalEncoding)
2. 多头注意力机制 (MultiHeadAttention)
3. 前馈网络 (FeedForward)
4. 编码器层 (EncoderLayer)
5. 解码器层 (DecoderLayer)
6. 完整Transformer模型 (Transformer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    位置编码层
    
    TODO: 实现位置编码
    - 使用sin和cos函数生成位置编码
    - 支持不同长度的序列
    - 与词嵌入相加
    """
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # TODO: 实现位置编码逻辑
        pass
    
    def forward(self, x):
        # TODO: 实现前向传播
        pass


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    TODO: 实现多头注意力
    - 线性变换生成Q, K, V
    - 分割成多个头
    - 计算注意力分数
    - 应用mask（如果需要）
    - 合并多头输出
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        # TODO: 实现多头注意力逻辑
        pass
    
    def forward(self, query, key, value, mask=None):
        # TODO: 实现前向传播
        pass


class FeedForward(nn.Module):
    """
    前馈网络
    
    TODO: 实现前馈网络
    - 两个线性层
    - ReLU激活函数
    - Dropout正则化
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        # TODO: 实现前馈网络逻辑
        pass
    
    def forward(self, x):
        # TODO: 实现前向传播
        pass


class EncoderLayer(nn.Module):
    """
    编码器层
    
    TODO: 实现编码器层
    - 自注意力机制
    - 残差连接
    - 层归一化
    - 前馈网络
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # TODO: 实现编码器层逻辑
        pass
    
    def forward(self, x, mask=None):
        # TODO: 实现前向传播
        pass


class DecoderLayer(nn.Module):
    """
    解码器层
    
    TODO: 实现解码器层
    - 自注意力机制（带mask）
    - 交叉注意力机制
    - 前馈网络
    - 残差连接和层归一化
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # TODO: 实现解码器层逻辑
        pass
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # TODO: 实现前向传播
        pass


class Transformer(nn.Module):
    """
    完整的Transformer模型
    
    TODO: 实现完整Transformer
    - 词嵌入层
    - 位置编码
    - 编码器堆叠
    - 解码器堆叠
    - 输出投影层
    - 创建mask的辅助方法
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_encoder_layers=6, n_decoder_layers=6, d_ff=2048, 
                 max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        # TODO: 实现完整Transformer逻辑
        pass
    
    def create_padding_mask(self, seq, pad_idx=0):
        # TODO: 实现padding mask
        pass
    
    def create_look_ahead_mask(self, size):
        # TODO: 实现look-ahead mask
        pass
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # TODO: 实现前向传播
        pass


def create_model(src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                n_encoder_layers=6, n_decoder_layers=6, d_ff=2048, dropout=0.1):
    """
    创建Transformer模型的便捷函数
    
    Args:
        src_vocab_size: 源语言词汇表大小
        tgt_vocab_size: 目标语言词汇表大小
        d_model: 模型维度
        n_heads: 注意力头数
        n_encoder_layers: 编码器层数
        n_decoder_layers: 解码器层数
        d_ff: 前馈网络隐藏层维度
        dropout: Dropout概率
    
    Returns:
        Transformer模型实例
    """
    return Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff,
        dropout=dropout
    )


if __name__ == "__main__":
    print("🧪 Transformer模型框架已创建")
    print("📝 请根据TODO注释逐步实现各个组件")
    print("🔧 实现完成后，可以运行训练脚本进行测试")
