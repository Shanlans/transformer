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
    
    使用sin和cos函数生成位置编码，为序列中的每个位置提供位置信息
    数学公式：
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000, position_factor=10000, fast_model=True):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # 创建位置索引 [max_len, 1]
        # position = [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算div_term，用于10000^(2i/d_model)
        if fast_model:
            # 快速模式：使用exp(-2i*ln(10000)/d_model)避免数值不稳定
            # div_term shape: [d_model//2]
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                               (-math.log(position_factor) / d_model))
        else:
            # 标准模式：直接计算10000^(2i/d_model)
            # div_term shape: [d_model//2]
            div_term = torch.float_power(position_factor, 
                                       torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
            div_term = torch.div(1, div_term)  # 取倒数
        
        # 对偶数维度应用sin函数 [max_len, d_model//2]
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # 对奇数维度应用cos函数 [max_len, d_model//2]
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 调整维度 [max_len, d_model] -> [max_len, 1, d_model]
        # 这样便于后续广播到 [seq_len, batch_size, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 注册为buffer（不参与梯度更新，但会随模型一起保存）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [seq_len, batch_size, d_model]
        
        Returns:
            输出张量 [seq_len, batch_size, d_model]
        """
        # 输入: x [seq_len, batch_size, d_model]
        # 位置编码: self.pe [max_len, 1, d_model]
        # 取前seq_len个位置: self.pe[:x.size(0), :] [seq_len, 1, d_model]
        # 广播相加: x + self.pe[:x.size(0), :] [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    数学公式：
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # 线性投影层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力机制
        
        Args:
            Q: Query矩阵 [batch_size, n_heads, seq_len, d_k]
            K: Key矩阵 [batch_size, n_heads, seq_len, d_k]
            V: Value矩阵 [batch_size, n_heads, seq_len, d_v]
            mask: 掩码 [batch_size, n_heads, seq_len, seq_len] 或 None
        
        Returns:
            output: 注意力输出 [batch_size, n_heads, seq_len, d_v]
            attention_weights: 注意力权重 [batch_size, n_heads, seq_len, seq_len]
        """
        d_k = Q.size(-1)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # [batch_size, n_heads, seq_len, seq_len]
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # 应用dropout
        attention_weights = self.dropout(attention_weights)
        
        # 计算加权输出
        output = torch.matmul(attention_weights, V)
        # [batch_size, n_heads, seq_len, d_v]
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        Args:
            query: 查询矩阵 [batch_size, seq_len, d_model]
            key: 键矩阵 [batch_size, seq_len, d_model]
            value: 值矩阵 [batch_size, seq_len, d_model]
            mask: 掩码 [batch_size, n_heads, seq_len, seq_len] 或 None
        
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = query.size()
        
        # 1. 线性变换
        Q = self.W_Q(query)  # [batch_size, seq_len, d_model]
        K = self.W_K(key)    # [batch_size, seq_len, d_model]
        V = self.W_V(value)  # [batch_size, seq_len, d_model]
        
        # 2. 重塑为多头
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # [batch_size, seq_len, n_heads, d_k] -> [batch_size, n_heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_v).transpose(1, 2)
        
        # 3. 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )
        # attention_output: [batch_size, n_heads, seq_len, d_v]
        
        # 4. 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        # [batch_size, n_heads, seq_len, d_v] -> [batch_size, seq_len, d_model]
        
        # 5. 输出投影
        output = self.W_O(attention_output)
        # [batch_size, seq_len, d_model]
        
        return output
        

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
