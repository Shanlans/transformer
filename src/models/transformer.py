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
    
    数学公式：FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    
    网络结构：
    输入 x: [batch_size, seq_len, d_model]
        ↓
    线性层1: [batch_size, seq_len, d_model] → [batch_size, seq_len, d_ff]
        ↓
    ReLU激活: max(0, x)
        ↓
    Dropout: 随机置零部分神经元
        ↓
    线性层2: [batch_size, seq_len, d_ff] → [batch_size, seq_len, d_model]
        ↓
    输出 y: [batch_size, seq_len, d_model]
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        # 第一个线性层：d_model → d_ff
        self.W_1 = nn.Linear(d_model, d_ff)
        # 第二个线性层：d_ff → d_model
        self.W_2 = nn.Linear(d_ff, d_model)
        # Dropout正则化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        # 第一个线性变换
        x = self.W_1(x)  # [batch_size, seq_len, d_model] → [batch_size, seq_len, d_ff]
        
        # ReLU激活函数
        x = F.relu(x)    # 引入非线性
        
        # Dropout正则化
        x = self.dropout(x)
        
        # 第二个线性变换
        x = self.W_2(x)  # [batch_size, seq_len, d_ff] → [batch_size, seq_len, d_model]
        
        return x


class EncoderLayer(nn.Module):
    """
    编码器层
    
    数学公式：
    LayerNorm(x + MultiHeadAttention(x))
    LayerNorm(x + FeedForward(x))
    
    网络结构：
    输入 x: [batch_size, seq_len, d_model]
        ↓
    自注意力: MultiHeadAttention(x, x, x)
        ↓
    残差连接: x + MultiHeadAttention(x, x, x)
        ↓
    层归一化: LayerNorm(x + MultiHeadAttention(x, x, x))
        ↓
    前馈网络: FeedForward(x')
        ↓
    残差连接: x' + FeedForward(x')
        ↓
    层归一化: LayerNorm(x' + FeedForward(x'))
        ↓
    输出 y: [batch_size, seq_len, d_model]
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # 自注意力子层
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        # 前馈网络子层
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # 层归一化层
        self.norm1 = nn.LayerNorm(d_model)  # 自注意力后的层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 前馈网络后的层归一化
        # Dropout正则化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, n_heads, seq_len, seq_len] 或 None
        
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        # 第一个子层：自注意力 + 残差连接 + 层归一化
        # 1. 计算自注意力
        attn_output = self.self_attention(x, x, x, mask)  # [batch_size, seq_len, d_model]
        
        # 2. 残差连接 + Dropout
        x = x + self.dropout(attn_output)  # [batch_size, seq_len, d_model]
        
        # 3. 层归一化
        x = self.norm1(x)  # [batch_size, seq_len, d_model]
        
        # 第二个子层：前馈网络 + 残差连接 + 层归一化
        # 1. 计算前馈网络
        ff_output = self.feed_forward(x)  # [batch_size, seq_len, d_model]
        
        # 2. 残差连接 + Dropout
        x = x + self.dropout(ff_output)  # [batch_size, seq_len, d_model]
        
        # 3. 层归一化
        x = self.norm2(x)  # [batch_size, seq_len, d_model]
        
        return x

class DecoderLayer(nn.Module):
    """
    解码器层
    
    数学公式：
    LayerNorm(x + MaskedMultiHeadAttention(x, tgt_mask))
    LayerNorm(x + MultiHeadAttention(x, encoder_output, encoder_output, src_mask))
    LayerNorm(x + FeedForward(x))
    
    网络结构：
    输入 x: [batch_size, tgt_seq_len, d_model]
        ↓
    掩码自注意力: MaskedMultiHeadAttention(x, x, x, tgt_mask)
        ↓
    残差连接: x + MaskedMultiHeadAttention(...)
        ↓
    层归一化: LayerNorm(x + MaskedMultiHeadAttention(...))
        ↓
    交叉注意力: MultiHeadAttention(x, encoder_output, encoder_output, src_mask)
        ↓
    残差连接: x + MultiHeadAttention(...)
        ↓
    层归一化: LayerNorm(x + MultiHeadAttention(...))
        ↓
    前馈网络: FeedForward(x)
        ↓
    残差连接: x + FeedForward(x)
        ↓
    层归一化: LayerNorm(x + FeedForward(x))
        ↓
    输出 y: [batch_size, tgt_seq_len, d_model]
    
    Mask说明：
    - tgt_mask: [batch_size, n_heads, tgt_seq_len, tgt_seq_len]
      * 防止解码器看到未来位置（causal mask）
      * 1表示允许注意力，0表示禁止注意力
      * 下三角矩阵，上三角为0
    - src_mask: [batch_size, n_heads, tgt_seq_len, src_seq_len]
      * 忽略编码器输入的padding位置
      * 1表示有效位置，0表示padding位置
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # 掩码自注意力子层（防止看到未来信息）
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        # 交叉注意力子层（关注编码器输出）
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        # 前馈网络子层
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # 层归一化层
        self.norm1 = nn.LayerNorm(d_model)  # 掩码自注意力后的层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 交叉注意力后的层归一化
        self.norm3 = nn.LayerNorm(d_model)  # 前馈网络后的层归一化
        # Dropout正则化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        前向传播
        
        Args:
            x: 解码器输入 [batch_size, tgt_seq_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_seq_len, d_model]
            src_mask: 源序列掩码 [batch_size, n_heads, tgt_seq_len, src_seq_len] 或 None
                    用于忽略编码器输入的padding位置
            tgt_mask: 目标序列掩码 [batch_size, n_heads, tgt_seq_len, tgt_seq_len] 或 None
                    用于防止解码器看到未来位置（causal mask）
        
        Returns:
            输出张量 [batch_size, tgt_seq_len, d_model]
        """
        # 第一个子层：掩码自注意力 + 残差连接 + 层归一化
        # 1. 计算掩码自注意力（防止看到未来信息）
        # query=key=value=x，使用tgt_mask防止看到未来位置
        attn_output = self.self_attention(x, x, x, tgt_mask)  # [batch_size, tgt_seq_len, d_model]
        
        # 2. 残差连接 + Dropout
        x = x + self.dropout(attn_output)  # [batch_size, tgt_seq_len, d_model]
        
        # 3. 层归一化
        x = self.norm1(x)  # [batch_size, tgt_seq_len, d_model]
        
        # 第二个子层：交叉注意力 + 残差连接 + 层归一化
        # 1. 计算交叉注意力（关注编码器输出）
        # query=x，key=value=encoder_output，使用src_mask忽略padding
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        # [batch_size, tgt_seq_len, d_model]
        
        # 2. 残差连接 + Dropout
        x = x + self.dropout(cross_attn_output)  # [batch_size, tgt_seq_len, d_model]
        
        # 3. 层归一化
        x = self.norm2(x)  # [batch_size, tgt_seq_len, d_model]
        
        # 第三个子层：前馈网络 + 残差连接 + 层归一化
        # 1. 计算前馈网络
        ff_output = self.feed_forward(x)  # [batch_size, tgt_seq_len, d_model]
        
        # 2. 残差连接 + Dropout
        x = x + self.dropout(ff_output)  # [batch_size, tgt_seq_len, d_model]
        
        # 3. 层归一化
        x = self.norm3(x)  # [batch_size, tgt_seq_len, d_model]
        
        return x


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


# ============================================================================
# Mask详细说明
# ============================================================================
"""
Transformer中的两种主要Mask类型：

1. Causal Mask (tgt_mask) - 目标序列掩码
   形状: [batch_size, n_heads, tgt_seq_len, tgt_seq_len]
   作用: 防止解码器在生成第i个词时看到第i+1个及以后的词
   示例: 对于序列长度4
   [[1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 1]]
   上三角为0，下三角为1

2. Padding Mask (src_mask) - 源序列掩码
   形状: [batch_size, n_heads, tgt_seq_len, src_seq_len]
   作用: 忽略编码器输入中的padding位置
   示例: 对于源序列长度5，其中位置3,4是padding
   [[1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0]]
   1表示有效位置，0表示padding位置

3. 在注意力计算中的应用：
   - 将mask为0的位置的注意力分数设为-1e9
   - 经过softmax后，这些位置的注意力权重接近0
   - 从而有效忽略被mask的位置
"""

if __name__ == "__main__":
    print("🧪 Transformer模型框架已创建")
    print("📝 请根据TODO注释逐步实现各个组件")
    print("🔧 实现完成后，可以运行训练脚本进行测试")
