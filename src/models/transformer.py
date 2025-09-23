"""
Transformer Model Implementation
Based on "Attention Is All You Need" (Vaswani et al., 2017)

This module implements the complete Transformer architecture including:
1. Positional Encoding (PositionalEncoding)
2. Multi-Head Attention Mechanism (MultiHeadAttention)
3. Feed-Forward Network (FeedForward)
4. Encoder Layer (EncoderLayer)
5. Decoder Layer (DecoderLayer)
6. Complete Transformer Model (Transformer)

All components are fully implemented with proper initialization and documentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding Layer
    
    Generates positional encodings using sine and cosine functions to provide
    position information for each position in the sequence.
    
    Mathematical formula:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        position_factor: Factor for position encoding (default: 10000)
        fast_model: Whether to use fast computation mode
    """
    
    def __init__(self, d_model, max_len=5000, position_factor=10000, fast_model=True):
        """
        Initialize positional encoding layer.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            position_factor: Factor for position encoding (default: 10000)
            fast_model: Whether to use fast computation mode
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # Create position indices [max_len, 1]
        # position = [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate div_term for 10000^(2i/d_model)
        if fast_model:
            # Fast mode: use exp(-2i*ln(10000)/d_model) to avoid numerical instability
            # div_term shape: [d_model//2]
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                               (-math.log(position_factor) / d_model))
        else:
            # Standard mode: directly calculate 10000^(2i/d_model)
            # div_term shape: [d_model//2]
            div_term = torch.float_power(position_factor, 
                                       torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
            div_term = torch.div(1, div_term)  # 取倒数
        
        # 对偶数维度应用sin函数 [max_len, d_model//2]
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # 对奇数维度应用cos函数 [max_len, d_model//2]
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 调整维度 [max_len, d_model] -> [1, max_len, d_model]
        # 这样便于后续广播到 [batch_size, seq_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 注册为buffer（不参与梯度更新，但会随模型一起保存）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass of positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Output tensor with positional encoding added [batch_size, seq_len, d_model]
        """
        # Input: x [batch_size, seq_len, d_model]
        # Positional encoding: self.pe [1, max_len, d_model]
        # Take first seq_len positions: self.pe[:, :x.size(1), :] [1, seq_len, d_model]
        # Broadcasting addition: x + self.pe[:, :x.size(1), :] [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Mechanism
    
    Implements the multi-head attention mechanism as described in the Transformer paper.
    
    Mathematical formula:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Initialize multi-head attention layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # Linear projection layers
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Scaled dot-product attention mechanism.
        
        Args:
            Q: Query matrix [batch_size, n_heads, query_seq_len, d_k]
            K: Key matrix [batch_size, n_heads, key_seq_len, d_k]
            V: Value matrix [batch_size, n_heads, value_seq_len, d_v]
            mask: Attention mask [batch_size, n_heads, query_seq_len, key_seq_len] or None
        
        Returns:
            output: Attention output [batch_size, n_heads, query_seq_len, d_v]
            attention_weights: Attention weights [batch_size, n_heads, query_seq_len, key_seq_len]
        """
        d_k = Q.size(-1)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # [batch_size, n_heads, query_seq_len, key_seq_len]
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # 应用dropout
        attention_weights = self.dropout(attention_weights)
        
        # 计算加权输出
        output = torch.matmul(attention_weights, V)
        # [batch_size, n_heads, query_seq_len, d_v]
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query matrix [batch_size, query_seq_len, d_model]
            key: Key matrix [batch_size, key_seq_len, d_model]
            value: Value matrix [batch_size, value_seq_len, d_model]
            mask: Attention mask [batch_size, n_heads, query_seq_len, key_seq_len] or None
        
        Returns:
            Output tensor [batch_size, query_seq_len, d_model]
        """
        batch_size, query_seq_len, d_model = query.size()
        key_seq_len = key.size(1)
        value_seq_len = value.size(1)
        
        # 1. 线性变换
        Q = self.W_Q(query)  # [batch_size, query_seq_len, d_model]
        K = self.W_K(key)    # [batch_size, key_seq_len, d_model]
        V = self.W_V(value)  # [batch_size, value_seq_len, d_model]
        
        # 2. 重塑为多头
        Q = Q.view(batch_size, query_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # [batch_size, query_seq_len, n_heads, d_k] -> [batch_size, n_heads, query_seq_len, d_k]
        K = K.view(batch_size, key_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # [batch_size, key_seq_len, n_heads, d_k] -> [batch_size, n_heads, key_seq_len, d_k]
        V = V.view(batch_size, value_seq_len, self.n_heads, self.d_v).transpose(1, 2)
        # [batch_size, value_seq_len, n_heads, d_v] -> [batch_size, n_heads, value_seq_len, d_v]
        
        # 3. 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )
        # attention_output: [batch_size, n_heads, query_seq_len, d_v]
        
        # 4. 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, query_seq_len, d_model
        )
        # [batch_size, n_heads, query_seq_len, d_v] -> [batch_size, query_seq_len, d_model]
        
        # 5. 输出投影
        output = self.W_O(attention_output)
        # [batch_size, query_seq_len, d_model]
        
        return output
        

class FeedForward(nn.Module):
    """
    Feed-Forward Network
    
    Implements the position-wise feed-forward network as described in the Transformer paper.
    
    Mathematical formula: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    
    Network structure:
    Input x: [batch_size, seq_len, d_model]
        ↓
    Linear layer 1: [batch_size, seq_len, d_model] → [batch_size, seq_len, d_ff]
        ↓
    ReLU activation: max(0, x)
        ↓
    Dropout: Randomly zero out some neurons
        ↓
    Linear layer 2: [batch_size, seq_len, d_ff] → [batch_size, seq_len, d_model]
        ↓
    Output y: [batch_size, seq_len, d_model]
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        dropout: Dropout rate
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super(FeedForward, self).__init__()
        # First linear layer: d_model → d_ff
        self.W_1 = nn.Linear(d_model, d_ff)
        # Second linear layer: d_ff → d_model
        self.W_2 = nn.Linear(d_ff, d_model)
        # Dropout regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
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
    Encoder Layer
    
    Implements a single encoder layer with self-attention and feed-forward network.
    
    Mathematical formula:
    LayerNorm(x + MultiHeadAttention(x))
    LayerNorm(x + FeedForward(x))
    
    Network structure:
    Input x: [batch_size, seq_len, d_model]
        ↓
    Self-attention: MultiHeadAttention(x, x, x)
        ↓
    Residual connection: x + MultiHeadAttention(x, x, x)
        ↓
    Layer normalization: LayerNorm(x + MultiHeadAttention(x, x, x))
        ↓
    Feed-forward network: FeedForward(x')
        ↓
    Residual connection: x' + FeedForward(x')
        ↓
    Layer normalization: LayerNorm(x' + FeedForward(x'))
        ↓
    Output y: [batch_size, seq_len, d_model]
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Initialize encoder layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super(EncoderLayer, self).__init__()
        # Self-attention sublayer
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        # Feed-forward sublayer
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)  # Layer norm after self-attention
        self.norm2 = nn.LayerNorm(d_model)  # Layer norm after feed-forward
        # Dropout regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass of encoder layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, n_heads, seq_len, seq_len] or None
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
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
    Decoder Layer
    
    Implements a single decoder layer with masked self-attention, cross-attention, and feed-forward network.
    
    Mathematical formula:
    LayerNorm(x + MaskedMultiHeadAttention(x, tgt_mask))
    LayerNorm(x + MultiHeadAttention(x, encoder_output, encoder_output, src_mask))
    LayerNorm(x + FeedForward(x))
    
    Network structure:
    Input x: [batch_size, tgt_seq_len, d_model]
        ↓
    Masked self-attention: MaskedMultiHeadAttention(x, x, x, tgt_mask)
        ↓
    Residual connection: x + MaskedMultiHeadAttention(...)
        ↓
    Layer normalization: LayerNorm(x + MaskedMultiHeadAttention(...))
        ↓
    Cross-attention: MultiHeadAttention(x, encoder_output, encoder_output, src_mask)
        ↓
    Residual connection: x + MultiHeadAttention(...)
        ↓
    Layer normalization: LayerNorm(x + MultiHeadAttention(...))
        ↓
    Feed-forward network: FeedForward(x)
        ↓
    Residual connection: x + FeedForward(x)
        ↓
    Layer normalization: LayerNorm(x + FeedForward(x))
        ↓
    Output y: [batch_size, tgt_seq_len, d_model]
    
    Mask explanation:
    - tgt_mask: [batch_size, n_heads, tgt_seq_len, tgt_seq_len]
      * 防止解码器看到未来位置（causal mask）
      * 1表示允许注意力，0表示禁止注意力
      * 下三角矩阵，上三角为0
    - src_mask: [batch_size, n_heads, tgt_seq_len, src_seq_len]
      * 忽略编码器输入的padding位置
      * 1表示有效位置，0表示padding位置
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Initialize decoder layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super(DecoderLayer, self).__init__()
        # Masked self-attention sublayer (prevents seeing future information)
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        # Cross-attention sublayer (attends to encoder output)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        # Feed-forward sublayer
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)  # Layer norm after masked self-attention
        self.norm2 = nn.LayerNorm(d_model)  # Layer norm after cross-attention
        self.norm3 = nn.LayerNorm(d_model)  # Layer norm after feed-forward
        # Dropout regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Forward pass of decoder layer.
        
        Args:
            x: Decoder input [batch_size, tgt_seq_len, d_model]
            encoder_output: Encoder output [batch_size, src_seq_len, d_model]
            src_mask: Source sequence mask [batch_size, n_heads, tgt_seq_len, src_seq_len] or None
                     Used to ignore padding positions in encoder input
            tgt_mask: Target sequence mask [batch_size, n_heads, tgt_seq_len, tgt_seq_len] or None
                     Used to prevent decoder from seeing future positions (causal mask)
        
        Returns:
            Output tensor [batch_size, tgt_seq_len, d_model]
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
    Complete Transformer Model
    
    Implements the full Transformer architecture for sequence-to-sequence tasks.
    
    Network structure:
    Input sequence → Word embedding → Positional encoding → Encoder stack → Encoder output
    Target sequence → Word embedding → Positional encoding → Decoder stack (using encoder output) → Output projection → Probability distribution
    
    Mathematical formula:
    - Word embedding: Embedding(x) * sqrt(d_model)
    - Positional encoding: x + PositionalEncoding(x)
    - Encoder: LayerNorm(x + MultiHeadAttention(x)) + LayerNorm(x + FeedForward(x))
    - Decoder: Three sublayers, each with residual connection and layer normalization
    - Output: Linear(d_model → tgt_vocab_size)
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension
        n_heads: Number of attention heads
        n_encoder_layers: Number of encoder layers
        n_decoder_layers: Number of decoder layers
        d_ff: Feed-forward dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_encoder_layers=6, n_decoder_layers=6, d_ff=2048, 
                 max_len=5000, dropout=0.1):
        """
        Initialize Transformer model.
        
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_encoder_layers: Number of encoder layers
            n_decoder_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(Transformer, self).__init__()
        
        # Save parameters
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码层
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # 编码器堆叠
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_encoder_layers)
        ])
        
        # 解码器堆叠
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_decoder_layers)
        ])
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        self.init_parameters()
    
    def init_parameters(self):
        """Initialize model parameters with Xavier initialization."""
        # Word embedding layers use Xavier initialization
        nn.init.xavier_uniform_(self.src_embedding.weight)
        nn.init.xavier_uniform_(self.tgt_embedding.weight)
        
        # Output projection layer uses Xavier initialization
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def create_padding_mask(self, seq, pad_idx=0):
        """
        Create padding mask for attention mechanism.
        
        Args:
            seq: Input sequence [batch_size, seq_len]
            pad_idx: Index of padding token
        
        Returns:
            mask: [batch_size, 1, 1, seq_len] - For multi-head attention
        """
        # Create padding mask: [batch_size, seq_len]
        mask = (seq != pad_idx)
        
        # Expand dimensions for multi-head attention: [batch_size, 1, 1, seq_len]
        mask = mask.unsqueeze(1).unsqueeze(2)
        
        return mask
    
    def create_look_ahead_mask(self, size):
        """
        Create look-ahead mask (causal mask) for decoder.
        
        Args:
            size: Sequence length
        
        Returns:
            mask: [size, size] - Lower triangular matrix
        """
        # Create upper triangular matrix, then invert to get lower triangular matrix
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0  # Lower triangle is True, upper triangle is False
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass of Transformer model.
        
        Args:
            src: Source sequence [batch_size, src_seq_len]
            tgt: Target sequence [batch_size, tgt_seq_len]
            src_mask: Source sequence mask [batch_size, 1, 1, src_seq_len] or None
            tgt_mask: Target sequence mask [batch_size, 1, 1, tgt_seq_len] or None
        
        Returns:
            Output probability distribution [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # 1. 创建mask（如果未提供）
        if src_mask is None:
            src_mask = self.create_padding_mask(src, 0)
        if tgt_mask is None:
            tgt_mask = self.create_look_ahead_mask(tgt.size(1))
            # 扩展维度以适配多头注意力
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_seq_len, tgt_seq_len]
        
        # 2. 词嵌入 + 缩放
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        # [batch_size, src_seq_len, d_model], [batch_size, tgt_seq_len, d_model]
        
        # 3. 位置编码
        # PositionalEncoding现在直接接受[batch_size, seq_len, d_model]格式
        src_encoded = self.positional_encoding(src_embedded)
        tgt_encoded = self.positional_encoding(tgt_embedded)
        # [batch_size, src_seq_len, d_model], [batch_size, tgt_seq_len, d_model]
        
        # 4. 编码器堆叠
        encoder_output = src_encoded
        for encoder_layer in self.encoder:
            encoder_output = encoder_layer(encoder_output, src_mask)
        # [batch_size, src_seq_len, d_model]
        
        # 5. 解码器堆叠
        decoder_output = tgt_encoded
        for decoder_layer in self.decoder:
            decoder_output = decoder_layer(
                decoder_output, encoder_output, src_mask, tgt_mask
            )
        # [batch_size, tgt_seq_len, d_model]
        
        # 6. 输出投影
        output = self.output_projection(decoder_output)
        # [batch_size, tgt_seq_len, tgt_vocab_size]
        
        return output   



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
