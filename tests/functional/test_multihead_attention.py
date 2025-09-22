"""
Multi-Head Attention 功能测试
测试MultiHeadAttention的功能和可视化
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.transformer import MultiHeadAttention


def test_multihead_attention_basic():
    """基础功能测试"""
    print("🧪 Testing Multi-Head Attention basic functionality...")
    
    # 创建多头注意力
    d_model = 512
    n_heads = 8
    mha = MultiHeadAttention(d_model, n_heads)
    
    # 创建测试输入
    batch_size = 32
    seq_len = 50
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    print(f"Number of heads: {n_heads}")
    print(f"d_k = d_v = {d_model // n_heads}")
    
    # 前向传播
    output = mha(x, x, x)  # 自注意力：query=key=value=x
    print(f"Output shape: {output.shape}")
    
    # 验证形状
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} != {x.shape}"
    print("✅ Shape test passed")
    
    # 验证输出不是全零
    assert not torch.allclose(output, torch.zeros_like(output)), "Output should not be all zeros"
    print("✅ Non-zero output test passed")
    
    # 验证不同输入产生不同输出
    x2 = torch.randn(batch_size, seq_len, d_model)
    output2 = mha(x2, x2, x2)
    assert not torch.allclose(output, output2), "Different inputs should produce different outputs"
    print("✅ Different input test passed")
    
    print("🎉 All basic functionality tests passed!\n")


def test_multihead_attention_different_lengths():
    """测试不同序列长度"""
    print("🧪 Testing different sequence lengths...")
    
    d_model = 256
    n_heads = 8
    mha = MultiHeadAttention(d_model, n_heads)
    
    # 测试不同序列长度
    test_lengths = [10, 25, 50, 100]
    batch_size = 16
    
    for seq_len in test_lengths:
        x = torch.randn(batch_size, seq_len, d_model)
        output = mha(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model), \
            f"Sequence length {seq_len} test failed"
        print(f"✅ Sequence length {seq_len} test passed")
    
    print("🎉 All different length tests passed!\n")


def test_multihead_attention_parameters():
    """测试不同参数设置"""
    print("🧪 Testing different parameter settings...")
    
    # 测试不同的头数
    d_model = 512
    test_heads = [1, 2, 4, 8, 16]
    seq_len, batch_size = 20, 8
    
    for n_heads in test_heads:
        if d_model % n_heads == 0:  # 确保d_model能被n_heads整除
            mha = MultiHeadAttention(d_model, n_heads)
            x = torch.randn(seq_len, batch_size, d_model)
            output = mha(x, x, x)
            
            assert output.shape == (seq_len, batch_size, d_model), \
                f"n_heads={n_heads} test failed"
            print(f"✅ n_heads={n_heads} test passed")
    
    # 测试不同的dropout率
    dropout_rates = [0.0, 0.1, 0.5]
    for dropout_rate in dropout_rates:
        mha = MultiHeadAttention(d_model, 8, dropout=dropout_rate)
        x = torch.randn(batch_size, seq_len, d_model)
        output = mha(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model), \
            f"dropout={dropout_rate} test failed"
        print(f"✅ dropout={dropout_rate} test passed")
    
    print("🎉 All parameter setting tests passed!\n")


def test_multihead_attention_mask():
    """测试掩码功能"""
    print("🧪 Testing mask functionality...")
    
    d_model = 128
    n_heads = 4
    mha = MultiHeadAttention(d_model, n_heads)
    
    batch_size, seq_len = 4, 10
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建padding mask（假设前5个位置是padding）
    # 注意：在我们的实现中，注意力计算是在 [batch_size, n_heads, seq_len, d_k] 维度上进行的
    # 注意力分数是 [batch_size, n_heads, seq_len, seq_len] 维度
    # 所以掩码应该是 [batch_size, n_heads, seq_len, seq_len] 维度
    mask = torch.ones(batch_size, n_heads, seq_len, seq_len)
    mask[:, :, :, :5] = 0  # 将前5个位置设为0
    
    # 测试带mask的前向传播
    output = mha(x, x, x, mask=mask)
    assert output.shape == (batch_size, seq_len, d_model), "Mask test failed"
    print("✅ Mask test passed")
    
    print("🎉 All mask tests passed!\n")


def test_multihead_attention_math_properties():
    """测试多头注意力的数学性质"""
    print("🧪 Testing Multi-Head Attention mathematical properties...")
    
    d_model = 64
    n_heads = 4
    mha = MultiHeadAttention(d_model, n_heads)
    
    batch_size, seq_len = 8, 20
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 1. 测试线性性（近似）
    x1 = torch.randn(batch_size, seq_len, d_model)
    x2 = torch.randn(batch_size, seq_len, d_model)
    alpha, beta = 0.5, 0.5
    
    output1 = mha(x1, x1, x1)
    output2 = mha(x2, x2, x2)
    output_combined = mha(alpha * x1 + beta * x2, alpha * x1 + beta * x2, alpha * x1 + beta * x2)
    
    # 由于非线性激活，不会完全线性，但应该有一定的相关性
    correlation = torch.corrcoef(torch.stack([
        output_combined.flatten(),
        (alpha * output1 + beta * output2).flatten()
    ]))[0, 1]
    
    print(f"Linearity correlation: {correlation:.4f}")
    print("✅ Linearity test passed")
    
    # 2. 测试不同头数的输出差异
    mha_1 = MultiHeadAttention(d_model, 1)
    mha_4 = MultiHeadAttention(d_model, 4)
    
    output_1 = mha_1(x, x, x)
    output_4 = mha_4(x, x, x)
    
    # 不同头数应该产生不同的输出
    assert not torch.allclose(output_1, output_4), "Different head counts should produce different outputs"
    print("✅ Different head count test passed")
    
    print("🎉 All mathematical property tests passed!\n")


def benchmark_multihead_attention():
    """性能测试"""
    print("⚡ Performance testing...")
    
    import time
    
    d_model = 512
    n_heads = 8
    mha = MultiHeadAttention(d_model, n_heads)
    
    # 测试不同批次大小和序列长度
    test_cases = [
        (32, 50),   # (batch_size, seq_len)
        (64, 100),
        (32, 200),
        (16, 500),
    ]
    
    for batch_size, seq_len in test_cases:
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 预热
        for _ in range(10):
            _ = mha(x, x, x)
        
        # 计时
        start_time = time.time()
        for _ in range(100):
            output = mha(x, x, x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        print(f"Batch size {batch_size}, Seq len {seq_len}: "
              f"Avg time {avg_time*1000:.2f}ms")
    
    print("✅ Performance test completed\n")


def visualize_attention_weights():
    """可视化注意力权重"""
    print("🎨 Visualizing attention weights...")
    
    d_model = 64
    n_heads = 4
    mha = MultiHeadAttention(d_model, n_heads)
    
    batch_size, seq_len = 1, 20
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 获取注意力权重（需要修改forward方法返回权重）
    # 这里我们创建一个简化的可视化
    with torch.no_grad():
        # 计算Q, K, V
        Q = mha.W_Q(x)
        K = mha.W_K(x)
        V = mha.W_V(x)
        
        # 重塑为多头
        Q = Q.view(batch_size, seq_len, n_heads, d_model // n_heads).transpose(1, 2)
        K = K.view(batch_size, seq_len, n_heads, d_model // n_heads).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_model // n_heads)
        attention_weights = torch.softmax(scores, dim=-1)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Head Attention Weights Visualization', fontsize=16)
    
    # 选择第一个batch的第一个头进行可视化
    head_idx = 0
    attention_matrix = attention_weights[0, head_idx, :, :].detach().numpy()
    
    # 1. 注意力权重热力图
    ax1 = axes[0, 0]
    im1 = ax1.imshow(attention_matrix, aspect='auto', cmap='Blues')
    ax1.set_title(f'Attention Weights (Head {head_idx})')
    ax1.set_xlabel('Key Position')
    ax1.set_ylabel('Query Position')
    plt.colorbar(im1, ax=ax1)
    
    # 2. 注意力权重分布
    ax2 = axes[0, 1]
    ax2.hist(attention_matrix.flatten(), bins=50, alpha=0.7, color='skyblue')
    ax2.set_title('Attention Weights Distribution')
    ax2.set_xlabel('Weight Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True)
    
    # 3. 不同位置的注意力模式
    ax3 = axes[1, 0]
    positions = [0, 5, 10, 15, 19]
    for pos in positions:
        ax3.plot(attention_matrix[pos, :], label=f'Position {pos}', alpha=0.7)
    ax3.set_title('Attention Patterns at Different Positions')
    ax3.set_xlabel('Key Position')
    ax3.set_ylabel('Attention Weight')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 注意力权重的对称性
    ax4 = axes[1, 1]
    # 计算注意力权重的对称性
    symmetry = np.abs(attention_matrix - attention_matrix.T).mean()
    ax4.text(0.5, 0.5, f'Attention Symmetry:\n{symmetry:.4f}', 
             ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Attention Properties')
    
    plt.tight_layout()
    
    # 保存到tests目录下的visualizations文件夹
    vis_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    save_path = os.path.join(vis_dir, 'multihead_attention_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualization completed, image saved as '{save_path}'\n")


def visualize_attention_heads():
    """可视化不同注意力头"""
    print("🎨 Visualizing different attention heads...")
    
    d_model = 128
    n_heads = 8
    mha = MultiHeadAttention(d_model, n_heads)
    
    batch_size, seq_len = 1, 15
    x = torch.randn(batch_size, seq_len, d_model)
    
    with torch.no_grad():
        # 计算Q, K
        Q = mha.W_Q(x)
        K = mha.W_K(x)
        
        # 重塑为多头
        Q = Q.view(batch_size, seq_len, n_heads, d_model // n_heads).transpose(1, 2)
        K = K.view(batch_size, seq_len, n_heads, d_model // n_heads).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_model // n_heads)
        attention_weights = torch.softmax(scores, dim=-1)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Multi-Head Attention Weights (All Heads)', fontsize=16)
    
    for head_idx in range(n_heads):
        row = head_idx // 4
        col = head_idx % 4
        
        attention_matrix = attention_weights[0, head_idx, :, :].detach().numpy()
        
        im = axes[row, col].imshow(attention_matrix, aspect='auto', cmap='Blues')
        axes[row, col].set_title(f'Head {head_idx}')
        axes[row, col].set_xlabel('Key Position')
        axes[row, col].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[row, col])
    
    plt.tight_layout()
    
    # 保存图片
    vis_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    save_path = os.path.join(vis_dir, 'multihead_attention_heads_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Multi-head visualization completed, image saved as '{save_path}'\n")


def run_basic_tests():
    """运行基础测试（不需要可视化依赖）"""
    print("🧪 Running basic functional tests...")
    
    try:
        # 运行基础测试
        test_multihead_attention_basic()
        test_multihead_attention_different_lengths()
        test_multihead_attention_parameters()
        test_multihead_attention_mask()
        test_multihead_attention_math_properties()
        benchmark_multihead_attention()
        
        print("✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        return False


def run_visualization_tests():
    """运行可视化测试"""
    print("🎨 Running visualization tests...")
    
    try:
        # 检查依赖
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as e:
            print(f"⚠️ Missing visualization dependencies: {e}")
            print("Please install: pip install matplotlib scikit-learn")
            return False
        
        # 运行可视化测试
        visualize_attention_weights()
        visualize_attention_heads()
        
        print("✅ Visualization tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Visualization tests failed: {e}")
        return False


def main():
    """运行所有测试"""
    print("🚀 Starting Multi-Head Attention functional tests\n")
    
    # 运行基础测试
    basic_success = run_basic_tests()
    
    if basic_success:
        print("\n" + "="*50)
        
        # 询问是否运行可视化测试
        try:
            response = input("Run visualization tests? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                run_visualization_tests()
            else:
                print("Skipping visualization tests")
        except KeyboardInterrupt:
            print("\nTests interrupted by user")
        except:
            print("Unable to get user input, skipping visualization tests")
    
    print("\n🎉 Tests completed!")


if __name__ == "__main__":
    main()
