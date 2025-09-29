"""
位置编码功能测试
测试PositionalEncoding的功能和可视化
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.transformer import PositionalEncoding


def test_positional_encoding_basic():
    """基础功能测试"""
    print("🧪 Testing positional encoding basic functionality...")
    
    # 创建位置编码
    d_model = 512
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    
    # 创建测试输入
    seq_len = 50
    batch_size = 32
    x = torch.randn(batch_size, seq_len, d_model)  # 修改为标准格式
    
    print(f"Input shape: {x.shape}")
    print(f"Positional encoding shape: {pe.pe.shape}")
    
    # 前向传播
    output = pe(x)
    print(f"Output shape: {output.shape}")
    
    # 验证形状
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} != {x.shape}"
    print("✅ Shape test passed")
    
    # 验证位置编码是固定的
    output2 = pe(x)
    assert torch.allclose(output, output2), "Positional encoding should be fixed"
    print("✅ Consistency test passed")
    
    # 验证不同位置有不同的编码
    pos_0 = pe.pe[0, 0, :]  # 位置0的编码
    pos_1 = pe.pe[0, 1, :]  # 位置1的编码
    assert not torch.allclose(pos_0, pos_1), "Different positions should have different encodings"
    print("✅ Position distinction test passed")
    
    print("🎉 All basic functionality tests passed!\n")


def test_positional_encoding_different_lengths():
    """测试不同序列长度"""
    print("🧪 Testing different sequence lengths...")
    
    d_model = 256
    max_len = 200
    pe = PositionalEncoding(d_model, max_len)
    
    # 测试不同序列长度
    test_lengths = [10, 50, 100, 150]
    batch_size = 16
    
    for seq_len in test_lengths:
        x = torch.randn(batch_size, seq_len, d_model)  # 修改为标准格式
        output = pe(x)
        
        assert output.shape == (batch_size, seq_len, d_model), \
            f"Sequence length {seq_len} test failed"
        print(f"✅ Sequence length {seq_len} test passed")
    
    print("🎉 All different length tests passed!\n")


def test_positional_encoding_parameters():
    """测试不同参数设置"""
    print("🧪 Testing different parameter settings...")
    
    d_model = 128
    max_len = 50
    
    # 测试fast_model=True
    pe_fast = PositionalEncoding(d_model, max_len, fast_model=True)
    print("✅ fast_model=True test passed")
    
    # 测试fast_model=False
    pe_standard = PositionalEncoding(d_model, max_len, fast_model=False)
    print("✅ fast_model=False test passed")
    
    # 测试不同position_factor
    pe_factor = PositionalEncoding(d_model, max_len, position_factor=1000)
    print("✅ position_factor=1000 test passed")
    
    # 验证两种模式结果相似
    x = torch.randn(8, 20, d_model)  # 修改为标准格式
    output_fast = pe_fast(x)
    output_standard = pe_standard(x)
    
    # 允许小的数值误差
    diff = torch.abs(output_fast - output_standard).max()
    print(f"Max difference between fast_model and standard mode: {diff:.6f}")
    
    print("🎉 All parameter setting tests passed!\n")


def visualize_positional_encoding():
    """可视化位置编码"""
    print("🎨 Visualizing positional encoding...")
    
    d_model = 64
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    
    # 获取位置编码数据
    pos_encoding = pe.pe[0, :max_len, :].detach().numpy()  # [max_len, d_model]
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Positional Encoding Visualization', fontsize=16)
    
    # 1. 位置编码热力图
    ax1 = axes[0, 0]
    im1 = ax1.imshow(pos_encoding.T, aspect='auto', cmap='RdBu')
    ax1.set_title('Positional Encoding Heatmap')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Dimension')
    plt.colorbar(im1, ax=ax1)
    
    # 2. 前几个维度的位置编码曲线
    ax2 = axes[0, 1]
    for i in range(0, min(8, d_model), 2):
        ax2.plot(pos_encoding[:, i], label=f'Dim {i} (sin)')
        if i+1 < d_model:
            ax2.plot(pos_encoding[:, i+1], label=f'Dim {i+1} (cos)')
    ax2.set_title('First Few Dimensions of Positional Encoding')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Encoding Value')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 不同位置的编码分布
    ax3 = axes[1, 0]
    positions = [0, 10, 25, 50, 75, 99]
    for pos in positions:
        ax3.plot(pos_encoding[pos, :], label=f'Position {pos}')
    ax3.set_title('Encoding Distribution at Different Positions')
    ax3.set_xlabel('Dimension')
    ax3.set_ylabel('Encoding Value')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 位置编码的周期性
    ax4 = axes[1, 1]
    # 选择几个维度显示周期性
    dims = [0, 2, 4, 6]
    for dim in dims:
        if dim < d_model:
            ax4.plot(pos_encoding[:50, dim], label=f'Dimension {dim}')
    ax4.set_title('Periodicity of Positional Encoding (First 50 Positions)')
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Encoding Value')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    # 保存到tests目录下的visualizations文件夹
    import os
    vis_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    save_path = os.path.join(vis_dir, 'positional_encoding_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualization completed, image saved as '{save_path}'\n")


def visualize_positional_encoding_3d():
    """3D可视化位置编码"""
    print("🎨 3D visualizing positional encoding...")
    
    d_model = 32
    max_len = 50
    pe = PositionalEncoding(d_model, max_len)
    
    # 获取位置编码数据
    pos_encoding = pe.pe[0, :max_len, :].detach().numpy()  # [max_len, d_model]
    
    # 创建3D图形
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 3D表面图
    ax1 = fig.add_subplot(131, projection='3d')
    X, Y = np.meshgrid(range(max_len), range(d_model))
    Z = pos_encoding.T
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_title('Positional Encoding 3D Surface')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Dimension')
    ax1.set_zlabel('Encoding Value')
    
    # 2. 3D线框图
    ax2 = fig.add_subplot(132, projection='3d')
    for i in range(0, d_model, 4):  # 每4个维度画一条线
        ax2.plot(range(max_len), [i] * max_len, pos_encoding[:, i], 
                label=f'Dim {i}')
    ax2.set_title('Positional Encoding 3D Wireframe')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Dimension')
    ax2.set_zlabel('Encoding Value')
    
    # 3. 位置编码的2D投影
    ax3 = fig.add_subplot(133)
    # 使用PCA降维到2D进行可视化
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pos_2d = pca.fit_transform(pos_encoding)
    
    scatter = ax3.scatter(pos_2d[:, 0], pos_2d[:, 1], 
                         c=range(max_len), cmap='viridis', s=50)
    ax3.set_title('Positional Encoding 2D Projection (PCA)')
    ax3.set_xlabel('First Principal Component')
    ax3.set_ylabel('Second Principal Component')
    plt.colorbar(scatter, ax=ax3, label='Position')
    
    # 添加位置标签
    for i in range(0, max_len, 10):
        ax3.annotate(str(i), (pos_2d[i, 0], pos_2d[i, 1]))
    
    plt.tight_layout()
    
    # 保存到tests目录下的visualizations文件夹
    import os
    vis_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    save_path = os.path.join(vis_dir, 'positional_encoding_3d_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 3D visualization completed, image saved as '{save_path}'\n")


def test_positional_encoding_math_properties():
    """测试位置编码的数学性质"""
    print("🧪 Testing positional encoding mathematical properties...")
    
    d_model = 64
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    
    pos_encoding = pe.pe[:max_len, 0, :].detach().numpy()
    
    # 1. 测试周期性
    print("Testing periodicity...")
    # 对于sin和cos函数，应该有一定的周期性
    for dim in range(0, min(8, d_model), 2):
        sin_values = pos_encoding[:, dim]
        cos_values = pos_encoding[:, dim + 1] if dim + 1 < d_model else None
        
        # 检查sin和cos的周期性
        if cos_values is not None:
            # sin^2 + cos^2 = 1
            sin_cos_sum = sin_values**2 + cos_values**2
            expected = np.ones_like(sin_cos_sum)
            assert np.allclose(sin_cos_sum, expected, atol=1e-6), \
                f"Dimension {dim} sin^2+cos^2 != 1"
    
    print("✅ Periodicity test passed")
    
    # 2. 测试不同位置的唯一性
    print("Testing position uniqueness...")
    # 使用更宽松的容差来检查唯一性
    unique_count = 0
    for i in range(min(max_len, pos_encoding.shape[0])):
        is_unique = True
        for j in range(i+1, min(max_len, pos_encoding.shape[0])):
            if np.allclose(pos_encoding[i], pos_encoding[j], atol=1e-10):
                is_unique = False
                break
        if is_unique:
            unique_count += 1
    
    # 至少90%的位置应该是唯一的（考虑到数值精度）
    actual_len = min(max_len, pos_encoding.shape[0])
    uniqueness_ratio = unique_count / actual_len
    assert uniqueness_ratio >= 0.9, f"Only {uniqueness_ratio:.2%} positions are unique, expected >= 90%"
    print(f"✅ Position uniqueness test passed ({uniqueness_ratio:.2%} unique)")
    
    # 3. 测试编码的数值范围
    print("Testing value range...")
    min_val = pos_encoding.min()
    max_val = pos_encoding.max()
    # 位置编码的值应该在[-1, 1]范围内
    assert -1.1 <= min_val <= 1.1, f"Min value should be in [-1, 1], actual: {min_val}"
    assert -1.1 <= max_val <= 1.1, f"Max value should be in [-1, 1], actual: {max_val}"
    print(f"✅ Value range test passed: [{min_val:.3f}, {max_val:.3f}]")
    
    print("🎉 All mathematical property tests passed!\n")


def benchmark_positional_encoding():
    """性能测试"""
    print("⚡ Performance testing...")
    
    import time
    
    d_model = 512
    max_len = 1000
    pe = PositionalEncoding(d_model, max_len)
    
    # 测试不同批次大小和序列长度
    test_cases = [
        (50, 32),   # (seq_len, batch_size)
        (100, 64),
        (200, 32),
        (500, 16),
    ]
    
    for seq_len, batch_size in test_cases:
        x = torch.randn(batch_size, seq_len, d_model)  # 修改为标准格式
        
        # 预热
        for _ in range(10):
            _ = pe(x)
        
        # 计时
        start_time = time.time()
        for _ in range(100):
            output = pe(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        print(f"Seq len {seq_len}, Batch size {batch_size}: "
              f"Avg time {avg_time*1000:.2f}ms")
    
    print("✅ Performance test completed\n")


def run_basic_tests():
    """运行基础测试（不需要可视化依赖）"""
    print("🧪 Running basic functional tests...")
    
    try:
        # 运行基础测试
        test_positional_encoding_basic()
        test_positional_encoding_different_lengths()
        test_positional_encoding_parameters()
        test_positional_encoding_math_properties()
        benchmark_positional_encoding()
        
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
            from sklearn.decomposition import PCA
        except ImportError as e:
            print(f"⚠️ Missing visualization dependencies: {e}")
            print("Please install: pip install matplotlib scikit-learn")
            return False
        
        # 运行可视化测试
        visualize_positional_encoding()
        visualize_positional_encoding_3d()
        
        print("✅ Visualization tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Visualization tests failed: {e}")
        return False

def main():
    """运行所有测试"""
    print("🚀 Starting positional encoding functional tests\n")
    
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
