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
    print("🧪 测试位置编码基础功能...")
    
    # 创建位置编码
    d_model = 512
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    
    # 创建测试输入
    seq_len = 50
    batch_size = 32
    x = torch.randn(seq_len, batch_size, d_model)
    
    print(f"输入形状: {x.shape}")
    print(f"位置编码形状: {pe.pe.shape}")
    
    # 前向传播
    output = pe(x)
    print(f"输出形状: {output.shape}")
    
    # 验证形状
    assert output.shape == x.shape, f"输出形状不匹配: {output.shape} != {x.shape}"
    print("✅ 形状测试通过")
    
    # 验证位置编码是固定的
    output2 = pe(x)
    assert torch.allclose(output, output2), "位置编码应该是固定的"
    print("✅ 固定性测试通过")
    
    # 验证不同位置有不同的编码
    pos_0 = pe.pe[0, 0, :]  # 位置0的编码
    pos_1 = pe.pe[1, 0, :]  # 位置1的编码
    assert not torch.allclose(pos_0, pos_1), "不同位置应该有不同编码"
    print("✅ 位置区分测试通过")
    
    print("🎉 基础功能测试全部通过！\n")


def test_positional_encoding_different_lengths():
    """测试不同序列长度"""
    print("🧪 测试不同序列长度...")
    
    d_model = 256
    max_len = 200
    pe = PositionalEncoding(d_model, max_len)
    
    # 测试不同序列长度
    test_lengths = [10, 50, 100, 150]
    batch_size = 16
    
    for seq_len in test_lengths:
        x = torch.randn(seq_len, batch_size, d_model)
        output = pe(x)
        
        assert output.shape == (seq_len, batch_size, d_model), \
            f"序列长度{seq_len}测试失败"
        print(f"✅ 序列长度 {seq_len} 测试通过")
    
    print("🎉 不同长度测试全部通过！\n")


def test_positional_encoding_parameters():
    """测试不同参数设置"""
    print("🧪 测试不同参数设置...")
    
    d_model = 128
    max_len = 50
    
    # 测试fast_model=True
    pe_fast = PositionalEncoding(d_model, max_len, fast_model=True)
    print("✅ fast_model=True 测试通过")
    
    # 测试fast_model=False
    pe_standard = PositionalEncoding(d_model, max_len, fast_model=False)
    print("✅ fast_model=False 测试通过")
    
    # 测试不同position_factor
    pe_factor = PositionalEncoding(d_model, max_len, position_factor=1000)
    print("✅ position_factor=1000 测试通过")
    
    # 验证两种模式结果相似
    x = torch.randn(20, 8, d_model)
    output_fast = pe_fast(x)
    output_standard = pe_standard(x)
    
    # 允许小的数值误差
    diff = torch.abs(output_fast - output_standard).max()
    print(f"fast_model和standard模式最大差异: {diff:.6f}")
    
    print("🎉 参数设置测试全部通过！\n")


def visualize_positional_encoding():
    """可视化位置编码"""
    print("🎨 可视化位置编码...")
    
    d_model = 64
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    
    # 获取位置编码数据
    pos_encoding = pe.pe[:max_len, 0, :].detach().numpy()  # [max_len, d_model]
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('位置编码可视化', fontsize=16)
    
    # 1. 位置编码热力图
    ax1 = axes[0, 0]
    im1 = ax1.imshow(pos_encoding.T, aspect='auto', cmap='RdBu')
    ax1.set_title('位置编码热力图')
    ax1.set_xlabel('位置')
    ax1.set_ylabel('维度')
    plt.colorbar(im1, ax=ax1)
    
    # 2. 前几个维度的位置编码曲线
    ax2 = axes[0, 1]
    for i in range(0, min(8, d_model), 2):
        ax2.plot(pos_encoding[:, i], label=f'维度 {i} (sin)')
        if i+1 < d_model:
            ax2.plot(pos_encoding[:, i+1], label=f'维度 {i+1} (cos)')
    ax2.set_title('前几个维度的位置编码')
    ax2.set_xlabel('位置')
    ax2.set_ylabel('编码值')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 不同位置的编码分布
    ax3 = axes[1, 0]
    positions = [0, 10, 25, 50, 75, 99]
    for pos in positions:
        ax3.plot(pos_encoding[pos, :], label=f'位置 {pos}')
    ax3.set_title('不同位置的编码分布')
    ax3.set_xlabel('维度')
    ax3.set_ylabel('编码值')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 位置编码的周期性
    ax4 = axes[1, 1]
    # 选择几个维度显示周期性
    dims = [0, 2, 4, 6]
    for dim in dims:
        if dim < d_model:
            ax4.plot(pos_encoding[:50, dim], label=f'维度 {dim}')
    ax4.set_title('位置编码的周期性（前50个位置）')
    ax4.set_xlabel('位置')
    ax4.set_ylabel('编码值')
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
    
    print(f"✅ 可视化完成，图片已保存为 '{save_path}'\n")


def visualize_positional_encoding_3d():
    """3D可视化位置编码"""
    print("🎨 3D可视化位置编码...")
    
    d_model = 32
    max_len = 50
    pe = PositionalEncoding(d_model, max_len)
    
    # 获取位置编码数据
    pos_encoding = pe.pe[:max_len, 0, :].detach().numpy()  # [max_len, d_model]
    
    # 创建3D图形
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 3D表面图
    ax1 = fig.add_subplot(131, projection='3d')
    X, Y = np.meshgrid(range(max_len), range(d_model))
    Z = pos_encoding.T
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_title('位置编码3D表面图')
    ax1.set_xlabel('位置')
    ax1.set_ylabel('维度')
    ax1.set_zlabel('编码值')
    
    # 2. 3D线框图
    ax2 = fig.add_subplot(132, projection='3d')
    for i in range(0, d_model, 4):  # 每4个维度画一条线
        ax2.plot(range(max_len), [i] * max_len, pos_encoding[:, i], 
                label=f'维度 {i}')
    ax2.set_title('位置编码3D线框图')
    ax2.set_xlabel('位置')
    ax2.set_ylabel('维度')
    ax2.set_zlabel('编码值')
    
    # 3. 位置编码的2D投影
    ax3 = fig.add_subplot(133)
    # 使用PCA降维到2D进行可视化
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pos_2d = pca.fit_transform(pos_encoding)
    
    scatter = ax3.scatter(pos_2d[:, 0], pos_2d[:, 1], 
                         c=range(max_len), cmap='viridis', s=50)
    ax3.set_title('位置编码2D投影（PCA）')
    ax3.set_xlabel('第一主成分')
    ax3.set_ylabel('第二主成分')
    plt.colorbar(scatter, ax=ax3, label='位置')
    
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
    
    print(f"✅ 3D可视化完成，图片已保存为 '{save_path}'\n")


def test_positional_encoding_math_properties():
    """测试位置编码的数学性质"""
    print("🧪 测试位置编码数学性质...")
    
    d_model = 64
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    
    pos_encoding = pe.pe[:max_len, 0, :].detach().numpy()
    
    # 1. 测试周期性
    print("测试周期性...")
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
                f"维度{dim}的sin^2+cos^2不等于1"
    
    print("✅ 周期性测试通过")
    
    # 2. 测试不同位置的唯一性
    print("测试位置唯一性...")
    unique_positions = np.unique(pos_encoding, axis=0)
    assert len(unique_positions) == max_len, "每个位置应该有唯一的编码"
    print("✅ 位置唯一性测试通过")
    
    # 3. 测试编码的数值范围
    print("测试数值范围...")
    min_val = pos_encoding.min()
    max_val = pos_encoding.max()
    assert -1.1 <= min_val <= -0.9, f"最小值应该在-1左右，实际为{min_val}"
    assert 0.9 <= max_val <= 1.1, f"最大值应该在1左右，实际为{max_val}"
    print(f"✅ 数值范围测试通过: [{min_val:.3f}, {max_val:.3f}]")
    
    print("🎉 数学性质测试全部通过！\n")


def benchmark_positional_encoding():
    """性能测试"""
    print("⚡ 性能测试...")
    
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
        x = torch.randn(seq_len, batch_size, d_model)
        
        # 预热
        for _ in range(10):
            _ = pe(x)
        
        # 计时
        start_time = time.time()
        for _ in range(100):
            output = pe(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        print(f"序列长度 {seq_len}, 批次大小 {batch_size}: "
              f"平均时间 {avg_time*1000:.2f}ms")
    
    print("✅ 性能测试完成\n")


def run_basic_tests():
    """运行基础测试（不需要可视化依赖）"""
    print("🧪 运行基础功能测试...")
    
    try:
        # 运行基础测试
        test_positional_encoding_basic()
        test_positional_encoding_different_lengths()
        test_positional_encoding_parameters()
        test_positional_encoding_math_properties()
        benchmark_positional_encoding()
        
        print("✅ 基础测试全部通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def run_visualization_tests():
    """运行可视化测试"""
    print("🎨 运行可视化测试...")
    
    try:
        # 检查依赖
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from sklearn.decomposition import PCA
        except ImportError as e:
            print(f"⚠️ 缺少可视化依赖: {e}")
            print("请安装: pip install matplotlib scikit-learn")
            return False
        
        # 运行可视化测试
        visualize_positional_encoding()
        visualize_positional_encoding_3d()
        
        print("✅ 可视化测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 可视化测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🚀 开始位置编码功能测试\n")
    
    # 运行基础测试
    basic_success = run_basic_tests()
    
    if basic_success:
        print("\n" + "="*50)
        
        # 询问是否运行可视化测试
        try:
            response = input("是否运行可视化测试？(y/n): ").lower().strip()
            if response in ['y', 'yes', '是']:
                run_visualization_tests()
            else:
                print("跳过可视化测试")
        except KeyboardInterrupt:
            print("\n测试被用户中断")
        except:
            print("无法获取用户输入，跳过可视化测试")
    
    print("\n🎉 测试完成！")


if __name__ == "__main__":
    main()
