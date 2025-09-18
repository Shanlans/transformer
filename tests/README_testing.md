# 测试说明文档

## 📁 测试目录结构

```
tests/
├── functional/                    # 功能测试
│   └── test_positional_encoding.py  # 位置编码测试（包含运行器）
├── visualizations/               # 可视化图片（自动创建）
│   ├── positional_encoding_visualization.png
│   └── positional_encoding_3d_visualization.png
└── README_testing.md               # 测试说明
```

## 🚀 快速开始

### 1. 安装项目依赖（包含测试依赖）
```bash
pip install -r config/requirements.txt
```

### 2. 运行位置编码测试
```bash
# 直接运行测试文件
python tests/functional/test_positional_encoding.py
```

## 🧪 测试内容

### 基础功能测试
- ✅ 形状验证
- ✅ 固定性测试
- ✅ 位置区分测试
- ✅ 不同序列长度测试
- ✅ 参数设置测试
- ✅ 数学性质测试
- ✅ 性能测试

### 可视化测试
- 🎨 位置编码热力图
- 🎨 位置编码曲线图
- 🎨 不同位置编码分布
- 🎨 位置编码周期性
- 🎨 3D表面图
- 🎨 3D线框图
- 🎨 2D投影图（PCA）

## 📊 测试输出

### 控制台输出
```
🧪 测试位置编码基础功能...
输入形状: torch.Size([50, 32, 512])
位置编码形状: torch.Size([100, 1, 512])
输出形状: torch.Size([50, 32, 512])
✅ 形状测试通过
✅ 固定性测试通过
✅ 位置区分测试通过
🎉 基础功能测试全部通过！
```

### 可视化输出
- `tests/visualizations/positional_encoding_visualization.png` - 2D可视化
- `tests/visualizations/positional_encoding_3d_visualization.png` - 3D可视化

**注意**: `visualizations` 文件夹会在运行测试时自动创建

## 🔧 自定义测试

### 修改测试参数
在 `test_positional_encoding.py` 中可以修改：
- `d_model`: 模型维度
- `max_len`: 最大序列长度
- `position_factor`: 位置因子
- `fast_model`: 是否使用快速模式

### 添加新测试
```python
def test_your_custom_test():
    """您的自定义测试"""
    # 测试代码
    pass
```

## 🐛 故障排除

### 常见问题

1. **ImportError: No module named 'sklearn'**
   ```bash
   pip install scikit-learn
   ```

2. **ImportError: No module named 'matplotlib'**
   ```bash
   pip install matplotlib
   ```

3. **路径问题**
   - 确保在项目根目录运行测试
   - 检查Python路径设置

### 调试技巧

1. **单独运行测试函数**
   ```python
   from tests.functional.test_positional_encoding import test_positional_encoding_basic
   test_positional_encoding_basic()
   ```

2. **查看中间结果**
   ```python
   import torch
   from src.models.transformer import PositionalEncoding
   
   pe = PositionalEncoding(64, 100)
   print(pe.pe.shape)
   print(pe.pe[0, 0, :10])  # 查看前10个维度
   ```

## 📈 性能基准

测试会输出不同配置下的性能数据：
```
序列长度 50, 批次大小 32: 平均时间 0.15ms
序列长度 100, 批次大小 64: 平均时间 0.28ms
序列长度 200, 批次大小 32: 平均时间 0.45ms
序列长度 500, 批次大小 16: 平均时间 0.89ms
```

## 🎯 下一步

测试通过后，可以继续实现下一个组件：
1. FeedForward（前馈网络）
2. MultiHeadAttention（多头注意力）
3. EncoderLayer（编码器层）
4. DecoderLayer（解码器层）
5. Transformer（完整模型）
