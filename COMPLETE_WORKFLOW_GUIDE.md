# Complete Training Workflow Guide

## 🎯 完整的训练流程

您的训练系统现在已经完全实现了以下5个核心需求：

### 1. ✅ 训练框架
- **主训练脚本**: `train.py` - 完整的训练系统
- **实验管理**: `manage_experiments.py` - 管理不同实验配置
- **配置版本管理**: `manage_config_versions.py` - 跟踪配置变更

### 2. ✅ 从JSON配置加载训练参数
```bash
# 使用默认配置训练
python train.py

# 使用指定配置文件训练
python train.py --config training_config.json
```

### 3. ✅ 训练参数可管理
```bash
# 创建新实验
python manage_experiments.py create --name my_experiment --epochs 2 --batch_size 16

# 列出所有实验
python manage_experiments.py list

# 运行特定实验
python manage_experiments.py run --name my_experiment
```

### 4. ✅ 训练参数和checkpoint版本保持可溯源
```bash
# 查看配置版本历史
python manage_config_versions.py list

# 查看checkpoint的配置历史
python manage_config_versions.py history --checkpoint checkpoints/run_20250924_080829

# 比较不同配置版本
python manage_config_versions.py compare --version1 baseline --version2 modified
```

### 5. ✅ 实验配置只需2个epoch
- 默认配置已调整为2个epoch
- 快速验证和测试

## 🚀 典型工作流程

### 场景1: 快速测试新配置
```bash
# 1. 创建快速测试实验
python manage_experiments.py create \
    --name quick_test \
    --description "Quick test with small model" \
    --epochs 2 \
    --batch_size 16 \
    --d_model 128

# 2. 运行实验
python manage_experiments.py run --name quick_test

# 3. 查看结果
python manage_config_versions.py list
```

### 场景2: 超参数调优
```bash
# 1. 创建不同学习率的实验
python manage_experiments.py create --name lr_001 --learning_rate 0.001 --epochs 2
python manage_experiments.py create --name lr_0001 --learning_rate 0.0001 --epochs 2
python manage_experiments.py create --name lr_00001 --learning_rate 0.00001 --epochs 2

# 2. 运行所有实验
python manage_experiments.py run --name lr_001
python manage_experiments.py run --name lr_0001
python manage_experiments.py run --name lr_00001

# 3. 比较结果
python manage_experiments.py compare --names lr_001 lr_0001 lr_00001
```

### 场景3: 模型架构对比
```bash
# 1. 创建不同模型大小的实验
python manage_experiments.py create --name small_model --d_model 128 --n_heads 4 --epochs 2
python manage_experiments.py create --name medium_model --d_model 256 --n_heads 8 --epochs 2
python manage_experiments.py create --name large_model --d_model 512 --n_heads 8 --epochs 2

# 2. 运行实验
python manage_experiments.py run --name small_model
python manage_experiments.py run --name medium_model
python manage_experiments.py run --name large_model

# 3. 分析结果
python manage_experiments.py compare --names small_model medium_model large_model
```

## 📊 自动记录的信息

### 配置版本记录
- **自动保存**: 每次训练自动保存配置版本
- **版本ID**: 时间戳格式的版本标识
- **描述**: 自动生成训练开始时间描述
- **标签**: 自动标记为"training"和"auto_saved"
- **哈希值**: 配置内容的MD5哈希，用于检测变更

### Checkpoint关联
- **自动链接**: 配置版本自动与checkpoint运行目录关联
- **链接类型**: 标记为"training"类型
- **时间戳**: 记录链接创建时间
- **双向追溯**: 可以从配置查checkpoint，也可以从checkpoint查配置

### 训练历史
- **训练指标**: 每个epoch的loss、学习率等
- **可视化**: 自动生成训练曲线图
- **评估结果**: BLEU、METEOR等翻译指标
- **模型检查点**: 最佳模型自动保存

## 🔍 溯源查询示例

### 查看特定checkpoint使用的配置
```bash
python manage_config_versions.py history --checkpoint checkpoints/run_20250924_080829
```

输出：
```
Configuration history for checkpoint: checkpoints/run_20250924_080829
================================================================================
Config Version: config_20250924_080830
Link Type: training
Linked At: 2025-09-24T08:08:44.944398
Description: Training run starting at 2025-09-24 08:08:30
Tags: training, auto_saved
--------------------------------------------------------------------------------
```

### 查看所有配置版本
```bash
python manage_config_versions.py list
```

输出：
```
========================================================================================================================
CONFIGURATION VERSIONS
========================================================================================================================
Version ID                Created              Description                    Tags                 Hash        
------------------------------------------------------------------------------------------------------------------------
config_20250924_080830    2025-09-24T08:08:30  Training run starting at 2025  training, auto_saved 0878908f340 
config_20250924_080620    2025-09-24T08:06:20  Training run starting at 2025  training, auto_saved 44a5dcb950f 
baseline                  2025-09-24T08:05:38  Baseline configuration for in  baseline, initial    44a5dcb950f 
========================================================================================================================
```

## 📁 文件结构

```
transformer/
├── train.py                           # 主训练脚本
├── training_config.json               # 默认训练配置
├── manage_experiments.py              # 实验管理工具
├── manage_config_versions.py         # 配置版本管理工具
├── experiments/                       # 实验配置存储
│   ├── configs/                      # 实验配置文件
│   └── results/                      # 实验结果（预留）
├── config_history/                    # 配置版本历史
│   ├── versions/                     # 配置版本文件
│   └── links/                        # 配置-checkpoint链接
├── checkpoints/                       # 模型检查点
│   └── run_YYYYMMDD_HHMMSS/         # 按时间戳组织的训练运行
└── src/                              # 源代码
    ├── models/                       # 模型实现
    ├── datasets/                     # 数据集
    ├── trainers/                     # 训练器
    └── utils/                        # 工具模块
```

## 🎯 关键优势

1. **完全自动化**: 配置版本和checkpoint关联完全自动
2. **可追溯性**: 任何checkpoint都能追溯到使用的配置
3. **版本控制**: 配置变更历史完整记录
4. **快速迭代**: 2个epoch的快速验证
5. **实验管理**: 系统化的实验配置管理
6. **可视化**: 自动生成训练和评估图表

## 🚀 开始使用

1. **快速开始**:
   ```bash
   conda activate torch2.5
   python train.py
   ```

2. **创建实验**:
   ```bash
   python manage_experiments.py create --name my_test --epochs 2
   ```

3. **运行实验**:
   ```bash
   python manage_experiments.py run --name my_test
   ```

4. **查看历史**:
   ```bash
   python manage_config_versions.py list
   ```

您的训练系统现在已经完全满足所有需求！🎉
