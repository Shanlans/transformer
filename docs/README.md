# Transformer Project

这是一个基于PyTorch的Transformer项目，配置了专门的conda环境。

## 环境配置

### 环境信息
- **环境名称**: torch2.5
- **Python版本**: 3.10.18
- **PyTorch版本**: 2.2.2
- **CUDA支持**: CPU版本

### 快速开始

#### 方法1: 使用conda环境文件（推荐）
```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate torch2.5
```

#### 方法2: 使用pip安装
```bash
# 创建虚拟环境
python -m venv torch2.5_env
source torch2.5_env/bin/activate  # macOS/Linux
# 或 torch2.5_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 方法3: 使用提供的脚本
```bash
# 激活现有环境
source activate_torch2.5.sh
```

### 验证安装
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 项目结构
```
transformer/
├── environment.yml          # Conda环境配置
├── requirements.txt         # Python依赖列表
├── pyproject.toml          # 现代Python项目配置
├── activate_torch2.5.sh     # 环境激活脚本
├── test_environment.py     # 环境测试脚本
├── init_git.sh             # Git初始化脚本
├── .gitignore              # Git忽略文件
├── .gitattributes          # Git属性配置
├── .gitmessage             # Git提交模板
├── git_workflow.md         # Git工作流程指南
├── commit_template.md      # 提交信息模板说明
├── .vscode/
│   └── settings.json       # VS Code配置
└── README.md               # 项目说明
```

## 开发指南

### 环境管理
- 使用 `conda activate torch2.5` 激活环境
- 使用 `conda deactivate` 退出环境
- 添加新依赖时，更新 `requirements.txt` 和 `environment.yml`

### 依赖管理
- 主要依赖在 `requirements.txt` 中
- 完整环境配置在 `environment.yml` 中
- 使用 `pip freeze > requirements.txt` 更新依赖列表

### 版本控制
- 运行 `bash init_git.sh` 初始化Git仓库
- 查看 `git_workflow.md` 了解详细工作流程
- 使用 `commit_template.md` 规范提交信息
- 遵循分支策略：`main`、`develop`、`feature/*`

### 代码质量
- 使用 `test_environment.py` 验证环境
- 遵循PEP 8代码规范
- 编写清晰的提交信息
- 定期同步主分支

## 快速开始

### 1. 环境设置
```bash
# 创建环境
conda env create -f environment.yml
conda activate torch2.5

# 验证安装
python test_environment.py
```

### 2. 版本控制
```bash
# 初始化Git
bash init_git.sh

# 添加远程仓库
git remote add origin <your-repo-url>
git push -u origin main
```

### 3. 开发流程
```bash
# 创建功能分支
git checkout -b feature/your-feature

# 开发并提交
git add .
git commit -m "feat: 添加新功能"

# 推送并创建PR
git push origin feature/your-feature
```

## 注意事项
- 本项目使用CPU版本的PyTorch，适合在Mac上运行
- NumPy版本限制为 <2.0 以确保与PyTorch兼容
- 建议在虚拟环境中开发，避免依赖冲突
- 遵循Git工作流程，保持代码质量

