# Transformer Project

基于PyTorch的Transformer模型实现项目，采用模块化的项目结构设计。

## 📁 项目结构

```
transformer/
├── config/                  # 配置文件
│   ├── environment.yml      # Conda环境配置
│   ├── requirements.txt     # Python依赖列表
│   ├── pyproject.toml       # 项目配置
│   ├── .gitignore          # Git忽略规则
│   └── test_environment.py  # 环境测试脚本
├── scripts/                 # 脚本文件
│   ├── activate_torch2.5.sh # 环境激活脚本
│   ├── init_git.sh         # Git初始化脚本
│   ├── setup_git.sh        # Git设置脚本
│   └── check_setup.py      # 项目状态检查脚本
├── docs/                   # 文档
│   ├── README.md           # 详细项目说明
│   ├── git_workflow.md     # Git工作流程指南
│   ├── commit_template.md  # 提交信息模板说明
│   └── setup_git_manual.md # Git手动设置指南
├── src/                    # 源代码
│   ├── models/             # 模型定义
│   ├── data/               # 数据处理
│   ├── training/           # 训练相关
│   └── utils/              # 工具函数
├── tests/                  # 测试文件
├── examples/               # 示例代码
├── .vscode/                # VS Code配置
└── .git/                   # Git仓库
```

## 🚀 快速开始

### 1. 环境设置
```bash
# 激活环境
source scripts/activate_torch2.5.sh

# 或手动激活
conda activate torch2.5
```

### 2. 验证环境
```bash
python scripts/check_setup.py
```

### 3. 查看文档
- [详细项目说明](docs/README.md)
- [Git工作流程](docs/git_workflow.md)
- [环境配置](config/)

## 📖 详细文档

请查看 `docs/` 目录下的详细文档。

## 🔧 开发

- **源代码**: 位于 `src/` 目录
- **测试文件**: 位于 `tests/` 目录
- **配置文件**: 位于 `config/` 目录
- **脚本工具**: 位于 `scripts/` 目录

## 📋 项目特性

- ✅ **模块化设计**: 清晰的目录结构，便于维护
- ✅ **环境管理**: 完整的Conda环境配置
- ✅ **版本控制**: 规范的Git工作流程
- ✅ **代码质量**: 配置文件和检查工具
- ✅ **文档完善**: 详细的使用说明和开发指南

## 🛠️ 技术栈

- **深度学习框架**: PyTorch 2.2.2
- **Python版本**: 3.10.18
- **环境管理**: Conda
- **版本控制**: Git
- **开发环境**: VS Code

## 📝 开发指南

### 环境管理
- 使用 `conda activate torch2.5` 激活环境
- 使用 `conda deactivate` 退出环境
- 添加新依赖时，更新 `config/requirements.txt` 和 `config/environment.yml`

### 版本控制
- 运行 `bash scripts/init_git.sh` 初始化Git仓库
- 查看 `docs/git_workflow.md` 了解详细工作流程
- 使用 `docs/commit_template.md` 规范提交信息
- 遵循分支策略：`main`、`develop`、`feature/*`

### 代码质量
- 使用 `python scripts/check_setup.py` 验证环境
- 遵循PEP 8代码规范
- 编写清晰的提交信息
- 定期同步主分支

## 🎯 下一步计划

1. 实现Transformer模型核心组件
2. 创建训练和评估脚本
3. 添加数据处理工具
4. 完善测试用例
5. 添加示例代码

## 📄 许可证

本项目采用MIT许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

---

**注意**: 这是一个学习和研究项目，用于深入理解Transformer架构的实现细节。