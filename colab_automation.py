#!/usr/bin/env python3
"""
Google Colab 自动化交互脚本

这个脚本可以自动处理：
1. 代码上传到 Colab
2. 依赖安装
3. 训练执行
4. 结果下载

使用方法：
python colab_automation.py --mode [upload|install|train|download|all]
"""

import os
import sys
import json
import time
import subprocess
import zipfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse

class ColabAutomation:
    """Google Colab 自动化管理器"""
    
    def __init__(self, project_dir: str = ".", colab_dir: str = "/content/transformer"):
        """
        初始化自动化管理器
        
        Args:
            project_dir: 本地项目目录
            colab_dir: Colab 中的项目目录
        """
        self.project_dir = Path(project_dir).resolve()
        self.colab_dir = colab_dir
        self.colab_notebook_path = "colab_automation.ipynb"
        
        print(f"🚀 ColabAutomation 初始化")
        print(f"   本地目录: {self.project_dir}")
        print(f"   Colab目录: {self.colab_dir}")
    
    def create_colab_notebook(self) -> str:
        """创建 Colab 自动化 Notebook"""
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# 🤖 Transformer 训练自动化\n",
                        "\n",
                        "这个 Notebook 会自动完成以下步骤：\n",
                        "1. 📥 克隆代码仓库\n",
                        "2. 📦 安装依赖\n",
                        "3. 🏃 执行训练\n",
                        "4. 📤 下载结果\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 1. 克隆代码仓库\n",
                        "!git clone https://github.com/Shanlans/transformer.git\n",
                        "!cd transformer && ls -la"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 2. 安装依赖\n",
                        "!cd transformer && pip install -r requirements.txt\n",
                        "!cd transformer && pip install colabcode"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 3. 检查环境\n",
                        "import torch\n",
                        "print(f\"PyTorch 版本: {torch.__version__}\")\n",
                        "print(f\"CUDA 可用: {torch.cuda.is_available()}\")\n",
                        "if torch.cuda.is_available():\n",
                        "    print(f\"GPU 设备: {torch.cuda.get_device_name(0)}\")\n",
                        "    print(f\"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 4. 执行训练\n",
                        "!cd transformer && python run.py --force-cloud --use-default"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 5. 检查训练结果\n",
                        "!cd transformer && ls -la checkpoints/\n",
                        "!cd transformer && find checkpoints/ -name \"*.pt\" -o -name \"*.json\" -o -name \"*.png\""
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 6. 压缩结果\n",
                        "!cd transformer && zip -r training_results.zip checkpoints/ config_history/ experiments/\n",
                        "!cd transformer && ls -la *.zip"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 7. 下载结果到本地\n",
                        "from google.colab import files\n",
                        "files.download('/content/transformer/training_results.zip')\n",
                        "print(\"✅ 结果已下载到本地！\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 8. 显示训练摘要\n",
                        "import json\n",
                        "import glob\n",
                        "\n",
                        "# 查找最新的训练历史\n",
                        "history_files = glob.glob('/content/transformer/checkpoints/run_*/training_history.json')\n",
                        "if history_files:\n",
                        "    latest_history = max(history_files, key=os.path.getctime)\n",
                        "    with open(latest_history, 'r') as f:\n",
                        "        history = json.load(f)\n",
                        "    \n",
                        "    print(\"📊 训练摘要:\")\n",
                        "    print(f\"   总轮数: {len(history.get('epochs', []))}\")\n",
                        "    print(f\"   最终损失: {history.get('epochs', [{}])[-1].get('train_loss', 'N/A'):.4f}\")\n",
                        "    print(f\"   最佳损失: {min([e.get('train_loss', float('inf')) for e in history.get('epochs', [])]):.4f}\")\n",
                        "else:\n",
                        "    print(\"⚠️ 未找到训练历史文件\")"
                    ]
                }
            ],
            "metadata": {
                "colab": {
                    "provenance": [],
                    "toc_visible": True
                },
                "kernelspec": {
                    "display_name": "Python 3",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 0
        }
        
        # 保存 Notebook
        with open(self.colab_notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_content, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Colab Notebook 已创建: {self.colab_notebook_path}")
        return self.colab_notebook_path
    
    def create_upload_script(self) -> str:
        """创建上传脚本"""
        upload_script = """#!/bin/bash
# 自动上传到 Google Colab 的脚本

echo "🚀 开始上传到 Google Colab..."

# 1. 创建项目压缩包
echo "📦 创建项目压缩包..."
zip -r transformer_project.zip . -x "*.git*" "checkpoints/*" "config_history/*" "experiments/results/*" "*.log" "*.tmp" "__pycache__/*" "*.pyc"

# 2. 上传到 Google Drive (需要手动操作)
echo "📤 请手动上传 transformer_project.zip 到 Google Drive"
echo "   然后运行以下命令解压："
echo "   !unzip transformer_project.zip -d /content/transformer"

# 3. 运行训练
echo "🏃 在 Colab 中运行训练："
echo "   !cd /content/transformer && python run.py --force-cloud --use-default"

echo "✅ 上传脚本完成！"
"""
        
        script_path = "upload_to_colab.sh"
        with open(script_path, 'w') as f:
            f.write(upload_script)
        
        # 添加执行权限
        os.chmod(script_path, 0o755)
        
        print(f"✅ 上传脚本已创建: {script_path}")
        return script_path
    
    def create_download_script(self) -> str:
        """创建下载脚本"""
        download_script = """#!/bin/bash
# 自动下载 Colab 结果的脚本

echo "📥 开始下载 Colab 结果..."

# 1. 检查本地结果目录
if [ ! -d "colab_results" ]; then
    mkdir colab_results
fi

# 2. 下载结果 (需要手动操作)
echo "📤 请手动下载 Colab 中的结果文件到 colab_results/ 目录"
echo "   或者使用以下 Python 代码："
echo ""
echo "   from google.colab import files"
echo "   files.download('/content/transformer/training_results.zip')"

# 3. 解压结果
if [ -f "colab_results/training_results.zip" ]; then
    echo "📦 解压结果..."
    cd colab_results
    unzip training_results.zip
    cd ..
    echo "✅ 结果已解压到 colab_results/ 目录"
else
    echo "⚠️ 未找到结果文件，请手动下载"
fi

echo "✅ 下载脚本完成！"
"""
        
        script_path = "download_from_colab.sh"
        with open(script_path, 'w') as f:
            f.write(download_script)
        
        # 添加执行权限
        os.chmod(script_path, 0o755)
        
        print(f"✅ 下载脚本已创建: {script_path}")
        return script_path
    
    def create_automation_guide(self) -> str:
        """创建自动化指南"""
        guide_content = """# 🤖 Google Colab 自动化指南

## 📋 **快速开始**

### **方法1：使用 Colab Notebook（推荐）**

1. **打开 Colab**：访问 [Google Colab](https://colab.research.google.com/)
2. **上传 Notebook**：上传 `colab_automation.ipynb`
3. **一键运行**：点击 "运行全部" 按钮
4. **自动下载**：训练完成后自动下载结果

### **方法2：手动步骤**

1. **克隆代码**：
   ```python
   !git clone https://github.com/Shanlans/transformer.git
   ```

2. **安装依赖**：
   ```python
   !cd transformer && pip install -r requirements.txt
   !cd transformer && pip install colabcode
   ```

3. **执行训练**：
   ```python
   !cd transformer && python run.py --force-cloud --use-default
   ```

4. **下载结果**：
   ```python
   from google.colab import files
   !cd transformer && zip -r results.zip checkpoints/
   files.download('/content/transformer/results.zip')
   ```

## 🔧 **自动化脚本**

- `colab_automation.ipynb`：完整的 Colab Notebook
- `upload_to_colab.sh`：上传脚本
- `download_from_colab.sh`：下载脚本

## 📊 **预期结果**

训练完成后，你将获得：
- ✅ 训练好的模型检查点
- ✅ 训练历史记录
- ✅ 可视化图表
- ✅ 评估结果
- ✅ 配置文件

## ⚠️ **注意事项**

1. **会话时间**：Colab 会话最长 12 小时
2. **GPU 限制**：免费用户有 GPU 使用限制
3. **存储限制**：Colab 有存储空间限制
4. **网络限制**：下载大文件可能较慢

## 🆘 **故障排除**

### **常见问题**：
- **导入错误**：检查依赖是否正确安装
- **内存不足**：减少批次大小或模型大小
- **训练中断**：检查网络连接和会话状态

### **获取帮助**：
- 查看训练日志
- 检查 Colab 控制台输出
- 使用 `!nvidia-smi` 检查 GPU 状态
"""
        
        guide_path = "COLAB_AUTOMATION_GUIDE.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        print(f"✅ 自动化指南已创建: {guide_path}")
        return guide_path
    
    def run_automation(self, mode: str = "all"):
        """运行自动化流程"""
        print(f"🚀 开始运行自动化流程: {mode}")
        
        if mode in ["all", "notebook"]:
            self.create_colab_notebook()
        
        if mode in ["all", "upload"]:
            self.create_upload_script()
        
        if mode in ["all", "download"]:
            self.create_download_script()
        
        if mode in ["all", "guide"]:
            self.create_automation_guide()
        
        print("✅ 自动化流程完成！")
        print("\n📋 下一步操作：")
        print("1. 打开 Google Colab")
        print("2. 上传 colab_automation.ipynb")
        print("3. 点击 '运行全部'")
        print("4. 等待训练完成")
        print("5. 自动下载结果")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Google Colab 自动化工具")
    parser.add_argument('--mode', choices=['all', 'notebook', 'upload', 'download', 'guide'], 
                       default='all', help='运行模式')
    parser.add_argument('--project-dir', default='.', help='项目目录')
    parser.add_argument('--colab-dir', default='/content/transformer', help='Colab 目录')
    
    args = parser.parse_args()
    
    # 创建自动化管理器
    automation = ColabAutomation(
        project_dir=args.project_dir,
        colab_dir=args.colab_dir
    )
    
    # 运行自动化流程
    automation.run_automation(args.mode)


if __name__ == "__main__":
    main()
