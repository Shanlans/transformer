#!/bin/bash
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
