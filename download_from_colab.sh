#!/bin/bash
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
