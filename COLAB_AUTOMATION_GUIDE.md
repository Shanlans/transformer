# 🤖 Google Colab 自动化指南

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
