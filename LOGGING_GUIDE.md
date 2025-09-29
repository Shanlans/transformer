# 结构化日志系统

## 📋 **概述**

我们已经实现了统一的结构化日志系统，替换了所有分散的print语句，提供了：

- **结构化输出**：统一的日志格式和分类
- **可配置的详细程度**：支持verbose、normal、quiet三种模式
- **操作跟踪**：自动跟踪操作开始和结束时间
- **错误处理**：统一的错误和警告处理
- **文件日志**：可选的日志文件记录

## 🎯 **日志级别**

### **LogLevel枚举**
- `DEBUG`: 🔍 调试信息
- `INFO`: ℹ️ 一般信息  
- `SUCCESS`: ✅ 成功操作
- `WARNING`: ⚠️ 警告信息
- `ERROR`: ❌ 错误信息
- `CRITICAL`: 🚨 严重错误

### **LogCategory枚举**
- `SYSTEM`: 系统操作
- `EXPERIMENT`: 实验管理
- `TRAINING`: 训练过程
- `CLOUD`: 云端操作
- `CHECKPOINT`: 检查点管理
- `CONFIG`: 配置管理
- `CLEANUP`: 清理操作
- `VALIDATION`: 参数验证

## 🚀 **使用方法**

### **命令行参数**

```bash
# 正常模式（默认）
python run.py --list-experiments

# 详细模式
python run.py --verbose --list-experiments

# 安静模式（只显示错误和警告）
python run.py --quiet --list-experiments
```

### **环境变量**

```bash
# 设置详细模式
export TRANSFORMER_VERBOSE=true

# 设置安静模式
export TRANSFORMER_QUIET=true

# 设置日志文件
export TRANSFORMER_LOG_FILE=logs/training.log

# 设置日志级别
export TRANSFORMER_LOG_LEVEL=DEBUG
```

## 📊 **输出示例**

### **正常模式**
```
06:38:49 - transformer - INFO - ℹ️ [SYSTEM] Starting Initializing components
06:38:49 - transformer - INFO - ✅ [SYSTEM] Initializing components completed in 0.00s
06:38:49 - transformer - INFO - ✅ [SYSTEM] Unified Trainer initialized
06:38:49 - transformer - INFO - ℹ️ [SYSTEM] Available resources:
06:38:49 - transformer - INFO - ℹ️ [SYSTEM]   COLABCODE: ColabCode SSH connection available (available)
06:38:49 - transformer - INFO - ℹ️ [EXPERIMENT] Starting Listing experiments
06:38:49 - transformer - INFO - ℹ️ [EXPERIMENT] No experiments found
06:38:49 - transformer - INFO - ✅ [EXPERIMENT] Listing experiments completed in 0.00s
```

### **安静模式**
```
# 只显示错误和警告，没有其他输出
```

### **详细模式**
```
# 显示所有调试信息，包括内部状态和详细操作步骤
```

## 🔧 **代码集成**

### **在代码中使用**

```python
from utils.logger import get_logger, LogLevel, LogCategory, OperationContext, track_operation

# 获取logger
logger = get_logger(verbose=True, quiet=False)

# 基本日志
logger.info(LogCategory.SYSTEM, "System message")
logger.success(LogCategory.EXPERIMENT, "Experiment created")
logger.warning(LogCategory.TRAINING, "Training warning")
logger.error(LogCategory.CLOUD, "Cloud error")

# 操作跟踪
with OperationContext("Creating experiment", LogCategory.EXPERIMENT):
    # 操作代码
    pass

# 装饰器跟踪
@track_operation("Training execution", LogCategory.TRAINING)
def run_training():
    pass

# 表格输出
logger.print_table(
    ["Name", "Value", "Status"],
    [["Item1", "100", "Active"], ["Item2", "200", "Inactive"]],
    "Test Table"
)

# 摘要输出
logger.print_summary("Training Summary", {
    "Epochs": 10,
    "Loss": 0.123,
    "Accuracy": 0.95
})
```

## 📁 **文件结构**

```
src/utils/
├── logger.py          # 统一日志系统
├── log_config.py      # 日志配置
└── ...
```

## 🎨 **格式化功能**

### **表格输出**
```python
logger.print_table(
    headers=["Name", "Value", "Status"],
    rows=[["Item1", "100", "Active"], ["Item2", "200", "Inactive"]],
    title="Available Items"
)
```

### **摘要输出**
```python
logger.print_summary("Training Results", {
    "Total Epochs": 10,
    "Final Loss": 0.123,
    "Best Accuracy": 0.95,
    "Training Time": "2.5 minutes"
})
```

### **进度条**
```python
logger.print_progress(current=5, total=10, operation="Training")
# 输出: Training: |██████████████████████████████| 50.0% (5/10)
```

## 🔍 **调试功能**

### **操作跟踪**
```python
# 自动跟踪操作时间
with OperationContext("Data loading", LogCategory.SYSTEM):
    load_data()
# 输出: ✅ [SYSTEM] Data loading completed in 2.34s
```

### **错误处理**
```python
try:
    risky_operation()
except Exception as e:
    logger.error(LogCategory.SYSTEM, f"Operation failed: {e}")
    if logger.verbose:
        import traceback
        traceback.print_exc()
```

## 📈 **性能优化**

- **延迟初始化**：Logger只在需要时创建
- **条件输出**：根据日志级别过滤消息
- **文件缓冲**：日志文件使用缓冲写入
- **内存优化**：避免不必要的字符串格式化

## 🛠️ **配置选项**

### **日志级别映射**
```python
QUIET_MODE = {
    "console": "WARNING",
    "file": "DEBUG"
}

NORMAL_MODE = {
    "console": "INFO", 
    "file": "DEBUG"
}

VERBOSE_MODE = {
    "console": "DEBUG",
    "file": "DEBUG"
}
```

### **类别级别**
```python
CATEGORY_LEVELS = {
    "SYSTEM": "INFO",
    "EXPERIMENT": "INFO", 
    "TRAINING": "INFO",
    "CLOUD": "INFO",
    "CHECKPOINT": "INFO",
    "CONFIG": "INFO",
    "CLEANUP": "INFO",
    "VALIDATION": "INFO"
}
```

## 🎯 **最佳实践**

1. **使用适当的日志级别**：不要滥用DEBUG级别
2. **包含上下文信息**：在日志消息中包含足够的上下文
3. **使用操作跟踪**：对长时间运行的操作使用OperationContext
4. **错误处理**：总是记录错误和异常
5. **性能考虑**：在性能关键路径上避免详细日志

## 🔄 **迁移指南**

### **从print语句迁移**
```python
# 旧代码
print("✅ Training completed successfully!")
print(f"Loss: {loss:.4f}")

# 新代码
logger.success(LogCategory.TRAINING, "Training completed successfully!")
logger.info(LogCategory.TRAINING, f"Loss: {loss:.4f}")
```

### **从分散的日志迁移**
```python
# 旧代码
print("=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)
print(f"Epochs: {epochs}")
print(f"Loss: {loss}")

# 新代码
logger.print_summary("Training Summary", {
    "Epochs": epochs,
    "Loss": loss
})
```

这个结构化日志系统大大提高了代码的可读性和可维护性，同时提供了灵活的配置选项来适应不同的使用场景。
