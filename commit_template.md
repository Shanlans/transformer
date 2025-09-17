# Git提交信息模板

## 提交类型 (Type)
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式调整（不影响功能）
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动
- `perf`: 性能优化
- `ci`: CI/CD相关

## 作用域 (Scope) - 可选
- `transformer`: Transformer模型相关
- `attention`: 注意力机制
- `data`: 数据处理
- `train`: 训练相关
- `eval`: 评估相关
- `config`: 配置文件
- `env`: 环境配置

## 提交信息格式
```
<type>(<scope>): <subject>

<body>

<footer>
```

## 示例
```
feat(transformer): 添加多头注意力机制

实现了Transformer模型中的多头注意力机制，包括：
- 查询、键、值的线性变换
- 注意力权重计算和缩放
- 多头拼接和输出投影层
- 添加了dropout和层归一化

测试覆盖率达到95%，性能提升15%

Closes #123
```

## 使用说明
1. 将提交信息保存到 `.gitmessage` 文件
2. 配置Git使用模板: `git config commit.template .gitmessage`
3. 提交时使用: `git commit` (会自动打开编辑器)
