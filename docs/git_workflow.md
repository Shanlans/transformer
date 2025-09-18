# Git工作流程指南

## 分支策略

### 主要分支
- `main`: 主分支，包含稳定的生产代码
- `develop`: 开发分支，用于集成功能
- `feature/*`: 功能分支，如 `feature/transformer-implementation`
- `bugfix/*`: 修复分支，如 `bugfix/attention-bug`
- `hotfix/*`: 紧急修复分支

### 分支命名规范
```
feature/功能描述
bugfix/问题描述
hotfix/紧急修复描述
release/版本号
```

## 工作流程

### 1. 开始新功能
```bash
# 切换到develop分支
git checkout develop
git pull origin develop

# 创建功能分支
git checkout -b feature/transformer-implementation

# 开始开发...
```

### 2. 提交更改
```bash
# 查看更改
git status
git diff

# 添加文件
git add .
# 或添加特定文件
git add src/transformer.py

# 提交更改
git commit -m "feat(transformer): 实现多头注意力机制"

# 推送到远程
git push origin feature/transformer-implementation
```

### 3. 创建Pull Request
1. 在GitHub/GitLab上创建PR
2. 填写PR描述
3. 请求代码审查
4. 通过CI/CD检查
5. 合并到目标分支

### 4. 代码审查检查清单
- [ ] 代码符合项目规范
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 没有破坏现有功能
- [ ] 提交信息清晰明确

## 常用命令

### 基础操作
```bash
# 查看状态
git status

# 查看差异
git diff
git diff --cached

# 查看历史
git log --oneline
git log --graph --oneline --all

# 查看分支
git branch -a
```

### 撤销操作
```bash
# 撤销工作区更改
git checkout -- <file>

# 撤销暂存区更改
git reset HEAD <file>

# 撤销最后一次提交（保留更改）
git reset --soft HEAD~1

# 撤销最后一次提交（丢弃更改）
git reset --hard HEAD~1
```

### 分支操作
```bash
# 创建并切换分支
git checkout -b <branch-name>

# 切换分支
git checkout <branch-name>

# 删除分支
git branch -d <branch-name>

# 合并分支
git merge <branch-name>
```

### 远程操作
```bash
# 添加远程仓库
git remote add origin <url>

# 查看远程仓库
git remote -v

# 拉取更改
git pull origin <branch>

# 推送更改
git push origin <branch>

# 设置上游分支
git push -u origin <branch>
```

## 最佳实践

### 提交规范
1. 使用清晰的提交信息
2. 每次提交只做一件事
3. 提交前运行测试
4. 使用提交模板

### 分支管理
1. 保持分支更新
2. 及时删除已合并的分支
3. 使用有意义的分支名
4. 定期同步主分支

### 代码质量
1. 编写清晰的代码
2. 添加必要的注释
3. 编写测试用例
4. 遵循代码规范

## 故障排除

### 常见问题
1. **合并冲突**: 手动解决冲突后提交
2. **提交到错误分支**: 使用 `git cherry-pick` 移动提交
3. **丢失提交**: 使用 `git reflog` 查找
4. **撤销推送**: 使用 `git revert` 创建反向提交

### 紧急情况
```bash
# 强制推送（谨慎使用）
git push --force-with-lease origin <branch>

# 重置到远程状态
git fetch origin
git reset --hard origin/<branch>
```
