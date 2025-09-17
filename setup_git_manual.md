# Git手动设置指南

由于自动脚本遇到权限问题，请按照以下步骤手动设置Git：

## 步骤1: 初始化Git仓库
```bash
cd /Users/shanlanshen/cursorproject/transformer
git init
```

## 步骤2: 配置Git用户信息
```bash
# 设置用户名
git config user.name "Your Name"

# 设置邮箱
git config user.email "your.email@example.com"

# 设置默认分支
git branch -M main
```

## 步骤3: 配置提交模板
```bash
git config commit.template .gitmessage
```

## 步骤4: 添加文件并提交
```bash
# 添加所有文件
git add .

# 创建初始提交
git commit -m "🎉 初始提交: 设置PyTorch Transformer项目

- 配置conda环境 (torch2.5)
- 添加PyTorch 2.2.2及相关依赖
- 设置项目结构和配置文件
- 添加环境测试脚本
- 配置Git工作流程"
```

## 步骤5: 添加远程仓库（可选）
```bash
# 添加远程仓库
git remote add origin <your-repo-url>

# 推送到远程
git push -u origin main
```

## 步骤6: 验证设置
```bash
# 检查状态
git status

# 查看提交历史
git log --oneline

# 查看分支
git branch
```

## 常用Git命令
```bash
# 查看状态
git status

# 添加文件
git add .
git add <filename>

# 提交更改
git commit -m "提交信息"

# 查看历史
git log --oneline

# 创建分支
git checkout -b feature/your-feature

# 切换分支
git checkout main

# 合并分支
git merge feature/your-feature

# 推送到远程
git push origin main

# 拉取远程更改
git pull origin main
```

## 项目文件说明
- `.gitignore` - Git忽略文件配置
- `.gitattributes` - Git属性配置
- `.gitmessage` - 提交信息模板
- `git_workflow.md` - 详细工作流程指南
- `commit_template.md` - 提交模板说明

## 下一步
1. 按照上述步骤设置Git
2. 开始开发新功能
3. 遵循Git工作流程
4. 定期提交和推送代码
