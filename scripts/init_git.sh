#!/bin/bash
# 简化的Git初始化脚本

echo "🚀 初始化Git版本控制..."

# 检查Git是否已安装
if ! command -v git &> /dev/null; then
    echo "❌ Git未安装，请先安装Git"
    exit 1
fi

# 初始化Git仓库
if [ ! -d ".git" ]; then
    echo "📁 初始化Git仓库..."
    git init
    echo "✅ Git仓库初始化完成"
else
    echo "✅ Git仓库已存在"
fi

# 设置默认分支
echo "🌿 设置默认分支为main..."
git branch -M main

# 配置Git（如果未设置）
if [ -z "$(git config user.name)" ]; then
    echo "👤 设置Git用户信息..."
    echo "请输入您的姓名:"
    read -r username
    git config user.name "$username"
fi

if [ -z "$(git config user.email)" ]; then
    echo "📧 设置Git邮箱..."
    echo "请输入您的邮箱:"
    read -r useremail
    git config user.email "$useremail"
fi

# 设置提交模板
echo "📝 配置提交模板..."
git config commit.template .gitmessage

# 添加所有文件
echo "📦 添加文件到Git..."
git add .

# 创建初始提交
echo "💾 创建初始提交..."
git commit -m "🎉 初始提交: 设置PyTorch Transformer项目

- 配置conda环境 (torch2.5)
- 添加PyTorch 2.2.2及相关依赖
- 设置项目结构和配置文件
- 添加环境测试脚本
- 配置Git工作流程"

echo ""
echo "✅ Git初始化完成！"
echo ""
echo "📋 下一步操作："
echo "1. 添加远程仓库: git remote add origin <your-repo-url>"
echo "2. 推送到远程: git push -u origin main"
echo ""
echo "🔧 常用命令："
echo "- git status          # 查看状态"
echo "- git add .           # 添加更改"
echo "- git commit -m 'msg' # 提交更改"
echo "- git push            # 推送到远程"
echo "- git log --oneline   # 查看历史"
echo ""
echo "📖 详细工作流程请查看: git_workflow.md"
