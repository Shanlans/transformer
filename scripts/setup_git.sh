#!/bin/bash
# Git仓库设置脚本

echo "🚀 设置Git版本控制..."

# 检查是否在Git仓库中
if [ ! -d ".git" ]; then
    echo "初始化Git仓库..."
    git init
    echo "✅ Git仓库初始化完成"
else
    echo "✅ Git仓库已存在"
fi

# 设置Git用户信息（如果未设置）
if [ -z "$(git config user.name)" ]; then
    echo "请设置Git用户信息："
    read -p "请输入您的姓名: " username
    git config user.name "$username"
fi

if [ -z "$(git config user.email)" ]; then
    echo "请设置Git邮箱："
    read -p "请输入您的邮箱: " useremail
    git config user.email "$useremail"
fi

# 设置默认分支为main
git branch -M main

# 添加所有文件
echo "添加文件到Git..."
git add .

# 创建初始提交
echo "创建初始提交..."
git commit -m "🎉 初始提交: 设置PyTorch Transformer项目

- 配置conda环境 (torch2.5)
- 添加PyTorch 2.2.2及相关依赖
- 设置项目结构和配置文件
- 添加环境测试脚本"

echo ""
echo "✅ Git设置完成！"
echo ""
echo "📋 下一步操作："
echo "1. 添加远程仓库: git remote add origin <your-repo-url>"
echo "2. 推送到远程: git push -u origin main"
echo "3. 查看状态: git status"
echo "4. 查看日志: git log --oneline"
echo ""
echo "🔧 常用Git命令："
echo "- git status          # 查看状态"
echo "- git add .           # 添加所有更改"
echo "- git commit -m 'msg' # 提交更改"
echo "- git push            # 推送到远程"
echo "- git pull            # 拉取远程更改"
echo "- git log --oneline   # 查看提交历史"
