#!/usr/bin/env python3
"""
项目设置检查脚本
检查环境配置、Git状态和项目结构
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, check=True):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"❌ 命令失败: {cmd}")
            print(f"错误: {result.stderr}")
        return result
    except Exception as e:
        print(f"❌ 执行命令时出错: {e}")
        return None

def check_python_environment():
    """检查Python环境"""
    print("🐍 检查Python环境...")
    
    # Python版本
    python_version = sys.version_info
    print(f"  Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查PyTorch
    try:
        import torch
        print(f"  ✅ PyTorch版本: {torch.__version__}")
        print(f"  ✅ CUDA可用: {torch.cuda.is_available()}")
    except ImportError:
        print("  ❌ PyTorch未安装")
        return False
    
    # 检查其他依赖
    required_packages = ['numpy', 'torchvision', 'torchaudio']
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package} 已安装")
        except ImportError:
            print(f"  ❌ {package} 未安装")
    
    return True

def check_git_status():
    """检查Git状态"""
    print("\n📁 检查Git状态...")
    
    # 检查是否在Git仓库中
    if not os.path.exists('.git'):
        print("  ❌ 不在Git仓库中")
        return False
    
    # 检查Git配置
    user_name = run_command("git config user.name")
    user_email = run_command("git config user.email")
    
    if user_name and user_name.stdout.strip():
        print(f"  ✅ Git用户: {user_name.stdout.strip()}")
    else:
        print("  ⚠️  Git用户未设置")
    
    if user_email and user_email.stdout.strip():
        print(f"  ✅ Git邮箱: {user_email.stdout.strip()}")
    else:
        print("  ⚠️  Git邮箱未设置")
    
    # 检查分支
    branch_result = run_command("git branch --show-current")
    if branch_result and branch_result.stdout.strip():
        print(f"  ✅ 当前分支: {branch_result.stdout.strip()}")
    
    # 检查未提交的更改
    status_result = run_command("git status --porcelain")
    if status_result and status_result.stdout.strip():
        print("  ⚠️  有未提交的更改:")
        for line in status_result.stdout.strip().split('\n'):
            print(f"    {line}")
    else:
        print("  ✅ 工作区干净")
    
    return True

def check_project_structure():
    """检查项目结构"""
    print("\n📂 检查项目结构...")
    
    required_files = [
        'environment.yml',
        'requirements.txt',
        'pyproject.toml',
        '.gitignore',
        '.gitattributes',
        'README.md',
        'test_environment.py',
        'init_git.sh'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} 缺失")
            missing_files.append(file)
    
    # 检查目录结构
    required_dirs = ['.vscode']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ 缺失")
    
    return len(missing_files) == 0

def check_conda_environment():
    """检查Conda环境"""
    print("\n🐍 检查Conda环境...")
    
    # 检查conda是否可用
    conda_result = run_command("conda --version", check=False)
    if not conda_result or conda_result.returncode != 0:
        print("  ❌ Conda未安装或不可用")
        return False
    
    print(f"  ✅ Conda版本: {conda_result.stdout.strip()}")
    
    # 检查torch2.5环境
    env_result = run_command("conda env list", check=False)
    if env_result and 'torch2.5' in env_result.stdout:
        print("  ✅ torch2.5环境存在")
    else:
        print("  ❌ torch2.5环境不存在")
        return False
    
    return True

def main():
    """主函数"""
    print("🔍 项目设置检查")
    print("=" * 50)
    
    checks = [
        ("Python环境", check_python_environment),
        ("Conda环境", check_conda_environment),
        ("Git状态", check_git_status),
        ("项目结构", check_project_structure)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ 检查 {name} 时出错: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("📊 检查结果汇总:")
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有检查都通过了！项目设置完成。")
        print("\n📋 下一步:")
        print("1. 开始开发: git checkout -b feature/your-feature")
        print("2. 查看工作流程: cat git_workflow.md")
        print("3. 运行测试: python test_environment.py")
    else:
        print("⚠️  部分检查失败，请根据上述信息修复问题。")
        print("\n🔧 修复建议:")
        print("1. 运行环境设置: conda env create -f environment.yml")
        print("2. 初始化Git: bash init_git.sh")
        print("3. 检查文件完整性")

if __name__ == "__main__":
    main()
