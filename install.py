#!/usr/bin/env python3
"""
Simple Python-based installation script for Transformer Training System
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_status(message):
    """Print status message with formatting."""
    print(f"🔧 {message}")

def print_success(message):
    """Print success message with formatting."""
    print(f"✅ {message}")

def print_error(message):
    """Print error message with formatting."""
    print(f"❌ {message}")

def print_warning(message):
    """Print warning message with formatting."""
    print(f"⚠️  {message}")

def check_command(command):
    """Check if a command is available."""
    try:
        subprocess.run([command, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def run_command(command, description=""):
    """Run a command and handle errors."""
    if description:
        print_status(f"Running: {description}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {command}")
        if e.stderr:
            print(e.stderr)
        return False

def check_prerequisites():
    """Check if required tools are installed."""
    print_status("Checking prerequisites...")
    
    # Check Python
    if not check_command("python"):
        print_error("Python is not installed or not in PATH")
        return False
    print_success("Python is available")
    
    # Check pip
    if not check_command("pip"):
        print_error("pip is not installed or not in PATH")
        return False
    print_success("pip is available")
    
    # Check conda (optional)
    if check_command("conda"):
        print_success("Conda is available")
        return True
    else:
        print_warning("Conda is not available. Will use pip for installation.")
        return True

def install_with_conda():
    """Install using conda environment."""
    print_status("Installing with conda...")
    
    # Check if environment.yml exists
    if not os.path.exists("environment.yml"):
        print_error("environment.yml not found")
        return False
    
    # Check if environment already exists
    try:
        result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, check=True)
        if "torch2.5" in result.stdout:
            print_warning("Conda environment 'torch2.5' already exists")
            response = input("Do you want to recreate it? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                print_status("Removing existing environment...")
                if not run_command("conda env remove -n torch2.5 -y", "Removing existing environment"):
                    return False
            else:
                print_status("Using existing environment")
                return True
    
    except subprocess.CalledProcessError:
        pass
    
    # Create conda environment
    if not run_command("conda env create -f environment.yml", "Creating conda environment"):
        return False
    
    print_success("Conda environment created successfully")
    return True

def install_with_pip():
    """Install using pip."""
    print_status("Installing with pip...")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print_error("requirements.txt not found")
        return False
    
    # Install PyTorch first (with CUDA if available)
    if platform.system() == "Linux" and check_command("nvidia-smi"):
        print_status("CUDA detected, installing PyTorch with CUDA support...")
        if not run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", "Installing PyTorch with CUDA"):
            return False
    else:
        print_status("Installing CPU-only PyTorch...")
        if not run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu", "Installing PyTorch CPU-only"):
            return False
    
    # Install other dependencies
    if not run_command("pip install -r requirements.txt", "Installing project dependencies"):
        return False
    
    print_success("Dependencies installed successfully")
    return True

def create_sample_dataset():
    """Create sample dataset."""
    print_status("Creating sample dataset...")
    
    if os.path.exists("examples/create_sample_dataset.py"):
        if not run_command("python examples/create_sample_dataset.py", "Creating sample dataset"):
            print_warning("Sample dataset creation failed, but continuing...")
    else:
        print_warning("Sample dataset creation script not found")

def test_installation():
    """Test the installation."""
    print_status("Testing installation...")
    
    test_code = """
import torch
import numpy as np
import matplotlib.pyplot as plt

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')

# Test basic imports
try:
    from src.models.transformer import Transformer
    from src.datasets.translation_dataset import TranslationDataset
    from src.trainers.transformer_trainer import TransformerTrainer
    print('All project modules imported successfully!')
except ImportError as e:
    print(f'Import error: {e}')
    print('Some project modules may not be available yet.')
"""
    
    if not run_command(f'python -c "{test_code}"', "Testing installation"):
        print_warning("Installation test failed, but continuing...")

def setup_git():
    """Setup Git configuration."""
    print_status("Setting up Git...")
    
    # Check if git is configured
    try:
        subprocess.run(["git", "config", "--global", "user.name"], check=True, capture_output=True)
        subprocess.run(["git", "config", "--global", "user.email"], check=True, capture_output=True)
        print_success("Git is already configured")
    except subprocess.CalledProcessError:
        print_warning("Git is not configured. Please configure it manually:")
        print("  git config --global user.name \"Your Name\"")
        print("  git config --global user.email \"your.email@example.com\"")

def main():
    """Main installation function."""
    print("🚀 Transformer Training System Installation")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        print_error("Prerequisites check failed")
        sys.exit(1)
    
    # Choose installation method
    use_conda = check_command("conda") and os.path.exists("environment.yml")
    
    if use_conda:
        print_status("Using conda for installation...")
        if not install_with_conda():
            print_error("Conda installation failed")
            sys.exit(1)
    else:
        print_status("Using pip for installation...")
        if not install_with_pip():
            print_error("Pip installation failed")
            sys.exit(1)
    
    # Create sample dataset
    create_sample_dataset()
    
    # Test installation
    test_installation()
    
    # Setup Git
    setup_git()
    
    print("\n" + "=" * 50)
    print_success("Installation completed successfully!")
    print("\nNext steps:")
    
    if use_conda:
        print("1. Activate the conda environment:")
        print("   conda activate torch2.5")
    else:
        print("1. Make sure you're in the project directory")
    
    print("2. Start training:")
    print("   python run.py --train --use-default")
    print("\n3. Create your first experiment:")
    print("   python run.py --create-experiment --name my_experiment --description \"My first experiment\"")
    print("\n4. List available experiments:")
    print("   python run.py --list-experiments")
    print("\nFor more information, see: UNIFIED_TRAINING_GUIDE.md")

if __name__ == "__main__":
    main()
