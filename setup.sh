#!/bin/bash
set -e

# Transformer Training System Setup Script
# This script sets up the environment for the Transformer training system

echo "🚀 Setting up Transformer Training System..."
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is installed
check_conda() {
    if command -v conda &> /dev/null; then
        print_success "Conda is installed"
        return 0
    else
        print_error "Conda is not installed. Please install Anaconda or Miniconda first."
        print_status "Visit: https://docs.conda.io/en/latest/miniconda.html"
        return 1
    fi
}

# Check if Python is available
check_python() {
    if command -v python &> /dev/null; then
        print_success "Python is available"
        return 0
    else
        print_error "Python is not available"
        return 1
    fi
}

# Create conda environment
create_conda_env() {
    local env_name=${1:-"torch2.5"}
    local python_version=${2:-"3.9"}
    
    print_status "Creating conda environment: $env_name"
    
    if conda env list | grep -q "^$env_name "; then
        print_warning "Environment '$env_name' already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Removing existing environment..."
            conda env remove -n "$env_name" -y
        else
            print_status "Using existing environment: $env_name"
            return 0
        fi
    fi
    
    print_status "Creating new conda environment with Python $python_version..."
    conda create -n "$env_name" python="$python_version" -y
    
    print_success "Conda environment '$env_name' created successfully"
}

# Install PyTorch
install_pytorch() {
    local env_name=${1:-"torch2.5"}
    
    print_status "Installing PyTorch in environment: $env_name"
    
    # Activate environment and install PyTorch
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$env_name"
    
    # Install PyTorch with CUDA support (if available)
    if command -v nvidia-smi &> /dev/null; then
        print_status "CUDA detected, installing PyTorch with CUDA support..."
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        print_status "No CUDA detected, installing CPU-only PyTorch..."
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    fi
    
    print_success "PyTorch installed successfully"
}

# Install project dependencies
install_dependencies() {
    local env_name=${1:-"torch2.5"}
    
    print_status "Installing project dependencies..."
    
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$env_name"
    
    # Install from requirements.txt
    if [ -f "requirements.txt" ]; then
        print_status "Installing dependencies from requirements.txt..."
        pip install -r requirements.txt
        print_success "Dependencies installed successfully"
    else
        print_error "requirements.txt not found"
        return 1
    fi
}

# Create sample dataset
create_sample_dataset() {
    local env_name=${1:-"torch2.5"}
    
    print_status "Creating sample dataset..."
    
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$env_name"
    
    # Run the sample dataset creation script
    if [ -f "examples/create_sample_dataset.py" ]; then
        python examples/create_sample_dataset.py
        print_success "Sample dataset created successfully"
    else
        print_warning "Sample dataset creation script not found"
    fi
}

# Test installation
test_installation() {
    local env_name=${1:-"torch2.5"}
    
    print_status "Testing installation..."
    
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$env_name"
    
    # Test basic imports
    python -c "
import torch
import numpy as np
import matplotlib.pyplot as plt
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
print('All imports successful!')
"
    
    print_success "Installation test completed successfully"
}

# Setup Git (if not already configured)
setup_git() {
    print_status "Setting up Git configuration..."
    
    # Check if git is configured
    if ! git config --global user.name &> /dev/null; then
        print_warning "Git user.name not configured"
        read -p "Enter your Git username: " git_username
        git config --global user.name "$git_username"
    fi
    
    if ! git config --global user.email &> /dev/null; then
        print_warning "Git user.email not configured"
        read -p "Enter your Git email: " git_email
        git config --global user.email "$git_email"
    fi
    
    print_success "Git configuration completed"
}

# Main setup function
main() {
    local env_name="torch2.5"
    local python_version="3.9"
    
    echo "Transformer Training System Setup"
    echo "================================="
    echo
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env-name)
                env_name="$2"
                shift 2
                ;;
            --python-version)
                python_version="$2"
                shift 2
                ;;
            --skip-conda)
                skip_conda=true
                shift
                ;;
            --skip-deps)
                skip_deps=true
                shift
                ;;
            --skip-dataset)
                skip_dataset=true
                shift
                ;;
            --skip-test)
                skip_test=true
                shift
                ;;
            --skip-git)
                skip_git=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo
                echo "Options:"
                echo "  --env-name NAME        Conda environment name (default: torch2.5)"
                echo "  --python-version VER   Python version (default: 3.9)"
                echo "  --skip-conda          Skip conda environment creation"
                echo "  --skip-deps           Skip dependency installation"
                echo "  --skip-dataset        Skip sample dataset creation"
                echo "  --skip-test           Skip installation test"
                echo "  --skip-git            Skip Git configuration"
                echo "  -h, --help            Show this help message"
                echo
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    print_status "Starting setup with environment: $env_name"
    print_status "Python version: $python_version"
    echo
    
    # Check prerequisites
    if [ "${skip_conda:-false}" != "true" ]; then
        if ! check_conda; then
            exit 1
        fi
        check_python
    fi
    
    # Create conda environment
    if [ "${skip_conda:-false}" != "true" ]; then
        create_conda_env "$env_name" "$python_version"
        install_pytorch "$env_name"
    fi
    
    # Install dependencies
    if [ "${skip_deps:-false}" != "true" ]; then
        install_dependencies "$env_name"
    fi
    
    # Create sample dataset
    if [ "${skip_dataset:-false}" != "true" ]; then
        create_sample_dataset "$env_name"
    fi
    
    # Test installation
    if [ "${skip_test:-false}" != "true" ]; then
        test_installation "$env_name"
    fi
    
    # Setup Git
    if [ "${skip_git:-false}" != "true" ]; then
        setup_git
    fi
    
    echo
    print_success "Setup completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Activate the conda environment:"
    echo "   conda activate $env_name"
    echo
    echo "2. Start training:"
    echo "   python run.py --train --use-default"
    echo
    echo "3. Create your first experiment:"
    echo "   python run.py --create-experiment --name my_experiment --description \"My first experiment\""
    echo
    echo "4. List available experiments:"
    echo "   python run.py --list-experiments"
    echo
    echo "For more information, see: UNIFIED_TRAINING_GUIDE.md"
}

# Run main function
main "$@"
