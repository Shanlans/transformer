#!/bin/bash
# Script to activate the torch2.5 conda environment

echo "Activating torch2.5 conda environment..."
source /Users/shanlanshen/opt/anaconda3/etc/profile.d/conda.sh
conda activate torch2.5

echo "Environment activated! PyTorch version:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

echo ""
echo "To use this environment in your current shell, run:"
echo "source /Users/shanlanshen/cursorproject/transformer/activate_torch2.5.sh"
echo ""
echo "Or manually activate with:"
echo "conda activate torch2.5"

