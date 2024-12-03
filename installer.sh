#!/bin/bash
conda create -n torch python=3.10 -y
conda activate torch
# Print a message to indicate the installation process has started
echo "Starting installation of required packages..."

# Step 1: Install PyTorch and related packages using pip
echo "Installing PyTorch, torchvision, and torchaudio..."
pip3 install torch torchvision torchaudio

# Step 2: Install additional packages using conda
echo "Installing einops, matplotlib, scipy, pandas, ninja, and pyyaml using conda..."
conda install einops matplotlib scipy pandas ninja pyyaml -y

# Step 3: Install flash-attn using pip
echo "Installing flash-attn with no-build-isolation..."
pip install flash-attn --no-build-isolation

# Print a message to indicate the installation is complete
echo "Installation of required packages is complete!"

