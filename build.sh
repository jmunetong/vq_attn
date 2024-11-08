#!/bin/bash
conda install -y matplotlib numpy pandas scikit-learn scikit-image
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install triton
