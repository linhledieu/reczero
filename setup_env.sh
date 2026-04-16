#!/bin/bash
set -e

echo "==== Activating environment ===="
conda activate /workspace/py39 || {
    echo "Creating conda env at /workspace/py39"
    conda create -p /workspace/py39 python=3.9 -y
    conda activate /workspace/py39
}

echo "==== Upgrading pip tools ===="
pip install --upgrade pip setuptools wheel

echo "==== Installing PyTorch ===="
pip install torch torchvision torchaudio

echo "==== Installing base requirements ===="
pip install -r /workspace/RecZero-main/requirements.txt

echo "==== Installing flash-attn separately ===="
pip install ninja packaging
pip install flash-attn --no-build-isolation || {
    echo "flash-attn failed, continuing without it..."
}

echo "==== Setup complete ===="