#!/bin/bash
# ============================================================
# FastFit Setup Script for g4dn.xlarge (T4 16GB)
# 
# This script sets up the FastFit footwear/bags try-on worker
# on a dedicated EC2 instance.
#
# Usage: bash setup_fastfit.sh
# ============================================================

set -e

WORKING_DIR="/srv/VTON/virtual-tryon"
FASTFIT_DIR="/srv/FastFit"
VENV_PATH="/srv/VTON/virtual-tryon/vton_venv"

echo "============================================"
echo "  FastFit Footwear Try-On Setup"
echo "============================================"

# Step 1: Clone FastFit repository
echo ""
echo "[1/5] Cloning FastFit repository..."
if [ ! -d "$FASTFIT_DIR" ]; then
    git clone https://github.com/Zheng-Chong/FastFit.git "$FASTFIT_DIR"
    echo "  ✓ FastFit cloned to $FASTFIT_DIR"
else
    echo "  ✓ FastFit already exists at $FASTFIT_DIR"
fi

# Step 2: Create virtual environment
echo ""
echo "[2/5] Setting up Python environment..."
if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment already exists"
fi

source "$VENV_PATH/bin/activate"

# Step 3: Install dependencies
echo ""
echo "[3/5] Installing dependencies..."
pip install --upgrade pip
pip install -r "$WORKING_DIR/requirements.txt"
pip install -r "$FASTFIT_DIR/requirements.txt"
pip install easy-dwpose --no-dependencies

echo "  ✓ Dependencies installed"

# Step 4: Add FastFit env variables to .env
echo ""
echo "[4/5] Updating .env configuration..."

# Check if fastfit vars already exist in .env
if ! grep -q "fastfit_repo_path" "$WORKING_DIR/.env" 2>/dev/null; then
    cat >> "$WORKING_DIR/.env" << 'FASTFIT_ENV'

# FastFit Configuration
fastfit_repo_path="/srv/FastFit"
fastfit_model_id="zhengchong/FastFit-SR-1024"
fastfit_device="cuda"
fastfit_num_steps=50
fastfit_guidance_scale=2.5
FASTFIT_ENV
    echo "  ✓ FastFit env variables added to .env"
else
    echo "  ✓ FastFit env variables already in .env"
fi

# Step 5: Create systemd service
echo ""
echo "[5/5] Creating systemd service..."

SERVICE_FILE="/etc/systemd/system/fastfit_poller.service"

sudo bash -c "cat > $SERVICE_FILE << EOF
[Unit]
Description=FastFit Footwear Try-On Poller Service
After=network.target

[Service]
Type=idle
WorkingDirectory=$WORKING_DIR

Environment='PATH=$VENV_PATH/bin'

ExecStart=$VENV_PATH/bin/python3 $WORKING_DIR/main_fastfit.py

Restart=always
RestartSec=10
StartLimitInterval=0

StandardOutput=append:$WORKING_DIR/fastfit_poller_logs.txt
StandardError=append:$WORKING_DIR/fastfit_poller_logs.txt

[Install]
WantedBy=multi-user.target
EOF"

sudo systemctl daemon-reload
sudo systemctl enable fastfit_poller.service
sudo systemctl restart fastfit_poller.service

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "  Service: fastfit_poller.service"
echo "  Status:  sudo systemctl status fastfit_poller.service"
echo "  Logs:    tail -f $WORKING_DIR/fastfit_poller_logs.txt"
echo "  Stop:    sudo systemctl stop fastfit_poller.service"
echo ""
echo "  The model weights will auto-download from HuggingFace"
echo "  on first inference (~2-4GB download)."
echo ""

sudo systemctl status fastfit_poller.service
