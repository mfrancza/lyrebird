#!/bin/bash
# Lyrebird Raspberry Pi Installation Script
# Run as: sudo ./install.sh

set -e

echo "=========================================="
echo "Lyrebird Raspberry Pi Installation"
echo "=========================================="
echo

# Check if running as root for system config
if [ "$EUID" -ne 0 ]; then
    echo "Note: Run with sudo for system-wide configuration"
    echo "Proceeding with user-level installation..."
    SYSTEM_INSTALL=false
else
    SYSTEM_INSTALL=true
fi

# Detect architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Project directory: $PROJECT_DIR"
echo

# Step 1: System packages
echo "Step 1: Installing system packages..."
if [ "$SYSTEM_INSTALL" = true ]; then
    apt-get update
    apt-get install -y \
        python3-pip \
        python3-venv \
        python3-dev \
        libasound2-dev \
        libportaudio2 \
        portaudio19-dev \
        libsndfile1 \
        git
else
    echo "Skipping system packages (run with sudo to install)"
fi
echo

# Step 2: Create virtual environment
echo "Step 2: Setting up Python virtual environment..."
VENV_DIR="$PROJECT_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Created virtual environment at $VENV_DIR"
else
    echo "Virtual environment already exists"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo

# Step 3: Install Python packages
echo "Step 3: Installing Python packages..."

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (ARM-specific)
if [[ "$ARCH" == "aarch64" ]]; then
    echo "Installing PyTorch for ARM64..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
elif [[ "$ARCH" == "armv7l" ]]; then
    echo "Installing PyTorch for ARM32..."
    echo "Note: ARM32 support is limited, consider using 64-bit OS"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "Installing PyTorch for $ARCH..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
pip install -r "$SCRIPT_DIR/requirements-pi.txt"
echo

# Step 4: Verify installation
echo "Step 4: Verifying installation..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
python3 -c "import torchaudio; print(f'TorchAudio {torchaudio.__version__}')"
python3 -c "import sounddevice; print(f'sounddevice {sounddevice.__version__}')"
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"

# Check ONNX Runtime (optional)
python3 -c "import onnxruntime; print(f'ONNX Runtime {onnxruntime.__version__}')" 2>/dev/null || echo "ONNX Runtime not installed (optional)"
echo

# Step 5: Audio configuration
echo "Step 5: Audio configuration..."
if [ "$SYSTEM_INSTALL" = true ]; then
    # Add user to audio group
    usermod -a -G audio pi 2>/dev/null || true

    echo "Checking audio devices..."
    aplay -l 2>/dev/null || echo "No playback devices found"
    arecord -l 2>/dev/null || echo "No capture devices found"

    echo
    echo "To configure your audio HAT, add the appropriate overlay to /boot/config.txt"
    echo "See pi/config/config.txt.example for examples"
else
    echo "Skipping audio configuration (run with sudo)"
fi
echo

# Step 6: Optional systemd service
if [ "$SYSTEM_INSTALL" = true ]; then
    echo "Step 6: Setting up systemd service (optional)..."
    read -p "Install systemd service for auto-start? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        SERVICE_FILE="$SCRIPT_DIR/config/lyrebird.service"
        if [ -f "$SERVICE_FILE" ]; then
            # Update service file paths (using # as delimiter for paths with special chars)
            sed -e "s#/home/pi/lyrebird#$PROJECT_DIR#g" \
                "$SERVICE_FILE" > /etc/systemd/system/lyrebird.service

            systemctl daemon-reload
            echo "Service installed. Enable with: sudo systemctl enable lyrebird"
            echo "Start with: sudo systemctl start lyrebird"
        else
            echo "Error: systemd service template not found at: $SERVICE_FILE"
            echo "Skipping systemd service installation."
        fi
    fi
else
    echo "Step 6: Skipping systemd setup (run with sudo)"
fi
echo

# Step 7: Performance tuning tips
echo "=========================================="
echo "Post-Installation Tips"
echo "=========================================="
echo
echo "1. Configure audio HAT in /boot/config.txt"
echo "   See: $SCRIPT_DIR/config/config.txt.example"
echo
echo "2. For lowest latency, add kernel parameters to /boot/cmdline.txt:"
echo "   isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3"
echo
echo "3. Test audio passthrough:"
echo "   source $VENV_DIR/bin/activate"
echo "   python $SCRIPT_DIR/realtime_processor.py --list-devices"
echo
echo "4. Run benchmarks to verify performance:"
echo "   python $SCRIPT_DIR/tools/benchmark_pi.py"
echo
echo "5. Copy a trained model to models/ and run:"
echo "   python $SCRIPT_DIR/realtime_processor.py --model models/small/model.pth"
echo
echo "Installation complete!"
