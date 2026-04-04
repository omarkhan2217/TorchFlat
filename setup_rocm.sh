#!/bin/bash
# TorchFlat ROCm setup helper for Linux
# Run: source setup_rocm.sh

echo "TorchFlat ROCm Setup"
echo "===================="
echo

# Auto-detect ROCm
if [ -d "/opt/rocm" ]; then
    export ROCM_HOME="/opt/rocm"
    echo "Found ROCm: $ROCM_HOME"
elif [ -n "$ROCM_HOME" ] && [ -d "$ROCM_HOME" ]; then
    echo "Using ROCM_HOME: $ROCM_HOME"
else
    echo "ERROR: ROCm not found at /opt/rocm and ROCM_HOME not set."
    echo "Install ROCm: https://rocm.docs.amd.com/projects/install-on-linux/"
    return 1 2>/dev/null || exit 1
fi

# Check for hipcc
if command -v hipcc &>/dev/null; then
    echo "hipcc found: $(which hipcc)"
elif [ -x "$ROCM_HOME/bin/hipcc" ]; then
    export PATH="$ROCM_HOME/bin:$PATH"
    echo "Added $ROCM_HOME/bin to PATH"
else
    echo "WARNING: hipcc not found. Kernel compilation may use pip ROCm SDK instead."
fi

# Check for C++ compiler
if command -v g++ &>/dev/null; then
    echo "g++ found: $(g++ --version | head -1)"
elif command -v clang++ &>/dev/null; then
    echo "clang++ found: $(clang++ --version | head -1)"
else
    echo "ERROR: No C++ compiler found."
    echo "  Ubuntu/Debian: sudo apt install build-essential"
    echo "  Fedora: sudo dnf install gcc-c++"
    return 1 2>/dev/null || exit 1
fi

# Test kernel
echo
echo "Testing kernel compilation..."
python -c "from torchflat._kernel_loader import _get_umi_kernel; k = _get_umi_kernel(); print('Kernel:', 'LOADED' if k else 'FAILED')"

echo
echo "Setup complete! Environment:"
echo "  ROCM_HOME=$ROCM_HOME"
echo
echo "To make permanent, add to ~/.bashrc:"
echo "  export ROCM_HOME=$ROCM_HOME"
