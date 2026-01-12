#!/bin/bash

# RPL (RPI Learn) Installation Script
# This script builds the C library and sets up the Python environment.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== RPL (RPI Learn) Installer ===${NC}"

# Check for dependencies
echo -e "\n${BLUE}[1/4] Checking dependencies...${NC}"
if [ -f /etc/debian_version ]; then
    echo "Detected Debian/Ubuntu system. Checking for required packages..."
    REQUIRED_PACKAGES="cmake build-essential python3-dev python3-numpy libopenblas-dev"
    MISSING_PACKAGES=""

    for pkg in $REQUIRED_PACKAGES; do
        if ! dpkg -l | grep -q "^ii  $pkg "; then
            MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
        fi
    done

    if [ -n "$MISSING_PACKAGES" ]; then
        echo -e "${RED}Missing packages:${MISSING_PACKAGES}${NC}"
        echo "Attempting to install missing packages (requires sudo)..."
        sudo apt update
        sudo apt install -y $MISSING_PACKAGES
    else
        echo "All system dependencies found."
    fi
else
    echo "Warning: Non-Debian system detected. Please ensure you have cmake, gcc, python3-dev, numpy, and OpenBLAS installed."
fi

# Build the C library
echo -e "\n${BLUE}[2/4] Building C core...${NC}"
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..

# Install Python package
echo -e "\n${BLUE}[3/4] Installing Python bindings...${NC}"
if [ -f setup.py ]; then
    pip3 install -e .
else
    echo "No setup.py found. Creating a symbolic link for development..."
    # Note: The library currently looks for the .so in ../../build/
    # This is handled by core.py
    echo "RPL is now available for import if you add $(pwd)/python to your PYTHONPATH."
    echo "Example: export PYTHONPATH=\$PYTHONPATH:$(pwd)/python"
fi

# Run a quick test
echo -e "\n${BLUE}[4/4] Verifying installation...${NC}"
if [ -f build/tests/test_core ]; then
    echo "Running C core tests..."
    ./build/tests/test_core || echo -e "${RED}C tests failed!${NC}"
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)/python
python3 -c "import rpl; print('RPL Python API loaded successfully!'); print('Version: 0.1.0')"

echo -e "\n${GREEN}=== Installation Complete! ===${NC}"
echo "To use RPL in your projects, add this to your .bashrc:"
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)/python"
