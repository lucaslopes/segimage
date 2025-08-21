#!/bin/bash

# segimage installation script
echo "Installing segimage library..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "Python version: $python_version"

# Install dependencies
echo "Installing dependencies..."
pip3 install scipy numpy click Pillow

# Install the library in development mode
echo "Installing segimage library..."
pip3 install -e .

echo ""
echo "✅ Installation complete!"
echo ""
echo "Usage examples:"
echo "  segimage process input.mat output_dir --process-type mat_to_image"
echo "  segimage process input.mat output_dir -f png"
echo "  segimage process input.mat output_dir -f jpg"
echo "  segimage formats"
echo "  segimage info"
echo ""
echo "For more information, run: segimage info"
