# segimage

A Python library for image segmentation and processing with command-line interface support.

## Features

- **MATLAB .mat file support**: Read and process MATLAB data files
- **Multiple output formats**: Convert to standard image formats (PNG, JPG, TIFF)
- **Command-line interface**: Easy-to-use CLI for batch processing
- **Extensible architecture**: Easy to add new processing methods

## Installation

### From PyPI (when published)
```bash
pip install segimage
```

### From source
```bash
git clone https://github.com/yourusername/segimage.git
cd segimage
pip install -e .
```

## Quick Start

### Command Line Usage

The library provides a command-line interface that can be used directly:

```bash
# Convert a MATLAB .mat file to PNG format (default)
segimage process input.mat output_directory --process-type mat_to_image

# Convert to JPG format
segimage process input.mat output_directory -t mat_to_image -f jpg

# With verbose output
segimage process input.mat output_directory -t mat_to_image -f png -v

# Show supported formats
segimage formats

# Show library information
segimage info
```

### Python API Usage

```python
from pathlib import Path
from segimage import ImageProcessor

# Initialize the processor
processor = ImageProcessor()

success = processor.process_image(
    Path("input.png"),
    Path("out/input_clustered.png"),
    "color_cluster",
    K=4,
    palette="rainbow",
)

if success:
    print("Conversion successful!")
else:
    print("Conversion failed!")
```

## Supported Formats

### Input Formats
- `.mat` - MATLAB data files
- `.npy` - NumPy array files
- `.tif`, `.tiff` - TIFF images
- `.png`, `.jpg`, `.jpeg` - Common image formats

### Output Formats
- `.png` - PNG images (default, lossless)
- `.jpg`, `.jpeg` - JPEG images (compressed)
- `.tif`, `.tiff` - TIFF images
- `.npy` - NumPy array files

## Processing Types

Currently supported processing types:

- **`mat_to_image`** (default): Convert MATLAB .mat files to standard image formats
- **`color_cluster`**: Group pixels by most frequent exact colors into up to K clusters
- **`slico`**: SLICO superpixels using scikit-image's SLIC with `slic_zero=True`

### SLICO usage examples

```bash
# Run SLICO with defaults
segimage process input.png output_dir -t slico

# Customize superpixel parameters
segimage process input.png output_dir -t slico --n-segments 500 --compactness 10 --sigma 1 --start-label 1
```

Python API:

```python
from pathlib import Path
from segimage import ImageProcessor

processor = ImageProcessor()
processor.process_image(
    Path("input.png"),
    Path("out/input_slico.png"),
    "slico",
    n_segments=280,
    compactness=2.0,
    sigma=1.0,
    start_label=1,
)
```

## Examples

### Basic MATLAB to PNG conversion
```bash
segimage process data/2018.mat output/ --process-type mat_to_image
```

### Convert to JPG format
```bash
segimage process input.mat output/ -t mat_to_image -f jpg
```

### Convert to TIFF format
```bash
segimage process input.mat output/ -t mat_to_image -f tif
```

### Verbose processing
```bash
segimage process input.mat output/ -t mat_to_image -f png -v
```

## How It Works

The library automatically:
1. **Reads MATLAB .mat files** and extracts numeric data
2. **Handles complex data structures** including object arrays and structured arrays
3. **Normalizes data** to appropriate ranges for image formats
4. **Converts to PIL Image objects** for proper image processing
5. **Saves in standard formats** that macOS and other systems recognize as images
6. **Preserves metadata** in companion .meta files

## Development

### Setup development environment
```bash
pip install -e ".[dev]"
```

### Run tests
```bash
pytest
```

### Code formatting
```bash
black src/
```

## Project Structure

```
segimage/
├── src/
│   └── segimage/
│       ├── __init__.py      # Main package exports
│       ├── processor.py     # Core image processing logic
│       └── cli.py          # Command-line interface
├── pyproject.toml          # Project configuration
└── README.md              # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the terms specified in the LICENSE file.

## Support

For issues and questions, please use the GitHub issue tracker.
