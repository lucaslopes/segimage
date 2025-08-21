# segimage

A Python library for image segmentation and processing with command-line interface support.

## Features

- **MATLAB .mat file support**: Read and process MATLAB data files
- **Multiple output formats**: Convert to standard image formats (PNG, JPG, TIFF) and graph formats (GraphML, GML, etc.)
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
# With uv (recommended for devs)
uv pip install -e '.[dev]'
# or using pip
pip install -e '.[dev]'
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

# Color clustering (top-K frequent colors)
segimage process input.png output_directory -t color_cluster -K 4 --palette rainbow

# LBP visualization (8-neighbor local binary pattern)
segimage process input.png output_directory -t lbp --palette bw

# SLICO superpixels
segimage process input.png output_directory -t slico --n-segments 300 --compactness 10

# Create a pixel adjacency graph (8-connected) and save as GraphML
segimage process input.png output_directory -t graph -f graphml

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
- Graphs: `.graphml`, `.gml`, `.lg`/`.lgl`, `.edgelist`/`.edges`/`.txt`, `.pickle`/`.pkl`
  - Note: Companion `.meta` files are only written for image outputs

## Processing Types

Currently supported processing types:

- **`mat_to_image`** (default): Convert MATLAB .mat files to standard image formats
- **`color_cluster`**: Group pixels by most frequent exact colors into up to K clusters
- **`lbp`**: Visualize 8-neighbor Local Binary Pattern values per pixel (palettes: `bw`, `rainbow`)
- **`slico`**: SLICO superpixels using scikit-image's SLIC with `slic_zero=True`
- **`graph`**: Build an 8-connected pixel adjacency graph and save to graph formats (GraphML, GML, etc.)

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

### LBP usage examples

```bash
# Black-and-white palette
segimage process input.png output_dir -t lbp --palette bw

# Rainbow palette (rank-normalized)
segimage process input.png output_dir -t lbp --palette rainbow
```

### Color clustering examples

```bash
# Cluster by top-3 most frequent colors (two top colors + remaining)
segimage process input.png output_dir -t color_cluster -K 3 --palette bw

# Rainbow palette for clusters
segimage process input.png output_dir -t color_cluster -K 5 --palette rainbow
```

### Graph creation examples

```bash
# Create 8-neighbor pixel graph and save as GraphML
segimage process input.png output_dir -t graph -f graphml

# Save as GML instead
segimage process input.png output_dir -t graph -f gml
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
# Recommended: uv (fast installer and runner)
uv pip install -e '.[dev]'

# Or using pip
pip install -e '.[dev]'
```

### Run tests
```bash
uv run -m pytest -q
# or
pytest -q
```

### Code formatting
```bash
uv run black src/
# or
black src/
```

## Project Structure

```
segimage/
├── src/
│   └── segimage/
│       ├── __init__.py      # Main package exports
│       ├── processor.py     # Core image processing logic and router
│       ├── cli/             # CLI entrypoint and commands
│       │   ├── main.py      # Click group and shared options
│       │   └── commands/    # Subcommands: process, formats, inspect, info
│       └── processors/      # Pluggable processors (color_cluster, lbp, slico, graph)
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
