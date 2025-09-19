# segimage

A Python library for image segmentation and processing with comprehensive command-line interface support.

## Overview
**SegImage** is designed to handle various image formats, including MATLAB `.mat` files, and provides multiple processing algorithms for image analysis and graph generation. It features an extensible plugin architecture for easy addition of new algorithms.

## Project Details
- **Name**: segimage
- **Version**: 0.0.2
- **Author**: Lucas Lopes Felipe
- **Python Version**: ≥3.12
- **License**: Custom (see LICENSE file)

## 🖼️ **Core Features**

- **MATLAB .mat file support**: Read and process MATLAB data files with automatic data extraction
- **Multiple input formats**: `.mat`, `.npy`, `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`
- **Multiple output formats**: PNG, JPG, TIFF, NumPy arrays, and various graph formats
- **Command-line interface**: Easy-to-use CLI for batch processing
- **Extensible architecture**: Easy to add new processing methods
- **Configurable graph edge filters**: Build pixel adjacency graphs with optional LBP/gray/RGB similarity-based edge filtering
- **Pixel vs Superpixel node mode**: All graph builders (`grid`, `affinity`, `prob4`, `contrast4`) can build graphs whose nodes are either pixels or SLICO superpixels, selectable via CLI
- **Graph view renderer**: Visualize graphs with adjustable node size, edge thickness, and weight-based edge opacity

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

# Graph with edge filtering:
# - Keep edges only if neighboring pixels have similar gray levels (exact match)
segimage process input.png output_directory -t graph -f graphml --edge-filter gray --edge-similarity 1.0
# - Keep edges for similar RGB colors (allow moderate difference)
segimage process input.png output_directory -t graph -f graphml --edge-filter rgb --edge-similarity 0.6
# - Keep edges for similar LBP codes (continuous threshold)
segimage process input.png output_directory -t graph -f graphml --edge-filter lbp --edge-similarity 0.8
# - Exact LBP code equality (legacy behavior)
segimage process input.png output_directory -t graph -f graphml --edge-filter lbp_eq

# Show supported formats
segimage formats

# Show library information
segimage info

# Graph → View (pixels as nodes)
segimage process input.png output_directory -t graph_view --graph-method grid --node-mode pixel --node-radius 2

# Graph → View (superpixels as nodes)
segimage process input.png output_directory -t graph_view --graph-method grid --node-mode superpixel \
  --n-segments 300 --compactness 10 --sigma 1.0 --start-label 0 --node-radius 12 --edge-width-max 8
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

## 🔧 **Processing Types**

Currently supported processing types:

- **`mat_to_image`** (default): Convert MATLAB .mat files to standard image formats
- **`color_cluster`**: Group pixels by most frequent exact colors into up to K clusters
- **`lbp`**: Visualize 8-neighbor Local Binary Pattern values per pixel (palettes: `bw`, `rainbow`)
- **`slico`**: SLICO superpixels using scikit-image's SLIC with `slic_zero=True`
- **`graph`**: Build an 8-connected pixel adjacency graph and save to graph formats (GraphML, GML, etc.).
  - Optional edge filters: `--edge-filter {none, lbp_eq, lbp, gray, rgb}`
  - Similarity control: `--edge-similarity [0..1]` where 1.0 requires exact match, 0.0 allows any difference (no filtering). Applies to `lbp`, `gray`, `rgb` and is ignored for `lbp_eq`.

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

# Gray-level similarity filtering (exact match)
segimage process input.png output_dir -t graph -f graphml --edge-filter gray --edge-similarity 1.0

# RGB similarity filtering (looser threshold)
segimage process input.png output_dir -t graph -f graphml --edge-filter rgb --edge-similarity 0.5

# LBP similarity filtering
segimage process input.png output_dir -t graph -f graphml --edge-filter lbp --edge-similarity 0.8

# LBP exact code equality
segimage process input.png output_dir -t graph -f graphml --edge-filter lbp_eq

# Graph view (render graph overlay)
segimage process input.png output_dir -t graph_view --graph-method grid --node-mode pixel --node-radius 2
segimage process input.png output_dir -t graph_view --graph-method affinity --node-mode superpixel \
  --n-segments 250 --compactness 8 --sigma 1.0 --start-label 1 --edge-width-max 12 --edge-min 0.0
```

## 🌐 **Graph Generation & Export**

- **Graph formats**: GraphML, GML, LGL, EdgeList, Pickle
- **Edge filtering options**:
  - `gray`: Filter by gray-level similarity
  - `rgb`: Filter by RGB color similarity
  - `lbp`: Filter by LBP code similarity
  - `lbp_eq`: Exact LBP code equality
  - `none`: No filtering (all edges included)

## 🎨 **Visualization Options**

- **Color palettes**: `bw` (black/white), `rainbow` (rank-normalized)
- **Configurable parameters**:
  - Node mode: `--node-mode {pixel, superpixel}`
  - SLICO params for superpixels: `--n-segments`, `--compactness`, `--sigma`, `--start-label`
  - Edge filtering: `--edge-filter {none, lbp_eq, lbp, gray, rgb}` + `--edge-similarity [0..1]`
  - Graph view sizing: `--node-radius` (pixel radius), `--edge-width-max` (max thickness)
  - Graph view filtering: `--edge-min` (minimum weight to draw)

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

## 🏗️ **Architecture**

### **Core Components**
```
src/segimage/
├── processor.py          # Main processing logic and router
├── utils.py             # Utility functions and helpers
├── cli/                 # Command-line interface
│   ├── main.py         # Click group and shared options
│   └── commands/       # Subcommands implementation
│       ├── process.py  # Main processing command
│       ├── formats.py  # Supported formats listing
│       ├── info.py     # Library information
│       └── inspect.py  # File inspection
└── processors/          # Pluggable processing algorithms
    ├── color_cluster.py # Color clustering implementation
    ├── lbp.py          # Local Binary Pattern processor
    ├── slico.py        # SLICO superpixels
    └── graph.py        # Graph generation and export
```

### **Design Patterns**
- **Plugin Architecture**: Extensible processor system for easy addition of new algorithms
- **Command Pattern**: CLI commands are modular and self-contained
- **Strategy Pattern**: Different processing strategies can be swapped easily

## 📦 **Dependencies**

### **Core Dependencies**
- `scipy≥1.7.0`: Scientific computing and MATLAB file support
- `click≥8.0.0`: Command-line interface framework
- `Pillow≥8.0.0`: Image processing and manipulation
- `scikit-image≥0.20.0`: Advanced image processing algorithms
- `hedonic≥0.0.7`: Additional utilities

### **Development Dependencies**
- `pytest≥6.0.0`: Testing framework
- `black≥21.0.0`: Code formatting
- `flake8≥3.8.0`: Linting and style checking

## 🚀 **Development**

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

### Lint code
```bash
uv run flake8 src/
# or
flake8 src/
```

### CLI Development
```bash
# Run CLI commands
uv run segimage --help
uv run segimage process --help
```

## 🎯 **Use Cases**

### **Academic Research**
- MATLAB data analysis and visualization
- Image segmentation algorithm development
- Graph-based image analysis
- Texture analysis with LBP

### **Data Science**
- Batch image processing
- Image format conversion
- Superpixel generation for ML preprocessing
- Graph representation of images

### **Image Analysis**
- Color-based segmentation
- Texture analysis
- Boundary detection
- Region-based analysis

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

## Project Status
- **Current Version**: 0.0.2
- **Development Stage**: Active development
- **Python Support**: Modern Python (3.12+)
- **Package Manager**: uv (recommended), pip supported

## Future Enhancements
- Additional segmentation algorithms
- More graph export formats
- Performance optimizations
- Extended CLI options
- Additional visualization palettes

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
