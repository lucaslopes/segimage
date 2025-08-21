#!/usr/bin/env python3
"""
Basic usage example for segimage library.

This script demonstrates how to use the ImageProcessor class
for common tasks like converting MATLAB .mat files, running SLICO superpixels,
LBP visualization, color clustering, and building pixel graphs.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from segimage import ImageProcessor


def main():
    """Main example function."""
    print("segimage - Basic Usage Example")
    print("=" * 40)
    
    # Initialize the processor
    processor = ImageProcessor()
    
    # Show supported formats
    print("\nSupported formats:")
    formats = processor.get_supported_formats()
    print(f"  Input:  {', '.join(formats['input'])}")
    print(f"  Output: {', '.join(formats['output'])}")
    
    # Example 1: Convert a MATLAB file (if it exists)
    input_mat = Path("data/2018.mat")
    output_png = Path("output/2018_processed.png")
    
    if input_mat.exists():
        print(f"\nProcessing {input_mat} (MAT to image)...")
        output_png.parent.mkdir(parents=True, exist_ok=True)
        success = processor.process_mat_to_image(input_mat, output_png, ".png")
        print("✅ Successfully converted to:" if success else "❌ Conversion failed", output_png if success else "")
    else:
        print(f"\nInput file {input_mat} not found.")
        print("You can use the command-line interface:")
        print("  segimage process input.mat output_dir --process-type mat_to_image")
        print("  segimage process input.mat output_dir -t mat_to_image -f png")
        print("  segimage process input.mat output_dir -t mat_to_image -f jpg")
    
    # Example 2: Run SLICO superpixels on a regular image (if it exists)
    input_img = Path("data/example.png")
    output_slico = Path("output/example_slico.png")
    
    if input_img.exists():
        print(f"\nProcessing {input_img} (SLICO superpixels)...")
        output_slico.parent.mkdir(parents=True, exist_ok=True)
        success = processor.process_image(
            input_img,
            output_slico,
            "slico",
            n_segments=280,
            compactness=2.0,
            sigma=1.0,
            start_label=1,
        )
        print("✅ SLICO result saved to:" if success else "❌ SLICO failed", output_slico if success else "")
    else:
        print(f"\nInput image {input_img} not found.")
        print("You can run SLICO via CLI:")
        print("  segimage process input.png output_dir -t slico")
        print("  segimage process input.png output_dir -t slico --n-segments 500 --compactness 10 --sigma 1 --start-label 1")

    # Example 3: LBP visualization
    output_lbp = Path("output/example_lbp.png")
    if input_img.exists():
        print(f"\nProcessing {input_img} (LBP visualization)...")
        output_lbp.parent.mkdir(parents=True, exist_ok=True)
        success = processor.process_image(
            input_img,
            output_lbp,
            "lbp",
            palette="bw",
        )
        print("✅ LBP saved to:" if success else "❌ LBP failed", output_lbp if success else "")
    else:
        print("\nYou can run LBP via CLI:")
        print("  segimage process input.png output_dir -t lbp --palette bw")

    # Example 4: Color clustering
    output_cluster = Path("output/example_cluster.png")
    if input_img.exists():
        print(f"\nProcessing {input_img} (color clustering)...")
        output_cluster.parent.mkdir(parents=True, exist_ok=True)
        success = processor.process_image(
            input_img,
            output_cluster,
            "color_cluster",
            K=3,
            palette="rainbow",
        )
        print("✅ Clusters saved to:" if success else "❌ Clustering failed", output_cluster if success else "")
    else:
        print("\nYou can run color clustering via CLI:")
        print("  segimage process input.png output_dir -t color_cluster -K 3 --palette rainbow")

    # Example 5: Pixel graph (GraphML)
    output_graph = Path("output/example_graph.graphml")
    if input_img.exists():
        print(f"\nProcessing {input_img} (graph build)...")
        output_graph.parent.mkdir(parents=True, exist_ok=True)
        success = processor.process_image(
            input_img,
            output_graph,
            "graph",
        )
        print("✅ Graph saved to:" if success else "❌ Graph build failed", output_graph if success else "")
    else:
        print("\nYou can build the pixel graph via CLI:")
        print("  segimage process input.png output_dir -t graph -f graphml")


if __name__ == "__main__":
    main()
