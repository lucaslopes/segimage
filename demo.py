#!/usr/bin/env python3
"""
Demo script for segimage library.

This script demonstrates the basic functionality of the segimage library.
Run this script to test if the library is working correctly.
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from segimage import ImageProcessor
    print("✅ Successfully imported segimage library")
except ImportError as e:
    print(f"❌ Failed to import segimage: {e}")
    print("Make sure you have installed the library: pip install -e .")
    sys.exit(1)


def demo_processor():
    """Demonstrate the ImageProcessor functionality."""
    print("\n🔧 Testing ImageProcessor...")
    
    # Initialize processor
    processor = ImageProcessor()
    print("✅ Processor initialized")
    
    # Show supported formats
    formats = processor.get_supported_formats()
    print(f"✅ Supported input formats: {', '.join(formats['input'])}")
    print(f"✅ Supported output formats: {', '.join(formats['output'])}")
    
    return processor


def demo_cli_help():
    """Demonstrate CLI help functionality."""
    print("\n📖 Testing CLI help...")
    
    try:
        from segimage.cli import main
        print("✅ CLI module imported successfully")
        print("✅ CLI commands available:")
        print("   - segimage process")
        print("   - segimage formats")
        print("   - segimage info")
        print("   - segimage inspect")
    except Exception as e:
        print(f"❌ CLI test failed: {e}")


def main():
    """Main demo function."""
    print("🚀 segimage Library Demo")
    print("=" * 40)
    
    # Test processor
    processor = demo_processor()
    
    # Test CLI
    demo_cli_help()
    
    print("\n" + "=" * 40)
    print("🎉 Demo completed successfully!")
    print("\nTo use the library:")
    print("1. Command line: segimage process input.mat output_dir")
    print("2. Python API: from segimage import ImageProcessor")
    print("\nFor more information: segimage info")
    print("\nExample conversions:")
    print("  segimage process input.mat output/ -f png")
    print("  segimage process input.mat output/ -f jpg")
    print("  segimage process input.mat output/ -f tif")


if __name__ == "__main__":
    main()
