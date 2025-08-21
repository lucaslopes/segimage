"""
Tests for the ImageProcessor class.
"""

import pytest
import numpy as np
from pathlib import Path
from segimage.processor import ImageProcessor


class TestImageProcessor:
    """Test cases for ImageProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ImageProcessor()
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up test files
        if self.test_dir.exists():
            for file in self.test_dir.iterdir():
                file.unlink()
            self.test_dir.rmdir()
    
    def test_processor_initialization(self):
        """Test that processor initializes correctly."""
        assert self.processor is not None
        assert hasattr(self.processor, 'supported_input_formats')
        assert hasattr(self.processor, 'supported_output_formats')
    
    def test_supported_formats(self):
        """Test that supported formats are returned correctly."""
        formats = self.processor.get_supported_formats()
        assert 'input' in formats
        assert 'output' in formats
        assert '.mat' in formats['input']
        assert '.png' in formats['output']
        assert '.jpg' in formats['output']
    
    def test_process_image_invalid_type(self):
        """Test that invalid process type returns False."""
        result = self.processor.process_image("dummy.mat", "dummy.png", "invalid_type")
        assert result is False
    
    def test_process_mat_to_image_nonexistent_file(self):
        """Test that processing non-existent file returns False."""
        result = self.processor.process_mat_to_image("nonexistent.mat", "output.png")
        assert result is False
    
    def test_process_mat_to_image_invalid_extension(self):
        """Test that processing file with wrong extension returns False."""
        # Create a dummy file with wrong extension
        dummy_file = self.test_dir / "dummy.txt"
        dummy_file.write_text("dummy content")
        
        result = self.processor.process_mat_to_image(dummy_file, "output.png")
        assert result is False
    
    def test_supported_output_formats(self):
        """Test that output formats are correctly configured."""
        formats = self.processor.get_supported_formats()
        assert '.png' in formats['output']
        assert '.jpg' in formats['output']
        assert '.jpeg' in formats['output']
        assert '.tif' in formats['output']
        assert '.tiff' in formats['output']
        # .img should not be in supported formats anymore
        assert '.img' not in formats['output']
