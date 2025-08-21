"""
Image processing module for segimage library.
Handles various image processing operations including MATLAB file conversion.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any
import numpy as np
from scipy import io
from PIL import Image
from .processors import get_processor, available_processors
from .utils import save_array_as_image, normalize_to_uint8


class ImageProcessor:
    """Main image processor class for handling various image operations."""
    
    def __init__(self):
        self.supported_input_formats = ['.mat', '.npy', '.tif', '.tiff', '.png', '.jpg', '.jpeg']
        self.supported_output_formats = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npy', '.graphml', '.gml', '.lg', '.lgl', '.edgelist', '.edges', '.txt', '.pickle', '.pkl']
    
    def process_mat_to_image(self, input_path: Union[str, Path], output_path: Union[str, Path], 
                           output_format: str = '.png') -> bool:
        """
        Convert MATLAB .mat file to image format (PNG, JPG, etc.).
        
        Args:
            input_path: Path to input .mat file
            output_path: Path for output image file
            output_format: Output format (default: .png)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            
            # Validate input file
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            if input_path.suffix.lower() != '.mat':
                raise ValueError(f"Input file must be a .mat file, got: {input_path.suffix}")
            
            # Validate output format
            if output_format.lower() not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                raise ValueError(f"Unsupported output format: {output_format}. Supported: .png, .jpg, .jpeg, .tif, .tiff")
            
            # Load MATLAB file
            mat_data = io.loadmat(str(input_path))
            
            # Find the main data array (exclude metadata keys)
            data_keys = [key for key in mat_data.keys() if not key.startswith('__')]
            
            if not data_keys:
                raise ValueError("No data arrays found in MATLAB file")
            
            # Use the first data array found
            main_key = data_keys[0]
            image_data = mat_data[main_key]
            
            print(f"Found data key: {main_key}")
            print(f"Data type: {type(image_data)}")
            print(f"Data shape: {image_data.shape if hasattr(image_data, 'shape') else 'N/A'}")
            
            # Handle different MATLAB data types
            if hasattr(image_data, 'dtype'):
                # It's a numpy array
                if image_data.dtype == np.object_:
                    # Handle object arrays (cell arrays, structs)
                    print("Detected object array, attempting to extract numeric data...")
                    
                    # Use recursive extraction function
                    numeric_data, message = self._extract_numeric_data(image_data)
                    
                    if numeric_data is not None:
                        image_data = numeric_data
                        print(f"✅ {message}")
                    else:
                        raise ValueError(f"Could not extract numeric data: {message}")
                
                # Convert to appropriate data type for image processing
                if image_data.dtype == np.float64:
                    image_data = image_data.astype(np.float32)
                elif image_data.dtype == np.int64:
                    image_data = image_data.astype(np.int32)
                elif image_data.dtype == np.uint64:
                    image_data = image_data.astype(np.uint32)
                
                print(f"Final data shape: {image_data.shape}")
                print(f"Final data type: {image_data.dtype}")
                
                # Convert to PIL Image and save
                success = self._save_as_image(image_data, output_path, output_format)
                
                if success:
                    print(f"Successfully converted {input_path} to {output_path}")
                    print(f"Data shape: {image_data.shape}, Data type: {image_data.dtype}")
                    return True
                else:
                    return False
            else:
                raise ValueError(f"Unexpected data type: {type(image_data)}")
            
        except Exception as e:
            print(f"Error processing file: {e}")
            return False
    
    def _save_as_image(self, image_data: np.ndarray, output_path: Path, output_format: str) -> bool:
        """Save numpy array as image file using shared utilities."""
        return save_array_as_image(image_data, output_path, output_format)
    
    def _extract_numeric_data(self, data, max_depth=5, current_depth=0):
        """
        Recursively extract numeric data from nested object arrays and structured arrays.
        
        Args:
            data: The data to extract from
            max_depth: Maximum recursion depth to prevent infinite loops
            current_depth: Current recursion depth
            
        Returns:
            tuple: (numeric_data, success_message) or (None, error_message)
        """
        if current_depth >= max_depth:
            return None, f"Maximum recursion depth ({max_depth}) reached"
        
        if not hasattr(data, 'dtype'):
            return None, f"Data has no dtype attribute: {type(data)}"
        
        # If it's already numeric, return it
        if data.dtype != np.object_:
            return data, f"Found numeric data with shape {data.shape} and dtype {data.dtype}"
        
        # Check if it's a structured array (record array)
        if hasattr(data.dtype, 'names') and data.dtype.names:
            print(f"  {'  ' * current_depth}Structured array with fields: {data.dtype.names}")
            
            # Try each field
            for field_name in data.dtype.names:
                field_data = data[field_name]
                print(f"  {'  ' * current_depth}Checking field '{field_name}'...")
                
                result, message = self._extract_numeric_data(field_data, max_depth, current_depth + 1)
                if result is not None:
                    return result, f"Found numeric data in field '{field_name}': {message}"
            
            # If no field worked, try to flatten and search
            print(f"  {'  ' * current_depth}No numeric data in fields, searching flattened structure...")
            for i, item in enumerate(data.flat):
                if i > 10:  # Limit search to first 10 items
                    break
                result, message = self._extract_numeric_data(item, max_depth, current_depth + 1)
                if result is not None:
                    return result, f"Found numeric data at index {i}: {message}"
        
        # Regular object array
        if data.size > 0:
            print(f"  {'  ' * current_depth}Object array with {data.size} elements")
            
            # Try to convert the entire array
            try:
                numeric_data = np.array(data, dtype=np.float32)
                return numeric_data, f"Successfully converted object array to numeric array with shape {numeric_data.shape}"
            except (ValueError, TypeError):
                pass
            
            # Try individual elements
            for i in range(min(data.size, 10)):  # Limit search to first 10 elements
                item = data.flat[i]
                print(f"  {'  ' * current_depth}Checking element {i}: type={type(item)}, dtype={getattr(item, 'dtype', 'N/A')}")
                
                # If this element is a structured array, extract from its fields
                if hasattr(item, 'dtype') and hasattr(item.dtype, 'names') and item.dtype.names:
                    print(f"  {'  ' * current_depth}Element {i} is a structured array with fields: {item.dtype.names}")
                    for field_name in item.dtype.names:
                        field_value = item[field_name]
                        print(f"  {'  ' * current_depth}Field '{field_name}' value type: {type(field_value)}, dtype: {getattr(field_value, 'dtype', 'N/A')}")
                        
                        if hasattr(field_value, 'dtype') and field_value.dtype != np.object_:
                            return field_value, f"Found numeric data in element {i}, field '{field_name}' with shape {field_value.shape}"
                        elif hasattr(field_value, 'dtype') and field_value.dtype == np.object_:
                            # Go deeper into this field
                            result, message = self._extract_numeric_data(field_value, max_depth, current_depth + 1)
                            if result is not None:
                                return result, f"Found numeric data in element {i}, field '{field_name}': {message}"
                
                # Regular recursive search
                result, message = self._extract_numeric_data(item, max_depth, current_depth + 1)
                if result is not None:
                    return result, f"Found numeric data at index {i}: {message}"
        
        return None, f"No numeric data found at depth {current_depth}"
    
    def _save_metadata(self, metadata_path: Path, image_data: np.ndarray, original_key: str, output_format: str):
        """Save metadata about the image data for future reference."""
        try:
            # Get the normalized image data for metadata
            normalized_data = normalize_to_uint8(image_data.copy())
            
            metadata = {
                "original_key": original_key,
                "shape": image_data.shape,
                "original_dtype": str(image_data.dtype),
                "size": image_data.size,
                "original_min_value": float(np.min(image_data)) if image_data.size > 0 else None,
                "original_max_value": float(np.max(image_data)) if image_data.size > 0 else None,
                "normalized_min_value": float(np.min(normalized_data)) if normalized_data.size > 0 else None,
                "normalized_max_value": float(np.max(normalized_data)) if normalized_data.size > 0 else None,
                "output_format": output_format,
                "description": f"Converted from MATLAB .mat file using segimage library to {output_format} format",
                "normalization_info": "Data normalized to 0-255 range (0 = black, 255 = white) for proper image display"
            }
            
            # Save as JSON for easy reading
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def inspect_mat_file(self, input_path: Union[str, Path]) -> bool:
        """
        Inspect the contents of a MATLAB .mat file to understand its structure.
        
        Args:
            input_path: Path to input .mat file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            input_path = Path(input_path)
            
            # Validate input file
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            if input_path.suffix.lower() != '.mat':
                raise ValueError(f"Input file must be a .mat file, got: {input_path.suffix}")
            
            # Load MATLAB file
            mat_data = io.loadmat(str(input_path))
            
            print(f"\n📁 MATLAB File Contents: {input_path}")
            print("=" * 50)
            
            # Show all keys
            print("Available data keys:")
            for key in sorted(mat_data.keys()):
                if not key.startswith('__'):
                    data = mat_data[key]
                    if hasattr(data, 'shape'):
                        print(f"  📊 {key}: {type(data).__name__} with shape {data.shape}")
                        if hasattr(data, 'dtype'):
                            print(f"         Data type: {data.dtype}")
                            if data.dtype == np.object_:
                                print(f"         Object array - may contain mixed data types")
                    else:
                        print(f"  📄 {key}: {type(data).__name__}")
                else:
                    print(f"  🔧 {key}: {type(mat_data[key]).__name__} (metadata)")
            
            # Show detailed info for first data array
            data_keys = [key for key in mat_data.keys() if not key.startswith('__')]
            if data_keys:
                main_key = data_keys[0]
                main_data = mat_data[main_key]
                
                print(f"\n🔍 Detailed analysis of '{main_key}':")
                print(f"   Type: {type(main_data).__name__}")
                
                if hasattr(main_data, 'shape'):
                    print(f"   Shape: {main_data.shape}")
                    print(f"   Size: {main_data.size}")
                    
                    if hasattr(main_data, 'dtype'):
                        print(f"   Data type: {main_data.dtype}")
                        
                        if main_data.dtype == np.object_:
                            print(f"   Object array detected!")
                            if main_data.size > 0:
                                first_element = main_data.flat[0]
                                print(f"   First element type: {type(first_element)}")
                                if hasattr(first_element, 'dtype'):
                                    print(f"   First element data type: {first_element.dtype}")
                                    if hasattr(first_element, 'shape'):
                                        print(f"   First element shape: {first_element.shape}")
            
            print("=" * 50)
            return True
            
        except Exception as e:
            print(f"Error inspecting file: {e}")
            return False
    
    def process_image(self, input_path: Union[str, Path], output_path: Union[str, Path], 
                     process_type: str = "mat_to_image", **options: Any) -> bool:
        """
        Main processing method that routes to appropriate processor based on type.
        
        Args:
            input_path: Path to input image file
            output_path: Path for output image file
            process_type: Type of processing to perform
            
        Returns:
            bool: True if successful, False otherwise
        """
        process_type = process_type.lower()
        
        if process_type == "mat_to_image":
            # Determine output format from output_path extension
            output_format = output_path.suffix.lower()
            if output_format not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                output_format = '.png'  # Default to PNG
            return self.process_mat_to_image(input_path, output_path, output_format)
        elif process_type == "inspect":
            return self.inspect_mat_file(input_path)
        else:
            # Look up pluggable processors
            proc = get_processor(process_type)
            if proc is not None:
                return bool(proc(Path(input_path), Path(output_path), **options))
            print(f"Unknown process type: {process_type}")
            builtins = ["mat_to_image", "inspect"]
            extra = list(available_processors().keys())
            all_types = builtins + extra
            print(f"Supported types: {', '.join(all_types)}")
            return False
    
    def get_supported_formats(self) -> dict:
        """Get supported input and output formats."""
        return {
            "input": self.supported_input_formats,
            "output": self.supported_output_formats
        }
