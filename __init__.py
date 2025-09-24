import os
import sys
import subprocess
import platform

def find_comfyui_root():
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):
        if os.path.basename(current) == 'custom_nodes':
            return os.path.dirname(current)
        current = os.path.dirname(current)
    return None

def is_package_installed(python_exe, package):
    try:
        result = subprocess.run([python_exe, '-m', 'pip', 'show', package], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def install_dependencies():
    comfy_root = find_comfyui_root()
    if not comfy_root:
        print("[SegImage] Could not find ComfyUI root directory!")
        return
        
    venv_dir = os.path.join(comfy_root, '.venv')
    if not os.path.exists(venv_dir):
        print("[SegImage] ComfyUI venv not found!")
        return
        
    bin_dir = 'Scripts' if platform.system() == 'Windows' else 'bin'
    python_exe = os.path.join(venv_dir, bin_dir, 'python.exe' if platform.system() == 'Windows' else 'python')
    
    if not os.path.exists(python_exe):
        print("[SegImage] Python executable not found in venv!")
        return
        
    # Check if uv is installed
    try:
        subprocess.check_call([python_exe, '-m', 'uv', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        installer = 'uv'
    except:
        print("[SegImage] uv not found, installing uv using pip...")
        try:
            subprocess.check_call([python_exe, '-m', 'pip', 'install', 'uv'])
            print("[SegImage] uv installed successfully.")
            installer = 'uv'
        except Exception as e:
            print(f"[SegImage] Failed to install uv: {e}")
            print("[SegImage] Falling back to pip for dependencies.")
            installer = 'pip'
    
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    # Check if key deps are missing to avoid unnecessary install
    key_deps = ['scikit-image', 'hedonic']
    if all(is_package_installed(python_exe, dep) for dep in key_deps):
        print("[SegImage] Key dependencies already installed. Skipping.")
        return
        
    try:
        if installer == 'uv':
            subprocess.check_call([python_exe, '-m', 'uv', 'pip', 'install', '-r', req_path])
        else:
            subprocess.check_call([python_exe, '-m', 'pip', 'install', '-r', req_path])
        print("[SegImage] Dependencies installed successfully.")
    except Exception as e:
        print(f"[SegImage] Failed to install dependencies: {e}")

install_dependencies()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
