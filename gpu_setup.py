"""
Utility to setup GPU environment for PyTorch 2.6.0+ on Windows
"""
import os
import torch
import sys
import subprocess
import pkg_resources

def get_torch_version():
    """Get the installed PyTorch version"""
    try:
        return pkg_resources.get_distribution("torch").version
    except pkg_resources.DistributionNotFound:
        return None

def get_available_gpu_device():
    """Return the first available GPU device ID"""
    if torch.cuda.is_available():
        # Return the first available GPU (usually 0)
        return 0
    return None

def setup_cuda_for_win_nvidia():
    """Setup CUDA environment for Windows with NVIDIA GPUs"""
    # Check if PyTorch is installed
    torch_version = get_torch_version()
    if not torch_version:
        print("PyTorch is not installed. Please install it first.")
        return False
    
    print(f"Found PyTorch version: {torch_version}")
    
    # Check if CUDA is available through PyTorch
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"✅ CUDA {cuda_version} is available")
        
        # Get number of devices and their properties
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s)")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        
        # Set environment variables for Windows
        if sys.platform == 'win32':
            # Try to locate CUDA installation
            cuda_path = os.environ.get('CUDA_PATH')
            if cuda_path:
                print(f"CUDA_PATH is set to: {cuda_path}")
                
                # Add CUDA bin to PATH if not already there
                cuda_bin = os.path.join(cuda_path, 'bin')
                if cuda_bin not in os.environ['PATH']:
                    os.environ['PATH'] = cuda_bin + os.pathsep + os.environ['PATH']
                    print(f"Added {cuda_bin} to PATH")
                
        return True
    else:
        print("❌ CUDA is not available through PyTorch")
        
        # Check if NVIDIA GPU is present but PyTorch can't see it
        try:
            if sys.platform == 'win32':
                # On Windows, use WMIC to check for NVIDIA GPU
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                       capture_output=True, text=True)
                output = result.stdout.lower()
                
                if 'nvidia' in output:
                    print("⚠️ NVIDIA GPU detected but PyTorch cannot access it!")
                    print("\nRecommendation: Reinstall PyTorch with CUDA support:")
                    print("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
                    return False
        except Exception as e:
            print(f"Error checking GPU: {e}")
        
        return False

if __name__ == "__main__":
    setup_cuda_for_win_nvidia()
