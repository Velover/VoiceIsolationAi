"""
Utility functions for GPU monitoring and optimization
"""
import torch
import time
import threading
from typing import Dict, List
import os
import random

class GPUMonitor:
    """
    A class to monitor GPU utilization in a separate thread
    """
    def __init__(self, device_id=0, interval=1.0):
        """
        Initialize the GPU monitor.
        
        Args:
            device_id: GPU device ID to monitor
            interval: Monitoring interval in seconds
        """
        self.device_id = device_id
        self.interval = interval
        self.running = False
        self.thread = None
        self.data = {
            'memory_allocated': [],
            'memory_reserved': [],
            'timestamps': []
        }
        
    def start(self):
        """Start monitoring GPU utilization"""
        if not torch.cuda.is_available():
            print("CUDA not available, cannot monitor GPU")
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True  # Thread will exit when main program exits
        self.thread.start()
        return True
        
    def stop(self):
        """Stop monitoring GPU utilization"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=self.interval*2)
            
    def _monitor_loop(self):
        """Monitor loop to collect GPU utilization data"""
        start_time = time.time()
        while self.running:
            try:
                memory_allocated = torch.cuda.memory_allocated(self.device_id) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(self.device_id) / 1024**2
                
                self.data['memory_allocated'].append(memory_allocated)
                self.data['memory_reserved'].append(memory_reserved)
                self.data['timestamps'].append(time.time() - start_time)
                
                time.sleep(self.interval)
            except Exception as e:
                print(f"Error monitoring GPU: {e}")
                break
    
    def get_summary(self) -> Dict:
        """
        Get a summary of GPU utilization.
        
        Returns:
            Dictionary with min, max, avg memory utilization
        """
        if not self.data['memory_allocated']:
            return {"error": "No data collected"}
            
        allocated = self.data['memory_allocated']
        reserved = self.data['memory_reserved']
        
        return {
            'memory_allocated_max': max(allocated),
            'memory_allocated_min': min(allocated),
            'memory_allocated_avg': sum(allocated) / len(allocated),
            'memory_reserved_max': max(reserved),
            'memory_reserved_min': min(reserved),
            'memory_reserved_avg': sum(reserved) / len(reserved),
            'duration': self.data['timestamps'][-1] if self.data['timestamps'] else 0
        }
        
    def print_summary(self):
        """Print a summary of GPU utilization"""
        summary = self.get_summary()
        if 'error' in summary:
            print(f"GPU Monitor: {summary['error']}")
            return
            
        print("\n=== GPU UTILIZATION SUMMARY ===")
        print(f"Duration: {summary['duration']:.1f} seconds")
        print(f"Memory Allocated: {summary['memory_allocated_avg']:.1f} MB (avg), {summary['memory_allocated_max']:.1f} MB (max)")
        print(f"Memory Reserved: {summary['memory_reserved_avg']:.1f} MB (avg), {summary['memory_reserved_max']:.1f} MB (max)")

def get_gpu_info() -> Dict:
    """
    Get information about available GPUs.
    
    Returns:
        Dictionary with GPU information
    """
    if not torch.cuda.is_available():
        return {"available": False}
        
    info = {
        "available": True,
        "count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": []
    }
    
    for i in range(info["count"]):
        props = torch.cuda.get_device_properties(i)
        info["devices"].append({
            "name": props.name,
            "total_memory": props.total_memory / 1024**3,  # GB
            "compute_capability": f"{props.major}.{props.minor}"
        })
        
    return info

def print_gpu_info():
    """Print information about available GPUs"""
    info = get_gpu_info()
    
    if not info["available"]:
        print("No CUDA-compatible GPUs detected")
        return
        
    print("\n=== GPU INFORMATION ===")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Available GPUs: {info['count']}")
    print(f"Current device: {info['current_device']}")
    
    for i, device in enumerate(info["devices"]):
        print(f"\nDevice {i}: {device['name']}")
        print(f"  Memory: {device['total_memory']:.2f} GB")
        print(f"  Compute Capability: {device['compute_capability']}")

def optimize_for_gpu():
    """Configure PyTorch for optimal GPU performance"""
    if not torch.cuda.is_available():
        return
        
    # Set tensor cores for faster computation if available
    if torch.cuda.get_device_capability(0)[0] >= 7:
        # For Volta, Turing, Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Set benchmark mode for optimized convolutions
    torch.backends.cudnn.benchmark = True

def estimate_optimal_samples(preprocessor, voice_files, noise_files, batch_size=64, memory_headroom=0.2) -> int:
    """
    Estimate the optimal number of samples based on hardware resources.
    
    Args:
        preprocessor: AudioPreprocessor instance
        voice_files: List of voice audio files
        noise_files: List of noise audio files
        batch_size: Training batch size
        memory_headroom: Fraction of memory to leave free (0.0-1.0)
        
    Returns:
        Estimated optimal number of samples
    """
    if not torch.cuda.is_available():
        print("GPU not available, using default sample count")
        return 2000  # Default for CPU
    
    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
    
    # Create a small batch of samples to measure memory usage
    print("Testing memory usage per sample...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        # Generate test samples
        test_samples = 10
        start_memory = torch.cuda.memory_allocated() / 1024**2
        
        for i in range(test_samples):
            # Get random voice and noise files
            voice_path = random.choice(voice_files)
            noise_path = random.choice(noise_files)
            
            # Create a sample
            preprocessor.create_training_example(voice_path, noise_path)
            print(f"  Created test sample {i+1}/{test_samples}", end="\r")
        
        end_memory = torch.cuda.memory_allocated() / 1024**2
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        # Calculate memory per sample
        memory_per_sample = (peak_memory - start_memory) / test_samples
        
        # Estimate the memory needed for training
        # We consider: 
        # 1. Memory for all samples
        # 2. Memory for the model
        # 3. Memory for optimizer states (roughly 2x model size)
        # 4. Memory for batches during training
        # 5. Memory for gradients and forward pass
        
        # We estimate model + optimizer + batch memory as roughly 1000MB
        training_overhead = 1000  
        
        # Calculate available memory considering headroom
        available_memory = total_memory * (1 - memory_headroom) - training_overhead
        
        # Calculate maximum samples that would fit in memory
        max_samples = int(available_memory / memory_per_sample)
        
        # Make it divisible by batch size for efficiency
        max_samples = (max_samples // batch_size) * batch_size
        
        # Apply reasonable limits
        max_samples = max(1000, min(10000, max_samples))
        
        print(f"\nMemory analysis:")
        print(f"  Total GPU memory: {total_memory:.0f} MB")
        print(f"  Memory per sample: {memory_per_sample:.2f} MB")
        print(f"  Estimated maximum samples: {max_samples}")
        
        return max_samples
        
    except Exception as e:
        print(f"Error estimating sample capacity: {e}")
        return 2000  # Default fallback value

def optimize_gpu_utilization():
    """
    Optimize settings to maximize GPU computational utilization.
    """
    if not torch.cuda.is_available():
        return
    
    # Set benchmark mode for optimized kernel selection
    torch.backends.cudnn.benchmark = True
    
    # Enable TensorFloat32 on Ampere GPUs for faster computation
    if torch.cuda.get_device_capability(0)[0] >= 8:
        torch.set_float32_matmul_precision('high')
    
    # Enable tensor cores for mixed precision
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set optimal threading configuration
    if hasattr(torch, 'set_num_threads'):
        # Use fewer CPU threads when GPU is primary compute device
        torch.set_num_threads(4)
    
    # Set environment variables for faster parallelism
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
