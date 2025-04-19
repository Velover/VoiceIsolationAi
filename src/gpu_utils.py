"""
Utility functions for GPU monitoring and optimization
"""
import torch
import time
import threading
from typing import Dict, List
import os

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
