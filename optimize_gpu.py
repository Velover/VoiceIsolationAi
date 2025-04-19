"""
Script to analyze and optimize GPU performance for Voice Isolation AI
"""
import torch
import os
import sys
import subprocess
import platform
import time
import argparse
from src.gpu_utils import GPUMonitor, optimize_for_gpu, print_gpu_info

def check_gpu_performance():
    """Run a GPU performance benchmark and optimization check"""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Cannot check GPU performance.")
        return False
    
    print("\n=== GPU PERFORMANCE CHECK ===")
    print_gpu_info()
    
    # Start GPU monitoring
    monitor = GPUMonitor(device_id=0)
    monitor.start()
    
    print("\nTesting matrix multiplication speed...")
    
    # Run benchmark with increasingly large matrices
    sizes = [1000, 2000, 4000, 8000]
    
    for size in sizes:
        # First run on CPU for comparison
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start = time.time()
        _ = torch.mm(a_cpu, b_cpu)
        cpu_time = time.time() - start
        
        # Now run on GPU
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        
        # Warm up
        _ = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        # Measure
        start = time.time()
        _ = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        
        print(f"{size}x{size} matrix multiplication:")
        print(f"  CPU: {cpu_time:.4f}s")
        print(f"  GPU: {gpu_time:.4f}s")
        print(f"  Speedup: {speedup:.1f}x\n")
        
        # Clean up
        del a_gpu, b_gpu
        torch.cuda.empty_cache()
    
    # Test batch processing (important for deep learning)
    print("\nTesting batch processing performance...")
    
    # Create a simple convolutional network
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(256, 10)
    ).cuda()
    
    batch_sizes = [1, 4, 16, 32, 64, 128]
    input_size = (224, 224)
    
    for batch_size in batch_sizes:
        # Create input
        inputs = torch.randn(batch_size, 3, *input_size).cuda()
        
        # Warm up
        _ = model(inputs)
        torch.cuda.synchronize()
        
        # Measure
        start = time.time()
        _ = model(inputs)
        torch.cuda.synchronize()
        batch_time = time.time() - start
        
        # Calculate throughput
        images_per_second = batch_size / batch_time
        
        print(f"Batch size {batch_size}:")
        print(f"  Processing time: {batch_time:.4f}s")
        print(f"  Throughput: {images_per_second:.1f} images/second\n")
        
        # Clean up
        del inputs
        torch.cuda.empty_cache()
    
    # Stop monitoring
    monitor.stop()
    monitor.print_summary()
    
    # Recommend optimal batch size and settings
    print("\n=== RECOMMENDATIONS ===")
    
    # Get GPU memory
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if gpu_mem > 8:
        print("✅ Your GPU has plenty of memory. Consider:")
        print("  - Using larger batch sizes (64-128)")
        print("  - Enabling mixed precision training with --mixed-precision")
    elif gpu_mem > 4:
        print("✅ Your GPU has moderate memory. Consider:")
        print("  - Using medium batch sizes (32-64)")
        print("  - Enabling mixed precision training with --mixed-precision")
    else:
        print("⚠️ Your GPU has limited memory. Consider:")
        print("  - Using smaller batch sizes (16-32)")
        print("  - Enabling mixed precision training with --mixed-precision")
    
    # Performance tuning recommendations
    print("\nPerformance tuning recommendations:")
    print("  - Add to config.py: BATCH_SIZE = 64  # Adjust based on your GPU memory")
    print("  - Add to config.py: DATALOADER_WORKERS = 2  # Use more workers if CPU-bound")
    print("  - Add to config.py: GPU_MEMORY_FRACTION = 0.95  # Use more GPU memory")
    print("  - Run training with: python main.py train --gpu --mixed-precision")
    
    return True

def optimize_windows_gpu():
    """Optimize Windows settings for GPU performance"""
    if platform.system() != "Windows":
        print("This optimization is only for Windows systems.")
        return
    
    print("\n=== OPTIMIZING WINDOWS FOR GPU PERFORMANCE ===")
    
    # Check for NVIDIA Control Panel settings
    print("Recommended NVIDIA Control Panel settings:")
    print("  1. Open NVIDIA Control Panel")
    print("  2. Manage 3D settings > Program Settings > Add 'python.exe'")
    print("  3. Set 'Power management mode' to 'Prefer maximum performance'")
    print("  4. Set 'Texture filtering - Quality' to 'High performance'")
    
    # Check Windows power settings
    print("\nChecking Windows power settings...")
    try:
        result = subprocess.run(['powercfg', '/list'], capture_output=True, text=True)
        if "High performance" in result.stdout:
            print("✅ High performance power plan is available.")
        else:
            print("⚠️ Consider switching to High performance power plan:")
            print("  Control Panel > Power Options > High performance")
    except Exception:
        print("Could not check power settings.")
    
    print("\nOptimizations complete! Restart your Python application for best results.")

def main():
    parser = argparse.ArgumentParser(description='Optimize GPU performance for Voice Isolation AI')
    parser.add_argument('--check', action='store_true', help='Run GPU performance check')
    parser.add_argument('--optimize', action='store_true', help='Optimize Windows for GPU performance')
    
    args = parser.parse_args()
    
    if args.check:
        check_gpu_performance()
    elif args.optimize:
        optimize_windows_gpu()
    else:
        # If no arguments, run both
        check_gpu_performance()
        print("\n" + "="*50 + "\n")
        optimize_windows_gpu()
    
if __name__ == "__main__":
    main()
