import os
import argparse
import sys
import torch

# Add GPU environment setup
from gpu_setup import setup_cuda_for_win_nvidia

# Attempt to setup CUDA environment first
setup_result = setup_cuda_for_win_nvidia()

from src.train import main as train_main
from src.inference import main as inference_main
from src.config import OUTPUT_DIR, USE_GPU, MIXED_PRECISION

def list_gpus():
    """List all available GPUs and their properties"""
    if not torch.cuda.is_available():
        print("No CUDA-compatible GPUs detected.")
        print("\nIMPORTANT: If you have an NVIDIA GPU, but PyTorch cannot see it:")
        print("Run the following command to reinstall PyTorch with CUDA support:")
        print("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return

    print(f"CUDA is available with {torch.cuda.device_count()} devices:")
    for i in range(torch.cuda.device_count()):
        try:
            properties = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {properties.name}")
            print(f"    - Total memory: {properties.total_memory / 1024**3:.2f} GB")
            print(f"    - CUDA Capability: {properties.major}.{properties.minor}")
        except Exception as e:
            print(f"  Device {i}: Error getting properties - {e}")
    
    print(f"Current device: {torch.cuda.current_device()}")

def main():
    parser = argparse.ArgumentParser(description='Voice Isolation AI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--window-size', choices=['small', 'medium', 'large'], 
                        default='medium', help='Window size for processing')
    train_parser.add_argument('--batch-size', type=int, default=32, 
                        help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs')
    train_parser.add_argument('--samples', type=int, default=2000, 
                        help='Number of training samples to generate')
    train_parser.add_argument('--gpu', action='store_true', default=USE_GPU,
                        help='Use GPU acceleration if available')
    train_parser.add_argument('--mixed-precision', action='store_true', default=MIXED_PRECISION,
                        help='Use mixed precision training for faster computation')
    
    # Inference command
    inference_parser = subparsers.add_parser('isolate', help='Isolate voice in audio file')
    inference_parser.add_argument('--input', required=True, help='Path to input audio file')
    inference_parser.add_argument('--output', help='Path to save output audio')
    inference_parser.add_argument('--model', required=True, help='Path to trained model')
    inference_parser.add_argument('--gpu', action='store_true', default=USE_GPU,
                        help='Use GPU acceleration if available')
    
    # GPU Info command
    subparsers.add_parser('gpu-info', help='List available GPUs and their properties')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        sys.argv = [sys.argv[0]] + [
            '--window-size', args.window_size,
            '--batch-size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--samples', str(args.samples)
        ]
        
        if args.gpu:
            sys.argv.append('--gpu')
        if args.mixed_precision:
            sys.argv.append('--mixed-precision')
            
        train_main()
    elif args.command == 'isolate':
        sys.argv = [sys.argv[0]] + [
            '--input', args.input,
            '--model', args.model
        ]
        if args.output:
            sys.argv += ['--output', args.output]
        if args.gpu:
            sys.argv.append('--gpu')
            
        inference_main()
    elif args.command == 'gpu-info':
        list_gpus()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
