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

    try:
        # Force CUDA initialization
        torch.cuda.init()
        torch.cuda.synchronize()
        
        # Test tensor creation on GPU
        test = torch.ones(1, device=f"cuda:0")
        
        print(f"CUDA is available with {torch.cuda.device_count()} devices:")
        for i in range(torch.cuda.device_count()):
            try:
                properties = torch.cuda.get_device_properties(i)
                print(f"  Device {i}: {properties.name}")
                print(f"    - Total memory: {properties.total_memory / 1024**3:.2f} GB")
                print(f"    - CUDA Capability: {properties.major}.{properties.minor}")
                
                # Test memory allocation
                memory_before = torch.cuda.memory_allocated(i)
                test_tensor = torch.ones((1000, 1000), device=f"cuda:{i}")
                memory_after = torch.cuda.memory_allocated(i)
                memory_used = (memory_after - memory_before) / 1024**2
                print(f"    - Memory allocation test: {memory_used:.2f} MB (should be ~4 MB)")
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Device {i}: Error getting properties - {e}")
        
        print(f"Current device: {torch.cuda.current_device()}")
        
        # Additional CUDA info
        print("\nCUDA Environment:")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    except Exception as e:
        print(f"Error testing CUDA: {e}")
        print("Your CUDA installation might be incomplete or corrupted.")

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
    train_parser.add_argument('--samples', type=int, default=0,  # Changed to 0 to favor auto-detection
                        help='Number of training samples to generate (0 for auto-detect)')
    train_parser.add_argument('--gpu', action='store_true', default=USE_GPU,
                        help='Use GPU acceleration if available')
    train_parser.add_argument('--mixed-precision', action='store_true', default=MIXED_PRECISION,
                        help='Use mixed precision training for faster computation')
    train_parser.add_argument('--auto-detect-samples', action='store_true', default=True,
                        help='Auto-detect optimal sample count based on GPU memory')
    train_parser.add_argument('--deep-model', action='store_true', default=False,
                        help='Use deeper model for higher GPU utilization')
    train_parser.add_argument('--use-cache', action='store_true', default=False,
                        help='Use cached preprocessed data for faster training')
    train_parser.add_argument('--preprocessed', action='store_true', default=False,
                        help='Use pre-generated samples for maximum speed (fastest)')
    train_parser.add_argument('--preprocessed-dir', default=None,
                        help='Directory containing pre-generated samples')
    
    # Inference command
    inference_parser = subparsers.add_parser('isolate', help='Isolate voice in audio file')
    inference_parser.add_argument('--input', required=True, help='Path to input audio file')
    inference_parser.add_argument('--output', help='Path to save output audio')
    inference_parser.add_argument('--model', required=True, help='Path to trained model')
    inference_parser.add_argument('--gpu', action='store_true', default=USE_GPU,
                        help='Use GPU acceleration if available')
    inference_parser.add_argument('--debug', action='store_true',
                        help='Generate additional debug files')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess audio files for faster training')
    preprocess_parser.add_argument('--workers', type=int, default=os.cpu_count(),
                          help='Number of worker threads for preprocessing')
    preprocess_parser.add_argument('--convert', action='store_true',
                          help='Convert WAV files to more efficient format')
    preprocess_parser.add_argument('--create-examples', action='store_true',
                          help='Create example audio files if directories are empty')
    preprocess_parser.add_argument('--samples', type=int, default=1000,
                          help='Number of training samples to preprocess and save')
    
    # Generate Samples command
    generate_parser = subparsers.add_parser('generate-samples', help='Generate and save training samples')
    generate_parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples to generate')
    generate_parser.add_argument('--window-size', choices=['small', 'medium', 'large'], 
                        default='medium', help='Window size for processing')
    generate_parser.add_argument('--gpu', action='store_true', default=USE_GPU,
                        help='Use GPU acceleration if available')
    generate_parser.add_argument('--force', action='store_true',
                        help='Force regeneration of existing samples')
    generate_parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto)')
    generate_parser.add_argument('--output-dir', default=None,
                        help='Directory to save preprocessed samples')
    generate_parser.add_argument('--use-cache', action='store_true', default=False,
                        help='Use cached spectrograms for faster sample generation')
    generate_parser.add_argument('--voice-cache-dir', default=None,
                        help='Directory with cached voice spectrograms')
    generate_parser.add_argument('--noise-cache-dir', default=None,
                        help='Directory with cached noise spectrograms')
    
    # GPU Info command
    subparsers.add_parser('gpu-info', help='List available GPUs and their properties')
    
    # Generate Test Audio command
    test_audio_parser = subparsers.add_parser('generate-test', help='Generate test audio by mixing voice and noise')
    test_audio_parser.add_argument('--output-dir', default='TEST', help='Directory to save test files')
    test_audio_parser.add_argument('--num-files', type=int, default=5, help='Number of test files to generate')
    test_audio_parser.add_argument('--duration', type=int, default=20, help='Duration of each file in seconds')
    test_audio_parser.add_argument('--min-snr', type=float, default=3.0, help='Minimum SNR in dB')
    test_audio_parser.add_argument('--max-snr', type=float, default=10.0, help='Maximum SNR in dB')
    
    # Add List Models command
    list_models_parser = subparsers.add_parser('list-models', help='List trained models and their information')
    list_models_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    list_models_parser.add_argument('--dir', default=OUTPUT_DIR, help='Directory containing models')
    
    # Add Batch Isolation command
    batch_parser = subparsers.add_parser('isolate-batch', help='Process all files in test directory')
    batch_parser.add_argument('--model', required=True, help='Path to trained model')
    batch_parser.add_argument('--input-dir', default='TEST/MIXED', help='Directory with mixed audio files')
    batch_parser.add_argument('--output-dir', default='TEST/ISOLATED', help='Directory to save isolated files')
    batch_parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU acceleration')
    batch_parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (1=sequential)')
    batch_parser.add_argument('--debug', action='store_true', help='Generate additional debug files')
    
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
        if args.auto_detect_samples:
            sys.argv.append('--auto-detect-samples')
        if args.deep_model:
            sys.argv.append('--deep-model')
        if args.use_cache:
            sys.argv.append('--use-cache')
        if args.preprocessed:
            sys.argv.append('--preprocessed')
        if args.preprocessed_dir:
            sys.argv.extend(['--preprocessed-dir', args.preprocessed_dir])
            
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
        if args.debug:
            sys.argv.append('--debug')
            
        inference_main()
    elif args.command == 'preprocess':
        # Import and run preprocessing script
        from preprocess_cache import main as preprocess_main
        # Set sys.argv to pass arguments
        sys.argv = [sys.argv[0]]
        if args.workers:
            sys.argv.extend(['--workers', str(args.workers)])
        if args.convert:
            sys.argv.append('--convert')
        if args.create_examples:
            sys.argv.append('--create-examples')
        if args.samples:
            sys.argv.extend(['--samples', str(args.samples)])
        preprocess_main()
    elif args.command == 'generate-samples':
        # Import and run sample generation script
        from preprocess_samples import main as generate_samples_main
        # Set sys.argv to pass arguments
        sys.argv = [sys.argv[0]]
        if args.samples:
            sys.argv.extend(['--samples', str(args.samples)])
        if args.window_size:
            sys.argv.extend(['--window-size', args.window_size])
        if args.gpu:
            sys.argv.append('--gpu')
        if args.force:
            sys.argv.append('--force')
        if args.workers:
            sys.argv.extend(['--workers', str(args.workers)])
        if args.output_dir:
            sys.argv.extend(['--output-dir', args.output_dir])
        if args.use_cache:
            sys.argv.append('--use-cache')
        if args.voice_cache_dir:
            sys.argv.extend(['--voice-cache-dir', args.voice_cache_dir])
        if args.noise_cache_dir:
            sys.argv.extend(['--noise-cache-dir', args.noise_cache_dir])
        generate_samples_main()
    elif args.command == 'gpu-info':
        list_gpus()
    elif args.command == 'generate-test':
        # Import and run test audio generation script
        from generate_test_audio import create_test_files
        create_test_files(
            output_dir=args.output_dir,
            num_files=args.num_files,
            duration=args.duration,
            min_snr=args.min_snr,
            max_snr=args.max_snr
        )
    elif args.command == 'list-models':
        # Import and run model listing utility
        from src.list_models import list_models
        list_models(args.dir, args.verbose)
    elif args.command == 'isolate-batch':
        # Import and run batch isolation
        from src.batch_isolate import batch_process
        batch_process(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_path=args.model,
            use_gpu=args.gpu,
            num_workers=args.workers,
            debug=args.debug  # Pass debug flag
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
