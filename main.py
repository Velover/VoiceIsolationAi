import os
import argparse
import sys

from src.train import main as train_main
from src.inference import main as inference_main
from src.config import OUTPUT_DIR, USE_GPU, MIXED_PRECISION

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
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
