"""
Utility to list and show information about trained models
"""
import os
import torch
import argparse
from pathlib import Path

from config import OUTPUT_DIR, WINDOW_SIZES

def list_models(models_dir=OUTPUT_DIR, verbose=False):
    """
    List all trained models and display their information
    
    Args:
        models_dir: Directory containing models
        verbose: Whether to show detailed information
    """
    if not os.path.exists(models_dir):
        print(f"Directory {models_dir} not found")
        return
    
    # Find all model files (.pth)
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        print(f"No model files found in {models_dir}")
        return
    
    # Sort models by creation time (newest first)
    model_files.sort(reverse=True)
    
    print(f"\n=== TRAINED MODELS ({len(model_files)}) ===")
    print(f"Location: {models_dir}")
    
    for i, model_file in enumerate(model_files):
        model_path = os.path.join(models_dir, model_file)
        
        try:
            # Load model info without loading the whole model
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract info
            window_size = checkpoint.get('window_size', 'unknown')
            window_ms = WINDOW_SIZES.get(window_size, 'unknown')
            deep_model = checkpoint.get('deep_model', False)
            creation_date = checkpoint.get('creation_date', 'unknown')
            samples = checkpoint.get('training_samples', 'unknown')
            
            # Basic info
            print(f"\n{i+1}. {model_file}")
            print(f"   - Window size: {window_size} ({window_ms}ms)")
            print(f"   - Model type: {'Deep' if deep_model else 'Standard'}")
            
            # Detailed info if requested
            if verbose:
                history = checkpoint.get('history', {})
                train_loss = history.get('train_loss', [])
                val_loss = history.get('val_loss', [])
                
                print(f"   - Created on: {creation_date}")
                print(f"   - Training samples: {samples}")
                
                if train_loss and val_loss:
                    print(f"   - Final train loss: {train_loss[-1]:.4f}")
                    print(f"   - Final validation loss: {val_loss[-1]:.4f}")
                    print(f"   - Best validation loss: {min(val_loss):.4f}")
                    
        except Exception as e:
            print(f"\n{i+1}. {model_file} - Error loading model info: {e}")
    
    print("\n=== USAGE GUIDE ===")
    print("To use a model for inference:")
    print(f"python main.py isolate --input your_audio.wav --model {os.path.join(models_dir, model_files[0])}")

def main():
    parser = argparse.ArgumentParser(description='List and show information about trained models')
    parser.add_argument('--dir', default=OUTPUT_DIR, help='Directory containing models')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    
    args = parser.parse_args()
    list_models(args.dir, args.verbose)

if __name__ == '__main__':
    main()
