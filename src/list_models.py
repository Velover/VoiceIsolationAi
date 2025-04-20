"""
Utility to list and show information about trained models
"""
import os
import torch
import argparse
from pathlib import Path
from typing import List, Dict
from tabulate import tabulate
import time
import datetime

from .config import OUTPUT_DIR

def load_model_info(model_path: str) -> Dict:
    """
    Load and return information about a model file.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Dictionary with model information
    """
    try:
        # Load the model file but only access metadata
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract information
        info = {
            'path': model_path,
            'filename': os.path.basename(model_path),
            'window_size': checkpoint.get('window_size', 'unknown'),
            'deep_model': checkpoint.get('deep_model', False),
            'creation_date': checkpoint.get('creation_date', 'unknown'),
            'training_samples': checkpoint.get('training_samples', 'unknown'),
            'n_fft': checkpoint.get('n_fft', 512),
            'file_size_mb': os.path.getsize(model_path) / (1024 * 1024),
            'modification_time': datetime.datetime.fromtimestamp(
                os.path.getmtime(model_path)
            ).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add extra information if available
        if 'history' in checkpoint:
            history = checkpoint['history']
            if 'train_loss' in history and history['train_loss']:
                info['final_train_loss'] = history['train_loss'][-1]
            if 'val_loss' in history and history['val_loss']:
                info['final_val_loss'] = history['val_loss'][-1]
        
        return info
    except Exception as e:
        return {
            'path': model_path,
            'filename': os.path.basename(model_path),
            'error': str(e)
        }

def find_models(directory: str) -> List[str]:
    """
    Find all model files in the specified directory.
    
    Args:
        directory: Directory to search for models
        
    Returns:
        List of model file paths
    """
    model_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pth'):
                model_files.append(os.path.join(root, file))
    
    return model_files

def list_models(directory: str = OUTPUT_DIR, verbose: bool = False):
    """
    List all models in the directory with their information.
    
    Args:
        directory: Directory containing models
        verbose: Whether to show detailed information
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return
    
    # Find model files
    model_files = find_models(directory)
    
    if not model_files:
        print(f"No model files found in {directory}")
        return
    
    print(f"Found {len(model_files)} model files in {directory}")
    
    # Load model information
    models_info = []
    for model_path in model_files:
        info = load_model_info(model_path)
        models_info.append(info)
    
    # Sort by modification time (newest first)
    models_info.sort(key=lambda x: x.get('modification_time', ''), reverse=True)
    
    # Display as table
    if verbose:
        # Detailed view
        headers = ['Filename', 'Window Size', 'Type', 'Created', 'Train Loss', 'Val Loss', 'Samples', 'Size (MB)', 'Modified']
        table_data = []
        
        for info in models_info:
            model_type = 'Deep' if info.get('deep_model', False) else 'Standard'
            train_loss = f"{info.get('final_train_loss', 'N/A'):.4f}" if 'final_train_loss' in info else 'N/A'
            val_loss = f"{info.get('final_val_loss', 'N/A'):.4f}" if 'final_val_loss' in info else 'N/A'
            
            table_data.append([
                info.get('filename', 'unknown'),
                info.get('window_size', 'unknown'),
                model_type,
                info.get('creation_date', 'unknown'),
                train_loss,
                val_loss,
                info.get('training_samples', 'N/A'),
                f"{info.get('file_size_mb', 0):.1f}",
                info.get('modification_time', 'unknown')
            ])
    else:
        # Simple view
        headers = ['Filename', 'Window Size', 'Type', 'Created', 'Size (MB)']
        table_data = []
        
        for info in models_info:
            model_type = 'Deep' if info.get('deep_model', False) else 'Standard'
            
            table_data.append([
                info.get('filename', 'unknown'),
                info.get('window_size', 'unknown'),
                model_type,
                info.get('creation_date', 'unknown'),
                f"{info.get('file_size_mb', 0):.1f}"
            ])
    
    # Print the table
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print path info
    print(f"\nModels directory: {os.path.abspath(directory)}")
    print(f"Use model path like: {os.path.join(directory, models_info[0]['filename'])} for inference")

def main():
    parser = argparse.ArgumentParser(description='List trained models and their information')
    parser.add_argument('--dir', default=OUTPUT_DIR, help='Directory containing models')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed information')
    
    args = parser.parse_args()
    
    list_models(args.dir, args.verbose)

if __name__ == "__main__":
    main()
