import os
import argparse
import torch
import numpy as np
from model import UNet
from utils import load_audio, prepare_spectrograms, reconstruct_signal, save_audio
from config import *

def process_audio(model, input_file, output_file, window_params=None):
    """Process audio file to isolate the target voice."""
    # Set default window parameters if not provided
    if window_params is None:
        window_params = {
            'window_samples': WINDOW_SIZE,
            'hop_length': HOP_LENGTH,
            'n_fft': N_FFT
        }
    
    # Load audio
    print(f"Loading audio file: {input_file}")
    audio = load_audio(input_file)
    
    # Compute STFT with the model's window parameters
    print("Computing STFT...")
    magnitude, phase = prepare_spectrograms(
        audio,
        n_fft=window_params['n_fft'],
        hop_length=window_params['hop_length'],
        win_length=window_params['window_samples']
    )
    
    # Convert to tensor
    magnitude_tensor = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # Get model prediction
    print("Applying voice isolation model...")
    model.eval()
    with torch.no_grad():
        mask = model(magnitude_tensor).squeeze().cpu().numpy()
    
    # Apply mask to isolate voice
    enhanced_magnitude = magnitude * mask
    
    # Reconstruct time-domain signal
    print("Reconstructing audio signal...")
    enhanced_audio = reconstruct_signal(
        enhanced_magnitude, 
        phase,
        hop_length=window_params['hop_length'],
        win_length=window_params['window_samples']
    )
    
    # Save output
    print(f"Saving output to: {output_file}")
    save_audio(enhanced_audio, output_file)
    
    return enhanced_audio

def main():
    parser = argparse.ArgumentParser(description='Isolate target voice from audio file')
    parser.add_argument('--input', required=True, help='Input audio file')
    parser.add_argument('--output', required=True, help='Output audio file')
    parser.add_argument('--window-size', type=str, choices=['SMALL', 'MEDIUM', 'LARGE', 'XLARGE'], 
                        default='SMALL', help='Window size to use (default: SMALL)')
    parser.add_argument('--model', help='Path to trained model (overrides window-size)')
    
    args = parser.parse_args()
    
    # Determine which model to use
    if args.model:
        model_path = args.model
        print(f"Using specified model: {model_path}")
    else:
        window_size = WindowSize[args.window_size]
        model_id = get_model_id(window_size)
        model_path = os.path.join(OUTPUT_DIR, f'{model_id}.pth')
        print(f"Using model for window size {window_size.name} ({window_size.value}ms): {model_path}")
    
    # Load model and its parameters
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Handle both old and new model format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        # Extract window parameters
        window_params = {
            'window_samples': checkpoint.get('window_samples', WINDOW_SIZE),
            'hop_length': checkpoint.get('hop_length', HOP_LENGTH),
            'n_fft': checkpoint.get('n_fft', N_FFT)
        }
        print(f"Using window parameters from model: {window_params}")
    else:
        model_state = checkpoint
        # Use default parameters
        window_params = None
        print("Using default window parameters")
    
    # Create and load model
    model = UNet().to(DEVICE)
    model.load_state_dict(model_state)
    
    # Process audio
    process_audio(model, args.input, args.output, window_params)
    print(f"Processing completed! Output saved to {args.output}")

if __name__ == "__main__":
    main()
