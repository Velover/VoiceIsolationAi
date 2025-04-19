import os
import torch
import torchaudio
import argparse
from typing import Optional, Tuple
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .config import OUTPUT_DIR, N_FFT, HOP_LENGTH, SAMPLE_RATE, SPEC_TIME_DIM
from .preprocessing import AudioPreprocessor
from .model import VoiceIsolationModel

def load_model(model_path: str, device: torch.device) -> Tuple[VoiceIsolationModel, str]:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to model file
        device: Device to load model on
        
    Returns:
        Tuple of (model, window_size)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model and load state
    model = VoiceIsolationModel(n_fft=checkpoint.get('n_fft', N_FFT))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    window_size = checkpoint.get('window_size', 'medium')
    
    return model, window_size

def process_audio(
    input_path: str,
    output_path: str,
    model_path: str,
    device: torch.device
) -> None:
    """
    Process audio file to isolate voice using the trained model.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save output audio
        model_path: Path to trained model
        device: Device to run inference on
    """
    # Load model
    print("Loading model...")
    model, window_size = load_model(model_path, device)
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(window_size=window_size)
    
    # Load audio
    print(f"Loading audio file: {input_path}")
    audio = preprocessor.load_audio(input_path)
    
    # Compute STFT
    print("Computing STFT...")
    stft = preprocessor.compute_stft(audio)
    
    # Get magnitude and phase
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    
    # Store original dimensions
    orig_shape = magnitude.shape
    
    # Standardize spectrogram for model input
    print("Processing audio with model...")
    magnitude_std = preprocessor.standardize_spectrogram(magnitude)
    
    # Prepare input for model (add batch and channel dimensions)
    model_input = magnitude_std.unsqueeze(0).unsqueeze(0).to(device)
    
    # Generate mask
    with torch.no_grad():
        print("Generating isolation mask...")
        mask = model(model_input).squeeze(0).squeeze(0).cpu()
    
    # If spectrogram was padded, make sure to use only the relevant part of the mask
    if orig_shape[1] < SPEC_TIME_DIM:
        mask = mask[:, :orig_shape[1]]
    else:
        # In case the original was longer (should not happen with standardized processing)
        # Pad the mask with zeros
        temp_mask = torch.zeros(orig_shape, dtype=mask.dtype)
        temp_mask[:, :mask.shape[1]] = mask[:, :orig_shape[1]]
        mask = temp_mask
    
    # Apply mask to isolate voice
    print("Applying mask to isolate voice...")
    isolated_magnitude = magnitude * mask
    
    # Reconstruct complex STFT
    isolated_stft = isolated_magnitude * torch.exp(1j * phase)
    
    # Convert back to audio
    print("Converting processed spectrogram back to audio...")
    isolated_audio = torch.istft(
        isolated_stft,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=torch.hann_window(N_FFT)
    )
    isolated_audio = isolated_audio.unsqueeze(0)  # Add channel dimension
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving processed audio to {output_path}")
    torchaudio.save(
        output_path,
        isolated_audio,
        SAMPLE_RATE
    )
    
    print(f"Processing complete: {input_path} → {output_path}")

def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description='Isolate voice in audio file')
    parser.add_argument('--input', required=True, help='Path to input audio file')
    parser.add_argument('--output', help='Path to save output audio')
    parser.add_argument('--model', required=True, help='Path to trained model')
    
    args = parser.parse_args()
    
    # Set output path if not provided
    if not args.output:
        input_filename = os.path.basename(args.input)
        input_name, input_ext = os.path.splitext(input_filename)
        args.output = os.path.join(OUTPUT_DIR, f"{input_name}_isolated{input_ext}")
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Process audio
    process_audio(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        device=device
    )

if __name__ == "__main__":
    main()
