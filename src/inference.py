import os
import torch
import torchaudio
import argparse
from typing import Optional, Tuple
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

from .config import OUTPUT_DIR, N_FFT, HOP_LENGTH, SAMPLE_RATE, SPEC_TIME_DIM, WINDOW_SIZES
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
    
    # Get model type (deep or standard)
    is_deep_model = checkpoint.get('deep_model', False)
    
    # Create appropriate model and load state
    if is_deep_model:
        from .model import VoiceIsolationModelDeep
        model = VoiceIsolationModelDeep(n_fft=checkpoint.get('n_fft', N_FFT))
    else:
        from .model import VoiceIsolationModel
        model = VoiceIsolationModel(n_fft=checkpoint.get('n_fft', N_FFT))
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    window_size = checkpoint.get('window_size', 'medium')
    
    # Print model information for clarity
    print(f"Model information:")
    print(f"  - Window size: {window_size}")
    print(f"  - Model type: {'Deep' if is_deep_model else 'Standard'}")
    print(f"  - Created on: {checkpoint.get('creation_date', 'unknown')}")
    print(f"  - Training samples: {checkpoint.get('training_samples', 'unknown')}")
    
    return model, window_size

def process_audio(
    input_path: str,
    output_path: str,
    model_path: str,
    device: torch.device,
    verbose: bool = True
) -> None:
    """
    Process audio file to isolate voice using the trained model.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save output audio
        model_path: Path to trained model
        device: Device to run inference on
        verbose: Whether to print detailed progress
    """
    process_start = time.time()
    if verbose:
        print(f"=== VOICE ISOLATION PROCESSING ===")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Model: {model_path}")
        print(f"Device: {device}")
    
    # Load model
    if verbose:
        print("\n[1/6] Loading model...", end="")
    start = time.time()
    model, window_size = load_model(model_path, device)
    duration = time.time() - start
    if verbose:
        print(f" Done in {duration:.2f}s")
        print(f"       Window size: {window_size} ({WINDOW_SIZES.get(window_size, 'unknown')}ms)")
    
    # Initialize preprocessor with use_gpu flag based on device
    use_gpu = device.type == 'cuda'
    preprocessor = AudioPreprocessor(window_size=window_size, use_gpu=use_gpu)
    
    # Load audio
    if verbose:
        print(f"[2/6] Loading audio file...", end="")
    start = time.time()
    audio = preprocessor.load_audio(input_path)
    duration = time.time() - start
    duration_seconds = audio.shape[1] / SAMPLE_RATE
    if verbose:
        print(f" Done in {duration:.2f}s (Audio length: {duration_seconds:.2f}s)")
    
    # Compute STFT
    if verbose:
        print("[3/6] Computing STFT...", end="")
    start = time.time()
    stft = preprocessor.compute_stft(audio)
    duration = time.time() - start
    if verbose:
        print(f" Done in {duration:.2f}s")
    
    # Get magnitude and phase
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    
    # Store original dimensions
    orig_shape = magnitude.shape
    
    # Standardize spectrogram for model input
    if verbose:
        print("[4/6] Processing with model...", end="")
    start = time.time()
    magnitude_std = preprocessor.standardize_spectrogram(magnitude)
    
    # Prepare input for model (add batch and channel dimensions)
    model_input = magnitude_std.unsqueeze(0).unsqueeze(0).to(device)
    
    # Display model input shape if verbose
    if verbose:
        print(f"\n       Input shape: {model_input.shape}", end="")
    
    # Generate mask with proper device handling for autocast
    with torch.no_grad():
        # Correct autocast usage for PyTorch 2.6.0
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=device.type == 'cuda', dtype=torch.float16):
            mask = model(model_input).squeeze(0).squeeze(0).cpu()
    
    duration = time.time() - start
    if verbose:
        print(f"\r[4/6] Processing with model... Done in {duration:.2f}s" + " " * 40)
    
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
    if verbose:
        print("[5/6] Applying isolation mask...", end="")
    start = time.time()
    
    # Move everything to CPU before operations to ensure consistent device
    magnitude = magnitude.cpu()
    phase = phase.cpu()
    mask = mask.cpu()
    
    isolated_magnitude = magnitude * mask
    duration = time.time() - start
    if verbose:
        print(f" Done in {duration:.2f}s")
    
    # Reconstruct complex STFT
    isolated_stft = isolated_magnitude * torch.exp(1j * phase)
    
    # Convert back to audio
    if verbose:
        print("[6/6] Converting back to audio...", end="")
    start = time.time()
    
    # Ensure window is on the same device as the STFT
    hann_window = torch.hann_window(N_FFT).to(isolated_stft.device)
    
    isolated_audio = torch.istft(
        isolated_stft,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=hann_window
    )
    isolated_audio = isolated_audio.unsqueeze(0)  # Add channel dimension
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(
        output_path,
        isolated_audio,
        SAMPLE_RATE
    )
    duration = time.time() - start
    if verbose:
        print(f" Done in {duration:.2f}s")
    
    total_time = time.time() - process_start
    if verbose:
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Processing ratio: {total_time/duration_seconds:.2f}x realtime")
        print(f"Output saved to: {output_path}")

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
