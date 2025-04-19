"""
Generate test audio files by mixing voice and noise samples
"""
import os
import torch
import torchaudio
import argparse
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import random

from src.preprocessing import AudioPreprocessor, get_audio_files
from src.config import VOICE_DIR, NOISE_DIR, SAMPLE_RATE

def create_mixed_test_file(
    voice_path: str, 
    noise_path: str, 
    output_path: str, 
    snr_db: float = 5.0,
    target_duration: int = 20,
    save_voice: bool = True,
    save_noise: bool = False,
    voice_output_dir: str = None,
    noise_output_dir: str = None
):
    """
    Create a test file by mixing voice and noise
    
    Args:
        voice_path: Path to voice file
        noise_path: Path to noise file
        output_path: Path to save mixed file
        snr_db: Signal-to-noise ratio in dB
        target_duration: Target duration in seconds
        save_voice: Whether to save the isolated voice file
        save_noise: Whether to save the isolated noise file
        voice_output_dir: Directory to save isolated voice
        noise_output_dir: Directory to save isolated noise
    """
    # Initialize preprocessor
    preprocessor = AudioPreprocessor()
    
    # Load voice and noise
    voice = preprocessor.load_audio(voice_path)
    noise = preprocessor.load_audio(noise_path)
    
    # Get file names for output
    voice_name = os.path.splitext(os.path.basename(voice_path))[0]
    noise_name = os.path.splitext(os.path.basename(noise_path))[0]
    
    # Calculate signal-to-noise ratio
    snr = 10 ** (snr_db / 20)
    
    # Get voice and noise durations
    voice_duration = voice.shape[1] / SAMPLE_RATE
    noise_duration = noise.shape[1] / SAMPLE_RATE
    
    # Adjust durations to match target
    target_samples = target_duration * SAMPLE_RATE
    
    # If voice is too short, repeat it
    if voice.shape[1] < target_samples:
        repeats = (target_samples // voice.shape[1]) + 1
        voice = voice.repeat(1, repeats)
    
    # If noise is too short, repeat it
    if noise.shape[1] < target_samples:
        repeats = (target_samples // noise.shape[1]) + 1
        noise = noise.repeat(1, repeats)
        
    # Trim to target duration
    voice = voice[:, :target_samples]
    noise = noise[:, :target_samples]
    
    # Normalize both to have peak amplitude of 1.0
    voice = voice / voice.abs().max()
    noise = noise / noise.abs().max()
    
    # Apply SNR adjustment to noise
    noise_adjusted = noise / snr
    
    # Mix voice and noise
    mixed = voice + noise_adjusted
    
    # Normalize mixed to have peak amplitude of 0.9 (avoid clipping)
    mixed = 0.9 * mixed / mixed.abs().max()
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # *** FIX: Move tensors to CPU before saving ***
    mixed_cpu = mixed.cpu() if mixed.device.type == 'cuda' else mixed
    voice_cpu = voice.cpu() if voice.device.type == 'cuda' else voice
    noise_adjusted_cpu = noise_adjusted.cpu() if noise_adjusted.device.type == 'cuda' else noise_adjusted
    
    # Save mixed audio
    torchaudio.save(output_path, mixed_cpu, SAMPLE_RATE)
    
    # Save isolated voice if requested
    if save_voice and voice_output_dir:
        voice_output_path = os.path.join(voice_output_dir, f"{voice_name}_clean.wav")
        os.makedirs(voice_output_dir, exist_ok=True)
        torchaudio.save(voice_output_path, voice_cpu, SAMPLE_RATE)
    
    # Save isolated noise if requested
    if save_noise and noise_output_dir:
        noise_output_path = os.path.join(noise_output_dir, f"{noise_name}_only.wav")
        os.makedirs(noise_output_dir, exist_ok=True)
        torchaudio.save(noise_output_path, noise_adjusted_cpu, SAMPLE_RATE)
    
    return {
        'voice_path': voice_path,
        'noise_path': noise_path,
        'mixed_path': output_path,
        'voice_duration': voice_duration,
        'noise_duration': noise_duration,
        'snr_db': snr_db
    }

def create_test_files(
    output_dir: str,
    num_files: int = 5,
    duration: int = 20,
    min_snr: float = 3.0,
    max_snr: float = 10.0,
    voice_dir: str = VOICE_DIR,
    noise_dir: str = NOISE_DIR
):
    """
    Create multiple test audio files
    
    Args:
        output_dir: Directory to save test files
        num_files: Number of test files to generate
        duration: Duration of each file in seconds
        min_snr: Minimum SNR in dB
        max_snr: Maximum SNR in dB
        voice_dir: Directory with voice files
        noise_dir: Directory with noise files
    """
    # Create output directories
    mixed_dir = os.path.join(output_dir, 'MIXED')
    voice_dir_out = os.path.join(output_dir, 'VOICE')
    noise_dir_out = os.path.join(output_dir, 'NOISE')
    
    os.makedirs(mixed_dir, exist_ok=True)
    os.makedirs(voice_dir_out, exist_ok=True)
    os.makedirs(noise_dir_out, exist_ok=True)
    
    # Get voice and noise files
    voice_files = get_audio_files(voice_dir)
    noise_files = get_audio_files(noise_dir)
    
    if not voice_files:
        raise ValueError(f"No voice files found in {voice_dir}")
    if not noise_files:
        raise ValueError(f"No noise files found in {noise_dir}")
    
    print(f"Found {len(voice_files)} voice files and {len(noise_files)} noise files")
    
    # Generate test files
    print(f"Generating {num_files} test files with duration {duration}s")
    print(f"SNR range: {min_snr}dB to {max_snr}dB")
    
    results = []
    for i in tqdm(range(num_files), desc="Generating test files"):
        # Select random voice and noise files
        voice_path = random.choice(voice_files)
        noise_path = random.choice(noise_files)
        
        # Generate random SNR
        snr_db = random.uniform(min_snr, max_snr)
        
        # Create output filename
        voice_name = os.path.splitext(os.path.basename(voice_path))[0]
        noise_name = os.path.splitext(os.path.basename(noise_path))[0]
        output_filename = f"mixed_{voice_name}_{noise_name}_snr{snr_db:.1f}_{i+1}.wav"
        output_path = os.path.join(mixed_dir, output_filename)
        
        # Create test file
        result = create_mixed_test_file(
            voice_path=voice_path,
            noise_path=noise_path,
            output_path=output_path,
            snr_db=snr_db,
            target_duration=duration,
            save_voice=True,
            save_noise=True,
            voice_output_dir=voice_dir_out,
            noise_output_dir=noise_dir_out
        )
        
        results.append(result)
    
    # Print summary
    print(f"\nGenerated {len(results)} test files")
    print(f"  Mixed files saved to: {mixed_dir}")
    print(f"  Clean voice files saved to: {voice_dir_out}")
    print(f"  Noise files saved to: {noise_dir_out}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Generate test audio files by mixing voice and noise')
    parser.add_argument('--output-dir', default='TEST', help='Directory to save test files')
    parser.add_argument('--num-files', type=int, default=5, help='Number of test files to generate')
    parser.add_argument('--duration', type=int, default=20, help='Duration of each file in seconds')
    parser.add_argument('--min-snr', type=float, default=3.0, help='Minimum SNR in dB')
    parser.add_argument('--max-snr', type=float, default=10.0, help='Maximum SNR in dB')
    parser.add_argument('--voice-dir', default=VOICE_DIR, help='Directory with voice files')
    parser.add_argument('--noise-dir', default=NOISE_DIR, help='Directory with noise files')
    
    args = parser.parse_args()
    
    create_test_files(
        output_dir=args.output_dir,
        num_files=args.num_files,
        duration=args.duration,
        min_snr=args.min_snr,
        max_snr=args.max_snr,
        voice_dir=args.voice_dir,
        noise_dir=args.noise_dir
    )

if __name__ == "__main__":
    main()
