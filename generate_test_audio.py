"""
Generate test audio files by mixing voice and noise samples
"""
import os
import torch
import torchaudio
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.preprocessing import get_audio_files
from src.config import VOICE_DIR, NOISE_DIR, SAMPLE_RATE

def calculate_snr(voice, noise, target_snr_db):
    """
    Scale noise to achieve target signal-to-noise ratio.
    
    Args:
        voice: Voice audio tensor
        noise: Noise audio tensor
        target_snr_db: Desired SNR in decibels
        
    Returns:
        Scaled noise tensor
    """
    # Calculate signal power
    voice_power = torch.mean(voice ** 2)
    noise_power = torch.mean(noise ** 2)
    
    # Calculate scaling factor for noise
    snr_linear = 10 ** (target_snr_db / 10)
    scaling_factor = torch.sqrt(voice_power / (noise_power * snr_linear))
    
    # Scale noise
    scaled_noise = noise * scaling_factor
    
    return scaled_noise

def create_test_files(output_dir='TEST', num_files=5, duration=20, min_snr=3.0, max_snr=10.0):
    """
    Create test audio files by mixing voice and noise with controlled SNR.
    
    Args:
        output_dir: Base directory for test files
        num_files: Number of test files to generate
        duration: Duration of each file in seconds
        min_snr: Minimum SNR in dB
        max_snr: Maximum SNR in dB
    """
    # Create output directories
    voice_dir = os.path.join(output_dir, 'VOICE')
    noise_dir = os.path.join(output_dir, 'NOISE')
    mixed_dir = os.path.join(output_dir, 'MIXED')
    
    os.makedirs(voice_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)
    os.makedirs(mixed_dir, exist_ok=True)
    
    # Get source files
    voice_files = get_audio_files(VOICE_DIR)
    noise_files = get_audio_files(NOISE_DIR)
    
    if not voice_files:
        print(f"No voice files found in {VOICE_DIR}")
        return
    
    if not noise_files:
        print(f"No noise files found in {NOISE_DIR}")
        return
    
    print(f"Found {len(voice_files)} voice files and {len(noise_files)} noise files")
    print(f"Generating {num_files} test files with {duration}s duration")
    print(f"SNR range: {min_snr} to {max_snr} dB")
    
    # Generate test files
    for i in tqdm(range(num_files), desc="Generating test files"):
        try:
            # Select random files
            voice_path = random.choice(voice_files)
            noise_path = random.choice(noise_files)
            
            voice_name = os.path.splitext(os.path.basename(voice_path))[0]
            noise_name = os.path.splitext(os.path.basename(noise_path))[0]
            
            # Load voice audio
            voice, sr_voice = torchaudio.load(voice_path)
            if sr_voice != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sr_voice, new_freq=SAMPLE_RATE)
                voice = resampler(voice)
            
            # Make mono if needed
            if voice.shape[0] > 1:
                voice = torch.mean(voice, dim=0, keepdim=True)
            
            # Load noise audio
            noise, sr_noise = torchaudio.load(noise_path)
            if sr_noise != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sr_noise, new_freq=SAMPLE_RATE)
                noise = resampler(noise)
            
            # Make mono if needed
            if noise.shape[0] > 1:
                noise = torch.mean(noise, dim=0, keepdim=True)
            
            # Create segments of the desired duration
            samples_needed = duration * SAMPLE_RATE
            
            # Loop voice and noise if needed
            while voice.shape[1] < samples_needed:
                voice = torch.cat([voice, voice], dim=1)
            
            while noise.shape[1] < samples_needed:
                noise = torch.cat([noise, noise], dim=1)
            
            # Trim to desired length
            voice = voice[:, :samples_needed]
            noise = noise[:, :samples_needed]
            
            # Normalize to prevent clipping
            voice = voice / voice.abs().max()
            noise = noise / noise.abs().max()
            
            # Generate random SNR
            snr_db = random.uniform(min_snr, max_snr)
            
            # Scale noise to achieve target SNR
            scaled_noise = calculate_snr(voice, noise, snr_db)
            
            # Create mixed audio
            mixed = voice + scaled_noise
            
            # Normalize mixed audio to prevent clipping
            mixed = mixed / mixed.abs().max() * 0.9
            
            # Save files
            voice_output = os.path.join(voice_dir, f"voice_{voice_name}_{i+1}.wav")
            noise_output = os.path.join(noise_dir, f"noise_{noise_name}_{i+1}.wav")
            mixed_output = os.path.join(mixed_dir, f"mixed_voice{i+1}_noise{i+1}_snr{snr_db:.1f}_{i+1}.wav")
            
            torchaudio.save(voice_output, voice, SAMPLE_RATE)
            torchaudio.save(noise_output, scaled_noise, SAMPLE_RATE)
            torchaudio.save(mixed_output, mixed, SAMPLE_RATE)
            
        except Exception as e:
            print(f"Error generating test file {i+1}: {e}")
    
    # Print summary
    print("\nTest file generation complete!")
    print(f"Clean voice files saved to: {voice_dir}")
    print(f"Noise files saved to: {noise_dir}")
    print(f"Mixed audio files saved to: {mixed_dir}")
    print("\nYou can test your model with:")
    print(f"python main.py isolate --input {os.path.join(mixed_dir, 'mixed_voice1_noise1_snr*.wav')} --model OUTPUT/your_model.pth")

def main():
    parser = argparse.ArgumentParser(description='Generate test audio by mixing voice and noise')
    parser.add_argument('--output-dir', default='TEST', help='Directory to save test files')
    parser.add_argument('--num-files', type=int, default=5, help='Number of test files to generate')
    parser.add_argument('--duration', type=int, default=20, help='Duration of each file in seconds')
    parser.add_argument('--min-snr', type=float, default=3.0, help='Minimum SNR in dB')
    parser.add_argument('--max-snr', type=float, default=10.0, help='Maximum SNR in dB')
    
    args = parser.parse_args()
    
    create_test_files(
        output_dir=args.output_dir,
        num_files=args.num_files,
        duration=args.duration,
        min_snr=args.min_snr,
        max_snr=args.max_snr
    )

if __name__ == "__main__":
    main()
