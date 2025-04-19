import os
import argparse
import numpy as np
import torch
import torchaudio
import random
from tqdm import tqdm
import librosa
import soundfile as sf

from utils import load_audio, mix_audio, save_audio
from config import *

def remove_silence(audio, sr=SAMPLE_RATE, threshold_db=-40, min_silence_duration=0.3):
    """Remove silence segments from audio."""
    # Convert to mono if stereo
    if isinstance(audio, torch.Tensor) and audio.dim() > 1:
        audio = torch.mean(audio, dim=0)
    
    # Convert to numpy if tensor
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
    
    # Calculate amplitude threshold
    threshold_amplitude = 10**(threshold_db/20)
    
    # Compute energy
    energy = librosa.feature.rms(y=audio)[0]
    
    # Find segments above threshold
    non_silent = energy > threshold_amplitude
    
    # Convert to frames to samples
    hop_length = 512  # Default hop_length for rms
    non_silent_samples = librosa.frames_to_samples(np.where(non_silent)[0], hop_length=hop_length)
    
    # No non-silent parts found
    if len(non_silent_samples) == 0:
        return np.array([])
    
    # Minimum silence duration in samples
    min_silence_samples = int(min_silence_duration * sr)
    
    # Initialize output audio
    output_audio = []
    
    # Process segments
    start_idx = non_silent_samples[0]
    for i in range(1, len(non_silent_samples)):
        if non_silent_samples[i] - non_silent_samples[i-1] > min_silence_samples:
            # End of segment
            end_idx = non_silent_samples[i-1] + hop_length
            if end_idx - start_idx > 0:
                output_audio.append(audio[start_idx:end_idx])
            # Start new segment
            start_idx = non_silent_samples[i]
    
    # Add last segment
    if len(non_silent_samples) > 0:
        end_idx = min(non_silent_samples[-1] + hop_length, len(audio))
        if end_idx - start_idx > 0:
            output_audio.append(audio[start_idx:end_idx])
    
    # Concatenate segments
    if output_audio:
        return np.concatenate(output_audio)
    else:
        return np.array([])

def process_audio_file(file_path, output_dir, remove_silence_flag=True):
    """Process a single audio file and save to output directory."""
    # Create output filename
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, filename)
    
    # Skip if output file already exists
    if os.path.exists(output_path):
        return output_path
    
    # Load audio
    try:
        audio = load_audio(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    # Remove silence if needed
    if remove_silence_flag:
        audio = remove_silence(audio)
        
        # Skip if no audio left after silence removal
        if len(audio) == 0:
            print(f"Warning: No audio left after silence removal in {file_path}")
            return None
    
    # Save processed audio
    try:
        os.makedirs(output_dir, exist_ok=True)
        save_audio(audio, output_path)
        return output_path
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
        return None

def generate_mixed_samples(voice_files, noise_files, output_dir, num_samples=5):
    """Generate mixed samples of voice and noise."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Select random files
        voice_file = random.choice(voice_files)
        noise_file = random.choice(noise_files)
        
        try:
            # Load audio
            voice = load_audio(voice_file)
            noise = load_audio(noise_file)
            
            # Ensure both are long enough
            min_length = min(len(voice), len(noise))
            if min_length < SAMPLE_RATE:  # Skip if less than 1 second
                continue
                
            # Trim to same length
            voice = voice[:min_length]
            noise = noise[:min_length]
            
            # Create different SNR mixes
            snr_values = [-5, 0, 5, 10, 15]
            
            for snr in snr_values:
                # Mix voice and noise
                mixed, _ = mix_audio(voice, noise, snr)
                
                # Create filename
                voice_name = os.path.splitext(os.path.basename(voice_file))[0]
                noise_name = os.path.splitext(os.path.basename(noise_file))[0]
                output_file = os.path.join(output_dir, f"mix_{voice_name}_{noise_name}_snr{snr}_{i}.wav")
                
                # Save mixed audio
                save_audio(mixed, output_file)
                
                print(f"Created mixed sample: {output_file}")
            
        except Exception as e:
            print(f"Error generating mixed sample: {e}")

def prepare_training_data(remove_silence_flag=True, generate_samples=False):
    """Process all audio files and prepare for training."""
    # Create output directories
    os.makedirs(PREPROCESSED_VOICE, exist_ok=True)
    os.makedirs(PREPROCESSED_NOISE, exist_ok=True)
    
    # Process voice files
    print("\nProcessing voice files...")
    voice_files = []
    for filename in tqdm(os.listdir(VOICE_DIR)):
        if filename.endswith(('.mp3', '.wav', '.webm')):
            file_path = os.path.join(VOICE_DIR, filename)
            output_path = process_audio_file(file_path, PREPROCESSED_VOICE, remove_silence_flag)
            if output_path:
                voice_files.append(output_path)
    
    # Process noise files
    print("\nProcessing noise files...")
    noise_files = []
    for filename in tqdm(os.listdir(NOISE_DIR)):
        if filename.endswith(('.mp3', '.wav', '.webm')):
            file_path = os.path.join(NOISE_DIR, filename)
            output_path = process_audio_file(file_path, PREPROCESSED_NOISE, remove_silence_flag)
            if output_path:
                noise_files.append(output_path)
    
    # Generate mixed samples if requested
    if generate_samples:
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        print("\nGenerating sample mixtures...")
        generate_mixed_samples(voice_files, noise_files, SAMPLES_DIR)
    
    print("\nData preparation complete!")
    print(f"Processed {len(voice_files)} voice files -> {PREPROCESSED_VOICE}")
    print(f"Processed {len(noise_files)} noise files -> {PREPROCESSED_NOISE}")
    if generate_samples:
        print(f"Sample mixtures saved to {SAMPLES_DIR}")

def main():
    parser = argparse.ArgumentParser(description='Prepare audio data for voice isolation training')
    parser.add_argument('--keep-silence', action='store_true', help='Keep silence in audio files (default: remove)')
    parser.add_argument('--generate-samples', action='store_true', help='Generate sample mixtures for listening')
    parser.add_argument('--window-size', type=str, choices=['SMALL', 'MEDIUM', 'LARGE', 'XLARGE'], 
                        default='SMALL', help='Window size to use for sample generation (default: SMALL)')
    
    args = parser.parse_args()
    
    # Process audio and generate samples
    print("Preparing training data...")
    prepare_training_data(
        remove_silence_flag=not args.keep_silence,
        generate_samples=args.generate_samples
    )

if __name__ == "__main__":
    main()
