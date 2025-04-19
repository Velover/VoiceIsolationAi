"""
Script to preprocess and cache audio files for faster training
"""
import os
import torch
import torchaudio
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import shutil
from src.config import VOICE_DIR, NOISE_DIR, CACHE_DIR, SAMPLE_RATE, SUPPORTED_FORMATS
from src.preprocessing import AudioPreprocessor, get_audio_files

def check_directories():
    """Check if required directories exist and have files, create if needed"""
    # Check and create directories
    dirs_to_check = {
        'VOICE': VOICE_DIR,
        'NOISE': NOISE_DIR,
        'CACHE': CACHE_DIR,
        'CACHE/voice': os.path.join(CACHE_DIR, 'voice'),
        'CACHE/noise': os.path.join(CACHE_DIR, 'noise')
    }
    
    for name, dir_path in dirs_to_check.items():
        os.makedirs(dir_path, exist_ok=True)
        
    # Check for audio files
    voice_files = get_audio_files(VOICE_DIR)
    noise_files = get_audio_files(NOISE_DIR)
    
    if not voice_files:
        print("=" * 80)
        print(f"WARNING: No voice files found in {VOICE_DIR}")
        print("Please add some voice audio files (.mp3, .wav, or .flac) to the VOICE directory.")
        print("Example: Recordings of the target person speaking alone")
        print("=" * 80)
        
    if not noise_files:
        print("=" * 80)
        print(f"WARNING: No noise files found in {NOISE_DIR}")
        print("Please add some noise audio files (.mp3, .wav, or .flac) to the NOISE directory.")
        print("Examples:")
        print("  - Recordings of other people talking")
        print("  - Background noise recordings")
        print("  - Any sounds that typically occur with the target voice")
        print("=" * 80)
    
    return len(voice_files) > 0 and len(noise_files) > 0

def preprocess_audio_file(file_path, output_dir, sample_rate=SAMPLE_RATE, window_size='medium'):
    """Preprocess a single audio file and save the spectrogram to disk"""
    try:
        # Get the relative path to maintain folder structure
        rel_path = os.path.basename(file_path)
        # Create output filename (replace extension with .pt)
        output_name = f"{os.path.splitext(rel_path)[0]}.pt"
        output_path = os.path.join(output_dir, output_name)
        
        # Skip if already processed
        if os.path.exists(output_path):
            return True, file_path
            
        # Load audio
        audio, sr = torchaudio.load(file_path)
        
        # Convert to mono if needed
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            audio = resampler(audio)
            
        # Initialize preprocessor
        preprocessor = AudioPreprocessor(window_size=window_size)
        
        # Compute spectrogram
        spect = preprocessor.compute_spectrogram(audio)
        
        # Save as tensor file
        torch.save(spect, output_path)
        
        return True, file_path
    except Exception as e:
        return False, f"Error processing {file_path}: {str(e)}"

def preprocess_directory(input_dir, output_dir, workers=4):
    """Preprocess all audio files in a directory using multiple workers"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all audio files
    audio_files = get_audio_files(input_dir)
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
        
    print(f"Found {len(audio_files)} files in {input_dir}")
    print(f"Processing with {workers} worker threads...")
    
    # Process files in parallel
    success_count = 0
    error_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(preprocess_audio_file, f, output_dir) for f in audio_files]
        
        # Show progress with tqdm
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            success, result = future.result()
            if success:
                success_count += 1
            else:
                error_count += 1
                print(result)
    
    print(f"Preprocessing complete: {success_count} succeeded, {error_count} failed")
    return success_count, error_count

def convert_to_efficient_format(directory, target_format='.flac', sample_rate=SAMPLE_RATE):
    """Convert WAV files to more efficient format to save space"""
    audio_files = [f for f in get_audio_files(directory) if os.path.splitext(f)[1].lower() == '.wav']
    
    if not audio_files:
        print(f"No WAV files found in {directory}")
        return 0
        
    print(f"Converting {len(audio_files)} WAV files to {target_format}...")
    
    converted = 0
    for file_path in tqdm(audio_files):
        try:
            # Load audio
            audio, sr = torchaudio.load(file_path)
            
            # Resample if needed
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                audio = resampler(audio)
            
            # Get new filename
            new_path = os.path.splitext(file_path)[0] + target_format
            
            # Save in new format
            torchaudio.save(new_path, audio, sample_rate)
            
            # Remove original file to save space
            os.remove(file_path)
            converted += 1
            
        except Exception as e:
            print(f"Error converting {file_path}: {e}")
    
    print(f"Converted {converted} files to {target_format}")
    return converted

def extract_example_files():
    """Extract example files if directories are empty"""
    # Check if both directories are empty
    voice_files = get_audio_files(VOICE_DIR)
    noise_files = get_audio_files(NOISE_DIR)
    
    if voice_files or noise_files:
        return  # Don't extract if files already exist
    
    print("No audio files found. Creating example files for testing...")
    
    # Create a simple sine wave audio file as an example
    duration = 3  # seconds
    sample_rate = SAMPLE_RATE
    t = torch.arange(0, duration, 1/sample_rate)
    
    # Voice example (440 Hz tone)
    voice_signal = 0.5 * torch.sin(2 * np.pi * 440 * t)
    # Save as both WAV and FLAC to demonstrate support for both formats
    voice_path_wav = os.path.join(VOICE_DIR, "example_voice.wav")
    voice_path_flac = os.path.join(VOICE_DIR, "example_voice.flac")
    torchaudio.save(voice_path_wav, voice_signal.unsqueeze(0), sample_rate)
    torchaudio.save(voice_path_flac, voice_signal.unsqueeze(0), sample_rate)
    
    # Noise example (white noise)
    noise_signal = 0.2 * torch.randn_like(t)
    noise_path_wav = os.path.join(NOISE_DIR, "example_noise.wav")
    noise_path_flac = os.path.join(NOISE_DIR, "example_noise.flac")
    torchaudio.save(noise_path_wav, noise_signal.unsqueeze(0), sample_rate)
    torchaudio.save(noise_path_flac, noise_signal.unsqueeze(0), sample_rate)
    
    print("Created example files:")
    print(f"  Voice: {voice_path_wav}, {voice_path_flac}")
    print(f"  Noise: {noise_path_wav}, {noise_path_flac}")
    print("NOTE: These are just placeholder examples. Replace them with real audio files.")

def main():
    parser = argparse.ArgumentParser(description='Preprocess audio files for faster training')
    parser.add_argument('--convert', action='store_true', 
                        help='Convert WAV files to more efficient FLAC format')
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                        help='Number of worker threads for preprocessing')
    parser.add_argument('--window-size', choices=['small', 'medium', 'large'], 
                        default='medium', help='Window size for processing')
    parser.add_argument('--create-examples', action='store_true',
                        help='Create example audio files if directories are empty')
    
    args = parser.parse_args()
    
    # Setup cache directories
    voice_cache_dir = os.path.join(CACHE_DIR, 'voice')
    noise_cache_dir = os.path.join(CACHE_DIR, 'noise')
    
    print("=== AUDIO PREPROCESSING UTILITY ===")
    
    # Check if directories exist and have files
    if not check_directories() and args.create_examples:
        extract_example_files()
    
    # Convert files if requested
    if args.convert:
        print("\n=== CONVERTING AUDIO FILES TO EFFICIENT FORMAT ===")
        convert_to_efficient_format(VOICE_DIR)
        convert_to_efficient_format(NOISE_DIR)
    
    # Preprocess files
    print("\n=== PREPROCESSING VOICE FILES ===")
    preprocess_directory(VOICE_DIR, voice_cache_dir, args.workers)
    
    print("\n=== PREPROCESSING NOISE FILES ===")
    preprocess_directory(NOISE_DIR, noise_cache_dir, args.workers)
    
    print("\nPreprocessing complete! You can now train with cached data.")
    print("Run: python main.py train --use-cache --gpu --mixed-precision --deep-model")
    
    # Final check - if directories are still empty after processing
    voice_files = get_audio_files(VOICE_DIR)
    noise_files = get_audio_files(NOISE_DIR)
    
    if not voice_files or not noise_files:
        print("\n" + "=" * 80)
        print("IMPORTANT: One or more directories are still empty!")
        if not voice_files:
            print(" - VOICE directory is empty. Add target person's voice recordings.")
        if not noise_files:
            print(" - NOISE directory is empty. Add background noise recordings.")
        print("\nUse the --create-examples flag to create test files for demonstration.")
        print("=" * 80)

if __name__ == "__main__":
    main()
