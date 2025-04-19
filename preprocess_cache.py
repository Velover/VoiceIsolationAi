"""
Script to preprocess and cache audio files for faster training
"""
import os
import argparse
import torch
import torchaudio
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
import traceback

# Import project modules
from src.preprocessing import AudioPreprocessor, get_audio_files
from src.config import (
    VOICE_DIR, NOISE_DIR, CACHE_DIR, SAMPLE_RATE, 
    SUPPORTED_FORMATS, WINDOW_SIZES, DEFAULT_WINDOW_SIZE
)

def process_file(file_path, output_dir, window_size='medium', convert=False):
    """
    Process a single audio file: convert format if needed and cache spectrogram.
    
    Args:
        file_path: Path to audio file
        output_dir: Directory to save processed files
        window_size: Processing window size
        convert: Whether to convert WAV files to more efficient format
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Initialize preprocessor (no GPU for multiprocessing compatibility)
        preprocessor = AudioPreprocessor(window_size=window_size, use_gpu=False)
        
        # Get filename and extension
        file_name = os.path.basename(file_path)
        file_base, file_ext = os.path.splitext(file_name)
        spec_cache_path = os.path.join(output_dir, f"{file_base}_spec.pt")
        
        # Skip if spectrogram cache already exists
        if os.path.exists(spec_cache_path):
            return (True, f"Skipped {file_name} (cache exists)")
        
        # Convert format if requested
        if convert and file_ext.lower() == '.wav':
            flac_path = os.path.join(output_dir, f"{file_base}.flac")
            
            # Skip if FLAC already exists
            if os.path.exists(flac_path):
                file_path = flac_path
            else:
                # Load audio
                audio, sr = torchaudio.load(file_path)
                
                # Convert to mono if needed
                if audio.shape[0] > 1:
                    audio = torch.mean(audio, dim=0, keepdim=True)
                
                # Resample if needed
                if sr != SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                    audio = resampler(audio)
                
                # Save as FLAC
                torchaudio.save(flac_path, audio, SAMPLE_RATE)
                file_path = flac_path
        
        # Load audio
        audio = preprocessor.load_audio(file_path)
        
        # Compute spectrogram
        spectrogram = preprocessor.compute_spectrogram(audio)
        
        # Save spectrogram
        torch.save(spectrogram, spec_cache_path)
        
        return (True, f"Processed {file_name}")
    except Exception as e:
        error_details = traceback.format_exc()
        return (False, f"Error processing {file_path}: {str(e)}\n{error_details}")

def create_example_files():
    """
    Create example audio files if the VOICE and NOISE directories are empty.
    
    This helps new users get started with sample data.
    """
    voice_files = get_audio_files(VOICE_DIR)
    noise_files = get_audio_files(NOISE_DIR)
    
    if voice_files and noise_files:
        print("Example files not needed - voice and noise directories already contain files.")
        return
    
    print("Creating example audio files...")
    
    # Create a simple sine wave for voice example
    if not voice_files:
        print("Creating example voice files...")
        for i in range(1, 4):
            # Create sine wave at different frequencies
            duration = 3.0  # 3 seconds
            frequency = 220 * i  # Different frequencies
            sample_rate = SAMPLE_RATE
            t = torch.arange(0, duration, 1.0/sample_rate)
            
            # Add some amplitude modulation to make it more voice-like
            modulation = 0.5 + 0.5 * torch.sin(2 * torch.pi * 2 * t)
            audio = torch.sin(2 * torch.pi * frequency * t) * modulation
            
            # Add a bit of noise
            audio = audio + 0.05 * torch.randn_like(audio)
            
            # Normalize
            audio = audio / audio.abs().max()
            
            # Save as WAV
            os.makedirs(VOICE_DIR, exist_ok=True)
            output_path = os.path.join(VOICE_DIR, f"example_voice_{i}.wav")
            torchaudio.save(output_path, audio.unsqueeze(0), sample_rate)
            print(f"Created {output_path}")
    
    # Create noise examples
    if not noise_files:
        print("Creating example noise files...")
        for i in range(1, 4):
            # Create different types of noise
            duration = 3.0  # 3 seconds
            sample_rate = SAMPLE_RATE
            samples = int(duration * sample_rate)
            
            if i == 1:
                # White noise
                audio = torch.randn(samples)
            elif i == 2:
                # Colored noise (brown-ish)
                audio = torch.zeros(samples)
                current = 0
                for j in range(samples):
                    current += random.uniform(-1, 1) * 0.01
                    current = max(-0.5, min(0.5, current))  # Clamp
                    audio[j] = current
            else:
                # Sine wave mixture (simulating background music)
                t = torch.arange(0, duration, 1.0/sample_rate)
                audio = torch.sin(2 * torch.pi * 440 * t) * 0.3
                audio += torch.sin(2 * torch.pi * 330 * t) * 0.2
                audio += torch.sin(2 * torch.pi * 247 * t) * 0.15
            
            # Normalize
            audio = audio / audio.abs().max() * 0.8
            
            # Save as WAV
            os.makedirs(NOISE_DIR, exist_ok=True)
            output_path = os.path.join(NOISE_DIR, f"example_noise_{i}.wav")
            torchaudio.save(output_path, audio.unsqueeze(0), sample_rate)
            print(f"Created {output_path}")
    
    print("Example files created successfully.")

def preprocess_files(workers=None, convert=False, window_size=DEFAULT_WINDOW_SIZE, create_examples=False):
    """
    Preprocess all audio files in VOICE and NOISE directories.
    
    Args:
        workers: Number of worker processes (None = auto-detect)
        convert: Whether to convert WAV files to more efficient FLAC format
        window_size: Processing window size
        create_examples: Whether to create example files if directories are empty
    """
    # Set up output directories
    voice_cache_dir = os.path.join(CACHE_DIR, 'voice')
    noise_cache_dir = os.path.join(CACHE_DIR, 'noise')
    os.makedirs(voice_cache_dir, exist_ok=True)
    os.makedirs(noise_cache_dir, exist_ok=True)
    
    # Create example files if requested
    if create_examples:
        create_example_files()
    
    # Get all audio files
    voice_files = get_audio_files(VOICE_DIR)
    noise_files = get_audio_files(NOISE_DIR)
    
    if not voice_files:
        print(f"No voice files found in {VOICE_DIR}")
        return
    
    if not noise_files:
        print(f"No noise files found in {NOISE_DIR}")
        return
    
    total_files = len(voice_files) + len(noise_files)
    print(f"Found {len(voice_files)} voice files and {len(noise_files)} noise files")
    
    # Determine number of workers
    if workers is None:
        workers = min(os.cpu_count(), 8)  # Limit to 8 workers maximum
        
    print(f"Starting preprocessing with {workers} workers")
    print(f"Window size: {window_size}")
    if convert:
        print("Will convert WAV files to FLAC for more efficient storage")
    
    # Process files with ProcessPoolExecutor
    start_time = time.time()
    
    # Process voice files
    print("\nProcessing voice files...")
    success_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_file, file_path, voice_cache_dir, window_size, convert): file_path
            for file_path in voice_files
        }
        
        # Process as they complete
        for future in tqdm(future_to_file, total=len(voice_files)):
            file_path = future_to_file[future]
            try:
                success, message = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    print(f"\nError: {message}")
            except Exception as e:
                error_count += 1
                print(f"\nError processing {file_path}: {e}")
    
    # Process noise files
    print("\nProcessing noise files...")
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_file, file_path, noise_cache_dir, window_size, convert): file_path
            for file_path in noise_files
        }
        
        # Process as they complete
        for future in tqdm(future_to_file, total=len(noise_files)):
            file_path = future_to_file[future]
            try:
                success, message = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    print(f"\nError: {message}")
            except Exception as e:
                error_count += 1
                print(f"\nError processing {file_path}: {e}")
    
    # Print summary
    duration = time.time() - start_time
    print(f"\nPreprocessing completed in {duration:.2f} seconds")
    print(f"Successfully processed {success_count}/{total_files} files")
    if error_count > 0:
        print(f"Encountered errors with {error_count} files")
    
    # Print cache stats
    voice_cache_files = [f for f in os.listdir(voice_cache_dir) if f.endswith('_spec.pt')]
    noise_cache_files = [f for f in os.listdir(noise_cache_dir) if f.endswith('_spec.pt')]
    
    print(f"\nCache summary:")
    print(f"  Voice spectrograms: {len(voice_cache_files)}")
    print(f"  Noise spectrograms: {len(noise_cache_files)}")
    
    if convert:
        voice_flac_files = [f for f in os.listdir(voice_cache_dir) if f.endswith('.flac')]
        noise_flac_files = [f for f in os.listdir(noise_cache_dir) if f.endswith('.flac')]
        print(f"  Converted FLAC files: {len(voice_flac_files) + len(noise_flac_files)}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess audio files for training')
    parser.add_argument('--workers', type=int, default=None, 
                        help='Number of worker processes (default: auto-detect)')
    parser.add_argument('--convert', action='store_true',
                        help='Convert WAV files to more efficient FLAC format')
    parser.add_argument('--window-size', choices=list(WINDOW_SIZES.keys()), default=DEFAULT_WINDOW_SIZE,
                        help=f'Window size for processing (default: {DEFAULT_WINDOW_SIZE})')
    parser.add_argument('--create-examples', action='store_true',
                        help='Create example audio files if directories are empty')
    parser.add_argument('--samples', type=int, default=0,
                        help='Number of training samples to preprocess (0 for all)')
    
    args = parser.parse_args()
    
    preprocess_files(
        workers=args.workers,
        convert=args.convert,
        window_size=args.window_size,
        create_examples=args.create_examples
    )

if __name__ == "__main__":
    main()
