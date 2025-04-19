"""
Preprocess and cache training samples to speed up training
"""
import os
import torch
import numpy as np
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time

from src.preprocessing import AudioPreprocessor, get_audio_files
from src.config import (
    VOICE_DIR, NOISE_DIR, CACHE_DIR, SAMPLE_RATE, 
    WINDOW_SIZES, DEFAULT_WINDOW_SIZE, SPEC_TIME_DIM
)

def create_sample(sample_id, voice_files, noise_files, output_dir, window_size='medium', use_gpu=False):
    """
    Create and save a single training sample.
    
    Args:
        sample_id: Sample identifier
        voice_files: List of voice audio files
        noise_files: List of noise audio files
        output_dir: Directory to save samples
        window_size: Processing window size
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Initialize preprocessor with given window size
        preprocessor = AudioPreprocessor(window_size=window_size, use_gpu=use_gpu)
        
        # Select random files
        voice_path = random.choice(voice_files)
        noise_path = random.choice(noise_files)
        
        # Random mix ratio
        mix_ratio = random.uniform(0.3, 0.7)
        
        # Create training example
        mixed_spec, mask = preprocessor.create_training_example(
            voice_path, noise_path, mix_ratio=mix_ratio
        )
        
        # Move tensors to CPU if they're on GPU
        if mixed_spec.device.type == 'cuda':
            mixed_spec = mixed_spec.cpu()
            mask = mask.cpu()
        
        # Save sample to disk
        output_path = os.path.join(output_dir, f"sample_{sample_id:05d}.pt")
        torch.save(
            {
                'mixed': mixed_spec,
                'mask': mask,
                'voice_path': voice_path,
                'noise_path': noise_path,
                'mix_ratio': mix_ratio
            },
            output_path
        )
        
        return (True, f"Created sample {sample_id}")
    except Exception as e:
        return (False, f"Error creating sample {sample_id}: {e}")

def generate_samples(num_samples=1000, window_size=DEFAULT_WINDOW_SIZE, 
                   output_dir=None, workers=None, use_gpu=False, force=False):
    """
    Generate training samples and save them to disk.
    
    Args:
        num_samples: Number of samples to generate
        window_size: Processing window size
        output_dir: Directory to save samples
        workers: Number of worker processes (None = auto-detect)
        use_gpu: Whether to use GPU acceleration
        force: Whether to force regeneration of existing samples
    """
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(CACHE_DIR, f"samples_{window_size}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get audio files
    voice_files = get_audio_files(VOICE_DIR)
    noise_files = get_audio_files(NOISE_DIR)
    
    if not voice_files:
        print(f"No voice files found in {VOICE_DIR}")
        return
    
    if not noise_files:
        print(f"No noise files found in {NOISE_DIR}")
        return
    
    print(f"Found {len(voice_files)} voice files and {len(noise_files)} noise files")
    
    # Check if samples already exist
    existing_samples = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
    if existing_samples and not force:
        print(f"Found {len(existing_samples)} existing samples in {output_dir}")
        print(f"Use --force to regenerate them if needed")
        
        if len(existing_samples) >= num_samples:
            print(f"Already have {len(existing_samples)} samples, which is >= requested {num_samples}")
            print(f"Skipping sample generation. Use --force to regenerate.")
            return
        else:
            print(f"Will generate {num_samples - len(existing_samples)} additional samples")
            start_idx = len(existing_samples)
    else:
        if existing_samples and force:
            print(f"Removing {len(existing_samples)} existing samples due to --force flag")
            for sample_file in existing_samples:
                os.remove(os.path.join(output_dir, sample_file))
        start_idx = 0
    
    # Determine number of workers
    if workers is None:
        # Choose based on CPU count and whether GPU is used
        if use_gpu:
            workers = 1  # Use single process with GPU to avoid memory issues
        else:
            workers = min(os.cpu_count(), 8)  # Limit to 8 workers maximum
    
    print(f"Generating {num_samples - start_idx} samples with {workers} workers")
    print(f"Window size: {window_size}")
    print(f"Output directory: {output_dir}")
    print(f"Using GPU: {use_gpu}")
    
    # For GPU processing, use a single worker
    if use_gpu:
        # Use a single process with GPU
        print("Using GPU for processing - running in single process mode")
        success_count = 0
        error_count = 0
        
        # Create preprocessor and move to GPU
        preprocessor = AudioPreprocessor(window_size=window_size, use_gpu=True)
        
        with tqdm(range(start_idx, num_samples), desc="Generating samples") as progress:
            for sample_id in progress:
                try:
                    # Select random files
                    voice_path = random.choice(voice_files)
                    noise_path = random.choice(noise_files)
                    
                    # Random mix ratio
                    mix_ratio = random.uniform(0.3, 0.7)
                    
                    # Create training example
                    mixed_spec, mask = preprocessor.create_training_example(
                        voice_path, noise_path, mix_ratio=mix_ratio
                    )
                    
                    # Move tensors to CPU
                    mixed_spec = mixed_spec.cpu()
                    mask = mask.cpu()
                    
                    # Save sample to disk
                    output_path = os.path.join(output_dir, f"sample_{sample_id:05d}.pt")
                    torch.save(
                        {
                            'mixed': mixed_spec,
                            'mask': mask,
                            'voice_path': voice_path,
                            'noise_path': noise_path,
                            'mix_ratio': mix_ratio
                        },
                        output_path
                    )
                    
                    success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    print(f"\nError creating sample {sample_id}: {e}")
                
                # Update progress
                progress.set_postfix({
                    'success': success_count, 
                    'errors': error_count
                })
                
                # Clean up GPU memory
                if sample_id % 10 == 0:
                    torch.cuda.empty_cache()
    else:
        # Use multiple CPU processes
        print(f"Using {workers} CPU processes for parallel processing")
        
        # Process files with ProcessPoolExecutor
        start_time = time.time()
        success_count = 0
        error_count = 0
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(
                    create_sample, sample_id, voice_files, noise_files, output_dir, window_size, False
                ): sample_id
                for sample_id in range(start_idx, num_samples)
            }
            
            # Process as they complete
            with tqdm(total=len(future_to_id), desc="Generating samples") as progress:
                for future in future_to_id:
                    sample_id = future_to_id[future]
                    try:
                        success, message = future.result()
                        if success:
                            success_count += 1
                        else:
                            error_count += 1
                            print(f"\n{message}")
                    except Exception as e:
                        error_count += 1
                        print(f"\nError processing sample {sample_id}: {e}")
                    
                    # Update progress
                    progress.update(1)
                    progress.set_postfix({
                        'success': success_count, 
                        'errors': error_count
                    })
    
    # Print summary
    duration = time.time() - start_time if 'start_time' in locals() else 0
    print(f"\nSample generation completed in {duration:.2f} seconds")
    print(f"Successfully generated {success_count}/{num_samples - start_idx} samples")
    if error_count > 0:
        print(f"Encountered errors with {error_count} samples")
    
    # Print final stats
    actual_samples = len([f for f in os.listdir(output_dir) if f.endswith('.pt')])
    print(f"\nFinal sample count: {actual_samples}")
    print(f"Samples saved to: {output_dir}")

def generate_samples_from_cache(
    output_dir=None, 
    voice_cache_dir=None, 
    noise_cache_dir=None, 
    num_samples=1000, 
    window_size=DEFAULT_WINDOW_SIZE, 
    force=False
):
    """
    Generate training samples using pre-cached spectrograms for maximum efficiency.
    
    Args:
        output_dir: Directory to save generated samples
        voice_cache_dir: Directory containing cached voice spectrograms
        noise_cache_dir: Directory containing cached noise spectrograms
        num_samples: Number of samples to generate
        window_size: Processing window size (for naming consistency)
        force: Whether to force regeneration of existing samples
    """
    # Set default paths if not provided
    if voice_cache_dir is None:
        voice_cache_dir = os.path.join(CACHE_DIR, 'voice')
    if noise_cache_dir is None:
        noise_cache_dir = os.path.join(CACHE_DIR, 'noise')
    if output_dir is None:
        output_dir = os.path.join(CACHE_DIR, f"samples_{window_size}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if cached spectrograms exist
    voice_specs = [f for f in os.listdir(voice_cache_dir) if f.endswith('_spec.pt')]
    noise_specs = [f for f in os.listdir(noise_cache_dir) if f.endswith('_spec.pt')]
    
    if not voice_specs:
        print(f"No cached voice spectrograms found in {voice_cache_dir}")
        print("Run preprocessing first: python main.py preprocess")
        return False
    
    if not noise_specs:
        print(f"No cached noise spectrograms found in {noise_cache_dir}")
        print("Run preprocessing first: python main.py preprocess")
        return False
    
    print(f"Found {len(voice_specs)} cached voice spectrograms and {len(noise_specs)} cached noise spectrograms")
    
    # Check if samples already exist
    existing_samples = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
    if existing_samples and not force:
        print(f"Found {len(existing_samples)} existing samples in {output_dir}")
        
        if len(existing_samples) >= num_samples:
            print(f"Already have {len(existing_samples)} samples, which is >= requested {num_samples}")
            print(f"Skipping sample generation. Use --force to regenerate.")
            return True
        else:
            print(f"Will generate {num_samples - len(existing_samples)} additional samples")
            start_idx = len(existing_samples)
    else:
        if existing_samples and force:
            print(f"Removing {len(existing_samples)} existing samples due to --force flag")
            for sample_file in existing_samples:
                os.remove(os.path.join(output_dir, sample_file))
        start_idx = 0
    
    print(f"\n=== GENERATING {num_samples - start_idx} TRAINING SAMPLES FROM CACHE ===")
    print(f"Using cached spectrograms for maximum speed")
    
    # Prepare full paths
    voice_spec_paths = [os.path.join(voice_cache_dir, f) for f in voice_specs]
    noise_spec_paths = [os.path.join(noise_cache_dir, f) for f in noise_specs]
    
    # Generate samples
    progress_bar = tqdm(range(start_idx, num_samples), desc="Generating samples")
    for i in progress_bar:
        try:
            # Select random spectrograms
            voice_spec_path = random.choice(voice_spec_paths)
            noise_spec_path = random.choice(noise_spec_paths)
            
            # Load cached spectrograms
            voice_spec = torch.load(voice_spec_path, map_location='cpu')
            noise_spec = torch.load(noise_spec_path, map_location='cpu')
            
            # Ensure matching dimensions
            freq_bins = min(voice_spec.shape[0], noise_spec.shape[0])
            
            # Standardize time dimension for both spectrograms
            voice_spec = AudioPreprocessor().standardize_spectrogram(voice_spec[:freq_bins])
            noise_spec = AudioPreprocessor().standardize_spectrogram(noise_spec[:freq_bins])
            
            # Apply random mixing ratio
            mix_ratio = random.uniform(0.3, 0.7)
            
            # Mix spectrograms directly
            mixed_spec = voice_spec * mix_ratio + noise_spec * (1 - mix_ratio)
            
            # Create mask (voice_spec / mixed_spec with proper bounds)
            epsilon = 1e-10
            mask = torch.clamp(voice_spec / (mixed_spec + epsilon), 0.0, 1.0)
            
            # Save sample
            output_path = os.path.join(output_dir, f"sample_{i:05d}.pt")
            torch.save(
                {
                    'mixed': mixed_spec,
                    'mask': mask,
                    'voice_path': os.path.basename(voice_spec_path),
                    'noise_path': os.path.basename(noise_spec_path),
                    'mix_ratio': mix_ratio
                },
                output_path
            )
            
            # Update progress
            if i % 50 == 0:
                progress_bar.set_postfix({
                    'voice': os.path.basename(voice_spec_path),
                    'noise': os.path.basename(noise_spec_path)
                })
                
        except Exception as e:
            print(f"\nError generating sample {i}: {e}")
    
    total_samples = len([f for f in os.listdir(output_dir) if f.endswith('.pt')])
    print(f"\n=== SAMPLE GENERATION COMPLETE ===")
    print(f"Total samples: {total_samples}")
    print(f"Samples saved to: {output_dir}")
    print(f"\nTo use these samples for training:")
    print(f"python main.py train --preprocessed --window-size {window_size}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate preprocessed training samples')
    parser.add_argument('--samples', type=int, default=1000, 
                        help='Number of samples to generate')
    parser.add_argument('--window-size', choices=list(WINDOW_SIZES.keys()), default=DEFAULT_WINDOW_SIZE,
                        help=f'Window size for processing (default: {DEFAULT_WINDOW_SIZE})')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save samples')
    parser.add_argument('--voice-cache-dir', type=str, default=None,
                        help='Directory containing cached voice spectrograms')
    parser.add_argument('--noise-cache-dir', type=str, default=None,
                        help='Directory containing cached noise spectrograms')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto-detect)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration if available')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration of existing samples')
    parser.add_argument('--use-cache', action='store_true',
                        help='Use cached spectrograms instead of raw audio (faster)')
    
    args = parser.parse_args()
    
    if args.use_cache:
        # Use cached spectrograms for sample generation
        generate_samples_from_cache(
            output_dir=args.output_dir,
            voice_cache_dir=args.voice_cache_dir,
            noise_cache_dir=args.noise_cache_dir,
            num_samples=args.samples,
            window_size=args.window_size,
            force=args.force
        )
    else:
        # Use original audio files for sample generation
        generate_samples(
            num_samples=args.samples,
            window_size=args.window_size,
            output_dir=args.output_dir,
            workers=args.workers,
            use_gpu=args.gpu,
            force=args.force
        )

if __name__ == "__main__":
    main()
