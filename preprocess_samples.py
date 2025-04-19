"""
Preprocess and cache training samples to speed up training
"""
import os
import torch
import torch.multiprocessing as mp
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import random
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.preprocessing import AudioPreprocessor, get_audio_files
from src.config import (
    VOICE_DIR, NOISE_DIR, CACHE_DIR, SAMPLE_RATE,
    USE_GPU, BATCH_SIZE, N_FFT
)

def process_batch(args):
    """
    Process a batch of samples (designed for multiprocessing)
    
    Args:
        args: Tuple of (batch_idx, num_samples, voice_files, noise_files, 
                       output_dir, window_size, use_gpu)
                       
    Returns:
        Number of successfully processed samples
    """
    batch_idx, batch_size, voice_files, noise_files, output_dir, window_size, use_gpu = args
    
    # Create local preprocessor for this process
    preprocessor = AudioPreprocessor(window_size=window_size, use_gpu=use_gpu)
    
    # Determine sample range for this batch
    start_idx = batch_idx * batch_size
    
    # Pre-load voice and noise files for this batch to avoid repeated loading
    preloaded_voice = {}
    preloaded_noise = {}
    
    # Track which files we need for this batch
    batch_voice_files = random.sample(voice_files, min(5, len(voice_files)))
    batch_noise_files = random.sample(noise_files, min(10, len(noise_files)))
    
    # Preload audio
    for voice_path in batch_voice_files:
        try:
            preloaded_voice[voice_path] = preprocessor.load_audio(voice_path)
        except Exception as e:
            print(f"Error loading {voice_path}: {e}")
    
    for noise_path in batch_noise_files:
        try:
            preloaded_noise[noise_path] = preprocessor.load_audio(noise_path)
        except Exception as e:
            print(f"Error loading {noise_path}: {e}")
    
    # Create GPU tensor batch containers if using GPU
    if use_gpu and torch.cuda.is_available():
        # Use a fixed batch size of 4-8 for GPU processing to avoid OOM
        gpu_batch_size = 4
        mixed_batch = []
        mask_batch = []
    
    # Process samples
    samples_created = 0
    for i in range(batch_size):
        try:
            # 80% mix with noise, 20% just voice
            use_noise = random.random() < 0.8
            mix_ratio = random.uniform(0.3, 0.7) if use_noise else 1.0
            
            # Select random files
            voice_path = random.choice(batch_voice_files)
            noise_path = random.choice(batch_noise_files) if use_noise else None
            
            # Use preloaded audio when available
            voice = preloaded_voice.get(voice_path)
            if voice is None:
                continue
                
            noise = preloaded_noise.get(noise_path) if noise_path else None
            if use_noise and noise is None:
                continue
            
            # Process on GPU in batches when possible
            if use_gpu and torch.cuda.is_available():
                # Create and add to batch
                if noise is not None:
                    # Adjust lengths to match
                    min_length = min(voice.shape[1], noise.shape[1])
                    voice_part = voice[:, :min_length]
                    noise_part = noise[:, :min_length]
                    
                    mixed = voice_part * mix_ratio + noise_part * (1 - mix_ratio)
                else:
                    mixed = voice
                
                # Compute spectrograms
                voice_spec = preprocessor.compute_spectrogram(voice if noise is None else voice_part)
                mixed_spec = preprocessor.compute_spectrogram(mixed)
                
                # Create mask
                epsilon = 1e-10
                mask = (voice_spec / (mixed_spec + epsilon)) > 0.5
                mask = mask.float()
                
                # Standardize
                mixed_spec = preprocessor.standardize_spectrogram(mixed_spec)
                mask = preprocessor.standardize_spectrogram(mask)
                
                # Move to CPU for saving
                mixed_spec = mixed_spec.cpu()
                mask = mask.cpu()
                
                # Save sample
                sample_idx = start_idx + samples_created
                sample_path = os.path.join(output_dir, f"sample_{sample_idx:06d}.pt")
                torch.save({
                    'mixed': mixed_spec,
                    'mask': mask
                }, sample_path)
                
                samples_created += 1
            else:
                # CPU processing
                mixed_spec, mask = preprocessor.create_training_example(
                    voice_path, noise_path, mix_ratio
                )
                
                # Save to disk with unique filename
                sample_idx = start_idx + samples_created
                sample_path = os.path.join(output_dir, f"sample_{sample_idx:06d}.pt")
                
                # Move tensors to CPU before saving if needed
                if mixed_spec.device.type != 'cpu':
                    mixed_spec = mixed_spec.cpu()
                if mask.device.type != 'cpu':
                    mask = mask.cpu()
                
                # Save as tensor dictionary
                torch.save({
                    'mixed': mixed_spec,
                    'mask': mask
                }, sample_path)
                
                samples_created += 1
                
        except Exception as e:
            print(f"Error creating sample in batch {batch_idx}: {e}")
    
    # Clean up explicitly to free memory
    del preprocessor
    del preloaded_voice
    del preloaded_noise
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return samples_created

def generate_training_samples(
    num_samples: int,
    window_size: str = 'medium',
    use_gpu: bool = USE_GPU,
    output_dir: str = None,
    force_regenerate: bool = False,
    num_workers: int = None
):
    """
    Generate and cache training samples for faster model training.
    
    Args:
        num_samples: Number of samples to generate
        window_size: Processing window size
        use_gpu: Whether to use GPU for preprocessing
        output_dir: Directory to save samples (defaults to CACHE_DIR/samples)
        force_regenerate: Whether to force regeneration of existing samples
        num_workers: Number of parallel workers (default: CPU count minus 1)
    """
    # Setup directories
    if output_dir is None:
        output_dir = os.path.join(CACHE_DIR, f'samples_{window_size}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if samples already exist
    existing_samples = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
    if len(existing_samples) >= num_samples and not force_regenerate:
        print(f"Found {len(existing_samples)} existing samples in {output_dir}")
        print("Use --force to regenerate these samples")
        return output_dir
    
    # Get voice and noise files
    voice_files = get_audio_files(VOICE_DIR)
    noise_files = get_audio_files(NOISE_DIR)
    
    if not voice_files:
        raise ValueError(f"No voice files found in {VOICE_DIR}")
    if not noise_files:
        raise ValueError(f"No noise files found in {NOISE_DIR}")
    
    print(f"Found {len(voice_files)} voice files and {len(noise_files)} noise files")
    
    # Determine optimal number of workers
    if num_workers is None:
        # Use all CPUs except one
        num_workers = max(1, mp.cpu_count() - 1)
    
    # Limit number of workers based on files
    num_workers = min(num_workers, len(voice_files), len(noise_files))
    
    print(f"Initializing preprocessor (window_size={window_size}, use_gpu={use_gpu})")
    print(f"Using {num_workers} worker processes")
    
    # Clear any existing samples if forcing regeneration
    if force_regenerate and existing_samples:
        print(f"Removing {len(existing_samples)} existing samples...")
        for sample_file in existing_samples:
            os.remove(os.path.join(output_dir, sample_file))
    
    # Prepare for multiprocessing
    batch_size = min(50, max(10, num_samples // (num_workers * 2)))
    total_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Generating {num_samples} training samples in {output_dir}...")
    print(f"Processing in {total_batches} batches with {batch_size} samples per batch")
    
    # Start timing
    start_time = time.time()
    
    # Set multiprocessing start method - 'spawn' is more compatible across platforms
    mp.set_start_method('spawn', force=True)
    
    # Prepare batch arguments
    batch_args = []
    for batch_idx in range(total_batches):
        # Adjust final batch size if needed
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        if current_batch_size <= 0:
            break
            
        batch_args.append(
            (batch_idx, current_batch_size, voice_files, noise_files, 
             output_dir, window_size, use_gpu)
        )
    
    # Process batches in parallel
    samples_created = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_batch, args) for args in batch_args]
        
        # Monitor progress with tqdm
        with tqdm(total=num_samples, desc="Generating samples") as pbar:
            for future in as_completed(futures):
                batch_samples = future.result()
                samples_created += batch_samples
                pbar.update(batch_samples)
                
                # Update progress display
                pbar.set_postfix({
                    'completed': f"{samples_created}/{num_samples}",
                    'workers': num_workers
                })
    
    # Report stats
    elapsed_time = time.time() - start_time
    print(f"\nSample generation complete!")
    print(f"Created {samples_created} samples in {elapsed_time:.1f}s")
    print(f"Average time per sample: {elapsed_time/samples_created:.2f}s")
    print(f"Samples per second: {samples_created/elapsed_time:.1f}")
    print(f"Samples saved to: {output_dir}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Preprocess and cache training samples')
    parser.add_argument('--samples', type=int, default=1000, 
                        help='Number of samples to generate')
    parser.add_argument('--window-size', choices=['small', 'medium', 'large'], 
                        default='medium', help='Window size for processing')
    parser.add_argument('--gpu', action='store_true', default=USE_GPU,
                        help='Use GPU acceleration if available')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration of existing samples')
    parser.add_argument('--output-dir', default=None,
                        help='Directory to save preprocessed samples')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count-1)')
    
    args = parser.parse_args()
    
    # Generate samples
    generate_training_samples(
        num_samples=args.samples,
        window_size=args.window_size,
        use_gpu=args.gpu,
        output_dir=args.output_dir,
        force_regenerate=args.force,
        num_workers=args.workers
    )

if __name__ == "__main__":
    main()
