import os
import torch
import numpy as np
import random
import time
import threading
import queue
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset

from .preprocessing import AudioPreprocessor, get_audio_files
from .config import CACHE_DIR, SPEC_TIME_DIM, PRELOAD_SAMPLES, BACKGROUND_WORKERS

class SpectrogramDataset(Dataset):
    """Dataset for generating training examples on-the-fly"""
    
    def __init__(self, voice_dir: str, noise_dir: str,
                 preprocessor: AudioPreprocessor,
                 num_samples: int = 2000,
                 use_gpu: bool = False):
        """
        Initialize dataset.
        
        Args:
            voice_dir: Directory containing voice files
            noise_dir: Directory containing noise files
            preprocessor: AudioPreprocessor for audio processing
            num_samples: Number of training samples to generate
            use_gpu: Whether to use GPU acceleration
        """
        self.voice_dir = voice_dir
        self.noise_dir = noise_dir
        self.preprocessor = preprocessor
        self.num_samples = num_samples
        self.use_gpu = use_gpu
        
        # Get list of files
        self.voice_files = get_audio_files(voice_dir)
        self.noise_files = get_audio_files(noise_dir)
        
        if not self.voice_files:
            raise ValueError(f"No voice files found in {voice_dir}")
        if not self.noise_files:
            raise ValueError(f"No noise files found in {noise_dir}")
            
        print(f"Found {len(self.voice_files)} voice files and {len(self.noise_files)} noise files")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a random mix of voice and noise"""
        # Select random files
        voice_path = random.choice(self.voice_files)
        noise_path = random.choice(self.noise_files)
        
        # Create training example
        try:
            # Random mix ratio between 0.3 and 0.7
            mix_ratio = random.uniform(0.3, 0.7)
            mixed_spec, mask = self.preprocessor.create_training_example(
                voice_path, noise_path, mix_ratio=mix_ratio
            )
            
            # Add channel dimension if needed
            if mixed_spec.dim() == 2:
                mixed_spec = mixed_spec.unsqueeze(0)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
                
            # Move to CPU if preprocessing was done on GPU
            if self.use_gpu and mixed_spec.device.type == 'cuda':
                mixed_spec = mixed_spec.cpu()
                mask = mask.cpu()
                
            return {
                'mixed': mixed_spec,
                'mask': mask
            }
        except Exception as e:
            print(f"Error creating example from {voice_path} and {noise_path}: {e}")
            # Try again with different files on error
            return self.__getitem__((idx + 1) % len(self))

class CachedSpectrogramDataset(Dataset):
    """Dataset that uses background threads to preload and cache data"""
    
    def __init__(self, voice_dir: str, noise_dir: str,
                 num_samples: int = 2000,
                 cache_size: int = PRELOAD_SAMPLES,
                 use_gpu: bool = False):
        """
        Initialize dataset with background caching.
        
        Args:
            voice_dir: Directory containing voice files
            noise_dir: Directory containing noise files
            num_samples: Number of training samples to generate
            cache_size: Number of samples to keep in memory
            use_gpu: Whether to use GPU acceleration
        """
        self.voice_dir = voice_dir
        self.noise_dir = noise_dir
        self.num_samples = num_samples
        self.cache_size = min(cache_size, num_samples)
        self.use_gpu = use_gpu
        
        # Get list of files
        self.voice_files = get_audio_files(voice_dir)
        self.noise_files = get_audio_files(noise_dir)
        
        if not self.voice_files:
            raise ValueError(f"No voice files found in {voice_dir}")
        if not self.noise_files:
            raise ValueError(f"No noise files found in {noise_dir}")
            
        print(f"Found {len(self.voice_files)} voice files and {len(self.noise_files)} noise files")
        
        # Initialize cache
        self.cache = {}
        self.cache_queue = queue.Queue(maxsize=self.cache_size)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(use_gpu=use_gpu)
        
        # Start background threads for filling cache
        self.workers = []
        num_workers = min(BACKGROUND_WORKERS, 4)  # Limit to 4 workers max
        print(f"Starting {num_workers} background workers for cache filling")
        
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._cache_filler_thread,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        # Wait for initial cache fill (at least 10% of requested size)
        min_cache = max(1, self.cache_size // 10)
        print(f"Waiting for initial cache fill ({min_cache} samples)...")
        
        while self.cache_queue.qsize() < min_cache:
            time.sleep(0.1)
            
        print(f"Initial cache filled with {self.cache_queue.qsize()} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __del__(self):
        """Clean up resources when object is deleted"""
        self.stop_event.set()
        for worker in self.workers:
            worker.join(timeout=1.0)
    
    def _cache_filler_thread(self, worker_id):
        """Background thread to fill the cache"""
        print(f"Cache filler thread {worker_id} started")
        
        while not self.stop_event.is_set():
            if self.cache_queue.full():
                # Cache is full, wait a bit
                time.sleep(0.1)
                continue
                
            try:
                # Create a new sample
                voice_path = random.choice(self.voice_files)
                noise_path = random.choice(self.noise_files)
                
                # Random mix ratio between 0.3 and 0.7
                mix_ratio = random.uniform(0.3, 0.7)
                
                mixed_spec, mask = self.preprocessor.create_training_example(
                    voice_path, noise_path, mix_ratio=mix_ratio
                )
                
                # Add channel dimension if needed
                if mixed_spec.dim() == 2:
                    mixed_spec = mixed_spec.unsqueeze(0)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                    
                # Move to CPU if preprocessing was done on GPU
                if self.use_gpu and mixed_spec.device.type == 'cuda':
                    mixed_spec = mixed_spec.cpu()
                    mask = mask.cpu()
                
                # Add to cache queue
                sample = {
                    'mixed': mixed_spec,
                    'mask': mask
                }
                
                self.cache_queue.put(sample, block=True, timeout=1.0)
            except queue.Full:
                # Queue is full, that's fine
                pass
            except Exception as e:
                print(f"Error in cache filler thread {worker_id}: {e}")
                time.sleep(0.5)  # Sleep on error to prevent spinning
    
    def __getitem__(self, idx):
        """Get a sample from the cache, or wait for one to be available"""
        try:
            # Get from cache if available
            sample = self.cache_queue.get(block=True, timeout=2.0)
            return sample
        except queue.Empty:
            # If cache is empty, generate on the fly as fallback
            print("Warning: Cache empty, generating sample on the fly")
            voice_path = random.choice(self.voice_files)
            noise_path = random.choice(self.noise_files)
            
            mixed_spec, mask = self.preprocessor.create_training_example(
                voice_path, noise_path, mix_ratio=random.uniform(0.3, 0.7)
            )
            
            # Add channel dimension if needed
            if mixed_spec.dim() == 2:
                mixed_spec = mixed_spec.unsqueeze(0)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
                
            return {
                'mixed': mixed_spec,
                'mask': mask
            }
