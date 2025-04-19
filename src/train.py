import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import random
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import gc
import threading
import queue

from .config import (
    VOICE_DIR, NOISE_DIR, OUTPUT_DIR, CACHE_DIR, BATCH_SIZE, EPOCHS, 
    LEARNING_RATE, N_FFT, VALIDATION_SPLIT, RANDOM_SEED,
    USE_GPU, GPU_DEVICE, MIXED_PRECISION, CUDNN_BENCHMARK, GPU_MEMORY_FRACTION,
    DATALOADER_WORKERS, DATALOADER_PREFETCH, DATALOADER_PIN_MEMORY, AUTO_DETECT_SAMPLES,
    USE_CACHED_DATA, PRELOAD_SAMPLES, BACKGROUND_WORKERS
)
from .preprocessing import AudioPreprocessor, get_audio_files
from .model import VoiceIsolationModel, MaskedLoss, VoiceIsolationModelDeep

# Import GPU utilities if available
try:
    from .gpu_utils import GPUMonitor, optimize_for_gpu, optimize_gpu_utilization, estimate_optimal_samples
except ImportError:
    GPUMonitor = None
    optimize_for_gpu = lambda: None
    optimize_gpu_utilization = lambda: None
    def estimate_optimal_samples(*args, **kwargs):
        return 2000

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set CUDA-specific configurations
if USE_GPU:
    # Optimize CUDA settings for maximum performance
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    torch.backends.cudnn.enabled = True
    
    # Enable TensorFloat32 precision if available (faster on newer GPUs)
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        torch.set_float32_matmul_precision('high')  # Use TF32 precision
    
    # Limit GPU memory usage if needed
    if GPU_MEMORY_FRACTION < 1.0:
        try:
            # Check if device exists before setting memory fraction
            if GPU_DEVICE < torch.cuda.device_count():
                torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, GPU_DEVICE)
            else:
                print(f"⚠️ GPU device {GPU_DEVICE} not found. Using default device 0.")
                torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
        except RuntimeError as e:
            print(f"⚠️ Could not set GPU memory fraction: {e}")
            print("Continuing without memory limits.")
    
    # Optimize GPU settings
    optimize_for_gpu()

class CachedSpectrogramDataset(Dataset):
    """Dataset that uses cached spectrograms for faster loading"""
    
    def __init__(self, voice_dir: str, noise_dir: str, 
                 num_samples: int = 1000, transform=None, use_gpu: bool = USE_GPU,
                 voice_cache_dir=None, noise_cache_dir=None):
        """
        Initialize dataset with cache support
        
        Args:
            voice_dir: Directory with voice audio files
            noise_dir: Directory with noise audio files
            num_samples: Number of samples to generate
            transform: Optional transforms to apply
            use_gpu: Whether to use GPU acceleration
            voice_cache_dir: Directory with cached voice spectrograms
            noise_cache_dir: Directory with cached noise spectrograms
        """
        # Initialize thread control attributes first to avoid errors in __del__
        self.preload_lock = threading.Lock()
        self.stop_preloading = threading.Event()
        self.preload_thread = None
        self.preload_queue = None
        
        self.transform = transform
        self.num_samples = num_samples
        self.use_gpu = use_gpu and USE_GPU
        self.device = torch.device(f"cuda:{GPU_DEVICE}" if self.use_gpu else "cpu")
        
        # Setup cache paths
        self.voice_cache_dir = voice_cache_dir or os.path.join(CACHE_DIR, 'voice')
        self.noise_cache_dir = noise_cache_dir or os.path.join(CACHE_DIR, 'noise')
        
        # Make sure directories exist
        os.makedirs(self.voice_cache_dir, exist_ok=True)
        os.makedirs(self.noise_cache_dir, exist_ok=True)
        os.makedirs(voice_dir, exist_ok=True)
        os.makedirs(noise_dir, exist_ok=True)
        
        # Check if cache exists
        self.voice_cache_available = os.path.exists(self.voice_cache_dir) and len(os.listdir(self.voice_cache_dir)) > 0
        self.noise_cache_available = os.path.exists(self.noise_cache_dir) and len(os.listdir(self.noise_cache_dir)) > 0
        
        # Get lists of files
        if self.voice_cache_available:
            self.voice_files = [os.path.join(self.voice_cache_dir, f) for f in os.listdir(self.voice_cache_dir) if f.endswith('.pt')]
            print(f"Using {len(self.voice_files)} cached voice spectrograms")
        else:
            self.voice_files = get_audio_files(voice_dir)
            print(f"Using {len(self.voice_files)} original voice audio files")
            
        if self.noise_cache_available:
            self.noise_files = [os.path.join(self.noise_cache_dir, f) for f in os.listdir(self.noise_cache_dir) if f.endswith('.pt')]
            print(f"Using {len(self.noise_files)} cached noise spectrograms")
        else:
            self.noise_files = get_audio_files(noise_dir)
            print(f"Using {len(self.noise_files)} original noise audio files")
        
        # Verify files exist and provide helpful error messages
        if not self.voice_files:
            print("=" * 80)
            print(f"ERROR: No voice files found in {voice_dir} or cache")
            print("Please add some voice audio files (.mp3, .wav, or .flac) to the VOICE directory.")
            print("For example:")
            print("  - Add .mp3, .wav, or .flac files of the target person speaking alone")
            print("  - Make sure the files are readable and not corrupted")
            print("=" * 80)
            raise ValueError(f"No voice files found in {voice_dir} or cache")
            
        if not self.noise_files:
            print("=" * 80)
            print(f"ERROR: No noise files found in {noise_dir} or cache")
            print("Please add some noise/background audio files (.mp3, .wav, or .flac) to the NOISE directory.")
            print("For example:")
            print("  - Add recordings of other people speaking (not the target person)")
            print("  - Add background noise recordings")
            print("  - Add any sounds that typically interfere with the target voice")
            print("=" * 80)
            raise ValueError(f"No noise files found in {noise_dir} or cache")
        
        # Initialize preprocessor for non-cached data
        if not (self.voice_cache_available and self.noise_cache_available):
            self.preprocessor = AudioPreprocessor(use_gpu=use_gpu)
        
        # Create sample indices
        self.indices = list(range(num_samples))
        
        # Preloaded sample cache
        self.preloaded_samples = {}
        
        # Setup background loading thread - increase preload size for better batch preparation
        self.preload_queue = queue.Queue(maxsize=max(PRELOAD_SAMPLES, 3 * BATCH_SIZE))
        self.preload_thread = threading.Thread(target=self._preload_samples_worker)
        self.preload_thread.daemon = True
        self.preload_thread.start()
        
        # Create multiple background threads for better CPU core utilization
        self.worker_threads = []
        num_workers = min(4, os.cpu_count() - 1)  # Use more worker threads but leave one CPU core free
        for i in range(num_workers):
            worker = threading.Thread(target=self._preload_samples_worker)
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)
        
        # Estimate time for sample creation
        self._estimate_creation_time()
    
    def _estimate_creation_time(self):
        print("\n=== PREPARING SAMPLE DATASET ===")
        sample_size = min(5, self.num_samples)
        
        progress_bar = tqdm(range(sample_size), desc="Creating test samples", unit="sample")
        start_time = time.time()
        for i in progress_bar:
            self._create_sample()
        
        avg_time = (time.time() - start_time) / sample_size
        print(f"Sample creation speed: {avg_time:.2f}s per sample")
        total_est_time = avg_time * self.num_samples / (len(self.worker_threads) + 1)
        print(f"Estimated total preparation time: {total_est_time:.1f}s with {len(self.worker_threads) + 1} workers")
        print(f"Using {'cached' if self.voice_cache_available else 'raw'} data")
        
        if self.use_gpu:
            print(f"GPU acceleration: ENABLED ✓")
        else:
            print(f"GPU acceleration: DISABLED ✗")
    
    def _load_cached_spectrogram(self, file_path):
        """Load a preprocessed spectrogram from cache"""
        return torch.load(file_path, map_location='cpu')
    
    def _create_sample(self):
        """Create a single training sample from cached or raw data"""
        # 80% mix with noise, 20% just voice
        use_noise = random.random() < 0.8
        mix_ratio = random.uniform(0.3, 0.7) if use_noise else 1.0
        
        # Select random files
        voice_path = random.choice(self.voice_files)
        noise_path = random.choice(self.noise_files) if use_noise else None
        
        try:
            # Check if using cached spectrograms
            if self.voice_cache_available and (not use_noise or self.noise_cache_available):
                # Load voice spectrogram
                voice_spec = self._load_cached_spectrogram(voice_path)
                
                if use_noise:
                    # Load noise spectrogram
                    noise_spec = self._load_cached_spectrogram(noise_path)
                    
                    # Adjust lengths to match
                    min_time_frames = min(voice_spec.shape[1], noise_spec.shape[1])
                    voice_spec = voice_spec[:, :min_time_frames]
                    noise_spec = noise_spec[:, :min_time_frames]
                    
                    # Mix spectrograms directly (approx - not as accurate as time domain mixing)
                    mixed_spec = voice_spec * mix_ratio + noise_spec * (1 - mix_ratio)
                else:
                    mixed_spec = voice_spec
                
                # Create mask
                epsilon = 1e-10
                mask = (voice_spec / (mixed_spec + epsilon)) > 0.5
                mask = mask.float()
                
            else:
                # Fallback to original method
                return self.preprocessor.create_training_example(
                    voice_path, noise_path, mix_ratio
                )
            
            # Standardize
            if hasattr(self, 'preprocessor'):
                mixed_spec = self.preprocessor.standardize_spectrogram(mixed_spec)
                mask = self.preprocessor.standardize_spectrogram(mask)
                
            return mixed_spec, mask
            
        except Exception as e:
            print(f"Error creating sample: {e}")
            # Try another sample
            return self._create_sample()
    
    def _preload_samples_worker(self):
        """Background worker to preload samples"""
        while not self.stop_preloading.is_set():
            try:
                # If we don't have enough preloaded samples, create more
                if self.preload_queue.qsize() < PRELOAD_SAMPLES:
                    sample = self._create_sample()
                    self.preload_queue.put(sample, block=False)
                else:
                    time.sleep(0.1)  # Don't spin if queue is full
            except queue.Full:
                time.sleep(0.1)  # Queue is full, wait a bit
            except Exception as e:
                print(f"Error in preload worker: {e}")
                time.sleep(0.5)  # Wait a bit after errors
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a sample from the preloaded queue if available, otherwise create one"""
        try:
            # First try to get a preloaded sample
            try:
                mixed_spec, mask = self.preload_queue.get(block=False)
            except queue.Empty:
                # If none are ready, create one on demand
                mixed_spec, mask = self._create_sample()
            
            # Apply transformations if any
            if self.transform:
                mixed_spec = self.transform(mixed_spec)
                mask = self.transform(mask)
            
            # Add channel dimension
            mixed_spec = mixed_spec.unsqueeze(0)
            mask = mask.unsqueeze(0)
            
            # IMPORTANT: Move tensors to CPU before returning
            if mixed_spec.device.type != 'cpu':
                mixed_spec = mixed_spec.cpu()
            if mask.device.type != 'cpu':
                mask = mask.cpu()
                
            return {
                'mixed': mixed_spec,
                'mask': mask
            }
        except Exception as e:
            print(f"Error in __getitem__: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'stop_preloading') and self.stop_preloading is not None:
            self.stop_preloading.set()
        if hasattr(self, 'preload_thread') and self.preload_thread is not None and self.preload_thread.is_alive():
            self.preload_thread.join(timeout=1.0)
        # Clean up worker threads
        if hasattr(self, 'worker_threads'):
            for thread in self.worker_threads:
                if thread.is_alive():
                    thread.join(timeout=0.5)

def train_model(
    window_size: str = 'medium',
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    num_samples: int = 2000,
    model_save_path: Optional[str] = None,
    use_gpu: bool = USE_GPU,
    use_mixed_precision: bool = MIXED_PRECISION,
    auto_detect_samples: bool = AUTO_DETECT_SAMPLES,
    use_deep_model: bool = False,
    use_cache: bool = USE_CACHED_DATA
) -> Tuple[nn.Module, Dict]:
    """
    Train the voice isolation model.
    
    Args:
        window_size: Size of the processing window
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        num_samples: Number of training samples to generate
        model_save_path: Path to save the model (optional)
        use_gpu: Whether to use GPU acceleration
        use_mixed_precision: Whether to use mixed precision training
        auto_detect_samples: Auto-detect optimal sample count
        use_deep_model: Use deeper model for higher GPU utilization
        use_cache: Use cached preprocessed data
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Better GPU detection and messaging
    if torch.cuda.is_available() and use_gpu:
        device = torch.device(f"cuda:{GPU_DEVICE}")
        
        # Force CUDA initialization to check if it's really working
        torch.cuda.synchronize()
        
        # Clear GPU cache before starting
        torch.cuda.empty_cache()
        gc.collect()  # Also collect Python garbage
        
        # Additional GPU optimization for maximum utilization
        optimize_gpu_utilization()
        
        # Test tensor creation
        test = torch.ones(1, device=device)
        if test.device.type == 'cuda':
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(GPU_DEVICE)
            gpu_mem = torch.cuda.get_device_properties(GPU_DEVICE).total_memory / 1024**3
            print(f"\n=== GPU ACCELERATION ENABLED ===")
            print(f"Device: {gpu_name} ({gpu_mem:.1f} GB)")
            
            # Measure GPU memory usage before training
            allocated_before = torch.cuda.memory_allocated(GPU_DEVICE) / 1024**2
            reserved_before = torch.cuda.memory_reserved(GPU_DEVICE) / 1024**2
            print(f"Initial GPU Memory: {allocated_before:.1f} MB allocated, {reserved_before:.1f} MB reserved")
            
            # Start GPU monitoring if available
            gpu_monitor = None
            if GPUMonitor is not None:
                gpu_monitor = GPUMonitor(device_id=GPU_DEVICE)
                gpu_monitor.start()
                print("Started GPU utilization monitoring")
        else:
            print("⚠️ Failed to create tensor on GPU despite CUDA being available!")
            print("⚠️ Falling back to CPU...")
            device = torch.device("cpu")
            use_gpu = False
            auto_detect_samples = False
    else:
        if not torch.cuda.is_available() and use_gpu:
            print("❌ GPU requested but no CUDA-compatible GPU detected!")
        device = torch.device("cpu")
        use_gpu = False
        auto_detect_samples = False
        print("\n=== RUNNING ON CPU ===")
    
    # Initialize preprocessor with GPU support
    preprocessor = AudioPreprocessor(window_size=window_size, use_gpu=use_gpu)
    
    # Get voice and noise files
    voice_files = get_audio_files(VOICE_DIR)
    noise_files = get_audio_files(NOISE_DIR)
    
    # Auto-detect optimal sample count if requested and possible
    if auto_detect_samples and use_gpu:
        print("\n=== AUTO-DETECTING OPTIMAL SAMPLE COUNT ===")
        estimated_samples = estimate_optimal_samples(
            preprocessor=preprocessor,
            voice_files=voice_files, 
            noise_files=noise_files,
            batch_size=batch_size
        )
        
        # Update sample count if auto-detection is successful
        if estimated_samples > 0:
            print(f"Auto-detected optimal sample count: {estimated_samples}")
            print(f"Original requested samples: {num_samples}")
            
            # Use the smaller of the two values to be safe
            num_samples = min(num_samples, estimated_samples) if num_samples > 0 else estimated_samples
            print(f"Using sample count: {num_samples}")
    
    print(f"\n=== CREATING DATASET ({num_samples} SAMPLES) ===")
    
    # Create the appropriate dataset (cached or on-the-fly)
    if use_cache:
        print("Using cached dataset with background workers for faster loading")
        dataset = CachedSpectrogramDataset(
            voice_dir=VOICE_DIR,
            noise_dir=NOISE_DIR,
            num_samples=num_samples,
            use_gpu=use_gpu
        )
        # When using CachedSpectrogramDataset, we must use num_workers=0
        # because thread locks and queues can't be pickled for multiprocessing
        actual_workers = 0
        print("Note: Using 0 dataloader workers with cached dataset (required for thread safety)")
    else:
        dataset = SpectrogramDataset(
            voice_dir=VOICE_DIR,
            noise_dir=NOISE_DIR,
            preprocessor=preprocessor,
            num_samples=num_samples,
            use_gpu=use_gpu
        )
        # Use configured number of workers for non-cached dataset
        actual_workers = DATALOADER_WORKERS
    
    # Split into train and validation sets
    val_size = int(VALIDATION_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n=== PREPARING DATA LOADERS ===")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {actual_workers}")
    print(f"Prefetch factor: {DATALOADER_PREFETCH}")
    
    # Important: For CachedSpectrogramDataset, optimize single-process loading
    if actual_workers == 0:
        # Safe configuration for datasets with unpicklable objects
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Force no workers
            pin_memory=True,  # Still use pin_memory for faster GPU transfers
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Force no workers
            pin_memory=True,  # Still use pin_memory for faster GPU transfers
            drop_last=True
        )
    else:
        # Standard configuration for regular datasets
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=actual_workers,
            pin_memory=DATALOADER_PIN_MEMORY and device.type == 'cuda',
            prefetch_factor=DATALOADER_PREFETCH,
            persistent_workers=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=actual_workers,
            pin_memory=DATALOADER_PIN_MEMORY and device.type == 'cuda',
            prefetch_factor=DATALOADER_PREFETCH,
            persistent_workers=True,
            drop_last=True
        )
    
    print(f"\n=== INITIALIZING MODEL ===")
    # Initialize model, loss, and optimizer
    if use_deep_model:
        print("Using deeper model for higher GPU utilization")
        model = VoiceIsolationModelDeep(n_fft=N_FFT)
    else:
        model = VoiceIsolationModel(n_fft=N_FFT)
    
    loss_fn = MaskedLoss()
    
    # Use AdamW optimizer for better performance
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Move model to GPU
    model.to(device)
    
    # Verify model is on correct device
    if use_gpu:
        model_device = next(model.parameters()).device
        print(f"Model is on: {model_device}")
        if model_device.type != 'cuda':
            print("⚠️ Warning: Model not on GPU despite GPU being enabled!")
    
    # Initialize scaler for mixed precision training
    scaler = torch.amp.GradScaler(enabled=use_mixed_precision and device.type == 'cuda')
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    # Training loop
    print(f"\n=== STARTING TRAINING ({epochs} EPOCHS) ===")
    if use_mixed_precision and device.type == 'cuda':
        print("Mixed precision (FP16): ENABLED ✓")
    
    # Calculate total steps for progress tracking
    total_steps = epochs * len(train_loader)
    start_training_time = time.time()
    
    # Create custom progress bar
    progress_bar = tqdm(
        range(epochs), 
        desc="Training Progress", 
        unit="epoch",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    # Store batch timing information
    batch_times = []
    
    # Create CUDA streams for overlapping operations
    if device.type == 'cuda':
        # Main computing stream
        compute_stream = torch.cuda.Stream()
        # Data transfer stream
        transfer_stream = torch.cuda.Stream()
        # Prefetch the first batch to warm up the pipeline
        next_batch = None
        try:
            # Get first batch in advance
            for prefetch_batch in train_loader:
                next_batch = prefetch_batch
                break
        except:
            pass
    
    for epoch in progress_bar:
        # Training phase
        model.train()
        train_loss = 0.0
        epoch_start_time = time.time()
        
        # Progress tracking
        batch_count = len(train_loader)
        batch_idx = 0
        
        # Create iterator once and reuse
        train_iter = iter(train_loader)
        
        while batch_idx < batch_count:
            batch_start = time.time()
            # Use prefetched batch if available, otherwise get next batch
            if device.type == 'cuda' and next_batch is not None:
                batch = next_batch
                # Prefetch next batch in background
                try:
                    next_batch = next(train_iter)
                except StopIteration:
                    next_batch = None
            else:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    break
            
            batch_idx += 1
            
            # Process batch (keep existing CUDA stream logic)
            if device.type == 'cuda':
                # Use CUDA streams for parallel operations
                with torch.cuda.stream(transfer_stream):
                    # Move data to GPU in the transfer stream
                    mixed_spec = batch['mixed'].to(device, non_blocking=True)
                    target_mask = batch['mask'].to(device, non_blocking=True)
                
                # Synchronize streams to ensure data is ready
                torch.cuda.current_stream().wait_stream(transfer_stream)
                
                # Process in compute stream
                with torch.cuda.stream(compute_stream):
                    # Forward pass with mixed precision
                    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                    
                    if use_mixed_precision:
                        with torch.autocast(device_type=device.type, dtype=torch.float16):
                            pred_mask = model(mixed_spec)
                            loss = loss_fn(pred_mask, target_mask)
                        
                        # Backward pass with gradient scaling for mixed precision
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard training path
                        pred_mask = model(mixed_spec)
                        loss = loss_fn(pred_mask, target_mask)
                        loss.backward()
                        optimizer.step()
            else:
                # CPU path - simpler without streams
                mixed_spec = batch['mixed'].to(device)
                target_mask = batch['mask'].to(device)
                
                optimizer.zero_grad(set_to_none=True)
                pred_mask = model(mixed_spec)
                loss = loss_fn(pred_mask, target_mask)
                loss.backward()
                optimizer.step()
            
            # Wait for compute stream to finish
            if device.type == 'cuda':
                torch.cuda.current_stream().wait_stream(compute_stream)
            
            # Update progress
            train_loss += loss.item()
            batch_times.append(time.time() - batch_start)
            
            # Calculate average stats for last 50 batches
            recent_batch_times = batch_times[-50:]
            avg_batch_time = sum(recent_batch_times) / len(recent_batch_times)
            global_step = epoch * batch_count + batch_idx
            progress_percent = 100.0 * global_step / total_steps
            
            # Calculate ETA
            elapsed = time.time() - start_training_time
            steps_done = global_step
            steps_remaining = total_steps - steps_done
            eta_seconds = avg_batch_time * steps_remaining if steps_done > 0 else 0
            
            # Calculate memory stats
            if device.type == 'cuda':
                mem_allocated = torch.cuda.memory_allocated(GPU_DEVICE) / 1024**3  # GB
                mem_reserved = torch.cuda.memory_reserved(GPU_DEVICE) / 1024**3  # GB
                mem_str = f", Mem: {mem_allocated:.1f}/{mem_reserved:.1f} GB"
            else:
                mem_str = ""
            
            # Create a custom progress indicator to avoid tqdm overhead
            if batch_idx % 5 == 0 or batch_idx == batch_count:
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                print(f"\rTraining: {progress_percent:.1f}% [{batch_idx}/{batch_count}] "
                      f"Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}, "
                      f"Batch: {avg_batch_time*1000:.1f}ms, ETA: {eta_str}{mem_str}", end="")
        
        print()  # New line after progress tracking
        
        # Validation phase with mixed precision
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_progress = tqdm(
                val_loader, 
                desc=f"Validating (Epoch {epoch+1}/{epochs})", 
                leave=False,
                unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
            
            for batch in val_progress:
                mixed_spec = batch['mixed'].to(device, non_blocking=True)
                target_mask = batch['mask'].to(device, non_blocking=True)
                
                # Correct autocast usage for PyTorch 2.6.0
                with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision and device.type == 'cuda', dtype=torch.float16):
                    pred_mask = model(mixed_spec)
                    loss = loss_fn(pred_mask, target_mask)
                
                val_loss += loss.item()
                
                # Update progress bar
                val_progress.set_postfix(loss=f"{loss.item():.4f}")
        
        # Average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate epoch stats
        epoch_time = time.time() - epoch_start_time
        epoch_percent = 100.0 * (epoch + 1) / epochs
        total_elapsed = time.time() - start_training_time
        eta_seconds = (total_elapsed / (epoch + 1)) * (epochs - epoch - 1)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} ({epoch_percent:.1f}%) - "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Time: {epoch_time:.1f}s, ETA: {eta_str}")
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Save current learning rate to history
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
    
    # Stop GPU monitoring if started
    if use_gpu and 'gpu_monitor' in locals() and gpu_monitor is not None:
        gpu_monitor.stop()
        gpu_monitor.print_summary()
    
    # Show GPU utilization after training
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(GPU_DEVICE) / 1024**2
        reserved = torch.cuda.memory_reserved(GPU_DEVICE) / 1024**2
        print(f"\n=== GPU MEMORY USAGE ===")
        print(f"Allocated: {allocated:.1f} MB")
        print(f"Reserved: {reserved:.1f} MB")
    
    # Save model if path provided
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'window_size': window_size,
            'n_fft': N_FFT
        }, model_save_path)
        print(f"Model saved to {model_save_path}")
    
    return model, history

def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    Plot training and validation loss.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training plot saved to {save_path}")
    
    plt.show()

def main():
    """Main function to start training"""
    parser = argparse.ArgumentParser(description='Train voice isolation model')
    parser.add_argument('--window-size', choices=['small', 'medium', 'large'], 
                        default='medium', help='Window size for processing')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=EPOCHS, 
                        help='Number of training epochs')
    parser.add_argument('--samples', type=int, default=2000, 
                        help='Number of training samples to generate')
    parser.add_argument('--gpu', action='store_true', default=USE_GPU, 
                        help='Use GPU acceleration if available')
    parser.add_argument('--mixed-precision', action='store_true', default=MIXED_PRECISION,
                        help='Use mixed precision training (FP16) for faster computation')
    parser.add_argument('--auto-detect-samples', action='store_true', default=AUTO_DETECT_SAMPLES,
                        help='Auto-detect optimal sample count for training')
    parser.add_argument('--deep-model', action='store_true', default=False,
                        help='Use deeper model for higher GPU utilization')
    parser.add_argument('--use-cache', action='store_true', default=USE_CACHED_DATA,
                        help='Use cached preprocessed data for faster loading')
    
    args = parser.parse_args()
    
    # Set paths
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_save_path = os.path.join(OUTPUT_DIR, f"voice_isolation_model_{timestamp}.pth")
    plot_save_path = os.path.join(OUTPUT_DIR, f"training_history_{timestamp}.png")
    
    # Train model with all options
    model, history = train_model(
        window_size=args.window_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_samples=args.samples,
        model_save_path=model_save_path,
        use_gpu=args.gpu,
        use_mixed_precision=args.mixed_precision,
        auto_detect_samples=args.auto_detect_samples,
        use_deep_model=args.deep_model,
        use_cache=args.use_cache
    )
    
    # Plot training history
    plot_training_history(history, save_path=plot_save_path)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()

