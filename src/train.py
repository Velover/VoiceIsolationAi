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
    USE_CACHED_DATA, PRELOAD_SAMPLES, BACKGROUND_WORKERS,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA, CHECKPOINT_FREQUENCY
)
from .preprocessing import AudioPreprocessor, get_audio_files
from .model import VoiceIsolationModel, MaskedLoss, VoiceIsolationModelDeep
from .dataset import SpectrogramDataset, CachedSpectrogramDataset

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

class PreprocessedDataset(Dataset):
    """Dataset that uses pre-generated samples for maximum speed"""
    
    def __init__(self, samples_dir: str, 
                 num_samples: int = None, transform=None):
        """
        Initialize dataset with pre-generated samples.
        
        Args:
            samples_dir: Directory containing preprocessed sample files
            num_samples: Number of samples to use (None = use all available)
            transform: Optional transforms to apply
        """
        self.samples_dir = samples_dir
        self.transform = transform
        
        # Get list of all sample files
        self.sample_files = [
            os.path.join(samples_dir, f) for f in os.listdir(samples_dir)
            if f.endswith('.pt')
        ]
        
        # Limit to requested number if specified
        if num_samples is not None and num_samples > 0 and num_samples < len(self.sample_files):
            self.sample_files = self.sample_files[:num_samples]
        
        print(f"Using {len(self.sample_files)} pre-generated samples from {samples_dir}")
        
        # Sample loading stats for diagnostics
        self.load_times = []
    
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        """Get a pre-generated sample from disk"""
        start_time = time.time()
        try:
            # Load sample from disk
            sample_path = self.sample_files[idx]
            sample_data = torch.load(sample_path, map_location='cpu')
            
            mixed_spec = sample_data['mixed']
            mask = sample_data['mask']
            
            # Apply transformations if any
            if self.transform:
                mixed_spec = self.transform(mixed_spec)
                mask = self.transform(mask)
            
            # Add channel dimension if needed
            if mixed_spec.dim() == 2:
                mixed_spec = mixed_spec.unsqueeze(0)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            
            # Track loading times for performance monitoring
            load_time = time.time() - start_time
            self.load_times.append(load_time)
            
            # Only keep the most recent times
            if len(self.load_times) > 100:
                self.load_times = self.load_times[-100:]
                
            return {
                'mixed': mixed_spec,
                'mask': mask
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Try another sample on error
            return self.__getitem__((idx + 1) % len(self))
    
    def get_loading_stats(self):
        """Get statistics about sample loading performance"""
        if not self.load_times:
            return {"avg_time": 0, "min_time": 0, "max_time": 0}
        
        return {
            "avg_time": sum(self.load_times) / len(self.load_times),
            "min_time": min(self.load_times),
            "max_time": max(self.load_times)
        }

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
    use_cache: bool = USE_CACHED_DATA,
    use_preprocessed: bool = False,  # New flag for pre-generated samples
    preprocessed_dir: Optional[str] = None,  # Path to pre-generated samples
    enable_early_stopping: bool = True  # Add early stopping parameter
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
        use_preprocessed: Whether to use pre-generated samples
        preprocessed_dir: Directory with pre-generated samples
        enable_early_stopping: Enable early stopping during training
        
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
    
    # Create the appropriate dataset based on what's available and requested
    if use_preprocessed and preprocessed_dir:
        # Use pre-generated samples (fastest option)
        if not os.path.exists(preprocessed_dir):
            print(f"Warning: Preprocessed samples directory {preprocessed_dir} not found")
            print("Falling back to standard dataset")
            use_preprocessed = False
        else:
            print(f"Using pre-generated samples from {preprocessed_dir} for maximum speed")
            dataset = PreprocessedDataset(
                samples_dir=preprocessed_dir,
                num_samples=num_samples
            )
            actual_workers = min(4, DATALOADER_WORKERS)  # Can use multiple workers safely
            print(f"Using {actual_workers} dataloader workers with pre-generated samples")
    
    if not use_preprocessed:
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
    
    # Add early stopping variables
    if enable_early_stopping:
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
    
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
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            
            batch_idx += 1
            
            # Process batch
            mixed_spec = batch['mixed'].to(device)
            target_mask = batch['mask'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            pred_mask = model(mixed_spec)
            loss = loss_fn(pred_mask, target_mask)
            loss.backward()
            optimizer.step()
            
            # Update progress
            train_loss += loss.item()
            batch_times.append(time.time() - batch_start)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                mixed_spec = batch['mixed'].to(device, non_blocking=True)
                target_mask = batch['mask'].to(device, non_blocking=True)
                
                pred_mask = model(mixed_spec)
                loss = loss_fn(pred_mask, target_mask)
                
                val_loss += loss.item()
        
        # Average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Save checkpoint at regular intervals
        if (epoch + 1) % CHECKPOINT_FREQUENCY == 0:
            checkpoint_path = os.path.join(
                OUTPUT_DIR, 
                f"checkpoint_{window_size}_{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'window_size': window_size,
                'n_fft': N_FFT,
                'deep_model': use_deep_model,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Early stopping logic
        if enable_early_stopping:
            if avg_val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"Early stopping patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
                
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    # Restore best model
                    model.load_state_dict(best_model_state)
                    break
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Save current learning rate to history
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
    
    # Save model if path provided
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'window_size': window_size,
            'n_fft': N_FFT,
            'deep_model': use_deep_model,  # Save if deep model was used
            'creation_date': time.strftime("%Y-%m-%d"),
            'training_samples': num_samples
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
    parser.add_argument('--preprocessed', action='store_true', default=False,
                        help='Use pre-generated samples for maximum speed')
    parser.add_argument('--preprocessed-dir', default=None,
                        help='Directory containing pre-generated samples')
    
    args = parser.parse_args()
    
    # Auto-detect preprocessed samples directory if not specified
    if args.preprocessed and args.preprocessed_dir is None:
        default_samples_dir = os.path.join(CACHE_DIR, f'samples_{args.window_size}')
        if os.path.exists(default_samples_dir):
            args.preprocessed_dir = default_samples_dir
            print(f"Auto-detected samples directory: {args.preprocessed_dir}")
        else:
            print(f"No preprocessed samples found at {default_samples_dir}")
            print("You can generate samples with: python preprocess_samples.py")
            args.preprocessed = False
    
    # Set paths with clear window size identification
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_type = "deep" if args.deep_model else "standard"
    model_save_path = os.path.join(
        OUTPUT_DIR, 
        f"voice_isolation_{args.window_size}_{model_type}_{timestamp}.pth"
    )
    plot_save_path = os.path.join(
        OUTPUT_DIR, 
        f"training_history_{args.window_size}_{model_type}_{timestamp}.png"
    )
    
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
        use_cache=args.use_cache,
        use_preprocessed=args.preprocessed,
        preprocessed_dir=args.preprocessed_dir
    )
    
    # Plot training history
    plot_training_history(history, save_path=plot_save_path)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()

