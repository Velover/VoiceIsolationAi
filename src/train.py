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

from .config import (
    VOICE_DIR, NOISE_DIR, OUTPUT_DIR, BATCH_SIZE, EPOCHS, 
    LEARNING_RATE, N_FFT, VALIDATION_SPLIT, RANDOM_SEED,
    USE_GPU, GPU_DEVICE, MIXED_PRECISION, CUDNN_BENCHMARK, GPU_MEMORY_FRACTION,
    DATALOADER_WORKERS, DATALOADER_PREFETCH, DATALOADER_PIN_MEMORY, AUTO_DETECT_SAMPLES
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

class SpectrogramDataset(Dataset):
    """Dataset for spectrogram and masks"""
    
    def __init__(self, voice_dir: str, noise_dir: str, preprocessor: AudioPreprocessor, 
                 num_samples: int = 1000, transform=None, use_gpu: bool = USE_GPU):
        self.voice_files = get_audio_files(voice_dir)
        self.noise_files = get_audio_files(noise_dir)
        self.preprocessor = preprocessor
        self.num_samples = num_samples
        self.transform = transform
        self.use_gpu = use_gpu and USE_GPU
        self.device = torch.device(f"cuda:{GPU_DEVICE}" if self.use_gpu else "cpu")
        
        # Generate indices for random sampling
        self.indices = list(range(num_samples))
        
        if not self.voice_files:
            raise ValueError(f"No voice files found in {voice_dir}")
        if not self.noise_files:
            raise ValueError(f"No noise files found in {noise_dir}")
            
        print(f"Found {len(self.voice_files)} voice files and {len(self.noise_files)} noise files")
        
        # Pre-calculate a few samples to estimate processing time - reduce verbosity
        print("\n=== PREPARING SAMPLE DATASET ===")
        sample_size = min(5, num_samples)
        
        # Preprocess a small batch to estimate time
        start_time = time.time()
        for i in range(sample_size):
            self._create_sample()
            print(f"  Created test sample {i+1}/{sample_size}", end="\r")
        print() # New line after progress
        
        avg_time = (time.time() - start_time) / sample_size
        estimated_time = avg_time * num_samples
        total_min = estimated_time // 60
        total_sec = estimated_time % 60
        
        # More concise output
        print(f"Sample creation speed: {avg_time:.2f}s per sample")
        if self.use_gpu:
            print(f"GPU acceleration: ENABLED ✓")
        else:
            print(f"GPU acceleration: DISABLED ✗")
        print(f"Estimated time to create {num_samples} samples: {total_min:.0f}m {total_sec:.0f}s")
        
    def _create_sample(self):
        """Create a single training sample"""
        # Select random files
        voice_path = random.choice(self.voice_files)
        
        # 80% mix with noise, 20% just voice
        if random.random() < 0.8:
            noise_path = random.choice(self.noise_files)
            mix_ratio = random.uniform(0.3, 0.7)
        else:
            noise_path = None
            mix_ratio = 1.0
            
        try:
            return self.preprocessor.create_training_example(
                voice_path, noise_path, mix_ratio
            )
        except Exception as e:
            print(f"Error creating sample from {voice_path}: {e}")
            # Return a different sample
            return self._create_sample()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a training example.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with mixed spectrogram and mask
        """
        try:
            # Create example on-the-fly for memory efficiency
            mixed_spec, mask = self._create_sample()
            
            # Apply transformations if any
            if self.transform:
                mixed_spec = self.transform(mixed_spec)
                mask = self.transform(mask)
                
            # Add channel dimension
            mixed_spec = mixed_spec.unsqueeze(0)
            mask = mask.unsqueeze(0)
            
            # IMPORTANT: Move tensors to CPU before returning - DataLoader will handle pinning/GPU transfer
            if mixed_spec.device.type != 'cpu':
                mixed_spec = mixed_spec.cpu()
            if mask.device.type != 'cpu':
                mask = mask.cpu()
            
            return {
                'mixed': mixed_spec,
                'mask': mask
            }
        except Exception as e:
            # If there's an error, return a different sample
            print(f"Error loading sample: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

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
    use_deep_model: bool = False
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
    # Create dataset with GPU support
    dataset = SpectrogramDataset(
        voice_dir=VOICE_DIR,
        noise_dir=NOISE_DIR,
        preprocessor=preprocessor,
        num_samples=num_samples,
        use_gpu=use_gpu
    )
    
    # Split into train and validation sets
    val_size = int(VALIDATION_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n=== PREPARING DATA LOADERS ===")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {DATALOADER_WORKERS}")
    print(f"Prefetch factor: {DATALOADER_PREFETCH}")
    
    # Create dataloaders with optimized settings for GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=DATALOADER_WORKERS,
        pin_memory=DATALOADER_PIN_MEMORY and device.type == 'cuda',
        prefetch_factor=DATALOADER_PREFETCH,
        persistent_workers=DATALOADER_WORKERS > 0,
        drop_last=True  # Drop incomplete batches to maintain optimal GPU utilization
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        pin_memory=DATALOADER_PIN_MEMORY and device.type == 'cuda',
        prefetch_factor=DATALOADER_PREFETCH,
        persistent_workers=DATALOADER_WORKERS > 0,
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
    progress_bar = tqdm(range(epochs), desc="Training", unit="epoch")
    
    # Create CUDA streams for overlapping operations
    if device.type == 'cuda':
        # Main computing stream
        compute_stream = torch.cuda.Stream()
        # Data transfer stream
        transfer_stream = torch.cuda.Stream()
    
    for epoch in progress_bar:
        # Training phase
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        # Create batch progress bar
        batch_progress = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{epochs}", 
            leave=False,
            unit="batch"
        )
        
        # Force garbage collection before epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        for batch in batch_progress:
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
            
            # Update progress bar with current loss
            batch_progress.set_postfix(loss=f"{loss.item():.4f}")
            
            train_loss += loss.item()
        
        # Validation phase with mixed precision
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_progress = tqdm(
                val_loader, 
                desc=f"Validating", 
                leave=False,
                unit="batch"
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
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Save current learning rate to history
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # Update epoch progress bar
        epoch_time = time.time() - start_time
        progress_bar.set_postfix({
            'train_loss': f"{avg_train_loss:.4f}",
            'val_loss': f"{avg_val_loss:.4f}",
            'lr': f"{current_lr:.6f}",
            'time': f"{epoch_time:.2f}s"
        })
    
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
    
    args = parser.parse_args()
    
    # Set paths
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_save_path = os.path.join(OUTPUT_DIR, f"voice_isolation_model_{timestamp}.pth")
    plot_save_path = os.path.join(OUTPUT_DIR, f"training_history_{timestamp}.png")
    
    # Train model with GPU options
    model, history = train_model(
        window_size=args.window_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_samples=args.samples,
        model_save_path=model_save_path,
        use_gpu=args.gpu,
        use_mixed_precision=args.mixed_precision,
        auto_detect_samples=args.auto_detect_samples,
        use_deep_model=args.deep_model  # Use args.deep_model here
    )
    
    # Plot training history
    plot_training_history(history, save_path=plot_save_path)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()

