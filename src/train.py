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

from .config import (
    VOICE_DIR, NOISE_DIR, OUTPUT_DIR, BATCH_SIZE, EPOCHS, 
    LEARNING_RATE, N_FFT, VALIDATION_SPLIT, RANDOM_SEED,
    USE_GPU, GPU_DEVICE, MIXED_PRECISION, CUDNN_BENCHMARK, GPU_MEMORY_FRACTION
)
from .preprocessing import AudioPreprocessor, get_audio_files
from .model import VoiceIsolationModel, MaskedLoss

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set CUDA-specific configurations
if USE_GPU:
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    
    # Limit GPU memory usage if needed
    if GPU_MEMORY_FRACTION < 1.0:
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, GPU_DEVICE)

class SpectrogramDataset(Dataset):
    """Dataset for spectrogram and masks"""
    
    def __init__(self, voice_dir: str, noise_dir: str, preprocessor: AudioPreprocessor, 
                 num_samples: int = 1000, transform=None, use_gpu: bool = USE_GPU):
        """
        Initialize the dataset.
        
        Args:
            voice_dir: Directory with voice audio files
            noise_dir: Directory with noise audio files
            preprocessor: AudioPreprocessor instance
            num_samples: Number of examples to generate
            transform: Optional transformations
            use_gpu: Whether to use GPU acceleration
        """
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
        print(f"Using device: {self.device} for sample creation")
        
        # Pre-calculate a few samples to estimate processing time
        print("Preparing sample dataset...")
        sample_size = min(5, num_samples)
        
        # Preprocess a small batch to estimate time
        start_time = time.time()
        for _ in tqdm(range(sample_size), desc="Creating test samples", unit="sample"):
            self._create_sample()
        avg_time = (time.time() - start_time) / sample_size
        estimated_time = avg_time * num_samples
        
        print(f"Estimated time to prepare {num_samples} samples: {estimated_time:.1f} seconds")
        if self.use_gpu:
            print(f"GPU accelerated sample creation enabled (expected {avg_time:.3f}s per sample)")
        
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
    use_mixed_precision: bool = MIXED_PRECISION
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
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Set device for training
    device = torch.device(f"cuda:{GPU_DEVICE}" if use_gpu and USE_GPU else "cpu")
    print(f"Training on device: {device}")
    
    if device.type == 'cuda':
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(GPU_DEVICE)
        gpu_mem = torch.cuda.get_device_properties(GPU_DEVICE).total_memory / 1024**3
        print(f"GPU: {gpu_name} with {gpu_mem:.1f} GB memory")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Initialize preprocessor with GPU support
    preprocessor = AudioPreprocessor(window_size=window_size, use_gpu=use_gpu)
    
    print(f"Creating dataset with {num_samples} samples...")
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
    
    print(f"Creating data loaders with batch size {batch_size}...")
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model, loss, and optimizer
    model = VoiceIsolationModel(n_fft=N_FFT)
    loss_fn = MaskedLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Move model to GPU
    model.to(device)
    
    # Initialize scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision and device.type == 'cuda')
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    progress_bar = tqdm(range(epochs), desc="Training", unit="epoch")
    
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
        
        for batch in batch_progress:
            mixed_spec = batch['mixed'].to(device)
            target_mask = batch['mask'].to(device)
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_mixed_precision and device.type == 'cuda'):
                pred_mask = model(mixed_spec)
                loss = loss_fn(pred_mask, target_mask)
            
            # Backward pass with gradient scaling for mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
                mixed_spec = batch['mixed'].to(device)
                target_mask = batch['mask'].to(device)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=use_mixed_precision and device.type == 'cuda'):
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
        
        # Update epoch progress bar
        epoch_time = time.time() - start_time
        progress_bar.set_postfix({
            'train_loss': f"{avg_train_loss:.4f}",
            'val_loss': f"{avg_val_loss:.4f}", 
            'time': f"{epoch_time:.2f}s"
        })
    
    # Add GPU memory stats after training
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(GPU_DEVICE) / 1024**2
        reserved = torch.cuda.memory_reserved(GPU_DEVICE) / 1024**2
        print(f"GPU Memory: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")
    
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
        use_mixed_precision=args.mixed_precision
    )
    
    # Plot training history
    plot_training_history(history, save_path=plot_save_path)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
