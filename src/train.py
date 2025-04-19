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
    LEARNING_RATE, N_FFT, VALIDATION_SPLIT, RANDOM_SEED
)
from .preprocessing import AudioPreprocessor, get_audio_files
from .model import VoiceIsolationModel, MaskedLoss

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class SpectrogramDataset(Dataset):
    """Dataset for spectrogram and masks"""
    
    def __init__(self, voice_dir: str, noise_dir: str, preprocessor: AudioPreprocessor, 
                 num_samples: int = 1000, transform=None):
        """
        Initialize the dataset.
        
        Args:
            voice_dir: Directory with voice audio files
            noise_dir: Directory with noise audio files
            preprocessor: AudioPreprocessor instance
            num_samples: Number of examples to generate
            transform: Optional transformations
        """
        self.voice_files = get_audio_files(voice_dir)
        self.noise_files = get_audio_files(noise_dir)
        self.preprocessor = preprocessor
        self.num_samples = num_samples
        self.transform = transform
        
        # Generate indices for random sampling
        self.indices = list(range(num_samples))
        
        if not self.voice_files:
            raise ValueError(f"No voice files found in {voice_dir}")
        if not self.noise_files:
            raise ValueError(f"No noise files found in {noise_dir}")
            
        print(f"Found {len(self.voice_files)} voice files and {len(self.noise_files)} noise files")
        
        # Pre-calculate a few samples to estimate processing time
        print("Preparing sample dataset...")
        sample_size = min(5, num_samples)
        self.samples = []
        
        # Preprocess a small batch to estimate time
        start_time = time.time()
        for _ in range(sample_size):
            self._create_sample()
        avg_time = (time.time() - start_time) / sample_size
        estimated_time = avg_time * num_samples
        
        print(f"Estimated time to prepare {num_samples} samples: {estimated_time:.1f} seconds")
        
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
    model_save_path: Optional[str] = None
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
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(window_size=window_size)
    
    print(f"Creating dataset with {num_samples} samples...")
    # Create dataset
    dataset = SpectrogramDataset(
        voice_dir=VOICE_DIR,
        noise_dir=NOISE_DIR,
        preprocessor=preprocessor,
        num_samples=num_samples
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
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
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
            
            # Forward pass
            optimizer.zero_grad()
            pred_mask = model(mixed_spec)
            
            # Compute loss
            loss = loss_fn(pred_mask, target_mask)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress bar with current loss
            batch_progress.set_postfix(loss=f"{loss.item():.4f}")
            
            train_loss += loss.item()
        
        # Validation phase
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
                
                # Forward pass
                pred_mask = model(mixed_spec)
                
                # Compute loss
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
    
    args = parser.parse_args()
    
    # Set paths
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_save_path = os.path.join(OUTPUT_DIR, f"voice_isolation_model_{timestamp}.pth")
    plot_save_path = os.path.join(OUTPUT_DIR, f"training_history_{timestamp}.png")
    
    # Train model
    model, history = train_model(
        window_size=args.window_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_samples=args.samples,
        model_save_path=model_save_path
    )
    
    # Plot training history
    plot_training_history(history, save_path=plot_save_path)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
