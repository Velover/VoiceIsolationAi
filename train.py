import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import torch.nn.functional as F

from model import UNet
from dataset import VoiceSeparationDataset
from config import *

def train_model(window_size):
    # Get window parameters
    params = get_window_params(window_size)
    window_size_ms = params['window_size_ms']
    window_samples = params['window_samples']
    hop_length = params['hop_length']
    n_fft = params['n_fft']
    
    print(f"Training model with window size: {window_size_ms}ms")
    print(f"Window samples: {window_samples}, Hop length: {hop_length}, FFT size: {n_fft}")
    
    # Create model
    model = UNet().to(DEVICE)
    print(f"Training on {DEVICE}")
    
    # Create dataset and dataloader
    dataset = VoiceSeparationDataset(
        VOICE_DIR, NOISE_DIR, 
        window_size_ms=window_size_ms,
        window_samples=window_samples,
        hop_length=hop_length,
        n_fft=n_fft
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Model ID
    model_id = get_model_id(window_size)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}") as pbar:
            for i, data in enumerate(pbar):
                mixed_mag = data['mixed_mag'].to(DEVICE)
                target_mask = data['target_mask'].to(DEVICE)
                
                # Ensure input has the right dimensions
                if mixed_mag.dim() == 3:  # [batch_size, frequency_bins, time_frames]
                    mixed_mag = mixed_mag.unsqueeze(1)  # Add channel dimension
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                predicted_mask = model(mixed_mag)
                
                # Print shapes for debugging (first batch only)
                if i == 0 and epoch == 0:
                    print(f"Mixed mag shape: {mixed_mag.shape}")
                    print(f"Predicted mask shape: {predicted_mask.shape}")
                    print(f"Target mask shape: {target_mask.shape}")
                
                # Resize target mask to match predicted mask size exactly
                if target_mask.shape != predicted_mask.shape:
                    print(f"Resizing target mask from {target_mask.shape} to {predicted_mask.shape}")
                    target_mask = F.interpolate(
                        target_mask, 
                        size=(predicted_mask.shape[2], predicted_mask.shape[3]),
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Calculate loss
                loss = criterion(predicted_mask, target_mask)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (i + 1))
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f'{model_id}_checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
                'window_size_ms': window_size_ms,
                'window_samples': window_samples,
                'hop_length': hop_length,
                'n_fft': n_fft
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(OUTPUT_DIR, f'{model_id}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'window_size_ms': window_size_ms,
        'window_samples': window_samples,
        'hop_length': hop_length,
        'n_fft': n_fft
    }, final_model_path)
    print(f'Training completed and model saved to {final_model_path}!')

def main():
    parser = argparse.ArgumentParser(description='Train voice isolation model with specific window size')
    parser.add_argument('--window-size', type=str, choices=['SMALL', 'MEDIUM', 'LARGE', 'XLARGE'], 
                      default='SMALL', help='Window size for training (default: SMALL)')
    parser.add_argument('--all', action='store_true', help='Train all window size models')
    
    args = parser.parse_args()
    
    if args.all:
        # Train all window sizes
        for window_size in WindowSize:
            train_model(window_size)
    else:
        # Train specific window size
        window_size = WindowSize[args.window_size]
        train_model(window_size)

if __name__ == "__main__":
    main()
