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
from dataset import VoiceSeparationDataset, PreGeneratedVoiceSeparationDataset, generate_dataset
from config import *
from prepare_data import prepare_training_data

class ComboLoss(nn.Module):
    """Combination of MSE and L1 losses for better training signal"""
    def __init__(self, mse_weight=0.7, l1_weight=0.3):
        super(ComboLoss, self).__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        return self.mse_weight * mse_loss + self.l1_weight * l1_loss

def train_model(window_size, use_pregenerated=False, num_examples=1000, loss_type='combo', 
               patience=10, max_epochs=None):
    # Get window parameters
    params = get_window_params(window_size)
    window_size_ms = params['window_size_ms']
    window_samples = params['window_samples']
    hop_length = params['hop_length']
    n_fft = params['n_fft']
    
    print(f"Training model with window size: {window_size_ms}ms")
    print(f"Window samples: {window_samples}, Hop length: {hop_length}, FFT size: {n_fft}")
    print(f"Using loss type: {loss_type}")
    
    # Set max epochs
    if max_epochs is None:
        max_epochs = NUM_EPOCHS
    
    # Create model
    model = UNet().to(DEVICE)
    print(f"Training on {DEVICE}")
    
    # Check if training data exists
    if not os.path.exists(VOICE_FOR_TRAINING) or not os.listdir(VOICE_FOR_TRAINING) or \
       not os.path.exists(NOISE_FOR_TRAINING) or not os.listdir(NOISE_FOR_TRAINING):
        print("WARNING: Training data folders are empty!")
        print(f"Please add audio files to {VOICE_FOR_TRAINING} and {NOISE_FOR_TRAINING}")
        print("You can use preprocessed files from PREPROCESSED_VOICE and PREPROCESSED_NOISE")
        print("Or use your own curated files")
        return
    
    # Generate dataset ID based on window size
    model_id = get_model_id(window_size)
    dataset_dir = os.path.join(DATASETS_DIR, f"{model_id}_dataset")
    
    # Create or use pre-generated dataset
    if use_pregenerated:
        # First check if dataset exists
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory {dataset_dir} does not exist. Creating it...")
            os.makedirs(dataset_dir, exist_ok=True)
        
        # Then check if it contains any data
        dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pkl')]
        if not dataset_files or len(dataset_files) < 10:  # Require at least 10 examples
            print(f"No pre-generated dataset found or dataset too small ({len(dataset_files)} files).")
            print(f"Pre-generating dataset with {num_examples} examples...")
            
            try:
                # Clear directory first to avoid mixing different dataset versions
                for file in dataset_files:
                    os.remove(os.path.join(dataset_dir, file))
                
                generate_dataset(
                    VOICE_FOR_TRAINING, 
                    NOISE_FOR_TRAINING, 
                    dataset_dir,
                    num_examples=num_examples,
                    window_size_ms=window_size_ms,
                    window_samples=window_samples,
                    hop_length=hop_length,
                    n_fft=n_fft
                )
                print("Dataset pre-generation completed.")
            except Exception as e:
                print(f"Error generating dataset: {e}")
                print("Falling back to on-the-fly dataset.")
                use_pregenerated = False
        else:
            print(f"Using existing pre-generated dataset with {len(dataset_files)-1} examples.") # -1 for metadata file
        
        # Use pre-generated dataset if it exists
        if use_pregenerated:
            try:
                dataset = PreGeneratedVoiceSeparationDataset(dataset_dir)
                print(f"Successfully loaded pre-generated dataset with {len(dataset)} examples.")
            except Exception as e:
                print(f"Error loading pre-generated dataset: {e}")
                print("Falling back to on-the-fly dataset.")
                use_pregenerated = False
    
    # Use on-the-fly dataset if needed
    if not use_pregenerated:
        print("Using on-the-fly dataset generation.")
        dataset = VoiceSeparationDataset(
            VOICE_FOR_TRAINING, NOISE_FOR_TRAINING, 
            window_size_ms=window_size_ms,
            window_samples=window_samples,
            hop_length=hop_length,
            n_fft=n_fft
        )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Define loss function based on user choice
    if loss_type == 'mse':
        criterion = nn.MSELoss()
        print("Using MSE loss")
    elif loss_type == 'l1':
        criterion = nn.L1Loss()
        print("Using L1 loss (MAE)")
    elif loss_type == 'combo':
        criterion = ComboLoss(mse_weight=0.7, l1_weight=0.3)
        print("Using combo loss (70% MSE + 30% L1)")
    else:
        print(f"Unknown loss type: {loss_type}, defaulting to MSE")
        criterion = nn.MSELoss()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    
    # Training loop
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}") as pbar:
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
                
                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (i + 1), 
                                lr=optimizer.param_groups[0]['lr'])
        
        # Calculate average loss for this epoch
        avg_loss = running_loss / len(dataloader)
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Check if loss improved
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            
            # Save best model
            best_model_path = os.path.join(OUTPUT_DIR, f'{model_id}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'window_size_ms': window_size_ms,
                'window_samples': window_samples,
                'hop_length': hop_length,
                'n_fft': n_fft
            }, best_model_path)
            print(f"New best model saved with loss: {best_loss:.6f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs. Best loss: {best_loss:.6f}")
            
            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch+1} epochs without improvement")
                break
        
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
    print(f'Best model saved to {os.path.join(OUTPUT_DIR, f"{model_id}_best.pth")} with loss: {best_loss:.6f}')

def main():
    parser = argparse.ArgumentParser(description='Train voice isolation model with specific window size')
    parser.add_argument('--window-size', type=str, choices=['SMALL', 'MEDIUM', 'LARGE', 'XLARGE'], 
                      default='SMALL', help='Window size for training (default: SMALL)')
    parser.add_argument('--all', action='store_true', help='Train all window size models')
    parser.add_argument('--prepare-only', action='store_true', help='Only prepare training data, don\'t train')
    
    # Existing arguments for pre-generated datasets
    parser.add_argument('--pregenerate', action='store_true', help='Use pre-generated dataset for training')
    parser.add_argument('--generate-only', action='store_true', help='Only generate dataset, don\'t train')
    parser.add_argument('--num-examples', type=int, default=1000, help='Number of examples to pre-generate')
    
    # New training control arguments
    parser.add_argument('--loss', type=str, choices=['mse', 'l1', 'combo'], default='combo',
                       help='Loss function to use (default: combo)')
    parser.add_argument('--patience', type=int, default=10, 
                       help='Early stopping patience - epochs without improvement (default: 10)')
    parser.add_argument('--max-epochs', type=int, default=None,
                       help='Maximum number of epochs to train (default: use NUM_EPOCHS from config)')
    
    args = parser.parse_args()
    
    # Check if only data preparation is requested
    if args.prepare_only:
        print("Preparing training data...")
        prepare_training_data(remove_silence_flag=True)
        print(f"\nNOTE: Please manually copy files from PREPROCESSED_VOICE to {VOICE_FOR_TRAINING}")
        print(f"and from PREPROCESSED_NOISE to {NOISE_FOR_TRAINING} that you want to use for training")
        return
    
    # Only generate dataset without training
    if args.generate_only:
        print("\n=== DATASET GENERATION MODE ===")
        print(f"Will generate datasets with {args.num_examples} examples each.")
        
        if args.all:
            # Generate datasets for all window sizes
            for window_size in WindowSize:
                params = get_window_params(window_size)
                model_id = get_model_id(window_size)
                dataset_dir = os.path.join(DATASETS_DIR, f"{model_id}_dataset")
                
                print(f"\nGenerating dataset for {window_size.name} ({window_size.value}ms) window size...")
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(dataset_dir, exist_ok=True)
                    
                    # Clear existing files if any
                    existing_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pkl')]
                    if existing_files:
                        print(f"Removing {len(existing_files)} existing files in {dataset_dir}")
                        for file in existing_files:
                            os.remove(os.path.join(dataset_dir, file))
                    
                    # Generate dataset
                    generate_dataset(
                        VOICE_FOR_TRAINING, 
                        NOISE_FOR_TRAINING, 
                        dataset_dir,
                        num_examples=args.num_examples,
                        window_size_ms=params['window_size_ms'],
                        window_samples=params['window_samples'],
                        hop_length=params['hop_length'],
                        n_fft=params['n_fft']
                    )
                    print(f"Dataset generation completed for {window_size.name}.")
                except Exception as e:
                    print(f"Error generating dataset for {window_size.name}: {e}")
                    import traceback
                    print(traceback.format_exc())
        else:
            # Generate dataset for specific window size
            window_size = WindowSize[args.window_size]
            params = get_window_params(window_size)
            model_id = get_model_id(window_size)
            dataset_dir = os.path.join(DATASETS_DIR, f"{model_id}_dataset")
            
            print(f"\nGenerating dataset for {window_size.name} ({window_size.value}ms) window size...")
            try:
                # Create directory if it doesn't exist
                os.makedirs(dataset_dir, exist_ok=True)
                
                # Clear existing files if any
                existing_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pkl')]
                if existing_files:
                    print(f"Removing {len(existing_files)} existing files in {dataset_dir}")
                    for file in existing_files:
                        os.remove(os.path.join(dataset_dir, file))
                
                # Generate dataset
                generate_dataset(
                    VOICE_FOR_TRAINING, 
                    NOISE_FOR_TRAINING, 
                    dataset_dir,
                    num_examples=args.num_examples,
                    window_size_ms=params['window_size_ms'],
                    window_samples=params['window_samples'],
                    hop_length=params['hop_length'],
                    n_fft=params['n_fft']
                )
                print(f"Dataset generation completed for {window_size.name}.")
            except Exception as e:
                print(f"Error generating dataset for {window_size.name}: {e}")
                import traceback
                print(traceback.format_exc())
        
        print("\nAll dataset generation completed.")
        return
    
    if args.all:
        # Train all window sizes
        for window_size in WindowSize:
            train_model(window_size, use_pregenerated=args.pregenerate, 
                      num_examples=args.num_examples, loss_type=args.loss,
                      patience=args.patience, max_epochs=args.max_epochs)
    else:
        # Train specific window size
        window_size = WindowSize[args.window_size]
        train_model(window_size, use_pregenerated=args.pregenerate, 
                  num_examples=args.num_examples, loss_type=args.loss,
                  patience=args.patience, max_epochs=args.max_epochs)

if __name__ == "__main__":
    main()
