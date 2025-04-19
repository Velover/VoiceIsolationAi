import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import load_audio, mix_audio, prepare_spectrograms
from config import *
from tqdm import tqdm
import pickle
import pathlib
import traceback

class VoiceSeparationDataset(Dataset):
    def __init__(self, voice_dir, noise_dir, segment_length=4, transform=None,
                 window_size_ms=None, window_samples=None, hop_length=None, n_fft=None):
        """
        Voice separation dataset.
        
        Args:
            voice_dir: Directory with clean voice files
            noise_dir: Directory with noise files
            segment_length: Length of audio segments in seconds
            transform: Optional transform to be applied on the samples
            window_size_ms: Window size in milliseconds
            window_samples: Window size in samples
            hop_length: Hop length
            n_fft: FFT size
        """
        self.voice_dir = voice_dir
        self.noise_dir = noise_dir
        self.segment_length = segment_length
        self.transform = transform
        
        # Set window parameters
        self.window_size_ms = window_size_ms or WINDOW_SIZE_MS
        self.window_samples = window_samples or WINDOW_SIZE
        self.hop_length = hop_length or HOP_LENGTH
        self.n_fft = n_fft or N_FFT
        
        # Get list of voice and noise files
        self.voice_files = [os.path.join(voice_dir, f) for f in os.listdir(voice_dir) 
                           if f.endswith(('.mp3', '.wav', '.webm'))]
        self.noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) 
                           if f.endswith(('.mp3', '.wav', '.webm'))]
        
        if not self.voice_files:
            raise ValueError(f"No audio files found in {voice_dir}")
        if not self.noise_files:
            raise ValueError(f"No audio files found in {noise_dir}")
    
    def __len__(self):
        return len(self.voice_files) * 10  # Multiple segments per file
    
    def __getitem__(self, idx):
        # Select random voice file
        voice_file = random.choice(self.voice_files)
        noise_file = random.choice(self.noise_files)
        
        # Load audio
        voice = load_audio(voice_file)
        noise = load_audio(noise_file)
        
        # Calculate segment length in samples
        segment_samples = int(self.segment_length * SAMPLE_RATE)
        
        # Ensure audio is long enough, if not, repeat
        if len(voice) < segment_samples:
            repetitions = int(np.ceil(segment_samples / len(voice)))
            voice = np.tile(voice, repetitions)[:segment_samples]
        
        if len(noise) < segment_samples:
            repetitions = int(np.ceil(segment_samples / len(noise)))
            noise = np.tile(noise, repetitions)[:segment_samples]
        
        # Randomly select segment
        if len(voice) > segment_samples:
            start = random.randint(0, len(voice) - segment_samples)
            voice = voice[start:start + segment_samples]
        
        if len(noise) > segment_samples:
            start = random.randint(0, len(noise) - segment_samples)
            noise = noise[start:start + segment_samples]
        
        # Mix audio with random SNR
        snr = random.uniform(MIN_SNR, MAX_SNR)
        mixed, clean = mix_audio(voice, noise, snr)
        
        # Convert to spectrograms
        mixed_mag, mixed_phase = prepare_spectrograms(
            mixed, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_samples
        )
        
        clean_mag, _ = prepare_spectrograms(
            clean,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_samples
        )
        
        # Create target mask (clean / mixed)
        target_mask = np.divide(clean_mag, mixed_mag + 1e-10)  # Avoid division by zero
        target_mask = np.clip(target_mask, 0, 1)
        
        # Make sure both have same dimensions - crucial for U-Net
        # Make dimensions even to avoid issues with max pooling and upsampling
        freq_dim = min(mixed_mag.shape[0], target_mask.shape[0])
        time_dim = min(mixed_mag.shape[1], target_mask.shape[1])
        
        # Ensure dimensions are even (for clean pooling operations)
        freq_dim = freq_dim - (freq_dim % 2)
        time_dim = time_dim - (time_dim % 8)  # Multiple of 8 for 3 pooling operations
        
        mixed_mag = mixed_mag[:freq_dim, :time_dim]
        target_mask = target_mask[:freq_dim, :time_dim]
        mixed_phase = mixed_phase[:freq_dim, :time_dim]
        
        # Convert to tensors
        mixed_mag = torch.tensor(mixed_mag, dtype=torch.float32)
        target_mask = torch.tensor(target_mask, dtype=torch.float32)
        mixed_phase = torch.tensor(mixed_phase, dtype=torch.float32)
        
        # Add channel dimension to target_mask to match model output
        target_mask = target_mask.unsqueeze(0)
        
        if self.transform:
            mixed_mag = self.transform(mixed_mag)
            target_mask = self.transform(target_mask)
        
        return {
            'mixed_mag': mixed_mag,
            'target_mask': target_mask,
            'mixed_phase': mixed_phase
        }

class PreGeneratedVoiceSeparationDataset(Dataset):
    """Dataset that loads pre-generated examples from disk."""
    
    def __init__(self, dataset_path):
        """
        Args:
            dataset_path: Path to the pre-generated dataset directory
        """
        self.dataset_path = dataset_path
        self.examples = []
        
        # Load all generated examples
        example_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.pkl')])
        
        for file in example_files:
            self.examples.append(os.path.join(dataset_path, file))
            
        print(f"Loaded {len(self.examples)} pre-generated examples from {dataset_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Load pre-generated example
        with open(self.examples[idx], 'rb') as f:
            example = pickle.load(f)
        return example


def generate_dataset(voice_dir, noise_dir, output_dir, num_examples=1000, segment_length=4,
                    window_size_ms=None, window_samples=None, hop_length=None, n_fft=None):
    """
    Pre-generate and save a dataset of voice separation examples.
    
    Args:
        voice_dir: Directory with clean voice files
        noise_dir: Directory with noise files
        output_dir: Directory to save the generated examples
        num_examples: Number of examples to generate
        segment_length: Length of audio segments in seconds
        window_size_ms: Window size in milliseconds
        window_samples: Window size in samples
        hop_length: Hop length
        n_fft: FFT size
    """
    # Set window parameters
    window_size_ms = window_size_ms or WINDOW_SIZE_MS
    window_samples = window_samples or WINDOW_SIZE
    hop_length = hop_length or HOP_LENGTH
    n_fft = n_fft or N_FFT
    
    print(f"Pre-generating dataset with parameters: window_size_ms={window_size_ms}, window_samples={window_samples}, hop_length={hop_length}, n_fft={n_fft}")
    print(f"Saving to: {output_dir}")
    
    # Get list of voice and noise files
    voice_files = [os.path.join(voice_dir, f) for f in os.listdir(voice_dir) 
                  if f.endswith(('.mp3', '.wav', '.webm'))]
    noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) 
                  if f.endswith(('.mp3', '.wav', '.webm'))]
    
    if not voice_files:
        raise ValueError(f"No audio files found in {voice_dir}")
    if not noise_files:
        raise ValueError(f"No audio files found in {noise_dir}")
    
    print(f"Found {len(voice_files)} voice files and {len(noise_files)} noise files")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate examples
    print(f"Generating {num_examples} examples...")
    
    successful_examples = 0
    for i in tqdm(range(num_examples)):
        try:
            # Select random voice and noise files
            voice_file = random.choice(voice_files)
            noise_file = random.choice(noise_files)
            
            # Load audio
            voice = load_audio(voice_file)
            noise = load_audio(noise_file)
            
            # Calculate segment length in samples
            segment_samples = int(segment_length * SAMPLE_RATE)
            
            # Ensure audio is long enough, if not, repeat
            if len(voice) < segment_samples:
                repetitions = int(np.ceil(segment_samples / len(voice)))
                voice = np.tile(voice, repetitions)[:segment_samples]
            
            if len(noise) < segment_samples:
                repetitions = int(np.ceil(segment_samples / len(noise)))
                noise = np.tile(noise, repetitions)[:segment_samples]
            
            # Randomly select segment
            if len(voice) > segment_samples:
                start = random.randint(0, len(voice) - segment_samples)
                voice = voice[start:start + segment_samples]
            
            if len(noise) > segment_samples:
                start = random.randint(0, len(noise) - segment_samples)
                noise = noise[start:start + segment_samples]
            
            # Mix audio with random SNR
            snr = random.uniform(MIN_SNR, MAX_SNR)
            mixed, clean = mix_audio(voice, noise, snr)
            
            # Convert to spectrograms
            mixed_mag, mixed_phase = prepare_spectrograms(
                mixed, 
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=window_samples
            )
            
            clean_mag, _ = prepare_spectrograms(
                clean,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=window_samples
            )
            
            # Create target mask (clean / mixed)
            target_mask = np.divide(clean_mag, mixed_mag + 1e-10)  # Avoid division by zero
            target_mask = np.clip(target_mask, 0, 1)
            
            # Make sure both have same dimensions - crucial for U-Net
            # Make dimensions even to avoid issues with max pooling and upsampling
            freq_dim = min(mixed_mag.shape[0], target_mask.shape[0])
            time_dim = min(mixed_mag.shape[1], target_mask.shape[1])
            
            # Ensure dimensions are even (for clean pooling operations)
            freq_dim = freq_dim - (freq_dim % 2)
            time_dim = time_dim - (time_dim % 8)  # Multiple of 8 for 3 pooling operations
            
            mixed_mag = mixed_mag[:freq_dim, :time_dim]
            target_mask = target_mask[:freq_dim, :time_dim]
            mixed_phase = mixed_phase[:freq_dim, :time_dim]
            
            # Convert to tensors
            mixed_mag = torch.tensor(mixed_mag, dtype=torch.float32)
            target_mask = torch.tensor(target_mask, dtype=torch.float32)
            mixed_phase = torch.tensor(mixed_phase, dtype=torch.float32)
            
            # Add channel dimension to target_mask to match model output
            target_mask = target_mask.unsqueeze(0)
            
            # Create example dict
            example = {
                'mixed_mag': mixed_mag,
                'target_mask': target_mask,
                'mixed_phase': mixed_phase
            }
            
            # Save example
            output_file = os.path.join(output_dir, f"example_{i:05d}.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(example, f)
            
            successful_examples += 1
            
            # Print progress occasionally
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Generated {i + 1}/{num_examples} examples. Size of last example: mixed_mag={mixed_mag.shape}, target_mask={target_mask.shape}")
        
        except Exception as e:
            print(f"Error generating example {i}: {e}")
            print(traceback.format_exc())
            continue
    
    print(f"Successfully generated {successful_examples}/{num_examples} examples and saved to {output_dir}")
    
    # Create metadata file with parameters
    metadata = {
        'num_examples': successful_examples,
        'window_size_ms': window_size_ms,
        'window_samples': window_samples,
        'hop_length': hop_length,
        'n_fft': n_fft,
        'segment_length': segment_length
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Dataset metadata saved to {os.path.join(output_dir, 'metadata.pkl')}")
    
    # Verify dataset can be loaded
    try:
        test_dataset = PreGeneratedVoiceSeparationDataset(output_dir)
        print(f"Successfully verified dataset loading: {len(test_dataset)} examples found")
    except Exception as e:
        print(f"WARNING: Could not verify dataset loading: {e}")
        print(traceback.format_exc())
