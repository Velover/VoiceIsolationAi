import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import load_audio, mix_audio, prepare_spectrograms
from config import *

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
        
        # Convert to tensors
        mixed_mag = torch.tensor(mixed_mag, dtype=torch.float32)
        target_mask = torch.tensor(target_mask, dtype=torch.float32)
        mixed_phase = torch.tensor(mixed_phase, dtype=torch.float32)
        
        if self.transform:
            mixed_mag = self.transform(mixed_mag)
            target_mask = self.transform(target_mask)
        
        return {
            'mixed_mag': mixed_mag,
            'target_mask': target_mask,
            'mixed_phase': mixed_phase
        }
