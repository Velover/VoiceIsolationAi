import os
import torch
import torchaudio
import numpy as np
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import time

from .config import (
    SAMPLE_RATE, N_FFT, HOP_LENGTH, 
    SUPPORTED_FORMATS, WINDOW_SIZES, SPEC_TIME_DIM,
    USE_GPU, GPU_DEVICE
)

class AudioPreprocessor:
    """
    Handles audio data preprocessing for voice isolation model
    """
    def __init__(self, window_size: str = 'medium', use_gpu: bool = USE_GPU):
        """
        Initialize the audio preprocessor.
        
        Args:
            window_size: Size of the window for processing ('small', 'medium', 'large')
            use_gpu: Whether to use GPU acceleration if available
        """
        if window_size not in WINDOW_SIZES:
            raise ValueError(f"Window size must be one of {list(WINDOW_SIZES.keys())}")
        
        self.window_size_ms = WINDOW_SIZES[window_size]
        self.window_size_samples = int(SAMPLE_RATE * self.window_size_ms / 1000)
        
        # Improved GPU handling
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device(f"cuda:{GPU_DEVICE}" if self.use_gpu else "cpu")
        
        # Move the Hann window to GPU for faster STFT computation
        self.window = torch.hann_window(N_FFT).to(self.device)
        
        if self.use_gpu:
            # Verify CUDA is working correctly
            test_tensor = torch.zeros(1, device=self.device)
            if test_tensor.device.type != 'cuda':
                print(f"⚠️ Warning: Failed to allocate tensor on GPU despite CUDA being available")
                self.use_gpu = False
                self.device = torch.device("cpu")
            else:
                # Print what GPU the processor is using
                print(f"AudioPreprocessor using GPU: {torch.cuda.get_device_name(GPU_DEVICE)}")
                # Warm up CUDA for faster first operation
                torch.cuda.synchronize()
        
    def load_audio(self, file_path: str) -> torch.Tensor:
        """
        Load and resample audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Resampled audio tensor
        """
        # Check if file exists and has supported format
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {SUPPORTED_FORMATS}")
        
        # Get file size for progress reporting
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"Loading: {os.path.basename(file_path)} ({file_size:.2f} MB)")
        
        # Load audio file
        audio, sr = torchaudio.load(file_path)
        
        # Convert to mono if needed
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            print(f"Resampling from {sr}Hz to {SAMPLE_RATE}Hz...")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            audio = resampler(audio)
        
        # Move tensor to GPU if enabled - ensure this works
        if self.use_gpu:
            audio = audio.to(self.device, non_blocking=True)
        
        # Print audio stats
        duration = audio.shape[1] / SAMPLE_RATE
        print(f"Audio loaded: {duration:.2f}s, {audio.shape[1]} samples")
        
        return audio
    
    def compute_stft(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute Short-Time Fourier Transform of audio signal.
        
        Args:
            audio: Audio tensor [1, samples]
            
        Returns:
            Complex STFT tensor [frequency_bins, time_frames]
        """
        # Make sure audio is on the correct device
        if audio.device != self.device:
            audio = audio.to(self.device)
            
        return torch.stft(
            audio[0],
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH,
            window=self.window,
            return_complex=True
        )
    
    def compute_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute the magnitude spectrogram of audio signal.
        
        Args:
            audio: Audio tensor [1, samples]
            
        Returns:
            Magnitude spectrogram tensor [frequency_bins, time_frames]
        """
        # Ensure computation happens on the GPU
        if self.use_gpu and audio.device.type != 'cuda':
            audio = audio.to(self.device, non_blocking=True)
        
        # Using a CUDA stream if available to parallelize with other GPU operations
        if self.use_gpu:
            with torch.cuda.stream(torch.cuda.Stream()):
                stft = self.compute_stft(audio)
                magnitude = torch.abs(stft)
                # Synchronize to ensure computation is complete
                torch.cuda.current_stream().synchronize()
        else:
            stft = self.compute_stft(audio)
            magnitude = torch.abs(stft)
        
        return magnitude
    
    def standardize_spectrogram(self, spectrogram: torch.Tensor, time_dim: int = SPEC_TIME_DIM) -> torch.Tensor:
        """
        Standardize spectrogram to have a fixed time dimension by padding or cropping.
        
        Args:
            spectrogram: Input spectrogram tensor [frequency_bins, time_frames]
            time_dim: Target time dimension
            
        Returns:
            Standardized spectrogram with shape [frequency_bins, time_dim]
        """
        freq_dim, spec_time = spectrogram.shape
        
        # If already the right size, return as is
        if spec_time == time_dim:
            return spectrogram
        
        # Create a new tensor filled with zeros
        standardized = torch.zeros((freq_dim, time_dim), dtype=spectrogram.dtype, 
                                   device=spectrogram.device)
        
        if spec_time > time_dim:
            # Crop if too long
            standardized = spectrogram[:, :time_dim]
        else:
            # Pad with zeros if too short
            standardized[:, :spec_time] = spectrogram
        
        return standardized
    
    def create_training_example(self, voice_path: str, noise_path: Optional[str] = None, 
                              mix_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a training example by mixing voice and noise.
        
        Args:
            voice_path: Path to voice audio file
            noise_path: Path to noise audio file (optional)
            mix_ratio: Ratio of voice to noise (0-1)
            
        Returns:
            Tuple of (mixed spectrogram, voice mask)
        """
        # Simple approach without autocast for more reliable GPU usage
        # Load voice
        voice = self.load_audio(voice_path)
        
        # Load noise if provided
        if noise_path:
            noise = self.load_audio(noise_path)
            
            # Adjust lengths to match
            min_length = min(voice.shape[1], noise.shape[1])
            voice = voice[:, :min_length]
            noise = noise[:, :min_length]
            
            # Mix voice and noise - ensure on same device
            if voice.device != noise.device:
                noise = noise.to(voice.device)
                
            noise = noise * (1 - mix_ratio)
            voice = voice * mix_ratio
            mixed = voice + noise
        else:
            mixed = voice
        
        # Compute spectrograms - ensure GPU acceleration if available
        start = time.time() if self.use_gpu else 0
        voice_spec = self.compute_spectrogram(voice)
        mixed_spec = self.compute_spectrogram(mixed)
        if self.use_gpu:
            torch.cuda.synchronize()
            # Uncomment for debugging
            # comp_time = time.time() - start
            # print(f"GPU spectrogram computation time: {comp_time*1000:.1f}ms")
        
        # Create mask (1 where voice is present, 0 elsewhere)
        # Using a simple threshold for mask creation
        epsilon = 1e-10  # Small value to avoid division by zero
        mask = (voice_spec / (mixed_spec + epsilon)) > 0.5
        mask = mask.float()
        
        # Standardize dimensions for batch processing
        mixed_spec = self.standardize_spectrogram(mixed_spec)
        mask = self.standardize_spectrogram(mask)
        
        return mixed_spec, mask
    
    def apply_mask(self, spectrogram: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply mask to spectrogram to isolate voice.
        
        Args:
            spectrogram: Magnitude spectrogram
            mask: Binary mask (0-1)
            
        Returns:
            Isolated voice spectrogram
        """
        return spectrogram * mask
    
    def spectrogram_to_audio(self, spectrogram: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        Convert spectrogram back to audio using phase information.
        
        Args:
            spectrogram: Magnitude spectrogram
            phase: Phase information from original STFT
            
        Returns:
            Reconstructed audio tensor
        """
        complex_spec = spectrogram * torch.exp(1j * phase)
        
        # Use torch's inverse STFT
        audio = torch.istft(
            complex_spec,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window=torch.hann_window(N_FFT)
        )
        
        return audio.unsqueeze(0)  # Add channel dimension

    def preload_files(self, file_paths: List[str], max_files: int = 10) -> Dict[str, torch.Tensor]:
        """
        Preload multiple audio files for faster batch processing
        
        Args:
            file_paths: List of audio file paths to preload
            max_files: Maximum number of files to load (to avoid memory issues)
            
        Returns:
            Dictionary mapping file paths to audio tensors
        """
        preloaded = {}
        files_to_load = file_paths[:max_files]
        
        for file_path in files_to_load:
            try:
                preloaded[file_path] = self.load_audio(file_path)
            except Exception as e:
                print(f"Error preloading {file_path}: {e}")
                
        return preloaded
    
    def batch_compute_spectrograms(self, audio_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute spectrograms for a batch of audio tensors
        
        Args:
            audio_list: List of audio tensors
            
        Returns:
            List of spectrogram tensors
        """
        results = []
        
        # Process in batches if we're using GPU for better throughput
        if self.use_gpu:
            # Process in smaller batches to avoid CUDA OOM
            batch_size = 4
            for i in range(0, len(audio_list), batch_size):
                batch = audio_list[i:i+batch_size]
                
                # Process batch
                with torch.cuda.stream(torch.cuda.Stream()):
                    batch_results = [self.compute_spectrogram(audio) for audio in batch]
                    # Move results to CPU to free GPU memory
                    batch_results = [spec.cpu() for spec in batch_results]
                    results.extend(batch_results)
                    
                # Clear CUDA cache
                torch.cuda.empty_cache()
        else:
            # Process individually for CPU
            results = [self.compute_spectrogram(audio) for audio in audio_list]
        
        return results
    
    def create_training_example_fast(self, voice: torch.Tensor, noise: Optional[torch.Tensor] = None, 
                                mix_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a training example using pre-loaded tensors (faster version)
        
        Args:
            voice: Voice audio tensor
            noise: Noise audio tensor (optional)
            mix_ratio: Ratio of voice to noise (0-1)
            
        Returns:
            Tuple of (mixed spectrogram, voice mask)
        """
        # Mix audio
        if noise is not None:
            # Adjust lengths to match
            min_length = min(voice.shape[1], noise.shape[1])
            voice = voice[:, :min_length]
            noise = noise[:, :min_length]
            
            # Mix voice and noise - ensure same device
            if voice.device != noise.device:
                noise = noise.to(voice.device)
                
            mixed = voice * mix_ratio + noise * (1 - mix_ratio)
        else:
            mixed = voice
        
        # Compute spectrograms
        voice_spec = self.compute_spectrogram(voice)
        mixed_spec = self.compute_spectrogram(mixed)
        
        # Create mask
        epsilon = 1e-10
        mask = (voice_spec / (mixed_spec + epsilon)) > 0.5
        mask = mask.float()
        
        # Standardize
        mixed_spec = self.standardize_spectrogram(mixed_spec)
        mask = self.standardize_spectrogram(mask)
        
        return mixed_spec, mask

def get_audio_files(directory: str) -> List[str]:
    """
    Get all supported audio files from a directory.
    
    Args:
        directory: Directory path
    
    Returns:
        List of file paths
    """
    files = []
    
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in SUPPORTED_FORMATS:
                files.append(file_path)
                
    return files
