import os
import torch
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
VOICE_DIR = os.path.join(BASE_DIR, 'VOICE')
NOISE_DIR = os.path.join(BASE_DIR, 'NOISE')
OUTPUT_DIR = os.path.join(BASE_DIR, 'OUTPUT')
CACHE_DIR = os.path.join(BASE_DIR, 'CACHE')  # New cache directory for preprocessed data

# Create directories if they don't exist
os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(NOISE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# GPU configurations
USE_GPU = torch.cuda.is_available()
GPU_DEVICE = 0
MIXED_PRECISION = True  # Use mixed precision training (FP16) for faster computation
CUDNN_BENCHMARK = True  # Set to True for fixed-size inputs for better performance
GPU_MEMORY_FRACTION = 0.90  # Increased to 90% for maximum utilization
FORCE_CPU_DATALOADING = True  # Force loading data on CPU regardless of GPU availability
AUTO_DETECT_SAMPLES = True  # Auto-detect optimal sample count based on hardware

# Performance tuning
BATCH_SIZE = 64  # Batch size for GPU efficiency
DATALOADER_WORKERS = 2  # Number of workers for data loading
DATALOADER_PREFETCH = 4  # Increased prefetch factor for better GPU feeding
DATALOADER_PIN_MEMORY = True  # Pin memory for faster CPU->GPU transfers

# Data loading optimization
USE_CACHED_DATA = True  # Enable using cached preprocessed data
PRELOAD_SAMPLES = 100   # Number of samples to keep preloaded in memory
CACHE_MEMORY_LIMIT = 2048  # Memory limit for cache in MB
BACKGROUND_WORKERS = max(1, os.cpu_count() - 1)  # Use all CPU cores except one

# Audio processing configurations
SAMPLE_RATE = 32000  # Hz - this is a good value for voice
WINDOW_SIZES = {
    'small': 30,    # 30ms window
    'medium': 500,  # 500ms window
    'large': 2000   # 2s window
}
DEFAULT_WINDOW_SIZE = 'medium'  # Default window size choice

# Model configurations - ENHANCED FFT SETTINGS
N_FFT = 2048  # Increased from 512 for better frequency resolution
HOP_LENGTH = 512  # Increased from 128 for better processing efficiency
N_MELS = 128  # Increased from 64 for more detailed spectral features
LEARNING_RATE = 0.001
EPOCHS = 50

# Fixed spectrogram time dimension for consistent batch processing
SPEC_TIME_DIM = 256  # Fixed time dimension for spectrograms

# Training configurations
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Supported audio formats
SUPPORTED_FORMATS = ['.mp3', '.wav', '.flac']  # Added FLAC support
