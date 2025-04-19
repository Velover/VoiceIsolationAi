import torch
import os
from enum import Enum

class WindowSize(Enum):
    SMALL = 30    # 30ms - good for detailed voice characteristics
    MEDIUM = 100  # 100ms - balanced approach
    LARGE = 500   # 500ms - better for distinguishing similar voices
    XLARGE = 2000 # 2s - captures more voice characteristics

# Audio processing parameters
SAMPLE_RATE = 16000  # 16kHz
DEFAULT_WINDOW_SIZE = WindowSize.SMALL  # Default window size

# Function to calculate window-dependent parameters
def get_window_params(window_size):
    # Convert enum to ms value if it's an enum
    if isinstance(window_size, WindowSize):
        window_size_ms = window_size.value
    else:
        window_size_ms = window_size
        
    window_samples = int(SAMPLE_RATE * window_size_ms / 1000)  # in samples
    hop_length = window_samples // 4  # 75% overlap
    
    # Scale FFT size based on window size
    if window_size_ms <= 50:
        n_fft = 2048
    elif window_size_ms <= 200:
        n_fft = 4096
    elif window_size_ms <= 1000:
        n_fft = 8192
    else:
        n_fft = 16384
        
    return {
        'window_size_ms': window_size_ms,
        'window_samples': window_samples,
        'hop_length': hop_length,
        'n_fft': n_fft
    }

# Get default window parameters
default_params = get_window_params(DEFAULT_WINDOW_SIZE)
WINDOW_SIZE_MS = default_params['window_size_ms']
WINDOW_SIZE = default_params['window_samples']
HOP_LENGTH = default_params['hop_length']
N_FFT = default_params['n_fft']

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
VOICE_DIR = "VOICE"
NOISE_DIR = "NOISE"
PREPROCESSED_VOICE = "PREPROCESSED_VOICE"
PREPROCESSED_NOISE = "PREPROCESSED_NOISE"
VOICE_FOR_TRAINING = "VOICE_FOR_TRAINING"
NOISE_FOR_TRAINING = "NOISE_FOR_TRAINING" 
SAMPLES_DIR = "SAMPLES"
OUTPUT_DIR = "OUTPUT"
DATASETS_DIR = "DATASETS"  # Directory for pre-generated datasets

# Model ID format
def get_model_id(window_size):
    if isinstance(window_size, WindowSize):
        return f"voice_isolation_ai_{window_size.value}"
    else:
        return f"voice_isolation_ai_{window_size}"

# Data augmentation
MIN_SNR = 0  # dB
MAX_SNR = 20  # dB

# Model parameters
N_CHANNELS = 32  # Starting number of channels in U-Net

# Create directories if they don't exist
for directory in [OUTPUT_DIR, PREPROCESSED_VOICE, PREPROCESSED_NOISE, VOICE_FOR_TRAINING, 
                 NOISE_FOR_TRAINING, SAMPLES_DIR, DATASETS_DIR]:
    os.makedirs(directory, exist_ok=True)
