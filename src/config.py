import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
VOICE_DIR = os.path.join(BASE_DIR, 'VOICE')
NOISE_DIR = os.path.join(BASE_DIR, 'NOISE')
OUTPUT_DIR = os.path.join(BASE_DIR, 'OUTPUT')

# Create directories if they don't exist
os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(NOISE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Audio processing configurations
SAMPLE_RATE = 16000  # Hz
WINDOW_SIZES = {
    'small': 30,    # 30ms window
    'medium': 500,  # 500ms window
    'large': 2000   # 2s window
}
DEFAULT_WINDOW_SIZE = 'medium'  # Default window size choice

# Model configurations
N_FFT = 512  # Number of frequency bins for FFT
HOP_LENGTH = 128  # Hop length for STFT
N_MELS = 64  # Number of Mel bands
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

# Training configurations
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Supported audio formats
SUPPORTED_FORMATS = ['.mp3', '.wav']
