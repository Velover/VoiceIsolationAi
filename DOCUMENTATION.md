# Voice Isolation AI - Documentation

This documentation provides detailed instructions on how to use the Voice Isolation AI system for training and processing audio files.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Configuration Guide](#configuration-guide)
5. [Window Size Options](#window-size-options)
6. [Data Preparation](#data-preparation)
7. [Training Guide](#training-guide)
8. [Processing Guide](#processing-guide)
9. [Troubleshooting](#troubleshooting)

## Project Overview

This project uses deep learning to isolate a specific person's voice from background noise, ambient sounds, and other voices. It works by:

1. Taking an audio file with mixed voices and noise
2. Converting it to the frequency domain using Short-Time Fourier Transform (STFT)
3. Using a U-Net model to generate a mask that identifies the target voice
4. Applying this mask to isolate the target voice
5. Converting back to the time domain

## Installation

### Requirements

- Python 3.7 or higher
- PyTorch 1.7 or higher
- librosa
- torchaudio
- numpy
- tqdm
- soundfile

### Setup

1. Clone this repository
2. Install dependencies:

```bash
pip install torch torchaudio librosa numpy tqdm soundfile
```

3. Prepare your data folders:
   - Create a `VOICE` folder with the target person's voice recordings
   - Create a `NOISE` folder with other voices and background noise

## Project Structure

```
├── config.py         # Configuration settings
├── utils.py          # Audio processing utilities
├── dataset.py        # Dataset for training data preparation
├── model.py          # U-Net model architecture
├── train.py          # Training script
├── process.py        # Audio processing script
├── prepare_data.py   # Data preprocessing script
├── VOICE/            # Target voice recordings (raw)
├── NOISE/            # Other voices and noise (raw)
├── PREPROCESSED_VOICE/ # Processed voice files with silence removed
├── PREPROCESSED_NOISE/ # Processed noise files with silence removed
├── SAMPLES/          # Generated sample mixtures for testing
└── OUTPUT/           # Trained models and outputs
```

## Configuration Guide

All configurable parameters are located in `config.py`. Here's what each parameter does:

### Audio Processing Parameters

```python
# Audio processing parameters
SAMPLE_RATE = 16000  # Sample rate in Hz
WINDOW_SIZE_MS = 30  # Size of each audio window in milliseconds
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)  # Window size in samples
HOP_LENGTH = WINDOW_SIZE // 4  # Hop length (75% overlap)
N_FFT = 2048  # FFT size
```

#### Examples:

- **30ms window (default)**: Good for capturing detailed voice characteristics

  ```python
  WINDOW_SIZE_MS = 30
  ```

- **50ms window**: Better for lower frequency resolution

  ```python
  WINDOW_SIZE_MS = 50
  ```

- **100ms window**: Better for isolated words, might lose some detail
  ```python
  WINDOW_SIZE_MS = 100
  ```

### Training Parameters

```python
# Training parameters
BATCH_SIZE = 8  # Number of samples per batch
NUM_EPOCHS = 100  # Number of training epochs
LEARNING_RATE = 0.001  # Learning rate for optimizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Training device
```

#### Examples:

- **Small dataset/limited memory**:

  ```python
  BATCH_SIZE = 4
  NUM_EPOCHS = 50
  ```

- **Large dataset/more memory**:
  ```python
  BATCH_SIZE = 16
  NUM_EPOCHS = 200
  ```

### Data Augmentation

```python
# Data augmentation
MIN_SNR = 0  # Minimum Signal-to-Noise Ratio in dB
MAX_SNR = 20  # Maximum Signal-to-Noise Ratio in dB
```

SNR (Signal-to-Noise Ratio) controls how much noise is mixed with the voice during training:

- Lower values (e.g., 0 dB): Voice and noise have equal power
- Higher values (e.g., 20 dB): Voice is much louder than noise

#### Examples:

- **Heavy noise environment** (useful if the target environment is very noisy):

  ```python
  MIN_SNR = -5  # Noise is louder than voice
  MAX_SNR = 10
  ```

- **Cleaner environment**:
  ```python
  MIN_SNR = 5
  MAX_SNR = 25
  ```

### Model Parameters

```python
# Model parameters
N_CHANNELS = 32  # Starting number of channels in U-Net
```

This controls the model capacity. Larger values create a bigger model that can capture more complex patterns but requires more memory and computation.

#### Examples:

- **Smaller model** (less memory, faster):

  ```python
  N_CHANNELS = 16
  ```

- **Larger model** (more memory, potentially better results):
  ```python
  N_CHANNELS = 64
  ```

## Window Size Options

The system now supports training and using models with different time-frequency resolutions through configurable window sizes. Each window size has its own advantages and use cases:

### Available Window Sizes

- **SMALL (30ms)**: Good for capturing detailed voice characteristics and transients

  - Higher time resolution
  - Good for removing quick noises
  - Best for general voice isolation

- **MEDIUM (100ms)**: Balanced approach

  - Moderate time and frequency resolution
  - Better voice/timbre preservation than SMALL
  - Still responsive to changes

- **LARGE (500ms)**: Better for distinguishing similar voices

  - Higher frequency resolution
  - Better maintains voice characteristics
  - Good for separating voice from music or similar voices

- **XLARGE (2000ms)**: Captures more long-term voice characteristics
  - Best frequency resolution
  - Good for speaker identification aspects
  - May introduce artifacts for rapid voice changes

### Training with Different Window Sizes

You can train a model with a specific window size:

```bash
python train.py --window-size MEDIUM
```

Or train all window size models sequentially:

```bash
python train.py --all
```

The trained models will be saved with their window size in the filename: `voice_isolation_ai_<size>.pth`

### Processing with Different Window Sizes

To process with a specific window size model:

```bash
python process.py --input noisy.mp3 --output clean.wav --window-size LARGE
```

## Data Preparation

Before training, you'll want to preprocess your audio data to remove silence and optionally generate sample mixtures to verify the quality of your dataset.

### Preprocessing Steps

The system includes a dedicated preprocessing script that:

1. Removes silence from audio files (can be disabled)
2. Normalizes audio formats and sample rates
3. Can generate sample mixtures with various SNR levels for testing

### Running Preprocessing

```bash
# Basic preprocessing (removes silence)
python prepare_data.py

# Keep silence in recordings
python prepare_data.py --keep-silence

# Generate sample mixtures for listening
python prepare_data.py --generate-samples

# Both keep silence and generate samples
python prepare_data.py --keep-silence --generate-samples
```

This creates:

- `PREPROCESSED_VOICE`: Contains processed voice files with silence removed
- `PREPROCESSED_NOISE`: Contains processed noise files with silence removed
- `SAMPLES`: Contains sample mixtures of voice and noise at different SNR levels (if --generate-samples is used)

### Silence Removal

The silence removal algorithm:

- Identifies segments with RMS energy below a threshold (-40dB by default)
- Removes segments of silence longer than a minimum duration (0.3s by default)
- Concatenates the remaining audio segments

This significantly improves the quality of training data by ensuring the model only learns from relevant audio content.

## Training Guide

### Prepare Your Data

1. **VOICE folder**: Add recordings of only the target person's voice

   - Recommended: At least 10 minutes of clean voice recordings
   - Multiple files of different lengths are fine
   - Supported formats: mp3, wav, webm

2. **NOISE folder**: Add recordings of:

   - Other people talking (especially those who typically appear in recordings with the target person)
   - Ambient noise from the environments where recordings typically happen
   - Any other noise sources that need to be removed
   - Supported formats: mp3, wav, webm

3. **Preprocess the data**:
   ```bash
   python prepare_data.py
   ```
   This creates the preprocessed folders that will be used for training.

### Start Training

Run the training script:

```bash
# Train with default window size (SMALL - 30ms)
python train.py

# Train with specific window size
python train.py --window-size MEDIUM

# Train all window size variants
python train.py --all

# Only prepare the data without training
python train.py --prepare-only
```

The script will:

1. Check if preprocessed data exists, and create it if needed
2. Mix voice and noise files with random SNRs to create training examples
3. Train the U-Net model to generate masks
4. Save checkpoints every 10 epochs in the OUTPUT directory
5. Save the final model as `voice_isolation_ai_{window_size}.pth`

### Training Tips

- Start with the default configuration and adjust if needed
- Training might take several hours depending on your hardware
- Check the loss values during training - they should decrease over time
- If the loss plateaus, try adjusting the learning rate or increasing the model capacity

## Processing Guide

### Basic Usage

```bash
python process.py --input noisy_audio.mp3 --output clean_voice.wav
```

### Advanced Usage

You can specify a different model file:

```bash
python process.py --input noisy_audio.mp3 --output clean_voice.wav --model OUTPUT/model_checkpoint_epoch_50.pth
```

The `--model` parameter takes precedence over the `--window-size` parameter when both are provided. Models are saved with the following naming convention:

- Final models: `voice_isolation_ai_{window_size}.pth` (e.g., `voice_isolation_ai_30.pth` for SMALL)
- Checkpoints: `voice_isolation_ai_{window_size}_checkpoint_epoch_{epoch}.pth`

### Processing Steps

The processing script will:

1. Load the audio file
2. Convert it to a spectrogram using STFT
3. Apply the trained model to generate a mask
4. Apply the mask to isolate the target voice
5. Convert back to the time domain and save the result

## Troubleshooting

### Common Issues

1. **Memory errors during training**:

   - Decrease BATCH_SIZE
   - Decrease N_CHANNELS
   - Use shorter audio segments by modifying the segment_length in dataset.py

2. **Poor isolation quality**:

   - Ensure you have enough diverse training data
   - Try increasing NUM_EPOCHS
   - Experiment with different WINDOW_SIZE_MS values
   - Try different SNR ranges

3. **Audio format errors**:

   - Ensure all audio files are in supported formats (mp3, wav, webm)
   - Convert any unsupported formats using tools like ffmpeg

4. **Model not improving during training**:
   - Try adjusting LEARNING_RATE
   - Ensure your VOICE folder contains only the target voice
   - Add more diverse noise samples to the NOISE folder

### Advanced Customization

For advanced users who want to modify the model architecture or training process:

- `model.py`: Contains the U-Net architecture. You can modify it to experiment with different architectures.
- `dataset.py`: Contains the dataset logic. You can modify it to change how training examples are generated.
- `utils.py`: Contains audio processing functions. You can modify these to change how audio is processed.
- `prepare_data.py`: Contains the preprocessing logic. You can modify it to change how silence is detected and removed.

## Performance Expectations

- The system works best when the model is trained on data similar to what it will process
- It may struggle with completely new environments or noise types
- For best results, include examples of all expected noise types in your NOISE folder
