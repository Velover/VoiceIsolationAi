# Voice Isolation AI

This project trains an AI to isolate a specific person's voice from background noises and other voices in audio recordings.

## Quick Start

1. **Installation**:

   ```bash
   pip install torch torchaudio librosa numpy tqdm soundfile
   ```

2. **Prepare data**:

   - Create a `VOICE` folder with the target person's voice recordings
   - Create a `NOISE` folder with other voices and background noise
   - Preprocess the data (removes silence and prepares for training):
     ```bash
     python prepare_data.py
     ```

3. **Train a model**:

   ```bash
   # Train with default settings (30ms window)
   python train.py

   # Train with specific window size (SMALL=30ms, MEDIUM=100ms, LARGE=500ms, XLARGE=2000ms)
   python train.py --window-size MEDIUM

   # Train all window sizes
   python train.py --all
   ```

4. **Process audio**:

   ```bash
   # Process using default model (SMALL window size)
   python process.py --input noisy_audio.mp3 --output clean_voice.wav

   # Process using specific window size model
   python process.py --input noisy_audio.mp3 --output clean_voice.wav --window-size LARGE

   # Process using specific model file
   python process.py --input noisy_audio.mp3 --output clean_voice.wav --model OUTPUT/voice_isolation_ai_100.pth
   ```

## Data Preparation

Before training, you can preprocess the data to remove silence and generate sample mixtures:

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

- `PREPROCESSED_VOICE`: Processed voice files
- `PREPROCESSED_NOISE`: Processed noise files
- `SAMPLES`: Sample mixtures of voice and noise (if --generate-samples is used)

The training automatically uses these preprocessed folders if they exist.

## Training Commands

Train a model to isolate the target voice:

```bash
# Basic usage - trains with default 30ms window
python train.py

# Train with specific window size
python train.py --window-size SMALL   # 30ms window
python train.py --window-size MEDIUM  # 100ms window
python train.py --window-size LARGE   # 500ms window
python train.py --window-size XLARGE  # 2000ms window

# Train models for all window sizes sequentially
python train.py --all

# Use pre-generated dataset for faster training
python train.py --pregenerate

# Only generate the dataset without training
python train.py --generate-only --num-examples 2000

# Generate datasets for all window sizes
python train.py --generate-only --all --num-examples 1000
```

Training will save models to the OUTPUT directory with names like `voice_isolation_ai_30.pth` for a 30ms window model.

## Pre-generated Datasets

To speed up training, you can pre-generate datasets:

```bash
# Generate 1000 examples (default) for SMALL window size model
python train.py --generate-only

# Generate 2000 examples for MEDIUM window size model
python train.py --generate-only --window-size MEDIUM --num-examples 2000

# Generate for all window sizes
python train.py --generate-only --all --num-examples 1000
```

Then train using the pre-generated datasets:

```bash
# Train using pre-generated dataset
python train.py --pregenerate

# Train all window sizes using pre-generated datasets
python train.py --all --pregenerate
```

This approach significantly speeds up training as it avoids dataset generation between epochs.

## Processing Commands

Process audio files to isolate the target voice:

```bash
# Basic usage with default model (30ms window)
python process.py --input path/to/noisy_audio.mp3 --output path/to/clean_voice.wav

# Use a specific window size model
python process.py --input input.wav --output output.wav --window-size MEDIUM

# Use a specific model file (useful for checkpoints or custom models)
python process.py --input input.wav --output output.wav --model OUTPUT/voice_isolation_ai_500.pth

# Full example with explicit paths
python process.py --input "recordings/noisy_conversation.mp3" --output "clean/isolated_voice.wav" --window-size LARGE
```

## Project Structure

- `config.py`: Configuration settings for audio processing and training
- `utils.py`: Utility functions for audio processing
- `dataset.py`: PyTorch Dataset for training data preparation
- `model.py`: U-Net based AI model for voice isolation
- `train.py`: Script for training the model
- `process.py`: Script for processing audio files with the trained model

## Data Organization

The project uses the following folder structure:

- `VOICE`: Contains original audio files of the target person's voice
- `NOISE`: Contains original audio files of other people's voices and background noise
- `PREPROCESSED_VOICE`: Contains processed voice files with silence removed
- `PREPROCESSED_NOISE`: Contains processed noise files with silence removed
- `VOICE_FOR_TRAINING`: Contains files that will be used for training (manually selected)
- `NOISE_FOR_TRAINING`: Contains files that will be used for training (manually selected)
- `SAMPLES`: Contains generated sample mixtures for testing
- `OUTPUT`: Contains trained models and checkpoints

### Training Workflow

1. Add raw audio files to `VOICE` and `NOISE` folders
2. Run preprocessing: `python prepare_data.py`
3. Manually copy files from `PREPROCESSED_VOICE` to `VOICE_FOR_TRAINING` and from `PREPROCESSED_NOISE` to `NOISE_FOR_TRAINING`
   - This allows you to select which files to use for training
   - You can also add your own custom files directly to these folders
4. Run training: `python train.py`

This workflow gives you more control over which files are used for training, allowing for more experimentation and targeted training sets.

Supported audio formats: mp3, wav, webm (all sample rates are automatically converted)

## Window Size Options

Different window sizes offer different tradeoffs:

- **SMALL (30ms)**: Good for detailed voice characteristics, best for general use
- **MEDIUM (100ms)**: Balanced approach with better timbre preservation
- **LARGE (500ms)**: Better for distinguishing similar voices
- **XLARGE (2000ms)**: Best for capturing long-term voice characteristics

## Technical Details

The system works by:

1. Converting audio to the frequency domain using Short-Time Fourier Transform (STFT)
2. Feeding the spectrogram to a U-Net neural network that generates a mask
3. Applying the mask to isolate the target voice
4. Converting back to the time domain using inverse STFT

For more detailed documentation, see `DOCUMENTATION.md`.
