# Voice Isolation AI

```
python -m pip install -r requirements.txt
```

A deep learning system for isolating a specific person's voice from background noise and other speakers.

## Quick Start Guide

### Setup

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data directories**:
   - Create `VOICE` folder with target person's voice recordings (.mp3 or .wav)
   - Create `NOISE` folder with other voices and background noise (.mp3 or .wav)

### Training the Model

```bash
# Basic training with default parameters
python main.py train

# Advanced training with custom parameters
python main.py train --window-size medium --epochs 50 --batch-size 32 --samples 2000
```

Training parameters:

- `--window-size`: Processing window size (`small`=30ms, `medium`=500ms, `large`=2s)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--samples`: Number of training samples to generate (default: 2000)

The trained model will be saved in the `OUTPUT` directory.

### Using the Model

```bash
# Isolate voice in an audio file
python main.py isolate --input path/to/recording.wav --model OUTPUT/voice_isolation_model.pth

# Specify custom output path
python main.py isolate --input recording.wav --model OUTPUT/model.pth --output isolated_voice.wav
```

## Project Overview

This system:

1. Processes audio using Short-Time Fourier Transform
2. Uses a CNN to generate isolation masks for the target voice
3. Applies the mask to isolate the specific voice
4. Outputs the cleaned audio file

## Detailed Instructions

### Data Preparation

- **VOICE folder**: Place multiple recordings of the target person speaking alone
- **NOISE folder**: Add recordings of:
  - Other people speaking
  - Background noises
  - Environmental sounds
  - Any sounds that typically interfere with the target voice

For best results:

- Use diverse recordings in different acoustic environments
- Include samples with different voice levels and tones
- Ensure recordings are clean and representative of real-world scenarios

### Training Tips

- Start with the `medium` window size for a good balance of context and detail
- For very noisy environments, increase the number of training samples
- If the model struggles with certain types of noise, add more similar noise samples to the training data

### Troubleshooting

- **Poor isolation results**: Try re-training with more diverse noise samples
- **Training errors**: Ensure audio files are properly formatted and not corrupted
- **Out of memory errors**: Reduce batch size or use a smaller window size

## Technical Details

The system combines:

- Audio preprocessing with configurable window sizes
- Convolutional neural network for mask generation
- Spectrogram manipulation for voice isolation

Supported audio formats: `.mp3` and `.wav`
