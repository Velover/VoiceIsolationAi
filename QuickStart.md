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
# Basic training with default parameters (auto-detects optimal sample count)
python main.py train

# Advanced training with custom parameters
python main.py train --window-size medium --epochs 50 --batch-size 64

# Maximum GPU utilization with auto-detection (recommended)
python main.py train --gpu --mixed-precision --deep-model

# Custom configuration with all options
python main.py train --gpu --mixed-precision --deep-model --window-size medium --epochs 50 --batch-size 64 --auto-detect-samples
```

Training parameters:

- `--window-size`: Processing window size (`small`=30ms, `medium`=500ms, `large`=2s)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 64)
- `--samples`: Number of training samples to generate (0 for auto-detect)
- `--gpu`: Enable GPU acceleration for faster training
- `--mixed-precision`: Use FP16 for additional performance on compatible GPUs
- `--auto-detect-samples`: Automatically determine optimal sample count based on GPU memory
- `--deep-model`: Use deeper model architecture for maximizing GPU computational utilization

The trained model will be saved in the `OUTPUT` directory.

### Using the Model

```bash
# Isolate voice in an audio file
python main.py isolate --input path/to/recording.wav --model OUTPUT/voice_isolation_model.pth

# Specify custom output path
python main.py isolate --input recording.wav --model OUTPUT/model.pth --output isolated_voice.wav

# Use GPU for faster processing
python main.py isolate --input recording.wav --model OUTPUT/model.pth --gpu
```

### Optimizing GPU Performance

For best GPU utilization:

```bash
# Check GPU information and performance
python main.py gpu-info

# Run training with optimal GPU settings
python main.py train --gpu --mixed-precision --deep-model
```

The system will:

1. Auto-detect the optimal number of training samples based on your GPU memory
2. Use CUDA streams for parallel data transfer and computation
3. Utilize a deeper model with more computations for higher GPU utilization
4. Apply all performance optimizations automatically

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

- The system now automatically determines the optimal number of samples for your hardware
- Use the `--deep-model` flag to fully utilize GPU computational power
- For very noisy environments, manually increase the number of training samples
- If the model struggles with certain types of noise, add more similar noise samples to the training data

### Troubleshooting

- **Poor isolation results**: Try re-training with more diverse noise samples
- **Training errors**: Ensure audio files are properly formatted and not corrupted
- **Out of memory errors**: Let the auto-detection handle sample count, or manually reduce batch size
- **Low GPU utilization**: Use the `--deep-model` flag to increase computational workload
- **GPU-related errors**: Make sure you have the latest GPU drivers installed
- **CUDA issues**: Verify your PyTorch installation includes CUDA support

## Technical Details

The system combines:

- Audio preprocessing with configurable window sizes
- Convolutional neural network for mask generation
- Spectrogram manipulation for voice isolation
- GPU acceleration with parallel CUDA streams for maximum performance
- Automatic sample count determination based on available GPU memory
- Optional deeper model for maximizing GPU computational utilization
- Mixed precision (FP16) training for modern NVIDIA GPUs

Supported audio formats: `.mp3` and `.wav`

Hardware recommendations:

- CPU: Any modern multi-core processor
- GPU: NVIDIA RTX 3060 or better for optimal performance
- RAM: 8GB minimum, 16GB recommended
