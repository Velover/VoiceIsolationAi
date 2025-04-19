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

   - Create `VOICE` folder with target person's voice recordings (.mp3, .wav, or .flac)
   - Create `NOISE` folder with other voices and background noise (.mp3, .wav, or .flac)

   If you don't have audio files yet, you can create test examples:

   ```bash
   python main.py preprocess --create-examples
   ```

3. **Preprocess audio for faster training** (highly recommended):

   ```bash
   # Convert large WAV files to more efficient FLAC format and cache spectrograms
   python main.py preprocess --convert

   # Only cache spectrograms (if you want to keep original WAV files)
   python main.py preprocess
   ```

### Training the Model

```bash
# Train with cached data (fastest method, recommended)
python main.py train --use-cache --gpu --mixed-precision --deep-model

# Basic training with default parameters
python main.py train

# Advanced training with custom parameters
python main.py train --window-size medium --epochs 50 --batch-size 64

# Maximum GPU utilization without preprocessing
python main.py train --gpu --mixed-precision --deep-model
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
- `--use-cache`: Use preprocessed cached data to dramatically speed up training

The trained model will be saved in the `OUTPUT` directory.

### Troubleshooting

- **"No voice/noise files found"**: Make sure you have audio files in the VOICE and NOISE directories
  - You can create test examples with `python main.py preprocess --create-examples`
  - The VOICE directory should contain recordings of the target person
  - The NOISE directory should contain background noises and other people's voices

// ...existing content...

```

These changes will:

1. Make the `CachedSpectrogramDataset` initialization more robust by initializing thread control attributes early
2. Add checks to create empty directories if they don't exist
3. Add more helpful error messages when no audio files are found
4. Add a feature to create example audio files if directories are empty
5. Update the QuickStart guide with troubleshooting information

The main issue was that either the NOISE directory was empty or didn't exist. With these changes, you'll get clearer error messages and have the option to create test examples.
```
