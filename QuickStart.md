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

### Preprocessing for Faster Training

For maximum training speed, preprocess audio data:

1. **Convert WAV files and create cached spectrograms**:

   ```bash
   python main.py preprocess --convert
   ```

2. **Generate training samples** (do this once, saves hours of training time):

   ```bash
   # Generate 1000 training samples using 8 CPU workers
   python main.py generate-samples --samples 1000 --gpu --workers 8

   # For maximum speed on multicore systems
   python main.py generate-samples --samples 5000 --gpu --workers 12

   # For lower memory systems
   python main.py generate-samples --samples 2000 --workers 4
   ```

### Training the Model

```bash
# FASTEST: Train with pre-generated samples (recommended after running generate-samples)
python main.py train --preprocessed --gpu --mixed-precision --deep-model

# Alternative: Train with cached spectrograms (no pre-generated samples)
python main.py train --use-cache --gpu --mixed-precision --deep-model

# Basic training with default parameters (slowest)
python main.py train
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
- `--preprocessed`: Use pre-generated samples for maximum training speed (requires running generate-samples first)

The trained model will be saved in the `OUTPUT` directory.

### Testing with Mixed Audio Files

Generate test audio files by mixing voice and noise from your dataset:

```bash
# Generate 5 test files by mixing voice and noise (20 seconds each)
python main.py generate-test

# Generate 10 test files with custom settings
python main.py generate-test --num-files 10 --duration 15 --min-snr 0 --max-snr 15
```

These commands will:

1. Create a TEST directory with subdirectories VOICE, NOISE, and MIXED
2. Generate mixed audio files with controlled SNR levels
3. Save the clean voice files for comparison

You can then use these files to test your model:

```bash
# Process a test file with your trained model
python main.py isolate --input TEST/MIXED/mixed_voice1_noise1_snr5.0_1.wav --model OUTPUT/your_model.pth
```

### Using Your Trained Model

You can process individual files:

```bash
# Process a single file
python main.py isolate --input TEST/MIXED/mixed_voice1_noise1_snr5.0_1.wav --model OUTPUT/your_model.pth
```

Or batch process all test files at once:

```bash
# Process all files in TEST/MIXED directory
python main.py isolate-batch --model OUTPUT/your_model.pth

# Process with multiple parallel workers (faster on multi-core systems)
python main.py isolate-batch --model OUTPUT/your_model.pth --workers 4

# Process files from a different directory
python main.py isolate-batch --model OUTPUT/your_model.pth --input-dir MY_FILES --output-dir RESULTS
```

This will:

1. Process all audio files in TEST/MIXED
2. Save the isolated voice files to TEST/ISOLATED
3. Show progress and timing information

### Model Management

List all available models in the `OUTPUT` directory:

```bash
python main.py list-models
```

This will display all trained models with their names and metadata.

### Troubleshooting

- **"No voice/noise files found"**: Make sure you have audio files in the VOICE and NOISE directories
  - You can create test examples with `python main.py preprocess --create-examples`
  - The VOICE directory should contain recordings of the target person
  - The NOISE directory should contain background noises and other people's voices

## Training Performance Tips

1. **For fastest training**:

   - First run: `python main.py generate-samples --samples 3000 --gpu --workers 8`
   - Then: `python main.py train --preprocessed --gpu --mixed-precision --deep-model`

2. **If sample generation is slow**:

   - Increase worker count with `--workers` (use fewer than your CPU core count)
   - For high-end GPUs, use them for preprocessing with `--gpu`
   - For low-memory systems, reduce worker count and generate fewer samples

3. **If training is still slow**:
   - Try reducing the number of samples using `--samples`
   - Ensure you're using a GPU with `--gpu`
   - Make sure you've run `generate-samples` first
