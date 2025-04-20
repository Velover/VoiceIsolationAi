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

2. **Generate training samples from cached spectrograms** (fastest method):

   ```bash
   # Generate 1000 samples using cached spectrograms (much faster)
   python main.py generate-samples --samples 1000 --use-cache

   # Specify custom cache directories if needed
   python main.py generate-samples --use-cache --voice-cache-dir CACHE/voice --noise-cache-dir CACHE/noise
   ```

3. **Or generate samples directly from audio files**:

   ```bash
   # Generate 1000 training samples using 8 CPU workers
   python main.py generate-samples --samples 1000 --gpu --workers 8

   # For maximum speed on multicore systems
   python main.py generate-samples --samples 5000 --gpu --workers 12

   # For lower memory systems
   python main.py generate-samples --samples 2000 --workers 4
   ```

   > **New Feature**: The system now selects random segments from longer audio files during training, providing more diverse training data and better model performance.

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
4. Select random segments from longer files for more realistic tests

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

# Process with debug mode to create additional output files for analysis
python main.py isolate --input TEST/MIXED/mixed_voice1_noise1_snr5.0_1.wav --model OUTPUT/your_model.pth --debug
```

The `--debug` flag generates additional files:

- A pure isolated version without any mixing
- A 70/30 mix of isolated and original audio
- The original audio for comparison
- A visualization of the isolation mask values

Or batch process all test files at once:

```bash
# Process all files in TEST/MIXED directory
python main.py isolate-batch --model OUTPUT/your_model.pth

# Process with debug files for detailed analysis
python main.py isolate-batch --model OUTPUT/your_model.pth --debug

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

- **Device mismatch errors**: If you see errors like "Expected all tensors to be on the same device":

  - This is usually caused by CUDA/CPU device mismatch during processing
  - Try updating to the latest version of the code which includes fixes for this issue
  - As a workaround, you can process without GPU: `python main.py isolate --input YOUR_FILE.wav --model YOUR_MODEL.pth --no-gpu`
  - For batch processing: `python main.py isolate-batch --model YOUR_MODEL.pth --no-gpu`

- **Silent or very quiet output**: If the isolated output is silent or too quiet:

  - Make sure your model was trained on similar audio conditions
  - Use the `--debug` flag to see the mask behavior
  - Try processing with `python main.py isolate --input YOUR_FILE.wav --model YOUR_MODEL.pth --debug`
  - Check the generated mask visualization to see if the model is recognizing voice segments
  - If the masks are consistently low (below 0.2), your model needs retraining with better voice samples

- **Poor voice isolation quality**: Try these approaches:
  - Ensure your VOICE directory contains clear samples of only the target voice
  - Use the `--debug` flag when processing to analyze mask behavior
  - Try a different window size (small windows work better for quick speech, large for steady speech)

## Training Performance Tips

1. **For fastest training**:

   - First run: `python main.py preprocess --convert`
   - Then: `python main.py generate-samples --samples 3000 --use-cache`
   - Finally: `python main.py train --preprocessed --gpu --mixed-precision --deep-model`

2. **If sample generation is slow**:

   - Use the `--use-cache` option with generate-samples
   - If not using cache, increase worker count with `--workers` (use fewer than your CPU core count)
   - For high-end GPUs, use them for preprocessing with `--gpu`
   - For low-memory systems, reduce worker count and generate fewer samples

3. **For best isolation quality**:
   - The default configuration uses enhanced FFT parameters (N_FFT=2048, HOP_LENGTH=512)
   - These provide better frequency resolution and voice separation
   - The new adaptive window processing adjusts overlap based on window size for optimal results
   - Frequency-enhanced mask application improves voice isolation in the typical voice frequency range
   - If isolation quality is poor, try the `--debug` flag to generate diagnostic files
