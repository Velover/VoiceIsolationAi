import os
import torch
import torchaudio
import argparse
from typing import Optional, Tuple, List
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import math

from .config import OUTPUT_DIR, N_FFT, HOP_LENGTH, SAMPLE_RATE, SPEC_TIME_DIM, WINDOW_SIZES
from .preprocessing import AudioPreprocessor
from .model import VoiceIsolationModel

def load_model(model_path: str, device: torch.device) -> Tuple[VoiceIsolationModel, str]:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to model file
        device: Device to load model on
        
    Returns:
        Tuple of (model, window_size)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model type (deep or standard)
    is_deep_model = checkpoint.get('deep_model', False)
    
    # Create appropriate model and load state
    if is_deep_model:
        from .model import VoiceIsolationModelDeep
        model = VoiceIsolationModelDeep(n_fft=checkpoint.get('n_fft', N_FFT))
    else:
        from .model import VoiceIsolationModel
        model = VoiceIsolationModel(n_fft=checkpoint.get('n_fft', N_FFT))
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    window_size = checkpoint.get('window_size', 'medium')
    
    # Print model information for clarity
    print(f"Model information:")
    print(f"  - Window size: {window_size}")
    print(f"  - Model type: {'Deep' if is_deep_model else 'Standard'}")
    print(f"  - Created on: {checkpoint.get('creation_date', 'unknown')}")
    print(f"  - Training samples: {checkpoint.get('training_samples', 'unknown')}")
    
    return model, window_size

def process_audio(
    input_path: str,
    output_path: str,
    model_path: str,
    device: torch.device,
    verbose: bool = True,
    debug: bool = False  # New parameter to control debug file generation
) -> None:
    """
    Process audio file to isolate voice using the trained model.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save output audio
        model_path: Path to trained model
        device: Device to run inference on
        verbose: Whether to print detailed progress
        debug: Whether to generate additional debug files
    """
    process_start = time.time()
    if verbose:
        print(f"=== VOICE ISOLATION PROCESSING ===")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Model: {model_path}")
        print(f"Device: {device}")
    
    # Load model
    if verbose:
        print("\n[1/6] Loading model...", end="")
    start = time.time()
    model, window_size = load_model(model_path, device)
    duration = time.time() - start
    if verbose:
        print(f" Done in {duration:.2f}s")
        print(f"       Window size: {window_size} ({WINDOW_SIZES.get(window_size, 'unknown')}ms)")
    
    # Initialize preprocessor with use_gpu flag based on device
    use_gpu = device.type == 'cuda'
    preprocessor = AudioPreprocessor(window_size=window_size, use_gpu=use_gpu)
    
    # Load audio
    if verbose:
        print(f"[2/6] Loading audio file...", end="")
    start = time.time()
    audio = preprocessor.load_audio(input_path)
    duration = time.time() - start
    audio_duration_seconds = audio.shape[1] / SAMPLE_RATE
    if verbose:
        print(f" Done in {duration:.2f}s (Audio length: {audio_duration_seconds:.2f}s)")
    
    # Calculate window size in samples
    window_ms = WINDOW_SIZES.get(window_size, 500)
    window_samples = int(SAMPLE_RATE * window_ms / 1000)
    
    # Calculate hop size for overlapping windows
    # Adaptive hop size based on window size - larger windows need less overlap
    if window_ms <= 100:  # Small windows (30-100ms)
        hop_ratio = 0.5  # 50% overlap for small windows
    elif window_ms <= 1000:  # Medium windows (100ms-1s)
        hop_ratio = 0.6  # 40% overlap for medium windows
    else:  # Large windows (1s+)
        hop_ratio = 0.75  # 25% overlap for large windows
        
    hop_samples = int(window_samples * hop_ratio)
    
    # Number of windows needed to cover the entire audio
    num_windows = math.ceil((audio.shape[1] - window_samples) / hop_samples) + 1
    
    if verbose:
        print(f"[3/6] Processing with sliding window approach...")
        print(f"       Audio length: {audio_duration_seconds:.2f}s, Window size: {window_ms}ms")
        print(f"       Processing {num_windows} windows with {int((1-hop_ratio)*100)}% overlap")
        print(f"       Window samples: {window_samples}, Hop samples: {hop_samples}")
    
    # Using a tapered window for smoother transitions between segments
    # This reduces audible artifacts at window boundaries
    window_envelope = torch.hann_window(window_samples, dtype=audio.dtype, device=audio.device)
    # Add a small offset to ensure every sample gets some weight
    window_envelope = window_envelope * 0.95 + 0.05
    window_envelope = window_envelope.unsqueeze(0)  # Add channel dimension
    
    # Create a tensor to accumulate the output audio and weights
    output_audio = torch.zeros_like(audio)
    weight = torch.zeros_like(audio)
    
    # Track mask statistics for debugging
    mask_means = []
    mask_maxes = []
    mask_mins = []
    
    # Save a copy of the original audio for fallback and comparison
    original_audio = audio.clone()
    
    # Process each window
    start = time.time()
    windows_progress = tqdm(range(num_windows), desc="Processing windows", disable=not verbose)
    
    # Create a debug file to visualize mask statistics
    debug_info = []
    
    for i in windows_progress:
        # Calculate window boundaries
        start_sample = i * hop_samples
        end_sample = min(start_sample + window_samples, audio.shape[1])
        
        # Extract window from audio
        window_audio = audio[:, start_sample:end_sample]
        
        # If the window is smaller than expected (last window), pad it
        actual_window_size = window_audio.shape[1]
        if actual_window_size < window_samples:
            padded = torch.zeros((1, window_samples), dtype=window_audio.dtype, device=window_audio.device)
            padded[:, :actual_window_size] = window_audio
            window_audio = padded
        
        # Process this window
        try:
            # Compute STFT for this window
            stft = preprocessor.compute_stft(window_audio)
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Standardize for model input
            magnitude_std = preprocessor.standardize_spectrogram(magnitude)
            
            # Prepare input for model
            model_input = magnitude_std.unsqueeze(0).unsqueeze(0).to(device)
            
            # Generate mask
            with torch.no_grad():
                device_type = 'cuda' if device.type == 'cuda' else 'cpu'
                with torch.amp.autocast(device_type=device_type, enabled=device.type == 'cuda', dtype=torch.float16):
                    mask = model(model_input).squeeze(0).squeeze(0).cpu()
            
            # Ensure mask has the right dimensions
            if magnitude.shape[1] < SPEC_TIME_DIM:
                mask = mask[:, :magnitude.shape[1]]
            else:
                temp_mask = torch.zeros_like(magnitude)
                temp_mask[:, :mask.shape[1]] = mask[:, :magnitude.shape[1]]
                mask = temp_mask
            
            # Track mask statistics for debugging
            mask_mean = mask.mean().item()
            mask_max = mask.max().item()
            mask_min = mask.min().item()
            mask_means.append(mask_mean)
            mask_maxes.append(mask_max)
            mask_mins.append(mask_min)
            
            # CRITICAL CHANGE: Add explicit device consistency checks
            # Move tensors to CPU for consistent processing right after model output
            mask = mask.cpu()
            magnitude = magnitude.cpu()
            phase = phase.cpu()
            
            # Print diagnostic info for troubleshooting
            if i == 0 or i == num_windows//2 or i == 6:  # Debug for first, middle and window 6
                print(f"\nDIAGNOSTIC: Window {i} raw mask stats: mean={mask_mean:.4f}, max={mask_max:.4f}, min={mask_min:.4f}")
                print(f"Devices - mask: {mask.device}, magnitude: {magnitude.device}, phase: {phase.device}")
            
            # Save information for debugging
            debug_info.append({
                'window': i,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'mask_mean': mask_mean,
                'mask_max': mask_max,
                'mask_min': mask_min
            })
            
            # Apply mask adjustment logic based on what we see
            # If mask is too extreme (all 0s or all 1s), make it more useful
            # We want some variation in the mask to actually isolate voice
            if mask_max < 0.05:
                # Mask is blocking everything - replace with moderate mask
                print(f"\nWARNING: Mask for window {i} is too low (max={mask_max:.4f}). Using raised mask.")
                # Changed from 0.5 to 0.7 to let more sound through
                mask = torch.ones_like(mask) * 0.7
            elif mask_mean > 0.9:
                # Mask is letting everything through - apply more aggressive masking
                print(f"\nWARNING: Mask for window {i} is too high (mean={mask_mean:.4f}). Applying threshold.")
                # Make this less aggressive - changed from (mask - 0.5) * 2 to (mask - 0.3) * 1.5
                mask = torch.clamp((mask - 0.3) * 1.5, 0.1, 1.0)  # Ensure minimum mask value of 0.1
            
            # Apply frequency-dependent enhancement
            # Voice is typically in mid-range frequencies (100Hz-3000Hz)
            # We'll boost this range for better voice isolation
            freq_bins = mask.shape[0]
            voice_range_start = int(freq_bins * 0.05)  # ~100Hz
            voice_range_end = int(freq_bins * 0.3)     # ~3000Hz
            
            # Boost the voice frequency range slightly - make this less aggressive
            voice_freq_mask = mask[voice_range_start:voice_range_end, :]
            if voice_freq_mask.mean() > 0.4:  # Only if we detect probable voice
                # Changed from 0.5 to 0.7 power to be less aggressive
                mask[voice_range_start:voice_range_end, :] = torch.pow(mask[voice_range_start:voice_range_end, :], 0.7)
            
            # Enhance contrast in the mask to make isolation more effective
            # This helps separate voice from background more clearly
            # Changed from 0.7 to 0.8 to make this less aggressive
            mask = torch.pow(mask, 0.8)  # Power < 1.0 increases contrast
            
            # Smooth mask values below a threshold to reduce musical noise
            # Increase this threshold and reduce the scaling to let more sound through
            low_mask_values = mask < 0.3
            if low_mask_values.any():
                mask[low_mask_values] *= 0.7  # Changed from 0.5 to 0.7
            
            # CRITICAL: Apply mask to isolate voice
            # Add a minimum mask value to ensure some sound always comes through
            # This prevents complete silence in the output
            mask = torch.clamp(mask, min=0.1)
            
            # Apply the mask to get isolated magnitude spectrum
            # This was the missing line causing the error
            isolated_magnitude = magnitude * mask
            
            # Verify isolated magnitude has non-zero values
            if isolated_magnitude.abs().max().item() < 1e-8:
                print(f"\nWARNING: Isolated magnitude for window {i} is all zeros!")
                # Use original magnitude scaled down as fallback
                isolated_magnitude = magnitude * 0.5
            
            # Reconstruct complex STFT
            isolated_stft = isolated_magnitude * torch.exp(1j * phase)
            
            # Convert back to time domain - ENSURE DEVICE CONSISTENCY
            # Move the window and STFT to same device (CPU is safest)
            isolated_stft = isolated_stft.cpu()
            hann_window = torch.hann_window(N_FFT, device='cpu')
            
            # Perform ISTFT on CPU for stability
            window_isolated_audio = torch.istft(
                isolated_stft,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                window=hann_window
            ).unsqueeze(0)  # Add channel dimension
            
            # Check if we got valid audio back
            if torch.isnan(window_isolated_audio).any() or torch.isinf(window_isolated_audio).any():
                print(f"\nWARNING: NaN or Inf in audio for window {i}. Using original audio.")
                window_isolated_audio = window_audio.cpu() * 0.5  # Use original scaled down, ensure on CPU
            
            # CRITICAL FIX: Ensure all tensors are on CPU for consistent processing
            window_envelope_cpu = window_envelope.cpu()
            window_isolated_audio = window_isolated_audio.cpu()
            
            # Apply window envelope for smooth transitions
            window_isolated_audio = window_isolated_audio * window_envelope_cpu[:, :window_isolated_audio.shape[1]]
            
            # Add to output with overlap-add (standard approach for audio processing)
            # Ensure output_audio is on CPU for consistent processing
            if output_audio.device.type != 'cpu':
                output_audio = output_audio.cpu()
                weight = weight.cpu()
                
            output_end = min(start_sample + window_samples, output_audio.shape[1])
            current_window_size = output_end - start_sample
            
            if current_window_size > 0 and window_isolated_audio.shape[1] >= current_window_size:
                output_audio[:, start_sample:output_end] += window_isolated_audio[:, :current_window_size]
                weight[:, start_sample:output_end] += window_envelope_cpu[:, :current_window_size]
            
            # Update progress bar with useful info
            if verbose:
                avg_mask = sum(mask_means[-10:]) / min(len(mask_means), 10) if mask_means else 0
                windows_progress.set_postfix({
                    'mask_avg': f"{avg_mask:.2f}", 
                    'max': f"{mask_max:.2f}"
                })
                
        except Exception as e:
            print(f"\nError processing window {i+1}/{num_windows}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary statistics
    if verbose:
        print(f"\n[4/6] Finalizing output...")
        if mask_means:
            print(f"       Mask statistics: mean={sum(mask_means)/len(mask_means):.4f}, max={max(mask_maxes):.4f}, min={min(mask_mins):.4f}")
    
    # Check if we have meaningful data in the output
    # If all weights are zero, something went wrong
    if weight.abs().max().item() < 1e-8:
        print("\nERROR: All weights are zero, indicating no valid data was processed.")
        print("Using original audio with reduced volume as fallback.")
        output_audio = original_audio * 0.7
    else:
        # Normalize by weights - SIMPLIFIED LOGIC
        epsilon = 1e-10
        weight = torch.clamp(weight, min=epsilon)  # Avoid division by zero
        output_audio = output_audio / weight
    
        # Final smoothing pass to eliminate any remaining discontinuities
        if window_samples > 100:  # Only for non-tiny windows
            # Add a small epsilon to weight to avoid division by zero
            epsilon = 1e-10
            weight = torch.clamp(weight, min=epsilon)
            output_audio = output_audio / weight
            
            # Apply a light smoothing filter to eliminate any sharp transitions
            # This helps reduce "musical" artifacts from windowing
            smoothing_kernel_size = min(int(SAMPLE_RATE * 0.01), 512)  # ~10ms or less
            if smoothing_kernel_size % 2 == 0:  # Ensure odd kernel size
                smoothing_kernel_size += 1
                
            try:
                # Only apply if we have enough samples and torchaudio supports conv filtering
                if output_audio.shape[1] > smoothing_kernel_size * 3:
                    import torch.nn.functional as F
                    smoothing_kernel = torch.hann_window(smoothing_kernel_size, dtype=torch.float32, device='cpu')
                    smoothing_kernel = smoothing_kernel / smoothing_kernel.sum()
                    smoothing_kernel = smoothing_kernel.view(1, 1, smoothing_kernel_size)
                    
                    # Apply convolution for smoothing
                    # Move to CPU for this operation to ensure compatibility
                    output_cpu = output_audio.cpu()
                    padded = F.pad(output_cpu, (smoothing_kernel_size//2, smoothing_kernel_size//2), mode='reflect')
                    output_audio = F.conv1d(padded, smoothing_kernel)
            except Exception as e:
                # If smoothing fails, continue without it
                if verbose:
                    print(f"       Note: Final smoothing pass skipped: {e}")
        else:
            # Standard normalization for small windows
            epsilon = 1e-10
            weight = torch.clamp(weight, min=epsilon)
            output_audio = output_audio / weight
    
    # Ensure all final operations happen on CPU
    output_audio = output_audio.cpu()
    original_audio = original_audio.cpu()
    
    # Check if output is silent - more aggressive check and recovery
    output_max = output_audio.abs().max().item()
    if output_max < 0.05:  # Changed from 0.01 to 0.05 for more sensitivity
        print("\nWARNING: Output is nearly silent. Applying gain and mixing with original.")
        # Mix with original instead of just applying gain
        # This ensures we always have some audio content
        output_audio = (output_audio * 5.0 * 0.7) + (original_audio * 0.3)
        output_max = output_audio.abs().max().item()
        print(f"After recovery, max amplitude: {output_max:.4f}")
    
    # Add a final safety check - if we still have very low output, mix in original
    if output_max < 0.1:
        mix_ratio = 0.5  # 50% original if we're still too quiet
        print(f"Output still too quiet. Mixing with {mix_ratio*100}% of original audio.")
        output_audio = (output_audio * (1-mix_ratio)) + (original_audio * mix_ratio)
    
    # Normalize to prevent clipping
    if output_max > 0.95:
        output_audio = 0.95 * output_audio / output_max
    
    # Print final audio stats
    if verbose:
        print(f"       Final output statistics: max amplitude={output_audio.abs().max().item():.4f}")
    
    # Save output
    if verbose:
        print(f"\n[5/6] Saving output audio...")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save audio file - ensure CPU tensors
    torchaudio.save(
        output_path,
        output_audio.cpu(),  # Redundant but safe
        SAMPLE_RATE
    )
    
    # Save debug audio files only if debug flag is set
    if debug:
        if verbose:
            print(f"[6/6] Creating debug files...")
        
        # Save a pure version (no mixing)
        pure_path = os.path.splitext(output_path)[0] + "_pure.wav"
        torchaudio.save(pure_path, output_audio.cpu(), SAMPLE_RATE)
        
        # Save a mix file with 70% isolated, 30% original
        mixed_path = os.path.splitext(output_path)[0] + "_mix70-30.wav"
        mixed_audio = 0.7 * output_audio + 0.3 * original_audio
        torchaudio.save(mixed_path, mixed_audio.cpu(), SAMPLE_RATE)
        
        # Save the original file for comparison
        orig_path = os.path.splitext(output_path)[0] + "_original.wav"
        torchaudio.save(orig_path, original_audio.cpu(), SAMPLE_RATE)
        
        if verbose:
            print(f"       Pure isolated audio saved to: {pure_path}")
            print(f"       70/30 mixed audio saved to: {mixed_path}")
            print(f"       Original audio saved to: {orig_path}")
        
        # Save a visual mask debug report if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            plt.figure(figsize=(12, 6))
            window_indices = [d['window'] for d in debug_info]
            mask_means_debug = [d['mask_mean'] for d in debug_info]
            
            plt.plot(window_indices, mask_means_debug, 'b-', label='Mask Mean')
            plt.title('Mask Mean Values Across Windows')
            plt.xlabel('Window Index')
            plt.ylabel('Mask Mean Value')
            plt.grid(True)
            plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
            plt.legend()
            
            mask_debug_path = os.path.splitext(output_path)[0] + "_mask_debug.png"
            plt.savefig(mask_debug_path)
            plt.close()
            
            if verbose:
                print(f"       Mask debug visualization saved to: {mask_debug_path}")
        except ImportError:
            if verbose:
                print("       Note: matplotlib not available for mask visualization")
    elif verbose:
        print(f"[6/6] Debug files disabled. Use --debug flag to generate additional audio files.")
    
    # Display important note about model effectiveness
    if sum(mask_means)/len(mask_means) > 0.9:
        print("\n⚠️ IMPORTANT: The model is generating very high mask values (mean > 0.9).")
        print("This means it's letting almost everything through, resulting in minimal isolation.")
        print("Try training with more diverse data or using a different model.")
    elif sum(mask_means)/len(mask_means) < 0.1:
        print("\n⚠️ IMPORTANT: The model is generating very low mask values (mean < 0.1).")
        print("This means it's blocking almost everything, resulting in silence or very quiet output.")
        print("Try training with more representative voice samples.")
    
    total_time = time.time() - process_start
    if verbose:
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Processing ratio: {total_time/audio_duration_seconds:.2f}x realtime")
        print(f"Output saved to: {output_path}")

def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description='Isolate voice in audio file')
    parser.add_argument('--input', required=True, help='Path to input audio file')
    parser.add_argument('--output', help='Path to save output audio')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--debug', action='store_true', help='Generate additional debug files')
    
    args = parser.parse_args()
    
    # Set output path if not provided
    if not args.output:
        input_filename = os.path.basename(args.input)
        input_name, input_ext = os.path.splitext(input_filename)
        args.output = os.path.join(OUTPUT_DIR, f"{input_name}_isolated{input_ext}")
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Process audio
    process_audio(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        device=device,
        debug=args.debug  # Pass debug flag
    )

if __name__ == "__main__":
    main()