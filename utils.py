import os
import numpy as np
import torch
import torchaudio
import librosa
from config import *

def load_audio(file_path, target_sr=SAMPLE_RATE):
    """Load audio file and convert to target sample rate."""
    # Get file extension
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.mp3', '.wav']:
        # Use torchaudio for mp3 and wav
        waveform, sr = torchaudio.load(file_path)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
    elif ext == '.webm':
        # Use librosa for webm
        waveform, sr = librosa.load(file_path, sr=None, mono=True)
        waveform = torch.tensor(waveform).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Resample if needed
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    
    return waveform.squeeze(0).numpy()

def stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WINDOW_SIZE):
    """STFT function."""
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def istft(stft_matrix, hop_length=HOP_LENGTH, win_length=WINDOW_SIZE):
    """Inverse STFT function."""
    return librosa.istft(stft_matrix, hop_length=hop_length, win_length=win_length)

def mix_audio(voice, noise, snr):
    """Mix voice and noise at the specified SNR."""
    # Calculate voice and noise power
    voice_power = np.mean(voice ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Calculate scaling factor for noise
    scaling_factor = np.sqrt(voice_power / (noise_power * 10 ** (snr / 10)))
    
    # Scale noise and add to voice
    scaled_noise = scaling_factor * noise
    
    # Ensure both signals are the same length
    min_len = min(len(voice), len(scaled_noise))
    mixed = voice[:min_len] + scaled_noise[:min_len]
    
    return mixed, voice[:min_len]

def prepare_spectrograms(audio_signal, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WINDOW_SIZE):
    """Convert audio signal to magnitude and phase spectrograms."""
    # Compute STFT
    stft_signal = stft(audio_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # Separate magnitude and phase
    magnitude = np.abs(stft_signal)
    phase = np.angle(stft_signal)
    
    return magnitude, phase

def apply_mask(magnitude, mask):
    """Apply mask to magnitude spectrogram."""
    return magnitude * mask

def reconstruct_signal(magnitude, phase, hop_length=HOP_LENGTH, win_length=WINDOW_SIZE):
    """Reconstruct time-domain signal from magnitude and phase."""
    # Combine magnitude and phase to get complex STFT
    stft_signal = magnitude * np.exp(1j * phase)
    
    # Compute inverse STFT
    return istft(stft_signal, hop_length=hop_length, win_length=win_length)

def save_audio(signal, file_path, sr=SAMPLE_RATE):
    """Save audio signal to file."""
    torchaudio.save(file_path, torch.tensor(signal).unsqueeze(0), sr)
