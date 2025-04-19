import torch
import torch.nn as nn
import torch.nn.functional as F

class VoiceIsolationModel(nn.Module):
    """
    CNN model for voice isolation using spectrogram masking
    """
    def __init__(self, n_fft=512):
        """
        Initialize the model.
        
        Args:
            n_fft: Number of frequency bins in the STFT
        """
        super(VoiceIsolationModel, self).__init__()
        
        # Input: [batch_size, 1, n_fft//2 + 1, time_frames]
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()  # Output mask values between 0 and 1
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input spectrogram [batch_size, 1, frequency_bins, time_frames]
            
        Returns:
            Predicted mask with same dimensions as input
        """
        # Reshape input if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
            
        # Encoder
        encoded = self.encoder(x)
        
        # Decoder
        mask = self.decoder(encoded)
        
        return mask

class MaskedLoss(nn.Module):
    """
    Custom loss function for voice isolation
    """
    def __init__(self):
        super(MaskedLoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred_mask, target_mask, mixed_spec=None, voice_spec=None):
        """
        Compute loss between predicted mask and target mask.
        
        Args:
            pred_mask: Predicted mask
            target_mask: Target mask
            mixed_spec: Mixed spectrogram (optional)
            voice_spec: Voice spectrogram (optional)
            
        Returns:
            Loss value
        """
        # Basic mask prediction loss
        mask_loss = self.mse(pred_mask, target_mask)
        
        # If spectrograms are provided, add spectral loss
        if mixed_spec is not None and voice_spec is not None:
            pred_voice_spec = mixed_spec * pred_mask
            spectral_loss = self.mse(pred_voice_spec, voice_spec)
            return mask_loss + spectral_loss
            
        return mask_loss
