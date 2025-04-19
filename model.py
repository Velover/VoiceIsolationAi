import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=N_CHANNELS, dropout_rate=0.2):
        super(UNet, self).__init__()
        
        # Add dropout to help prevent overfitting
        self.dropout = nn.Dropout2d(dropout_rate)
        
        self.encoder1 = ConvBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ConvBlock(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ConvBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ConvBlock(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = ConvBlock(features * 8, features * 16)
        
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(features * 2, features)
        
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Ensure input has the right dimensions
        if x.dim() == 3:  # [batch_size, frequency_bins, time_frames]
            x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, frequency_bins, time_frames]
        
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck with dropout
        bottleneck = self.bottleneck(self.pool4(enc4))
        bottleneck = self.dropout(bottleneck)
        
        # Decoder with dropout at deeper layers
        dec4 = self.upconv4(bottleneck)
        # Crop enc4 to match dec4 size
        enc4_cropped = self.crop(enc4, dec4)
        dec4 = torch.cat((dec4, enc4_cropped), dim=1)
        dec4 = self.decoder4(dec4)
        dec4 = self.dropout(dec4)
        
        dec3 = self.upconv3(dec4)
        # Crop enc3 to match dec3 size
        enc3_cropped = self.crop(enc3, dec3)
        dec3 = torch.cat((dec3, enc3_cropped), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        # Crop enc2 to match dec2 size
        enc2_cropped = self.crop(enc2, dec2)
        dec2 = torch.cat((dec2, enc2_cropped), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        # Crop enc1 to match dec1 size
        enc1_cropped = self.crop(enc1, dec1)
        dec1 = torch.cat((dec1, enc1_cropped), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final layer
        mask = self.sigmoid(self.final_conv(dec1))
        
        return mask
    
    def crop(self, encoder_feature, decoder_feature):
        """
        Center-crop encoder_feature to match the size of decoder_feature.
        """
        if encoder_feature.size() == decoder_feature.size():
            return encoder_feature
            
        _, _, h_e, w_e = encoder_feature.size()
        _, _, h_d, w_d = decoder_feature.size()
        
        # Crop height and width
        h_diff = h_e - h_d
        w_diff = w_e - w_d
        
        if h_diff > 0:
            encoder_feature = encoder_feature[:, :, h_diff//2:h_diff//2+h_d, :]
        if w_diff > 0:
            encoder_feature = encoder_feature[:, :, :, w_diff//2:w_diff//2+w_d]
        
        return encoder_feature
