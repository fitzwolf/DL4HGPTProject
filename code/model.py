import torch
import torch.nn as nn

class TRACEEncoderMasked(nn.Module):
    """
    TRACE-style convolutional encoder with masking capability.
    Inputs: 
        x: tensor of shape (B, C, T)
        mask: tensor of same shape as x with 0s where masked, 1s elsewhere
    Returns:
        reconstructed summary (B, C)
        latent embedding (B, latent_dim)
    """
    def __init__(self, input_channels=10, latent_dim=128):
        super(TRACEEncoderMasked, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_channels)

    def forward(self, x, mask=None):
        x_input = x.clone()
        if mask is not None:
            x_input = x_input * mask
        encoded = self.encoder(x_input)
        encoded = encoded.squeeze(-1)
        latent = self.fc(encoded)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class TransferClassifier(nn.Module):
    """
    Transfer learning classifier using frozen TRACE encoder.
    """
    def __init__(self, encoder, latent_dim=128, num_classes=3):
        super(TransferClassifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            latent = self.encoder.encoder(x)
            latent = latent.squeeze(-1)
            latent = self.encoder.fc(latent)
        out = self.fc(latent)
        return out
