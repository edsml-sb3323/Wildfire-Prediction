import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    A convolutional autoencoder model for image compression and decompression.

    The model consists of an encoder and a decoder.
    The encoder compresses the input image
    into a smaller representation, and the decoder reconstructs the
    image from this compressed representation.

    Args:
        nn.Module: Inherits from the PyTorch base module class.
    """

    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # (1, 256, 256) -> (8, 128, 128)
            nn.Conv2d(1, 8, 5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),  # (8, 128, 128) -> (8, 64, 64)
            # (8, 64, 64) -> (16, 32, 32)
            nn.Conv2d(8, 16, 5, stride=2, padding=2),
            nn.LeakyReLU(),
            # (16, 32, 32) -> (1, 32, 32)
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            # (1, 32, 32) -> (16, 32, 32)
            nn.ConvTranspose2d(1, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(),

            # (16, 32, 32) -> (8, 64, 64)
            nn.ConvTranspose2d(
                16, 8, 5, stride=2, padding=2, output_padding=1
                ),
            nn.LeakyReLU(),

            # (8, 64, 64) -> (8, 128, 128)
            nn.ConvTranspose2d(8, 8, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(),

            # (8, 128, 128) -> (1, 256, 256)
            nn.ConvTranspose2d(8, 1, 5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()  # Using Sigmoid to output values in the range [0, 1]
        )

    def forward(self, x):
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 256, 256),
            representing a batch of grayscale images.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, 256, 256),
            representing the reconstructed images.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
