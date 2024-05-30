import torch
import numpy as np
from WildfireLogan.assimilation import ConvAutoencoder
from WildfireLogan.assimilation import encoding
from WildfireLogan.assimilation import decoding


def test_conv_autoencoder():
    model = ConvAutoencoder()
    assert isinstance(model, ConvAutoencoder), (
        "Model is not an instance of ConvAutoencoder"
    )


def test_forward_pass():
    model = ConvAutoencoder()
    input_tensor = torch.randn(1, 1, 256, 256)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape, (
        "Output shape is not the same as input shape"
    )


def test_encoding():
    model = ConvAutoencoder()
    images = np.random.rand(5, 256, 256).astype(np.float32)
    compressed_images = encoding(model, images)
    assert compressed_images.shape == (5, 32, 32), (
        "Compressed images shape is not correct"
    )


def test_decoding():
    model = ConvAutoencoder()
    compressed_images = np.random.rand(5, 32, 32).astype(np.float32)
    decompressed_images = decoding(model, compressed_images)
    assert decompressed_images.shape == (5, 256, 256), (
        "Decompressed images shape is not correct"
    )
