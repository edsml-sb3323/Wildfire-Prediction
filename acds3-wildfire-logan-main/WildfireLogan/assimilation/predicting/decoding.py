import torch


def decoding(model, images_compr):
    """
    Decode compressed images using the provided model.

    Args:
        model (torch.nn.Module): The model with a decoder
                                 method used for decompressing images.
        images_compr (array-like): A list or array of compressed
                                   images to be decompressed.

    Returns:
        numpy.ndarray: A numpy array containing the decompressed
                       images with shape (5, 256, 256).
    """
    data_decompr = model.decoder(
        torch.tensor(images_compr).float().reshape(5, 1, 32, 32)
    ).detach().numpy().reshape(5, 256, 256)
    return data_decompr
