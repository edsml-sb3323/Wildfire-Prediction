import torch


def encoding(model, images):
    """
    Encode images using the provided model.

    Args:
        model (torch.nn.Module): The model with an encoder
                                 method used for compressing images.
        images (array-like): A list or array of images to be compressed.

    Returns:
        numpy.ndarray: A numpy array containing the compressed
                       images with shape (5, 32, 32).
    """
    data_compr = model.encoder(
        torch.tensor(images).float().reshape(5, 1, 256, 256)
    ).detach().numpy().reshape(5, 32, 32)
    return data_compr
