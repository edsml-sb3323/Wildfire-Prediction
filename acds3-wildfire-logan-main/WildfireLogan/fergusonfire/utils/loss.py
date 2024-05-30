import torch
import torch.nn.functional as F


def loss_function(recon_x, x, mu, logvar, alpha=1.0, beta=2.0):
    """
    Compute the loss function for the Variational Autoencoder (VAE).

    The loss function is a combination of the Binary Cross Entropy (BCE) and
    the Kullback-Leibler Divergence (KLD). The BCE measures the reconstruction
    error, while the KLD measures the divergence between the learned latent
    distribution and the prior distribution.

    Args:
        recon_x (torch.Tensor): Reconstructed input.
        x (torch.Tensor): Original input.
        mu (torch.Tensor): Mean of the latent space.
        logvar (torch.Tensor): Log variance of the latent space.
        alpha (float, optional):
            Weight for the reconstruction loss (BCE). Defaults to 1.0.
        beta (float, optional):
            Weight for the Kullback-Leibler Divergence (KLD). Defaults to 2.0.

    Returns:
        torch.Tensor: The computed loss value.
    """
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return alpha * BCE + beta * KLD
