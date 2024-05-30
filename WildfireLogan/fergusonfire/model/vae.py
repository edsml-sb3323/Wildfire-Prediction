import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.

    This class defines a Variational Autoencoder (VAE) with an encoder and
    decoder network, along with reparameterization to handle the stochastic
    nature of the latent space.

    Args:
        nn.Module (_type_): Base class for all neural network modules.
        input_dim (int): The dimension of the input data.
        hidden_dim (int, optional):
                The dimension of the hidden layer. Defaults to 128.
        latent_dim (int, optional):
                The dimension of the latent space. Defaults to 32.
    """
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout(h1)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
