import pytest
import torch
from WildfireLogan.fergusonfire import VAE
from WildfireLogan.fergusonfire import loss_function


@pytest.fixture
def vae_model():
    input_dim = 256 * 256  # Size of wildfire data 65,536
    hidden_dim = 128
    latent_dim = 32
    model = VAE(input_dim, hidden_dim, latent_dim)
    return model


@pytest.fixture
def sample_data():
    # Create sample input data for testing
    return torch.randint(0, 2, (2, 256 * 256), dtype=torch.int32).float()


def test_encode(vae_model, sample_data):
    mu, logvar = vae_model.encode(sample_data)
    assert mu.shape == (2, vae_model.fc21.out_features)
    assert logvar.shape == (2, vae_model.fc22.out_features)
    assert not torch.isnan(mu).any()
    assert not torch.isnan(logvar).any()


def test_reparameterize(vae_model, sample_data):
    mu, logvar = vae_model.encode(sample_data)
    z = vae_model.reparameterize(mu, logvar)
    assert z.shape == (2, vae_model.fc21.out_features)
    assert not torch.isnan(z).any()


def test_decode(vae_model):
    z = torch.randn(2, vae_model.fc21.out_features)
    recon_x = vae_model.decode(z)
    assert recon_x.shape == (2, 256*256)
    assert (recon_x >= 0).all() and (recon_x <= 1).all()


def test_forward(vae_model, sample_data):
    recon_x, mu, logvar = vae_model(sample_data)
    assert recon_x.shape == (2, 256*256)
    assert mu.shape == (2, vae_model.fc21.out_features)
    assert logvar.shape == (2, vae_model.fc22.out_features)
    assert (recon_x >= 0).all() and (recon_x <= 1).all()


def test_loss_function(vae_model, sample_data):
    recon_x, mu, logvar = vae_model(sample_data)
    loss = loss_function(recon_x, sample_data, mu, logvar)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss).any()


if __name__ == "__main__":
    pytest.main()
