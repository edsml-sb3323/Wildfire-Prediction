import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from WildfireLogan.fergusonfire import VAE
from WildfireLogan.fergusonfire import Trainer


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def vae_model(device):
    input_dim = 256*256  # Example input dimension
    hidden_dim = 128
    latent_dim = 32
    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    return model


@pytest.fixture
def data_loaders():
    # Create dummy data
    train_data = torch.randint(0, 2, (1, 256, 256), dtype=torch.int32).float()
    val_data = torch.randint(0, 2, (1, 256, 256), dtype=torch.int32).float()
    train_dataset = TensorDataset(train_data, train_data)
    val_dataset = TensorDataset(val_data, val_data)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    return train_loader, val_loader


@pytest.fixture
def trainer(vae_model, data_loaders, device):
    train_loader, val_loader = data_loaders
    return Trainer(vae_model, train_loader, val_loader, device)


def test_trainer_initialization(trainer):
    assert isinstance(trainer.model, VAE)
    assert isinstance(trainer.train_loader, DataLoader)
    assert isinstance(trainer.val_loader, DataLoader)
    assert trainer.device.type in ["cuda", "cpu"]
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert isinstance(
        trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
    )


def test_train_epoch(trainer):
    initial_loss = trainer.train_epoch(1)
    assert isinstance(initial_loss, float)
    assert initial_loss > 0


def test_validate_epoch(trainer):
    validation_loss = trainer.validate_epoch(1)
    assert isinstance(validation_loss, float)
    assert validation_loss > 0


def test_train(trainer):
    num_epochs = 2
    train_losses, val_losses = trainer.train(num_epochs)
    assert len(train_losses) == num_epochs
    assert len(val_losses) == num_epochs
    assert all(isinstance(loss, float) for loss in train_losses)
    assert all(isinstance(loss, float) for loss in val_losses)


if __name__ == "__main__":
    pytest.main()
