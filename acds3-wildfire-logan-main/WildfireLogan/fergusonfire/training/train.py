import torch
from torch import optim
from livelossplot import PlotLosses
from WildfireLogan.fergusonfire import loss_function


class Trainer:
    """
    Trainer class to handle training and validation of a model.

    This class encapsulates the training and validation loops for a given
    model, including optimization, learning rate scheduling, and loss tracking.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader):
                DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader):
                DataLoader for the validation data.
        device (torch.device):
                The device to run the model on (e.g., 'cpu' or 'cuda').
        lr (float, optional):
                Learning rate for the optimizer. Defaults to 1e-3.
    """
    def __init__(self, model, train_loader, val_loader, device, lr=1e-3):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=3, verbose=True
        )
        self.liveloss = PlotLosses()

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        for data, target in self.train_loader:
            data = data.view(data.size(0), -1).to(self.device)
            target = target.view(target.size(0), -1).to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = loss_function(recon_batch, target, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        avg_train_loss = train_loss / len(self.train_loader.dataset)
        return avg_train_loss

    def validate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.view(data.size(0), -1).to(self.device)
                target = target.view(target.size(0), -1).to(self.device)
                recon_batch, mu, logvar = self.model(data)
                loss = loss_function(recon_batch, target, mu, logvar)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(self.val_loader.dataset)
        return avg_val_loss

    def train(self, num_epochs):
        train_losses = []
        val_losses = []
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            self.scheduler.step(val_loss)
            self.liveloss.update({
                'log loss': train_loss,
                'val_log loss': val_loss
            })
            self.liveloss.draw()
            print(
                f"Epoch {epoch}, Training loss: {train_loss:.4f}, "
                f"Validation loss: {val_loss:.4f}"
            )
        return train_losses, val_losses
