import torch
from livelossplot import PlotLosses


def train_model(model, train_loader, val_loader, device, num_epochs=5):
    """Trains the model and plots training and validation
       losses using livelossplot.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        device (torch.device): The device to train the model on
                               (e.g., 'cpu' or 'cuda').
        num_epochs (int, optional): Number of epochs to train the model.
                                    Defaults to 5.
    """
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    liveloss = PlotLosses()

    for epoch in range(num_epochs):
        logs = {}

        model.train()
        total_train_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        logs['' + 'log_loss'] = avg_train_loss

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs.float())
                loss = criterion(outputs, targets.float())
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        logs['val_' + 'log_loss'] = avg_val_loss

        liveloss.update(logs)
        liveloss.draw()

        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {avg_train_loss:.4f} - '
              f'Val Loss: {avg_val_loss:.4f}')
