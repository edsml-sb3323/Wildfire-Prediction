import torch
import numpy as np
from WildfireLogan.convLSTM import prediction_on_background
from WildfireLogan.convLSTM import ConvLSTMNetwork


def test_prediction_on_background():
    input_dim = 1
    hidden_dim = [32, 32]
    kernel_size = (3, 3)
    num_layers = 2
    image_size = (128, 128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ConvLSTMNetwork(input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            kernel_size=kernel_size,
                            num_layers=num_layers,
                            image_size=image_size,
                            batch_first=True).to(device)

    # Generate random data
    data = np.random.rand(5, 256, 256)

    # Call the function
    predicted_image = prediction_on_background(model, data, device)

    # Assertions
    assert isinstance(predicted_image, np.ndarray), (
        "The output should be a numpy array"
    )
    assert predicted_image.shape == (256, 256), (
        "The output shape should be (4, 4)"
    )
