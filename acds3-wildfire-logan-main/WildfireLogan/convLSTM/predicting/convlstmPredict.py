import torch
import random


def prediction_on_background(model, data, device):
    """Generates a prediction from the given model using
       the provided (background) data.

    Args:
        model (torch.nn.Module): The neural network model
                                 to be used for prediction.
        data (np.ndarray): The input data for the model,
                           expected to be an array of images.
        device (torch.device): The device to run the model on
                               (e.g., 'cpu' or 'cuda').

    Returns:
        np.ndarray: The predicted image as a NumPy array.
    """

    input_images = data[:4]
    # Add batch and channel dimension
    input_tensor = torch.tensor(input_images,
                                dtype=torch.float32).unsqueeze(0).unsqueeze(2)
    # Move model to device
    model.to(device)
    input_tensor = input_tensor.to(device)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predicted_output = model(input_tensor)
    predicted_image = predicted_output.squeeze(0).squeeze(0).cpu().numpy()

    return predicted_image


def prediction_on_test(model, dataloader, device):
    """Generates a prediction from the given model using a
       randomly selected sample from the testset.

    Args:
        model: The neural network model to be used for prediction.
        dataloader: The dataloader providing the test data.
        device: The device to run the model on (e.g., 'cpu' or 'cuda').

    Returns:
        tuple: A tuple containing the predicted image and
               the real image as NumPy arrays.
    """

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Randomly select an index
    random_index = random.randint(0, len(dataloader.dataset) - 1)

    with torch.no_grad():
        for i, (input_data, target_data) in enumerate(dataloader):
            if i == random_index:
                input_data = input_data.to(device)
                target_data = target_data.to(device)

                # Perform the prediction
                output = model(input_data)
                predicted_image = output.view(256, 256).cpu().numpy()
                real_image = target_data.view(256, 256).cpu().numpy()

    return predicted_image, real_image
