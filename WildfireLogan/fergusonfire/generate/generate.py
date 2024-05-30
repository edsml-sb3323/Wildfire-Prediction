import torch
from sklearn.metrics import mean_squared_error


def generate_images(
    model, latent_dim, num_images, device,
    threshold_low=0.3, threshold_high=0.5
):
    """
    Generate images using a trained model from random latent vectors.

    Args:
        model (torch.nn.Module): The trained model with a decode method.
        latent_dim (int): The dimensionality of the latent space.
        num_images (int): The number of images to generate.
        device (torch.device): The device to run the model on
                               (e.g., 'cpu' or 'cuda').
        threshold_low (float, optional): Lower threshold for binarizing
                                         the generated images. Defaults to 0.3.
        threshold_high (float, optional): Upper threshold for binarizing the
                                          generated images. Defaults to 0.5.

    Returns:
        numpy.ndarray: Generated images after thresholding, reshaped
                       to (num_images, 256, 256).
    """
    z = torch.randn(num_images, latent_dim).to(device)
    with torch.no_grad():
        generated_images = model.decode(z).cpu().numpy()
    generated_images[generated_images < threshold_low] = 0
    generated_images[generated_images > threshold_high] = 1
    return generated_images.reshape(num_images, 256, 256)


def find_min_mse_image(generated_images, satellite_data):
    """
    Find the generated image with the minimum Mean Squared Error (MSE)
    compared to satellite data.

    Args:
        generated_images (numpy.ndarray): An array of generated images.
        satellite_data (numpy.ndarray): An array of satellite images.

    Returns:
        tuple: A tuple containing:
            - min_mse_generated_image (numpy.ndarray):
                            The generated image with the minimum MSE.
            - min_mse_satellite_image (numpy.ndarray):
                            The corresponding satellite image with the min MSE.
            - min_mse_gen_idx (int):
                            The index of the generated image with the min MSE.
            - min_mse_sat_idx (int):
                            The index of the satellite image with the min MSE.
    """
    min_mse = float('inf')
    min_mse_generated_image = None
    min_mse_satellite_image = None
    min_mse_gen_idx = 0
    min_mse_sat_idx = 0

    for i, gen_img in enumerate(generated_images):
        for j, satellite_img in enumerate(satellite_data):
            mse = mean_squared_error(gen_img.flatten(),
                                     satellite_img.flatten())
            if mse < min_mse:
                min_mse = mse
                min_mse_generated_image = gen_img
                min_mse_satellite_image = satellite_img
                min_mse_gen_idx = i
                min_mse_sat_idx = j

    return (
        min_mse_generated_image,
        min_mse_satellite_image,
        min_mse_gen_idx,
        min_mse_sat_idx
    )


def generate_sequence(model, initial_latent_vector, num_steps, device):
    """
    Generate a sequence of images using a trained model starting
    from an initial latent vector.

    Args:
        model (torch.nn.Module):
                The trained model with encode and decode methods.
        initial_latent_vector (torch.Tensor):
                The initial latent vector to start the sequence.
        num_steps (int):
                The number of steps (images) to generate in the sequence.
        device (torch.device):
                The device to run the model on (e.g., 'cpu' or 'cuda').

    Returns:
        list of numpy.ndarray: A list of generated images in the sequence,
        each of shape (256, 256).
    """
    model.eval()
    generated_images = []
    z = initial_latent_vector.to(device)
    with torch.no_grad():
        for _ in range(num_steps):
            generated_image = model.decode(z)
            generated_images.append(
                generated_image.view(256, 256).cpu().numpy()
                )
            generated_image_flat = generated_image.view(1, -1)
            mu, logvar = model.encode(generated_image_flat)
            z = model.reparameterize(mu, logvar)
    return generated_images
