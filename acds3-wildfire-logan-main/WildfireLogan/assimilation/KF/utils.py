import numpy as np
from numpy.linalg import inv
import torch

R_factors = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
B_factors = [1, 10, 100, 1000, 10000, 100000]


def covariance_matrix(X):
    """
    Calculate the covariance matrix for the given data.

    Args:
        X (numpy.ndarray): Input data matrix of shape (n_samples, n_features).

    Returns:
        numpy.ndarray: Covariance matrix of the input data.
    """
    means = np.mean(X, axis=0, keepdims=True)
    dev_matrix = X - means
    res = np.dot(dev_matrix.T, dev_matrix) / (X.shape[0] - 1)
    return res


def KalmanGain(B, H, R):
    """
    Compute the Kalman gain.

    Args:
        B (numpy.ndarray): Covariance matrix of the model state.
        H (numpy.ndarray): Observation matrix.
        R (numpy.ndarray): Covariance matrix of the observation noise.

    Returns:
        numpy.ndarray: Kalman gain matrix.
    """
    tempInv = inv(R + np.dot(H, np.dot(B, H.T)))
    res = np.dot(B, np.dot(H.T, tempInv))
    return res


def update_prediction(x, K, H, y):
    """
    Update the prediction using the Kalman gain.

    Args:
        x (numpy.ndarray): Current state estimate.
        K (numpy.ndarray): Kalman gain matrix.
        H (numpy.ndarray): Observation matrix.
        y (numpy.ndarray): Observation vector.

    Returns:
        numpy.ndarray: Updated state estimate.
    """
    res = x + np.dot(K, (y - np.dot(H, x)))
    return res


def mse(y_obs, y_pred):
    """
    Calculate the mean squared error (MSE) between
    observed and predicted values.

    Args:
        y_obs (numpy.ndarray): Observed values.
        y_pred (numpy.ndarray): Predicted values.

    Returns:
        float: Mean squared error.
    """
    return np.square(np.subtract(y_obs, y_pred)).mean()


def grid_search(observation_data, model_data, R_factors, B_factors):
    """
    Perform grid search to find the best R and B factors for the Kalman filter.

    Args:
        observation_data (numpy.ndarray): Observation data.
        model_data (numpy.ndarray): Model data.
        R_factors (list): List of factors to scale the initial observation
                          noise covariance matrix.
        B_factors (list): List of factors to scale the
                          initial state covariance matrix.

    Returns:
        tuple: Best R matrix, best B matrix, and the best MSE value.
    """
    num_timesteps, height, width = model_data.shape
    num_channels = height * width
    model_data_flat = model_data.reshape(num_timesteps, num_channels)
    observation_data_flat = observation_data.reshape(
        num_timesteps, num_channels
        )

    initial_R = covariance_matrix(observation_data_flat)
    initial_B = np.diag(np.diag(covariance_matrix(model_data_flat)))

    best_R = None
    best_B = None
    best_mse = float('inf')

    for r_factor in R_factors:
        for b_factor in B_factors:
            R = initial_R * r_factor
            B = initial_B * b_factor

            H = np.eye(num_channels)  # Observation operator
            K = KalmanGain(B, H, R)

            updated_model_data = []
            for i in range(num_timesteps):
                x_flat = model_data_flat[i]
                y_flat = observation_data_flat[i]
                updated_flat = update_prediction(x_flat, K, H, y_flat)
                updated_model_data.append(updated_flat.reshape(height, width))

            updated_model_data = np.array(updated_model_data)

            current_mse = mse(observation_data, updated_model_data)

            if current_mse < best_mse:
                best_mse = current_mse
                best_R = R
                best_B = B

    return best_R, best_B, best_mse


def update_data(
    task_data_compr, satellite_data_compr, cae, K, H,
    task_images, satellite_images, update_prediction, task
):
    """
    Update the compressed task data using the Kalman filter
    and reconstruct the images.

    Args:
        task_data_compr (numpy.ndarray): Compressed task data.
        satellite_data_compr (numpy.ndarray): Compressed satellite data.
        cae (ConvAutoencoder): Convolutional autoencoder model.
        K (numpy.ndarray): Kalman gain matrix.
        H (numpy.ndarray): Observation matrix.
        task_images (numpy.ndarray): Original task images.
        satellite_images (numpy.ndarray): Original satellite images.
        update_prediction (function): Function to update the state prediction.
        task (int): Task identifier (1 or 2).

    Returns:
        numpy.ndarray: Reconstructed task data after updating.
    """
    updated_task_data_compr = []

    task_name = "Task 1" if task == 1 else "Task 2"

    for i in range(5):
        y = satellite_data_compr.reshape(5, -1)[i]
        x = task_data_compr.reshape(5, -1)[i]
        x = update_prediction(x, K, H, y)
        updated_task_data_compr.append(x)

    updated_task_data_compr = np.array(updated_task_data_compr)

    mse_before_DA = mse(satellite_data_compr, task_data_compr)
    mse_after_DA = mse(satellite_data_compr.reshape(5, -1),
                       updated_task_data_compr)

    updated_task_data_recon = cae.decoder(
        torch.Tensor(updated_task_data_compr).reshape(5, 1, 32, 32)
    ).detach().numpy().reshape(5, 256, 256)
    updated_task_data_recon = updated_task_data_recon.reshape(5, 256, 256)

    physical_mse_before_da = mse(satellite_images, task_images)
    physical_mse_after_da = mse(satellite_images, updated_task_data_recon)

    print(task_name)
    print(f'MSE before DA in latent space: {mse_before_DA}')
    print(f'MSE after DA in latent space: {mse_after_DA}')

    print(f'MSE before DA in physical space: {physical_mse_before_da}')
    print(f'MSE after DA in physical space: {physical_mse_after_da}')

    return updated_task_data_recon
