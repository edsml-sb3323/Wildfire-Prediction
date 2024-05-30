from sklearn.metrics import mean_squared_error


def compare_mse(output, data):
    """Calculates the Mean Squared Error (MSE) between
       the output and the target data.

    Args:
        output (np.ndarray or torch.Tensor): The predicted output data.
        data (np.ndarray or torch.Tensor): The actual target data,
                                           usually a single image.

    Returns:
        float: The calculated Mean Squared Error (MSE) between
               the output and the target data.
    """

    mse = mean_squared_error(output, data)
    return mse
