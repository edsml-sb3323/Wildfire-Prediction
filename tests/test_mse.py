import pytest
import numpy as np
from sklearn.metrics import mean_squared_error
from WildfireLogan.convLSTM import compare_mse


def test_compare_mse():
    # Generate random output and data
    output = np.random.rand(10, 10)
    data = np.random.rand(10, 10)

    # Compute MSE using the compare_mse function
    mse_function = compare_mse(output, data)

    # Compute MSE using sklearn's mean_squared_error
    mse_expected = mean_squared_error(output, data)

    # Assertions
    assert isinstance(mse_function, float), "The output should be a float"
    assert mse_function == pytest.approx(mse_expected), (
        f"The MSE should be {mse_expected}, but got {mse_function}"
    )
