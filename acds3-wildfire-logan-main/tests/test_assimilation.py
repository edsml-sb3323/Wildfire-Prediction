import numpy as np
from WildfireLogan.assimilation import (
    covariance_matrix, KalmanGain, update_prediction, mse, grid_search
)


# Test data for covariance_matrix
def test_covariance_matrix():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    expected_result = np.array([[4, 4], [4, 4]])
    np.testing.assert_array_almost_equal(covariance_matrix(X), expected_result)


# Test data for KalmanGain
def test_KalmanGain():
    B = np.array([[1, 0], [0, 1]])
    H = np.array([[1, 0], [0, 1]])
    R = np.array([[1, 0], [0, 1]])
    expected_result = np.array([[0.5, 0], [0, 0.5]])
    np.testing.assert_array_almost_equal(KalmanGain(B, H, R), expected_result)


# Test data for update_prediction
def test_update_prediction():
    x = np.array([1, 2])
    K = np.array([[0.5, 0], [0, 0.5]])
    H = np.array([[1, 0], [0, 1]])
    y = np.array([2, 4])
    expected_result = np.array([1.5, 3])
    np.testing.assert_array_almost_equal(
        update_prediction(x, K, H, y), expected_result
    )


# Test data for mse
def test_mse():
    y_obs = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    expected_result = 0.0
    assert mse(y_obs, y_pred) == expected_result


# Test data for grid_search
def test_grid_search():
    observation_data = np.random.rand(5, 2, 2)
    model_data = np.random.rand(5, 2, 2)
    R_factors = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    B_factors = [1, 10, 100, 1000, 10000, 100000]
    best_R, best_B, best_mse = grid_search(
        observation_data, model_data, R_factors, B_factors
    )
    assert best_R is not None
    assert best_B is not None
    assert best_mse is not None
