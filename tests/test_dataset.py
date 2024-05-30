import pytest
import numpy as np
from torch.utils.data import DataLoader
from WildfireLogan.fergusonfire import load_data, FireDataset


# Fixtures for temporary test data
@pytest.fixture
def create_test_data(tmp_path):
    train_data = np.random.rand(500, 256, 256)
    test_data = np.random.rand(500, 256, 256)
    train_path = tmp_path / "train.npy"
    test_path = tmp_path / "test.npy"
    np.save(train_path, train_data)
    np.save(test_path, test_data)
    return str(train_path), str(test_path)


# Test for load_data function
def test_load_data(create_test_data):
    train_path, test_path = create_test_data
    train_dataset, test_dataset = load_data(train_path, test_path)

    assert isinstance(train_dataset, np.ndarray)
    assert isinstance(test_dataset, np.ndarray)
    assert train_dataset.shape == (500, 256, 256)
    assert test_dataset.shape == (500, 256, 256)


# Test for FireDataset class
def test_fire_dataset(create_test_data):
    train_path, test_path = create_test_data
    dataset = FireDataset(train_path, test_path)

    assert len(dataset.train_dataset) == 500
    assert len(dataset.test_dataset) == 500
    assert len(dataset.train_cycle_indices) == 5
    assert len(dataset.test_cycle_indices) == 5


# Test for create_pairs method
def test_create_pairs(create_test_data):
    train_path, test_path = create_test_data
    dataset = FireDataset(train_path, test_path)

    train_pairs = dataset.get_train_pairs()
    val_pairs = dataset.get_val_pairs()

    assert train_pairs.shape[0] > 0
    assert train_pairs.shape[1] == 2
    assert val_pairs.shape[0] > 0
    assert val_pairs.shape[1] == 2


# Test for get_dataloaders method
def test_get_dataloaders(create_test_data):
    train_path, test_path = create_test_data
    dataset = FireDataset(train_path, test_path)
    train_loader, val_loader = dataset.get_dataloaders(batch_size=9)

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert len(train_loader) > 0
    assert len(val_loader) > 0

    for data in train_loader:
        inputs, targets = data
        assert inputs.shape == (9, 256, 256)
        assert targets.shape == (9, 256, 256)
        break  # Only check the first batch

    for data in val_loader:
        inputs, targets = data
        assert inputs.shape == (9, 256, 256)
        assert targets.shape == (9, 256, 256)
        break  # Only check the first batch


if __name__ == "__main__":
    pytest.main()
