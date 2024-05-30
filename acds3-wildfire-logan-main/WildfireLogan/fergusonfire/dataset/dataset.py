import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class FireDataset:
    """
    A dataset class for handling training and testing data for fire detection.

    This class loads datasets from provided file paths, detects cycles within
    the data, and creates pairs of data points for training and validation.
    It also provides data loaders for batch processing.

    Args:
        train_path (str): Path to the training dataset file (numpy .npy format)
        test_path (str): Path to the testing dataset file (numpy .npy format).
    """
    def __init__(self, train_path, test_path):
        self.train_dataset = np.load(train_path, mmap_mode='r')
        self.test_dataset = np.load(test_path, mmap_mode='r')
        self.train_cycle_indices = self.detect_cycles(len(self.train_dataset))
        self.test_cycle_indices = self.detect_cycles(len(self.test_dataset))

    def detect_cycles(self, length, threshold=0.1):
        cycle_indices = [i for i in range(0, length, 100)]
        return cycle_indices

    def create_pairs(self, data, cycle_start_indices):
        pairs = []
        for start_idx in cycle_start_indices:
            cycle_end_idx = start_idx + 100
            for i in range(start_idx, cycle_end_idx, 10):
                if i + 10 < cycle_end_idx:
                    pair = (data[i], data[i + 10])
                    pairs.append(pair)
        return np.array(pairs)

    def get_train_pairs(self):
        return self.create_pairs(self.train_dataset, self.train_cycle_indices)

    def get_val_pairs(self):
        return self.create_pairs(self.test_dataset, self.test_cycle_indices)

    def get_dataloaders(self, batch_size=9):
        train_pairs = self.get_train_pairs()
        val_pairs = self.get_val_pairs()

        train_dataset = TensorDataset(
            torch.tensor(train_pairs[:, 0], dtype=torch.float32),
            torch.tensor(train_pairs[:, 1], dtype=torch.float32)
        )

        val_dataset = TensorDataset(
            torch.tensor(val_pairs[:, 0], dtype=torch.float32),
            torch.tensor(val_pairs[:, 1], dtype=torch.float32)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        return train_loader, val_loader


def load_data(train_path, test_path):
    """
    Load training and testing datasets from the given file paths.

    Args:
        train_path (str): Path to the training dataset file (numpy .npy format)
        test_path (str): Path to the testing dataset file (numpy .npy format).

    Returns:
        tuple: A tuple containing:
            - train_dataset (numpy.memmap): The training dataset loaded in
                                            memory-mapped mode.
            - test_dataset (numpy.ndarray): The testing dataset.
    """
    train_dataset = np.load(train_path, mmap_mode='r')
    test_dataset = np.load(test_path)
    return train_dataset, test_dataset
