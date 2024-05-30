import torch
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    """
    A custom dataset class grouping images to sequence,
    suitable for use with RNN, LSTM networks.

    Args:
        data (np.ndarray): The dataset containing all data points.
        input_indices (list of lists): Indices to access the input sequences.
        target_indices (list of lists or ints): Indices to access the target
            sequences or single target values. If target_indices is a single
            value, the returned target_data does not have a sequence
            dimension. If target_indices is a list, the returned target_data
            has a sequence dimension.
        oneD (bool, optional): If True, reshapes image data to one dimension.
            Default is False.

    Raises:
        TypeError: If the type of target data is not int or list.

    Returns:
        tuple: A tuple containing input and target data in tensor format.
    """

    def __init__(self, data, input_indices, target_indices, oneD=False):
        self.data = data
        self.input_indices = input_indices
        self.target_indices = target_indices
        self.oneD = oneD

    def __len__(self):
        return len(self.input_indices)

    def __getitem__(self, idx):
        input_idx = self.input_indices[idx]
        target_idx = self.target_indices[idx]

        # Retrieve the data points using the indices
        input_data = self.data[input_idx]
        target_data = self.data[target_idx]

        # Convert to tensors
        input_data = torch.tensor(input_data, dtype=torch.float32)
        target_data = torch.tensor(target_data, dtype=torch.float32)

        if isinstance(self.target_indices[0], int):
            input_data = input_data.unsqueeze(1)
            target_data = target_data.unsqueeze(0)
        elif isinstance(self.target_indices[0], list):
            input_data = input_data.unsqueeze(1)
            target_data = target_data.unsqueeze(1)
        else:
            raise TypeError("Check the type of target data")

        if self.oneD:
            seq_length = len(self.input_indices[0])
            image_width = input_data.shape[2]
            image_height = input_data.shape[3]
            input_data = input_data.view(
                seq_length, image_width * image_height)
            if isinstance(self.target_indices[0], int):
                target_data = target_data.view(image_width * image_height)
            elif isinstance(self.target_indices[0], list):
                target_data = target_data.view(
                    seq_length, image_width * image_height)

        return input_data, target_data
