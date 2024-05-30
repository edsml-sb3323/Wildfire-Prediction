from torch.utils.data import random_split


def train_test_split(train_dataset, train_per=0.8):
    """Splits the given dataset into training and validation subsets.

    Args:
        train_dataset (Dataset): The dataset to be split.
        train_per (float, optional): The proportion of the dataset to
                                     include in the training subset.
                                     Defaults to 0.8.

    Returns:
        tuple: A tuple containing the training subset and
               the validation subset.
    """

    train_size = int(train_per * len(train_dataset))  # 80% for training
    val_size = len(train_dataset) - train_size  # 20% for validation

    # Use random_split to split the dataset
    train_subset, val_subset = random_split(train_dataset,
                                            [train_size, val_size])

    return train_subset, val_subset
