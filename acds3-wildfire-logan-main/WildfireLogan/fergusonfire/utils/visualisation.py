import matplotlib.pyplot as plt


def plot_images(images, title, rows=1, cols=9):
    """
    Plot a grid of images.

    This function creates a grid of images with the specified number of rows
    and columns. Each image is displayed with its index as the title.

    Args:
        images (numpy.ndarray): An array of images to be plotted.
        title (str): The title for the entire plot.
        rows (int, optional): The number of rows in the grid. Defaults to 1.
        cols (int, optional): The number of columns in the grid. Defaults to 9.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(20, 2))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(images[i])
        ax.axis('off')
        ax.set_title(f'Image {i + 1}')
    plt.suptitle(title)
    plt.show()


def plot_losses(train_losses, val_losses):
    """
    Plot the training and validation losses over epochs.

    This function creates a line plot of the training and validation losses
    over the course of training epochs.

    Args:
        train_losses (list of float): A list of training losses.
        val_losses (list of float): A list of validation losses.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()
