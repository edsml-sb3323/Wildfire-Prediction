import matplotlib.pyplot as plt
import numpy as np


def show_images(predicted_image, actual_image, threshold=0.5, show=True):
    """Displays the predicted image after applying a threshold
       and the actual image side by side.

    Args:
        predicted_image (np.ndarray): The predicted image array.
        actual_image (np.ndarray): The actual image array.
        threshold (float): The threshold value to apply to the predicted image.
                           Greater than threshold is set to 1. Less than
                           threshold is set to 0. Defaults to 0.5.
        show (bool, optional): If True, displays the images. If False, closes
                               the figure without displaying. Defaults to True.
    """

    # Apply thresholding
    thresholded_image = np.where(predicted_image > threshold, 1, 0)

    # Display the thresholded predicted image and the actual last image
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(thresholded_image, cmap='gray')
    axs[0].set_title('Thresholded Predicted Image')
    axs[1].imshow(actual_image, cmap='gray')
    axs[1].set_title('Actual Image')

    # Remove axis for better visualization
    for ax in axs:
        ax.axis('off')

    if show:
        plt.show()
    else:
        plt.close(fig)
