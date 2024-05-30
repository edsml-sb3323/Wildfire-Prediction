import matplotlib.pyplot as plt
import numpy as np
from WildfireLogan.convLSTM import show_images


def test_show_images():
    # Use the Agg backend to prevent images from displaying during the test
    plt.switch_backend('Agg')

    # Generate random predicted and actual images
    predicted_image = np.random.rand(100, 100)
    actual_image = np.random.rand(100, 100)
    threshold = 0.5

    # Call the function with show=False to avoid displaying the figure
    show_images(predicted_image, actual_image, threshold, show=False)

    # Check if a figure was created
    assert plt.gcf().number == 1

    # Clear the figure after test to avoid memory issues
    plt.clf()
