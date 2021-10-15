import numpy as np


def image_transform(sensor_reading, upscale_factor=48):
    """
    Function that takes in a 2d array representing an image, upscales it and transforms it into
    RGB format for use with opencv

    Args:
        sensor_reading: The 2d array of values from the sensor
        upscale_factor: The upscaling factor being used in the np.kron call

    Returns:
        A 3d numpy array representing the inputted image, upscaled and in RGB format
    """

    upscale = np.kron(sensor_reading, np.ones((upscale_factor, upscale_factor))).astype('uint8')

    img_size = len(upscale)
    transformed_image = np.zeros((img_size, img_size, 3), dtype='uint8')

    for i in range(img_size):
        for j in range(img_size):
            pixel = upscale[i, j]
            transformed_image[i, j] = np.array([pixel, pixel, pixel])

    return transformed_image
