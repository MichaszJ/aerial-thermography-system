import numpy as np
from scipy import integrate
from scipy.interpolate import CubicSpline
import streamlit as st

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

@st.cache
def numerical_position(times, acceleration_x, acceleration_y, acceleration_z):
    accel_x_interp = CubicSpline(times, acceleration_x)
    accel_y_interp = CubicSpline(times, acceleration_y)
    accel_z_interp = CubicSpline(times, acceleration_z)

    vel_x = np.array([integrate.quad(accel_x_interp, times[0], time)[0] for time in times])
    vel_x_interp = CubicSpline(times, vel_x)
    pos_x = np.array([integrate.quad(vel_x_interp, times[0], time)[0] for time in times])

    vel_y = np.array([integrate.quad(accel_y_interp, times[0], time)[0] for time in times])
    vel_y_interp = CubicSpline(times, vel_y)
    pos_y = np.array([integrate.quad(vel_y_interp, times[0], time)[0] for time in times])

    vel_z = np.array([integrate.quad(accel_z_interp, times[0], time)[0] for time in times])
    vel_z_interp = CubicSpline(times, vel_z)
    pos_z = np.array([integrate.quad(vel_z_interp, times[0], time)[0] for time in times])

    return pos_x, pos_y, pos_z