import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import cv2
from scipy import integrate
from scipy.interpolate import CubicSpline

from scanning_utils import image_transform, numerical_position

st.title('Aerial Thermography System - Analysis Software')

st.sidebar.markdown('## Import Data File')
data_file = st.sidebar.file_uploader('Data File (.csv)', type=['csv'])

if data_file is not None:
    data_frame = pd.read_csv(data_file, header=None).rename(columns={0: 'Time', 1: 'Acceleration', 2: 'Image'})

    def str_to_vec(str):
        return [float(num) for num in str.split(',')]

    data_frame['Time'] = data_frame['Time'].apply(lambda time: float(time) / 1e3)
    data_frame['Acceleration'] = data_frame['Acceleration'].apply(str_to_vec)
    data_frame['Acceleration'] = data_frame['Acceleration'].apply(lambda acc: [a * 9.80665 for a in acc])
    data_frame['Acceleration'] = data_frame['Acceleration'].apply(lambda acc: [acc[0], acc[1], acc[2] - 9.80665])

    data_frame['Image'] = data_frame['Image'].apply(str_to_vec)

    st.markdown('## Flight Data')
    st.write(data_frame)

    images = [np.array(image).reshape(8,8) for image in data_frame['Image']]
    accelerations = [acceleration for acceleration in data_frame['Acceleration']]

    st.markdown('## Flight Position')
    times = np.array([time for time in data_frame['Time']])
    accel_x = np.array([accel[0] for accel in accelerations])
    accel_y = np.array([accel[1] for accel in accelerations])
    accel_z = np.array([accel[2] for accel in accelerations])

    pos_x, pos_y, pos_z = numerical_position(times, accel_x, accel_y, accel_z)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=pos_x, y=pos_y, z=pos_z))
    st.plotly_chart(fig)

    st.markdown('## Image Preview')
    img_prev = st.slider('Select Image', min_value=1, max_value=len(images), step=1)

    fig, ax = plt.subplots()
    img = ax.imshow(images[img_prev-1])
    ax.invert_yaxis()
    fig.colorbar(img, label='Temperature (°C)')
    st.pyplot(fig)

    left_col, right_col = st.columns(2)
    with left_col:
        up_factor = st.slider('Upscale Factor', min_value=1, max_value=100, step=1)

        st.markdown('Select image range to merge:')
        first_image = st.number_input('First Image', min_value=1, max_value=len(images))
        last_image = st.number_input('Last Image', value=len(images), min_value=1, max_value=len(images))

    with right_col:
        run_stitch = st.checkbox('Merge images?')
    
    if run_stitch:
        images_transform = [image_transform(image, upscale_factor=up_factor) for image in images]

        stitcher = cv2.Stitcher_create(mode=1)
        status, stitched = stitcher.stitch(images_transform[first_image-1:last_image])
        status_codes = ['OK', 'ERR_NEED_MORE_IMGS', 'ERR_HOMOGRAPHY_EST_FAIL', 'ERR_CAMERA_PARAMS_ADJUST_FAIL']

        st.write('Stitch Status Code: ', status)

        if status != 1:
            fig2, ax2 = plt.subplots()
            ax2.imshow(stitched)

            st.pyplot(fig2)
        else:
            st.markdown(f'**Error:** {status_codes[status]}')