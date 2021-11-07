import streamlit as st
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import cv2

from scanning_utils import image_transform

st.title('Aerial Thermography System - Analysis Software')

st.sidebar.markdown('## Import Data File')
data_file = st.sidebar.file_uploader('Data File (.csv)', type=['csv'])

if data_file is not None:
    data_frame = pd.read_csv(data_file)

    st.write(data_frame)

    def str_to_vec(str):
        return [float(num) for num in str.split(',')]

    data_frame['acceleration'] = data_frame['acceleration'].apply(str_to_vec)
    data_frame['image'] = data_frame['image'].apply(str_to_vec)

    images = [np.array(image).reshape(8,8) for image in data_frame['image']]
    accelerations = [np.array(acceleration) for acceleration in data_frame['acceleration']]

    img_prev = st.slider('Image Preview', min_value=1, max_value=len(images), step=1)

    fig, ax = plt.subplots()
    ax.imshow(images[img_prev-1])
    st.pyplot(fig)

    left_col, right_col = st.columns(2)
    with left_col:
        up_factor = st.slider('Upscale Factor', min_value=1, max_value=100, step=1)

    with right_col:
        run_stitch = st.checkbox('Merge images?')
    
    if run_stitch:
        images_transform = [image_transform(image, upscale_factor=up_factor) for image in images]

        stitcher = cv2.Stitcher_create(mode=1)
        status, stitched = stitcher.stitch(images_transform)

        st.write('Stitch Status Code: ', status)

        fig2, ax2 = plt.subplots()
        ax2.imshow(stitched)

        st.pyplot(fig2)