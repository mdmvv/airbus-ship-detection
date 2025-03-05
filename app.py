import os
import streamlit as st
from ship_detection import build_model, detect


image_option = st.selectbox(
    "Target image for ship detection",
    os.listdir("ship_images")
)


model = build_model()
fig = detect(model, image_option)

st.pyplot(fig)
