import streamlit as st
#import pandas as pd
from PIL import Image 
from app import get_captions

st.write("""
    ### Image Caption Generator
""")
         
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)
    label = "How many different caption required"
    options = [1,2,3,4,5]
    option = st.selectbox(label,options= options)
    if option is not None :
        captions = get_captions(image,option)
        for caption in captions:
            st.write(caption)
