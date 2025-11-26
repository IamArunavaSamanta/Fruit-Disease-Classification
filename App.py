import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import keras

st.image("https://github.com/IamArunavaSamanta/CNN-Streamlit-Vehicles-Classification-Project/blob/main/images/All_Vehicles.png?raw=true", width=400)
st.markdown('''#### :red-background[:orange[CNN]]:orange[, or Convolutional Neural Network,] ''')

import requests

HF_API_URL = "https://api-inference.huggingface.co/models/google/gemma-2b"  # Example model
HF_TOKEN = st.secrets["HF_TOKEN"]  # Store token in Streamlit secrets

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(payload):
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    return response.json()

st.title("Free LLM on Streamlit Cloud")
user_input = st.text_area("Ask something:")

if st.button("Generate"):
    with st.spinner("Generating response..."):
        output = query({"inputs": user_input})
        st.write(output[0]["generated_text"])

