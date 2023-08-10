import base64

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from model import CNN
from torchvision.io import read_image
from PIL import Image
from torchvision import transforms

@st.cache_resource
def get_model(device):
    model = CNN()
    return model

def get_transform(input):
    norm_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: x.float()),
        transforms.Resize((200, 200)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return norm_transform(input)

def streamlit_app(device, lti, itl):
    st.title("Butterfly Classification using CNN")
    file_ = open("butGIF.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            unsafe_allow_html=True,
        )
    st.write("Hello. Enter a Butterfly image and I will tell you which species it is :sunglasses: :butterfly:")
    butterfly_input = st.file_uploader("Upload your butterfly image here")
    if butterfly_input:
        model = get_model(device)
        model.load_state_dict(torch.load("./butterflynet.pth", map_location=device))
        model.eval()
        input = Image.open(butterfly_input)
        input_tensor = get_transform(input).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top3probs = torch.topk(probs[0], 3)
        st.image(input)
        for prob, index in zip(top3probs[0], top3probs[1]):
            string = "" + itl[index.item()] + " with probability " + str(round(prob.item() * 100, 3)) + "%"
            st.write(string)

