import streamlit as st
import pandas as pd
import pickle
from scipy.sparse import hstack

# Load the saved model
with open("laptop_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the saved encoder
with open("laptop_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


# Feature names used during training
numerical_features = [
    'spec_rating',
    'Ram',
    'ROM',
    'display_size',
    'resolution_width',
    'resolution_height',
    'warranty'
]

categorical_features = [
    'brand',
    'processor',
    'CPU',
    'Ram_type',
    'ROM_type',
    'GPU',
    'OS'
]


# App title
st.title("💻 Laptop Price Prediction")

st.write(
    "Enter the laptop specifications to predict its estimated price."
)


# User inputs
brand = st.text_input("Brand", "HP")

processor = st.text_input(
    "Processor",
    "Intel Core i5"
)

CPU = st.text_input(
    "CPU",
    "Intel Core i5"
)

Ram = st.number_input(
    "RAM (GB)",
    min_value=1,
    value=16
)

Ram_type = st.text_input(
    "RAM Type",
    "DDR4"
)

ROM = st.number_input(
    "ROM / Storage (GB)",
    min_value=1,
    value=512
)

ROM_type = st.text_input(
    "ROM Type",
    "SSD"
)

GPU = st.text_input(
    "GPU",
    "Intel Integrated"
)

display_size = st.number_input(
    "Display Size (inches)",
    min_value=1.0,
    value=15.6
)

resolution_width = st.number_input(
    "Resolution Width",
    min_value=1,
    value=1920
)

resolution_height = st.number_input(
    "Resolution Height",
    min_value=1,
    value=1080
)

OS = st.text_input(
    "Operating System",
    "Windows 11"
)

warranty = st.number_input(
    "Warranty (years)",
    min_value=0,
    value=1
)

spec_rating = st.number_input(
    "Specification Rating",
    min_value=0.0,
    value=75.0
)


# Prediction
if st.button("Predict Laptop Price"):

    sample_laptop = pd.DataFrame([{
        'brand': brand,
        'processor': processor,
        'CPU': CPU,
        'Ram': Ram,
        'Ram_type': Ram_type,
        'ROM': ROM,
        'ROM_type': ROM_type,
        'GPU': GPU,
        'display_size': display_size,
        'resolution_width': resolution_width,
        'resolution_height': resolution_height,
        'OS': OS,
        'warranty': warranty,
        'spec_rating': spec_rating
    }])

    # Encode categorical features
    sample_encoded = encoder.transform(
        sample_laptop[categorical_features]
    )

    # Select numerical features
    sample_numeric = sample_laptop[numerical_features]

    # Combine numerical and encoded features
    sample_final = hstack([
        sample_numeric,
        sample_encoded
    ])

    # Predict price
    predicted_price = model.predict(sample_final)

    st.success(
        f"Predicted Laptop Price: ₹{predicted_price[0]:,.2f}"
    )