import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set up the Streamlit app interface
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image and get a diagnosis.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))  # Match model's expected input size
    input_data = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get prediction result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)
    predicted_label = labels[predicted_index]
    confidence = np.max(output_data)

    # Show results
    st.success(f"**Prediction:** {predicted_label}")
    st.info(f"**Confidence:** {confidence:.2f}")
