# 🌱 Plant Disease Detection App

A simple web app to detect plant diseases using leaf images and a TensorFlow Lite model. Built with Streamlit.

## 🚀 Features
- Upload a plant leaf image
- Fast prediction using TFLite
- Simple web interface (no Android coding needed)

## 🧰 Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/plant-disease-app.git
   cd plant-disease-app

2. Install requirements:
```bash
    pip install -r requirements.txt

3. Run the app:
```bash
   streamlit run app.py

## 📁 Files

1. app.py – Main Streamlit app
2. model.tflite – Pretrained TFLite model
3. labels.txt – Output class names

## 🧠 Convert Keras to TFLite
```bash 
    python to_tflite.py

## 🌍 Deployment
Use Streamlit Cloud or any platform that supports Python.

MIT License | Made with ❤️ using Streamlit & TFLite
