# ☁️ WeatherVision AI

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-FF6F00?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Production%20Ready-black?logo=flask)
![License](https://img.shields.io/badge/license-MIT-green)

**WeatherVision AI** is an enterprise-grade, lightweight Convolutional Neural Network (CNN) application designed for real-time weather classification. Built with a focus on minimalistic architecture and high performance, WeatherVision provides a seamless pipeline from model training to a production-ready web interface.

---

## ✨ Key Features

- **🧠 Robust AI Engine:** A custom-built, optimized CNN architecture leveraging TensorFlow/Keras capable of distinguishing complex weather patterns (Sunny, Rainy, Cloudy, Snowy).
- **🚀 Production-Ready Serving:** Fully integrated with `Waitress`, a production-grade WSGI server optimized for Windows environments, capable of handling concurrent requests efficiently.
- **🎨 Premium UI/UX:** A stunning frontend interface featuring modern glassmorphism design, fluid micro-animations, drag-and-drop file uploads, and graceful error handling.
- **⚙️ Plug-and-Play Architecture:** Highly modular backend design that allows easy swapping of datasets, retraining of models, and seamless API integrations.

## 🛠️ Technology Stack

- **Machine Learning Core:** TensorFlow, Keras, NumPy, Pillow
- **Backend Infrastructure:** Python, Flask, Werkzeug, Waitress
- **Frontend Interface:** HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (Fetch API)

## 📂 Project Structure

```text
WeatherVision-App/
│
├── app.py                 # Development Flask application & API routes
├── serve.py               # Production Waitress WSGI server entry point
├── model.py               # CNN architecture definition
├── train.py               # Automated dataset generation & model training script
├── predict.py             # CLI inference and prediction script
├── requirements.txt       # Project dependencies
│
├── saved_models/          # Directory containing compiled .keras model binaries
├── static/
│   └── uploads/           # Ephemeral storage for user-uploaded images
└── templates/
    └── index.html         # Premium frontend UI template
```

## 🚀 Getting Started

### 1. Environment Setup
Clone the repository and install the required dependencies:
```bash
# It is recommended to use a virtual environment
pip install -r requirements.txt
```

### 2. Model Training
Before starting the server, initialize and train the CNN model:
```bash
python train.py
```
*Note: The default script utilizes a mock dataset for immediate deployment. For real-world use, replace the mock generator in `train.py` with `tf.keras.preprocessing.image_dataset_from_directory` pointing to your local image folders.*

### 3. Launching the Application

**Production Mode (Recommended)**
Run the application using the Waitress WSGI server for optimal stability and multithreading:
```bash
python serve.py
```
*The service will be available at `http://localhost:8080`*

**Development Mode**
For debugging and active development:
```bash
python app.py
```

## 🖥️ Usage Guide

1. Navigate to `http://localhost:8080` in your web browser.
2. Drag and drop a weather-related image into the designated upload area, or click to browse your file system.
3. Click **Analyze Weather**.
4. The proprietary WeatherVision engine will process the image and immediately return the classified weather condition along with the AI's confidence interval.

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 🤝 Support
For enterprise support, integration requests, or general inquiries, please contact the development team.
