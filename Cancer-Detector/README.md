# Simple Cancer Detection CNN

This is a small, lightweight project demonstrating how to build a Convolutional Neural Network (CNN) for binary classification (Cancerous vs. Healthy) using TensorFlow and Keras.

## Project Structure
- `generate_data.py`: Creates a synthetic dataset of 64x64 images. Healthy tissue is represented by pinkish noise, while cancerous tissue has an additional dark spot (tumor).
- `train.py`: Defines and trains a simple CNN model on the generated dataset. It outputs the model as `cancer_detector_model.h5` and a training history plot `training_history.png`.
- `predict.py`: Loads the trained model, runs inference on a few sample images, and generates a visual output `sample_predictions.png` showing the true labels and predicted results.

## Requirements
To run this project, you need the following Python packages:
- `tensorflow`
- `numpy`
- `pillow`
- `matplotlib`

Install them using:
```bash
pip install -r requirements.txt
```

## How to Run

1. **Generate the dataset**:
   ```bash
   python generate_data.py
   ```
   This will create a `dataset` folder with `train` and `test` subdirectories.

2. **Train the model**:
   ```bash
   python train.py
   ```
   This will train the CNN and save `cancer_detector_model.h5` and `training_history.png`.

3. **Run predictions (Sample Outputs)**:
   ```bash
   python predict.py
   ```
   This will select a few images from the test set, predict their class, and save the result as `sample_predictions.png`.
