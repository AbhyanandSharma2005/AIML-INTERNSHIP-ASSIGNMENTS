import os
import numpy as np
from model import create_weather_cnn

def generate_mock_data(num_samples=100, image_size=(64, 64, 3), num_classes=4):
    """
    Generates random mock image data and labels for testing the pipeline.
    Replace this with real data loading (e.g., tf.keras.preprocessing.image_dataset_from_directory).
    """
    print("Generating mock dataset...")
    X = np.random.rand(num_samples, *image_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return X, y

def main():
    # 1. Define classes and parameters
    weather_classes = ['Sunny', 'Rainy', 'Cloudy', 'Snowy']
    image_shape = (64, 64, 3)
    num_classes = len(weather_classes)
    
    # 2. Get Data
    # For a real project, point to your dataset directory
    X_train, y_train = generate_mock_data(num_samples=200, image_size=image_shape, num_classes=num_classes)
    X_val, y_val = generate_mock_data(num_samples=40, image_size=image_shape, num_classes=num_classes)
    
    # 3. Create Model
    print("Initializing model...")
    model = create_weather_cnn(input_shape=image_shape, num_classes=num_classes)
    model.summary()
    
    # 4. Train Model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=5,
        validation_data=(X_val, y_val),
        batch_size=16
    )
    
    # 5. Save Model
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "weather_vision_model.keras")
    model.save(model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    main()
