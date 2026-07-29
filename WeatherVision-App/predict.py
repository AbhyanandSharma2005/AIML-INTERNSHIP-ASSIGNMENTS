import numpy as np
import tensorflow as tf
from PIL import Image

def load_and_preprocess_image(image_path, target_size=(64, 64)):
    """
    Loads an image from path and preprocesses it for the model.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def main():
    weather_classes = ['Sunny', 'Rainy', 'Cloudy', 'Snowy']
    model_path = "saved_models/weather_vision_model.keras"
    
    # Check if model exists
    try:
        model = tf.keras.models.load_model(model_path)
    except OSError:
        print(f"Error: Model not found at {model_path}. Please run train.py first.")
        return
        
    # Generate a dummy image to test if no real image is provided
    print("Generating a test image (dummy_test_image.jpg)...")
    dummy_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    dummy_img.save("dummy_test_image.jpg")
    
    image_path = "dummy_test_image.jpg"
    print(f"Predicting for image: {image_path}")
    
    img_array = load_and_preprocess_image(image_path)
    if img_array is not None:
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        predicted_weather = weather_classes[predicted_class_idx]
        print("\n--- Prediction Results ---")
        print(f"Weather: {predicted_weather}")
        print(f"Confidence: {confidence*100:.2f}%")
        print("--------------------------")

if __name__ == "__main__":
    main()
