import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def load_and_preprocess_image(path):
    img = Image.open(path).convert('RGB')
    img_resized = img.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    return img_array

def main():
    model_path = 'cancer_detector_model.h5'
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Please train the model first.")
        return
        
    model = tf.keras.models.load_model(model_path)
    
    test_dir = 'dataset/test'
    if not os.path.exists(test_dir):
        print("Dataset not found. Please run generate_data.py first.")
        return
        
    # Get a few sample images
    cancerous_dir = os.path.join(test_dir, 'cancerous')
    healthy_dir = os.path.join(test_dir, 'healthy')
    
    cancerous_samples = [os.path.join(cancerous_dir, f) for f in os.listdir(cancerous_dir)[:2]]
    healthy_samples = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir)[:2]]
    
    all_samples = cancerous_samples + healthy_samples
    
    plt.figure(figsize=(12, 6))
    
    for i, img_path in enumerate(all_samples):
        img_array = load_and_preprocess_image(img_path)
        
        # Predict (model outputs probability of being class 1 - which is healthy)
        # Class 0: cancerous, Class 1: healthy
        prediction = model.predict(np.expand_dims(img_array, axis=0))[0][0]
        
        predicted_class = "Healthy" if prediction >= 0.5 else "Cancerous"
        confidence = prediction if prediction >= 0.5 else (1 - prediction)
        
        true_label = os.path.basename(os.path.dirname(img_path))
        
        plt.subplot(1, 4, i+1)
        plt.imshow(img_array)
        color = 'green' if predicted_class.lower() == true_label.lower() else 'red'
        plt.title(f"True: {true_label.capitalize()}\nPred: {predicted_class}\nConf: {confidence:.2f}", color=color)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    print("Predictions saved to sample_predictions.png")

if __name__ == '__main__':
    main()
