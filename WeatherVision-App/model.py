import tensorflow as tf
from tensorflow.keras import layers, models

def create_weather_cnn(input_shape=(64, 64, 3), num_classes=4):
    """
    Creates a minimalistic CNN for weather classification.
    Classes could be: Sunny, Rainy, Cloudy, Snowy.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
