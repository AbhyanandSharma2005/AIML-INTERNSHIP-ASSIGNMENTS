import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

def build_model():
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Binary classification
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    base_dir = 'dataset'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    
    # Check if dataset exists
    if not os.path.exists(train_dir):
        print("Dataset not found. Please run generate_data.py first.")
        return

    # Data generators
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(64, 64),
        batch_size=32,
        class_names=['cancerous', 'healthy'], # Note: cancerous=0, healthy=1
        color_mode='rgb'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(64, 64),
        batch_size=32,
        class_names=['cancerous', 'healthy'],
        color_mode='rgb'
    )
    
    # Normalize pixel values
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    model = build_model()
    model.summary()
    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )
    
    # Save the model
    model.save('cancer_detector_model.h5')
    print("Model saved to cancer_detector_model.h5")
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved to training_history.png")

if __name__ == '__main__':
    main()
