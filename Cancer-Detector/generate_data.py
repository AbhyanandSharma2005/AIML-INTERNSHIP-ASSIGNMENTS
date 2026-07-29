import os
import numpy as np
from PIL import Image

def generate_synthetic_data(num_samples, output_dir):
    """
    Generates synthetic dataset for cancer detection.
    Healthy: Pinkish noise
    Cancerous: Pinkish noise with a dark blob
    """
    classes = ['healthy', 'cancerous']
    
    for cls in classes:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)
        
    for i in range(num_samples):
        # Base tissue: random pinkish noise
        # R: 200-255, G: 100-150, B: 150-200
        img_array = np.zeros((64, 64, 3), dtype=np.uint8)
        img_array[:, :, 0] = np.random.randint(200, 255, (64, 64))
        img_array[:, :, 1] = np.random.randint(100, 150, (64, 64))
        img_array[:, :, 2] = np.random.randint(150, 200, (64, 64))
        
        is_cancerous = np.random.choice([True, False])
        cls = 'cancerous' if is_cancerous else 'healthy'
        
        if is_cancerous:
            # Add a dark blob (tumor)
            center_x = np.random.randint(15, 49)
            center_y = np.random.randint(15, 49)
            radius = np.random.randint(5, 12)
            
            y, x = np.ogrid[-center_y:64-center_y, -center_x:64-center_x]
            mask = x*x + y*y <= radius*radius
            
            # Darken the blob area
            img_array[mask, 0] = np.random.randint(50, 100, img_array[mask, 0].shape)
            img_array[mask, 1] = np.random.randint(20, 70, img_array[mask, 1].shape)
            img_array[mask, 2] = np.random.randint(50, 100, img_array[mask, 2].shape)
            
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, cls, f"sample_{i}.png"))
        
    print(f"Generated {num_samples} samples in '{output_dir}'.")

if __name__ == "__main__":
    # Generate Train and Test sets
    np.random.seed(42)
    generate_synthetic_data(800, "dataset/train")
    generate_synthetic_data(200, "dataset/test")
