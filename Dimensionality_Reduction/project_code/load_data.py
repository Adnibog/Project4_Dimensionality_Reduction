import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder, target_size=(92, 112)):
    images = []
    labels = []
    label_map = {}
    current_label = 0
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.pgm')

    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path) and label != 'figures':
            if label not in label_map:
                label_map[label] = current_label
                current_label += 1
            image_count = 0
            for filename in os.listdir(label_path):
                if filename.lower().endswith(valid_extensions):
                    img_path = os.path.join(label_path, filename)
                    try:
                        img = Image.open(img_path).convert('L')  # Convert to grayscale
                        img = img.resize(target_size)  # Resize image
                        img_array = np.array(img)
                        images.append(img_array)
                        labels.append(label_map[label])
                        image_count += 1
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
            if image_count != 10:
                raise ValueError(f"Expected 10 images for subject {label}, but found {image_count}.")
    if len(images) != 400:
        raise ValueError(f"Expected 400 images, but loaded {len(images)} images.")
    print(f"Loaded {len(images)} images from {folder}")
    return np.array(images), np.array(labels)

