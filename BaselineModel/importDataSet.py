import os
import random
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_datasets(data_dir, train_dir, val_dir, test_dir, test_size=0.30, val_test_split=0.50, seed=42):
    random.seed(seed)

    # Ensure base directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    #Get all images and labels
    all_images = []
    labels = []

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                if os.path.isfile(img_path):
                    all_images.append(img_path)
                    labels.append(label)

    # Convert lists to arrays for convenience
    all_images = np.array(all_images)
    labels = np.array(labels)

    # Create directories for each label in the train, val, and test directories
    for label in np.unique(labels):
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
        os.makedirs(os.path.join(val_dir, label), exist_ok=True)
        os.makedirs(os.path.join(test_dir, label), exist_ok=True)

    # Split the dataset within each label
    # 70-30 for train_imgs and temp_imgs
    # 50-50 for val and test images 
    for label in np.unique(labels):
        label_images = all_images[labels == label]
        train_imgs, temp_imgs = train_test_split(label_images, test_size=test_size, random_state=seed)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=val_test_split, random_state=seed)

        # Copy images to their respective directories
        for img_path in train_imgs:
            shutil.copy(img_path, os.path.join(train_dir, label))
        for img_path in val_imgs:
            shutil.copy(img_path, os.path.join(val_dir, label))
        for img_path in test_imgs:
            shutil.copy(img_path, os.path.join(test_dir, label))

    print("Dataset preparation complete.")
