import random
import os
import shutil
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import DataLoader

def prepare_datasets(data_dir, train_dir, val_dir, test_dir, test_size=0.30, val_test_split=0.50, seed=30):
    random.seed(seed)
    
    # Ensure base directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all individuals (directories) within the data directory
    individuals = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

    # Split individuals into train, val, test sets
    train_individuals, temp_individuals = train_test_split(individuals, test_size=test_size, random_state=seed)
    val_individuals, test_individuals = train_test_split(temp_individuals, test_size=val_test_split, random_state=seed)

    # Define a function to copy images to new directories
    def copy_images(individuals, source_dir, dest_dir):
        for individual in individuals:
            individual_dir = os.path.join(source_dir, individual)
            if os.path.isdir(individual_dir):
                shutil.copytree(individual_dir, os.path.join(dest_dir, individual), dirs_exist_ok=True)

    # Copy individuals into respective directories
    copy_images(train_individuals, data_dir, train_dir)
    copy_images(val_individuals, data_dir, val_dir)
    copy_images(test_individuals, data_dir, test_dir)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 pixels
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize
    ])

    # Load datasets using ImageFolder
    train_dataset = ImageFolder(train_dir, transform=transform)
    val_dataset = ImageFolder(val_dir, transform=transform)
    test_dataset = ImageFolder(test_dir, transform=transform)

    # Convert datasets to numpy arrays for Random Forest
    def dataset_to_numpy(dataset):
        images, labels = [], []
        for img, label in DataLoader(dataset, batch_size=1):
            img = img.view(-1).numpy()  # Flatten the image
            images.append(img)
            labels.append(label.numpy()[0])
        return np.array(images), np.array(labels)

    X_train, y_train = dataset_to_numpy(train_dataset)
    X_val, y_val = dataset_to_numpy(val_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)

    # Output information about the datasets
    print(f"Number of training images: {len(y_train)}")
    print(f"Number of validation images: {len(y_val)}")
    print(f"Number of test images: {len(y_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test
