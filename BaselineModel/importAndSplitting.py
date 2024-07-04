import random
import os
import shutil
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

def prepare_datasets(data_dir, train_dir, val_dir, test_dir, test_size=0.30, val_test_split=0.50, seed=30):
    """
    Prepare datasets by splitting into training, validation, and test sets and copying them into respective directories.

    Parameters:
        data_dir (str): Directory containing the original dataset.
        train_dir (str): Directory to store training data.
        val_dir (str): Directory to store validation data.
        test_dir (str): Directory to store test data.
        test_size (float): Proportion of the dataset to include in the test split (relative to the entire dataset).
        val_test_split (float): Proportion of the test set to include in the validation split.
        seed (int): Random seed for reproducibility.
    """
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

    # Load datasets using ImageFolder
    train_dataset = ImageFolder(train_dir)
    val_dataset = ImageFolder(val_dir)
    test_dataset = ImageFolder(test_dir)

    # Output information about the datasets
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    print(f"Number of test images: {len(test_dataset)}")

