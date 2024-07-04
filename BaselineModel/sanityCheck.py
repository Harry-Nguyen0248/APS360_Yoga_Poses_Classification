import torchvision.transforms as transforms
import torch
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from helperFunctions import get_model_name

import os
import shutil
import random
from pathlib import Path

def create_small_dataset(source_dir, target_dir, num_images=10):
    """
    Creates a small dataset by randomly selecting a specified number of images 
    from each class folder in the source directory and copying them to a new 
    target directory.

    Args:
        source_dir (str): Path to the source directory where the original dataset is stored.
        target_dir (str): Path to the target directory where the small dataset will be created.
        num_images (int): Number of images to randomly select from each class folder.
    """
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # List all folders in the source directory
    yoga_poses = [folder for folder in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, folder))]
    
    # Loop through each yoga pose directory in the source directory
    for pose in yoga_poses:
        pose_path = os.path.join(source_dir, pose)
        images = os.listdir(pose_path)
        
        # Randomly select a specified number of images
        selected_images = random.sample(images, min(len(images), num_images))
        
        # Create a corresponding folder in the target directory
        target_pose_path = os.path.join(target_dir, pose)
        os.makedirs(target_pose_path, exist_ok=True)
        
        # Copy the selected images to the target directory
        for image in selected_images:
            src_path = os.path.join(pose_path, image)
            dest_path = os.path.join(target_pose_path, image)
            shutil.copy2(src_path, dest_path)

    print(f"Small dataset created at {target_dir}")


def get_small_data_loader(batch_size):

    transform = transforms.Compose([
        transforms.Resize((640,480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    #Loadd small dataset
    small_dataset = datasets.ImageFolder('/Users/harrynguyen/Documents/GitHub/APS360_Yoga_Poses_Classification/BaselineModel/small_dataset',
                                         transform=transform)
    small_loader = torch.utils.data.DataLoader(small_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True,
                                                num_workers=1)
    return small_loader

def train_on_small_dataset(net, batch_size = 8, learning_rate=0.001, num_epochs=20):
    # Fixed PyTorch random seed for reproducibility
    torch.manual_seed(1000)
    
    # Obtain the small dataset loader
    small_loader = get_small_data_loader(batch_size=batch_size)
    
    # Define the Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    
    checkpoint_dir = '/Users/harrynguyen/Documents/GitHub/APS360_Yoga_Poses_Classification/BaselineModel/small_checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)

    net.train()
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        
        for i, data in enumerate(small_loader, 0):
            inputs, labels = data
            labels = labels.long()  # Convert labels to Long type
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            corr = (outputs.argmax(dim=1) != labels).sum().item()
            total_train_err += corr
            total_train_loss += loss.item()
            total_epoch += len(labels)
        
        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i + 1)
        
        print(f"Epoch {epoch + 1}: Train err: {train_err[epoch]}, Train loss: {train_loss[epoch]}")
        
        # Check if training error is zero (i.e., model has overfitted)
        if train_err[epoch] == 0.0:
            print(f"Model has memorized the dataset in {epoch + 1} epochs.")
            break
    
        # model_path = get_model_name(net.name, batch_size, learning_rate, epoch)
        # torch.save(net.state_dict(), model_path)
    print('Finished Training on Small Dataset')

    #Save metrics to CSV
    model_path = get_model_name(net.name, batch_size, learning_rate, epoch)
    train_err_path = os.path.join(checkpoint_dir, f"{model_path}_train_err.csv")
    train_loss_path = os.path.join(checkpoint_dir, f"{model_path}_train_loss.csv")
    epochs = np.arange(1, num_epochs + 1)
    np.savetxt(train_err_path, train_err, delimiter= ',')
    np.savetxt(train_loss_path, train_loss, delimiter= ',')

    return net, train_err, train_loss

