import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import os
import torch
from torch.utils.data import DataLoader


###############################################################################
#Data loading
def get_folder_names(dataset_path):
    """
    This function returns a list of folder names contained within the given dataset directory.

    Parameters:
        dataset_path (str): The path to the dataset directory.
    
    Returns:
        list: A list of folder names representing different yoga pose types.
    """
    # Check if the given path is a directory
    if not os.path.isdir(dataset_path):
        raise ValueError("Provided path is not a directory")
    
    # List all entries in the directory specified by path
    folder_names = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    return folder_names

###############################################################################
# Data Loading

def get_relevant_indices(dataset, target_classes):
    """ Return the indices for datapoints in the dataset that belong to the
    desired target classes, a subset of all possible classes.

    Args:
        dataset: Dataset object with an attribute 'classes' that includes all class names
        target_classes: A list of strings denoting the name of desired classes
    
    Returns:
        indices: list of indices that have labels corresponding to one of the target classes
    """
    indices = []
    for i in range(len(dataset)):
        _, label_index = dataset[i]  # Get the label index directly from the dataset
        if dataset.classes[label_index] in target_classes:
            indices.append(i)
    return indices


def get_data_loader(target_classes, batch_size):
    """ Loads images, splits the data into training, validation, and testing datasets. Returns data loaders.

    Args:
        target_classes: A list of strings denoting the name of the desired classes
        batch_size: An integer representing the number of samples per batch
    
    Returns:
        train_loader, val_loader, test_loader: iterable datasets organized by batch size
        classes: A list of class names in the dataset
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load training data
    trainset = datasets.ImageFolder('/Users/harrynguyen/Documents/GitHub/APS360_Yoga_Poses_Classification/BaselineModel/yoga_train', 
                                    transform=transform)
    relevant_indices = get_relevant_indices(trainset, target_classes)
    np.random.seed(1000)
    np.random.shuffle(relevant_indices)
    split = int(0.8 * len(relevant_indices))

    # Create training and validation loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(relevant_indices[:split]), num_workers=1)
    val_loader = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(relevant_indices[split:]), num_workers=1)

    # Load testing data
    testset = datasets.ImageFolder('/Users/harrynguyen/Documents/GitHub/APS360_Yoga_Poses_Classification/BaselineModel/yoga_test', 
                                   transform=transform)
    test_indices = get_relevant_indices(testset, target_classes)
    test_loader = DataLoader(testset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices), num_workers=1)

    return train_loader, val_loader, test_loader, trainset.classes


###############################################################################
# Training
def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

def normalize_label(labels):
    """
    Given a tensor containing 2 possible values, normalize this to 0/1

    Args:
        labels: a 1D tensor containing two possible scalar values
    Returns:
        A tensor normalize to 0/1 value
    """
    max_val = torch.max(labels)
    min_val = torch.min(labels)
    norm_labels = ((labels - min_val)/(max_val - min_val))
    return norm_labels.long()

def evaluate(net, loader, criterion):
    """ Evaluate the network on the validation set.

     Args:
         net: PyTorch neural network object
         loader: PyTorch data loader for the validation set
         criterion: The loss function
     Returns:
         err: A scalar for the avg classification error over the validation set
         loss: A scalar for the average loss function over the validation set
     """
    # net.eval()
    # total_loss = 0.0
    # total_err = 0.0
    # total_epoch = 0
    # with torch.no_grad():
    #     for data in loader:
    #         inputs, labels = data
    #         labels = normalize_label(labels)
    #         labels = labels.long() 
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels.float())
    #         corr = (outputs.argmax(dim=1) != labels).sum().item()
    #         total_err += corr
    #         total_loss += loss.item()
    #         total_epoch += len(labels)
    # err = float(total_err) / total_epoch
    # loss = float(total_loss) / len(loader)
    # return err, loss

    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        labels = normalize_label(labels)  # Convert labels to 0/1
        labels = labels.long()

        # labels = labels.unsqueeze(1)

        outputs = net(inputs)
        loss = criterion(outputs, labels.float())
        corr = (outputs > 0.0).squeeze().long() != labels
        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)
    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err, loss

def save_files(model_name, batch_size, learning_rate, epoch, checkpoint_dir, 
               train_err, train_loss, val_err=None, val_loss=None):
    """
    Saves training and optionally validation error and loss to CSV files.

    Args:
        model_name (str): Name of the model.
        batch_size (int): Batch size used during training.
        learning_rate (float): Learning rate used during training.
        epoch (int): Current epoch number.
        checkpoint_dir (str): Directory to save the checkpoint files.
        train_err (np.array): Array of training errors.
        train_loss (np.array): Array of training losses.
        val_err (np.array, optional): Array of validation errors.
        val_loss (np.array, optional): Array of validation losses.
    """
    model_path = get_model_name(model_name, batch_size, learning_rate, epoch)
    
    train_err_path = os.path.join(checkpoint_dir, f"{model_path}_train_err.csv")
    train_loss_path = os.path.join(checkpoint_dir, f"{model_path}_train_loss.csv")
    np.savetxt(train_err_path, train_err, delimiter=',')
    np.savetxt(train_loss_path, train_loss, delimiter=',')
    
    if val_err is not None and val_loss is not None:
        val_err_path = os.path.join(checkpoint_dir, f"{model_path}_val_err.csv")
        val_loss_path = os.path.join(checkpoint_dir, f"{model_path}_val_loss.csv")
        np.savetxt(val_err_path, val_err, delimiter=',')
        np.savetxt(val_loss_path, val_loss, delimiter=',')

###############################################################################
# Training Curve
def plot_training_curve(path, checkpoint_dir, smalldata = False):
    import matplotlib.pyplot as plt
    train_err_path = os.path.join(checkpoint_dir, f"{path}_train_err.csv")
    train_loss_path = os.path.join(checkpoint_dir, f"{path}_train_loss.csv")
    train_err = np.loadtxt(train_err_path, delimiter= ',')
    train_loss = np.loadtxt(train_loss_path, delimiter= ',')
    if not smalldata:
        val_err_path = os.path.join(checkpoint_dir, f"{path}_val_err.csv")
        val_loss_path = os.path.join(checkpoint_dir, f"{path}_val_loss.csv")
        val_err = np.loadtxt(val_err_path, delimiter= ',')
        val_loss = np.loadtxt(val_loss_path, delimiter= ',')
        plt.title("Train vs Validation Error")
        n = len(train_err) # number of epochs
        plt.plot(range(1,n+1), train_err, label="Train")
        plt.plot(range(1,n+1), val_err, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.legend(loc='best')
        plt.show()

        plt.title("Train vs Validation Loss")
        plt.plot(range(1,n+1), train_loss, label="Train")
        plt.plot(range(1,n+1), val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='best')
        plt.show()
    else:
        plt.title("Train Error and Train Loss")
        n = len(train_err)
        plt.plot(range(1,n+1), train_err, label="Train Error")
        plt.plot(range(1,n+1), train_loss, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Value Ratio")
        plt.legend(loc='best')
        plt.show()
