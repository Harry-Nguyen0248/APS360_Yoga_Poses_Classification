import helperFunctions
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time

def train_model(net, batch_size, learning_rate, num_epochs, checkpoint_frequency = 5):
    torch.manual_seed(1000)
    data_dir = "/Users/harrynguyen/Documents/GitHub/APS360_Yoga_Poses_Classification/Dataset/dataset"
    filenames = helperFunctions.get_folder_names(data_dir)

    train_loader, val_loader, test_loader, classes = helperFunctions.get_data_loader(
            target_classes = filenames, batch_size=batch_size)
    
    #Define the Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)

    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    # Create directories for checkpoints if they don't exist
    checkpoint_dir = '/Users/harrynguyen/Documents/GitHub/APS360_Yoga_Poses_Classification/BaselineModel/model_checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_time = time.time()
    net.train()
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = labels.long()  # Convert labels to the required format
            
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
        val_err[epoch], val_loss[epoch] = helperFunctions.evaluate(net, val_loader, criterion)
        print(("Epoch {}: Train err: {}, Train loss: {} |"+
               "Validation err: {}, Validation loss: {}").format(
                   epoch + 1,
                   train_err[epoch],
                   train_loss[epoch],
                   val_err[epoch],
                   val_loss[epoch]))
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_frequency == 0:
            # save_files(net.name, batch_size, learning_rate, epoch, checkpoint_dir,
            #            train_err, train_loss, val_err, val_loss)
            print(f"Model checkpoint saved at {epoch + 1}th epoch")
    
    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")
    
    # # Save metrics to CSV
    model_path = helperFunctions.get_model_name(net.name, batch_size, learning_rate, epoch)
    
    train_err_path = os.path.join(checkpoint_dir, f"{model_path}_train_err.csv")
    train_loss_path = os.path.join(checkpoint_dir, f"{model_path}_train_loss.csv")
    np.savetxt(train_err_path, train_err, delimiter=',')
    np.savetxt(train_loss_path, train_loss, delimiter=',')

    val_err_path = os.path.join(checkpoint_dir, f"{model_path}_val_err.csv")
    val_loss_path = os.path.join(checkpoint_dir, f"{model_path}_val_loss.csv")
    np.savetxt(val_err_path, val_err, delimiter=',')
    np.savetxt(val_loss_path, val_loss, delimiter=',')
    
    return net, train_err, train_loss, val_err, val_loss