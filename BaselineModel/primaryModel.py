import torch
import torch.nn as nn
import torch.nn.functional as F

class YogaClassfier(nn.Module):
    def __init__(self, num_classes = 107): #number of yoga poses
        super(YogaClassfier, self).__init__()

        #Convolutional layers (increasing numbers of filters)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=64, 
                               kernel_size = 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels=128, 
                               kernel_size = 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels=256, 
                               kernel_size = 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels=512, 
                               kernel_size = 3, stride=1, padding=1)
        
        #Max-pooling layers 
        #reducing spatial dimensinons by half after each layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #Fully connected layers
        #assume dimension is 640 x 480
        self.fc1 = nn.Linear(in_features=512 * 40 * 30, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        #Convlutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        #Flatten the feature map
        x = x.view(-1, 512*40*30)

        #Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) #handled by loss fuction 

        return x