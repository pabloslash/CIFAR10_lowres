import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import IPython as IP
from my_helper_pytorch import *

class Net_cifar(nn.Module):
    def __init__(self, num_classes=10):
        super(Net_cifar, self).__init__()
        #convolutional layers
        #the input is 3 RBM
        self.conv1 = nn.Conv2d(3, 96, 3, padding = 1) #padding = 1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding = 1) #padding = 1)
        self.conv3 = nn.Conv2d(96, 192, 3, padding = 1)
        self.conv4 = nn.Conv2d(192, 192, 3,padding = 1)
        # self.conv5 = nn.Conv2d(96, 96, 3, stride = 2)
        self.conv5 = nn.Conv2d(192, 192, 1, padding = 1)
        self.conv6 = nn.Conv2d(192, 256, 1, padding = 1)

        #batch normalization
        self.batch96 = nn.BatchNorm2d(96, momentum=0.6)
        self.batch192 = nn.BatchNorm2d(192, momentum=0.6)
        self.batch256 = nn.BatchNorm2d(256, momentum=0.6)

        self.dropOut = nn.Dropout2d(p=0.4)

        #max pooling
        self.mp = nn.MaxPool2d(2, stride=2) #2X2 with stride 2

        self.fc = nn.Linear(256*7*7, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):

        x = self.batch96(F.relu(self.conv1(x)))
        x = self.mp(self.batch96(F.relu(self.conv2(x))))

        x = self.batch192(F.relu(self.conv3(x)))
        x = self.mp(self.batch192(F.relu(self.conv4(x))))

        x = self.dropOut(x)

        x = self.batch192(F.relu(self.conv4(x)))
        x = self.mp(self.batch192(F.relu(self.conv5(x))))

        x = self.batch256(F.relu(self.conv6(x)))
        x = x.view(x.size(0), -1) #flatten for fc
        #print (x.size(1))
        x = F.relu(self.fc(x))

        x = self.dropOut(x)

        x = self.fc2(x)

        # x = self.dropOut(x)
        return F.log_softmax(x,dim=1) #softmax classifier
