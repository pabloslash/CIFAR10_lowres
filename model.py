import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import IPython as IP
from my_helper_pytorch import *

class Net_cifar(nn.Module):
    def __init__(self):
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
        self.batch96 = nn.BatchNorm1d(96)
        self.batch192 = nn.BatchNorm1d(192)
        self.batch256 = nn.BatchNorm1d(256)

        # self.dropOut = nn.Dropout2d(p=0.4)

        #max pooling
        self.mp = nn.MaxPool2d(2, stride=2) #2X2 with stride 2

        self.fc = nn.Linear(256*7*7, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):

        x = F.relu(self.batch96(self.conv1(x)))
        x = self.mp(F.relu(self.batch96(self.conv2(x))))
        x = F.relu(self.batch192(self.conv3(x)))
        x = self.mp(F.relu(self.batch192(self.conv4(x))))

        # x = self.dropOut(x)

        x = F.relu(self.batch192(self.conv4(x)))
        x = self.mp(F.relu(self.batch192(self.conv5(x))))
        x = F.relu(self.batch256(self.conv6(x)))

        # IP.embed()

        x = x.view(x.size(0), -1) #flatten for fc
        #print (x.size(1))
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        # x = self.fc2(x)
        # x = self.fc2(x)
        return F.log_softmax(x) #softmax classifier
