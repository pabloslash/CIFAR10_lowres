 #######
# Developed by Pablo Tostado Marcos
# Last modified: Feb 15th 2018
#######


from __future__ import print_function
from data_loader import *
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
from  PIL import Image
import numpy as np
import random
import sys
sys.dont_write_bytecode = True
import IPython as IP


######################## Data Path ########################
cifar10_path = '/home/pablotostado/Desktop/PT/ML_Datasets/cifar10/'

######################## FUNCTIONS ########################

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

def get_accuracy(dataloader, net, classes,cuda=0):
    correct = 0
    total = 0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.cuda(cuda), labels.cuda(cuda)

        outputs = net(Variable(inputs))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return 100.0 * correct / total

def get_class_accuracy(dataloader, net, classes, cuda=0):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.cuda(cuda), labels.cuda(cuda)
        outputs = net(Variable(inputs))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    class_perc = []

    for i in range(10):
        class_perc.append(100.0 * class_correct[i] / class_total[i])
    return class_perc


######################## IMPORT DATA ####################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].



global batch_size
batch_size = 20

# Load Training + Validation
trainloader, validationloader = get_train_valid_loader(data_dir=cifar10_path,
                                                       batch_size=batch_size,
                                                       augment=True,
                                                       random_seed=1,
                                                       shuffle=False,
                                                       show_sample=False)
# Load Testing
testloader = get_test_loader(data_dir=cifar10_path,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


###
# Training: 1800 batches of 25 images (45000 images)
# Validation: 200 batches of 25 images (5000 images)
# Testing: 400 batches of 25 images (10000 images)
###


####################### ALTERNATE IMPORT ###########################3

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                           std=[0.229, 0.224, 0.225])
#
#
# trainset = torchvision.datasets.CIFAR10(root=cifar10_path, train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)
# #
# testset = torchvision.datasets.CIFAR10(root=cifar10_path, train=False,
#                                        transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)


# ####################### VISUALIZE IMAGES ############################################
#
# #
# ##### Print figure with 1 random image from each class
# train_labels = [] # Get labels
# for im in xrange(0, len(trainset)):
#     train_labels.append(trainset[im][1])
# train_labels = np.array(train_labels)
#
# fig = plt.figure(figsize=(8,3))
# for i in range(10):
#     ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
#     idx = np.where(train_labels==i)     # Find images with target label
#     idx = random.choice(idx[0])         # Pick random idx of current class
#     img = trainset[idx][0] #Take image
#     ax.set_title(classes[i])
#     imshow(img)
# plt.show(block = False)



######################## NET 4 -  ########################

# conv2(in_channels, out_channels, kernel, stride, padding)
# MaxPool2d(kernel, stride, padding)


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
####################### RUN NET ###################################

cuda = 0
model = 1



net = Net_cifar()
net.cuda(cuda)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
# optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
print('Defined Everything')


train_accuracy = []
test_accuracy = []
validation_accuracy = []

train_class_accuracy = []
test_class_accuracy = []
validation_class_accuracy = []


epochs = 150
for epoch in range(epochs):
    print (epoch)

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        # inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = Variable(inputs.cuda(cuda)), Variable(labels.cuda(cuda))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        # if i % 2000 == 1999:    # print every 50 mini-batches
        # print('[%d, %5d] loss: %.3f' %
        #       (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

    print('Epoch {} | Test acc: {}'.format((epoch + 1), get_accuracy(testloader, net, classes, cuda)))
    train_accuracy.append(get_accuracy(trainloader, net, classes, cuda))
    test_accuracy.append(get_accuracy(testloader, net, classes, cuda))
    validation_accuracy.append(get_accuracy(validationloader, net, classes, cuda))

    train_class_accuracy.append(get_class_accuracy(trainloader, net, classes, cuda))
    test_class_accuracy.append(get_class_accuracy(testloader, net, classes, cuda))
    validation_class_accuracy.append(get_class_accuracy(validationloader, net, classes, cuda))

print('test accuracy:\n')
print(get_accuracy(testloader, net, classes, cuda))

print('validation accuracy:\n')
print(get_accuracy(validationloader, net, classes, cuda))


import os

model_name = 'networks/networks_NOdropout/cifar10_NOdropout_0'+str(model+1)+'.pt'
save_dir = os.getcwd() + '/' + model_name

#SAVE
torch.save(net.state_dict(), save_dir)







######################### PLOTS ################################
'''
Plotting
'''

plt.style.use('ggplot')

'''
Total accuracy
'''
plt.figure()
plt.plot(range(epochs), train_accuracy, label='Train accuracy')
plt.plot(range(epochs), test_accuracy, label='Test accuracy')
plt.plot(range(epochs), validation_accuracy, label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Percent Accuracy')
plt.title('Training accuracy over: \n{} Iterations'.format(epochs), fontsize=16)
plt.legend(loc='lower right')
plt.show(block=False)



'''
Accuracy by class.
'''

f, axarr = plt.subplots(2, 5, figsize=(18,9))
for i in range(len(classes)):
    if int((i) / 5) > 0:
        row = 1
        col = i % 5
    else:
        row = 0
        col = i

    print(row, col)
    axarr[row, col].plot(range(len(train_class_accuracy)), list(np.array(train_class_accuracy)[:, i]), label='Train accuracy')
    axarr[row, col].plot(range(len(test_class_accuracy)), list(np.array(test_class_accuracy)[:, i]), label='Test accuracy')
    axarr[row, col].plot(range(len(validation_class_accuracy)), list(np.array(validation_class_accuracy)[:, i]), label='Validation accuracy')
    axarr[row, col].set_title('Accuracy for\nclass: {}'.format(classes[i]))

# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.suptitle('Accuracy By Class over {} Epochs'.format(len(train_accuracy)), fontsize=16)
plt.figlegend(loc = 'lower center', ncol=5, labelspacing=0. )
plt.show()
