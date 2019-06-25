#######
# Developed by Pablo Tostado Marcos
# Last modified: Feb 15th 2018
#######

# TEST

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
from model_cifar10 import *
import os
import datetime
import time
import torch.nn as nn


####################################################################
###### Initialize Net & variables:
####################################################################

cifar10 = False # False if you desire to load CIFAR-100
num_classes=10 if (cifar10) else 100

cuda = 0
net = Net_cifar(num_classes=num_classes)
if torch.cuda.is_available():
    net.cuda(cuda)

model = 1
batch_size = 20
lr = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
# optimizer = optim.SGD(net.parameters(), lr=lr)#, momentum=0.9)

######################## IMPORT DATA ####################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
cifar_path = '/home/pablotostado/Desktop/PT/ML_Datasets/cifar10/' if (cifar10) else '/home/pablotostado/Desktop/PT/ML_Datasets/cifar100/'
trainloader, validationloader, testloader, classes = load_cifar(cifar_path, batch_size, cifar10=cifar10, augment=False)



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



####################### RUN NET ###################################
print('Net Initialized')


def train(ep):
    train_accuracy = []
    test_accuracy = []
    validation_accuracy = []

    train_class_accuracy = []
    test_class_accuracy = []
    validation_class_accuracy = []

    for e in range(ep):
        # net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            # inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = Variable(inputs.cuda(cuda)), Variable(labels.cuda(cuda))

            # zero the parameter gradients
            # print(optimizer)
            optimizer.zero_grad()

            # Stochastic binarization of weights:
            for layer in net.children():
                if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
                    layer.weight.data = binarize_and_stochRound(layer.weight.data)


            # DITHER.
            # This dithers convolutional & fully connected.
            layers, count = [], 0
            for layer in net.children():
                if (isinstance(layer, nn.Linear)): #or isinstance(layer, nn.Conv2d) or
                    count+=1
                    # if (count == 4):
                        # IP.embed()
                    layers.append(layer.weight.data)
                    # layer.weight.data = weight_dithering(layer.weight.data, 40, cuda=cuda, dith_levels=1)
                    layer.weight.data = fullPrec_grid_dithering(layer.weight.data)

            # forward + backward + optimize

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Restore UNDITHER layers and update
            l, count = 0, 0
            for layer in net.children():
                if (isinstance(layer, nn.Linear)): # or isinstance(layer, nn.Conv2d)
                    count+=1
                    # if (count == 4):
                        # IP.embed()
                    layer.weight.data = layers[l]
                    l += 1


            optimizer.step()
            # running_loss += loss.data[0]
            # running_loss = 0.0

        net.eval()
        train_accuracy.append(get_accuracy(trainloader, net, cuda))
        test_accuracy.append(get_accuracy(testloader, net, cuda))
        validation_accuracy.append(get_accuracy(validationloader, net, cuda))

        # train_class_accuracy.append(get_class_accuracy(trainloader, net, classes, cuda))
        # test_class_accuracy.append(get_class_accuracy(testloader, net, classes, cuda))
        # validation_class_accuracy.append(get_class_accuracy(validationloader, net, classes, cuda))
        print('Epoch {} | Test acc: {} | Validation Accuracy {} | Train Accuracy {}'.format(e+1, test_accuracy[-1], validation_accuracy[-1], train_accuracy[-1]))

    print('test accuracy:\n')
    print(get_accuracy(testloader, net, classes, cuda))

    print('validation accuracy:\n')
    print(get_accuracy(validationloader, net, classes, cuda))

'''NOT ADAPTED. FIX THIS FUNCTION TO SAVE MODELS'''
def test():
    net.eval()
    test_performance = get_accuracy(testloader, net, classes, cuda)
    print ('Test accuracy is {}'.format(test_performance))
    return test_performance

#####################################
### Load / Save STATE DIR

def save_model():
    save_dir = os.getcwd() + "/results/NIPS/"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    # date = datetime.datetime.now()
    # date_dir = save_dir + str(date.year) + str(date.month) + str(date.day) + '_' + str(date.hour) + str(date.minute)+ '/'  # Save in todays date.
    # os.mkdir(date_dir)
    model_name = save_dir + 'acc91.37_cifar10_fullPrec' + '_ep300_Adam-SGD_lr' + str(lr) + 'bs' + str(batch_size) + '_dither40_onlyLayer4.pt'
    torch.save(net.state_dict(), model_name)

def load_model():
    model_name = os.getcwd() + "/results/NIPS/acc89.75_cifar10_model_fullPrec_Adam_lr0.001bs20_1layerdropout40_1layer.pt"
    net.load_state_dict(torch.load(model_name))

#########################################################################################################
### PLOTS
def plot_results():
    plt.style.use('ggplot')

    # Total accuracy
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='Train accuracy')
    plt.plot(range(epochs), test_accuracy, label='Test accuracy')
    plt.plot(range(epochs), validation_accuracy, label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Percent Accuracy')
    plt.title('Training accuracy over: \n{} Iterations'.format(epochs), fontsize=16)
    plt.legend(loc='lower right')
    plt.show(block=False)

    #Accuracy by class.
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
