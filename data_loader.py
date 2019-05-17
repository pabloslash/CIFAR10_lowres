# This is an example for the CIFAR-10 dataset.
# There's a function for creating a train and validation iterator.
# There's also a function for creating a test iterator.
# Inspired by https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
from data_loader import *
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from utils import *
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import IPython as IP

def load_cifar(cifar_path, batch_size, cifar10=True, augment=True, shuffle=False):
    # Load Training + Validation
    trainloader, validationloader = get_train_valid_loader(data_dir=cifar_path,
                                                           batch_size=batch_size,
                                                           augment=augment,
                                                           random_seed=1,
                                                           shuffle=shuffle,
                                                           show_sample=False,
                                                           cifar10=cifar10)
    print('Augment: {}'.format(augment))
    print('cifar10: {}'.format(cifar10))
    # Load Testing
    testloader = get_test_loader(data_dir=cifar_path,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 cifar10=cifar10)

    if cifar10:
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        classes = (	'beaver', 'dolphin', 'otter', 'seal', 'whale',
                    'aquarium', 'fish', 'flatfish', 'ray', 'shark', 'trout',
                    'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
                    'bottles', 'bowls', 'cans', 'cups', 'plates',
                    'apples', 'mushrooms', 'oranges', 'pears', 'sweet' 'peppers',
                    'clock', 'computer' 'keyboard', 'lamp', 'telephone', 'television',
                    'bed', 'chair', 'couch', 'table', 'wardrobe',
	                'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
	                'bear', 'leopard', 'lion', 'tiger', 'wolf',
                    'bridge', 'castle', 'house', 'road', 'skyscraper',
	                'cloud', 'forest', 'mountain', 'plain', 'sea',
	                'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
	                'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                    'crab', 'lobster', 'snail', 'spider', 'worm',
                    'baby', 'boy', 'girl', 'man', 'woman',
                    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
	                'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
	                'maple', 'oak', 'palm', 'pine', 'willow',
	                'bicycle', 'bus', 'motorcycle', 'pickup' 'truck', 'train',
	                'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor')

    return trainloader, validationloader, testloader, classes


def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           cifar10=True,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    print(augment)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    # load the dataset
    if cifar10:
        train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                    download=True, transform=train_transform)

        valid_dataset = datasets.CIFAR10(root=data_dir, train=True,
                    download=True, transform=valid_transform)

    else:
        train_dataset = datasets.CIFAR100(root=data_dir, train=True,
                    download=True, transform=train_transform)

        valid_dataset = datasets.CIFAR100(root=data_dir, train=True,
                    download=True, transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=batch_size, sampler=train_sampler,
                    num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                    batch_size=batch_size, sampler=valid_sampler,
                    num_workers=num_workers, pin_memory=pin_memory)


    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=9,
                                                    shuffle=shuffle,
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory)


        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)

def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    cifar10=True,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if cifar10:
        dataset = datasets.CIFAR10(root=data_dir,
                                   train=False,
                                   download=True,
                                   transform=transform)

    else:
        dataset = datasets.CIFAR100(root=data_dir,
                                   train=False,
                                   download=True,
                                   transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)

    return data_loader
