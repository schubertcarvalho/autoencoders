import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import helper
import os

PROJECT_ROOT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "data", "Cat_Dog_data")
TRAIN_PATH = os.path.join(DATA_PATH, "train")
TEST_PATH = os.path.join(DATA_PATH, "test")

CIFAR10_PATH = os.path.join(PROJECT_ROOT_DIR, 'data')


def load_cats_dogs_dataset(batch_size=64, dev_size=0.2, num_workers=0):

    # obtain training indices that will be used for validation
    
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(TRAIN_PATH, transform=train_transform)
    test_data = datasets.ImageFolder(TEST_PATH, transform=test_transform)

    train_sampler, dev_sampler = train_dev_split(len(train_data), dev_size)

    trainloader = torch.utils.data.DataLoader(train_data,
                                              batch_size=batch_size,
                                              sampler=train_sampler,
                                              num_workers=num_workers)
    devloader = torch.utils.data.DataLoader(train_data,
                                            batch_size=batch_size,
                                            sampler=dev_sampler,
                                            num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    return trainloader, devloader, testloader


# Load the Cifar10 dataset
def load_CIFAR_dataset(batch_size=64, dev_size=0.2, num_workers=0):

    # train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
    #                                       transforms.RandomRotation(30),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize((0.5, 0.5, 0.5),
    #                                                            (0.5, 0.5, 0.5))])

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(32),
                                          transforms.RandomRotation(20),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5),
                                                               (0.5, 0.5, 0.5))])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5),
                                                              (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root=CIFAR10_PATH, train=True,
                                download=False,
                                transform=train_transform)

    testset = datasets.CIFAR10(root=CIFAR10_PATH, train=False,
                               download=False,
                               transform=test_transform)

    train_sampler, dev_sampler = train_dev_split(len(trainset), dev_size)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              sampler=train_sampler,
                                              num_workers=num_workers)

    devloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            sampler=dev_sampler,
                                            num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)

    return trainloader, devloader, testloader


def cifar10_classes():
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return classes


def train_dev_split(train_size, dev_size):
    indices = list(range(train_size))
    np.random.shuffle(indices)
    split = int(np.floor(dev_size * train_size))
    train_idx, dev_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    dev_sampler = SubsetRandomSampler(dev_idx)

    return train_sampler, dev_sampler

if __name__ == "__main__":
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 32
    # percentage of training set to use as validation
    dev_size = 0.2

    # trainloader, devloader, testloader = load_cats_dogs_dataset(batch_size, dev_size, num_workers)
    trainloader, devloader, testloader = load_CIFAR_dataset(batch_size, dev_size, num_workers)

    # print(len(trainloader))
    # print(len(devloader))
    # print(len(testloader))

    data_iter = iter(trainloader)

    images, labels = next(data_iter)
    print(images.shape)
    print((labels == labels).sum().item())

    # helper.imshow(images[0], normalize=False)

    # fig, axes = plt.subplots(figsize=(10,4), ncols=4)
    # for ii in range(4):
    #     ax = axes[ii]
    #     helper.imshow(images[ii], ax=ax, normalize=False)