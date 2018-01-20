""" Extract CNN codes for CIFAR-10 dataset using pretrained ResNet-101."""

# Import libs
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
import argparse


def main():
    """ Extract CNN codes for CIFAR-10 dataset"""
    
    # Prepare transformation of CIFAR10 dataset
    # Dataset should be normalized, converted to torch.Tensor
    # and resized to resnet's input size: 224 
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Download train dataset from torchvision repo and prepare data loader
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.jobs)

    # Download pretrained ResNet-100 model
    resnet = torchvision.models.resnet101(pretrained=True)
    resnet = resnet.cuda()

    # Delete last FC layer of resnet to get CNN codes
    resnet_modules = list(resnet.children())[:-1]
    resnet = nn.Sequential(*resnet_modules)

    # Open files, where CNN codes will be saved (append binary mode)
    file1 = open(config.y, 'ab')
    file2 = open(config.x, 'ab')

    # Iterate through train set
    for (inputs, targets) in trainloader:
        # Save resnet output for CIFAR-10 images (that's CNN codes)
        X = resnet(Variable(inputs.cuda())).data.cpu().numpy()
        np.savetxt(file2, X.squeeze())

        # Save classes for CIFAR-10 images
        y = Variable(targets.cuda()).data.cpu().numpy()
        np.savetxt(file1, y)

    # Close files
    file1.close()
    file2.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths for CNN codes and categories
    parser.add_argument('--x', type=str, default='x.csv')
    parser.add_argument('--y', type=str, default='y.csv')

    # Parameters for data loader
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--jobs', type=int, default=2)

    config = parser.parse_args()
    main()
