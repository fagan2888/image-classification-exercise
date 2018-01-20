""" Create a 10x10 grid of images from CIFAR-10 dataset"""

# Import libs
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse


def check_if_all_images_ready(images, n_classes, n_images):
    """Helps not iterate the whole dataset.""" 
    for i in range(n_classes):
        if(len(images[i])<n_images):
            # Return false when not all images are collected
            return False
    # Return true when there are at least n_images of each class
    return True
    
def denorm(image):
    """Convert image range (-1, 1) to (0, 1)."""
    out = (image + 1) / 2
    return out.clamp(0, 1)

def main():
    """Create grid image with 10 classes x 10 random example images from CIFAR-10 dataset"""

    # Prepare transformation of CIFAR-10 dataset
    # Dataset should be normalized and converted to torch.Tensor
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Download train dataset from torchvision repo
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    # Number of classes in CIFAR-10 dataset
    n_classes = 10

    # Load dataset, one random image at once
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

    # Prepare empty list of lists to store images from n_classes
    images = [[] for x in range(n_classes)]

    # Iterate through dataset and store images grouped by class
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        images[targets[0]].append(inputs[0])
        # Break when 10 images for each class is ready
        if check_if_all_images_ready(images, n_classes, n_images=10):
            break

    # Get only 10 images for each class
    for i in range(n_classes):
        images[i] = images[i][0:10]

    # Save grid image (10x10) to output file
    torchvision.utils.save_image(denorm(torch.stack([item for sublist in images for item in sublist])), config.out, nrow=n_classes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path for output image
    parser.add_argument('--out', type=str, default='images.png')
    config = parser.parse_args()
    main()
