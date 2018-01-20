""" Test SVM classifier using features (CNN codes) extracted from CIFAR-10 dataset by transfer learning """

# Import libs
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.externals import joblib
import numpy as np
import argparse

def main():
    """ Test SVM classifier, print average accuracy. """

    # Load trained SVM model from file
    svm = joblib.load(config.model)

    # Prepare transformation of CIFAR-10 dataset
    # Dataset should be normalized and converted to torch.Tensor
    # and resized to resnet's input size: 224 
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Download test dataset from torchvision repo and prepare data loader
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=config.jobs)


    # Download pretrained ResNet-100 model
    resnet = torchvision.models.resnet101(pretrained=True)
    resnet = resnet.cuda()

    # Delete last FC layer of resnet to get CNN codes
    resnet_modules = list(resnet.children())[:-1]
    resnet = nn.Sequential(*resnet_modules)

    # Prepare list for accuracy metric
    accuracy = []

    # Iterate through test dataset
    for (inputs, targets) in testloader:
        # Get resnet output for CIFAR-10 images (that's CNN codes)
        X = resnet(Variable(inputs.cuda())).data.cpu().numpy()
    
        # Get image category
        y = Variable(targets.cuda()).data.cpu().numpy()

        # Test SVM model (get classification's accuracy)
        accuracy.append(svm.score(X.squeeze(), y))

    # Print average accuracy for SVM model
    print("SVM accuracy:")
    print(np.mean(accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parameters for data loader
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--jobs', type=int, default=1)

    # Path of SVM model
    parser.add_argument('--model', type=str, default="svm.pkl")
    config = parser.parse_args()
    main()
