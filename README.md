# Exercise
In this exercise we will build machine learning and deep learning models that will recognize images from the CIFAR-10 dataset.
# 01 Visualize dataset
First of all let's visualize the images from the CIFAR-10 collection. At the beginnig we download images from torchvision repository:
```
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
```

We apply transformation to normalize dataset and converted it to torch.Tensor:

```
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
```

The result is as follows:
(https://github.com/witold-oleszkiewicz/image-classification-exercise/blob/master/images.png)
