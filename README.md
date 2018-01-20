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

# 02 CNN codes extraction

Once we've downloaded CIFAR-10 dataset, let's extract visual features using a pre-trained CNN network. We'll use pre-trained ResNet-101. 

```
resnet = torchvision.models.resnet101(pretrained=True)

```
We'll use penultimate layer of ResNet, since in deep neural networks subsequent layers can have more knowledge.

```
resnet_modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*resnet_modules)
```

We'll get CNN codes by passing images from CIFAR-10 dataset through the resnet network:

```
X = resnet(Variable(inputs.cuda())).data.cpu().numpy()
```

# 03 CNN codes visualization

In order to visualize CNN codes in two dimensions we'll use two algorithms: PCA (Principal component analysis) and tSNE (t-distributed Stochastic Neighbor Embedding). tSNE is considered to be the best technique for visualizing multidimensional data in 2D, however in our case CNN codes are 2048-dimensional (due to the size of the penultimate layer of ResNet) what could be too big for tSNE. Therefore at the beginning we will use PCA to reduce the number of dimensions from 2048 to 200
```
pca = PCA(n_components=config.pca_components)
pca_result = pca.fit_transform(X)

```
and then use tSNE to reduce it from 200 to 2
```
tsne = TSNE(n_components=2, verbose=0, perplexity=config.tsne_perplexity, n_iter=config.tsne_iter)
tsne_results = tsne.fit_transform(pca_result)

```

The result of 2D visualization is:
