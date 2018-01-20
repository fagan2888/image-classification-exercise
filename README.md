# Exercise
In this exercise we will build machine learning and deep learning models that will recognize images from the CIFAR-10 dataset. The plan is as follows:

1. We load and visulize images from CIFAR-10 dataset.
2. We do feautre extraction for CIFAR-10 images by means of transfer learning (use of pre-trained neural network).
3. We visualize these features by embedding them in two dimensions.
4. We train (and tune) SVM model on top of the features extracted by means of transfer learning.
5. We evaluate the qulity of SVM model. We measure the accuracy of class prediction for images from the CIFAR-10 dataset.
6. We check another variation of transfer learning.

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
![](https://github.com/witold-oleszkiewicz/image-classification-exercise/blob/master/images.png)

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
![](https://github.com/witold-oleszkiewicz/image-classification-exercise/blob/master/CNN_codes_2D.png)

# 04 SVM training

SVM classifier will be trained on the top of CNN codes. But first of all we should tune the parameters of SVM classifier and choose those values of the parameters for which classifier has the top accuracy. To do this we'll perform Grid Search. (Random Search would be better if we had to tune many parameters at once):

```
C_range = np.logspace(-3, 10, 10)
tuned_parameters = [{'C':C_range}, {'kernel':['linear']}, {'decision_function_shape':['ovo']}]
svm = GridSearchCV(SVC(), param_grid=tuned_parameters, n_jobs=config.jobs)
```

During Grid search the k-fold validation is performed (train SVM on one part of the dataset and validate SVM's accuracy on the another part of dataset). The optimal value of C parameter is determined as a result of Grid search: C=100.0. The accuracy on valiadtion set for C=100.0 is 83%.

# 05 Accuracy of the SVM model

The model obtained during the training will be tested on the test set of CIFAR-10. We download test images from torchvision repository:
```
testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=config.jobs)
```
Once we've downloaded CIFAR-10 dataset, let's extract visual features using a pre-trained CNN network. We'll use penultimate layer of pre-trained ResNet-101
```
resnet = torchvision.models.resnet101(pretrained=True)
resnet_modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*resnet_modules)
```

We'll get CNN codes by passing images from CIFAR-10 dataset through the resnet network:

```
X = resnet(Variable(inputs.cuda())).data.cpu().numpy()
```

Finally, we'll use a trained SVM to get the accuracy of the classification:
```
accuracy.append(svm.score(X.squeeze(), y))
```
The accuracy of classification measured on the test set is 84.5%.

# 06 Another approach to Transfer Learning

In this paragraph another approach to Transfer Learning will be presented. Instead of using CNN codes to train a classifier on top of it, we'll get pretrained ResNet model and then replace the last layer of it:

```
model_ft = models.resnet50(pretrained=True)
num_features = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_features, 10)
```
Now the neural network has 10 outputs, which is the number of categories in the CIFAR-10 dataset. Then we'll train such a neural network in online mode, using mini-batches of images from CIFAR-10 dataset. Adam optimizer is used and the optimization criterion is cross-entropy:

```
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.fc.parameters())
```

The accuracy of classification and the value of loss function in subsequent epochs is shown in the picture:
![](https://github.com/witold-oleszkiewicz/image-classification-exercise/blob/master/adam_acc.png)
![](https://github.com/witold-oleszkiewicz/image-classification-exercise/blob/master/adam_loss.png)

The accuracy is 85% what is close to what was achieved by training SVM model on top of the CNN codes.
