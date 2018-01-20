""" Visualize CNN codes from CIFAR-10 dataset and pretrained ResNet using PCA and tSNE methods """

# Import libs
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def main():
    """ Visulize CNN codes """

    # Load CNN codes and categories from file
    X = genfromtxt(config.x)
    y = genfromtxt(config.y)

    # Run PCA algorihtm
    pca = PCA(n_components=config.pca_components)
    pca_result = pca.fit_transform(X)

    # Run tSNE algorithm
    tsne = TSNE(n_components=2, verbose=0, perplexity=config.tsne_perplexity, n_iter=config.tsne_iter)
    tsne_results = tsne.fit_transform(pca_result)
    
    # Save results in the form of 2D plot
    results = np.column_stack((tsne_results, y))
    plt.scatter(results[:,0], results[:,1], c=results[:,2])
    plt.savefig(config.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths for CNN codes and categories
    parser.add_argument('--x', type=str, default='x.csv')
    parser.add_argument('--y', type=str, default='y.csv')

    # Path for output image
    parser.add_argument('--out', type=str, default='CNN_codes_2D.png')

    # PCA and tSNE parameters
    parser.add_argument('--pca_components', type=int, default=200)
    parser.add_argument('--tsne_perplexity', type=int, default=100)
    parser.add_argument('--tsne_iter', type=int, default=1000)

    config = parser.parse_args()
    main()
