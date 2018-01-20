""" Train SVM classifier using features (CNN codes) extracted from CIFAR-10 dataset by transfer learning """

# Import libs
import numpy as np
from numpy import genfromtxt
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
import argparse

def main():
    """ Train SVM model, use CNN codes for CIFAR-10 dataset"""

    # Load CNN codes and categories from file
    X = genfromtxt(config.x)
    y = genfromtxt(config.y)

    # Specify tuned parameters of SVM
    C_range = np.logspace(-3, 10, 10)
    tuned_parameters = [{'C':C_range}, {'kernel':['linear']}, {'decision_function_shape':['ovo']}]

    # Use grid search to choose optimal parameters of SVM
    # (Random search would be better for more parameters)
    svm = GridSearchCV(SVC(), param_grid=tuned_parameters, n_jobs=config.jobs)

    # Train SVM using best value of hyperparameters
    svm.fit(X, y)
    print("The best parameters are %s with a score of %0.2f"
          % (svm.best_params_, svm.best_score_))

    # Save SVM model to file
    joblib.dump(svm, 'svm.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths for CNN codes and categories
    parser.add_argument('--x', type=str, default='x.csv')
    parser.add_argument('--y', type=str, default='y.csv')

    # Parameters for grid search
    parser.add_argument('--jobs', type=int, default=2)

    # Path for storage of SVM model
    parser.add_argument('--out', type=str, default="svm.pkl")
    config = parser.parse_args()
    main()
