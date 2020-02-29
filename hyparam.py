from time import perf_counter
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from hpsklearn import HyperoptEstimator, svc, random_forest, knn
from hyperopt import tpe
from sklearn.metrics import f1_score

def scorer(yt, yp): return 1 - f1_score(yt, yp, average='macro')

if __name__=='__main__':
    np.random.seed(42)
    train_X = np.load('data/train_X.npy')
    test_X = np.load('data/test_X.npy')
    train_Y = np.load('data/train_Y.npy')
    test_Y = np.load('data/test_Y.npy')

    estim = HyperoptEstimator(classifier=random_forest('rf'),algo=tpe.suggest,loss_fn=scorer,max_evals=200,trial_timeout=1200)
    estim.fit(train_X, train_Y)
    yp = estim.predict(test_X)
    print(f1_score(test_Y, yp, average='macro'))