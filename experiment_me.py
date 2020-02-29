from time import perf_counter 
import numpy as np
from sklearn import clone
from sklearn.cluster import KMeans
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode

from tuner import DE

param_grid = {
    'svm': {"kernel": ["linear", "poly", "rbf", "sigmoid"],
            "C": [1, 50],
            "coef0": [0.0, 1],
            "gamma": [0.0, 1]},
    'knn': {"n_neighbors": [2, 10],
            "weights": ['uniform', 'distance']},
    'rf': {'min_samples_split':[2,10]}
}
def tune_learner(clf, params, train_X, train_Y, tune_X, tune_Y, goal):
    """
    :param learner:
    :param train_X:
    :param train_Y:
    :param tune_X:
    :param tune_Y:
    :param goal:
    :return:
    """
    tuner = DE(clf, params, goal)
    return tuner.tune(train_X, train_Y, tune_X, tune_Y, goal)

def run_baseline(clf, train_X, train_Y, test_X, test_Y,scorer):
    start = perf_counter()
    clf.fit(train_X, train_Y)
    stop = perf_counter()
    Yp = clf.predict(test_X)
    
    print("accuracy  ", scorer(Yp,test_Y))
    print("Model training time: ", stop - start)


def run_tuning_clf(learner, param_dist, train_X, train_Y, test_X, test_Y,scorer, repeats=1, fold=10):
    start = perf_counter()
    accs = []
    clfs = []
    kf = RepeatedStratifiedKFold(n_splits=fold, n_repeats=repeats)
    for fit_index, val_index in kf.split(train_X, train_Y):
        fit_X, fit_Y = train_X[fit_index], train_Y[fit_index]
        val_X, val_Y = train_X[val_index], train_Y[val_index]
        clf = clone(learner)
        params, evaluation = tune_learner(
            clf, param_dist, fit_X, fit_Y, val_X, val_Y, goal=scorer)
        print("Best Hyp:", params)
        clf.set_params(**params)
        clf = clf.fit(fit_X, fit_Y)  # Wrong, use whole trainset
        acc = scorer(test_Y, clf.predict(test_X))
        print(acc)
        accs.append(acc)
        clfs.append(clf)
    stop = perf_counter()
    print(sum(accs)/len(accs))
    print("Model tuning time: ", stop - start)
    return clfs

def optimalK(data, nrefs=3, maxClusters=15):  #
    gaps = np.zeros((len(range(1, maxClusters)),))
    for gap_index, k in enumerate(range(1, maxClusters)):
        refDisps = np.zeros(nrefs)
        for i in range(nrefs):
            randomReference = np.random.random_sample(size=data.shape)
            km = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=1)
            km.fit(randomReference)
            refDisps[i] = km.inertia_

        # Fit cluster to original data and create dispersion
        km = KMeans(n_clusters=k,init='k-means++',n_init=10,max_iter=300)
        km.fit(data)

        origDisp = km.inertia_
        gaps[gap_index] = np.log(np.mean(refDisps)) - np.log(origDisp)
        print(gap_index,k,gaps[gap_index])
    return gaps.argmax()


def run_kmeans(clf,params,train_X, train_Y, test_X, test_Y, scorer):
    start = perf_counter()
    #numClusters = optimalK(train_X)
    stop = perf_counter()
    numClusters = 13
    print(f"Found optimal k: {numClusters} in {stop-start} secs")

    kmeans = KMeans(n_clusters=numClusters,init='k-means++', max_iter=200, n_init=1)
    kmeans.fit(train_X)

    models = {}  # maintain a list of svms
    CY = kmeans.labels_
    for cl in range(numClusters):
        idx = CY == cl
        Xc, Yc = train_X[idx], train_Y[idx]
        print(cl,idx.sum(),np.unique(Yc,return_counts=True))
        clfs = run_tuning_clf(clf,params,Xc,Yc,test_X,test_Y,scorer,fold=5)
        models[cl] = clfs

    accs = []
    test_Cls = kmeans.predict(test_X)
    for ind in range(len(models[0])):
        Yp = np.zeros_like(test_Y) - 1
        for cl in range(numClusters):
            idx = test_Cls==cl
            Yp[idx] = models[cl][ind].predict(test_X[idx])
        assert np.all(Yp>=0)
        acc = scorer(test_Y,Yp)
        print(acc)
        accs.append(acc)
    print("Final:",sum(accs)/len(accs))

def run_skf(clf,params,train_X, train_Y, test_X, test_Y, scorer,numClusters = 13):
    start = perf_counter()
    skf = StratifiedKFold(n_splits=numClusters,shuffle=True).split(train_X,train_Y)

    models = {}
    for (cl,(_,val_idx)) in enumerate(skf):
        print("Starting cluster:",cl)
        idx = val_idx
        Xc, Yc = train_X[idx], train_Y[idx]
        print(len(idx),np.unique(Yc,return_counts=True))
        clfs = run_tuning_clf(clf,params,Xc,Yc,test_X,test_Y,scorer,fold=5)
        models[cl] = clfs

    print("Total Training time:",perf_counter()-start)

    start = perf_counter()
    accs = []
    test_Cls = np.random.choice(range(numClusters),size=test_Y.shape)
    print(np.unique(test_Cls, return_counts=True))

    for ind in range(len(models[0])):
        Yp = np.zeros((len(test_Y),numClusters))
        for cl in range(numClusters):
            Yp[:,cl] = models[cl][ind].predict(test_X)
        Yp = mode(Yp,axis=1)[0]
        assert np.all(Yp>=0)
        acc = scorer(test_Y,Yp)
        print(acc)
        accs.append(acc)
    print("Total Testing time:",perf_counter()-start)
    print("Final:",sum(accs)/len(accs))


if __name__ == '__main__':

    np.random.seed(0)
    train_X = np.load('data/train_X.npy')
    test_X = np.load('data/test_X.npy')
    train_Y = np.load('data/train_Y.npy')
    test_Y = np.load('data/test_Y.npy')
    def scorer(yt, yp): return f1_score(yt, yp, average='macro')

    #run_tuning_clf(SVC(), param_grid['svm'],train_X, train_Y, test_X, test_Y,scorer, fold=10)
    #print("Final:",optimalK(train_X,maxClusters=20))
    #run_skf(SVC(),param_grid['svm'],train_X,train_Y,test_X,test_Y,scorer,numClusters=13)
    run_skf(KNeighborsClassifier(n_jobs=-1),param_grid['knn'],train_X,train_Y,test_X,test_Y,scorer)
    #run_kmeans(RandomForestClassifier(n_jobs=-1),param_grid['rf'],train_X,train_Y,test_X,test_Y,scorer)
    #run_skf(RandomForestClassifier(n_jobs=-1), param_grid['rf'], train_X, train_Y, test_X, test_Y, scorer)