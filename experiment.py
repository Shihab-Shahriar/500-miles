from __future__ import division, print_function
import pickle
import pdb
import os
import time
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn import metrics
import gensim
import random
from learners import SK_SVM, SK_KNN, SK_MLP
from tuner import DE
from model import PaperData
from utility import study
from results import results_process
import numpy as np
#import wget
import zipfile
from sklearn import neighbors
from sklearn import metrics
import threading
from threading import Barrier
import timeit
import multiprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import collections
from multiprocessing import Queue
import pandas as pd
import warnings
from sklearn.neural_network import MLPClassifier


param_grid = {
    'svm': {"kernel": ["linear", "poly", "rbf", "sigmoid"],
            "C": [1, 50],
            "coef0": [0.0, 1],
            "gamma": [0.0, 1]},
    'knn': {"n_neighbors": [2, 10],
            "weights": ['uniform', 'distance']}
}


def tune_learner(clf, params, train_X, train_Y, tune_X, tune_Y, goal,
                 target_class=None):
    """
    :param learner:
    :param train_X:
    :param train_Y:
    :param tune_X:
    :param tune_Y:
    :param goal:
    :param target_class:
    :return:
    """
    tuner = DE(clf, params, goal)
    return tuner.tune(train_X, train_Y, tune_X, tune_Y, goal)


def load_vec(d, data, use_pkl=False, file_name=None):
    if use_pkl:
        if os.path.isfile(file_name):
            with open(file_name, "rb") as my_pickle:
                return pickle.load(my_pickle)
    else:
        # print("call get_document_vec")
        return d.get_document_vec(data, file_name)


def print_results(clfs, stop, start):
    file_name = time.strftime(os.path.sep.join([".", "results",
                                                "%Y%m%d_%H:%M:%S.txt"]))
    file_name = os.path.sep.join(["20171103.txt"])
    content = ""
    for each in clfs:
        content += each.confusion
    print(content)
    print("Model training time: ", stop - start)
    with open(file_name, "w") as f:
        f.write(content)
    results_process.reports(file_name)


def get_acc(cm):
    out = []
    for i in range(4):
        out.append(cm[i][i] / 400)
    return out


@study
def run_tuning_SVM(word2vec_src, repeats=3,
                   fold=10,
                   tuning=True):
    """
    :param word2vec_src:str, path of word2vec model
    :param repeats:int, number of repeats
    :param fold: int,number of folds
    :param tuning: boolean, tuning or not.
    :return: None
    """
    print("# word2vec:", word2vec_src)
    word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
    data = PaperData(word2vec=word2vec_model)
    train_pd = load_vec(data, data.train_data, file_name=False)
    print(train_pd)
    test_pd = load_vec(data, data.test_data, file_name=False)
    learner = [SK_SVM][0]
    goal = {0: "PD", 1: "PF", 2: "PREC", 3: "ACC", 4: "F", 5: "G", 6: "Macro_F",
            7: "Micro_F"}[6]
    print(goal)
    F = {}
    clfs = []
    start = timeit.default_timer()
    for i in range(repeats):  # repeat n times here
        kf = StratifiedKFold(train_pd.loc[:, "LinkTypeId"].values, fold,
                             shuffle=True)
        for train_index, tune_index in kf:
            print(train_pd)
            print(train_index)
            train_data = train_pd.ix[train_index]
            print(train_data)
            tune_data = train_pd.ix[tune_index]
            train_X = train_data.loc[:, "Output"].values
            train_Y = train_data.loc[:, "LinkTypeId"].values
            tune_X = tune_data.loc[:, "Output"].values
            tune_Y = tune_data.loc[:, "LinkTypeId"].values
            test_X = test_pd.loc[:, "Output"].values
            test_Y = test_pd.loc[:, "LinkTypeId"].values
            params, evaluation = tune_learner(learner, train_X, train_Y, tune_X,
                                              tune_Y, goal) if tuning else ({}, 0)
            clf = learner(train_X, train_Y, test_X, test_Y, goal)
            F = clf.learn(F, **params)
            clfs.append(clf)
    stop = timeit.default_timer()
    print("Model training time: ", stop - start)
    print_results(clfs, stop, start)


@study
def run_tuning_MLP(word2vec_src, repeats=1,
                   fold=2,
                   tuning=True):
    """
    :param word2vec_src:str, path of word2vec model
    :param repeats:int, number of repeats
    :param fold: int,number of folds
    :param tuning: boolean, tuning or not.
    :return: None
    """
    print("# word2vec:", word2vec_src)
    word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
    data = PaperData(word2vec=word2vec_model)
    train_pd = load_vec(data, data.train_data, file_name=False)
    print(train_pd)
    test_pd = load_vec(data, data.test_data, file_name=False)
    learner = [SK_MLP][0]
    goal = {0: "PD", 1: "PF", 2: "PREC", 3: "ACC", 4: "F", 5: "G", 6: "Macro_F",
            7: "Micro_F"}[6]
    print(goal)
    F = {}
    clfs = []
    start = timeit.default_timer()
    for i in range(repeats):  # repeat n times here
        kf = StratifiedKFold(train_pd.loc[:, "LinkTypeId"].values, fold,
                             shuffle=True)
        for train_index, tune_index in kf:
            print(train_pd)
            print(train_index)
            train_data = train_pd.ix[train_index]
            print(train_data)
            tune_data = train_pd.ix[tune_index]
            train_X = train_data.loc[:, "Output"].values
            train_Y = train_data.loc[:, "LinkTypeId"].values
            tune_X = tune_data.loc[:, "Output"].values
            tune_Y = tune_data.loc[:, "LinkTypeId"].values
            test_X = test_pd.loc[:, "Output"].values
            test_Y = test_pd.loc[:, "LinkTypeId"].values
            params, evaluation = tune_learner(learner, train_X, train_Y, tune_X,
                                              tune_Y, goal) if tuning else ({}, 0)
            clf = learner(train_X, train_Y, test_X, test_Y, goal)
            F = clf.learn(F, **params)
            clfs.append(clf)
    stop = timeit.default_timer()
    print("Model training time: ", stop - start)
    print_results(clfs, stop, start)


@study
def run_tuning_KNN(word2vec_src, repeats=6,
                   fold=10,
                   tuning=True):
    """
    :param word2vec_src:str, path of word2vec model
    :param repeats:int, number of repeats
    :param fold: int,number of folds
    :param tuning: boolean, tuning or not.
    :return: None
    """
    print("# word2vec:", word2vec_src)
    word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
    data = PaperData(word2vec=word2vec_model)
    train_pd = load_vec(data, data.train_data, file_name=False)
    test_pd = load_vec(data, data.test_data, file_name=False)
    learner = [SK_KNN][0]
    goal = {0: "PD", 1: "PF", 2: "PREC", 3: "ACC", 4: "F", 5: "G", 6: "Macro_F",
            7: "Micro_F"}[6]
    F = {}
    clfs = []
    start = timeit.default_timer()
    for i in range(repeats):  # repeat n times here
        kf = StratifiedKFold(train_pd.loc[:, "LinkTypeId"].values, fold,
                             shuffle=True)
        for train_index, tune_index in kf:
            train_data = train_pd.ix[train_index]
            tune_data = train_pd.ix[tune_index]
            train_X = train_data.loc[:, "Output"].values
            train_Y = train_data.loc[:, "LinkTypeId"].values
            tune_X = tune_data.loc[:, "Output"].values
            tune_Y = tune_data.loc[:, "LinkTypeId"].values
            test_X = test_pd.loc[:, "Output"].values
            test_Y = test_pd.loc[:, "LinkTypeId"].values
            params, evaluation = tune_learner(learner, train_X, train_Y, tune_X,
                                              tune_Y, goal) if tuning else ({}, 0)
            clf = learner(train_X, train_Y, test_X, test_Y, goal)
            F = clf.learn(F, **params)
            clfs.append(clf)
    stop = timeit.default_timer()
    print("Model training time: ", stop - start)
    print_results(clfs)


@study
def run_SVM_baseline(word2vec_src):
    """
    Run SVM+word embedding experiment !
    This is the baseline method.
    :return:None
    """
    # Create a subplot with 1 row and 2 columns
    print("# word2vec:", word2vec_src)
    clf = svm.SVC(kernel="rbf", gamma=0.005)
    word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
    data = PaperData(word2vec=word2vec_model)
    train_pd = load_vec(data, data.train_data, use_pkl=False)
    test_pd = load_vec(data, data.test_data, use_pkl=False)
    train_X = train_pd.loc[:, "Output"].tolist()
    train_Y = train_pd.loc[:, "LinkTypeId"].tolist()
    test_X = test_pd.loc[:, "Output"].tolist()
    test_Y = test_pd.loc[:, "LinkTypeId"].tolist()
    start = timeit.default_timer()
    clf.fit(train_X, train_Y)
    stop = timeit.default_timer()
    predicted = clf.predict(test_X)
    print(metrics.classification_report(test_Y, predicted,
                                        labels=["1", "2", "3", "4"],
                                        digits=3))
    cm = metrics.confusion_matrix(
        test_Y, predicted, labels=["1", "2", "3", "4"])
    print("accuracy  ", get_acc(cm))
    print("Model training time: ", stop - start)


@study
def run_KNN_baseline(word2vec_src):
    """
    Run KNN+word embedding experiment !
    This is the baseline method.
    :return:None
    """
    # Create a subplot with 1 row and 2 columns
    print("# word2vec:", word2vec_src)
    clf = neighbors.KNeighborsClassifier(n_neighbors=5)
    word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
    data = PaperData(word2vec=word2vec_model)
    train_pd = load_vec(data, data.train_data, use_pkl=False)
    test_pd = load_vec(data, data.test_data, use_pkl=False)
    train_X = train_pd.loc[:, "Output"].tolist()
    train_Y = train_pd.loc[:, "LinkTypeId"].tolist()
    test_X = test_pd.loc[:, "Output"].tolist()
    test_Y = test_pd.loc[:, "LinkTypeId"].tolist()
    start = timeit.default_timer()
    clf.fit(train_X, train_Y)
    stop = timeit.default_timer()
    predicted = clf.predict(test_X)
    print(metrics.classification_report(test_Y, predicted,
                                        labels=["1", "2", "3", "4"],
                                        digits=3))
    cm = metrics.confusion_matrix(
        test_Y, predicted, labels=["1", "2", "3", "4"])
    print("accuracy  ", get_acc(cm))
    print("Model training time: ", stop - start)


@study
def run_MLP(word2vec_src):
    """
    Run SVM+word embedding experiment !
    This is the baseline method.
    :return:None
    """
    # Create a subplot with 1 row and 2 columns
    print("# word2vec:", word2vec_src)
    clf = MLPClassifier()
    word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
    data = PaperData(word2vec=word2vec_model)
    train_pd = load_vec(data, data.train_data, use_pkl=False)
    test_pd = load_vec(data, data.test_data, use_pkl=False)
    train_X = train_pd.loc[:, "Output"].tolist()
    train_Y = train_pd.loc[:, "LinkTypeId"].tolist()
    test_X = test_pd.loc[:, "Output"].tolist()
    test_Y = test_pd.loc[:, "LinkTypeId"].tolist()
    start = timeit.default_timer()
    clf.fit(train_X, train_Y)
    stop = timeit.default_timer()
    predicted = clf.predict(test_X)
    print(metrics.classification_report(test_Y, predicted,
                                        labels=["1", "2", "3", "4"],
                                        digits=3))
    cm = metrics.confusion_matrix(
        test_Y, predicted, labels=["1", "2", "3", "4"])
    print("accuracy  ", get_acc(cm))
    print("Model training time: ", stop - start)

# Katie's Code +++++++++++++++++++++++++++++++
# returns the svm model


def run_SVM_C(word2vec_src, train_pd, queue, l, test_pd_n):
    clf = svm.SVC(kernel="rbf", gamma=0.005)
    clfs = []
#  word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
#  data = PaperData(word2vec=word2vec_model)
#  print("Train data: " + str(train_pd.shape))
#  if train_pd is None: train_pd = load_vec(
#      data, data.train_data, use_pkl=False)
    train_X = train_pd.loc[:, "Output"].tolist()
    train_Y = train_pd.loc[:, "LinkTypeId"].tolist()
    start = timeit.default_timer()
    clf.fit(train_X, train_Y)
    stop = timeit.default_timer()
    print("SVM Model Train Time", (stop-start))
    clfs.append(clf)
    clfs.append(l)
    queue.put(clfs)
    return clf


def run_KNN_C(word2vec_src, train_pd, queue, l, test_pd_n):
    clf = neighbors.KNeighborsClassifier(n_neighbors=5)
    clfs = []
#  word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
#  data = PaperData(word2vec=word2vec_model)
#  print("Train data: " + str(train_pd.shape))
#  if train_pd is None: train_pd = load_vec(
#      data, data.train_data, use_pkl=False)
    train_X = train_pd.loc[:, "Output"].tolist()
    train_Y = train_pd.loc[:, "LinkTypeId"].tolist()
    start = timeit.default_timer()
    clf.fit(train_X, train_Y)
    stop = timeit.default_timer()
    print("Th", l)
    print("KNN Model Train Time", (stop-start))
    clfs.append(clf)
    clfs.append(l)
    queue.put(clfs)
    return clf


@study
def run_tuning_SVM_C(word2vec_src, train_pd_c, queue, l, test_pd_c, repeats=1,
                     fold=2,
                     tuning=True):
    """
    :param word2vec_src:str, path of word2vec model
    :param repeats:int, number of repeats
    :param fold: int,number of folds
    :param tuning: boolean, tuning or not.
    :return: None
    """
    # print("# word2vec:", word2vec_src)
    #word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
    #data = PaperData(word2vec=word2vec_model)
    train_pd_c = train_pd_c.reset_index()
    train_pd = train_pd_c
    test_pd = test_pd_c
    learner = [SK_SVM][0]
    goal = {0: "PD", 1: "PF", 2: "PREC", 3: "ACC", 4: "F", 5: "G", 6: "Macro_F",
            7: "Micro_F"}[6]
    F = {}
    clfs = []
    for i in range(repeats):  # repeat n times here
        kf = StratifiedKFold(train_pd.loc[:, "LinkTypeId"].values, fold,
                             shuffle=True)
        for train_index, tune_index in kf:
            train_data = train_pd.ix[train_index]
            tune_data = train_pd.ix[tune_index]
            train_X = train_data.loc[:, "Output"].values
            train_Y = train_data.loc[:, "LinkTypeId"].values
            tune_X = tune_data.loc[:, "Output"].values
            tune_Y = tune_data.loc[:, "LinkTypeId"].values
            test_X = test_pd.loc[:, "Output"].values
            test_Y = test_pd.loc[:, "LinkTypeId"].values
            params, evaluation = tune_learner(learner, train_X, train_Y, tune_X,
                                              tune_Y, goal) if tuning else ({}, 0)
            clf = learner(train_X, train_Y, test_X, test_Y, goal)
            F = clf.learn(F, **params)
            clfs.append(clf)
    clfs.append(l)
    queue.put(clfs)
    return clfs


@study
def run_tuning_KNN_C(word2vec_src, train_pd_c, queue, l, test_pd_c, repeats=10,
                     fold=10,
                     tuning=True):
    """
    :param word2vec_src:str, path of word2vec model
    :param repeats:int, number of repeats
    :param fold: int,number of folds
    :param tuning: boolean, tuning or not.
    :return: None
    """
    # print("# word2vec:", word2vec_src)
    #word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
    #data = PaperData(word2vec=word2vec_model)
    train_pd_c = train_pd_c.reset_index()
    train_pd = train_pd_c
    test_pd = test_pd_c
    learner = [SK_KNN][0]
    goal = {0: "PD", 1: "PF", 2: "PREC", 3: "ACC", 4: "F", 5: "G", 6: "Macro_F",
            7: "Micro_F"}[6]
    F = {}
    clfs = []
    for i in range(repeats):  # repeat n times here
        kf = StratifiedKFold(train_pd.loc[:, "LinkTypeId"].values, fold,
                             shuffle=True)
        for train_index, tune_index in kf:
            train_data = train_pd.ix[train_index]
            tune_data = train_pd.ix[tune_index]
            train_X = train_data.loc[:, "Output"].values
            train_Y = train_data.loc[:, "LinkTypeId"].values
            tune_X = tune_data.loc[:, "Output"].values
            tune_Y = tune_data.loc[:, "LinkTypeId"].values
            test_X = test_pd.loc[:, "Output"].values
            test_Y = test_pd.loc[:, "LinkTypeId"].values
            params, evaluation = tune_learner(learner, train_X, train_Y, tune_X,
                                              tune_Y, goal) if tuning else ({}, 0)
            clf = learner(train_X, train_Y, test_X, test_Y, goal)
            F = clf.learn(F, **params)
            clfs.append(clf)
    clfs.append(l)
    queue.put(clfs)
    return clfs

# parses and returns a given svm in the format of dictionary -
# [class](precision, recall, f1score, support)


def results_SVM(clf, test_X, test_Y):
    predicted = clf.predict(test_X)
    # labels: ["Duplicates", "DirectLink","IndirectLink", "Isolated"]
    report_gen = metrics.classification_report(
        test_Y, predicted, labels=["1", "2", "3", "4"], digits=3)
    parsed_report = parse_classification_report(report_gen)
    return parsed_report
   #cm=metrics.confusion_matrix(test_Y, predicted, labels=["1", "2", "3", "4"])
    # print("accuracy  ", get_acc(cm)


def results_SVM_C(predicted, test_Y):
    #predicted = clf.predict(test_X)
    # labels: ["Duplicates", "DirectLink","IndirectLink", "Isolated"]
    report_gen = metrics.classification_report(
        test_Y, predicted, labels=["1", "2", "3", "4"], digits=3)
    print(report_gen)
    classifaction_report_csv(report_gen)
    parsed_report = parse_classification_report(report_gen)
    return parsed_report


def classifaction_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('     ')
        row['class'] = row_data[2]
        row['precision'] = float(row_data[3].strip())
        row['recall'] = float(row_data[4])
        row['f1_score'] = float(row_data[5])
        row['support'] = float(row_data[6].strip())
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report.csv', mode='a', index=False)


def total_summary(result_set, num_rows, start0, start1, stop0, stop1, start, stop):
    weightedAvgs = [0, 0, 0]
    for l in result_set:
        avg_list = l['avg']
        for i in range(3):
            support_count = avg_list[3]
            weightedAvgs[i] += (avg_list[i] * support_count)/num_rows

    result = {}
    result['precision'] = weightedAvgs[0]
    result['recall'] = weightedAvgs[1]
    result['f1'] = weightedAvgs[2]
    # print(result)
    print("GAP statistics Time:", (stop - start))
    print("1st Model training time: ", (stop0 - start0))
    print("layer 2 Models training time: ", (stop1 - start1))
    print("Total Model training time: ", (stop1 - start1))
    print("Total training time: ", (stop1 - start))


def run_kmeans(word2vec_src):

    print("# word2vec:", word2vec_src)
    word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
    data = PaperData(word2vec=word2vec_model)
    train_pd = load_vec(data, data.train_data, use_pkl=False)
    test_pd = load_vec(data, data.test_data, use_pkl=False)
    train_X = train_pd.loc[:, "Output"].tolist()
    queue = Queue()

    start = timeit.default_timer()
    numClusters = optimalK(pd.DataFrame(train_X))
    stop = timeit.default_timer()
    #numClusters = 5
    print("Found optimal k: " + str(numClusters))
    clf = KMeans(n_clusters=numClusters,
                 init='k-means++', max_iter=200, n_init=1)

    start0 = timeit.default_timer()
    clf.fit(train_X)
    stop0 = timeit.default_timer()

    svm_models = []  # maintain a list of svms
    s1 = timeit.default_timer()
    data.train_data['clabel'] = clf.labels_
    s2 = timeit.default_timer()
    print("Inter - ", (s2-s1))
    start1 = timeit.default_timer()
    #b = Barrier(numClusters)
    # Change the target here as this will be used result validation purpose
    target_model = run_tuning_KNN_C
    for l in range(numClusters):
        cluster = data.train_data.loc[data.train_data['clabel'] == l]
        print("Thread No", l)
        t = threading.Thread(target=run_tuning_KNN_C, args=[
                             word2vec_src, cluster, queue, l])
        threads.append(t)
        t.start()
        response = queue.get()
        svm_models.append(response)
    # b.wait()
    for thread in threads:
        thread.join()
    stop1 = timeit.default_timer()
    print("Done all models - ", (stop1 - start0))

    svm_results = []  # maintain a list of svm results
    test_X = test_pd.loc[:, "Output"].tolist()
    predicted = clf.predict(test_X)
    data.test_data['clabel'] = predicted
    total_predicted = []
    total_cluster_Y = []
    avg_predicted = []
    avg_cluster_Y = []
    for i in range(len(svm_models[l])-1):
        total_predicted = []
        total_cluster_Y = []
        for l in range(numClusters):
            cluster = data.test_data.loc[data.test_data['clabel'] == l]
            svm_model = svm_models[l][i]
            cluster_X = cluster.loc[:, "Output"].tolist()
            cluster_Y = cluster.loc[:, "LinkTypeId"].tolist()
            total_cluster_Y = np.append(total_cluster_Y, cluster_Y)
            avg_cluster_Y = np.append(avg_cluster_Y, cluster_Y)
            if target_model == run_tuning_SVM_C or target_model == run_tuning_KNN_C:
                predicted_C = svm_model.learner.predict(cluster_X)
            else:
                predicted_C = svm_model.predict(cluster_X)
            total_predicted = np.append(total_predicted, predicted_C)
            avg_predicted = np.append(avg_predicted, predicted_C)
        # store all the SVM result report in a dictionary
        svm_results.append(results_SVM_C(total_predicted, total_cluster_Y))
    svm_results.append(results_SVM_C(avg_predicted, avg_cluster_Y))
    # call the helper method to summarize the svm results
    total_summary(
        svm_results, test_pd.shape[0], start0, start1, stop0, stop1, start, stop)


def run_kmeans_m(word2vec_src):

    print("# word2vec:", word2vec_src)
    word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
    data = PaperData(word2vec=word2vec_model)
    train_pd = load_vec(data, data.train_data, use_pkl=False)
    test_pd = load_vec(data, data.test_data, use_pkl=False)
    train_X = train_pd.loc[:, "Output"].tolist()
    queue = Queue()

    start = timeit.default_timer()
    numClusters = optimalK(pd.DataFrame(train_X))
    stop = timeit.default_timer()
    #numClusters = 5
    print("Found optimal k: " + str(numClusters))
    clf = KMeans(n_clusters=numClusters,
                 init='k-means++', max_iter=200, n_init=1)

    start0 = timeit.default_timer()
    clf.fit(train_X)
    stop0 = timeit.default_timer()

    svm_models = []  # maintain a list of svms
    s1 = timeit.default_timer()
    data.train_data['clabel'] = clf.labels_
    s2 = timeit.default_timer()
    print("Inter - ", (s2-s1))
    start1 = timeit.default_timer()
    # Change the target here as this will be used result validation purpose
    target_model = run_tuning_SVM_C
    for l in range(numClusters):
        cluster = data.train_data.loc[data.train_data['clabel'] == l]
        print("Thread No", l)
        #result.append(pool.apply_async(run_KNN_C, args = (word2vec_src,cluster,queue,)))
        t = threading.Thread(target=run_tuning_SVM_C, args=[
                             word2vec_src, cluster, queue, l, test_pd])
        threads.append(t)
    for th in threads:
        th.start()
    for th in threads:
        response = queue.get()
        svm_models.append(response)
    svm_models = sorted(svm_models, key=lambda th: th[-1])
    stop1 = timeit.default_timer()
    svm_results = []  # maintain a list of svm results
    test_X = test_pd.loc[:, "Output"].tolist()
    predicted = clf.predict(test_X)
    data.test_data['clabel'] = predicted
    total_predicted = []
    total_cluster_Y = []
    avg_predicted = []
    avg_cluster_Y = []
    print(len(svm_models[l])-1)
    for i in range(len(svm_models[l])-1):
        total_predicted = []
        total_cluster_Y = []
        for l in range(numClusters):
            cluster = data.test_data.loc[data.test_data['clabel'] == l]
            svm_model = svm_models[l][i]
            cluster_X = cluster.loc[:, "Output"].tolist()
            cluster_Y = cluster.loc[:, "LinkTypeId"].tolist()
            total_cluster_Y = np.append(total_cluster_Y, cluster_Y)
            avg_cluster_Y = np.append(avg_cluster_Y, cluster_Y)
            if target_model == run_tuning_SVM_C or target_model == run_tuning_KNN_C:
                predicted_C = svm_model.learner.predict(cluster_X)
            else:
                predicted_C = svm_model.predict(cluster_X)
            total_predicted = np.append(total_predicted, predicted_C)
            avg_predicted = np.append(avg_predicted, predicted_C)
        # store all the SVM result report in a dictionary
        svm_results.append(results_SVM_C(total_predicted, total_cluster_Y))
    svm_results.append(results_SVM_C(avg_predicted, avg_cluster_Y))
    # call the helper method to summarize the svm results
    total_summary(
        svm_results, test_pd.shape[0], start0, start1, stop0, stop1, start, stop)


def run_kmeans_mp(word2vec_src):

    print("# word2vec:", word2vec_src)
    word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
    data = PaperData(word2vec=word2vec_model)
    train_pd = load_vec(data, data.train_data, use_pkl=False)
    test_pd = load_vec(data, data.test_data, use_pkl=False)
    train_X = train_pd.loc[:, "Output"].tolist()
    queue = Queue()
    pool = multiprocessing.Pool()
    processes = []

    start = timeit.default_timer()
    numClusters = optimalK(pd.DataFrame(train_X))
    stop = timeit.default_timer()
    #numClusters = 5
    print("Found optimal k: " + str(numClusters))
    clf = KMeans(n_clusters=numClusters,
                 init='k-means++', max_iter=200, n_init=1)

    start0 = timeit.default_timer()
    clf.fit(train_X)
    stop0 = timeit.default_timer()

    svm_models = []  # maintain a list of svms
    s1 = timeit.default_timer()
    data.train_data['clabel'] = clf.labels_
    s2 = timeit.default_timer()
    print("Inter - ", (s2-s1))
    start1 = timeit.default_timer()
    # Change the target here as this will be used result validation purpose
    target_model = run_tuning_KNN_C
    for l in range(numClusters):
        cluster = data.train_data.loc[data.train_data['clabel'] == l]
        print("Thread No", l)
        pool.apply_async(run_tuning_KNN_C, (word2vec_src,
                                            cluster, queue, l, test_pd,))
        #t = threading.Thread(target = run_tuning_SVM_C, args = [word2vec_src,cluster,queue,l,test_pd])

#  for pr in processes:
#      pr.start()
    for pr in range(numClusters):
        response = queue.get()
        svm_models.append(response)
    print(svm_models)
    svm_models = sorted(svm_models, key=lambda th: th[-1])
    stop1 = timeit.default_timer()
    print(svm_models)

    svm_results = []  # maintain a list of svm results
    test_X = test_pd.loc[:, "Output"].tolist()
    predicted = clf.predict(test_X)
    data.test_data['clabel'] = predicted
    total_predicted = []
    total_cluster_Y = []
    avg_predicted = []
    avg_cluster_Y = []
    for i in range(len(svm_models[l])-1):
        total_predicted = []
        total_cluster_Y = []
        for l in range(numClusters):
            cluster = data.test_data.loc[data.test_data['clabel'] == l]
            svm_model = svm_models[l][i]
            cluster_X = cluster.loc[:, "Output"].tolist()
            cluster_Y = cluster.loc[:, "LinkTypeId"].tolist()
            total_cluster_Y = np.append(total_cluster_Y, cluster_Y)
            avg_cluster_Y = np.append(avg_cluster_Y, cluster_Y)
            if target_model == run_tuning_SVM_C or target_model == run_tuning_KNN_C:
                predicted_C = svm_model.learner.predict(cluster_X)
            else:
                predicted_C = svm_model.predict(cluster_X)
            total_predicted = np.append(total_predicted, predicted_C)
            avg_predicted = np.append(avg_predicted, predicted_C)
        # store all the SVM result report in a dictionary
        svm_results.append(results_SVM_C(total_predicted, total_cluster_Y))
    svm_results.append(results_SVM_C(avg_predicted, avg_cluster_Y))
    # call the helper method to summarize the svm results
    total_summary(
        svm_results, test_pd.shape[0], start0, start1, stop0, stop1, start, stop)
# Source: https://anaconda.org/milesgranger/gap-statistic/notebook


def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):

            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=1)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_
    # print(str(i+1) + ": " + str(origDisp))

    # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

    # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append(
            {'clusterCount': k, 'gap': gap}, ignore_index=True)

    # return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
    return gaps.argmax()

# Not used, but wanted to put this code somewhere


def results_kmeans(clf, train_X, train_Y, test_X, test_Y):
    predicted = clf.predict(test_X)
    print("Homogeneity: %0.3f" %
          metrics.homogeneity_score(train_Y, clf.labels_))
    print("Completeness: %0.3f" %
          metrics.completeness_score(train_Y, clf.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(train_Y, clf.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(train_Y, clf.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(train_X, clf.labels_, sample_size=1000))


"""
Parse a sklearn classification report into a dict keyed by class name
and containing a tuple (precision, recall, fscore, support) for each class
Reference: https://gist.github.com/julienr/6b9b9a03bd8224db7b4f
"""


def parse_classification_report(clfreport):

    lines = clfreport.split('\n')
    # Remove empty lines
    lines = list(filter(lambda l: not len(l.strip()) == 0, lines))

    # Starts with a header, then score for each class and finally an average
    header = lines[0]
    cls_lines = lines[1:-1]
    avg_line = lines[-1]

    assert header.split() == ['precision', 'recall', 'f1-score', 'support']
    assert avg_line.split()[0] == 'avg'

    # class names can have spaces - figure the width of the class field
    # using indentation of the precision header
    cls_field_width = len(header) - len(header.lstrip())

    # Now, collect all the class names and score in a dict
    def parse_line(l):
        """Parse a line of classification_report"""
        cls_name = l[:cls_field_width].strip()
        precision, recall, fscore, support = l[cls_field_width:].split()
        precision = float(precision)
        recall = float(recall)
        fscore = float(fscore)
        support = int(support)
        return (cls_name, precision, recall, fscore, support)

    data = collections.OrderedDict()
    for l in cls_lines:
        ret = parse_line(l)
        cls_name = ret[0]
        scores = ret[1:]
        data[cls_name] = scores

    data['avg'] = parse_line(avg_line)[1:]  # average
    return data

# Katie's Code +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def prepare_word2vec():
    import wget
    print("Downloading pretrained word2vec models")
    url = "https://zenodo.org/record/807727/files/word2vecs_models.zip"
    file_name = wget.download(url)
    with zipfile.ZipFile(file_name, "r") as zip_ref:
        zip_ref.extractall()


if __name__ == "__main__":
    word_src = "word2vecs_models"
    threads = []
    warnings.filterwarnings("ignore")
    if not os.path.exists(word_src):
        prepare_word2vec()
    elif len(os.listdir(word_src)) == 0:
        os.rmdir(word_src)
        prepare_word2vec()
    for x in range(1):
        random.seed(x)
        np.random.seed(x)
        myword2vecs = [os.path.join(word_src, i) for i in os.listdir(word_src)
                       if "syn" not in i]
        # run_MLP(myword2vecs[x])
        # run_tuning_MLP(myword2vecs[x])
        run_KNN_baseline(myword2vecs[x])
        run_SVM_baseline(myword2vecs[x])
        print("Run completed for baseline model--------------------------------------------------")
        # run_tuning_SVM(myword2vecs[x])
        # run_tuning_KNN(myword2vecs[x])
        #print("Run completed for DE model--------------------------------------------------")
