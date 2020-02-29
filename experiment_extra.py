from __future__ import division, print_function
import pickle
import pdb
import os
import time
from sklearn.cross_validation import StratifiedKFold
from sklearn import svm
from sklearn import metrics
import gensim
import random
from learners import SK_SVM,SK_KNN,SK_LDA
from tuner import DE_Tune_ML
from model import PaperData
from utility import study
from results import results_process
import numpy as np
#import wget
import zipfile
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import threading
from threading import Barrier
import timeit
import multiprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.lda import LDA
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import collections
from multiprocessing import Queue
import pandas as pd

def tune_learner(learner, train_X, train_Y, tune_X, tune_Y, goal,
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
  if not target_class:
    target_class = goal
  clf = learner(train_X, train_Y, tune_X, tune_Y, goal)
  tuner = DE_Tune_ML(clf, clf.get_param(), goal, target_class)
  return tuner.Tune()


def load_vec(d, data, use_pkl=False, file_name=None):
  if use_pkl:
    if os.path.isfile(file_name):
      with open(file_name, "rb") as my_pickle:
        return pickle.load(my_pickle)
  else:
    # print("call get_document_vec")
    return d.get_document_vec(data, file_name)


def print_results(clfs):
  file_name = time.strftime(os.path.sep.join([".", "results",
                                              "%Y%m%d_%H:%M:%S.txt"]))
  file_name = os.path.sep.join(["20171103.txt"])
  content = ""
  for each in clfs:
    content += each.confusion
  print(content)
  with open(file_name, "w") as f:
    f.write(content)
  results_process.reports(file_name)


def get_acc(cm):
  out = []
  for i in range(4):
    out.append(cm[i][i] / 400)
  return out


@study
def run_tuning_SVM(word2vec_src, repeats=1,
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
  print_results(clfs)

@study
def run_tuning_KNN(word2vec_src, repeats=1,
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
def run_tuning_LDA(word2vec_src, repeats=1,
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
  learner = [SK_LDA][0]
  goal = {0: "PD", 1: "PF", 2: "PREC", 3: "ACC", 4: "F", 5: "G", 6: "Macro_F",
          7: "Micro_F"}[6]
  F = {}
  clfs = []
  for i in range(repeats):  # repeat n times here
    kf = StratifiedKFold(train_pd.loc[:, "LinkTypeId"].values, fold,
                         shuffle=True)
    for train_index, tune_index in kf:
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
  cm=metrics.confusion_matrix(test_Y, predicted, labels=["1", "2", "3", "4"])
  print("accuracy  ", get_acc(cm))
  print("Model training time: ", stop - start)

@study
def run_LDA(word2vec_src): 
  """
  Run LDA+word embedding experiment !
  This is the baseline method.
  :return:None
  """
  # Create a subplot with 1 row and 2 columns
  print("# word2vec:", word2vec_src)
  clf = LDA(solver='lsqr', shrinkage='auto')
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
  cm=metrics.confusion_matrix(test_Y, predicted, labels=["1", "2", "3", "4"])
  print("accuracy  ", get_acc(cm))
  print("Model training time: ", stop - start)
  
@study
def run_LinearDiscriminantAnalysis(word2vec_src): 
  """
  Run LinearDiscriminantAnalysis+word embedding experiment !
  This is the baseline method.
  :return:None
  """
  # Create a subplot with 1 row and 2 columns
  def select_n_components(var_ratio, goal_var: float) -> int:
    # Set initial variance explained so far
    total_variance = 0.0

    # Set initial number of features
    n_components = 0

    # For the explained variance of each feature:
    for explained_variance in var_ratio:

        # Add the explained variance to the total
        total_variance += explained_variance

        # Add one to the number of components
        n_components += 1

        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break

    # Return the number of components
    return n_components
  print("# word2vec:", word2vec_src)
  clf = LinearDiscriminantAnalysis(n_components=None)
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
  lda_var_ratios = clf.explained_variance_ratio_
  n_com = select_n_components(lda_var_ratios, 0.99)
  clf = LinearDiscriminantAnalysis(n_components=n_com)
  clf.fit(train_X, train_Y)
  stop = timeit.default_timer()
  predicted = clf.predict(test_X)

  print(metrics.classification_report(test_Y, predicted,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  cm=metrics.confusion_matrix(test_Y, predicted, labels=["1", "2", "3", "4"])
  print("accuracy  ", get_acc(cm))
  print("Model training time: ", stop - start)  


@study
def run_KNN(word2vec_src):
  """
  Run KNN+word embedding experiment !
  This is the baseline method.
  :return:None
  """
  # Create a subplot with 1 row and 2 columns
  print("# word2vec:", word2vec_src)
  clf = neighbors.KNeighborsClassifier(n_neighbors = 5)
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
  cm=metrics.confusion_matrix(test_Y, predicted, labels=["1", "2", "3", "4"])
  print("accuracy  ", get_acc(cm))
  print("Model training time: ", stop - start)
 
@study
def run_RNN(word2vec_src):
  """
  Run KNN+word embedding experiment !
  This is the baseline method.
  :return:None
  """
  # Create a subplot with 1 row and 2 columns
  print("# word2vec:", word2vec_src)
  clf = neighbors.RadiusNeighborsClassifier(radius=5.0)
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
  cm=metrics.confusion_matrix(test_Y, predicted, labels=["1", "2", "3", "4"])
  print("accuracy  ", get_acc(cm))
  print("Model training time: ", stop - start)    
  
@study
def run_SVM_KNN(word2vec_src):
  """
  Run SVM->KNN+word embedding experiment !
  This is the baseline method.
  :return:None
  """
  classX1 = []
  classX2 = []
  classX3 = []
  classX4 = []
  classY1 = []
  classY2 = []
  classY3 = []
  classY4 = []
  classTX1 = []
  classTX2 = []
  classTX3 = []
  classTX4 = []
  classTY1 = []
  classTY2 = []
  classTY3 = []
  classTY4 = []
  predicted_F = []
  finalY = []
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
  predicted = clf.predict(train_X)
#  predicted = pd.DataFrame(predicted)
#  train_X = pd.DataFrame(train_X)
#  t = predicted.index[predicted.loc[1] == 1].tolist()
#  print(predicted.axes)
#  print(t)
  for i in range(len(predicted)):
      if predicted[i] == '1':
          classX1.append(train_X[i])
          classY1.append(train_Y[i])
      elif predicted[i] == '2':
          classX2.append(train_X[i])
          classY2.append(train_Y[i])
      elif predicted[i] == '3':
          classX3.append(train_X[i])
          classY3.append(train_Y[i])
      elif predicted[i] == '4':
          classX4.append(train_X[i])
          classY4.append(train_Y[i])      
  clf2 = neighbors.KNeighborsClassifier(n_neighbors = 5)
  clf3 = neighbors.KNeighborsClassifier(n_neighbors = 5)
  clf4 = neighbors.KNeighborsClassifier(n_neighbors = 5)
  clf5 = neighbors.KNeighborsClassifier(n_neighbors = 5)
  clf2.fit(classX1,classY1)
  clf3.fit(classX2,classY2)
  clf4.fit(classX3,classY3)
  clf5.fit(classX4,classY4)
  stop = timeit.default_timer()
  predicted0 = clf.predict(test_X)
  for i in range(len(predicted0)):
      if predicted0[i] == '1':
          classTX1.append(test_X[i])
          classTY1.append(test_Y[i])
      elif predicted0[i] == '2':
          classTX2.append(test_X[i])
          classTY2.append(test_Y[i])
      elif predicted0[i] == '3':
          classTX3.append(test_X[i])
          classTY3.append(test_Y[i])
      elif predicted0[i] == '4':
          classTX4.append(test_X[i])
          classTY4.append(test_Y[i])
  predicted1 = clf2.predict(classTX1)
  predicted2 = clf3.predict(classTX2)
  predicted3 = clf4.predict(classTX3)
  predicted4 = clf5.predict(classTX4)  
  finalY = np.append(classTY1, classTY2)
  finalY = np.append(finalY, classTY3)
  finalY = np.append(finalY, classTY4)
  predicted_F = np.append(predicted1, predicted2)
  predicted_F = np.append(predicted_F, predicted3)
  predicted_F = np.append(predicted_F, predicted4)
  print("+++++++++++++++++++Original Predcition Result+++++++++++++++++++++++++")
  print(metrics.classification_report(test_Y, predicted0,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(test_Y, predicted0, labels=["1", "2", "3", "4"])
  print("accuracy  ", get_acc(cm))
  print("+++++++++++++++++++2nd Layer 1st Prediction Model+++++++++++++++++++++++++")
  print(metrics.classification_report(classTY1, predicted1,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(classTY1, predicted1, labels=["1", "2", "3", "4"])
  print("+++++++++++++++++++2nd Layer 2nd Prediction Model+++++++++++++++++++++++++")
  print(metrics.classification_report(classTY2, predicted2,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(classTY2, predicted2, labels=["1", "2", "3", "4"])
  print("+++++++++++++++++++2nd Layer 3rd Prediction Model+++++++++++++++++++++++++")
  print(metrics.classification_report(classTY3, predicted3,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(classTY3, predicted3, labels=["1", "2", "3", "4"])
  print("+++++++++++++++++++2nd Layer 4th Prediction Model+++++++++++++++++++++++++")
  print(metrics.classification_report(classTY4, predicted4,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(classTY4, predicted4, labels=["1", "2", "3", "4"])
  print("+++++++++++++++++++combined result+++++++++++++++++++++++++")
  print(metrics.classification_report(finalY, predicted_F,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  cm=metrics.confusion_matrix(finalY, predicted_F, labels=["1", "2", "3", "4"])
  print("accuracy  ", get_acc(cm))
  print("Model training time: ", stop - start)

@study
def run_SVM_KNN_thread(word2vec_src):
  """
  Run SVM->KNN+word embedding experiment !
  This is the baseline method.
  :return:None
  """
  classX1 = []
  classX2 = []
  classX3 = []
  classX4 = []
  classY1 = []
  classY2 = []
  classY3 = []
  classY4 = []
  classTX1 = []
  classTX2 = []
  classTX3 = []
  classTX4 = []
  classTY1 = []
  classTY2 = []
  classTY3 = []
  classTY4 = []
  TrainingSamplesX = []
  TrainingSamplesY = []
  models = []
  predicted_F = []
  finalY = []
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
  start0 = timeit.default_timer()
  clf.fit(train_X, train_Y)
  stop0 = timeit.default_timer()
  predicted = clf.predict(train_X)
  for i in range(len(predicted)):
      if predicted[i] == '1':
          classX1.append(train_X[i])
          classY1.append(train_Y[i])
      elif predicted[i] == '2':
          classX2.append(train_X[i])
          classY2.append(train_Y[i])
      elif predicted[i] == '3':
          classX3.append(train_X[i])
          classY3.append(train_Y[i])
      elif predicted[i] == '4':
          classX4.append(train_X[i])
          classY4.append(train_Y[i]) 
  TrainingSamplesX.append(classX1)
  TrainingSamplesY.append(classY1)
  TrainingSamplesX.append(classX2)
  TrainingSamplesY.append(classY2)
  TrainingSamplesX.append(classX3)
  TrainingSamplesY.append(classY3)
  TrainingSamplesX.append(classX4)
  TrainingSamplesY.append(classY4)        
  clf2 = neighbors.KNeighborsClassifier(n_neighbors = 5)
  clf3 = neighbors.KNeighborsClassifier(n_neighbors = 5)
  clf4 = neighbors.KNeighborsClassifier(n_neighbors = 5)
  clf5 = neighbors.KNeighborsClassifier(n_neighbors = 5)
  
  models.append(clf2)
  models.append(clf3)
  models.append(clf4)
  models.append(clf5)
  start1 = timeit.default_timer()
  for i in range((len(TrainingSamplesX))):
      t = threading.Thread(target= models[i].fit, args = [TrainingSamplesX[i],TrainingSamplesY[i]])
      threads.append(t)
      t.start()
  stop1 = timeit.default_timer()
  predicted0 = clf.predict(test_X)
  for i in range(len(predicted0)):
      if predicted0[i] == '1':
          classTX1.append(test_X[i])
          classTY1.append(test_Y[i])
      elif predicted0[i] == '2':
          classTX2.append(test_X[i])
          classTY2.append(test_Y[i])
      elif predicted0[i] == '3':
          classTX3.append(test_X[i])
          classTY3.append(test_Y[i])
      elif predicted0[i] == '4':
          classTX4.append(test_X[i])
          classTY4.append(test_Y[i])
  predicted1 = clf2.predict(classTX1)
  predicted2 = clf3.predict(classTX2)
  predicted3 = clf4.predict(classTX3)
  predicted4 = clf5.predict(classTX4)  
  finalY = np.append(classTY1, classTY2)
  finalY = np.append(finalY, classTY3)
  finalY = np.append(finalY, classTY4)
  predicted_F = np.append(predicted1, predicted2)
  predicted_F = np.append(predicted_F, predicted3)
  predicted_F = np.append(predicted_F, predicted4)
  print("+++++++++++++++++++Original Predcition Result+++++++++++++++++++++++++")
  print(metrics.classification_report(test_Y, predicted0,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(test_Y, predicted0, labels=["1", "2", "3", "4"])
  print("accuracy  ", get_acc(cm))
  print("+++++++++++++++++++2nd Layer 1st Prediction Model+++++++++++++++++++++++++")
  print(metrics.classification_report(classTY1, predicted1,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(classTY1, predicted1, labels=["1", "2", "3", "4"])
  print("+++++++++++++++++++2nd Layer 2nd Prediction Model+++++++++++++++++++++++++")
  print(metrics.classification_report(classTY2, predicted2,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(classTY2, predicted2, labels=["1", "2", "3", "4"])
  print("+++++++++++++++++++2nd Layer 3rd Prediction Model+++++++++++++++++++++++++")
  print(metrics.classification_report(classTY3, predicted3,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(classTY3, predicted3, labels=["1", "2", "3", "4"])
  print("+++++++++++++++++++2nd Layer 4th Prediction Model+++++++++++++++++++++++++")
  print(metrics.classification_report(classTY4, predicted4,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(classTY4, predicted4, labels=["1", "2", "3", "4"])
  print("+++++++++++++++++++combined result+++++++++++++++++++++++++")
  print(metrics.classification_report(finalY, predicted_F,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  cm=metrics.confusion_matrix(finalY, predicted_F, labels=["1", "2", "3", "4"])
  print("accuracy  ", get_acc(cm))
  print("1st Model training time: ", (stop0 - start0))
  print("layer 2 Models training time: ", (stop1 - start1))
  print("Total Model training time: ", (stop1 - start0))


@study
def run_KNN_SVM(word2vec_src):
  """
  Run KNN -> SVM+word embedding experiment !
  This is the baseline method.
  :return:None
  """
  classX1 = []
  classX2 = []
  classX3 = []
  classX4 = []
  classY1 = []
  classY2 = []
  classY3 = []
  classY4 = []
  classTX1 = []
  classTX2 = []
  classTX3 = []
  classTX4 = []
  classTY1 = []
  classTY2 = []
  classTY3 = []
  classTY4 = []
  TrainingSamplesX = []
  TrainingSamplesY = []
  models = []
  predicted_F = []
  finalY = []
  # Create a subplot with 1 row and 2 columns
  print("# word2vec:", word2vec_src)
  clf = neighbors.KNeighborsClassifier(n_neighbors = 5)
  word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
  data = PaperData(word2vec=word2vec_model)
  train_pd = load_vec(data, data.train_data, use_pkl=False)
  test_pd = load_vec(data, data.test_data, use_pkl=False)
  train_X = train_pd.loc[:, "Output"].tolist()
  train_Y = train_pd.loc[:, "LinkTypeId"].tolist()
  test_X = test_pd.loc[:, "Output"].tolist()
  test_Y = test_pd.loc[:, "LinkTypeId"].tolist()
  #print("before train")
  start0 = timeit.default_timer()
  clf.fit(train_X, train_Y)
  stop0 = timeit.default_timer()
  predicted = clf.predict(train_X)
  for i in range(len(predicted)):
      if predicted[i] == '1':
          classX1.append(train_X[i])
          classY1.append(train_Y[i])
      elif predicted[i] == '2':
          classX2.append(train_X[i])
          classY2.append(train_Y[i])
      elif predicted[i] == '3':
          classX3.append(train_X[i])
          classY3.append(train_Y[i])
      elif predicted[i] == '4':
          classX4.append(train_X[i])
          classY4.append(train_Y[i])
  TrainingSamplesX.append(classX1)
  TrainingSamplesY.append(classY1)
  TrainingSamplesX.append(classX2)
  TrainingSamplesY.append(classY2)
  TrainingSamplesX.append(classX3)
  TrainingSamplesY.append(classY3)
  TrainingSamplesX.append(classX4)
  TrainingSamplesY.append(classY4) 
  clf2 = svm.SVC(kernel="rbf", gamma=0.005)
  clf3 = svm.SVC(kernel="rbf", gamma=0.005)
  clf4 = svm.SVC(kernel="rbf", gamma=0.005)
  clf5 = svm.SVC(kernel="rbf", gamma=0.005)
  models.append(clf2)
  models.append(clf3)
  models.append(clf4)
  models.append(clf5)
  start1 = timeit.default_timer()
  for i in range((len(TrainingSamplesX))):
      t = threading.Thread(target= models[i].fit, args = [TrainingSamplesX[i],TrainingSamplesY[i]])
      threads.append(t)
      t.start()
  stop1 = timeit.default_timer()
  predicted0 = clf.predict(test_X)
  for i in range(len(predicted0)):
      if predicted0[i] == '1':
          classTX1.append(test_X[i])
          classTY1.append(test_Y[i])
      elif predicted0[i] == '2':
          classTX2.append(test_X[i])
          classTY2.append(test_Y[i])
      elif predicted0[i] == '3':
          classTX3.append(test_X[i])
          classTY3.append(test_Y[i])
      elif predicted0[i] == '4':
          classTX4.append(test_X[i])
          classTY4.append(test_Y[i])
  predicted1 = clf2.predict(classTX1)
  predicted2 = clf3.predict(classTX2)
  predicted3 = clf4.predict(classTX3)
  predicted4 = clf5.predict(classTX4)
  
  finalY = np.append(classTY1, classTY2)
  finalY = np.append(finalY, classTY3)
  finalY = np.append(finalY, classTY4)
  predicted_F = np.append(predicted1, predicted2)
  predicted_F = np.append(predicted_F, predicted3)
  predicted_F = np.append(predicted_F, predicted4)
  print("+++++++++++++++++++Original Predcition Result+++++++++++++++++++++++++")
  print(metrics.classification_report(test_Y, predicted0,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(test_Y, predicted0, labels=["1", "2", "3", "4"])
  print("accuracy  ", get_acc(cm))
  print("+++++++++++++++++++2nd Layer 1st Prediction Model+++++++++++++++++++++++++")
  print(metrics.classification_report(classTY1, predicted1,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(classTY1, predicted1, labels=["1", "2", "3", "4"])
  print("+++++++++++++++++++2nd Layer 2nd Prediction Model+++++++++++++++++++++++++")
  print(metrics.classification_report(classTY2, predicted2,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(classTY2, predicted2, labels=["1", "2", "3", "4"])
  print("+++++++++++++++++++2nd Layer 3rd Prediction Model+++++++++++++++++++++++++")
  print(metrics.classification_report(classTY3, predicted3,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(classTY3, predicted3, labels=["1", "2", "3", "4"])
  print("+++++++++++++++++++2nd Layer 4th Prediction Model+++++++++++++++++++++++++")
  print(metrics.classification_report(classTY4, predicted4,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(classTY4, predicted4, labels=["1", "2", "3", "4"])
  print("+++++++++++++++++++combined result+++++++++++++++++++++++++")
  print(metrics.classification_report(finalY, predicted_F,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  cm=metrics.confusion_matrix(finalY, predicted_F, labels=["1", "2", "3", "4"])
  print("accuracy  ", get_acc(cm))
  print("1st Model training time: ", (stop0 - start0))
  print("layer 2 Models training time: ", (stop1 - start1))
  print("Total Model training time: ", (stop1 - start0))
  
  
@study
def run_KNN_KNN(word2vec_src):
  """
  Run KNN+word embedding experiment !
  This is the baseline method.
  :return:None
  """
  classX1 = []
  classX2 = []
  classX3 = []
  classX4 = []
  classY1 = []
  classY2 = []
  classY3 = []
  classY4 = []
  classTX1 = []
  classTX2 = []
  classTX3 = []
  classTX4 = []
  classTY1 = []
  classTY2 = []
  classTY3 = []
  classTY4 = []
  TrainingSamplesX = []
  TrainingSamplesY = []
  models = []
  predicted_F = []
  finalY = []
  # Create a subplot with 1 row and 2 columns
  print("# word2vec:", word2vec_src)
  #clf = svm.SVC(kernel="rbf", gamma=0.005)
  clf = neighbors.KNeighborsClassifier(n_neighbors = 5)
  #clf = KMeans(n_clusters=4, init='k-means++', max_iter=100, n_init=1)
  word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
  data = PaperData(word2vec=word2vec_model)
  train_pd = load_vec(data, data.train_data, use_pkl=False)
  test_pd = load_vec(data, data.test_data, use_pkl=False)
  train_X = train_pd.loc[:, "Output"].tolist()
  train_Y = train_pd.loc[:, "LinkTypeId"].tolist()
  test_X = test_pd.loc[:, "Output"].tolist()
  test_Y = test_pd.loc[:, "LinkTypeId"].tolist()
  #print("before train")
  start0 = timeit.default_timer()
  clf.fit(train_X, train_Y)
  stop0 = timeit.default_timer()
  predicted = clf.predict(train_X)
  for i in range(len(predicted)):
      if predicted[i] == '1':
          classX1.append(train_X[i])
          classY1.append(train_Y[i])
      elif predicted[i] == '2':
          classX2.append(train_X[i])
          classY2.append(train_Y[i])
      elif predicted[i] == '3':
          classX3.append(train_X[i])
          classY3.append(train_Y[i])
      elif predicted[i] == '4':
          classX4.append(train_X[i])
          classY4.append(train_Y[i])
  #print(classX1)
  TrainingSamplesX.append(classX1)
  TrainingSamplesY.append(classY1)
  TrainingSamplesX.append(classX2)
  TrainingSamplesY.append(classY2)
  TrainingSamplesX.append(classX3)
  TrainingSamplesY.append(classY3)
  TrainingSamplesX.append(classX4)
  TrainingSamplesY.append(classY4)
  clf2 = neighbors.KNeighborsClassifier(n_neighbors = 10)
  clf3 = neighbors.KNeighborsClassifier(n_neighbors = 10)
  clf4 = neighbors.KNeighborsClassifier(n_neighbors = 10)
  clf5 = neighbors.KNeighborsClassifier(n_neighbors = 10)
  models.append(clf2)
  models.append(clf3)
  models.append(clf4)
  models.append(clf5)
  start1 = timeit.default_timer()
  for i in range((len(TrainingSamplesX))):
      t = threading.Thread(target= models[i].fit, args = [TrainingSamplesX[i],TrainingSamplesY[i]])
      threads.append(t)
      t.start()
  stop1 = timeit.default_timer()
  predicted0 = clf.predict(test_X)
  for i in range(len(predicted0)):
      if predicted0[i] == '1':
          classTX1.append(test_X[i])
          classTY1.append(test_Y[i])
      elif predicted0[i] == '2':
          classTX2.append(test_X[i])
          classTY2.append(test_Y[i])
      elif predicted0[i] == '3':
          classTX3.append(test_X[i])
          classTY3.append(test_Y[i])
      elif predicted0[i] == '4':
          classTX4.append(test_X[i])
          classTY4.append(test_Y[i])
  predicted1 = clf2.predict(classTX1)
  predicted2 = clf3.predict(classTX2)
  predicted3 = clf4.predict(classTX3)
  predicted4 = clf5.predict(classTX4)
  
  finalY = np.append(classTY1, classTY2)
  finalY = np.append(finalY, classTY3)
  finalY = np.append(finalY, classTY4)
  predicted_F = np.append(predicted1, predicted2)
  predicted_F = np.append(predicted_F, predicted3)
  predicted_F = np.append(predicted_F, predicted4)
  print("+++++++++++++++++++Original Predcition Result+++++++++++++++++++++++++")
  print(metrics.classification_report(test_Y, predicted0,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(test_Y, predicted0, labels=["1", "2", "3", "4"])
  print("accuracy  ", get_acc(cm))
  print("+++++++++++++++++++2nd Layer 1st Prediction Model+++++++++++++++++++++++++")
  print(metrics.classification_report(classTY1, predicted1,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(classTY1, predicted1, labels=["1", "2", "3", "4"])
  print("+++++++++++++++++++2nd Layer 2nd Prediction Model+++++++++++++++++++++++++")
  print(metrics.classification_report(classTY2, predicted2,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(classTY2, predicted2, labels=["1", "2", "3", "4"])
  print("+++++++++++++++++++2nd Layer 3rd Prediction Model+++++++++++++++++++++++++")
  print(metrics.classification_report(classTY3, predicted3,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(classTY3, predicted3, labels=["1", "2", "3", "4"])
  print("+++++++++++++++++++2nd Layer 4th Prediction Model+++++++++++++++++++++++++")
  print(metrics.classification_report(classTY4, predicted4,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  #print("print classification data")
  cm=metrics.confusion_matrix(classTY4, predicted4, labels=["1", "2", "3", "4"])
  print("+++++++++++++++++++combined result+++++++++++++++++++++++++")
  print(metrics.classification_report(finalY, predicted_F,
                                      labels=["1", "2", "3", "4"],
                                      digits=3))
  cm=metrics.confusion_matrix(finalY, predicted_F, labels=["1", "2", "3", "4"])
  print("accuracy  ", get_acc(cm))
  print("1st Model training time: ", (stop0 - start0))
  print("layer 2 Models training time: ", (stop1 - start1))
  print("Total Model training time: ", (stop1 - start0))


@study
def run_KMeans_Wpair(word2vec_src):
  """
  Run KMeans+word embedding experiment !
  This is the baseline method.
  :return:None
  """
  # Create a subplot with 1 row and 2 columns
  print("# word2vec:", word2vec_src)
  #clf = svm.SVC(kernel="rbf", gamma=0.005)
  #clf = neighbors.KNeighborsClassifier(n_neighbors = 5)
  clf = KMeans(n_clusters=4, init='k-means++', max_iter=100, n_init=1)
  word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
  data = PaperData(word2vec=word2vec_model)
  train_pd = load_vec(data, data.train_data, use_pkl=False)
  test_pd = load_vec(data, data.test_data, use_pkl=False)
  train_X = train_pd.loc[:, "PostIdVec"].tolist()
  train_Y = train_pd.loc[:, "LinkTypeId"].tolist()
  train_X1 = train_pd.loc[:, "RelatedPostIdVec"].tolist()
  train_Y1 = train_pd.loc[:, "LinkTypeId"].tolist()
  np.append(train_X,train_X1)
  np.append(train_Y,train_Y1)
  test_X = test_pd.loc[:, "PostIdVec"].tolist()
  test_Y = test_pd.loc[:, "LinkTypeId"].tolist()
  clf.fit(train_X, train_Y)
  predicted = clf.predict(test_X)
  print(predicted)
  x =  list(np.asarray(clf.labels_) + 1)
  print("Homogeneity: %0.3f" % metrics.homogeneity_score(train_Y, x))
  print("Completeness: %0.3f" % metrics.completeness_score(train_Y, clf.labels_))
  print("V-measure: %0.3f" % metrics.v_measure_score(train_Y, clf.labels_))
  print("Adjusted Rand-Index: %.3f"
        % metrics.adjusted_rand_score(train_Y, clf.labels_))
  print("Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(train_X, clf.labels_, sample_size=1000))

#################Katie's Code +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# returns the svm model
def run_SVM(word2vec_src, train_pd, queue):
  clf = svm.SVC(kernel="rbf", gamma=0.005)
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
  queue.put(clf)
  return clf
def run_KNN_clustering(word2vec_src, train_pd, queue):
  print("# word2vec:", word2vec_src)
  clf = neighbors.KNeighborsClassifier(n_neighbors = 10)
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
  queue.put(clf)
  return clf

@study
def run_tuning_SVM_C(word2vec_src,train_pd_c,queue, repeats=1,
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
  train_pd_c = train_pd_c.reset_index()
  train_pd = train_pd_c
  test_pd = load_vec(data, data.test_data, file_name=False)
  learner = [SK_SVM][0]
  goal = {0: "PD", 1: "PF", 2: "PREC", 3: "ACC", 4: "F", 5: "G", 6: "Macro_F",
          7: "Micro_F"}[6]
  print(goal)
  F = {}
  clfs = []
  for i in range(repeats):  # repeat n times here
    kf = StratifiedKFold(train_pd.loc[:, "LinkTypeId"].values, fold,
                         shuffle=True)
    for train_index, tune_index in kf:
      print(train_pd)
      train_data = train_pd.ix[train_index]
      print(train_index)
      print(train_data)
      tune_data = train_pd.ix[tune_index]
      train_X = train_data.loc[:, "Output"].values
      print(train_X)
      train_Y = train_data.loc[:, "LinkTypeId"].values
      print(train_Y)
      tune_X = tune_data.loc[:, "Output"].values
      tune_Y = tune_data.loc[:, "LinkTypeId"].values
      test_X = test_pd.loc[:, "Output"].values
      test_Y = test_pd.loc[:, "LinkTypeId"].values
      params, evaluation = tune_learner(learner, train_X, train_Y, tune_X,
                                        tune_Y, goal) if tuning else ({}, 0)
      clf = learner(train_X, train_Y, test_X, test_Y, goal)
      F = clf.learn(F, **params)
      clfs.append(clf)
  queue.put(clfs)    
  print_results(clfs)

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
  #print("accuracy  ", get_acc(cm)


def total_summary(result_set, num_rows, start0,start1,stop0,stop1):
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
  print(result)
  print("1st Model training time: ", (stop0 - start0))
  print("layer 2 Models training time: ", (stop1 - start1))
  print("Total Model training time: ", (stop1 - start0))
  
def run_kmeans(word2vec_src):

  print("# word2vec:", word2vec_src)
  word2vec_model = gensim.models.Word2Vec.load(word2vec_src)
  data = PaperData(word2vec=word2vec_model)
  train_pd = load_vec(data, data.train_data, use_pkl=False)
  test_pd = load_vec(data, data.test_data, use_pkl=False)
  train_X = train_pd.loc[:, "Output"].tolist()
  queue = Queue()

  numClusters = optimalK(train_X)
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
  #b = Barrier(numClusters-1)
  for l in range(numClusters):
    cluster = data.train_data.loc[data.train_data['clabel'] == l] 
    t = threading.Thread(target=run_tuning_SVM_C, args = [word2vec_src,cluster,queue])
    threads.append(t)
    t.start()
    response = queue.get()
    svm_models.append(response)
  #b.wait()
  t.join()
  stop1 = timeit.default_timer()

  svm_results = [] # maintain a list of svm results
  test_X = test_pd.loc[:, "Output"].tolist()
  predicted = clf.predict(test_X)
  data.test_data['clabel'] = predicted
  for l in range(numClusters):
    #print("Label " + str(l))
    cluster = data.test_data.loc[data.test_data['clabel'] == l]
    svm_model = svm_models[l]
    cluster_X = cluster.loc[:, "Output"].tolist()
    cluster_Y = cluster.loc[:, "LinkTypeId"].tolist()
    svm_results.append(results_SVM(svm_model, cluster_X, cluster_Y))# store all the SVM result report in a dictionary

    # call the helper method to summarize the svm results
  total_summary(svm_results, test_pd.shape[0],start0,start1,stop0,stop1)

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
      # randomReference = np.random.random_sample(size=data.shape)

      # Fit to it
      km = KMeans(n_clusters=k, init='k-means++', max_iter=200, n_init=1)
      km.fit(data)

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
  print("Homogeneity: %0.3f" % metrics.homogeneity_score(train_Y, clf.labels_))
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
  
#################Katie's Code +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def prepare_word2vec():
  print("Downloading pretrained word2vec models")
  url = "https://zenodo.org/record/807727/files/word2vecs_models.zip"
  file_name = wget.download(url)
  with zipfile.ZipFile(file_name, "r") as zip_ref:
    zip_ref.extractall()


if __name__ == "__main__":
  word_src = "word2vecs_models"
  threads = []
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
#    t = threading.Thread(target=run_tuning_SVM_KNN, args = [myword2vecs[x]])
#    threads.append(t)
#    t.start()
    run_SVM_baseline(myword2vecs[x])
    #run_SVM_KNN_thread(myword2vecs[x])
    #run_LinearDiscriminantAnalysis(myword2vecs[x])
    #run_KNN(myword2vecs[x])
    #run_SVM_KNN(myword2vecs[x])
    #run_KMeans_Wpair(myword2vecs[x])
    #run_kmeans(myword2vecs[x])
    #run_KNN_SVM(myword2vecs[x])
    #run_KNN_KNN(myword2vecs[x])
    #Srun_LDA(myword2vecs[x])
    #run_RNN(myword2vecs[x])
    #print("Run completed for baseline model--------------------------------------------------")
    #run_tuning_SVM(myword2vecs[x])
    #run_tuning_LDA(myword2vecs[x])
    #run_tuning_KNN(myword2vecs[x])
    #print("Run completed for DE model--------------------------------------------------")