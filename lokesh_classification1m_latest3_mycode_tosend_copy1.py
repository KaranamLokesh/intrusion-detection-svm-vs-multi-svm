


import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
# from PIL import image
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
#import mahotas
import h5py
import six
from sklearn.decomposition import PCA
from imblearn.ensemble import EasyEnsemble,BalanceCascade
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
import os
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import sys
import matplotlib.pyplot as plt
import pandas as pd
# import scipy
from skimage import feature
from skimage.feature import greycomatrix,greycoprops
from sklearn import neighbors, datasets, preprocessing
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.feature import hog
from skimage import data, exposure
from skimage.morphology import disk

#Packages for training

import glob
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,mean_squared_error,mean_absolute_error
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

# For ROC
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import model_selection
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


from sklearn import svm, datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
#from ggplot import *

# For Training models

import h5py
import numpy as np
import os
import glob
import cv2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# from skrvm import RVC

import csv
from scipy import stats

from sklearn.metrics import mean_squared_error,r2_score
# from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier


# filter all the warnings
import warnings
warnings.filterwarnings('ignore')

from time import sleep

# Feature selection:
from sklearn.feature_selection import RFECV, RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import make_pipeline
# from skrebate import ReliefF
# import ibmdbpy.feature_selection.gain_ratio
from sklearn.metrics.pairwise import pairwise_distances
import math
from random import randint
from copy import deepcopy
from fractions import Fraction
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier

import pickle
# %run GainRatioClass.ipynb

from sklearn.svm import SVR
from sklearn.linear_model import Lasso
# Lables:
# 1: class2
# 2: class3
# 3: class5
# 4: class4
# 5: class10

# data1m = pd.read_csv('C:/Users/pcuv48/Desktop/2020_Data/1m/FinalGT1m_Combined.csv') # Training Data
data1m = pd.read_csv('FinalGT1m_2m_5m_Combined_Train+Test.csv')

# data1m = pd.read_csv('C:/Users/pcuv48/Desktop/2020_Data/1m/FinalGT1m_Combined_ForHistMatch.csv')  # Training + Testing Data

subsetting = data1m['Subsetting'].astype('category')
tape=subsetting.cat.categories.tolist()
print (tape)
print("Total number of tapes",len(tape))

DataCategory = data1m['DataType'].astype('category')
DataClass = data1m['Class5'].astype('category')

# # Data location address:
# DataAddress = "C:/Users/parth/Desktop/2020_Data/1m/FeaturesAndLabelsStored1m/LBP_Radius/LBP3/"
# LabelAddress = "C:/Users/parth/Desktop/2020_Data/1m/FeaturesAndLabelsStored1m/LBP_Radius/LBP3/CombinedLabel/"
# AnalysisAddress = "C:/Users/parth/Desktop/2020_Data/1m/FeaturesAndLabelsStored1m/LBP_Radius/LBP3/Analysis/Combined/Class5/"

# Loading Features and Label vectors
global_features_load = np.load("Data_81Features.npy")
global_label5 = np.load("label_70_Features.npy")     # Class-5 labels

#AnalysisAddress = "C:/Users/parth/Desktop/2020_Data/ForLokesh/Dataset_WithoutShape/WithLBP2/"

# Based on a Class problem, assigning label variable:
finalLabel = global_label5
# finalLabel = npy_percet

global_features_load = global_features_load.reshape(88,50,81)

global_features_load[0][0]

data1 = []      # It will store all images in 1D array
data2 = []      # It will store number of images per tape
num1 = 0
for i in range(len(global_features_load)):
    num1 = len(global_features_load[i])
    data2.append(num1)

for z in global_features_load:
    for i in z:
        data1.append(i)
print ("Total number of dimension Before PCA or LDA:",len(data1[0]))
print ("Total number of data:",len(data1))



# Getting 1D array of all labels- class#

label1 = []      # It will store all images in 1D array
label2 = []      # It will store number of images per tape
num1 = 0
for i in range(len(finalLabel)):
    num1 = len(finalLabel[i])
    label2.append(num1)

for z in finalLabel:
    for i in z:
        label1.append(i)

# train_test_split size
test_size = 0.2

# seed for reproducing same results
seed = 9

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainGroundsGlobal, testGroundsGlobal) = train_test_split(np.array(data1),
                                                                                          np.array(label1),
                                                                                          test_size=test_size,
                                                                                          random_state=seed,
                                                                                           stratify = label1)

print ("[STATUS] splitted train and test data...")
print ("Train data  : {}".format(trainDataGlobal.shape))
print ("Test data   : {}".format(testDataGlobal.shape))
print ("Train labels: {}".format(trainGroundsGlobal.shape))
print ("Test labels : {}".format(testGroundsGlobal.shape))

# Assigning variable name as per my code
data1 = []
label1 = []
data1 = trainDataGlobal
label1 = trainGroundsGlobal

testData = testDataGlobal
testLabel = testGroundsGlobal

# from sklearn.neighbors import LocalOutlierFactor
# lof = LocalOutlierFactor(n_neighbors=2)
# yhat = lof.fit_predict(data1)
# mask = yhat != -1
# data1, label1 = data1[mask, :], label1[mask]

# pca = PCA()
# data1 = pca.fit_transform(data1)
# testData = pca.transform(testData)

# under_sampler = BalanceCascade(random_state=0)
# data1,label1 = under_sampler.fit_resample(data1, label1)



def remove_outliers_bis(arr, k):
    mask = np.ones((arr.shape[0],), dtype=np.bool)
    mu, sigma = np.mean(arr, axis=0), np.std(arr, axis=0, ddof=1)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        mask[mask] &= np.abs((col[mask] - mu[j]) / sigma[j]) < k
    return arr[mask]

x = remove_outliers_bis(data1,3)



data1.shape

# Creating Different Dataset for bi-class problem

Data12 = []
Data23 = []
Data34 = []
Data45 = []
Data15 = []

Label12 = []
Label23 = []
Label34 = []
Label45 = []
Label15 = []

# Creating dataset for class1 and class2 only
for i in range(len(data1)):
    if (label1[i] == 1 or label1[i] == 2):
        Data12.append(data1[i])
        Label12.append(label1[i])

# Creating dataset for class2 and class3 only
for i in range(len(data1)):
    if (label1[i] == 2 or label1[i] == 3):
        Data23.append(data1[i])
        Label23.append(label1[i])

# Creating dataset for class3 and class4 only
for i in range(len(data1)):
    if (label1[i] == 3 or label1[i] == 4):
        Data34.append(data1[i])
        Label34.append(label1[i])

# Creating dataset for class4 and class5 only
for i in range(len(data1)):
    if (label1[i] == 4 or label1[i] == 5):
        Data45.append(data1[i])
        Label45.append(label1[i])

# Creating dataset for class1 and class5 only
for i in range(len(data1)):
    if (label1[i] == 1 or label1[i] == 5):
        Data15.append(data1[i])
        Label15.append(label1[i])

data1.shape



# Number of dataset for each two classes problem
print (len(Label12))
print (len(Label23))
print (len(Label34))
print (len(Label45))
print (len(Label15))

def featureRanking(features):
    featureSequence = list(range(len(features)))
    sortFeatures = []
    for f in range(len(features)):
        for i in range(len(features)):
            if(featureSequence[f] == features[i]):
                sort1 = i
                sortFeatures.append(sort1)
                break
#     print (sortFeatures)
    return sortFeatures

# Defining selected importance
def selectRFEImportance(order1, X, k=5):
    return X[:,order1[::1][:k]]

# Feature selection method - Recursive Feature Elimination (RFECV)
estimator = SVC(kernel='linear')
selector2 = RFECV(estimator, step = 1, scoring = 'accuracy', cv = StratifiedKFold(n_splits=10))
# selector2 = RFE(estimator, n_features_to_select=1)
selector = selector2.fit(data1,label1)
print("[STATUS]Optimal number of features after RFECV: ", selector.n_features_)

rank = selector.ranking_
ranks = sorted(range(len(rank)),key=rank.__getitem__)
print("Features in descending order based on ranking: ",ranks)

# To get ranking of selected features from RFECV using RFE
################################################
fea = []
fea = selectRFEImportance(ranks,np.array(data1),selector.n_features_) # Change "k" as per optimal number of features "selector.n_features_" or manually.
selector3 = RFE(estimator, n_features_to_select=1)
selector4 = selector3.fit(fea,label1)

rank1 = selector4.ranking_
ranks1 = sorted(range(len(rank1)),key=rank1.__getitem__)
print("Features in descending order based on ranking using RFE: ",ranks1)
temp = []
for i in range(len(rank1)):
    temp.append(ranks[ranks1[i]])

for i in range(len(rank1)):
    ranks[i] = temp[i]
###################################################

print("Features in descending order based on ranking (Final): ",ranks)
OriginalRank6 = featureRanking(ranks)
# print (OriginalRank6)



# Getting n selected features for RFE:
features6 = []
features6Rank = []
for i in range(len(data1[0])):
    features6Rank.append(ranks[:i+1])
    features6.append(selectRFEImportance(ranks,np.array(data1),i+1))


# text = open(AnalysisAddress+"/FeatureRankings.txt","a")
# text.write("\n\nOptimal number of features after RFECV:\n %i"%selector.n_features_)
# text.write("\nRecursive Feature Elimination Rankings Cross -Validation Rankings:\n %s"%ranks)
# text.write("\n\nOriginal Recursive Feature Elimination Cross -Validation Rankings:\n %s"%OriginalRank6)
# text.close()

# Plot number of features VS. cross-validation scores
plt.figure()
plt.title("Reccurssive Feature Elimination based Feature Selection")
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.savefig('FeatureImportance_RFE.png')
plt.show()


# Getting selected features

SelectedFeatures = []
for i in range(len(data1)):
    SelectedFeatures.append([f for f,s in zip(data1[i], selector.support_) if s])
print('The selected features are:')
# print ('{}'.format(SelectedFeatures))
print(len(SelectedFeatures[0]))
data0 = SelectedFeatures
data0 = np.array(data0)



# Validation of each feature selection methods
svc = SVC(probability=True,class_weight='balanced')
clf = svc
# clf = RandomForestClassifier()

# variables to hold the results and names
cv_results6 = []
scoring = "accuracy"

# kfold cross validation:
# kfold = KFold(n_splits=10)

# # Stratified kfold cross validation:
kfold = StratifiedKFold(n_splits=10)

for i in range(len(features6)):
#     print (i)

    cv_result6 = cross_val_score(clf, features6[i], label1, cv=kfold, scoring=scoring)
    cv_results6.append(cv_result6.mean())

    df = pd.read_csv('RFECV_Score.csv')
    for j in range(10):
        df.at[j+(i*10),'Epoch']=i+1
        df.at[j+(i*10),'Iteration']=j+1
        df.at[j+(i*10),'Score']=cv_result6[j]
        df.at[j+(i*10),'MeanScore']=cv_result6.mean()
    df.to_csv('RFECV_Score.csv',index = False)


# Plot number of features VS. cross-validation scores with selected features-6
plt.figure()
plt.title("Reccursive Feature Elemination based feature selection")
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(features6) + 1), cv_results6)
plt.savefig('FeatureValidation_RFE.png')
plt.show()

cv_results6.index(max(cv_results6))

cv_results6[41],cv_results6[42]

x =cv_results6.index(max(cv_results6))

##select the top 42 features which give high cross validation score
ranks = ranks1[:41]
len(ranks)

cols = ranks
len(cols)

# Transforming bi-class based dataset with selected features only
dataImp = Data12
dataImp = np.array(dataImp)
dta = list( dataImp[:,i] for i in cols )
dta = np.transpose(dta)
Data12 = dta

dataImp = Data23
dataImp = np.array(dataImp)
dta = list( dataImp[:,i] for i in cols )
dta = np.transpose(dta)
Data23 = dta

dataImp = Data34
dataImp = np.array(dataImp)
dta = list( dataImp[:,i] for i in cols )
dta = np.transpose(dta)
Data34 = dta

dataImp = Data45
dataImp = np.array(dataImp)
dta = list( dataImp[:,i] for i in cols )
dta = np.transpose(dta)
Data45 = dta

dataImp = Data15
dataImp = np.array(dataImp)
dta = list( dataImp[:,i] for i in cols )
dta = np.transpose(dta)
Data15 = dta

dataImp = data1
dataImp = np.array(dataImp)
# Creating dataset with only selected ELEVEN features in main model. Features - 1, 7, 23, 25, 29, 32, 38, 40, 41, 42, 48.
# dta = list( dataImp[:,i] for i in [26, 45, 46, 48, 50, 55, 56] )  #2m Features
# dta = list( dataImp[:,i] for i in [4, 26, 45, 55, 56, 46] )  #5m Features
# dta = list( dataImp[:,i] for i in [26, 38, 42, 44, 47, 51, 57, 58, 62, 63, 64, 55, 46] )  #1m Features
dta = list( dataImp[:,i] for i in cols )

dta = np.transpose(dta)
data1 = dta
print (len(data1[0]))

dataImp1 = testData

# Creating dataset with only selected ELEVEN features in main model. Features - 1, 7, 23, 25, 29, 32, 38, 40, 41, 42, 48.
# dta1 = list( dataImp1[:,i] for i in [26, 45, 46, 48, 50, 55, 56] )  #2m Features
# dta1 = list( dataImp1[:,i] for i in [4, 26, 45, 55, 56, 46] )  #5m Features
# dta1 = list( dataImp1[:,i] for i in [26, 38, 42, 44, 47, 51, 57, 58, 62, 63, 64, 55, 46] )  #1m Features
dta1 = list( dataImp1[:,i] for i in cols )

dta1 = np.transpose(dta1)
testData = dta1
print (len(testData[0]))

##starting estimator for each classifier
SVMEstimator = SVC(C = 100, kernel='rbf')
svc = SVC(probability=True,class_weight='balanced')

##parameters for grid search
parameters = {'kernel':('linear', 'rbf'), 'C':[ 0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
# parameters_ovo = {'estimator__C':[1,10,100],'estimator__kernel':['rbf']}
clf_cv  =  GridSearchCV(svc, parameters)

##classifiers for binary classification
clf12  =  GridSearchCV(svc, parameters)
clf23  =  GridSearchCV(svc, parameters)
clf34  =  GridSearchCV(svc, parameters)
clf45  =  GridSearchCV(svc, parameters)

# from sklearn.neighbors import KernelDensity
# kde12 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(Data12)
# Data12 = kde12.score_samples(Data12)
# kde23 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(Data23)
# Data23 = kde23.score_samples(Data23)
# kde34 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(Data34)
# Data34 = kde34.score_samples(Data34)
# kde45 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(Data45)
# Data45 = kde45.score_samples(Data45)

# Data12 = Data12.reshape(-1,1)
# Data23 = Data23.reshape(-1,1)
# Data34 = Data34.reshape(-1,1)
# Data45 = Data45.reshape(-1,1)

Data34.shape

# from sklearn.neighbors import KernelDensity

# def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs):
#     """Build 2D kernel density estimate (KDE)."""

#     # create grid of sample locations (default: 100x100)
#     xx, yy = np.mgrid[x.min():x.max():xbins,
#                       y.min():y.max():ybins]

#     xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
#     xy_train  = np.vstack([y, x]).T

#     kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
#     kde_skl.fit(xy_train)

#     # score_samples() returns the log-likelihood of the samples
#     z = np.exp(kde_skl.score_samples(xy_sample))
#     return xx, yy, np.reshape(z, xx.shape)

# import scipy.stats as stats
# from sklearn.neighbors import KernelDensity
# kde34 = stats.gaussian_kde(Data34)
# Data34 = kde34.evaluate(Data34)

testData.shape

##classifiers for binary classification
clf12  =  clf12.fit(Data12,Label12)
print (1)
clf23  =  clf23.fit(Data23,Label23)
print (2)
clf34  =  clf34.fit(Data34,Label34)
print (3)
clf45  =  clf45.fit(Data45,Label45)

# Set the svc to the best combination of parameters
svc12 = clf12.best_estimator_
svc23 = clf23.best_estimator_
svc34 = clf34.best_estimator_
svc45 = clf45.best_estimator_

## best parameters
print(svc12)
print(svc23)
print(svc34)
print(svc45)



scoring = "accuracy"
model = SVMEstimator

# For validation Score
kfold = KFold(n_splits=10, random_state=None)

cv_results12 = cross_val_score(svc12, Data12, Label12, cv=kfold, scoring=scoring)
cv_results23 = cross_val_score(svc23, Data23, Label23, cv=kfold, scoring=scoring)
cv_results34 = cross_val_score(svc34, Data34, Label34, cv=kfold, scoring=scoring)
cv_results45 = cross_val_score(svc45, Data45, Label45, cv=kfold, scoring=scoring)

cv_results = cross_val_score(model, data1, label1, cv=kfold, scoring=scoring)


msg = "%s: %f (%f)" % ("Class12", cv_results12.mean(), cv_results12.std())
print(msg)
msg = "%s: %f (%f)" % ("Class23", cv_results23.mean(), cv_results23.std())
print(msg)
msg = "%s: %f (%f)" % ("Class34", cv_results34.mean(), cv_results34.std())
print(msg)
msg = "%s: %f (%f)" % ("Class45", cv_results45.mean(), cv_results45.std())
print(msg)

msg = "%s: %f (%f)" % ("AllClass", cv_results.mean(), cv_results.std())
print(msg)



# Fitting datasets

svc12.fit(Data12,Label12)
svc23.fit(Data23,Label23)
svc34.fit(Data34,Label34)
svc45.fit(Data45,Label45)

total_data = len(data1)
weightage_1 = total_data/(2*len(Data12))
weightage_2 = total_data/(2*len(Data23))
weightage_3 = total_data/(2*len(Data34))
weightage_4 = total_data/(2*len(Data45))

(weightage_4)



##get the probability of each class
predict_proba_12 = svc12.predict_proba(data1)
predict_proba_23 = svc23.predict_proba(data1)
predict_proba_34 = svc34.predict_proba(data1)
predict_proba_45 = svc45.predict_proba(data1)



#y = np.vstack((proba_train_weight_1,proba_train_weight_2,proba_train_weight_3,proba_train_weight_4,proba_train_weight_5)).T

# train_1 = predict_proba_12[:,0]
# train_2 = predict_proba_12[:,1]
# train_3 = predict_proba_23[:,0]
# train_4 = predict_proba_23[:,1]
# train_5 = predict_proba_34[:,0]
# train_6 = predict_proba_34[:,1]
# train_7 = predict_proba_45[:,0]
# train_8 = predict_proba_45[:,1]

predict_proba_12

##calculating new features based on predict proba method

proba_train_weight_1 =predict_proba_12[:,0]
proba_train_weight_2 = (predict_proba_23[:,0])
proba_train_weight_3 = (predict_proba_34[:,0])
proba_train_weight_4 =(predict_proba_45[:,0])
# proba_train_weight_5 =weightage_4* predict_proba_45[:,1]

# #sorted in descending order
# sorted_a = np.sort(proba_train_weight_1)[::-1]
# sorted_b = np.sort(proba_train_weight_2)[::-1]
# sorted_c = np.sort(proba_train_weight_3)[::-1]
# sorted_d = np.sort(proba_train_weight_4)[::-1]
# sorted_e = np.sort(proba_train_weight_5)[::-1]

# s_1 = (sorted_a - sorted_b)*(1/1)
# s_2 = (sorted_b - sorted_c)*(1/2)
# s_3 = (sorted_c - sorted_d)*(1/3)
# s_4 = (sorted_d - sorted_e)*(1/4)

# ## normalising data
# norm_1 = np.linalg.norm(proba_train_weight_1)
# proba_train_weight_1 = proba_train_weight_1/norm_1
# norm_2 = np.linalg.norm(proba_train_weight_2)
# proba_train_weight_2 = proba_train_weight_2/norm_2
# norm_3 = np.linalg.norm(proba_train_weight_3)
# proba_train_weight_3 = proba_train_weight_3/norm_3
# norm_4 = np.linalg.norm(proba_train_weight_4)
# proba_train_weight_4 = proba_train_weight_4/norm_4
# norm_5 = np.linalg.norm(proba_train_weight_5)
# proba_train_weight_5 = proba_train_weight_5/norm_5

##standardizing te data
proba_train_weight_1 = (proba_train_weight_1 - proba_train_weight_1.mean())/(proba_train_weight_1.std())
proba_train_weight_2 = (proba_train_weight_2 - proba_train_weight_2.mean())/(proba_train_weight_2.std())
proba_train_weight_3 = (proba_train_weight_3 - proba_train_weight_3.mean())/(proba_train_weight_3.std())
proba_train_weight_4 = (proba_train_weight_4 - proba_train_weight_4.mean())/(proba_train_weight_4.std())
# proba_train_weight_5 = (proba_train_weight_5 - proba_train_weight_5.mean())/(proba_train_weight_5.std())
#y = np.vstack((proba_train_weight_1,proba_train_weight_2,proba_train_weight_3,proba_train_weight_4,proba_train_weight_5)).T

# y = np.vstack((s_1,s_2,s_3,s_4,proba_train_weight_1,proba_train_weight_2,proba_train_weight_3,proba_train_weight_4,proba_train_weight_5)).T

y = np.vstack((proba_train_weight_1,proba_train_weight_2,proba_train_weight_3,proba_train_weight_4)).T

# y = np.vstack((train_1,train_2,train_3,train_4,train_5,train_6,train_7,train_8)).T

print(y[0])
print(y[1])
print(y[10])

data_all_method3 = np.append(data1,y,axis=1)

print(data_all_method3.shape)

# Full training Data(which has all classes)
clf_cv  =  GridSearchCV(svc, parameters)
clf_cv  =  clf_cv.fit(data1,label1)
svcFinal = clf_cv.best_estimator_

clf_clf  = svcFinal
# fit the training data to the model
clf_clf.fit(data1,label1)
#print(clf_clf.best_params_)

cv_results1 = cross_val_score(svcFinal, data1, label1, cv=10, scoring="accuracy")

msg = "%s: %f (%f)" % ("AllClass", cv_results1.mean(), cv_results1.std())
print(msg)



from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_pred = cross_val_predict(svcFinal, data1, label1, cv=kfold)
conf_mat = confusion_matrix(label1, y_pred)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print(confusion_matrix(label1, y_pred))
print(classification_report(label1, y_pred))

from sklearn.metrics import f1_score,recall_score,accuracy_score,plot_roc_curve
from sklearn.metrics import precision_score
print("accuracy:", accuracy_score(label1, y_pred))
print("precision:", precision_score(label1, y_pred, average='weighted'))
print("recall:", recall_score(label1, y_pred, average='weighted'))
print("f1_score:", f1_score(label1, y_pred, average='weighted'))

# Full training Data(which has all classes)
clf_cv2  =  GridSearchCV(svc, parameters)
clf_cv2  =  clf_cv2.fit(data_all_method3,label1)
svcFinal2 = clf_cv2.best_estimator_

clf_method2  = svcFinal2
# fit the training data to the model
clf_method2.fit(data_all_method3,label1)
#print(clf_clf.best_params_)

cv_results2 = cross_val_score(svcFinal2, data_all_method3, label1, cv=kfold, scoring=scoring)

msg = "%s: %f (%f)" % ("AllClass", cv_results2.mean(), cv_results2.std())
print(msg)

y_pred = cross_val_predict(svcFinal2, data_all_method3, label1, cv=kfold)
conf_mat = confusion_matrix(label1, y_pred)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print(confusion_matrix(label1, y_pred))
print(classification_report(label1, y_pred))

from sklearn.metrics import f1_score,recall_score,accuracy_score,plot_roc_curve
from sklearn.metrics import precision_score
print("accuracy:", accuracy_score(label1, y_pred))
print("precision:", precision_score(label1, y_pred, average='weighted'))
print("recall:", recall_score(label1, y_pred, average='weighted'))
print("f1_score:", f1_score(label1, y_pred, average='weighted'))

## for whole data svm
parameters_ovo = {'estimator__C':[1,10,100,1000],'estimator__kernel':['rbf','linear'],'estimator__gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
##OneVsOne Classifier for all training set
ovo = OneVsOneClassifier(svc)
# Training data after extracting probabilities
clf_method3  = GridSearchCV(ovo, parameters_ovo)
# fit the training data to the model
clf_method3.fit(data_all_method3,label1)
print(clf_method3.best_params_)

# cv_results3 = cross_val_score(clf_method3, data_all_method3, label1, cv=kfold, scoring=scoring)

# msg = "%s: %f (%f)" % ("AllClass", cv_results3.mean(), cv_results3.std())
# print(msg)

# For 5 separate models
# Test model on Trainig Data
trainScore12 = svc12.score(Data12,Label12)
trainScore23 = svc23.score(Data23,Label23)
trainScore34 = svc34.score(Data34,Label34)
trainScore45 = svc45.score(Data45,Label45)
trainScore = clf_clf.score(data1,label1)

trainScore_method2 = clf_method2.score(data_all_method3,label1)
trainScore_method3 = clf_method3.score(data_all_method3,label1)

# Test models on Trainig Data
print ("Training Score_12: ",trainScore12)
print ("Training Score_23: ",trainScore23)
print ("Training Score_34: ",trainScore34)
print ("Training Score_45: ",trainScore45)
print ("Training Score_Train using Normal SVM: ",trainScore)
print ("Training Score_Train using ensembled SVM: ",trainScore_method2)
print ("Training Score_Train_method3 with onevsone classifier: ",trainScore_method3)

# oversample = SMOTE()
# testData,testLabel = oversample.fit_resample(testData, testLabel)

len(testData)

# ##calculating new features based on predict proba method

predict_proba_test_12 = svc12.predict_proba(testData)
predict_proba_test_23 = svc23.predict_proba(testData)
predict_proba_test_34 = svc34.predict_proba(testData)
predict_proba_test_45 = svc45.predict_proba(testData)

# test_1 = predict_proba_test_12[:,0]
# test_2 = predict_proba_test_12[:,1]
# test_3 = predict_proba_test_23[:,0]
# test_4 = predict_proba_test_23[:,1]
# test_5 = predict_proba_test_34[:,0]
# test_6 = predict_proba_test_34[:,1]
# test_7 = predict_proba_test_45[:,0]
# test_8 = predict_proba_test_45[:,1]

proba_test_weight_1 =predict_proba_test_12[:,0]
proba_test_weight_2 =predict_proba_test_23[:,0]
proba_test_weight_3 = predict_proba_test_34[:,0]
proba_test_weight_4 = (predict_proba_test_45[:,0])
# proba_test_weight_5 = weightage_4*predict_proba_test_45[:,1]

## method 3 appending 5 separate probabilities to all data
#data_all_test_method3 = np.concatenate((Data12_test,Data23_test,Data34_test,Data45_test),axis=0)
#y = np.vstack((proba_test_weight_1,proba_test_weight_2,proba_test_weight_3,proba_test_weight_4,proba_test_weight_5)).T

# #sorted in descending order
# sorted_at = np.sort(proba_test_weight_1)[::-1]
# sorted_bt = np.sort(proba_test_weight_2)[::-1]
# sorted_ct = np.sort(proba_test_weight_3)[::-1]
# sorted_dt = np.sort(proba_test_weight_4)[::-1]
# sorted_et = np.sort(proba_test_weight_5)[::-1]


# s_1t = (sorted_at - sorted_bt)*(1/1)
# s_2t = (sorted_bt - sorted_ct)*(1/2)
# s_3t = (sorted_ct - sorted_dt)*(1/3)
# s_4t = (sorted_dt - sorted_et)*(1/4)

# ## normalising data
# norm_1 = np.linalg.norm(proba_test_weight_1)
# proba_test_weight_1 = proba_test_weight_1/norm_1
# norm_2 = np.linalg.norm(proba_test_weight_2)
# proba_test_weight_2 = proba_test_weight_2/norm_2
# norm_3 = np.linalg.norm(proba_test_weight_3)
# proba_test_weight_3 = proba_test_weight_3/norm_3
# norm_4 = np.linalg.norm(proba_test_weight_4)
# proba_test_weight_4 = proba_test_weight_4/norm_4
# norm_5 = np.linalg.norm(proba_test_weight_5)
# proba_test_weight_5 = proba_test_weight_5/norm_5

##standardizing data
proba_test_weight_1 = (proba_test_weight_1 - proba_test_weight_1.mean())/(proba_test_weight_1.std())
proba_test_weight_2 = (proba_test_weight_2 - proba_test_weight_2.mean())/(proba_test_weight_2.std())
proba_test_weight_3 = (proba_test_weight_3 - proba_test_weight_3.mean())/(proba_test_weight_3.std())
proba_test_weight_4 = (proba_test_weight_4 - proba_test_weight_4.mean())/(proba_test_weight_4.std())
# proba_test_weight_5 = (proba_test_weight_5 - proba_test_weight_5.mean())/(proba_test_weight_5.std())
# y = np.vstack((proba_train_weight_1,proba_train_weight_2,proba_train_weight_3,proba_train_weight_4,proba_train_weight_5)).T
#y = np.vstack((proba_test_weight_1,proba_test_weight_2,proba_test_weight_3,proba_test_weight_4,proba_test_weight_5)).T

# y = np.sum((proba_test_weight_1,proba_test_weight_2,proba_test_weight_3,proba_test_weight_4,proba_test_weight_5),axis=0)
# y = y.reshape(880,1)

y = np.vstack((proba_test_weight_1,proba_test_weight_2,proba_test_weight_3,proba_test_weight_4)).T

# y = np.vstack((test_1,test_2,test_3,test_4,test_5,test_6,test_7,test_8)).T

data_all_test_method3 = np.append(testData,y,axis=1)
data_all_test_method3.shape

# For 5 separate models
# # Test model on testing Data
# testScore12 = svc12.score(testData,testLabel)
# testScore23 = svc23.score(testData,testLabel)
# testScore34 = svc34.score(testData,testLabel)
# testScore45 = svc45.score(testData,testLabel)

testScore_clf = clf_clf.score(testData,testLabel)

testScore_method2 = clf_method2.score(data_all_test_method3,testLabel)

testScore_method3 = clf_method3.score(data_all_test_method3,testLabel)

# # # Test models on Trainig Data
# print ("Testing Score_12: ",testScore12)
# print ("Testing Score_23: ",testScore23)
# print ("Testing Score_34: ",testScore34)
# print ("Testing Score_45: ",testScore45)

print ("full test data on single model Testing Score_Test_clf using Normal SVM: ",testScore_clf)

print ("full test data on single model Testing Score_Test method with ensembled SVM: ",testScore_method2)

print ("full test data on single model Testing Score_Test method 3: ",testScore_method3)

predictedLabel = clf_method2.predict(data_all_test_method3)

c = confusion_matrix(testLabel,predictedLabel)

print("confusion matrix for SVM: \n ",c)

testScore = clf_method2.score(data_all_test_method3,testLabel)
print(testScore)
y_pred = cross_val_predict(clf_method2, data_all_test_method3, testLabel, cv=kfold)
from sklearn.metrics import f1_score,recall_score,accuracy_score,plot_roc_curve
from sklearn.metrics import precision_score
print("accuracy:", accuracy_score(testLabel, predictedLabel))
print("precision:", precision_score(testLabel, predictedLabel, average='weighted'))
print("recall:", recall_score(testLabel, predictedLabel, average='weighted'))
print("f1_score:", f1_score(testLabel, predictedLabel, average='weighted'))
print(classification_report(testLabel, y_pred))

predictedLabel = clf_clf.predict(testData)

c = confusion_matrix(testLabel,predictedLabel)

print("confusion matrix for SVM: \n ",c)

testScore = clf_clf.score(testData,testLabel)
print(testScore)
from sklearn.metrics import f1_score,recall_score,accuracy_score,plot_roc_curve
from sklearn.metrics import precision_score
print("accuracy:", accuracy_score(testLabel, predictedLabel))
print("precision:", precision_score(testLabel, predictedLabel, average='weighted'))
print("recall:", recall_score(testLabel, predictedLabel, average='weighted'))
print("f1_score:", f1_score(testLabel, predictedLabel, average='weighted'))

y_pred = cross_val_predict(clf_method3, data_all_test_method3, testLabel, cv=kfold)
conf_mat = confusion_matrix(testLabel, y_pred)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print(confusion_matrix(testLabel, y_pred))
print(classification_report(testLabel, y_pred))












##Combined classifier
combinedClassifier = EnsembleVoteClassifier(clfs=[svc12, svc23,
                                    svc34,svc45],voting='soft')

combinedClassifier.fit(data1,label1)

score = combinedClassifier.score(testData,testLabel)

print(score)

predictedLabel = combinedClassifier.predict(testData)

c = confusion_matrix(testLabel,predictedLabel)

print("confusion matrix for SVM: \n ",c)

predictedLabel = combinedClassifier.predict(testData)

c = confusion_matrix(testLabel,predictedLabel)

print("confusion matrix for SVM: \n ",c)

testScore = combinedClassifier.score(testData,testLabel)
print(testScore)
from sklearn.metrics import f1_score,recall_score,accuracy_score,plot_roc_curve
from sklearn.metrics import precision_score
print("accuracy:", accuracy_score(testLabel, predictedLabel))
print("precision:", precision_score(testLabel, predictedLabel, average='weighted'))
print("recall:", recall_score(testLabel, predictedLabel, average='weighted'))
print("f1_score:", f1_score(testLabel, predictedLabel, average='weighted'))

from sklearn.ensemble import AdaBoostClassifier,VotingClassifier

adaBoostClassifier = AdaBoostClassifier(base_estimator = combinedClassifier)
adaBoostClassifier.fit(data1,label1)
score = adaBoostClassifier.score(testData,testLabel)

print(score)

predictedLabel = adaBoostClassifier.predict(testData)

c = confusion_matrix(testLabel,predictedLabel)

print("confusion matrix for SVM: \n ",c)



testData.shape

##weighted distance method
weighted_distance_12 = pd.DataFrame([0] * 6215)
weighted_distance_23 = pd.DataFrame([0] * 6215)
weighted_distance_34 = pd.DataFrame([0] * 6215)
weighted_distance_45 = pd.DataFrame([0] * 6215)

#distances from hyperplanes
y_12 = svc_12.decision_function(Data12)
w_norm_12 = np.linalg.norm(svc_12.coef_)
dist_12 = y_12 / w_norm_12
df_12 = pd.DataFrame(dist_12)

## class 23
y_23 = svc_23.decision_function(Data23)
w_norm_23 = np.linalg.norm(svc_23.coef_)
dist_23 = y_23 / w_norm_23
df_23 = pd.DataFrame(dist_23)

##class 34
y_34 = svc_34.decision_function(Data34)
w_norm_34 = np.linalg.norm(svc_34.coef_)
dist_34 = y_34 / w_norm_34
df_34 = pd.DataFrame(dist_34)

## class 45
y_45 = svc_45.decision_function(Data45)
w_norm_45 = np.linalg.norm(svc_45.coef_)
dist_45 = y_45 / w_norm_45
df_45 = pd.DataFrame(dist_45)

param = svc_12.get_params()

nv = svc_12.n_support_
a  = svc_12.dual_coef_
b  = svc_12._intercept_
cs = svc_12.classes_
print(nv,b,cs)

y_all = svc_15.decision_function(data1)
w_norm_all = np.linalg.norm(svc_45.coef_)
dist_all = y_all / w_norm_all
df_all = pd.DataFrame(dist_all)
df_all.describe()



## for individual svms

weighted_distance_12_1 = 1/(df_12)
weighted_distance_23_2 = 1/(df_23)
weighted_distance_34_3 = 1/(df_34)
weighted_distance_45_4 = 1/(df_45)

## for all training data and for single svm
j=0
for i in range(0,2541):
    weighted_distance_12[0][i] = weighted_distance_12_1[0][j]
    j+=1

j=0
for i in range(2541,3865):
    weighted_distance_23[0][i] = weighted_distance_23_2[0][j]
    j+=1

j=0
for i in range(3865,4905):
    weighted_distance_34[0][i] = weighted_distance_34_3[0][j]
    j+=1

j=0
for i in range(4905,6215):
    weighted_distance_45[0][i] = weighted_distance_45_4[0][j]
    j+=1







scaler = StandardScaler()
weighted_distance_12 = scaler.fit_transform(weighted_distance_12)
weighted_distance_23 = scaler.fit_transform(weighted_distance_23)
weighted_distance_34 = scaler.fit_transform(weighted_distance_34)
weighted_distance_45 = scaler.fit_transform(weighted_distance_45)

## individual data should also be standardised
weighted_distance_12_1 = scaler.fit_transform(weighted_distance_12_1)
weighted_distance_23_2 = scaler.fit_transform(weighted_distance_23_2)
weighted_distance_34_3 = scaler.fit_transform(weighted_distance_34_3)
weighted_distance_45_4 = scaler.fit_transform(weighted_distance_45_4)



## for individual model training

Data12_1 = np.append(Data12,weighted_distance_12_1,axis=1)
Data23_1 = np.append(Data23,weighted_distance_23_2,axis=1)
Data34_1 = np.append(Data34,weighted_distance_34_3,axis=1)
Data45_1 = np.append(Data45,weighted_distance_45_4,axis=1)

#all data used for training
data_all = np.concatenate((Data12_1,Data23_1,Data34_1,Data45_1),axis=0)
data_all_clf = np.concatenate((Data12,Data23,Data34,Data45),axis=0)
#data_all = np.concatenate((data_all,weighted_distance_12,weighted_distance_23,weighted_distance_34,weighted_distance_45),axis=1)
label_all = np.concatenate((Label12,Label23,Label34,Label45),axis=0)

data_all_clf.shape,label_all.shape

## method 2 appending 4 separate weights to all data
data_all_method2 = np.concatenate((Data12,Data23,Data34,Data45),axis=0)
y = np.concatenate((weighted_distance_12,weighted_distance_23,weighted_distance_34,weighted_distance_45),axis=1)
data_all_method2 = np.append(data_all_method2,y,axis=1)
data_all_method2.shape

## for whole data svm
##OneVsOne Classifier for all training set
ovo = OneVsOneClassifier(svc)
# Full training Data(which has all classes)
clf  = GridSearchCV(ovo, parameters_ovo)
# fit the training data to the model
clf.fit(data_all,label_all)
print(clf.best_params_)

## for whole data svm
##OneVsOne Classifier for all training set
ovo = OneVsOneClassifier(svc)
# Full training Data(which has all classes)
clf_clf  = GridSearchCV(ovo, parameters_ovo)
# fit the training data to the model
clf_clf.fit(data_all_clf,label_all)
print(clf.best_params_)

## for whole data svm method 2
##OneVsOne Classifier for all training set
ovo = OneVsOneClassifier(svc)
# Full training Data(which has all classes)
clf_method2  = GridSearchCV(ovo, parameters_ovo)
# fit the training data to the model
clf_method2.fit(data_all_method2,label_all)
print(clf_method2.best_params_)

## for whole data svm
##OneVsOne Classifier for all training set
ovo = OneVsOneClassifier(svc)
# Full training Data(which has all classes)
clf_ovo  = GridSearchCV(ovo, parameters_ovo)
# fit the training data to the model
clf_ovo.fit(data1,label1)
print(clf.best_params_)



##calculating new features based on predict proba method
proba_1 = pd.DataFrame([0] * 6215)
proba_2 = pd.DataFrame([0] * 6215)
proba_3 = pd.DataFrame([0] * 6215)
proba_4 = pd.DataFrame([0] * 6215)
proba_5 = pd.DataFrame([0] * 6215)

#training 4 svms for fitting the data
svc_12 = SVC(C=100,kernel='linear',probability=True)
svc_23 = SVC(C=100,kernel='linear',probability=True)
svc_34 = SVC(C=100,kernel='linear',probability=True)
svc_45 = SVC(C=1,kernel='linear',probability=True)
svc_12.fit(Data12,Label12)
svc_23.fit(Data23,Label23)
svc_34.fit(Data34,Label34)
svc_45.fit(Data45,Label45)

predict_proba_12 = svc_12.predict_proba(Data12)
predict_proba_23 = svc_23.predict_proba(Data23)
predict_proba_34 = svc_34.predict_proba(Data34)
predict_proba_45 = svc_45.predict_proba(Data45)
predict_proba_45.shape

proba_1[0:2541] = predict_proba_12[:,0]
proba_2[0:2541] = predict_proba_12[:,1]
proba_2[2541:3865] = predict_proba_23[:,0].reshape(1324,1)
proba_3[2541:3865] = predict_proba_23[:,1].reshape(1324,1)
proba_3[3865:4905] = predict_proba_34[:,0].reshape(1040,1)
proba_4[3865:4905] = predict_proba_34[:,1].reshape(1040,1)
proba_4[4905:6215] = predict_proba_45[:,0].reshape(1310,1)
proba_5[4905:6215] = predict_proba_45[:,1].reshape(1310,1)

## method 3 appending 5 separate probabilities to all data
data_all_method3 = np.concatenate((Data12,Data23,Data34,Data45),axis=0)
y = np.concatenate((proba_1,proba_2,proba_3,proba_4,proba_5),axis=1)
data_all_method3 = np.append(data_all_method3,y,axis=1)
data_all_method3.shape

## for whole data svm
##OneVsOne Classifier for all training set
ovo = OneVsOneClassifier(svc)
# Full training Data(which has all classes)
clf_method3  = GridSearchCV(ovo, parameters_ovo)
# fit the training data to the model
clf_method3.fit(data_all_method3,label_all)
print(clf_method3.best_params_)















def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

plt.figure(figsize=(12,2.7))

svc_rand = SVC(C=1000,kernel='linear',probability=True)
svc_rand.fit(Data12_1,Label12)

plot_svc_decision_boundary(svc_rand,-1,1)

#training 4 svms after appending the new feature to the data
svc_12 = SVC(C=100,kernel='linear',probability=True)
svc_23 = SVC(C=100,kernel='linear',probability=True)
svc_34 = SVC(C=100,kernel='linear',probability=True)
svc_45 = SVC(C=1,kernel='linear',probability=True)

svc_12.fit(Data12_1,Label12)
svc_23.fit(Data23_1,Label23)
svc_34.fit(Data34_1,Label34)
svc_45.fit(Data45_1,Label45)

# For 5 separate models
# Test model on Trainig Data
trainScore12 = svc_12.score(Data12_1,Label12)
trainScore23 = svc_23.score(Data23_1,Label23)
trainScore34 = svc_34.score(Data34_1,Label34)
trainScore45 = svc_45.score(Data45_1,Label45)
trainScore = clf.score(data_all,label_all)
trainScore_ovo = clf_ovo.score(data1,label1)
trainScore_clf = clf_clf.score(data_all_clf,label_all)
trainScore_method2 = clf_method2.score(data_all_method2,label_all)
trainScore_method3 = clf_method3.score(data_all_method3,label_all)

# Test models on Trainig Data
print ("Training Score_12: ",trainScore12)
print ("Training Score_23: ",trainScore23)
print ("Training Score_34: ",trainScore34)
print ("Training Score_45: ",trainScore45)
print ("Training Score_Train: ",trainScore)
print ("Training Score_Train_ovo: ",trainScore_ovo)
print ("Training Score_Train_clf_32: ",trainScore_clf)
print ("Training Score_Train_method2: ",trainScore_method2)
print ("Training Score_Train_method3: ",trainScore_method3)

data_all_method2.shape



# Creating Different test Dataset for bi-class problem

Data12_test = []
Data23_test = []
Data34_test = []
Data45_test = []
Data15_test = []


Label12_test = []
Label23_test = []
Label34_test = []
Label45_test = []
Label15_test = []


# Creating dataset for class1 and class2 only
for i in range(len(testData)):
    if (testLabel[i] == 1 or testLabel[i] == 2):
        Data12_test.append(testData[i])
        Label12_test.append(testLabel[i])

# Creating dataset for class2 and class3 only
for i in range(len(testData)):
    if (testLabel[i] == 2 or testLabel[i] == 3):
        Data23_test.append(testData[i])
        Label23_test.append(testLabel[i])

# Creating dataset for class3 and class4 only
for i in range(len(testData)):
    if (testLabel[i] == 3 or testLabel[i] == 4):
        Data34_test.append(testData[i])
        Label34_test.append(testLabel[i])

# Creating dataset for class4 and class5 only
for i in range(len(testData)):
    if (testLabel[i] == 4 or testLabel[i] == 5):
        Data45_test.append(testData[i])
        Label45_test.append(testLabel[i])

# Creating dataset for class4 and class5 only
for i in range(len(testData)):
    if (testLabel[i] == 1 or testLabel[i] == 5):
        Data15_test.append(testData[i])
        Label15_test.append(testLabel[i])

#svms for testing data to fit and get weighted distances
svc_12_test = SVC(C=100,kernel='linear',probability=True)
svc_23_test = SVC(C=100,kernel='linear',probability=True)
svc_34_test = SVC(C=100,kernel='linear',probability=True)
svc_45_test = SVC(C=1,kernel='linear',probability=True)
svc_15_test = SVC(C=10,kernel='linear',probability=True)
svc_12_test.fit(Data12_test,Label12_test)
svc_23_test.fit(Data23_test,Label23_test)
svc_34_test.fit(Data34_test,Label34_test)
svc_45_test.fit(Data45_test,Label45_test)
svc_15_test.fit(Data15,Label15)

#distances from hyperplanes
y_12_test = svc_12_test.decision_function(Data12_test)
w_norm_12_test = np.linalg.norm(svc_12_test.coef_)
dist_12_test = y_12_test / w_norm_12_test
df_12_test = pd.DataFrame(dist_12_test)

## class 23
y_23_test = svc_23_test.decision_function(Data23_test)
w_norm_23_test = np.linalg.norm(svc_23_test.coef_)
dist_23_test = y_23_test / w_norm_23_test
df_23_test = pd.DataFrame(dist_23_test)

##class 34
y_34_test = svc_34_test.decision_function(Data34_test)
w_norm_34_test = np.linalg.norm(svc_34_test.coef_)
dist_34_test = y_34_test / w_norm_34_test
df_34_test = pd.DataFrame(dist_34_test)

## class 45
y_45_test = svc_45_test.decision_function(Data45_test)
w_norm_45_test = np.linalg.norm(svc_45_test.coef_)
dist_45_test = y_45_test / w_norm_45_test
df_45_test = pd.DataFrame(dist_45_test)

## class 15
y_15_test = svc_15_test.decision_function(Data15_test)
w_norm_15_test = np.linalg.norm(svc_15_test.coef_)
dist_15_test = y_15_test / w_norm_15_test
df_15_test = pd.DataFrame(dist_15_test)



##calculating weighted distance of samples from the hyperplane
weighted_distance_12_test = 1/(df_12_test)
weighted_distance_23_test = 1/(df_23_test)
weighted_distance_34_test = 1/(df_34_test)
weighted_distance_45_test = 1/(df_45_test)
weighted_distance_15_test = 1/(df_15_test)

weighted_distance_12_test_method2 = pd.DataFrame([0] * 1216)
weighted_distance_23_test_method2 = pd.DataFrame([0] * 1216)
weighted_distance_34_test_method2 = pd.DataFrame([0] * 1216)
weighted_distance_45_test_method2 = pd.DataFrame([0] * 1216)
weighted_distance_15_test_method2 = pd.DataFrame([0] * 1216)

##calculating weighted distance of samples from the hyperplane method2
## for all testing data and for single svm
j=0
for i in range(0,501):
    weighted_distance_12_test_method2[0][i] = weighted_distance_12_test[0][j]
    j+=1

j=0
for i in range(501,733):
    weighted_distance_23_test_method2[0][i] = weighted_distance_23_test[0][j]
    j+=1

j=0
for i in range(733,937):
    weighted_distance_34_test_method2[0][i] = weighted_distance_34_test[0][j]
    j+=1

j=0
for i in range(937,1216):
    weighted_distance_45_test_method2[0][i] = weighted_distance_45_test[0][j]
    j+=1

j=0
for i in range(1216,1760):
    weighted_distance_15_test_method2[0][i] = weighted_distance_15_test[0][j]
    j+=1

weighted_distance_12_test_method2 = scaler.fit_transform(weighted_distance_12_test_method2)
weighted_distance_23_test_method2 = scaler.fit_transform(weighted_distance_23_test_method2 )
weighted_distance_34_test_method2 = scaler.fit_transform(weighted_distance_34_test_method2)
weighted_distance_45_test_method2 = scaler.fit_transform(weighted_distance_45_test_method2)
weighted_distance_15_test_method2 = scaler.fit_transform(weighted_distance_15_test_method2)


scaler = StandardScaler()
weighted_distance_12_test = scaler.fit_transform(weighted_distance_12_test)
weighted_distance_23_test = scaler.fit_transform(weighted_distance_23_test )
weighted_distance_34_test = scaler.fit_transform(weighted_distance_34_test)
weighted_distance_45_test = scaler.fit_transform(weighted_distance_45_test)

#weighted_distance_15_test.shape





#appending the newly added column to the
Data12_test_1 = np.append(Data12_test,weighted_distance_12_test,axis=1)
Data23_test_1 = np.append(Data23_test,weighted_distance_23_test,axis=1)
Data34_test_1 = np.append(Data34_test,weighted_distance_34_test,axis=1)
Data45_test_1 = np.append(Data45_test,weighted_distance_45_test,axis=1)
Data15_test_1 = np.append(Data15_test,weighted_distance_15_test,axis=1)


#all data used for training
data_all_test = np.concatenate((Data12_test_1,Data23_test_1,Data34_test_1,Data45_test_1,Data15_test_1),axis=0)

data_all_test_method4 = np.concatenate((Data12_test,Data23_test,Data34_test,Data45_test,Data15_test),axis=0)
#data_all = np.concatenate((data_all,weighted_distance_12,weighted_distance_23,weighted_distance_34,weighted_distance_45),axis=1)
label_all_test = np.concatenate((Label12_test,Label23_test,Label34_test,Label45_test),axis=0)
label_all_test_method4 = np.concatenate((Label12_test,Label23_test,Label34_test,Label45_test,Label15_test),axis=0)
data_all_test.shape

## method 2 appending 4 separate weights to all data
data_all_test_method2 = np.concatenate((Data12_test,Data23_test,Data34_test,Data45_test),axis=0)
data_all_test_method2_clf = np.concatenate((Data12_test,Data23_test,Data34_test,Data45_test),axis=0)
print(data_all_test_method2_clf.shape)
y_test = np.concatenate((weighted_distance_12_test_method2,weighted_distance_23_test_method2,
                    weighted_distance_34_test_method2,weighted_distance_45_test_method2),axis=1)
data_all_test_method2 = np.append(data_all_test_method2,y_test,axis=1)
data_all_test_method2.shape

##calculating new features based on predict proba method
proba_test_1 = pd.DataFrame([0] * 1216)
proba_test_2 = pd.DataFrame([0] * 1216)
proba_test_3 = pd.DataFrame([0] * 1216)
proba_test_4 = pd.DataFrame([0] * 1216)
proba_test_5 = pd.DataFrame([0] * 1216)


predict_proba_test_12 = svc_12_test.predict_proba(Data12_test)
predict_proba_test_23 = svc_23_test.predict_proba(Data23_test)
predict_proba_test_34 = svc_34_test.predict_proba(Data34_test)
predict_proba_test_45 = svc_45_test.predict_proba(Data45_test)


proba_test_1[0:501] = predict_proba_test_12[:,0]
proba_test_2[0:501] = predict_proba_test_12[:,1]
proba_test_2[501:733] = predict_proba_test_23[:,0].reshape(232,1)
proba_test_3[501:733] = predict_proba_test_23[:,1].reshape(232,1)
proba_test_3[733:937] = predict_proba_test_34[:,0].reshape(204,1)
proba_test_4[733:937] = predict_proba_test_34[:,1].reshape(204,1)
proba_test_4[937:1216] = predict_proba_test_45[:,0].reshape(279,1)
proba_test_5[937:1216] = predict_proba_test_45[:,1].reshape(279,1)


proba_test_2 = proba_test_2/2
proba_test_3 = proba_test_3/2
proba_test_4 = proba_test_4/2



## method 3 appending 5 separate probabilities to all data
data_all_test_method3 = np.concatenate((Data12_test,Data23_test,Data34_test,Data45_test),axis=0)
y = np.concatenate((proba_test_1,proba_test_2,proba_test_3,proba_test_4,proba_test_5),axis=1)
data_all_test_method3 = np.append(data_all_test_method3,y,axis=1)
data_all_test_method3.shape

##calculating new features based on predict proba method for weight based classification
proba_test_weight_1 = pd.DataFrame([0] * 1760)
proba_test_weight_2 = pd.DataFrame([0] * 1760)
proba_test_weight_3 = pd.DataFrame([0] * 1760)
proba_test_weight_4 = pd.DataFrame([0] * 1760)
proba_test_weight_5 = pd.DataFrame([0] * 1760)


predict_proba_test_weight_12 = svc_12_test.predict_proba(data_all_test_method4)
predict_proba_test_weight_23 = svc_23_test.predict_proba(data_all_test_method4)
predict_proba_test_weight_34 = svc_34_test.predict_proba(data_all_test_method4)
predict_proba_test_weight_45 = svc_45_test.predict_proba(data_all_test_method4)
predict_proba_test_weight_15 = svc_15_test.predict_proba(data_all_test_method4)

proba_test_weight_1 = predict_proba_test_weight_12[:,0]+predict_proba_test_weight_15[:,0]
proba_test_weight_2 = predict_proba_test_weight_12[:,1]+predict_proba_test_weight_23[:,0]
proba_test_weight_3 = predict_proba_test_weight_23[:,1]+predict_proba_test_weight_34[:,0]
proba_test_weight_4 = predict_proba_test_weight_34[:,1]+predict_proba_test_weight_45[:,0]
proba_test_weight_5 = predict_proba_test_weight_45[:,1]+predict_proba_test_weight_15[:,1]

final_test_predictions_array = np.vstack((proba_test_weight_1,proba_test_weight_2,proba_test_weight_3,proba_test_weight_4,proba_test_weight_5)).T

final_test_predictions_method4 = np.argmax((final_test_predictions_array),axis=1)

final_test_predictions_method4 = final_test_predictions_method4+1

c = confusion_matrix(label_all_test_method4,final_test_predictions_method4)

print("confusion matrix for SVM: \n ",c)

label_all_test.shape

data_all_test.shape



##fitting the new data to the 4 svms
svc_12_test.fit(Data12_test,Label12_test)
svc_23_test.fit(Data23_test,Label23_test)
svc_34_test.fit(Data34_test,Label34_test)
svc_45_test.fit(Data45_test,Label45_test)

## for whole data svm
##OneVsOne Classifier for all testing set
ovo = OneVsOneClassifier(svc)
# Full training Data(which has all classes)
clf_test  = GridSearchCV(ovo, parameters_ovo)
# fit the training data to the model
clf_test.fit(data_all_test,label_all_test_method4)
print(clf_test.best_params_)

## for whole data svm
##OneVsOne Classifier for all testing set
ovo = OneVsOneClassifier(svc)
# Full training Data(which has all classes)
clf_test_method2  = GridSearchCV(ovo, parameters_ovo)
# fit the training data to the model
clf_test_method2.fit(data_all_test_method2,label_all_test)
print(clf_test_method2.best_params_)

## for whole data svm
##OneVsOne Classifier for all testing set
ovo = OneVsOneClassifier(svc)
# Full training Data(which has all classes)
clf_test_method3  = GridSearchCV(ovo, parameters_ovo)
# fit the training data to the model
clf_test_method3.fit(data_all_test_method3,label_all_test)
print(clf_test_method3.best_params_)

# For 5 separate models
# Test model on testing Data
testScore12 = svc_12_test.score(Data12_test,Label12_test)
testScore23 = svc_23_test.score(Data23_test,Label23_test)
testScore34 = svc_34_test.score(Data34_test,Label34_test)
testScore45 = svc_45_test.score(Data45_test,Label45_test)
testScore = clf.score(data_all_test,label_all_test_method4)
testScore_ovo = clf_ovo.score(testData,testLabel)
testScore_clf = clf_clf.score(data_all_test_method2_clf,label_all_test)
testScore_method2 = clf_method2.score(data_all_test_method2,label_all_test)
testScore_method3 = clf_method3.score(data_all_test_method3,label_all_test)

# Test models on Trainig Data
print ("Testing Score_12: ",testScore12)
print ("Testing Score_23: ",testScore23)
print ("Testing Score_34: ",testScore34)
print ("Testing Score_45: ",testScore45)
print ("full test data on single model Testing Score_Test: ",testScore)
print ("full test data on single model Testing Score_Test_ovo: ",testScore_ovo)
print ("full test data on single model Testing Score_Test_clf: ",testScore_clf)
print ("full test data on single model Testing Score_Test method 2: ",testScore_method2)
print ("full test data on single model Testing Score_Test method 3: ",testScore_method3)

data_all_test_method3.shape

final_test_predictions_array_3 = np.concatenate((proba_test_1,proba_test_2,proba_test_3,proba_test_4,proba_test_5),axis=1)

final_test_predictions_method3 = np.argmax((final_test_predictions_array_3),axis=1)

final_test_predictions_method3 = final_test_predictions_method3+1

c = confusion_matrix(label_all_test,final_test_predictions_method3)

predictedLabel = svc_12_test.predict(Data12_test)

c = confusion_matrix(Label12_test,predictedLabel)

print("confusion matrix for SVM: \n ",c)

testScore = svc_12_test.score(Data12_test,Label12_test)
print(testScore)
from sklearn.metrics import f1_score,recall_score,accuracy_score,plot_roc_curve
from sklearn.metrics import precision_score
print("accuracy:", accuracy_score(Label12_test, predictedLabel))
print("precision:", precision_score(Label12_test, predictedLabel, average='weighted'))
print("recall:", recall_score(Label12_test, predictedLabel, average='weighted'))
print("f1_score:", f1_score(Label12_test, predictedLabel, average='weighted'))

predictedLabel = svc_23_test.predict(Data23_test)

c = confusion_matrix(Label23_test,predictedLabel)

print("confusion matrix for SVM: \n ",c)

from sklearn.metrics import f1_score,recall_score,accuracy_score,plot_roc_curve
from sklearn.metrics import precision_score
print("accuracy:", accuracy_score(Label23_test, predictedLabel))
print("precision:", precision_score(Label23_test, predictedLabel, average='weighted'))
print("recall:", recall_score(Label23_test, predictedLabel, average='weighted'))
print("f1_score:", f1_score(Label23_test, predictedLabel, average='weighted'))

predictedLabel = svc_34_test.predict(Data34_test)

c = confusion_matrix(Label34_test,predictedLabel)

print("confusion matrix for SVM: \n ",c)

from sklearn.metrics import f1_score,recall_score,accuracy_score,plot_roc_curve
from sklearn.metrics import precision_score
print("accuracy:", accuracy_score(Label34_test, predictedLabel))
print("precision:", precision_score(Label34_test, predictedLabel, average='weighted'))
print("recall:", recall_score(Label34_test, predictedLabel, average='weighted'))
print("f1_score:", f1_score(Label34_test, predictedLabel, average='weighted'))

predictedLabel = svc_45_test.predict(Data45_test)

c = confusion_matrix(Label45_test,predictedLabel)

print("confusion matrix for SVM: \n ",c)

from sklearn.metrics import f1_score,recall_score,accuracy_score,plot_roc_curve
from sklearn.metrics import precision_score
print("accuracy:", accuracy_score(Label45_test, predictedLabel))
print("precision:", precision_score(Label45_test, predictedLabel, average='weighted'))
print("recall:", recall_score(Label45_test, predictedLabel, average='weighted'))
print("f1_score:", f1_score(Label45_test, predictedLabel, average='weighted'))

from sklearn.metrics import f1_score,recall_score,accuracy_score,plot_roc_curve
from sklearn.metrics import precision_score
print("accuracy:", accuracy_score(label_all_test, final_test_predictions_method3))
print("precision:", precision_score(label_all_test, final_test_predictions_method3, average='weighted'))
print("recall:", recall_score(label_all_test, final_test_predictions_method3, average='weighted'))
print("f1_score:", f1_score(label_all_test, final_test_predictions_method3, average='weighted'))

predictedLabel = clf_ovo.predict(testData)

c = confusion_matrix(testLabel,predictedLabel)

print("confusion matrix for SVM: \n ",c)

testScore = clf_ovo.score(testData,testLabel)
print(testScore)
from sklearn.metrics import f1_score,recall_score,accuracy_score,plot_roc_curve
from sklearn.metrics import precision_score
print("accuracy:", accuracy_score(testLabel, predictedLabel))
print("precision:", precision_score(testLabel, predictedLabel, average='weighted'))
print("recall:", recall_score(testLabel, predictedLabel, average='weighted'))
print("f1_score:", f1_score(testLabel, predictedLabel, average='weighted'))

predictedLabel = clf_method3.predict(data_all_test_method3)

c = confusion_matrix(label_all_test,predictedLabel)

print("confusion matrix for SVM: \n ",c)

testScore = clf_method3.score(data_all_test_method3,label_all_test)
print(testScore)
from sklearn.metrics import f1_score,recall_score,accuracy_score,plot_roc_curve
from sklearn.metrics import precision_score
print("accuracy:", accuracy_score(label_all_test, predictedLabel))
print("precision:", precision_score(label_all_test, predictedLabel, average='weighted'))
print("recall:", recall_score(label_all_test, predictedLabel, average='weighted'))
print("f1_score:", f1_score(label_all_test, predictedLabel, average='weighted'))

plot_roc_curve(svc_23_test, Data23_test, Label23_test)

votingClassifier1 = SVC(C=100,kernel='linear',probability=True)
votingClassifier2 = SVC(C=100,kernel='linear',probability=True)
votingClassifier3 = SVC(C=100,kernel='linear',probability=True)
votingClassifier4 = SVC(C=10,kernel='linear',probability=True)
votingClassifier5 = SVC(C=10,kernel='linear',probability=True)

final_test_predictions_method3.shape



##Combined classifier
combinedClassifier = EnsembleVoteClassifier(clfs=[votingClassifier1, votingClassifier2,
                                    votingClassifier3,votingClassifier4,votingClassifier5],voting='soft')

combinedClassifier.fit(data1,label1)

score = combinedClassifier.score(testData,testLabel)

print(score)

predictedLabel = combinedClassifier.predict(testData)

c = confusion_matrix(testLabel,predictedLabel)

print("confusion matrix for SVM: \n ",c)

# Commented out IPython magic to ensure Python compatibility.
labels = ['svm1','svm2','svm3','svm4','svm5']
for clf, label in zip([votingClassifier1,votingClassifier2, votingClassifier3, votingClassifier4,votingClassifier5], labels):

    scores = model_selection.cross_val_score(clf, data1, label1,
                                              cv=5,
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
#           % (scores.mean(), scores.std(), label))

# votingClassifers.get_params().keys()
## parameters to be used for voting classifiers
params = {
      'estimator__svm__C': [1,10,100],'estimator__svm__kernel':["linear","rbf"]}







##OneVsOne Classifier for all training set
ovo = OneVsOneClassifier(svc)
# Full training Data(which has all classes)
clf  = GridSearchCV(ovo, parameters_ovo)
# fit the training data to the model
clf.fit(data1,label1)
print(clf.best_params_)

##OneVsRest Classifier for all training set
ovo = OneVsRestClassifier(svc)
# Full training Data(which has all classes)
clf  = GridSearchCV(ovo, parameters_ovo)
# fit the training data to the model
clf.fit(data1,label1)
print(clf.best_params_)

##Combined classifier
combinedClassifier = EnsembleVoteClassifier(clfs=[clf12, clf23,
                                    clf34,clf45,clf15],voting='soft')

combinedClassifier.fit(data1,label1)

score = combinedClassifier.score(testData,testLabel)

print(score)

predictedLabel = combinedClassifier.predict(testData)

c = confusion_matrix(testLabel,predictedLabel)

print("confusion matrix for SVM: \n ",c)

# For C = 100
# Test model on Trainig Data
trainScore12 = clf12.score(Data12,Label12)
trainScore23 = clf23.score(Data23,Label23)
trainScore34 = clf34.score(Data34,Label34)
trainScore45 = clf45.score(Data45,Label45)
trainScore15 = clf15.score(Data15,Label15)
trainScore = clf.score(data1,label1)

# Test models on Trainig Data
print ("Training Score_12: ",trainScore12)
print ("Training Score_23: ",trainScore23)
print ("Training Score_34: ",trainScore34)
print ("Training Score_45: ",trainScore45)
print ("Training Score_15: ",trainScore15)
print ("Training Score_Train: ",trainScore)



# Test model on Testing Data
testScore = clf.score(testData,testLabel)

print ("Testing Score: ", testScore)

predictedLabel = clf.predict(testData)

c = confusion_matrix(testLabel,predictedLabel)

print("confusion matrix for SVM: \n ",c)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

knn = KNeighborsClassifier()
gnb = GaussianNB()
rf = RandomForestClassifier()
lr = LogisticRegression()
svm = SVC(C=100,kernel='rbf',probability=True)

classifiers = [('knn', knn),
               ('gnb', gnb),
               ('rf', rf),
               ('svm', svm)]
vc = VotingClassifier(estimators=classifiers, voting='hard')

params = {'voting':['hard', 'soft'],
          'weights':[(1,1,1,1), (2,1,1,1), (1,2,1,1), (1,1,2,1), (1,1,1,2)]}

# find the best set of parameters
grid = GridSearchCV(estimator=vc, param_grid=params, cv=5, scoring='accuracy')
grid.fit(data1, label1)

print(grid.best_params_)
print('vc cross_val_score with GridSearch:' + str(cross_val_score(grid, data1, label1, scoring='accuracy', cv=10).mean()))

print('vc cross_val_score with GridSearch:' + str(cross_val_score(svm, data1, label1, scoring='accuracy', cv=10).mean()))
print('vc cross_val_score with GridSearch:' + str(cross_val_score(rf, data1, label1, scoring='accuracy', cv=10).mean()))
print('vc cross_val_score with GridSearch:' + str(cross_val_score(gnb, data1, label1, scoring='accuracy', cv=10).mean()))
print('vc cross_val_score with GridSearch:' + str(cross_val_score(knn, data1, label1, scoring='accuracy', cv=10).mean()))



##creating 4 different classifiers for voting
votingClassifier1 = SVC(C=100,kernel='rbf',probability=True)
votingClassifier2 = knn
votingClassifier3 = gnb
votingClassifier4 = rf


##Combined classifier
combinedClassifier = EnsembleVoteClassifier(clfs=[votingClassifier1, votingClassifier2,
                                    votingClassifier3,votingClassifier4],voting='hard',weights=[])

combinedClassifier.fit(data1,label1)

score = combinedClassifier.score(testData,testLabel)

print(score)

predictedLabel = combinedClassifier.predict(testData)

c = confusion_matrix(testLabel,predictedLabel)

print("confusion matrix for SVM: \n ",c)

from sklearn import svm
from sklearn import datasets
from numpy import argmax, zeros
from itertools import combinations

# do pairwise comparisons, return class with most +1 votes
def ovo_vote(classes, decision_function):
    combos = list(combinations(classes, 2))
    votes = zeros(len(classes))
    for i in range(len(decision_function[0])):
        if decision_function[0][i] > 0:
            votes[combos[i][0]] = votes[combos[i][0]] + 1
        else:
            votes[combos[i][1]] = votes[combos[i][1]] + 1
    winner = argmax(votes)
    return classes[winner]

# load the digits data set
digits = datasets.load_digits()

X, y = digits.data, digits.target

# set the SVC's decision function shape to "ovo"
estimator = svm.SVC(gamma=0.001, C=100., decision_function_shape='ovo')

# train SVC on all but the last digit
estimator.fit(X.data[:-1], y[:-1])

# print the value of the last digit
print("To be classified digit: ", y[-1:][0])

# print the predicted class
pred = estimator.predict(X[-1:])
print("Perform classification using predict: ", pred[0])

# get decision function
df = estimator.decision_function(X[-1:])

# print the decision function itself
print("Decision function consists of",len(df[0]),"elements:")
print(df)

# get classes, here, numbers 0 to 9
digits = estimator.classes_

# print which class has most votes
vote = ovo_vote(digits, df)
print("Perform classification using decision function: ", vote)

Training :

1. fit training data with 4 svms (12,23,34,45)
2. Get the distance of each point from each svm.
3. take the inverse of the distance and weigh them based on their distance. (1/distance)
4. get the weight feature for each svm. so, we now have 4 weight features
5. create a new training data by adding these 4 weight features to the original feature set.
6. train a new svm with the updated feature set.

n = 6
unique_value = 25
df = pd.DataFrame([0] * 6215)

df[0:2415] =1

df[0:1324] = weighted_distance_23





a =[0.22452880315825172,
  0.048579893342551146,
  0.04048607116243962,
  0.03724156869254235,
  0.035561160719685854,
  0.0345079590962536,
  0.033745513447485435,
  0.03321040686134292,
  0.032778946340255355,
  0.0326324795833093,
  0.03220555809294355,
  0.032026106438160624,
  0.03182049629803019,
  0.03165729025956886,
  0.031726337134324274,
  0.031496734807974304,
  0.03154588534614478,
  0.03139975507497445,
  0.03131634203567245,
  0.031187965630971152,
  0.031127746313296515,
  0.031158252544272905,
  0.031067658515497185,
  0.03112247082051532,
  0.030989020816639238,
  0.031053107028463107,
  0.030938184520379566,
  0.03102094028145075,
  0.030949111171494955,
  0.030972676779860736,
  0.03087784610046395,
  0.030942272052340125,
  0.030855305578516817,
  0.03081783686561146,
  0.03089760424804756,
  0.03075878873424626,
  0.030962879758799213,
  0.03073169847672013,
  0.030775556843941926,
  0.030665225649102665,
  0.030720330946061802,
  0.03070017345378111,
  0.030704624640444916,
  0.030642559109576816,
  0.030601925762562918,
  0.03064092469198265,
  0.030603523494611526,
  0.03059471559164853,
  0.030651794072112132,
  0.030546564437534618]
b =  [3.589091078988437,
  4.481736035182558,
  5.016183943584047,
  5.442383198902525,
  5.764519009096869,
  6.057287947884921,
  6.296977536431674,
  6.514841416786457,
  6.710365936673921,
  6.880084341969983,
  7.0606723736072405,
  7.212363152668394,
  7.35696869883044,
  7.49849035822112,
  7.637488233632054,
  7.764148531288936,
  7.863014517159297,
  7.987774240559545,
  8.096586605598187,
  8.210740977320178,
  8.296955881447628,
  8.418228297398008,
  8.51115267852257,
  8.597273185335357,
  8.700600361001902,
  8.7906368025418,
  8.861971953819538,
  8.952669324546024,
  9.034415623237347,
  9.135198938435522,
  9.207485083875985,
  9.288046771082385,
  9.381099092549292,
  9.4654855070443,
  9.527684951650686,
  9.595995623489905,
  9.664462007325271,
  9.730439580720047,
  9.820095473322375,
  9.895090547101251,
  9.971030925882273,
  9.990084105524524,
  10.079096695472455,
  10.115385483051169,
  10.166918968332224,
  10.248737269434436,
  10.293530793025576,
  10.360718102290713,
  10.42850819949446,
  10.478480191066348]

c =[3.589091078988437]
for i in range(len(b)-1):
    c.append(b[i+1]-b[i])

c

from matplotlib import pyplot

fig, axs = pyplot.subplots()

line1 = axs.plot(
    [i for i in range(len(a))], a, label="Train loss",
)
# axs[2].title.set_text("AKI Recall")
line2 = axs.plot(
    [i for i in range(len(c))], c, label="Validation loss"
)

# axs[0].set_axisbelow(True)
# axs[0].yaxis.grid(color="gray", linestyle="dashed")
# axs[0].xaxis.grid(color="gray", linestyle="dashed")

axs.legend()

pyplot.show()

