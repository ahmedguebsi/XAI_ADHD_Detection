
import scipy.io
from glob import glob
import numpy as np
from tqdm import tqdm
import pandas as pd


from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import KFold,LeaveOneOut,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import xgboost as xgb
import matplotlib.pyplot as plt

IDD_rest=[]
for i in glob(IDD+'/Music/*.mat'):
    data=scipy.io.loadmat(i)['clean_data'].reshape(14,-1,128*4)
    data=np.swapaxes(data,0,1)
    IDD_rest.append(data )

TDC_rest=[]
for i in glob(TDC+'/Music/*.mat'):
    data=scipy.io.loadmat(i)['clean_data'].reshape(14,-1,128*4)
    data=np.swapaxes(data,0,1)
    TDC_rest.append(data )


fig,ax=plt.subplots(nrows=8,ncols=1,figsize=(8,10))
label=['A6','D6','D5','D4','D3','D2','D1']
for i,l in zip(range(1,8),label):
  ax[0].plot(x,color='b')
  #ax[0].set_xticks([],[])
  ax[0].set_title('Signal',x=-0.1,y=0.1)
  ax[i].plot(coefs[i-1],color='b')
  ax[i].set_title(l,x=-0.1,y=0.1)
  #ax[i].set_xticks(np.arange(0,250,50))
#plt.subplots_adjust(wspace=0, hspace=0)

classifiers = [
    KNeighborsClassifier(),
    SVC(),
    NuSVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    #xgb.XGBClassifier(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]


def classifiers_calc(X, y):
    accuracy_avg = []
    accuracy_std = []
    f1_avg = []
    f1_std = []
    for clfs in classifiers:
        # print('====================================')
        # name = clfs.__class__.__name__
        # print(name)
        acc_scores = []
        f1_scores = []
        for train, test in StratifiedKFold(10).split(X, y):
            X_train = X[train].reshape(-1, X.shape[2])
            X_test = X[test].reshape(-1, X.shape[2])
            y_train = np.concatenate([[i] * X.shape[1] for i in y[train]])
            y_test = np.concatenate([[i] * X.shape[1] for i in y[test]])
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            X_train, y_train = shuffle(X_train, y_train)

            clfs.fit(X_train, y_train)
            y_pred = clfs.predict(X_test)
            acc_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))
        accuracy_avg.append(np.mean(acc_scores))
        accuracy_std.append(np.std(acc_scores))
        f1_avg.append(np.mean(f1_scores))
        f1_std.append(np.std(f1_scores))


classifier = ['KNN', 'c-SVM', 'nu-SVM', 'DT', 'RF', 'AB', 'GB', 'NB', 'LDA', 'QDA', 'LR']


def plot_classifiers(accuracy_avg, f1_avg, title):
    y_pos = np.arange(len(classifier))
    w = 0.4
    plt.figure(figsize=(10, 5))
    plt.bar(y_pos, np.array(accuracy_avg), align='center', width=w, label='Accuracy')
    plt.bar(y_pos + w, np.array(f1_avg), align='center', width=w, label='F1-score')
    ya = np.array(accuracy_avg)
    for index, value in enumerate(ya):
        plt.text(index - 0.1, value - 0.2, str(np.round(value, 2)), rotation=90, color='white', fontsize=11)

    yf = np.array(f1_avg)
    for index, value in enumerate(yf):
        plt.text(index + w - 0.1, value - 0.1, str(np.round(value, 2)), rotation=90, color='white', fontsize=11)

    plt.xticks(y_pos, classifier, fontsize=11)
    plt.yticks(fontsize=11)

    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Classifiers', fontsize=12)
    plt.title('Classifiers Performance of {} Coefficents Features'.format(title), fontsize=12)
    plt.legend()
    plt.savefig('{} Feature.eps'.format(title), dip=300)

meanFeatureAcc,meanFeatureAccStd,meanFeatureF1,meanFeatureF1Std=classifiers_calc(X,y) #average k fold cross validation of mean features extraction


df=pd.DataFrame(zip(meanFeatureAcc,meanFeatureAccStd,meanFeatureF1,meanFeatureF1Std)).round(4)*100
df.columns=['Acc Avg', 'Acc Std.','F1 Avg', 'F1 Std.']
mean_df=df.rename(index=dict(zip(list(range(0,11)),classifier)))
mean_df



plot_classifiers(meanFeatureAcc,meanFeatureF1,title='Average')