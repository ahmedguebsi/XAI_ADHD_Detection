from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
#import pywt

import pandas as pd
import numpy as np
import pickle, sys

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, log_loss, roc_auc_score, roc_curve

from xgboost import XGBClassifier

from fancyimpute import MatrixFactorization, SimpleFill

from impute_transform import ImputeTransform

def get_metrics(X, y, clf_dict,
                    scoring, metric_df_cols,
                    X_test, y_test,
                    cv,
                    n_folds=10):
    """Runs cross validation to obtain error metrics for several classifiers.
    Outputs a formatted dataframe.

    INPUTS
    ------
    - X: dataframe representing feature matrix for training data
    - y: series representing target for training data
    - clf_dict: dict of list of objects and characteristics of prepared classifiers
    - scoring: dict of strings and objects for sklearn scoring
    - metric_df_cols: dict of strings for desired columns of output DataFrame
    - n_folds: int, number of folds for k-fold cross validation
    - multiclass: bool, whether target has multiple classes
    - cv: bool, whether to run cross validation

    OUTPUTS
    -------
    """
    if cv == True:
        clf_metrics = _run_clfs(clf_dict,
                                X, y, scoring,
                                n_folds)
    else:
        clf_metrics = _run_train_test(clf_dict,
                                    X, y,
                                    X_test, y_test,
                                    scoring)

    return clf_metrics

def _run_clfs(clf_dict,
                X, y, scoring,
                n_folds):
    """Runs cross validation on classifiers"""
    for name in clf_dict.keys():
        clf = clf_dict[name]['clf']
        scores = cross_validate(clf, X, y,
                                scoring=scoring, cv=n_folds,
                                return_train_score=True)
        clf_dict[name]['metrics'] = scores
    return clf_dict

def _run_train_test(clf_dict,
                    X_train, y_train,
                    X_test, y_test,
                    scoring):
    for name in clf_dict.keys():
        clf_metric_dict = {}
        clf = clf_dict[name]['clf']
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)
        clf_metric_dict['test_neg_log_loss'] = log_loss(y_test, y_pred_proba)
        clf_metric_dict['test_roc_auc'] = roc_auc_score(pd.get_dummies(y_test), y_pred_proba)
        clf_dict[name]['metrics'] = clf_metric_dict
    return clf_dict

def multiclass_roc_auc_score(truth, pred, average=None):
    """Returns multiclass roc auc score"""
    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)

    # with open('multiclass.txt', 'a') as f:
    #     data = list(roc_auc_score(truth, pred, average=None))
    #     data.append(truth.shape[0])
    #     f.write(str(data))
    #     f.write('\n')

    return roc_auc_score(truth, pred, average='macro')

def prep_x_y(df, target, feature):
    if target == 'DX':
        y = df[target].map({3:1, 1:0})
    else:
        y = df[target]

    if feature == 'tmcq':
        cols = ['Y1_P_TMCQ_ACTIVCONT', 'Y1_P_TMCQ_ACTIVITY', 'Y1_P_TMCQ_AFFIL',
          'Y1_P_TMCQ_ANGER', 'Y1_P_TMCQ_FEAR', 'Y1_P_TMCQ_HIP',
           'Y1_P_TMCQ_IMPULS', 'Y1_P_TMCQ_INHIBIT', 'Y1_P_TMCQ_SAD',
           'Y1_P_TMCQ_SHY', 'Y1_P_TMCQ_SOOTHE', 'Y1_P_TMCQ_ASSERT',
           'Y1_P_TMCQ_ATTFOCUS', 'Y1_P_TMCQ_LIP', 'Y1_P_TMCQ_PERCEPT',
           'Y1_P_TMCQ_DISCOMF', 'Y1_P_TMCQ_OPENNESS', 'Y1_P_TMCQ_SURGENCY',
           'Y1_P_TMCQ_EFFCONT', 'Y1_P_TMCQ_NEGAFFECT']
    elif feature == 'neuro':
        cols = ['STOP_SSRTAVE_Y1', 'DPRIME1_Y1', 'DPRIME2_Y1', 'SSBK_NUMCOMPLETE_Y1',
            'SSFD_NUMCOMPLETE_Y1', 'V_Y1', 'Y1_CLWRD_COND1', 'Y1_CLWRD_COND2',
            'Y1_DIGITS_BKWD_RS', 'Y1_DIGITS_FRWD_RS', 'Y1_TRAILS_COND2',
            'Y1_TRAILS_COND3', 'CW_RES', 'TR_RES', 'Y1_TAP_SD_TOT_CLOCK']
    elif feature == 'all':
        cols = ['Y1_P_TMCQ_ACTIVCONT', 'Y1_P_TMCQ_ACTIVITY', 'Y1_P_TMCQ_AFFIL',
          'Y1_P_TMCQ_ANGER', 'Y1_P_TMCQ_FEAR', 'Y1_P_TMCQ_HIP',
           'Y1_P_TMCQ_IMPULS', 'Y1_P_TMCQ_INHIBIT', 'Y1_P_TMCQ_SAD',
           'Y1_P_TMCQ_SHY', 'Y1_P_TMCQ_SOOTHE', 'Y1_P_TMCQ_ASSERT',
           'Y1_P_TMCQ_ATTFOCUS', 'Y1_P_TMCQ_LIP', 'Y1_P_TMCQ_PERCEPT',
           'Y1_P_TMCQ_DISCOMF', 'Y1_P_TMCQ_OPENNESS', 'Y1_P_TMCQ_SURGENCY',
           'Y1_P_TMCQ_EFFCONT', 'Y1_P_TMCQ_NEGAFFECT',
           'STOP_SSRTAVE_Y1', 'DPRIME1_Y1', 'DPRIME2_Y1', 'SSBK_NUMCOMPLETE_Y1',
           'SSFD_NUMCOMPLETE_Y1', 'V_Y1', 'Y1_CLWRD_COND1', 'Y1_CLWRD_COND2',
           'Y1_DIGITS_BKWD_RS', 'Y1_DIGITS_FRWD_RS', 'Y1_TRAILS_COND2',
           'Y1_TRAILS_COND3', 'CW_RES', 'TR_RES', 'Y1_TAP_SD_TOT_CLOCK']
    X = df[cols]

    if feature == 'tmcq':
        X_no_null = X[X.isnull().sum(axis=1) == 0]
        y_no_null = y[X.isnull().sum(axis=1) == 0]
    else:
        X_no_null = X[X.isnull().sum(axis=1) != X.shape[1]]
        y_no_null = y[X.isnull().sum(axis=1) != X.shape[1]]

    return X_no_null, y_no_null

def prep_clfs(feature):
    if feature == 'tmcq':
        log_reg_clf = make_pipeline(LogisticRegression(random_state=56))

        rf_clf = make_pipeline(RandomForestClassifier(n_jobs=-1, random_state=56))

        gb_clf = make_pipeline(GradientBoostingClassifier(random_state=56))

        xgb_clf = make_pipeline(XGBClassifier(max_depth=3, learning_rate=0.1,
                                random_state=56, n_jobs=-1))
    else:
        log_reg_clf = make_pipeline(ImputeTransform(strategy=MatrixFactorization()),
                            LogisticRegression(random_state=56))

        rf_clf = make_pipeline(ImputeTransform(strategy=MatrixFactorization()),
                               RandomForestClassifier(n_jobs=-1, random_state=56))

        gb_clf = make_pipeline(ImputeTransform(strategy=MatrixFactorization()),
                               GradientBoostingClassifier(random_state=56))

        xgb_clf = make_pipeline(ImputeTransform(strategy=MatrixFactorization()),
                                XGBClassifier(max_depth=3, learning_rate=0.1,
                                random_state=56, n_jobs=-1))
    classifier_dict = {'LogReg':
                            {'clf': log_reg_clf},
                       'RandomForest':
                            {'clf': rf_clf},
                       'GradientBoosting':
                            {'clf': gb_clf},
                       'XGB':
                            {'clf': xgb_clf}}
    return classifier_dict

def prep_scoring(target):
    scoring_dict = {'accuracy': 'accuracy',
                    'neg_log_loss': 'neg_log_loss'}
    if target == 'DXSUB':
        multiclass_roc = make_scorer(multiclass_roc_auc_score,
                                 greater_is_better=True)
        scoring_dict['roc_auc'] = multiclass_roc
    else:
        scoring_dict['roc_auc'] = 'roc_auc'
    return scoring_dict

def notch_filter(x, samplerate, plot=False):
    x = x - np.mean(x)

    high_cutoff_notch = 59 / (samplerate / 2)
    low_cutoff_notch = 61 / (samplerate / 2)

    # Band Stop Filter (BSF) or Band Reject Filter
    [b, a] = signal.butter(4, [high_cutoff_notch, low_cutoff_notch], btype='stop')

    x_filt = signal.filtfilt(b, a, x.T)

    if plot:
        t = np.arange(0, len(x) / samplerate, 1 / samplerate)
        plt.plot(t, x)
        plt.plot(t, x_filt.T, 'k')
        plt.autoscale(tight=True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        plt.show()

    return x_filt


def bp_filter(x, low_f, high_f, samplerate, plot=False):
    # x = x - np.mean(x)

    low_cutoff_bp = low_f / (samplerate / 2)
    high_cutoff_bp = high_f / (samplerate / 2)

    [b, a] = signal.butter(5, [low_cutoff_bp, high_cutoff_bp], btype='bandpass')

    x_filt = signal.filtfilt(b, a, x)

    if plot:
        t = np.arange(0, len(x) / samplerate, 1 / samplerate)
        plt.plot(t, x)
        plt.plot(t, x_filt, 'k')
        plt.autoscale(tight=True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        plt.show()

    return x_filt


def plot_signal(x, samplerate, chname):
    t = np.arange(0, len(x) / samplerate, 1 / samplerate)
    plt.plot(t, x)
    plt.autoscale(tight=True)
    plt.xlabel('Time')
    plt.ylabel('Amplitude (mV)')
    plt.title(chname)
    plt.show()

def plot_features(signal, channel_name, fs, feature_matrix, step):
    """
    Argument:
    signal -- python numpy array representing recording of a signal.
    channel_name -- string variable with the EMG channel name in analysis (Title).
    fs -- int variable with the sampling frequency used to acquire the signal.
    feature_matrix -- python Dataframe.
    step -- int variable with the step size used in the sliding window method.
    """

    ts = np.arange(0, len(signal) / fs, 1 / fs)
    # for idx, f in enumerate(tfeatures.T):
    for key in feature_matrix.T:
        tf = step * (np.arange(0, len(feature_matrix.T[key]) / fs, 1 / fs))
        fig = plt.figure()

        ax = fig.add_subplot(111, label="1")
        ax2 = fig.add_subplot(111, label="2", frame_on=False)
        ax.plot(ts, signal, color="C0")
        ax.autoscale(tight=True)
        plt.title(channel_name + ": " + key)
        ax.set_xlabel("Time")
        ax.set_ylabel("mV")

        ax2.plot(tf, feature_matrix.T[key], color="red")
        ax2.yaxis.tick_right()
        ax2.autoscale(tight=True)
        ax2.set_xticks([])
        ax2.set_yticks([])
        # mng = plt.get_current_fig_manager()
        # mng.window.state('zoomed')
        plt.show()



def plot_roc():
    plt.plot(fpr, tpr, label = 'ROC curve', linewidth = 2)
    plt.plot([0,1],[0,1], 'k--', linewidth = 2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for LoR')
    plt.show()
fpr, tpr, t = roc_curve(y1_test, prediction)
plot_roc()

cm = confusion_matrix(y1_test, prediction)
cm
#Plotting the confusion matrix
plt.figure(figsize=(10,7))
p = sns.heatmap(cm, annot=True, cmap="Reds", fmt='g')
plt.title('Confusion matrix  - Cancer')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()

# Correlation matrix

plt.figure(figsize=(9,9))
sns.heatmap(df1.iloc[:,1:12].corr(),yticklabels=True,annot=True)