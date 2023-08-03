import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from pandas import read_pickle, Series
from pandas._config.config import set_option
from pandas.core.frame import DataFrame
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, precision_score, auc, roc_curve, recall_score,confusion_matrix
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, KFold, cross_validate, train_test_split
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
from tqdm import tqdm

from models import model_knn, model_mlp, model_rfc, model_svc
from environment import training_columns_regex
#from utils_file_saver import TIMESTAMP_FORMAT, save_model
from helper_functions import (get_mat_filename, glimpse_df, serialize_functions,isnull_any, rows_with_null)

timestamp = datetime.today()
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
set_option("display.max_columns", None)

TIMESTAMP_FORMAT = "%Y-%m-%d-%H-%M-%S"
output_dir = r"C:\Users\Ahmed Guebsi\Desktop\Data_test"
#stdout_to_file(Path(output_dir, "-".join(["train-models", timestamp.strftime(TIMESTAMP_FORMAT)]) + ".txt"))



def loo_generator(X, y):
    groups = X["child_id"].to_numpy()
    scaler = MinMaxScaler()

    for train_index, test_index in LeaveOneGroupOut().split(X, y, groups):
        X_train, X_test = X.loc[train_index, training_columns], X.loc[test_index, training_columns]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train.loc[:, training_columns] = scaler.fit_transform(X_train.loc[:, training_columns])
        X_test.loc[:, training_columns] = scaler.transform(X_test.loc[:, training_columns])
        yield X_train, X_test, y_train, y_test

def split_and_normalize(X: Series, y: Series, test_size: float, columns_to_scale, scaler: MinMaxScaler = MinMaxScaler()):
    """Columns to scale can be both string list or list of bools"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    X_train: Series
    X_test: Series
    y_train: Series
    y_test: Series

    X_train.loc[:, columns_to_scale] = scaler.fit_transform(X_train.loc[:, columns_to_scale])
    X_test.loc[:, columns_to_scale] = scaler.transform(X_test.loc[:, columns_to_scale])
    return X_train, X_test, y_train, y_test

def split_generator(X, y):
    X_train, X_test, y_train, y_test = split_and_normalize(X.loc[:, training_columns], y, test_size=0.5, columns_to_scale=training_columns)
    yield X_train, X_test, y_train, y_test

def cross_validation_generator(X, y):
    # cross validation
    Kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    Kfold.get_n_splits(X, y)
    foldNum = 0  # initializing fold = 0
    for train_index, test_index in Kfold.split(X, y):
        foldNum += 1
        print("fold", foldNum)
        X_train, X_test = X.loc[train_index, training_columns], X.loc[test_index, training_columns]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        yield X_train, X_test, y_train, y_test



def specificity(y_true, y_pred):
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp)
def sensitivity(y_true, y_pred):
    tp = sum((y_true == 1) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn)
def false_prediction_rate(y_true, y_pred):
    fp = sum((y_true == 0) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    return fp / (fp + tn)
def matthews_correlation_coefficient(y_true, y_pred):
    tp = sum((y_true == 1) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return numerator / denominator
def negative_predicted_value(y_true, y_pred):
    tn = sum((y_true == 0) & (y_pred == 0))
    fn = sum((y_true == 1) & (y_pred == 0))
    return tn / (tn + fn)
def critical_success_index(y_true, y_pred):
    tp = sum((y_true == 1) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fn + fp)

def balanced_accuracy(y_true, y_pred):
    return (sensitivity(y_true, y_pred) + specificity(y_true, y_pred)) / 2
def f1_score_weighted(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted")
def f1_score_none(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None)

def precision_score_weighted(y_true, y_pred):
    return precision_score(y_true, y_pred, average="weighted")
def precision_score_none(y_true, y_pred):
    return precision_score(y_true, y_pred, average=None)
def recall_score_weighted(y_true, y_pred):
    return recall_score(y_true, y_pred, average="weighted")
def recall_score_none(y_true, y_pred):
    return recall_score(y_true, y_pred, average=None)
def roc_auc_score_weighted(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average="weighted")
def roc_auc_score_macro(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average="macro")
def roc_auc_score_micro(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average="micro")
def roc_auc_score_none(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average=None)


#df_path=r"C:\Users\Ahmed Guebsi\Downloads\complete-clean-2022-02-26-is_complete_dataset_true___brains_true___reref_false.pickle"
df_path=r"C:\Users\Ahmed Guebsi\Desktop\Data_test\.clean_raw_df_adhd.pkl"
df: DataFrame = read_pickle(df_path)
print(isnull_any(df))
#glimpse_df(df)
print(rows_with_null(df))
X = df.drop("is_adhd", axis=1)
y = df.loc[:, "is_adhd"]

training_columns = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)
# Remove rows with null values


#strategies = {"leaveoneout": loo_generator, "split": split_generator}
strategies = {"cv": cross_validation_generator, "split": split_generator}
scorings = ["f1"]
models = [model_svc, model_rfc, model_mlp, model_knn]
#models = [model_mlp]

training_generators = map(lambda strategy_name: (strategy_name, strategies[strategy_name]),strategies)

for (training_generator_name, training_generator), model, scoring in tqdm(list(product(training_generators, models, scorings)), desc="Training model"):
    scoring: str
    model: GridSearchCV
    model_name = type(model.estimator).__name__
    model.scoring = scoring

    y_trues = []
    y_preds = []
    means = []
    stds = []
    params_dict = {}
    for X_train, X_test, y_train, y_test in tqdm(list(training_generator(X, y)), desc="Model {}".format(model_name)):
        model.fit(X_train.values, y_train.values)
        y_true_test, y_pred_test = y_test, model.predict(X_test.values)
        print(y_true_test)
        print(type(y_true_test))
        y_trues.append(y_true_test)
        y_preds.append(y_pred_test)

        #accuracy = accuracy_score(y_trues, y_preds)
        #f1_score = f1_score(y_true_test, y_pred_test)


        for mean, std, params in zip(model.cv_results_["mean_test_score"], model.cv_results_["std_test_score"], model.cv_results_["params"]):
            params = frozenset(params.items())
            if params not in params_dict:
                params_dict[params] = {}
                params_dict[params]["means"] = []
                params_dict[params]["stds"] = []
            params_dict[params]["means"].append(mean)
            params_dict[params]["stds"].append(std)
    f1_average = sum((map(lambda x: f1_score(x[0], x[1]), zip(y_trues, y_preds)))) / len(y_trues)
    acc_average = sum((map(lambda x: accuracy_score(x[0], x[1]), zip(y_trues, y_preds)))) / len(y_trues)
    print_table = {"Model": [model_name], "f1_average": [f1_average], "accuracy_average": [acc_average]}

    print_table.update({k: [v] for k, v in model.best_params_.items()})
    print(tabulate(print_table, headers="keys"), "\n")

    for params in params_dict.keys():
        params_dict[params]["mean"] = sum(params_dict[params]["means"]) / len(params_dict[params]["means"])
        params_dict[params]["std"] = sum(params_dict[params]["stds"]) / len(params_dict[params]["stds"])

    for params, mean, std in map(lambda x: (x[0], x[1]["mean"], x[1]["std"]), sorted(params_dict.items(), key=lambda x: x[1]["mean"], reverse=True)):
        print("%0.6f (+/-%0.6f) for %r" % (mean, std * 2, dict(params)))

stdout_to_file(Path(output_dir, "-".join(["train-models", timestamp.strftime(TIMESTAMP_FORMAT)]) + ".txt"))
#glimpse_df(df)