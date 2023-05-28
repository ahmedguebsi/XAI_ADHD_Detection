
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

from pandas import read_pickle
from pandas._config.config import set_option
from pandas.core.frame import DataFrame
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
from tqdm import tqdm

from models import model_knn, model_mlp, model_rfc, model_svc
from preprocess import split_and_normalize
from environment import training_columns_regex
#from utils_file_saver import TIMESTAMP_FORMAT, save_model
from helper_functions import (get_cnt_filename, glimpse_df, serialize_functions,isnull_any, rows_with_null)

timestamp = datetime.today()
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
set_option("display.max_columns", None)

TIMESTAMP_FORMAT = "%Y-%m-%d-%H-%M-%S"
output_dir = r"C:\Users\Ahmed Guebsi\Desktop\Data_test"
#stdout_to_file(Path(output_dir, "-".join(["train-models", timestamp.strftime(TIMESTAMP_FORMAT)]) + ".txt"))



def loo_generator(X, y):
    groups = X["driver_id"].to_numpy()
    scaler = MinMaxScaler()

    for train_index, test_index in LeaveOneGroupOut().split(X, y, groups):
        X_train, X_test = X.loc[train_index, training_columns], X.loc[test_index, training_columns]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train.loc[:, training_columns] = scaler.fit_transform(X_train.loc[:, training_columns])
        X_test.loc[:, training_columns] = scaler.transform(X_test.loc[:, training_columns])
        yield X_train, X_test, y_train, y_test


def split_generator(X, y):
    X_train, X_test, y_train, y_test = split_and_normalize(X.loc[:, training_columns], y, test_size=0.5, columns_to_scale=training_columns)
    yield X_train, X_test, y_train, y_test

#df_path=r"C:\Users\Ahmed Guebsi\Downloads\complete-clean-2022-02-26-is_complete_dataset_true___brains_true___reref_false.pickle"
df_path=r"C:\Users\Ahmed Guebsi\Desktop\Data_test\.clean_raw_df.pkl"
df: DataFrame = read_pickle(df_path)
print(isnull_any(df))
#glimpse_df(df)
print(rows_with_null(df))

training_columns = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)
X = df.drop("is_fatigued", axis=1)
y = df.loc[:, "is_fatigued"]

strategies = {"leaveoneout": loo_generator, "split": split_generator}
scorings = ["f1"]
#models = [model_svc, model_rfc, model_mlp, model_knn]
models = [model_mlp]

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

        y_trues.append(y_true_test)
        y_preds.append(y_pred_test)

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

    print_table = {"Model": [model_name], "f1": [f1_average], "accuracy": [acc_average]}
    print_table.update({k: [v] for k, v in model.best_params_.items()})
    print(tabulate(print_table, headers="keys"), "\n")

    for params in params_dict.keys():
        params_dict[params]["mean"] = sum(params_dict[params]["means"]) / len(params_dict[params]["means"])
        params_dict[params]["std"] = sum(params_dict[params]["stds"]) / len(params_dict[params]["stds"])

    for params, mean, std in map(lambda x: (x[0], x[1]["mean"], x[1]["std"]), sorted(params_dict.items(), key=lambda x: x[1]["mean"], reverse=True)):
        print("%0.6f (+/-%0.6f) for %r" % (mean, std * 2, dict(params)))

stdout_to_file(Path(output_dir, "-".join(["train-models", timestamp.strftime(TIMESTAMP_FORMAT)]) + ".txt"))
#glimpse_df(df)