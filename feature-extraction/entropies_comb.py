"""
Create all possible combinations of entropies
Train GridSearch SVM model on each entropy combination
Find out which entropy combinations performs the best
"""

from itertools import product

from pandas import DataFrame, read_pickle
from sklearn.metrics import classification_report
from tqdm import tqdm

from models import model_svc
from preprocess import split_and_normalize
from environment import entropy_names, training_columns_regex
from helper_functions import (get_dictionary_leaves, powerset)

df_path=r"C:\Users\Ahmed Guebsi\Downloads\complete-clean-2022-02-26-is_complete_dataset_true___brains_true___reref_false.pickle"
output_dir = r"C:\Users\Ahmed Guebsi\Desktop\Data_test"
df: DataFrame = read_pickle(df_path)
X = df.loc[:, ~df.columns.isin(["is_fatigued"])]
y = df.loc[:, "is_fatigued"]

training_columns = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)
X_train_org, X_test_org, y_train, y_test = split_and_normalize(X, y, 0.5, training_columns)

entropy_excluded_powerset = list(powerset(entropy_names))[:-1]  # exclude last element (PE, AE, FE, SE)
models = [model_svc]
scorings = ["accuracy"]
results = []

for i, pair in enumerate(tqdm(list(product(scorings, models, entropy_excluded_powerset)))):
    scoring, model, entropies_exclude = pair
    (X_train, X_test) = (X_train_org.copy(), X_test_org.copy())

    for entropy in entropies_exclude:
        X_train = X_train.loc[:, ~X_train.columns.str.contains(entropy)]
        X_test = X_test.loc[:, ~X_test.columns.str.contains(entropy)]

    model.scoring = scoring
    model.fit(X_train, y_train)

    y_true_train, y_pred_train = y_train, model.predict(X_train)
    y_true_test, y_pred_test = y_test, model.predict(X_test)

    classification_report_string = classification_report(y_true_test, y_pred_test, digits=6, output_dict=True)

    results.append([i, list(set(entropy_names) - set(entropies_exclude)), model.best_score_, get_dictionary_leaves(classification_report_string)])

for result in sorted(results, key=lambda x: x[2], reverse=True):
    print(result[0], result[1], result[2])