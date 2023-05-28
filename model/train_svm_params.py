
from itertools import product

from pandas import DataFrame, read_pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from tqdm import tqdm

from models import wide_params
from environment import NUM_USERS, training_columns_regex
from helper_functions import (get_cnt_filename, glimpse_df, serialize_functions,isnull_any, rows_with_null)
from preprocess import (df_replace_values)


#df_path = r"C:\Users\Ahmed Guebsi\Downloads\complete-raw-2022-02-26-is_complete_dataset_true___brains_true___reref_false.pickle"
df_path=r"C:\Users\Ahmed Guebsi\Downloads\complete-clean-2022-02-26-is_complete_dataset_true___brains_true___reref_false.pickle"
output_dir = r"C:\Users\Ahmed Guebsi\Desktop\Data_test"
#stdout_to_file(Path(args.output_report, "-".join(["svm-parameters", get_timestamp()]) + ".txt"))

df: DataFrame = read_pickle(df_path)
print(df.head())
print(isnull_any(df))
print(rows_with_null(df))
df = df_replace_values(df)
print(isnull_any(df))
print(rows_with_null(df))
training_columns = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)
print(training_columns)
#df_replace_values(df)
X = df.loc[:, ~df.columns.isin(["is_fatigued"])]
X = X[X.columns[X.max() != -1]]  # remove constant attributes
y = df.loc[:, "is_fatigued"]

groups = X["driver_id"].to_numpy()
acc_parameters = []

for C, gamma in tqdm(list(product(wide_params, wide_params))):
    model = SVC(kernel="rbf", C=C, gamma=gamma)
    acc_total = 0

    for train_index, test_index in LeaveOneGroupOut().split(X, y, groups):
        X_train, X_test = X.loc[train_index, training_columns], X.loc[test_index, training_columns]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_test_pred)
        acc_total += acc
    acc_parameters.append([acc_total / NUM_USERS, C, gamma])

print("Acc\t\t\tC\tgamma")
accs = sorted(acc_parameters, key=lambda x: x[0], reverse=True)
for acc in accs:
    print(acc)