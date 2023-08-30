import numpy as np
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import warnings
from tqdm import tqdm

from environment import channels_good
from pandas import DataFrame, read_pickle
from sklearn.model_selection import (train_test_split)
from sklearn.feature_selection import SelectKBest, f_classif

from helper_functions import ( glimpse_df, serialize_functions,isnull_any)

class RandomForest:
    def __init__(self):
        #self.path = path
        self.SEED = 42
        #assert variable in (0, 1)
        #self.variable = variable
        self.normalizer = StandardScaler()

        # CV ranges
        self.folds = 5
        self.n_trees = [3, 10, 50, 100, 300, 1000]
        self.max_features = [ 'sqrt', 'log2']
        self.max_depths = [10, 30, 50, 100]
        self.criterions = ['gini', 'entropy']
        self.min_samples_splits = [2, 5, 10]

    def fit(self):
        # load data
        #df_path = r"C:\Users\Ahmed Guebsi\Desktop\Data_test\final_modified_df.pkl"
        df_path = r"C:\Users\Ahmed Guebsi\Desktop\Data_test\hilbert_dataframe.pkl"
        df: DataFrame = read_pickle(df_path)
        glimpse_df(df)

        # Check for infinity in the DataFrame
        has_infinity = df.isin([np.inf, -np.inf]).any()

        # Check for values too large for float64 dtype
        dtype_max_value = np.finfo(np.float64).max
        has_large_values = df.max() > dtype_max_value
        print(has_infinity)

        # Get column names with infinity or large values
        columns_with_infinity = df.columns[has_infinity].tolist()
        columns_with_large_values = df.columns[has_large_values].tolist()

        # Print column names
        if columns_with_infinity:
            print("Columns with infinity:", columns_with_infinity)

        if columns_with_large_values:
            print("Columns with values too large for dtype('float64'):", columns_with_large_values)

        columns_to_drop = df.columns[df.columns.str.contains('Fp1_LEN|SHAN')]
        df_test = df.drop(columns=columns_to_drop)
        df_test = df_test.dropna(axis=1)

        X = df_test.drop("is_adhd", axis=1)
        y = df_test.loc[:, "is_adhd"]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        # normalize training set
        self.normalizer.fit(X_train) # fit accord. to training set
        X_train = self.normalizer.transform(X_train, copy=True)

        # inner CV (hyperparameter tuning)
        inner_cv = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.SEED)
        combinations = {}
        for n_tree in tqdm(self.n_trees):
            for max_feature in self.max_features:
                for max_depth in self.max_depths:
                    for criterion in self.criterions:
                        for min_sample_split in self.min_samples_splits:
                            # model
                            rf = RandomForestClassifier(n_estimators=n_tree,
                                                        criterion=criterion,
                                                        max_depth=max_depth,
                                                        min_samples_split=min_sample_split,
                                                        max_features=max_feature)

                            # CV
                            scores = cross_val_score(rf, X_train, y_train, cv=inner_cv, scoring='f1_weighted')

                            # store score
                            combination = (n_tree, max_feature, max_depth, criterion, min_sample_split)
                            combinations[combination] = np.mean(scores)

        # best hyperparams
        best_combination, best_score = sorted(list(combinations.items()), key=lambda item: item[1])[-1]
        print(best_combination, best_score)
        # use model with best hyperparams
        self.model = RandomForestClassifier(n_estimators=best_combination[0],
                                            criterion=best_combination[3],
                                            max_depth=best_combination[2],
                                            min_samples_split=best_combination[4],
                                            max_features=best_combination[1])

        self.model.fit(X_train, y_train)

    def predict(self, test_indices):
        # load data
        X_test, _ = self.load_data(test_indices)

        # normalize test set
        X_test = self.normalizer.transform(X_test, copy=True)

        return self.model.predict(X_test)


if __name__ =="__main__":
    rfc=RandomForest()
    rfc.fit()
    importances = rfc.feature_importances