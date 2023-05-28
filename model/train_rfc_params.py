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
from helper_functions import (get_cnt_filename, glimpse_df, serialize_functions,isnull_any)

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
        self.max_features = ['auto', 'sqrt', 'log2']
        self.max_depths = [10, 30, 50, 100]
        self.criterions = ['gini', 'entropy']
        self.min_samples_splits = [2, 5, 10]

    def fit(self):
        # load data
        df_path = r"C:\Users\Ahmed Guebsi\Downloads\complete-clean-2022-02-26-is_complete_dataset_true___brains_true___reref_false.pickle"
        df: DataFrame = read_pickle(df_path)
        glimpse_df(df)

        X = df.drop("is_fatigued", axis=1)
        y = df.loc[:, "is_fatigued"]

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