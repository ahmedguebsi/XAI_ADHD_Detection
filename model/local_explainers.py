import shap
import xgboost
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (train_test_split)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from typing import Dict, List

from tqdm import tqdm
from pandas import DataFrame, read_pickle, read_csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import (train_test_split)
from sklearn.svm import SVC
from tqdm import tqdm
from environment import channels_good, Chs
from pandas import DataFrame, read_pickle
#from train_models import split_generator
from helper_functions import (glimpse_df,isnull_any, rows_with_null)


def caculate_mode_all(model: ExplainableBoostingClassifier, X_train_org: DataFrame, X_test_org: DataFrame, y_train_org: DataFrame, y_test_org: DataFrame, channels_good: list) -> List:
    """ Calculate accuracy for each channel (Acc_i) by training on the whole dataset """

    channel_acc: Dict[str, float] = {}

    for ch in tqdm(channels_good):
        X_train = X_train_org.loc[:, X_train_org.columns.str.contains(ch)]
        X_test = X_test_org.loc[:, X_test_org.columns.str.contains(ch)]
        #y_train = y_train_org["is_fatigued"]
        y_train = y_train_org.values
        #y_test = y_test_org["is_fatigued"]
        y_test = y_test_org.values
        #model = ExplainableBoostingClassifier()
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        channel_acc[ch] = accuracy_score(y_test, y_test_pred)

    """ Calculate weight for the whole dataset for each channel (V_i). """

    channel_weights = {}
    for channel_a_name in tqdm(channels_good):
        sum_elements = []
        for channel_b_name in channels_good:
            """ Calculate Acc(i,j) and add it to sum expression """
            if channel_b_name == channel_a_name:
                break

            X_train = X_train_org.loc[:, X_train_org.columns.str.contains("|".join([channel_a_name, channel_b_name]))]

            X_test = X_test_org.loc[:, X_test_org.columns.str.contains("|".join([channel_a_name, channel_b_name]))]

            # y_train = y_train_org["is_adhd"]
            y_train = y_train_org.values
            # y_test = y_test_org["is_adhd"]
            y_test = y_test_org.values

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            acc_ij = accuracy_score(y_test, y_test_pred)
            print(acc_ij)
            sum_elements.append(acc_ij + channel_acc[channel_a_name] - channel_acc[channel_b_name])

        sum_expression = sum(sum_elements)
        acc_i = channel_acc[channel_a_name]
        weight = (acc_i + sum_expression) / len(channels_good)
        channel_weights[channel_a_name] = acc_i

    return sorted(channel_weights.items(), key=lambda x: x[1], reverse=True)

if __name__ == '__main__':

    df_path = r"C:\Users\Ahmed Guebsi\Desktop\Data_test\ica_adhd_dataframe_fp1.pkl"
    df: DataFrame = read_pickle(df_path)
    print("shaaaaaaaaaaaape",df.shape)
    glimpse_df(df)
    print(isnull_any(df))
    # glimpse_df(df)
    print(rows_with_null(df))
    print("null columns df", df.columns[df.isnull().any()].tolist())

    channels_good = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4',
                     'P8', '01', '02']

    feature_names = ['mean', 'std', 'MIN', 'MAX', 'MED', 'PEAK', 'SKEW', 'KURT', 'R1', 'RMS', 'M1','IQR', 'Q1', 'Q2', 'Q3', 'WL', 'IEEG',
                     'SPF', 'MOM2','PE', 'MOM3', 'MAV', 'MAV1', 'MAV2', 'COV', 'CF', 'AAC', 'HURST', 'HjC', 'HA', 'HM', 'HFD',
                        'FE','RE','TE','PEN','KEN','SHAN','SUE','CE']

    seed = 42
    np.random.seed(seed)
    # Find columns with null values
    columns_with_null = df.columns[df.isnull().any()]

    print(len(columns_with_null))

    # Access columns with "Fp1" in their names
    columns_with_f1 = [col for col in df.columns if 'Fp1' in col]

    # Access the data of selected columns
    data_with_f1 = df[columns_with_f1]

    # Print the data of selected columns
    print(data_with_f1.head(60))

    # Remove rows with null values
    # df_cleaned = df.dropna().reset_index(drop=True)
    df_cleaned = df.dropna(axis=1)

    print(df_cleaned.shape)

    columns_to_drop = df_cleaned.columns[df_cleaned.columns.str.contains('LEN|PFD|KFD|LE|SE|psd|delta|theta|alpha|beta|alpha_ratio|theta_ratio|beta_ratio|delta_ratio|theta_bata_ratio')]
    df_test = df_cleaned.drop(columns=columns_to_drop)
    print(df_test.shape)
    # Load data and labels
    X = df_test.drop(["is_adhd","epoch_id","child_id"], axis=1)
    y = df_test.loc[:, "is_adhd"]
    # Split data into training and testing sets
    #X_train, X_test, y_train, y_test = split_generator(X, y)
    X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(X, y, test_size=0.2, shuffle=True,random_state=seed)

    # train an XGBoost model
    #X, y = shap.datasets.boston()
    #model = xgboost.XGBRegressor().fit(X, y)
    model = xgboost.XGBClassifier().fit(X_train_org, y_train_org)
    y_pred=model.predict(X_test_org)
    acc= accuracy_score(y_test_org, y_pred)
    print("accuracy",acc)

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0])

    # visualize the first prediction's explanation with a force plot
    shap.plots.force(shap_values[0])