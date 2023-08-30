from alibi.explainers import PartialDependence, plot_pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (train_test_split)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from tqdm import tqdm
from pandas import DataFrame, read_pickle, read_csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay, partial_dependence # when you need raw values of the partial dependence function rather than the plots
from helper_functions import (glimpse_df,isnull_any, rows_with_null)


def partial_dep(model: ExplainableBoostingClassifier,X_train_org: DataFrame, X_test_org: DataFrame, y_train_org: DataFrame, y_test_org: DataFrame, channels_good: list):
    for ch in tqdm(channels_good):
        X_train = X_train_org.loc[:, X_train_org.columns.str.contains(ch)]
        X_test = X_test_org.loc[:, X_test_org.columns.str.contains(ch)]
        # y_train = y_train_org["is_adhd"]
        y_train = y_train_org.values
        #y_test = y_test_org["is_adhd"]
        y_test = y_test_org.values

        features = X_train.columns
        print(features)

        model.fit(X_train, y_train)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Explainable Boosting Classifier")
        ebc_disp =PartialDependenceDisplay.from_estimator(model, X_train, features, kind='average',n_cols=2, ax=ax)
        plt.show()

        results = partial_dependence(model, X_train, [0])
    return ebc_disp



def alibi_partial_dep(model: ExplainableBoostingClassifier,X_train_org: DataFrame, X_test_org: DataFrame, y_train_org: DataFrame, y_test_org: DataFrame, channels_good: list):
    # define and fit the oridnal encoder
    #categorical_columns_names= X_train_org.columns[X_train_org.columns.str.contains("ch")].tolist()
    #oe = OrdinalEncoder().fit(X_train[categorical_columns_names])

    X_train = X_train_org.loc[:, X_train_org.columns.str.contains(ch)]
    X_test = X_test_org.loc[:, X_test_org.columns.str.contains(ch)]
    y_train = y_train_org["is_adhd"]
    y_test = y_test_org["is_adhd"]

    # convert data to numpy
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
    feature_names = X_train.columns
    # define target names
    target_names = ['Number of bikes']

    numerical_columns_indices = [feature_names.index(fn) for fn in feature_names]

    # define numerical standard sclaer
    num_transf = StandardScaler()

    # define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transf, numerical_columns_indices),
        ],
        sparse_threshold=0
    )

    # define and fit regressor - feel free to play with the hyperparameters
    #predictor = ExplainableBoostingClassifier()
    model.fit(X_train_ohe, y_train)

    # compute scores
    print('Train score: %.2f' % (model.score(X_train_ohe, y_train)))
    print('Test score: %.2f' % (model.score(X_test_ohe, y_test)))

    prediction_fn = lambda x: model.predict(preprocessor.transform(x))

    # define explainer
    explainer = PartialDependence(predictor=prediction_fn,feature_names=feature_names,target_names=target_names)

    # select temperature, humidity, wind speed, and season
    features = [feature_names.index(fn) for fn in feature_names]

    # compute explanations
    exp = explainer.explain(X=X_train,
                            features=features,
                            kind='average')

    # plot partial dependece curves
    plot_pd(exp=exp,
            n_cols=3,
            sharey='row',
            fig_kw={'figheight': 10, 'figwidth': 15});
    return 0

if __name__ == "__main__":
    df_path = r"C:\Users\Ahmed Guebsi\Desktop\Data_test\final_modified_df.pkl"
    df: DataFrame = read_pickle(df_path)
    print("shaaaaaaaaaaaape", df.shape)
    glimpse_df(df)
    print(isnull_any(df))
    # glimpse_df(df)
    print(rows_with_null(df))
    print("null columns df", df.columns[df.isnull().any()].tolist())

    channels_good = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4','P8', '01', '02']

    feature_names = ['mean', 'std', 'MIN', 'MAX', 'MED', 'PEAK', 'SKEW', 'KURT', 'R1', 'RMS', 'M1', 'IQR', 'Q1', 'Q2',
                     'Q3', 'WL', 'IEEG', 'SPF', 'MOM2', 'PE', 'MOM3', 'MAV', 'MAV1', 'MAV2', 'COV', 'CF', 'AAC', 'HURST', 'HjC', 'HA', 'HM',
                     'HFD','FE', 'RE', 'TE', 'PEN', 'KEN', 'SHAN', 'SUE', 'CE']

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

    columns_to_drop = df_cleaned.columns[df_cleaned.columns.str.contains(
        'LEN|PFD|KFD|LE|SE|psd|delta|theta|alpha|beta|alpha_ratio|theta_ratio|beta_ratio|delta_ratio|theta_bata_ratio')]
    df_test = df_cleaned.drop(columns=columns_to_drop)
    print(df_test.shape)
    # Load data and labels
    X = df_test.drop(["is_adhd", "epoch_id", "child_id"], axis=1)
    y = df_test.loc[:, "is_adhd"]
    # Split data into training and testing sets
    # X_train, X_test, y_train, y_test = split_generator(X, y)
    X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(X, y, test_size=0.2, shuffle=True,
                                                                        random_state=seed)
    model = ExplainableBoostingClassifier()
    res=partial_dep(model, X_train_org, X_test_org, y_train_org, y_test_org, channels_good)
    res.show()