

import numpy as np
import pandas as pd
from pandas import DataFrame, read_pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from environment import training_columns_regex

import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
from interpret.blackbox import ShapKernel, LimeTabular, MorrisSensitivity, PartialDependence
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())


#df_path=r"C:\Users\Ahmed Guebsi\Downloads\complete-clean-2022-02-26-is_complete_dataset_true___brains_true___reref_false.pickle"
df_path=r"C:\Users\Ahmed Guebsi\Desktop\Data_test\.clean_raw_df_adhd.pkl"
df: DataFrame = read_pickle(df_path)
# Convert complex values to real
df_real = df.applymap(lambda x: x.real)

#print(isnull_any(df))
#glimpse_df(df)
#print(rows_with_null(df))

seed = 42
np.random.seed(seed)

training_columns = list(df.iloc[:, df.columns.str.contains(training_columns_regex)].columns)

# Remove rows with null values
df_cleaned = df.dropna().reset_index(drop=True)

X = df_cleaned.drop("is_adhd", axis=1)
y = df_cleaned.loc[:, "is_adhd"]
print(y)
print(df_cleaned.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)
auc = roc_auc_score(y_test, ebm.predict_proba(X_test)[:, 1])
print("AUC: {:.3f}".format(auc))

pca = PCA()
rf = RandomForestClassifier(random_state=seed)

blackbox_model = Pipeline([('pca', pca), ('rf', rf)])
blackbox_model.fit(X_train, y_train)

lime = LimeTabular(blackbox_model, X_train)
show(lime.explain_local(X_test[:5], y_test[:5]), 0)

plt.show()

pdp = PartialDependence(blackbox_model, X_train)
show(pdp.explain_global(), 0)

msa = MorrisSensitivity(blackbox_model, X_train)
show(msa.explain_global())

shap = ShapKernel(blackbox_model, X_train)
shap_local = shap.explain_local(X_test[:5], y_test[:5])

show(shap_local, 0)