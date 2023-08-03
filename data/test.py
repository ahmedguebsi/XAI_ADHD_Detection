import numpy as np
from pandas import DataFrame, read_pickle
from pathlib import Path
df_path=r"C:\Users\Ahmed Guebsi\Desktop\Data_test\.clean_raw_df_adhd.pkl"
df_path_new=r"C:\Users\Ahmed Guebsi\Desktop\Data_test\.clean_raw_df_adhd_new.pkl"


df: DataFrame = read_pickle(df_path)
print(df.describe())
print(df.columns)
num_rows = df.shape[0]
num_columns =df.shape[1]
print("Number of rows:", num_rows)
print("Number of columns:", num_columns)
print(df.head(60))
print(df.columns)
def rows_with_null(df):
    return df[df.isnull().any(axis=1)]
def isnull_any(df):
    # Null and NaN are the same in Pandas :)
    return df.isnull().any()

def isnull_values_sum(df):
    return df.isnull().values.sum() > 0
def isnull_sum(df):
    return df.isnull().sum() > 0
print(rows_with_null(df))
print(isnull_any(df))
print(isnull_values_sum(df))
print(isnull_sum(df))

output_dir = r"C:\Users\Ahmed Guebsi\Desktop\Data_test"
null_rows=df[df.isnull().any(axis=1)]
print(null_rows)
#null_rows.to_csv(str(Path(output_dir, "null_rows_adhd.csv")))

# Find columns with null values
columns_with_null = df.columns[df.isnull().any()]

print(columns_with_null)
# Remove columns containing "KEN"
df_cleaned = null_rows.drop(columns=[col for col in df.columns if 'KEN' in col])

print(df_cleaned.shape)

y=[ 0.02515075 , 1.40937459  ,1.90836891,  1.90522701 ,-2.48040572 , 1.73020856]
y=np.array(y)
print(type(y))
print(int(np.round(y.max(),0)))

# Access columns with "Fp1" in their names
columns_with_f1 = [col for col in df.columns if 'Fp1' in col]

# Access the data of selected columns
data_with_f1 = df[columns_with_f1]

# Print the data of selected columns
print(data_with_f1.head(60))


df_new: DataFrame = read_pickle(df_path_new)
print(df_new.describe())
print(df_new.columns)