import numpy as np
from pandas import DataFrame, read_pickle, read_csv
from pathlib import Path
import pandas as pd
df_path=r"C:\Users\Ahmed Guebsi\Desktop\Data_test\.clean_raw_df_adhd.pkl"
df_path_new=r"C:\Users\Ahmed Guebsi\Desktop\Data_test\.clean_raw_df_adhd_new.pkl"
df_path_no_ica=r"C:\Users\Ahmed Guebsi\Desktop\Data_test\.clean_raw_df_adhd_no_ica.pkl"

output_dir = r"C:\Users\Ahmed Guebsi\Desktop\Data_test"







def check_dataframe(df: DataFrame, drop_columns: bool = False):


    has_null = df.isnull().any().any()
    print("Has null values:", has_null)
    # Find columns with null values
    null_final_df_columns = df.columns[df.isnull().any()]

    print(null_final_df_columns)

    # Check for infinity in the DataFrame
    has_infinity = df.isin([np.inf, -np.inf]).any()

    # Check for values too large for float64 dtype
    dtype_max_value = np.finfo(np.float64).max
    has_large_values = df.max() > dtype_max_value

    # Get column names with infinity or large values
    columns_with_infinity = df.columns[has_infinity].tolist()
    columns_with_large_values = df.columns[has_large_values].tolist()

    # Print column names
    df_cleaned = df

    if columns_with_infinity:
        print("Columns with infinity:", columns_with_infinity)

    if columns_with_large_values:
        print("Columns with values too large for dtype('float64'):", columns_with_large_values)

    if has_null and drop_columns:
        df_cleaned = df.drop(columns=null_final_df_columns, axis=1)
    if len(has_infinity) != 0 and drop_columns:
        df_cleaned = df_cleaned.drop(columns=columns_with_infinity, axis=1)
    if len(has_large_values) != 0 and drop_columns:
        df_cleaned = df_cleaned.drop(columns=columns_with_large_values, axis=1)


    return df_cleaned
df_vmd_dsa_path=r"C:\Users\Ahmed Guebsi\Desktop\Data_test\kraskov_dataframe.pkl"
df_vmd_dsa: DataFrame = read_pickle(df_vmd_dsa_path)
df_test_copy = df_vmd_dsa.copy()
print(df_vmd_dsa.shape)
print(df_vmd_dsa.head(60))
# Add '_SHAN_standard' to column names
new_columns = [col + '_KE' for col in df_test_copy.columns]
df_test_copy.rename(columns=dict(zip(df_test_copy.columns, new_columns)), inplace=True)

df_test_copy.to_pickle(str(Path(output_dir, "KE_dataframe.pkl")))



df_vmd_dsa_path=r"C:\Users\Ahmed Guebsi\Desktop\Data_test\vmd_dsa_dataframe.pkl"
df_vmd_dsa: DataFrame = read_pickle(df_vmd_dsa_path)
print(df_vmd_dsa.shape)
print(check_dataframe(df_vmd_dsa,drop_columns=False).shape)

df_vmd_dsa_fp1_path=r"C:\Users\Ahmed Guebsi\Desktop\Data_test\krask_ch_load.pkl"
df_vmd_dsa_fp1: DataFrame = read_pickle(df_vmd_dsa_fp1_path)
print(df_vmd_dsa_fp1.shape)
print(df_vmd_dsa_fp1[0][22:47])
print(df_vmd_dsa_fp1.head(60))
print(max(df_vmd_dsa_fp1[0]))
channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8','P7', 'P3', 'Pz', 'P4', 'P8', '01', '02']
# Reverse the rows
reversed_df = df_vmd_dsa_fp1.iloc[::-1].reset_index(drop=True)

print(reversed_df.head(20))

# Define the number of columns you want
num_columns = 19
kraskov_data =np.array(df_vmd_dsa_fp1[0])

# Define the value(s) you want to filter and remove
values_to_remove = [float(i) for i in range(84)]

# Filter and remove rows containing the specified values
filtered_df = df_vmd_dsa_fp1[~df_vmd_dsa_fp1[0].isin(values_to_remove)]
# Reset the index labels
filtered_df = filtered_df.reset_index(drop=True)
# Display the filtered DataFrame
print("\nDataFrame after removing rows:")
# Define the number of columns you want
num_columns = 19
kraskov_data =np.array(filtered_df[0])
kraskov_data=np.resize(kraskov_data,(kraskov_data.shape[0]//num_columns, num_columns))
print(kraskov_data.shape)

kraskov_dataframe =pd.DataFrame(kraskov_data, columns=channel_names)
kraskov_dataframe.to_pickle(str(Path(output_dir, "kraskov_dataframe.pkl")))


print(check_dataframe(df_vmd_dsa_fp1,drop_columns=True).shape)
df_cleaned=check_dataframe(df_vmd_dsa_fp1,drop_columns=True)
print(df_cleaned.columns)
columns_to_drop = df_cleaned[df_cleaned.columns.str.contains('Fp1|Fp2|F7|F3|Fz|F4|F8|T7|C3|Cz|C4|T8|P7|P3|Pz|P8|01|02')]
print(columns_to_drop)
