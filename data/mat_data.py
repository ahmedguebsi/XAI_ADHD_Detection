import scipy.io


from environment import (ADHD_STR, FREQ, USE_REREF, LOW_PASS_FILTER_RANGE_HZ,
                              NOTCH_FILTER_HZ, SIGNAL_OFFSET,CHANNELS,Chs,
                              channels_good, attention_states, feature_names,
                              get_brainwave_bands, NUM_CHILDREN, SIGNAL_DURATION_SECONDS_DEFAULT,custom_mapping)

from helper_functions import (get_mat_filename, get_mat_file_name, serialize_functions, glimpse_df)
import scipy.io as sio
from pathlib import Path
from tqdm import tqdm
from itertools import product
from typing import Dict, List, Tuple
import numpy as np
from pandas import DataFrame, read_pickle, to_pickle
import pandas as pd

data_directory = "ADHD_part1"
output_dir = r"C:\Users\Ahmed Guebsi\Desktop\Data_test"
PATH_DATASET_MAT= r"C:\Users\Ahmed Guebsi\Downloads\ADHD_part1"
children_num = NUM_CHILDREN

# Example label for each .mat file
label = 0  # Replace with the actual label
# Add label information to each .mat file
mat_file_paths = []



for child_id, attention_state in tqdm(list(product(range(0, children_num), attention_states))):
    is_adhd = 1 if attention_state == ADHD_STR else 0
    signal_filepath = str(Path(PATH_DATASET_MAT, get_mat_filename(child_id + 1, attention_state)))

    print(signal_filepath)
    mat_file_paths.append(signal_filepath)
print(len(mat_file_paths))

for mat_file_path in mat_file_paths:
    mat_data = scipy.io.loadmat(mat_file_path)
    # Specify the key to delete
    key_to_delete = 'label'

    # Delete the element with the specified key
    if key_to_delete in mat_data:
        del mat_data[key_to_delete]

    scipy.io.savemat(mat_file_path, mat_data)

signal_filepath = str(Path(PATH_DATASET_MAT, get_mat_filename(0 + 1, "normal")))
mat_data = sio.loadmat(signal_filepath)
last_key, last_value = list(mat_data.items())[-1]


# Specify the key to delete
key_to_delete = 'label'

# Delete the element with the specified key
if key_to_delete in mat_data:
    del mat_data[key_to_delete]

print(mat_data.keys())

print("Last Key:", last_key)
print("Last Value:", last_value.shape)

ica_path=r"C:\Users\Ahmed Guebsi\Desktop\Data_test\ica_dataframe.pkl"
ica_df: DataFrame = read_pickle(ica_path)
print(ica_df.shape)
print(ica_df.describe())
print(ica_df.head(10))

second_ica_path=r"C:\Users\Ahmed Guebsi\Desktop\Data_test\second_ica_dataframe.pkl"
second_ica_df: DataFrame = read_pickle(second_ica_path)
print(second_ica_df.shape)
final_ica_df = ica_df._append(second_ica_df)
print(final_ica_df.shape)

final_ica_df.to_pickle(str(Path(output_dir, "final_ica_dataframe.pkl")))