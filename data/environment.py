NUM_CHILDREN = 60
ADHD_STR = "adhd"
NORMAL_STR = "normal"
SIGNAL_OFFSET = -20


attention_states = [NORMAL_STR, ADHD_STR]
USE_ICA = False
USE_REREF = False


""" Signal config"""
FREQ = 128
EPOCH_SECONDS = 1
SIGNAL_FILE_DURATION_SECONDS = 600
SIGNAL_DURATION_SECONDS_DEFAULT = 300
NOTCH_FILTER_HZ = 50
LOW_PASS_FILTER_RANGE_HZ = (0.1, 60)


""" Channels config """
channels_all = ["HEOL", "HEOR", "FP1", "FP2", "VEOU", "VEOL", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ", "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ", "CP4", "TP8", "A1", "T5", "P3", "PZ", "P4", "T6", "A2", "O1", "OZ", "O2", "FT9", "FT10", "PO1", "PO2"]
channels_good = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ", "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ", "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2"]
channels_bad = list(set(channels_all) - set(channels_good))
CHANNELS =['Channel 1','Channel 2','Channel 3','Channel 4','Channel 5','Channel 6','Channel 7','Channel 8','Channel 9','Channel 10','Channel 11','Channel 12','Channel 13','Channel 14','Channel 15','Channel 16','Channel 17','Channel 18','Channel 19']
CHS_PAPAER=['Fz', 'Cz', 'Pz', 'C3', 'T3', 'C4', 'T4', 'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8','P3', 'P4', 'T5', 'T6', 'O1', 'O2']
CHS=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8','T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
Chs=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8','P7', 'P3', 'Pz', 'P4', 'P8', '01', '02']

regions = ["FP", "F", "C", "P", "O", "T"]
custom_mapping = {
    'Fp1': 'Prefrontal',
    'Fp2': 'Prefrontal',
    'Fz': 'Frontal',
    'F3': 'Frontal',
    'F4': 'Frontal',
    'F7': 'Frontal',
    'F8': 'Frontal',
    'Cz': 'Central',
    'C3': 'Central',
    'C4': 'Central',
    'T7': 'Temporal',
    'T8': 'Temporal',
    'Pz': 'Parietal',
    'P3': 'Parietal',
    'P4': 'Parietal',
    'P7': 'Parietal',
    'P8': 'Parietal',
    'O1': 'Occipital',
    'O2': 'Occipital',
    # Add more mappings as needed
}
regions_mapping = {"FP": ["FP1", "FP2"], "F": ["F7", "F3", "FZ", "F4", "F8"], "C": ["FT7", "FC3", "FCZ", "FC4", "FT8"], "P": ["TP7", "CP3", "CPZ", "CP4", "TP8"], "O": ["T7", "P3", "PZ", "P4", "T8"], "T": ["T3", "C3", "CZ", "C4", "T4"]}
def get_brainwave_bands():
    return {"AL": (8, 10 + 1), "AH": (10, 12 + 1), "BL": (13, 19 + 1), "BH": (19, 25 + 1)}


additional_feature_names = ["psd", "mean", "std","MIN","MAX","MED","PEAK","SKEW","KURT","R1","RMS","M1","IQR","Q1","Q2","Q3","WL","IEEG","SPF","MOM2",
                            "MOM3","MAV","MAV1","MAV2","COV","CF","AAC","HURST","HjC","HA","HM","HFD","alpha","beta","theta","delta","alpha_ratio",
                            "beta_ratio","theta_ratio","delta_ratio","theta_beta_ratio"]
entropy_names = ["PE", "AE", "SE", "FE","RE","TE","PEN","KEN","SHAN","LEN","SUE"]
feature_names = entropy_names + additional_feature_names
feature_indices = dict((name, i) for i, name in enumerate(feature_names))


#training_columns_regex = "|".join(channels_good)
training_columns_regex = "|".join(Chs)


# [PE_FP1, PE_FP2, ... , PE_C3, AE_FP1, AE_FP2, ..., FE_C3]
entropy_channel_combinations = ["{}_{}".format(entropy, channel) for entropy in entropy_names for channel in channels_good]