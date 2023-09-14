import scipy.io as sio
import h5py
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew,kurtosis, gaussian_kde
import mne
import pyedflib
import matplotlib.pyplot as plt
from pathlib import Path
import os
from mne.epochs import make_fixed_length_epochs
from mne.io.base import BaseRaw
from mne import Epochs
from mne.preprocessing import ICA
from autoreject import (get_rejection_threshold, AutoReject)

import time
from vmdpy import VMD

from mne.filter import notch_filter, filter_data
from itertools import product
from typing import List, Union
from tqdm import tqdm

from pandas import DataFrame, Series, read_pickle, set_option
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from environment import (ADHD_STR, FREQ, USE_REREF, LOW_PASS_FILTER_RANGE_HZ,
                              NOTCH_FILTER_HZ, SIGNAL_OFFSET,CHANNELS,Chs,
                              channels_good, attention_states, feature_names,
                              get_brainwave_bands, NUM_CHILDREN, SIGNAL_DURATION_SECONDS_DEFAULT,custom_mapping)

from helper_functions import (get_mat_filename, get_mat_file_name, serialize_functions, glimpse_df)
from signals import SignalPreprocessor

import tensorflow as tf

import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.optimizers import Adam

from tf_explain.callbacks.smoothgrad import SmoothGradCallback

from matplotlib import pyplot as plt

from deep_models import cnnlstm, square, log
from data_preprocess import data_load, get_k_fold_data, flatten_test, easy_shuffle, split_data
from eeg_visualize import test_visualize


epoch_events_num = FREQ
children_num = NUM_CHILDREN
signal_duration = SIGNAL_DURATION_SECONDS_DEFAULT
#channels = CHANNELS
channels = Chs

PATH_DATASET_MAT= r"C:\Users\Ahmed Guebsi\Downloads\ADHD_part1"



def load_mat_data(signal_filepath):
    # Load the MATLAB .mat file

    #mat_data = sio.loadmat(r"C:\Users\Ahmed Guebsi\Downloads\ADHD_Data\ADHD_part1\v1p.mat")
    mat_data = sio.loadmat(signal_filepath)
    #filename= get_mat_file_name(signal_filepath)

    # Extract the EEG signal data from the loaded .mat file
    #eeg_signal = mat_data['eeg']
    print(mat_data.keys())

    last_key, last_value = list(mat_data.items())[-1]

    eeg_signal = np.transpose(last_value)
    print("teeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeest",eeg_signal.shape)


    # Create the info dictionary
    n_channels, n_samples = eeg_signal.shape

    #ch_names = [f'Channel {i}' for i in range(1, n_channels + 1)]
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8','01', '02']
    #ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=128,ch_types='eeg')

    # Create the RawArray object
    raw = mne.io.RawArray(eeg_signal, info)
    print("raaaaaaaaaaaaaaaaw",raw.get_data().shape)

    return raw, info, ch_names,eeg_signal


def get_electrodes_coordinates(ch_names):
    #df = pd.read_csv("/content/drive/My Drive/Standard-10-20-Cap19new.txt", sep="\t")
    df = pd.read_csv(r"C:\Users\Ahmed Guebsi\Downloads\Standard-10-20-Cap19new.txt", sep="\t")
    df.head()

    electrodes_coordinates = []

    for i in range(len(ch_names)):
        xi, yi, zi = df.loc[i, 'X':'Z']
        #electrodes_coordinates.append([xi, yi, zi])
        electrodes_coordinates.append([-yi, xi, zi])

    return electrodes_coordinates


def get_electrodes_positions(ch_names, electrodes_coordinates):

   dig_points=[]
   for i in range(len(ch_names)):
       electrode_positions = dict()
       electrode_positions[ch_names[i]]=np.array(electrodes_coordinates[i])
       dig_points.append(electrode_positions)

   return dig_points


def set_electrodes_montage(ch_names, electrodes_coordinates,eeg_signal):
  # Create a montage with channel positions
  montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, electrodes_coordinates)))
  info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types='eeg')

  info.set_montage(montage)

  # Create a Raw object
  raw = mne.io.RawArray(eeg_signal, info)

  # Access the channel positions from the info attribute
  print(raw.info['dig'])

  return raw,info

def montage_data(raw, fname_mon):

  dig_montage = read_custom_montage(fname_mon, head_size=0.095, coord_frame="head")
  # read_custom_montage(fname, head_size=0.095, coord_frame=None)

  raw.set_montage(dig_montage)
  #   set_montage(montage, match_case=True, match_alias=False, on_missing='raise', verbose=None)

  std_montage = make_standard_montage('standard_1005')
  #             make_standard_montage(kind, head_size='auto')

  # raw.set_montage(std_montage)

  trans = compute_native_head_t(dig_montage)

  return dig_montage, trans

def filt_data(eegData, lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filt_eegData = signal.lfilter(b, a, eegData, axis = 1)
    return filt_eegData

def preprocess_eeg_data(raw,info):
    # Apply notch filter at 50 Hz
    notch_filtered = notch_filter(raw.get_data(), Fs=raw.info['sfreq'], freqs=50)

    # Apply sixth-order Butterworth bandpass filter (0.1 - 60 Hz)
    #bandpass_filtered = filter_data(notch_filtered, sfreq=raw.info['sfreq'], l_freq=0.1, h_freq=60, method='fir')
    bandpass_filtered = filt_data(notch_filtered, lowcut=0.1, highcut=60, fs=raw.info['sfreq'], order=6)

    signal_filtered = mne.io.RawArray(bandpass_filtered, info)
    return signal_filtered
def gpu_start():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    physical_devices = tf.config.list_physical_devices('GPU')
    for i in range(4):
        print('gpu', i)
        tf.config.experimental.set_memory_growth(physical_devices[i], True)



def cross_validation_generator(X, y,k=10):
    # cross validation
    Kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    Kfold.get_n_splits(X, y)
    foldNum = 0  # initializing fold = 0
    for train_index, test_index in Kfold.split(X, y):
        foldNum += 1
        print("fold", foldNum)
        X_train, X_test = X[train_index], X[test_inde]
        y_train, y_test = y[train_index], y[test_index]
        yield X_train, X_test, y_train, y_test, foldNum

def plot_history(history, result_dir, fold_index):
    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy-' + str(fold_index) + '.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss-' + str(fold_index) + '.png'))
    plt.close()


def save_history(history, result_dir, fold_index):
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    nb_epoch = len(acc)


    with open(os.path.join(result_dir, 'train-result-' + str(fold_index) + '.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))

def train_eeg_model(params, train_dataset, valid_dataset, fold_index):
    chans, samples = params['chans'], params['samples']
    # multi gpu
    strategy = params['strategy']
    with strategy.scope():

        model = cnnlstm(nb_classes=2, Chans=chans, Samples=samples, dropoutRate=0.5, weight_decay=0.1)

        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=params['lr']),
                      metrics=['accuracy'])
        # count number of parameters in the model
        numParams = model.count_params()
        print('Total number of parameters: ' + str(numParams))

    models_path = os.path.join(params['output_dir'], 'saved_models')
    if not os.path.isdir(models_path):
        os.mkdir(models_path)
    save_model_weights_path = models_path + '/model-weights-' + str(fold_index) + '.h5'

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath=save_model_weights_path, verbose=1, monitor='val_loss',
                                   mode='auto', save_best_only=True)
    # initialize the weights all to be 1
    class_weights = {0: 1, 1: 1}

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    print('Training...')
    print(type(valid_dataset))


    #model.fit(x_train, y_train, batch_size=2, epochs=2, callbacks=callbacks)
    model_history = model.fit(train_dataset,epochs=params['num_epochs'],
                              verbose=2, validation_data=valid_dataset,
                              callbacks=[checkpointer, reduce_lr, early_stop], class_weight=class_weights)

    plot_history(model_history, params['output_dir'], fold_index)
    save_history(model_history, params['output_dir'], fold_index)
    model.summary()

    model_structure_json = model.to_json()
    save_models_structure_path = os.path.join(models_path, 'model_architecture' + str(fold_index) + '.json')
    open(save_models_structure_path, 'w').write(model_structure_json)
    params['model_structure_path'] = save_models_structure_path
    params['model_weights_path'] = save_model_weights_path


def test_main(params, test_x, test_y, fold_index):
    probs, real, acc = test_model(params, X_test=test_x, Y_test=test_y, fold_index=fold_index)


def test_model(params, X_test, Y_test):
    X_test = X_test.reshape(X_test.shape[0], params['chans'], params['samples'], params['kernels'])
    print(X_test.shape)
    model = tf.keras.models.load_model(params['model_weights_path'], custom_objects={'square': square, 'log': log})
    probs = model.predict(X_test)
    print("tttttttttttttttttttttttttttt",probs)
    print(probs.shape)
    print(y_test.shape)
    preds = probs.argmax(axis=-1)
    print(preds)
    real = Y_test.argmax(axis=-1)
    acc = np.mean(preds == real)
    params['acc'].append(acc.item())
    print(probs.shape)
    print(X_test.shape)
    print("Classification accuracy: %f " % acc)

    return probs, real, acc


def train_main(params, X_data, y_data):

    # ---------------------------- k fold validation part------------------------
    X_data, y_data = easy_shuffle(X_data, y_data)
    X_test, y_test = X_data[0:800, :], y_data[0:800,:]
    print('y_test:  ', y_test)
    print('y_test.shape', y_test.shape)
    kpart_x, kpart_y = X_data[801:, :], y_data[801:, :]

    params['mode_'] = 'k_fold'
    k = params['k']

    for i in range(k):
        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, kpart_x, kpart_y)

        #for X_train, y_train, X_valid, y_valid , nbFold in cross_validation(X_data, y_data):
        print('the fold times: ', nbFold)
        print('X_train shape:', X_train.shape)
        print('X_valid shape:', X_valid.shape)
        print('y_train shape:', y_train.shape)
        print('y_valid shape:', y_valid.shape)
        print('batch_size: ', params['batch_size'])

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        print('train_dataset: ', train_dataset)
        train_dataset = train_dataset.shuffle(buffer_size=X_train.shape[0]).batch(params['batch_size'])
        #print('train_dataset batch: ', train_dataset)

        valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
        print('valid_dataset: ', valid_dataset)
        valid_dataset = valid_dataset.shuffle(buffer_size=X_valid.shape[0]).batch(params['batch_size'])
        print('valid_dataset batch: ', valid_dataset)


        train_eeg_model(params=params, train_dataset=train_dataset, valid_dataset=valid_dataset, fold_index=nbFold)
        print('ready for testing....')
        test_main(params=params, test_x=X_test, test_y=y_test, fold_index=i)
    print('cross subjects test was finished.............: ')
    print('cross subjects all acc list: ', params['acc'])
    print('cross subjects mean acc: ', np.mean(params['acc']))

def apply_ica(raw, info, plot_components=False, plot_scores=False,plot_sources=False, plot_overlay=False):

    raw.load_data()

    raw_ica = raw.copy()
    raw_ica.load_data()
    # Break raw data into 1 s epochs
    tstep = 4.0
    events_ica = mne.make_fixed_length_events(raw_ica, duration=tstep)
    epochs_ica = mne.Epochs(raw_ica, events_ica,
                            tmin=0.0, tmax=tstep,
                            baseline=None,
                            preload=True)
    reject = get_rejection_threshold(epochs_ica);
    ica_z_thresh = 1.96
    ica_v = mne.preprocessing.ICA(n_components=.99, random_state=42)


    ica_v.fit(raw_ica,picks='eeg', reject=reject, tstep=tstep)

    if plot_components:
        ica_v.plot_components();

    explained_var_all_ratio = ica_v.get_explained_variance_ratio(
        raw, components=[0], ch_type="eeg"
    )
    # This time, print as percentage.
    ratio_percent = round(100 * explained_var_all_ratio["eeg"])
    print(
        f"Fraction of variance in EEG signal explained by first component: "
        f"{ratio_percent}%"
    )
    eog_indices, eog_scores = ica_v.find_bads_eog(raw_ica,
                                                  ch_name=['Fp1', 'F8'],
                                                  threshold=ica_z_thresh)
    ica_v.exclude = eog_indices
    print(ica_v.exclude)

    #list_removed_components.append(ica_v.exclude)

    if plot_scores:
        p = ica_v.plot_scores(eog_scores);

    if plot_overlay:
        o = ica_v.plot_overlay(raw_ica, exclude=eog_indices, picks='eeg');

    reconstructed_raw = ica_v.apply(raw_ica)

    if plot_sources:
        res = ica_v.plot_sources(reconstructed_raw);

    #sources = ica_v.get_sources(epochs_ica);
    #print(type(sources))
    # Apply ICA to EEG data
    # ica.apply(raw)
    # print(ica.exclude)


    return reconstructed_raw


def data_load(path):
    x_data, y_data = [], []
    subIdx =[]

    x_data = np.array(x_data)
    signal_preprocessor = SignalPreprocessor()

    filter_frequencies = {"standard": LOW_PASS_FILTER_RANGE_HZ}

    """ preprocessing procedures that will filter frequencies defined in filter_frequencies"""
    for freq_name, freq_range in filter_frequencies.items():
        low_freq, high_freq = freq_range

        procedure = serialize_functions(
            lambda s: preprocess_eeg_data(s, info),
        )

        signal_preprocessor.register_preprocess_procedure(freq_name, procedure=procedure,
                                                          context={"freq_filter_range": freq_range})

    nb_epoch_adhd =0
    nb_epoch_normal =0
    compt =1
    for child_id, attention_state in tqdm(list(product(range(1, children_num +1), attention_states))):
        is_adhd = 1 if attention_state == ADHD_STR else 0
        signal_filepath = str(Path(path, get_mat_filename(child_id, attention_state)))

        print(signal_filepath)
        raw, info, ch_names, eeg_signal = load_mat_data(signal_filepath)
        print(raw)
        print(type(raw))
        print(raw.info)

        # raw.plot(duration=10, n_channels=10, scalings='auto', title='Auto-scaled Data from arrays')

        # print(f"Duration of the signal is {dur_tot} seconds")

        electrodes_coordinates = get_electrodes_coordinates(ch_names)
        # print(electrodes_coordinates)
        dig_points = get_electrodes_positions(ch_names, electrodes_coordinates)
        raw_with_dig_pts, info = set_electrodes_montage(ch_names, electrodes_coordinates, eeg_signal)
        # print(type(raw_with_dig_pts))
        # raw_with_dig_pts.plot_sensors(show_names=True)
        signal_preprocessor.fit(raw_with_dig_pts)
        print(type(signal_preprocessor))
        # apply_ica(raw_with_dig_pts, info)
        for proc_index, (signal_processed, proc_name, proc_context) in tqdm(enumerate(signal_preprocessor.get_preprocessed_signals())):
            print(type(signal_processed))
            scan_durn = signal_processed._data.shape[1] / signal_processed.info['sfreq']
            # signal_duration = int(scan_durn)
            signal_duration = round((signal_processed._data.shape[1] - 1) / signal_processed.info["sfreq"],
                                    3)  # corrected

            signal_processed.crop(tmin=0, tmax=signal_duration // 4 * 4)

            # assuming you have a Raw and ICA instance previously fitted
            print(type(signal_processed))

            signal_processed_ica = apply_ica(signal_processed, info, plot_components=False, plot_scores=False,plot_sources=False, plot_overlay=False)

            epochs = make_fixed_length_epochs(signal_processed_ica, duration=4.0, preload=True, verbose=False)
            print(epochs)
            print(len(epochs))  # 47
            # epochs.plot(picks="all", scalings="auto", n_epochs=3, n_channels=19, title="plotting epochs")
            # epochs = mne.Epochs(signal_processed, epochs, tmin=0, tmax=1, baseline=None, detrend=1, preload=True)
            print(epochs.get_data().shape)  # (47,19, 512) we added 2s overlap

            n_epochs = len(epochs)


            print("epoch check shape", epochs.get_data().T.shape)  # (512,19, 47)

            if x_data.size == 0:
                x_data = epochs.get_data().T
            else:
                x_data = np.concatenate((x_data, epochs.get_data().T), axis=2)
                print(x_data.shape)
            y_data.extend([is_adhd] * n_epochs)
            if is_adhd == 1:
                subIdx.extend([compt +1 ] * n_epochs)
                compt += 2
                nb_epoch_adhd +=n_epochs
                #print(subIdx)
            else:
                subIdx.extend([compt] * n_epochs)
                nb_epoch_normal += n_epochs
                #print(subIdx)
            print(len(y_data))
    subIdx = np.array(subIdx).reshape(1,len(subIdx))
    y_data = np.array(y_data).reshape(1,len(y_data))

    print(y_data.shape)
    print(y_data.ndim)

    # Create labels for the classes
    class_labels = ['Normal ', 'ADHD ']

    # Create a bar plot
    plt.bar(class_labels, [nb_epoch_normal, nb_epoch_adhd], color=['blue', 'red'])

    # Add labels and a title
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Distribution of Dataset')

    # Add counts as text labels on top of the bars
    for i, count in enumerate([nb_epoch_normal, nb_epoch_adhd]):
        plt.text(i, count, str(count), ha='center', va='bottom', fontsize=12)
    #plt.show()
    return x_data, y_data, subIdx


if __name__ == '__main__':

    params = {
        'output_dir': r'C:\Users\Ahmed Guebsi\Desktop\ahmed_files\NO_ICA',
        'strategy': tf.distribute.MultiWorkerMirroredStrategy(),
        'num_epochs': 300, 'batch_size': 32,
        'k': 5, 'lr': 1e-3,
        'kernels': 1, 'chans': 19, 'samples': 512,
        'mode_': 'k-fold', 'acc': [],
        'model_weights_path': '', 'model_structure_path': ''
    }

    x_data, y_data, subIdx = data_load(PATH_DATASET_MAT)
    x_data = np.swapaxes(x_data, 2, 0)
    y_data = np.swapaxes(y_data, 1, 0)
    print(y_data[0:600, 1:4])

    print('x_data.shape: ', x_data.shape)
    print('y_data.shape: ', y_data.shape)

    #train_main(params, x_data, y_data)

    X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(x_data, y_data, test_size=0.2, shuffle=True,
                                                                        random_state=42)

    X_test = np.array(X_test_org)
    y_test = np.array(y_test_org)
    params[
        'model_weights_path'] = r'C:\Users\Ahmed Guebsi\Desktop\ahmed_files\OUTPUTS\results\saved_models\model-weights-1.hdf5'

    saved_model_file = r"C:\Users\Ahmed Guebsi\Desktop\ahmed_files\OUTPUTS\results\saved_models\model-weights-1.hdf5"
    keras_model = tf.keras.models.load_model(saved_model_file)
    summary = keras_model.summary()

    X_test, y_test = x_data[0:800, :], y_data[0:800, :]
    print(X_test.shape)
    print("type", type(X_test))


    test_model(params, X_test,y_test)
    # Evaluate the model on the test set
    loss, accuracy = keras_model.evaluate(X_test_org, y_test_org)

    print(f'Test loss: {loss:.4f}')
    print(f'Test accuracy: {accuracy:.4f}')
    print(keras_model.predict(X_test_org))
    class_probabilities = keras_model.predict(X_test_org)
    predicted_class = tf.argmax(class_probabilities)
    print(predicted_class[0])
    real = y_test_org.argmax(axis=-1)
    print(real[0])

    acc = np.mean(predicted_class == real)
    print("accuracy", acc)
    print(summary)






