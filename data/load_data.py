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
import time
from vmdpy import VMD

import dsatools
from dsatools import operators
import dsatools.utilits as ut
from dsatools import decomposition


from pandas import DataFrame, set_option
#from mne.io.egi  import write_raw_egi
from mne.filter import notch_filter, filter_data
from itertools import product
from typing import List, Union
from tqdm import tqdm
import EntropyHub as eh
import antropy as an
import dit
from dit.other import tsallis_entropy, renyi_entropy
from pandas import DataFrame, Series, read_pickle, set_option
from features_extraction import log_entropy,log2_entropy,permutation_entropy,cal_permutation_entropy,permut_entropy,fuzzy_entropy,kraskov_entropy,krask_entropy, cal_shannon_entropy,\
    cor_cond_entropy, spectral_entropy, svd_entropy, approximate_entropy, sample_entropy,shannon_entropy, cond_entropy,\
    tsalis_entropy,reyni_entropy, hjorth,hurst, hurst_x,hfd,HFD,higuchi_fractal_dimension,hjorth_mob,hjorth_params,hjorthParameters, spectral_flatness, calculate_renyi_entropy,\
    calculate_tsallis_entropy,calculate_log_energy_entropy,sure_entropy, calculate_prob_distribution, bin_power,calculate_shannon_entropy,\
    psd_welch, lyapunov,outcome_probabalities,conditional_entropy,cal_renyi_entropy,cal_log_energy_entropy,cal_tsallis_entropy,cal_conditional_entropy\


from environment import (ADHD_STR, FREQ, USE_REREF, LOW_PASS_FILTER_RANGE_HZ,
                              NOTCH_FILTER_HZ, SIGNAL_OFFSET,CHANNELS,Chs,
                              channels_good, attention_states, feature_names,
                              get_brainwave_bands, NUM_CHILDREN, SIGNAL_DURATION_SECONDS_DEFAULT,custom_mapping)

from helper_functions import (get_mat_filename, get_mat_file_name, serialize_functions, glimpse_df)
from electrodes_positions import (get_electrodes_positions, get_electrodes_coordinates, set_electrodes_montage)
from signals import SignalPreprocessor
from features_extraction import FeatureExtractor
from autoreject import (get_rejection_threshold, AutoReject)
import nolds as nlds
import neurokit2 as nk
from mne_icalabel import label_components
from mne_icalabel.gui import label_ica_components


epoch_events_num = FREQ
children_num = NUM_CHILDREN
signal_duration = SIGNAL_DURATION_SECONDS_DEFAULT
#channels = CHANNELS
channels = Chs
data_directory = "ADHD_part1"
output_dir = r"C:\Users\Ahmed Guebsi\Desktop\Data_test"
PATH_DATASET_MAT= r"C:\Users\Ahmed Guebsi\Downloads\ADHD_part1"



datapath = r"C:\Users\Ahmed Guebsi\Downloads\data"# data file path
def data_load(index):
    directory = datapath + '\\d'
    x_data, y_data = [], []
    x_data = np.array(x_data)

    for fileIndex in range(index):
        filename = directory + str(fileIndex + 1) + '.mat'
        feature = h5py.File(filename, mode='r')
        a = list(feature.keys())
        x_sub = feature[a[0]]
        x_sub = np.array(x_sub)
        if x_data.size == 0:
            x_data = x_sub
        else:
            x_data = np.concatenate((x_data, x_sub), axis=2)

    feature = h5py.File(datapath + '\\y_stim.mat', mode='r')
    a = list(feature.keys())
    y_data = feature[a[0]]
    y_data = np.array(y_data)

    return x_data, y_data


def reference_data(raw, reference):
    raw.set_eeg_reference(ref_channels=reference, projection=False, ch_type='eeg', verbose=None)
    # set_eeg_reference(ref_channels='average', projection=False, ch_type='auto', forward=None, verbose=None)
    return raw

def downsample(raw, freq=250):
    raw = raw.resample(sfreq=freq)
    return raw, freq

def signal_crop(signal: BaseRaw, freq: float, signal_offset: float, signal_duration_wanted: float):
    signal_total_duration = floor(len(signal) / freq)
    start = signal_total_duration - signal_duration_wanted + signal_offset
    end = signal_total_duration + signal_offset
    return signal.crop(tmin=start, tmax=end)

def signal_filter_notch(signal: BaseRaw, filter_hz):
    return signal.copy().notch_filter(np.arange(filter_hz, (filter_hz * 5) + 1, filter_hz))

def low_high_pass_filter(signal: BaseRaw, l_freq, h_freq):
    return signal.copy().filter(l_freq=l_freq, h_freq=h_freq)

def filt_data(eegData, lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filt_eegData = signal.lfilter(b, a, eegData, axis = 1)
    return filt_eegData


def epochs_to_dataframe(epochs: Epochs, drop_columns=["time", "condition"]):
    df: DataFrame = epochs.to_data_frame(scalings=dict(eeg=1))
    df = df.drop(drop_columns, axis=1)
    return df

def get_column_names(channels, feature_names, preprocess_procedure_names : List[str]):
    prod= product(channels, feature_names, preprocess_procedure_names)
    return list(map(lambda strs:"_".join(strs), prod))

def get_column_name(feature: str, channel: str, suffix: Union[str, None] = None):
    result = "_".join([channel, feature])
    result = result if suffix is None else "_".join([result, suffix])
    return result

def load_mat_data(signal_filepath):
    # Load the MATLAB .mat file

    #mat_data = sio.loadmat(r"C:\Users\Ahmed Guebsi\Downloads\ADHD_Data\ADHD_part1\v1p.mat")
    mat_data = sio.loadmat(signal_filepath)
    #filename= get_mat_file_name(signal_filepath)

    # Extract the EEG signal data from the loaded .mat file
    #eeg_signal = mat_data['eeg']
    print(mat_data.keys())
    # Transpose the signal array if necessary
    #print(type(mat_data['v1p']))
    #print(mat_data['v1p'].shape)
    #print(mat_data['__header__'])

    last_key, last_value = list(mat_data.items())[-1]

    #eeg_signal = np.transpose(mat_data['v1p'])
    #eeg_signal = np.transpose(mat_data[filename])

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

def preprocess_eeg_data(raw,info):
    # Apply notch filter at 50 Hz
    notch_filtered = notch_filter(raw.get_data(), Fs=raw.info['sfreq'], freqs=50)

    # Apply sixth-order Butterworth bandpass filter (0.1 - 60 Hz)
    #bandpass_filtered = filter_data(notch_filtered, sfreq=raw.info['sfreq'], l_freq=0.1, h_freq=60, method='fir')
    bandpass_filtered = filt_data(notch_filtered, lowcut=0.1, highcut=60, fs=raw.info['sfreq'], order=6)

    # Segment the filtered data into 4-second durations
    #duration = 4  # Duration of each segment in seconds
    #segment_length = int(duration * raw.info['sfreq'])  # Convert duration to number of samples
    #segmented_data = [bandpass_filtered[:, i:i + segment_length] for i in range(0, bandpass_filtered.shape[1], segment_length)]

    signal_filtered = mne.io.RawArray(bandpass_filtered, info)
    return signal_filtered

list_removed_components=[]

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

def plot_raw(raw, raw_filtered, raw_ica):
    # Plot the raw data
    result=nk.signal_plot([raw.get_data()[0, 0:500], raw_filtered.get_data()[0, 0:500], raw_ica.get_data()[0, 0:500]],
                   labels=["Raw", "Preprocessed", "ICA"],
                   sampling_rate=raw.info["sfreq"])

    return result

def get_duration():
    duration = []
    nb_epoch_list=[]
    nb_epoch_four=[]
    for child_id, attention_state in tqdm(list(product(range(0, children_num), attention_states))):
        is_adhd = 1 if attention_state == ADHD_STR else 0
        if attention_state == "adhd":
            signal_filepath = str(Path(PATH_DATASET_MAT, get_mat_filename(child_id + 1, attention_state)))
            # signal_filepath = str(Path(PATH_DATASET_MAT, get_mat_file_name(PATH_DATASET_MAT)))
            print(signal_filepath)
            raw, info, ch_names, eeg_signal = load_mat_data(signal_filepath)
            print(raw)
            print(type(raw))
            print(raw.info)
            dur = raw._data.shape[1] / raw.info["sfreq"]
            dur = round((raw._data.shape[1] - 1/ raw.info["sfreq"]),3)

            duration_epoch = dur // 4 * 4
            nb_epoch_list.append(duration_epoch//4)
            duration.append(dur)


        #print(sum(duration))
    return int(sum(duration)), int(sum(nb_epoch_list)), int(sum(nb_epoch_four))

# Feature extraction fucntion
def featExtract(f, Fs, welchWin=1024):
    """
    features,featLabels = featExtract(f,Fs, welchWin = 1024):
        f: input signal
        Fs: sampling frequency, in Hz
        welchWin: window size (in samples) for evaluating Welch's PSD, from which spectral features are calculated
    Returns:
        features: calculated features
        featLabels: Feature labels - ["AM","BM","ent","pow","Cent","pk","freq","skew","kurt","Hmob","Hcomp"]
    """
    # from scipy.ndimage.filters import gaussian_filter

    # AM and BM
    fhilbert = signal.hilbert(f)  # hilbert transform
    fhilbert = fhilbert[150:-150]  # to avoid border effects
    fphase = np.unwrap(np.angle(fhilbert))
    A = abs(fhilbert)  # instantaneous amplitude
    inst_freq = np.diff(fphase) * Fs / (2 * np.pi)  # instantaneous frequency
    E = (np.linalg.norm(fhilbert) ** 2) / len(fhilbert)
    CW = np.sum(np.diff(fphase) * Fs * (A[0:-1] ** 2)) / (2 * np.pi * E)
    AM = np.sqrt(np.sum((np.diff(A) * Fs) ** 2)) / E
    BM = np.sqrt(np.sum(((inst_freq - CW) ** 2) * (A[0:-1] ** 2)) / E)

    # spectral features - Welch
    w, Pxx = signal.welch(f, Fs, nperseg=welchWin, noverlap=round(0.85 * welchWin))
    PxxNorm = Pxx / sum(Pxx)  # normalized spectrum

    Sent = -sum(PxxNorm * np.log2(PxxNorm))  # spectral entropy
    Spow = np.mean(Pxx ** 2)  # spectral power
    Cent = np.sum(w * PxxNorm)  # frequency centroid
    Speak = np.max(Pxx)  # peak amplitude
    Sfreq = w[np.argmax(PxxNorm)]  # peak frequency
    # skewness, kurtosis
    fskew = skew(f)
    fkurt = kurtosis(f)
    # Hjorth Parameters
    dy_f = np.diff(f)
    Hmob = np.sqrt(np.var(dy_f) / np.var(f))
    Hcomp = np.sqrt(np.var(np.diff(dy_f)) / np.var(dy_f)) / Hmob

    features = [AM, BM, Sent, Spow, Cent, Speak, Sfreq, fskew, fkurt, Hmob, Hcomp]
    featLabels = ["AM", "BM", "ent", "pow", "Cent", "pk", "freq", "skew", "kurt", "Hmob", "Hcomp"]

    return features, featLabels
def apply_vmd(signal):

    featsTuple = {"EMD": 0, "EEMD": 0, "CEEMDAN": 0, "EWT": 0, "VMD": 0, "Orig": 0}
    """
    u,u_hat,omega = VMD(f, alpha, tau, K, DC, init, tol)

     Input and Parameters
        ---------------------
        f       - the time domain signal (1D) to be decomposed
        alpha   - the balancing parameter of the data-fidelity constraint
        tau     - time-step of the dual ascent ( pick 0 for noise-slack )
        Nmodes / K       - the number of modes to be recovered
        DC      - true if the first mode is put and kept at DC (0-freq)
        init    - 0 = all omegas start at 0
                           1 = all omegas start uniformly distributed
                          2 = all omegas initialized randomly
        tol     - tolerance of convergence criterion; typically around 1e-6

        Output:
        -------
        u       - the collection of decomposed modes
        u_hat   - spectra of the modes
        omega   - estimated mode center-frequencies
    """
    Fs=128
    # some sample parameters for VMD
    alpha = 2000  # moderate bandwidth constraint
    tau = 0.  # noise-tolerance (no strict fidelity enforcement)
    Nmodes = 5  # 5 modes
    DC = 0  # no DC part imposed
    init = 1  # initialize omegas uniformly
    tol = 1e-3
    #% VMD features
    #DC = np.mean(fp)    no DC part imposed
    tic = time.time()
    vmd,_,_ = VMD(signal, alpha, tau, Nmodes, DC, init, tol)
    toc = time.time()
    featsTuple["VMD"]  = toc-tic #execution time (decomposition )
    if Nmodes != vmd.shape[0]:
        print("\nCheck number of VMD modes")
    #print("VMD decomposition time: ",toc-tic)

    return vmd, featsTuple


def VMD(f, alpha, tau, K, DC, init, tol):
    """
    u,u_hat,omega = VMD(f, alpha, tau, K, DC, init, tol)
    Variational mode decomposition
    Python implementation by Vinícius Rezende Carvalho - vrcarva@gmail.com
    code based on Dominique Zosso's MATLAB code, available at:
    https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
    Original paper:
    Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’,
    IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.


    Input and Parameters:
    ---------------------
    f       - the time domain signal (1D) to be decomposed
    alpha   - the balancing parameter of the data-fidelity constraint
    tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    K       - the number of modes to be recovered
    DC      - true if the first mode is put and kept at DC (0-freq)
    init    - 0 = all omegas start at 0
                       1 = all omegas start uniformly distributed
                      2 = all omegas initialized randomly
    tol     - tolerance of convergence criterion; typically around 1e-6

    Output:
    -------
    u       - the collection of decomposed modes
    u_hat   - spectra of the modes
    omega   - estimated mode center-frequencies
    """
    print("first length of input signal: ", len(f))
    if len(f) % 2:
        f = f[:-1]

    # Period and sampling frequency of input signal
    fs = 1. / len(f)
    #fs=128
    print("sampling frequency: ", fs)
    print("signal length: ", len(f))
    ltemp = len(f) // 2
    fMirr = np.append(np.flip(f[:ltemp], axis=0), f)
    fMirr = np.append(fMirr, np.flip(f[-ltemp:], axis=0))

    # Time Domain 0 to T (of mirrored signal)
    T = len(fMirr)
    t = np.arange(1, T + 1) / T
    print(len(t))

    # Spectral Domain discretization
    freqs = t - 0.5 - (1 / T)

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    Niter = 500
    # For future generalizations: individual alpha for each mode
    Alpha = alpha * np.ones(K)

    # Construct and center f_hat
    f_hat = np.fft.fftshift((np.fft.fft(fMirr)))
    f_hat_plus = np.copy(f_hat)  # copy f_hat
    f_hat_plus[:T // 2] = 0

    # Initialization of omega_k
    omega_plus = np.zeros([Niter, K])

    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5 / K) * (i)
    elif init == 2:
        omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(1, K)))
    else:
        omega_plus[0, :] = 0

    # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0, 0] = 0

    # start with empty dual variables
    lambda_hat = np.zeros([Niter, len(freqs)], dtype=complex)

    # other inits
    uDiff = tol + np.spacing(1)  # update step
    n = 0  # loop counter
    sum_uk = 0  # accumulator
    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = np.zeros([Niter, len(freqs), K], dtype=complex)

    # *** Main loop for iterative updates***

    while (uDiff > tol and n < Niter - 1):  # not converged and below iterations limit
        # update first mode accumulator
        k = 0
        sum_uk = u_hat_plus[n, :, K - 1] + sum_uk - u_hat_plus[n, :, 0]

        # update spectrum of first mode through Wiener filter of residuals
        u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                    1. + Alpha[k] * (freqs - omega_plus[n, k]) ** 2)

        # update first omega if not held at 0
        if not (DC):
            omega_plus[n + 1, k] = np.dot(freqs[T // 2:T], (abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)) / np.sum(
                abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)

        # update of any other mode
        for k in np.arange(1, K):
            # accumulator
            sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]
            # mode spectrum
            u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                        1 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2)
            # center frequencies
            omega_plus[n + 1, k] = np.dot(freqs[T // 2:T], (abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)) / np.sum(
                abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)

        # Dual ascent
        lambda_hat[n + 1, :] = lambda_hat[n, :] + tau * (np.sum(u_hat_plus[n + 1, :, :], axis=1) - f_hat_plus)

        # loop counter
        n = n + 1

        # converged yet?
        uDiff = np.spacing(1)
        for i in range(K):
            uDiff = uDiff + (1 / T) * np.dot((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]),
                                             np.conj((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i])))

        uDiff = np.abs(uDiff)

        # Postprocessing and cleanup

    # discard empty space if converged early
    Niter = np.min([Niter, n])
    omega = omega_plus[:Niter, :]

    idxs = np.flip(np.arange(1, T // 2 + 1), axis=0)
    # Signal reconstruction
    u_hat = np.zeros([T, K], dtype=complex)
    u_hat[T // 2:T, :] = u_hat_plus[Niter - 1, T // 2:T, :]
    u_hat[idxs, :] = np.conj(u_hat_plus[Niter - 1, T // 2:T, :])
    u_hat[0, :] = np.conj(u_hat[-1, :])

    u = np.zeros([K, len(t)])
    print("inside vmd u shape is ", u.shape)
    print("number of samples is ", len(t))
    for k in range(K):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))

    # remove mirror part
    u = u[:, T // 4:3 * T // 4]
    print("inside vmd after mirroring u shape is ", u.shape)

    # recompute spectrum
    u_hat = np.zeros([u.shape[1], K], dtype=complex)
    print("inside vmd u_hat shape is ", u_hat.shape)
    for k in range(K):
        u_hat[:, k] = np.fft.fftshift(np.fft.fft(u[k, :]))

    return u, u_hat, omega

def zero_crossing(X, th=0):
    zcross = 0
    for cont in range(len(X) - 1):
        can = X[cont] * X[cont + 1]
        can2 = abs(X[cont] - X[cont + 1])
        if can < 0 and can2 > th:
            zcross = zcross + 1
    return zcross
# threshold the signal and make it discrete, normalize it and then compute entropy
def shannonEntropy(eegData, bin_min, bin_max, binWidth):

    H = np.zeros((eegData.shape[0], eegData.shape[2]))
    for chan in range(H.shape[0]):
        for epoch in range(H.shape[1]):
            counts, binCenters = np.histogram(eegData[chan,:,epoch], bins=np.arange(bin_min+1, bin_max, binWidth))
            nz = counts > 0
            prob = counts[nz] / np.sum(counts[nz])
            H[chan, epoch] = -np.dot(prob, np.log2(prob/binWidth))
    return H

# Extract the tsallis Entropy
def tsalisEntropy(eegData, bin_min, bin_max, binWidth, orders=[2]):
  H = [np.zeros((eegData.shape[0], eegData.shape[2]))] * len(orders)
  for chan in range(H[0].shape[0]):
    for epoch in range(H[0].shape[1]):
      counts, bins = np.histogram(eegData[chan, :, epoch], bins=np.arange(-200 + 1, 200, 2))
      dist = dit.Distribution([str(bc).zfill(5) for bc in bins[:-1]], counts / sum(counts))
      for ii, order in enumerate(orders):
        H[ii][chan, epoch] = tsallis_entropy(dist, order)
  return H

# Extract the Reyni Entropy
def ReyniEntropy(eegData, bin_min, bin_max, binWidth, orders = [1]):
    H = [np.zeros((eegData.shape[0], eegData.shape[2]))]*len(orders)
    for chan in range(H[0].shape[0]):
        for epoch in range(H[0].shape[1]):
            counts, bins = np.histogram(eegData[chan,:,epoch], bins=np.arange(-200+1, 200, 2))
            dist = dit.Distribution([str(bc).zfill(5) for bc in bins[:-1]],counts/sum(counts))
            for ii,order in enumerate(orders):
                H[ii][chan,epoch] = renyi_entropy(dist,order)
    return H


feature_extractor = FeatureExtractor(selected_feature_names=feature_names)
signal_preprocessor = SignalPreprocessor()


filter_frequencies = {"standard": LOW_PASS_FILTER_RANGE_HZ}


""" preprocessing procedures that will filter frequencies defined in filter_frequencies"""
for freq_name, freq_range in filter_frequencies.items():
    low_freq, high_freq = freq_range

    procedure = serialize_functions(
        lambda s: preprocess_eeg_data(s,info),
        #lambda s:apply_ica(s,info),

        #lambda s, freq=FREQ, signal_offset=SIGNAL_OFFSET, signal_duration=signal_duration: signal_crop(s, freq,signal_offset,signal_duration),
        #lambda s, notch_filter_hz=NOTCH_FILTER_HZ: signal_filter_notch(s, notch_filter_hz),
        #lambda s, low_freq=low_freq, high_freq=high_freq: s.copy().filter(low_freq, high_freq),
        #lambda s: s.copy().apply_hilbert(envelope=True),
    )

    signal_preprocessor.register_preprocess_procedure(freq_name, procedure=procedure, context={"freq_filter_range": freq_range})

plot_channel_data = False
plot_raw_eeg= False
plot_hilbert = False
plot_prob_dist =False

list_inst_amp=[]
list_inst_freq=[]
list_inst_hilbert=[]
renyi_entropy_all_channels_list=[]

shannon_entropy_all_channels_list=[]

tsallis_entropy_all_channels_list=[]

log_energy_entropy_all_channels_list=[]
df_channels =pd.DataFrame(columns=channels)
df_kraskov =pd.DataFrame(columns=channels)

shannon_dataframe= pd.DataFrame(columns=channels)
tsallis_dataframe= pd.DataFrame(columns=channels)
renyi_dataframe= pd.DataFrame(columns=channels)

training_cols = get_column_names(channels, feature_extractor.get_feature_names(), signal_preprocessor.get_preprocess_procedure_names())
df_dict = {k: [] for k in ["is_adhd", "child_id", "epoch_id", *training_cols]}
duration=[]
for child_id, attention_state in tqdm(list(product(range(0, children_num), attention_states))):
    is_adhd = 1 if attention_state == ADHD_STR else 0
    signal_filepath = str(Path(PATH_DATASET_MAT, get_mat_filename(child_id + 1, attention_state)))

    print(signal_filepath)
    raw, info, ch_names, eeg_signal = load_mat_data(signal_filepath)
    print(raw)
    print(type(raw))
    print(raw.info)

    #print(f"Duration of the signal is {dur_tot} seconds")

    electrodes_coordinates = get_electrodes_coordinates(ch_names)
    #print(electrodes_coordinates)
    dig_points = get_electrodes_positions(ch_names, electrodes_coordinates)
    raw_with_dig_pts,info = set_electrodes_montage(ch_names, electrodes_coordinates,eeg_signal)
    #print(type(raw_with_dig_pts))
    #raw_with_dig_pts.plot_sensors(show_names=True)
    signal_preprocessor.fit(raw_with_dig_pts)
    print(type(signal_preprocessor))
    #apply_ica(raw_with_dig_pts, info)
    for proc_index, (signal_processed, proc_name, proc_context) in tqdm(enumerate(signal_preprocessor.get_preprocessed_signals())):
        print(type(signal_processed))
        scan_durn = signal_processed._data.shape[1] / signal_processed.info['sfreq']
        #signal_duration = int(scan_durn)
        signal_duration = round((signal_processed._data.shape[1] - 1) / signal_processed.info["sfreq"], 3) # corrected

        # epoch_events_num = int(signal_duration // 4)
        signal_processed.crop(tmin=0, tmax=signal_duration // 4 * 4)

        # assuming you have a Raw and ICA instance previously fitted
        print(type(signal_processed))

        #ica = ICA(n_components=0.99, random_state=42)
        #ica.fit(signal_processed)
        #ica.plot_components()
        #mne.viz.set_browser_backend("qt", verbose=None)
        #label_components(raw_with_dig_pts, ica, method='iclabel')
        #gui = label_ica_components(raw, ica)
        #print(ica.labels_)

        signal_processed_ica = apply_ica(signal_processed, info,plot_components=False, plot_scores=False,plot_sources=False, plot_overlay=False)

        if plot_raw:
            plot_raw(raw,signal_processed,signal_processed_ica)
        # By default epoch duration = 1

        epochs = make_fixed_length_epochs(signal_processed, duration=4.0, preload=True,verbose=False)
        print(epochs)
        print(len(epochs))  # 47
        #epochs.plot(picks="all", scalings="auto", n_epochs=3, n_channels=19, title="plotting epochs")
        # epochs = mne.Epochs(signal_processed, epochs, tmin=0, tmax=1, baseline=None, detrend=1, preload=True)
        print(epochs.get_data().shape)  # (47,19, 512) we added 2s overlap
        print(type(signal_processed))
        #signal_processed.apply_hilbert(picks="all", envelope=True, n_jobs=-1, n_fft='auto', verbose=None)
        print(signal_processed._data.shape) # (19, 12289)
        # epochs = epochs.apply_baseline(baseline=(None, 0))
        df = epochs_to_dataframe(epochs)

        #print(df.describe())
        print(df.head())
        num_rows = df.shape[0]
        #num_columns = df.shape[1]
        print("Number of rows:", num_rows) # 24064 because of the added overlap
        #print("Number of columns:", num_columns)

        freq_filter_range = proc_context["freq_filter_range"]
        feature_extractor.fit(signal_processed, FREQ)

        #features, featLabels =featExtract(epochs[1].get_data(), FREQ)
        #print(features)
        #print(featLabels)

        for epoch_id, epoch_data in enumerate(epochs):
            print(epoch_data.T.shape) # (512, 19)
            df_epoch_test_mode_1 = pd.DataFrame(epoch_data.T, columns=ch_names)
            df_epoch_test_mode_2 = pd.DataFrame(epoch_data.T, columns=ch_names)
            df_epoch_test_mode_3 = pd.DataFrame(epoch_data.T, columns=ch_names)
            df_epoch_test_mode_4 = pd.DataFrame(epoch_data.T, columns=ch_names)
            df_epoch_test_mode_0 = pd.DataFrame(epoch_data.T, columns=ch_names)

            print(df_epoch_test_mode_0.head())

            for i in range(len(ch_names)):
                #print("channel number is ", i)
                #print("length of epoch data is ", len(epoch_data[i, :]))
                #print(type(epoch_data[i, :]))
                #print(epoch_data[i, :].shape)  # (512,)
                #print(epoch_data.shape) # (19, 512)

                if plot_channel_data:
                    plt.figure()
                    plt.plot(epoch_data[i, :])
                    plt.title(f'EEG data of a single channel {ch_names[i]} of epoch {epoch_id} of child {child_id}')
                    plt.show()

                vmd_ch, featsTuple_ch = apply_vmd(epoch_data[i, :])
                #print("length of vmd_ch is ", len(vmd_ch))  # 5
                #print(type(vmd_ch))
                #print(vmd_ch.shape) # (5, 512)

                for mode in range(4,5):
                    #df_dict["mode"].append(mode)

                    df_epoch_test_vmd = pd.DataFrame(vmd_ch[mode], columns=[f'vmd_{mode}_ch_{ch_names[i]}'])
                    print(df_epoch_test_vmd.shape) # (512, 1)
                    print(df_epoch_test_vmd.head())
                    vmd_ch_test = np.copy(vmd_ch[mode])
                    print("mode number is ", mode)
                    hilbert = signal.hilbert(vmd_ch_test)
                    print(hilbert.shape) # (512,)
                    fphase = np.unwrap(np.angle(hilbert))
                    A = abs(hilbert)  # instantaneous amplitude
                    print(type(A))
                    inst_freq = np.diff(fphase) * 128 / (2 * np.pi)  # instantaneous frequency

                    # Calculate the probability distribution using histogram analysis
                    hist, bins = np.histogram(A, bins='auto', density=True)
                    #print(bins)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    #print("bin centers",bin_centers)

                    # Convert the data to a NumPy array if it's not already
                    time_series_data = np.array(A)

                    # Create a KDE object with the time series data
                    kde = gaussian_kde(time_series_data)

                    # Calculate the probability estimates for individual samples
                    probability_estimates = kde.evaluate(time_series_data)

                    # Normalize the probability estimates so that the sum is equal to one
                    normalized_prob_estimates = probability_estimates / np.sum(probability_estimates)
                    #print("shape kde",normalized_prob_estimates.shape)
                    #print(normalized_prob_estimates)
                    #print(np.sum(normalized_prob_estimates))
                    #print("shape kde",probability_estimates.shape)
                    #print(probability_estimates)
                    #print(np.sum(probability_estimates))
                    print(calculate_renyi_entropy(A))
                    sampling_freq = 128  # Sampling frequency in Hz
                    # time = np.arange(len(fhilbert)) / sampling_freq  # Time array in seconds

                    # Plot the probability distribution
                    #plt.plot(bin_centers, hist)
                    if plot_prob_dist:
                        plt.plot(A, normalized_prob_estimates)
                        plt.xlabel('Instantaneous Amplitude')
                        plt.ylabel('Probability Density')
                        plt.title('Probability Distribution of Instantaneous Amplitude')
                        plt.show()

                    if plot_hilbert:
                        plt.figure()
                        plt.plot(hilbert)
                        # plt.title(f'instantaneous amp num_mode={num_mode}, epoch_id={epoch_id}')
                        plt.title(f'Hilbert num_mode={mode}, epoch_id={epoch_id}')
                        plt.show()

                    #df_epoch_test_hilbert = pd.DataFrame(hilbert, columns=[f'hilbert_{mode}_ch_{ch_names[i]}'])
                    df_epoch_test_hilbert = pd.DataFrame(A, columns=[f'{ch_names[i]}'])
                    df_epoch_test_mode_0[ch_names[i]] = df_epoch_test_hilbert[ch_names[i]]
                    df_epoch_test_hilbert['mode'] = mode
                    df_epoch_test_hilbert['epoch_id'] = epoch_id
                    #print(df_epoch_test_hilbert.shape) # (512, 1)
                    #print(df_epoch_test_hilbert.head())
                    #print(df_epoch_test[ch_names[i]].head())
                    print(df_epoch_test_mode_0.head())

            df_epoch = df.loc[df["epoch"] == epoch_id, channels].head(epoch_events_num)
            #print(df_epoch)
            #print(df_epoch.shape) # (512,19)
            #feature_dict = feature_extractor.get_features(df_epoch, epoch_id=epoch_id,freq_filter_range=freq_filter_range)

            #df_epoch = df_epoch_test_mode_0.loc[df_epoch_test_mode_0["epoch"] == epoch_id, channels]

            df_epoch_test_mode_0.insert(loc=0,column='epoch_id',value=epoch_id)
            print(type(df_epoch_test_mode_0))
            print(df_epoch_test_mode_0.head())

            feature_dict = feature_extractor.get_features(df_epoch_test_mode_0, epoch_id=epoch_id,freq_filter_range=freq_filter_range)
            for channel_idx, channel in enumerate(channels):
                for feature_name, feature_array in feature_dict.items():
                    df_dict[get_column_name(feature_name, channel, proc_name)].append(feature_array[channel_idx])
            if proc_index == 0:
                df_dict["epoch_id"].append(epoch_id)
                df_dict["child_id"].append(child_id)
                df_dict["is_adhd"].append(is_adhd)
                #df_dict["mode"].append(mode)

            """Create dataframe from rows and columns"""
        df = DataFrame.from_dict(df_dict)
        df["is_adhd"] = df["is_adhd"].astype(int)
        df["child_id"] = df["child_id"].astype(int)
        df["epoch_id"] = df["epoch_id"].astype(int)
        #df["mode"] = df["mode"].astype(int)
        glimpse_df(df)

        df.to_pickle(str(Path(output_dir, ".clean_raw_df_adhd_new.pkl")))
        df.to_csv(str(Path(output_dir, ".clean_raw_df_adhd_new.csv")))


if __name__ == "__main__":
    print(get_duration())

    signal_filepath = r"C:\Users\Ahmed Guebsi\Downloads\data\d1.mat"
    feature = h5py.File(signal_filepath, mode='r')
    a = list(feature.keys())
    print(a)
    x_sub = feature[a[0]]
    x_sub = np.array(x_sub)
    print(x_sub.shape)

    x_data, y_data = data_load(7)
    print(x_data.shape)
    print(y_data.shape)
    print(y_data[0])
    print(y_data[1][9900:10000])
    y_data = np.swapaxes(y_data, 1, 0)
    print(y_data[0:600, 1:3])

