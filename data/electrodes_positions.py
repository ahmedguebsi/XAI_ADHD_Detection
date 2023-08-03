import pandas as pd
import numpy as np
import antropy as an
import EntropyHub as eh
import dit
import mne
import time
from vmdpy import VMD
import scipy.io as sio
from helper_functions import get_mat_file_name
from mne.channels import compute_native_head_t, read_custom_montage, make_standard_montage
import matplotlib.pyplot as plt
from mne.epochs import make_fixed_length_epochs
from dit.other import tsallis_entropy, renyi_entropy
from scipy import stats, signal, integrate
from mne.epochs import make_fixed_length_epochs
from mne.io.base import BaseRaw
from mne import Epochs
from mne.preprocessing import ICA
from mne.filter import notch_filter, filter_data

from autoreject import (get_rejection_threshold, AutoReject)

from pyPTE import get_phase, get_binsize,get_discretized_phase, get_bincount, get_delay, \
    compute_PTE, compute_dPTE_rawPTE, PTE_from_dataframe, PTE_from_mne,PTE
import seaborn as sns
import scipy.stats
import mne
import neurokit2 as nk

import networkx as nx

#plot connectivity

from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_sensors_connectivity
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.viz import plot_topomap


from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_stat_map, plot_anat, plot_img, show

from spectral_connectivity import Multitaper, Connectivity


print(__doc__)

#from load_data import preprocess_eeg_data, apply_vmd, apply_ica

# helper function to remove 1st dimension
def convert_2d(in_array):

    rows = in_array.shape[0]    # vertical
    cols = in_array.shape[1]    # horizontal

    out_array = np.zeros((rows, cols))    # create new array to hold image data

    for r in range(rows):
        for c in range(cols):
            #out_array[r, c] = in_array[:, r, c]
            out_array[r, c] = in_array[r, c,:]
    return out_array
def convert_3d_to_2d(array_3d):
    # Get the shape of the 3D array
    shape_3d = array_3d.shape

    # Remove the second dimension by reshaping the array
    #array_2d = np.reshape(array_3d, (shape_3d[0], shape_3d[2]))

    # Compute the average along the second dimension
    array_2d = np.mean(array_3d, axis=1)
    return array_2d
def load_data(signal_filepath):
  # Load the MATLAB .mat file

  # mat_data = sio.loadmat(r"C:\Users\Ahmed Guebsi\Downloads\ADHD_Data\ADHD_part1\v1p.mat")
  mat_data = sio.loadmat(signal_filepath)
  filename = get_mat_file_name(signal_filepath)

  # Extract the EEG signal data from the loaded .mat file
  # eeg_signal = mat_data['eeg']
  print(mat_data.keys())

  last_key, last_value = list(mat_data.items())[-1]

  eeg_signal = np.transpose(last_value)

  # Create the info dictionary
  n_channels, n_samples = eeg_signal.shape
  #ch_names = [f'Channel {i}' for i in range(1, n_channels + 1)]
  ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8','01', '02']
  # ch_types = ['eeg'] * n_channels
  info = mne.create_info(ch_names=ch_names, sfreq=128)

  # Create the RawArray object
  raw = mne.io.RawArray(eeg_signal, info)

  return raw, info, ch_names, eeg_signal


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


def name_channels(raw, ch_names):
  mapping = {}
  for i in range(0, raw.info.get('nchan'), 1):
    mapping[str(i)] = ch_names[i]
  raw.rename_channels(mapping, allow_duplicates=False, verbose=None)

  return raw

def cor_cond_entropy(x):
    return eh.CondEn(x, tau=1, c=6, Logx=np.exp(1), Norm=False)

def permutation_entropy(x):
    return an.perm_entropy(x, order=3, normalize=True)

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
def tsalisEntropy(eegData, bin_min, bin_max, binWidth, orders=[1]):
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


# Extract the Conditional Entropy
def CondEntropy(eegData,raw, bin_min, bin_max, binWidth):
  H = [np.zeros((eegData.shape[0], eegData.shape[2]))]
  for chan in range(H[0].shape[0]):
    for epoch in range(H[0].shape[1]):
      counts, bins = np.histogram(eegData[chan, :, epoch], bins=np.arange(-200 + 1, 200, 2))
      dist = dit.Distribution([str(bc).zfill(5) for bc in bins[:-1]], counts / sum(counts))
      adhd_labels = np.array([0, 1])
      # Define the random variable representing the ADHD labels
      rv_Y = dit.ScalarDistribution(adhd_labels)

      # Define the random variables representing the EEG channels
      #rvs_X = [dit.ScalarDistribution(data) for data in epochs.get_data()]  # Each epoch represents a channel
      rvs_X = [dit.ScalarDistribution(data) for data in raw.T]
      H[chan, epoch] = dit.shannon.conditional_entropy(dist, rvs_X=rvs_X, rvs_Y=rv_Y)

  return H


def hjorth(a):
    r"""
    Compute Hjorth parameters [HJO70]_.
    .. math::
        Activity = m_0 = \sigma_{a}^2
    .. math::
        Complexity = m_2 = \sigma_{d}/ \sigma_{a}
    .. math::
        Morbidity = m_4 =  \frac{\sigma_{dd}/ \sigma_{d}}{m_2}
    Where:
    :math:`\sigma_{x}^2` is the mean power of a signal :math:`x`. That is, its variance, if it's mean is zero.
    :math:`a`, :math:`d` and :math:`dd` represent the original signal, its first and second derivatives, respectively.
    .. note::
        **Difference with PyEEG:**
        Results is different from [PYEEG]_ which appear to uses a non normalised (by the length of the signal) definition of the activity:
        .. math::
            \sigma_{a}^2 = \sum{\mathbf{x}[i]^2}
        As opposed to
        .. math::
            \sigma_{a}^2 = \frac{1}{n}\sum{\mathbf{x}[i]^2}
    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :return: activity, complexity and morbidity
    :rtype: tuple(float, float, float)

    """

    first_deriv = np.diff(a)
    second_deriv = np.diff(a,2)

    var_zero = np.mean(a ** 2)
    var_d1 = np.mean(first_deriv ** 2)
    var_d2 = np.mean(second_deriv ** 2)

    activity = var_zero
    morbidity = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / morbidity

    return activity, morbidity, complexity

def compute_hjorth_mobility(data):
    """Hjorth mobility (per channel).
    Hjorth mobility parameter computed in the time domain.
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **hjorth_mobility**. See [1]_.
    References
    ----------
    .. [1] Paivinen, N. et al. (2005). Epileptic seizure detection: A
           nonlinear viewpoint. Computer methods and programs in biomedicine,
           79(2), 151-159.
    """
    x = np.insert(data, 0, 0, axis=-1)
    dx = np.diff(x, axis=-1)
    sx = np.std(x, ddof=1, axis=-1)
    sdx = np.std(dx, ddof=1, axis=-1)
    mobility = np.divide(sdx, sx)
    return mobility

def compute_hjorth_complexity(data):
    """Hjorth complexity (per channel).
    Hjorth complexity parameter computed in the time domain.
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **hjorth_complexity**. See [1]_.
    References
    ----------
    .. [1] Paivinen, N. et al. (2005). Epileptic seizure detection: A
           nonlinear viewpoint. Computer methods and programs in biomedicine,
           79(2), 151-159.
    """
    x = np.insert(data, 0, 0, axis=-1)
    dx = np.diff(x, axis=-1)
    m_dx = compute_hjorth_mobility(dx)
    m_x = compute_hjorth_mobility(data)
    complexity = np.divide(m_dx, m_x)
    return complexity


def hjorth_params(x, axis=-1):
    """Calculate Hjorth mobility and complexity on given axis.
    .. versionadded: 0.1.3
    Parameters
    ----------
    x : list or np.array
        1D or N-D data.
    axis : int
        The axis along which to perform the computation. Default is -1 (last).
    Returns
    -------
    mobility, complexity : float
        Mobility and complexity parameters.
    Notes
    -----
    Hjorth Parameters are indicators of statistical properties used in signal
    processing in the time domain introduced by Bo Hjorth in 1970. The
    parameters are activity, mobility, and complexity. EntroPy only returns the
    mobility and complexity parameters, since activity is simply the variance
    of :math:`x`, which can be computed easily with :py:func:`numpy.var`.
    The **mobility** parameter represents the mean frequency or the proportion
    of standard deviation of the power spectrum. This is defined as the square
    root of variance of the first derivative of :math:`x` divided by the
    variance of :math:`x`.
    The **complexity** gives an estimate of the bandwidth of the signal, which
    indicates the similarity of the shape of the signal to a pure sine wave
    (where the value converges to 1). Complexity is defined as the ratio of
    the mobility of the first derivative of :math:`x` to the mobility of
    :math:`x`.
    References
    ----------
    - https://en.wikipedia.org/wiki/Hjorth_parameters
    - https://doi.org/10.1016%2F0013-4694%2870%2990143-4
    Examples
    --------
    Hjorth parameters of a pure sine
    >>> import numpy as np
    >>> import entropy as ent
    >>> sf, f, dur = 100, 1, 4
    >>> N = sf * dur # Total number of discrete samples
    >>> t = np.arange(N) / sf # Time vector
    >>> x = np.sin(2 * np.pi * f * t)
    >>> np.round(ent.hjorth_params(x), 4)
    array([0.0627, 1.005 ])
    Random 2D data
    >>> np.random.seed(42)
    >>> x = np.random.normal(size=(4, 3000))
    >>> mob, com = ent.hjorth_params(x)
    >>> print(mob)
    [1.42145064 1.4339572  1.42186993 1.40587512]
    >>> print(com)
    [1.21877527 1.21092261 1.217278   1.22623163]
    Fractional Gaussian noise with H = 0.5
    >>> import stochastic.processes.noise as sn
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
    >>> np.round(ent.hjorth_params(x), 4)
    array([1.4073, 1.2283])
    Fractional Gaussian noise with H = 0.9
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.9, rng=rng).sample(10000)
    >>> np.round(ent.hjorth_params(x), 4)
    array([0.8395, 1.9143])
    Fractional Gaussian noise with H = 0.1
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.1, rng=rng).sample(10000)
    >>> np.round(ent.hjorth_params(x), 4)
    array([1.6917, 1.0717])
    """
    x = np.asarray(x)
    # Calculate derivatives
    dx = np.diff(x, axis=axis)
    ddx = np.diff(dx, axis=axis)
    # Calculate variance
    x_var = np.var(x, axis=axis)  # = activity
    dx_var = np.var(dx, axis=axis)
    ddx_var = np.var(ddx, axis=axis)
    # Mobility and complexity
    mob = np.sqrt(dx_var / x_var)
    com = np.sqrt(ddx_var / dx_var) / mob
    return mob, com


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
    tol = 1e-7
    #% VMD features
    #DC = np.mean(fp)    no DC part imposed
    tic = time.time()
    vmd,_,_ = VMD(signal, alpha, tau, Nmodes, DC, init, tol)
    toc = time.time()
    featsTuple["VMD"]  = toc-tic #execution time (decomposition )
    if Nmodes != vmd.shape[0]:
        print("\nCheck number of VMD modes")
    #print("VMD decomposition time: ",toc-tic)
    return vmd


##########
# false nearest neighbor descriptor
def falseNearestNeighbor(eegData, fast=True):
    # Average Mutual Information
    # There exist good arguments that if the time delayed mutual
    # information exhibits a marked minimum at a certain value of tex2html_wrap_inline6553,
    # then this is a good candidate for a reasonable time delay.
    npts = 1000  # not sure about this?
    maxdims = 50
    max_delay = 2  # max_delay = 200  # TODO: need to use 200, but also need to speed this up
    distance_thresh = 0.5

    out = np.zeros((eegData.shape[0], eegData.shape[2]))
    for chan in range(eegData.shape[0]):
        for epoch in range(eegData.shape[2]):
            if fast:
                out[chan, epoch] = 0
            else:
                cur_eegData = eegData[chan, :, epoch]
                lagidx = 0  # we are looking for the index of the lag that makes the signal maximally uncorrelated to the original
                # # minNMI = 1  # normed_mutual_info is from 1 (perfectly correlated) to 0 (not at all correlated)
                # # for lag in range(1, max_delay):
                # #     x = cur_eegData[:-lag]
                # #     xlag = cur_eegData[lag:]
                # #     # convert float data into histogram bins
                # #     nbins = int(np.floor(1 + np.log2(len(x)) + 0.5))
                # #     x_discrete = np.histogram(x, bins=nbins)[0]
                # #     xlag_discrete = np.histogram(xlag, bins=nbins)[0]
                # #     cNMI = normed_mutual_info(x_discrete, xlag_discrete)
                # #     if cNMI < minNMI:
                # #         minNMI = cNMI
                # #         lagidx = lag
                # nearest neighbors part
                knn = int(max(2, 6 * lagidx))  # heuristic (number of nearest neighbors to look up)
                m = 1  # lagidx + 1

                # y is the embedded version of the signal
                y = np.zeros((maxdims + 1, npts))
                for d in range(maxdims + 1):
                    tmp = cur_eegData[d * m:d * m + npts]
                    y[d, :tmp.shape[0]] = tmp

                nnd = np.ones((npts, maxdims))
                nnz = np.zeros((npts, maxdims))

                # see where it tends to settle
                for d in range(1, maxdims):
                    for k in range(0, npts):
                        # get the distances to all points in the window (distance given embedding dimension)
                        dists = []
                        for nextpt in range(1, knn + 1):
                            if k + nextpt < npts:
                                dists.append(np.linalg.norm(y[:d, k] - y[:d, k + nextpt]))
                        if len(dists) > 0:
                            minIdx = np.argmin(dists)
                            if dists[minIdx] == 0:
                                dists[minIdx] = 0.0000001  # essentially 0 just silence the error
                            nnd[k, d - 1] = dists[minIdx]
                            nnz[k, d - 1] = np.abs(y[d + 1, k] - y[d + 1, minIdx + 1 + k])
                # aggregate results
                mindim = np.mean(nnz / nnd > distance_thresh, axis=0) < 0.1
                # get the index of the first occurence of the value true
                # (a 1 in the binary representation of true and false)
                out[chan, epoch] = np.argmax(mindim)

    return out

# Coherence in the Delta Band
def CoherenceDelta(eegData, i, j, fs=100):
    nfft=eegData.shape[1]
    f, Cxy = signal.coherence(eegData[i,:,:], eegData[j,:,:], fs=fs, nfft=nfft, axis=0)#, window=np.hanning(nfft))
    out = np.mean(Cxy[np.all([f >= 0.5, f<=4], axis=0)], axis=0)
    return out

##########
# Filter the eegData, midpass filter
#	eegData: 3D np array [chans x ms x epochs]
def filt_data(eegData, lowcut, highcut, fs, order=7):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filt_eegData = signal.lfilter(b, a, eegData, axis = 1)
    return filt_eegData

def preprocess_eeg_data(raw,info):
    # Apply notch filter at 50 Hz
    notch_filtered = notch_filter(raw.get_data(), Fs=raw.info['sfreq'], freqs=50)
    print("hhhhhhhhhhhhhhhhhhhhhhhhhh",notch_filtered.shape)

    # Apply sixth-order Butterworth bandpass filter (0.1 - 60 Hz)
    #bandpass_filtered = filter_data(notch_filtered, sfreq=raw.info['sfreq'], l_freq=0.1, h_freq=60, method='fir')
    bandpass_filtered= filt_data(notch_filtered, lowcut=0.1, highcut=60, fs=raw.info['sfreq'], order=6)

    # Segment the filtered data into 4-second durations
    #duration = 4  # Duration of each segment in seconds
    #segment_length = int(duration * raw.info['sfreq'])  # Convert duration to number of samples
    #segmented_data = [bandpass_filtered[:, i:i + segment_length] for i in range(0, bandpass_filtered.shape[1], segment_length)]

    signal_filtered = mne.io.RawArray(bandpass_filtered, info)
    return signal_filtered

def apply_ica(raw, info, plot_components=False, plot_scores=False,plot_sources=False, plot_overlay=False):
    #raw.plot_sensors(show_names=True)
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



# Replace these coordinates with the actual coordinates of your EEG electrodes
# Coordinates should be in 3D (x, y, z) space
electrode_coordinates = {
    'EEG1': [0, 0, 0],
    'EEG2': [1, 0, 0],
    'EEG3': [0, 1, 0],
    # Add more electrodes here...
}

# Function to calculate the Euclidean distance between two electrodes
def calculate_distance(coord1, coord2):
    return np.sqrt(sum((coord1[i] - coord2[i]) ** 2 for i in range(3)))

# Create an empty graph
G = nx.Graph()

# Add the electrodes as nodes to the graph
G.add_nodes_from(electrode_coordinates.keys())

# Calculate the pairwise distances and form the adjacency matrix with weights
adjacency_matrix = np.zeros((len(electrode_coordinates), len(electrode_coordinates)))
for i, (electrode1, coord1) in enumerate(electrode_coordinates.items()):
    for j, (electrode2, coord2) in enumerate(electrode_coordinates.items()):
        if i != j:  # Avoid self-loops
            distance = calculate_distance(coord1, coord2)
            G.add_edge(electrode1, electrode2, weight=1/distance)
            adjacency_matrix[i, j] = 1/distance

# Draw the graph
pos = {electrode: coord[:2] for electrode, coord in electrode_coordinates.items()}
labels = {electrode: electrode for electrode in G.nodes()}
weights = nx.get_edge_attributes(G, 'weight')

plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, labels=labels, node_size=1000, font_size=10, font_weight='bold')
nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_size=8)
plt.title('Weighted and Undirected EEG Electrodes Graph')
plt.show()

print("Adjacency Matrix:")
print(adjacency_matrix)

def plot_eeg_raw(raw ,ch_names):
    fig, ax = plt.subplots(figsize=[15, 5])
    n_channels = len(ch_names)
    n_samples = raw.n_times

    time = np.arange(n_samples) / raw.info['sfreq']
    start_time = 20
    end_time = 21
    print(time.shape)
    segment_time = time[start_time:end_time + 1]
    print(raw.__dict__)
    print(raw.info)
    raw.info['dig']
    for i in range(n_channels):
        ax.plot(raw.get_data(picks='all', tmin=start_time, tmax=end_time).T, label=f'ch_names[i+1]')
        # plt.plot(time, eeg_signal[i, :], label=f'Channel {i+1}')
    plt.show()

def plot_all(raw, raw_filtered, reconstructed_raw):
    nk.signal_plot([raw.get_data()[0, 0:50], raw_filtered.get_data()[0, 0:50], reconstructed_raw.get_data()[0, 0:50]],
                   labels=["Raw", "Preprocessed", "ICA"],
                   sampling_rate=raw.info["sfreq"])
    plt.show()


if __name__=="__main__":

  raw, info, ch_names, eeg_signal = load_data(r"C:\Users\Ahmed Guebsi\Downloads\ADHD_Data\ADHD_part1\v1p.mat")
  print("eeg siiiiiignal",type(eeg_signal))
  plot_eeg_raw(raw, ch_names)
  print(raw._data.shape)
  raw_test_cond=np.transpose(raw.get_data())
  electrodes_coordinates = get_electrodes_coordinates(ch_names)
  # print(electrodes_coordinates)
  dig_points = get_electrodes_positions(ch_names, electrodes_coordinates)
  raw_with_dig_pts, info = set_electrodes_montage(ch_names, electrodes_coordinates, eeg_signal)

  raw_filtered= preprocess_eeg_data(raw_with_dig_pts,info)

  plot_eeg_raw(raw_filtered,ch_names)

  reconstructed_raw = apply_ica(raw_filtered, info,False,False,False,False)

  plot_eeg_raw(reconstructed_raw,ch_names)

  plot_all(raw, raw_filtered, reconstructed_raw)

  epochs = make_fixed_length_epochs(reconstructed_raw, duration=4.0, overlap=2.0, verbose=False)
  epochs.load_data().pick_types(eeg=True)

  dPTE, rPTE = PTE_from_mne(reconstructed_raw)

  print(rPTE.max(), rPTE.min())
  sns.heatmap(rPTE)

  plt.show()

  # Compute connectivity for band containing the evoked response.
  # We exclude the baseline period:
  fmin, fmax = 4., 9.
  sfreq = raw.info['sfreq']  # the sampling frequency
  tmin = 0.0  # exclude the baseline period
  con = spectral_connectivity_epochs(
      epochs, method='pli', mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
      faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

  print(con.shape)
  # Now, visualize the connectivity in 3D:
  plot_sensors_connectivity(
      epochs.info,
      #rPTE,
      con.get_data(output='dense')[:, :, 0]
      )




  # print(type(raw_with_dig_pts))
  # raw_with_dig_pts.plot_sensors(show_names=True)
  # plt.show()

  epochs = make_fixed_length_epochs(reconstructed_raw, duration=4.0, overlap=2.0, verbose=False)

  raw_auto = raw.copy()
  epochs_auto = mne.make_fixed_length_epochs(raw_auto, duration=4, preload=True)
  print(epochs_auto.get_data().shape)
  print("shaaaaaaape",epochs.get_data().shape)
  for epoch_id, epoch_data in enumerate(epochs):
      print(epoch_data.shape)

      print(get_phase(epoch_data).shape)
      phase = get_phase(epoch_data)
      delay = get_delay(phase)
      print("delay", delay)
      binsize = get_binsize(phase)
      print(binsize)
      print(get_bincount(binsize))

      d_phase = get_discretized_phase(phase, binsize)
      print(type(d_phase))
      print(d_phase.shape)

  epochs.load_data()
  print(len(epochs))
  print(len(epochs[0].times))
  scan_durn = raw._data.shape[1] / raw.info['sfreq']
  # Create the 3D array
  eeg_data = np.zeros((len(raw.ch_names), 512, len(epochs)))
  # print(eeg_data)

  # Fill the array with epoch data
  for i, epoch in enumerate(epochs):
      eeg_data[:, :, i] = epoch
  # raw_with_dig_pts.plot_sensors(show_names=True)
  # plt.show()
  eegData_list = []
  eegData_list.extend([len(raw.ch_names), scan_durn, raw.n_times])
  eegData = np.array((len(raw.ch_names), scan_durn, scan_durn // 4))
  print(eeg_data)
  print(eeg_data.shape[0], eeg_data.shape[2])
  ShannonRes = shannonEntropy(eeg_data, bin_min=-200, bin_max=200, binWidth=2)
  print(ShannonRes.shape)
  TsallisRes = tsalisEntropy(eeg_data, bin_min=-200, bin_max=200, binWidth=2, orders=[1])
  print(TsallisRes[0].shape)
  ReyRes = ReyniEntropy(eeg_data, bin_min=-200, bin_max=200, binWidth=2, orders=[1])
  print(ReyRes[0].shape)
  print(type(eeg_data))
  eeg_data_converted = convert_3d_to_2d(eeg_data)
  # CondRes = CondEntropy(eeg_data, raw_test_cond,bin_min=-200, bin_max=200, binWidth=2)
  # print("cond ",CondRes)
  activity, morbidity, complexity = hjorth(eeg_data)
  print(morbidity)
  print(complexity)
  mob = compute_hjorth_mobility(eeg_data)
  print(mob)
  com = compute_hjorth_complexity(eeg_data)
  print(com.shape)
  mobi, con = hjorth_params(eeg_data)
  print(mobi.shape)
  print(con.shape)
  print(epochs[0].get_data().shape)

  for epoch_data in epochs:
      # print(type(epoch_data))
      vmd = apply_vmd(epoch_data)
  print(len(vmd))
  print(vmd[0].shape)
  print(epochs[0].get_data().shape)
  vmd_test = np.copy(vmd[0])
  CCENRes = cor_cond_entropy(vmd_test)
  print(CCENRes)
  print(CoherenceDelta(eeg_data, 0, 1, fs=128))
  # PermRes = permutation_entropy(eeg_data)
  # print(PermRes.shape)






