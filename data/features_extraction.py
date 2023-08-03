"""
Class which extracts features for a given dataframe's epoch

PE - special entropy - calculated by applying the Shannon function to the normalized power spectrum based on the peaks of a Fourier transform
AE - Approximate entropy - calculated in time domain without phase-space reconstruction of signal (short-length time series data)
SE - Sample entropy - similar to AE. Se is less sensitive to changes in data length with larger values corresponding to greater complexity or irregularity in the data
FE - Fuzzy entropy - stable results for different parameters. Best noise resistance using fuzzy membership function.

"""
from math import ceil, sqrt, log, floor, gamma
from typing import Dict, List, Optional, Tuple, TypedDict
from collections.abc import Iterable
from scipy import stats, signal, integrate
import antropy as an
import EntropyHub as eh
from pyentrp import entropy
from scipy.stats import entropy as scipy_entropy
import numpy as np
from pandas import DataFrame, Series
from scipy.signal.spectral import periodogram
from scipy.stats import gaussian_kde # for kernel density estimation
from mne.time_frequency import psd_array_welch, psd_array_multitaper
from environment import FREQ
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, norm
from scipy.signal import find_peaks
from scipy.integrate import simps
import dit
#import pyeeg
from dit.other import tsallis_entropy, renyi_entropy
from scipy.stats.mstats import gmean

def petrosian_fractal_dimension(x):
    #return eh.PetrosianFD(x)
    return an.petrosian_fd(x)
def katz_fractal_dimension(x):
    #return eh.KatzFD(x)
    return an.katz_fd(x, axis=-1)
def higuchi_fractal_dimension(x):
    #return eh.HiguchiFD(x)
    return an.higuchi_fd(x, kmax=10)


def bin_power(X, Band=[0.5, 4, 8, 12, 30], Fs=128):
    """Compute power in each frequency bin specified by Band from FFT result of
    X. By default, X is a real signal.
    """
    C = np.fft.fft(X)
    C = abs(C)
    Power = np.zeros(len(Band) - 1)
    for Freq_Index in range(0, len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])
        Power[Freq_Index] = sum(
            C[int(np.floor(Freq / Fs * len(X))):
                int(np.floor(Next_Freq / Fs * len(X)))]
        )
    Power_Ratio = Power / sum(Power)
    theta_beta_ratio = Power[1] / Power[3]
    return Power, Power_Ratio, theta_beta_ratio




def conditional_entropy(x):
    return 0
def cond_entropy(x):
    source = dit.Distribution(x[:-1])  # Exclude the last sample for conditioning
    target = dit.Distribution(x[1:])  # Exclude the first sample for conditioning
    return dit.shannon.conditional_entropy(target, source)

def calculate_prob_distribution(signal):
    unique_values, counts = np.unique(signal, return_counts=True)
    prob_distribution = counts / len(signal)
    return unique_values, prob_distribution


def sure_entropy(x):
    #unique_values, prob_distribution = calculate_prob_distribution(signal)

    # Calculate SURE entropy using the log, probability, and sum functions
    #sure_entropy = -np.sum(prob_distribution * np.log2(prob_distribution))
    threshold = 3 #5
    unique_values, prob_distribution = calculate_prob_distribution(x)
    above_threshold = x[x > threshold]

    sure_entropy = np.sum(min(xi ** 2,threshold ** 2 ) for xi in x)
    print(sure_entropy)
    #below_threshold_values=[value for value in x if abs(value) < threshold]
    below_threshold_values = []
    for index , value in enumerate(x):
        if abs(value) < threshold:
            sure_entropy -= index
            below_threshold_values.append(value)
    print(below_threshold_values)

    return sure_entropy

def svd_entropy(x):
    return an.svd_entropy(x, order=3, delay=1, normalize=True)
def multiscale_entropy(x):
    return entropy.multiscale_entropy(x, )


############################################################################################################
def fuzzy_entropy(x):
    return eh.FuzzEn(x, m=2, r=(np.std(x, ddof=0) * 0.2, 1))[0][-1]


def sample_entropy(x):
    return an.sample_entropy(x)


# don't normalize because you have to normalze across all drivers and not based on 1 driver and 1 sample
def spectral_entropy(x, freq: float):
    axis = -1
    sf = freq
    normalize = False

    x = np.asarray(x)
    _, psd = periodogram(x, sf, axis=axis)
    psd_norm = psd[1:] / psd[1:].sum(axis=axis, keepdims=True)
    se = -(psd_norm * np.log2(psd_norm)).sum(axis=axis)
    if normalize:
        se /= np.log2(psd_norm.shape[axis])
    return se


def approximate_entropy(x):
    return an.app_entropy(x, order=2)

############################################################################################################

def approx_entropy(x):
    return eh.ApEn(x, m=2, r=(np.std(x, ddof=0) * 0.2, 1))[0][-1]

def reyni_entropy(x):
    # Discretize the EEG signal into probability distribution
    counts, bins = np.histogram(x, bins=np.arange(-200 + 1, 200, 2), density=True)
    #probabilities = hist / np.sum(hist)

    # Create the distribution object
    #dist = dit.Distribution('custom', outcomes=range(num_bins), probabilities=probabilities)
    #d = dit.Distribution.from_ndarray(pmf, bins)
    dist = dit.Distribution([str(bc).zfill(5) for bc in bins[:-1]], counts / sum(counts))

    # Calculate the Renyi entropy
    #alpha = 2  # Renyi entropy parameter
    #renyi_entropy = dit.shannon.renyi_entropy(dist, alpha)
    return renyi_entropy(dist,order=2)


def regularized_gaussian_kde(data, bw_method=None, reg_factor=1e-6):
    # Calculate the covariance matrix of the data
    #cov_matrix = np.cov(data, rowvar=False)

    # Regularize the covariance matrix by adding a small factor to its diagonal
    #cov_matrix_regularized = cov_matrix + reg_factor * np.eye(cov_matrix.shape[0])

    # Calculate the regularized Gaussian KDE
    #kde = gaussian_kde(data, bw_method=bw_method, cov=cov_matrix_regularized)

    # Calculate the variance of the data
    variance = np.var(data)

    # Regularize the variance by adding a small factor
    variance_regularized = variance + reg_factor

    # Calculate the regularized Gaussian KDE
    kde = gaussian_kde(data, bw_method=bw_method)
    kde.covariance_factor = lambda: variance_regularized
    kde._compute_covariance()

    return kde

def calculate_renyi_entropy(x, order=2):
    if order <= 0:
        raise ValueError("The order should be positive.")

    if len(x) == 0:
        raise ValueError("The signal is empty.")

        # Create a probability distribution from the signal
    unique_values, counts = np.unique(x, return_counts=True)
    probabilities = counts / len(x)

#################################################### prob estimation ##############################"
    # Convert the data to a NumPy array if it's not already
    time_series_data = np.array(x)

    # Create a KDE object with the time series data
    #kde = gaussian_kde(time_series_data)
    kde =regularized_gaussian_kde(time_series_data)

    # Calculate the probability estimates for individual samples
    probability_estimates = kde.evaluate(time_series_data)

    # Normalize the probability estimates so that the sum is equal to one
    normalized_prob_estimates = probability_estimates / np.sum(probability_estimates)

    ren_entropy = np.sum(normalized_prob_estimates ** order)
    ren_entropy = np.log(ren_entropy) / (1 - order)
####################################################################################################

    # Calculate the Rényi entropy
    #ren_entropy = np.sum(probabilities ** order)
    #ren_entropy = np.log(ren_entropy) / (1 - order)
    return ren_entropy

def tsalis_entropy(x):
    counts, bins = np.histogram(x, bins=np.arange(-200 + 1, 200, 2))
    dist = dit.Distribution([str(bc).zfill(5) for bc in bins[:-1]], counts / sum(counts))
    return tsallis_entropy(dist, order=2)


def calculate_tsallis_entropy(x, q=2):
    """
    Calculate the Tsallis entropy of an EEG signal using natural logarithm (ln) and summation functions.

    Args:
        signal (numpy.ndarray): The EEG signal.
        q (float): The order of the Tsallis entropy.

    Returns:
        float: The calculated Tsallis entropy.

    Raises:
        ValueError: If q is less than or equal to 0 or the signal is empty.
    """
    if q <= 0:
        raise ValueError("q should be greater than 0.")

    if len(x) == 0:
        raise ValueError("The signal is empty.")

    # Create a probability distribution from the signal
    unique_values, counts = np.unique(x, return_counts=True)
    probabilities = counts / len(x)

    #################################################### prob estimation ##############################"
    # Convert the data to a NumPy array if it's not already
    time_series_data = np.array(x)

    # Create a KDE object with the time series data
    kde = gaussian_kde(time_series_data)

    # Calculate the probability estimates for individual samples
    probability_estimates = kde.evaluate(time_series_data)

    # Normalize the probability estimates so that the sum is equal to one
    normalized_prob_estimates = probability_estimates / np.sum(probability_estimates)

    tsallis_entropy = (1 - np.sum(normalized_prob_estimates ** q)) / (q - 1)

    ######################################################################################################
    # Calculate the Tsallis entropy , resulted in negative tsallis entropy
    #tsallis_entropy = np.log(probabilities ** q)
    #tsallis_entropy = np.sum(probabilities * tsallis_entropy) / (q - 1)

    # Calculate the Tsallis entropy , worked correctly
    #tsallis_entropy = (1 - np.sum(probabilities ** q)) / (q - 1)
    return tsallis_entropy


def calculate_log_energy_entropy(x):
    """
    Calculate the Log Energy Entropy (LEEn) of an EEG signal using logarithm and summation functions.

    Args:
        signal (numpy.ndarray): The EEG signal.

    Returns:
        float: The calculated Log Energy Entropy.

    Raises:
        ValueError: If the signal is empty.
    """
    if len(x) == 0:
        raise ValueError("The signal is empty.")

    # Create a probability distribution from the signal
    unique_values, counts = np.unique(x, return_counts=True)
    probabilities = counts / len(x)

    # Calculate the power spectrum of the signal
    power_spectrum = np.abs(np.fft.fft(x)) ** 2

    # Normalize the power spectrum
    normalized_power_spectrum = power_spectrum / np.sum(power_spectrum)

    #################################################### prob estimation ##############################"
    # Convert the data to a NumPy array if it's not already
    time_series_data = np.array(x)

    # Create a KDE object with the time series data
    kde = gaussian_kde(time_series_data)

    # Calculate the probability estimates for individual samples
    probability_estimates = kde.evaluate(time_series_data)

    # Normalize the probability estimates so that the sum is equal to one
    normalized_prob_estimates = probability_estimates / np.sum(probability_estimates)
    #print("log",np.log2(normalized_prob_estimates))

    log_energy_entropy = -np.sum((np.log2(normalized_prob_estimates)) ** 2)
    ######################################################################################################
    # Calculate the log energy entropy
    #log_energy_entropy = -np.sum(normalized_power_spectrum * np.log2(normalized_power_spectrum))
    #log_energy_entropy = -np.sum((np.log(probabilities))**2) worked correctly

    return log_energy_entropy

def calculate_shannon_entropy(x):
    """
    Calculate the Shannon entropy of an EEG signal using logarithm, probability, and summation functions.

    Args:
        signal (numpy.ndarray): The EEG signal.

    Returns:
        float: The calculated Shannon entropy.

    Raises:
        ValueError: If the signal is empty.
    """
    if len(x) == 0:
        raise ValueError("The signal is empty.")

    # Create a probability distribution from the signal
    unique_values, counts = np.unique(x, return_counts=True)
    probabilities = counts / len(x)

    #################################################### prob estimation ##############################"
    # Convert the data to a NumPy array if it's not already
    time_series_data = np.array(x)

    # Create a KDE object with the time series data
    kde = gaussian_kde(time_series_data)

    # Calculate the probability estimates for individual samples
    probability_estimates = kde.evaluate(time_series_data)

    # Normalize the probability estimates so that the sum is equal to one
    normalized_prob_estimates = probability_estimates / np.sum(probability_estimates)

    shannon_entropy = -np.sum(normalized_prob_estimates * np.log2(normalized_prob_estimates))
    ######################################################################################################
    # Calculate the Shannon entropy
    #shannon_entropy = -np.sum(probabilities * np.log2(probabilities))

    return shannon_entropy

def kraskov_entropy(x):
    return eh.K2En(x, m=2, tau=1, r=0.2)[0][0]
def permutation_entropy(x):
    return eh.PermEn(x)[0][-1]
def cor_cond_entropy(x):
    return eh.CondEn(x, tau=1, c=6, Logx=np.exp(1), Norm=False)[0][-1]
def permutation_entroy(x):
    return eh.PermEn(x)
def permut_entropy(x):
    return an.perm_entropy(x, order=3, normalize=True)

def log_entropy(x):
    return scipy_entropy(x, base=2)
def log2_entropy(x):
    # Calculate Shannon entropy
    shannon_entropy = entropy.shannon_entropy(x)

    # Calculate Log entropy
    log_entropy = -np.log2(shannon_entropy)
    return log_entropy

def shannon_entropy(x):
    return entropy.shannon_entropy(x)

def wiener_entropy(x): #Spectral flatness (Wiener entropy)
    return 0
def spectral_flatness(x,freq):
    # Calculate the power spectrum of the EEG signal
    power_spectrum = np.abs(np.fft.fft(x)) ** 2

    # Calculate the geometric mean and arithmetic mean of the power spectrum
    geometric_mean = gmean(power_spectrum)
    arithmetic_mean = np.mean(power_spectrum)

    # Calculate the spectral flatness
    #spectral_flatness = 10 * np.log10(geometric_mean / arithmetic_mean)

    spectral_flatness: np.ndarray=(geometric_mean / arithmetic_mean)
    return spectral_flatness

def psd_welch(x: Series, fs=FREQ):
    _, psd = signal.welch(x, fs=fs)
    return psd
##########
# Hjorth Mobility
# Hjorth Complexity
# variance = mean(signal^2) iff mean(signal)=0
# which it is be because I normalized the signal
# Assuming signals have mean 0
# Mobility = sqrt( mean(dx^2) / mean(x^2) )
def hjorthParameters(xV):
    dxV = np.diff(xV, axis=1)
    ddxV = np.diff(dxV, axis=1)

    mx2 = np.mean(np.square(xV), axis=1)
    mdx2 = np.mean(np.square(dxV), axis=1)
    mddx2 = np.mean(np.square(ddxV), axis=1)

    mob = mdx2 / mx2
    complexity = np.sqrt((mddx2 / mdx2) / mob)
    mobility = np.sqrt(mob)

    # PLEASE NOTE that Mohammad did NOT ACTUALLY use hjorth complexity,
    # in the matlab code for hjorth complexity subtraction by mob not division was used
    return mobility, complexity



def compute_mean(data):
    """Mean of the data (per channel).
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **mean**
    """
    return np.mean(data, axis=-1)


def compute_variance(data):
    """Variance of the data (per channel).
    Parameters
    ----------
    data : shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **variance**
    """
    return np.var(data, axis=-1, ddof=1)


def compute_std(data):
    """Standard deviation of the data.
    Parameters
    ----------
    data : shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels)
    Notes
    -----
    Alias of the feature function: **std**
    """
    return np.std(data, axis=-1, ddof=1)

def compute_rms(data):
    """Root-mean squared value of the data (per channel).
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: *rms*
    """
    return np.sqrt(np.mean(np.power(data, 2), axis=-1))

def compute_skewness(data):
    """Skewness of the data (per channel).
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **skewness**
    """
    ndim = data.ndim
    return stats.skew(data, axis=ndim - 1)


def compute_kurtosis(data):
    """Kurtosis of the data (per channel).
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **kurtosis**
    """
    ndim = data.ndim
    return stats.kurtosis(data, axis=ndim - 1, fisher=False)

def compute_quantile(data, q=0.75):
    """Quantile of the data (per channel).
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    q : float or list
        Quantile or sequence of quantiles to compute, which must be between 0
        and 1 inclusive.
    Returns
    -------
    output : ndarray, shape (n_channels * len(q),)
    Notes
    -----
    Alias of the feature function: *quantile*
    """
    return np.ravel(np.quantile(data, q, axis=-1), order='F')

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

# calculate band power
def bandPower(eegData, lowcut, highcut, fs):
	eegData_band = filt_data(eegData, lowcut, highcut, fs, order=7)
	freqs, powers = signal.periodogram(eegData_band, fs, axis=1)
	bandPwr = np.mean(powers,axis=1)
	return bandPwr

##########
# α/δ Ratio
def eegRatio(eegData,fs):
	# alpha (8–12 Hz)
	eegData_alpha = filt_data(eegData, 8, 12, fs)
	# delta (0.5–4 Hz)
	eegData_delta = filt_data(eegData, 0.5, 4, fs)
	# calculate the power
	powers_alpha = bandPower(eegData, 8, 12, fs)
	powers_delta = bandPower(eegData, 0.5, 4, fs)
	ratio_res = np.sum(powers_alpha,axis=0) / np.sum(powers_delta,axis=0)
	return np.expand_dims(ratio_res, axis=0)

def power_spectrum(sfreq, data, fmin=0., fmax=256., psd_method='welch',
                   welch_n_fft=256, welch_n_per_seg=None, welch_n_overlap=0,
                   verbose=False):
    """Power Spectral Density (PSD).
    Utility function to compute the (one-sided) Power Spectral Density which
    acts as a wrapper for :func:`mne.time_frequency.psd_array_welch` (if
    ``method='welch'``) or :func:`mne.time_frequency.psd_array_multitaper`
    (if ``method='multitaper'``). The multitaper method, although more
    computationally intensive than Welch's method or FFT, should be prefered
    for 'short' windows. Welch's method is more suitable for 'long' windows.
    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.
    data : ndarray, shape (..., n_times).
    fmin : float (default: 0.)
        Lower bound of the frequency range to consider.
    fmax : float (default: 256.)
        Upper bound of the frequency range to consider.
    psd_method : str (default: 'welch')
        Method used to estimate the PSD from the data. The valid values for
        the parameter ``method`` are: ``'welch'``, ``'fft'`` or
        ``'multitaper'``.
    welch_n_fft : int (default: 256)
        The length of the FFT used. The segments will be zero-padded if
        `welch_n_fft > welch_n_per_seg`. This parameter will be ignored if
        `method = 'fft'` or `method = 'multitaper'`.
    welch_n_per_seg : int or None (default: None)
        Length of each Welch segment (windowed with a Hamming window). If
        None, `welch_n_per_seg` is equal to `welch_n_fft`. This parameter
        will be ignored if `method = 'fft'` or `method = 'multitaper'`.
    welch_n_overlap : int (default: 0)
        The number of points of overlap between segments. Should be
        `<= welch_n_per_seg`. This parameter will be ignored if
        `method = 'fft'` or `method = 'multitaper'`.
    verbose : bool (default: False)
        Verbosity parameter. If True, info and warnings related to
        :func:`mne.time_frequency.psd_array_welch` or
        :func:`mne.time_frequency.psd_array_multitaper` are printed.
    Returns
    -------
    psd : ndarray, shape (..., n_freqs)
        Estimated PSD.
    freqs : ndarray, shape (n_freqs,)
        Array of frequency bins.
    """
    _verbose = 40 * (1 - int(verbose))
    _fmin, _fmax = max(0, fmin), min(fmax, sfreq / 2)
    if psd_method == 'welch':
        _n_fft = min(data.shape[-1], welch_n_fft)
        return psd_array_welch(data, sfreq, fmin=_fmin, fmax=_fmax,
                               n_fft=_n_fft, verbose=_verbose,
                               n_per_seg=welch_n_per_seg,
                               n_overlap=welch_n_overlap)
    elif psd_method == 'multitaper':
        return psd_array_multitaper(data, sfreq, fmin=_fmin, fmax=_fmax,
                                    verbose=_verbose)
    elif psd_method == 'fft':
        n_times = data.shape[-1]
        m = np.mean(data, axis=-1)
        _data = data - m[..., None]
        spect = np.fft.rfft(_data, n_times)
        mag = np.abs(spect)
        freqs = np.fft.rfftfreq(n_times, 1. / sfreq)
        psd = np.power(mag, 2) / (n_times ** 2)
        psd *= 2.
        psd[..., 0] /= 2.
        if n_times % 2 == 0:
            psd[..., -1] /= 2.
        mask = np.logical_and(freqs >= _fmin, freqs <= _fmax)
        return psd[..., mask], freqs[mask]
    else:
        raise ValueError('The given method (%s) is not implemented. Valid '
                         'methods for the computation of the PSD are: '
                         '`welch`, `fft` or `multitaper`.' % str(psd_method))

################################################### Mne-features #####################################################################
def _psd_params_checker(params):
    """Utility function to check parameters to be passed to `power_spectrum`.
    Parameters
    ----------
    params : dict or None
        Optional parameters to be passed to
        :func:`mne_features.utils.power_spectrum`. If `params` contains a key
        which is not an optional parameter of
        :func:`mne_features.utils.power_spectrum`, an error is raised.
    Returns
    -------
    valid_params : dict
    """
    if params is None:
        return dict()
    elif not isinstance(params, dict):
        raise ValueError('The parameter `psd_params` has type %s. Expected '
                         'dict instead.' % type(params))
    else:
        expected_keys = ['welch_n_fft', 'welch_n_per_seg', 'welch_n_overlap']
        valid_keys = list()
        for n in params:
            if n not in expected_keys:
                raise ValueError('The key %s in `psd_params` is not valid and '
                                 'will be ignored. Valid keys are: %s' %
                                 (n, str(expected_keys)))
            else:
                valid_keys.append(n)
        valid_params = {n: params[n] for n in valid_keys}
        return valid_params

def compute_hjorth_mobility_spect(sfreq, data, normalize=False,
                                  psd_method='welch', psd_params=None):
    """Hjorth mobility (per channel).
    Hjorth mobility parameter computed from the Power Spectrum of the data.
    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.
    data : ndarray, shape (n_channels, n_times)
    normalize : bool (default: False)
        Normalize the result by the total power.
    psd_method : str (default: 'welch')
        Method used for the estimation of the Power Spectral Density (PSD).
        Valid methods are: ``'welch'``, ``'multitaper'`` or ``'fft'``.
    psd_params : dict or None (default: None)
        If not None, dict with optional parameters (`welch_n_fft`,
        `welch_n_per_seg`, `welch_n_overlap`) to be passed to
        :func:`mne_features.utils.power_spectrum`. If None, default parameters
        are used (see doc for :func:`mne_features.utils.power_spectrum`).
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **hjorth_mobility_spect**. See [1]_ and
    [2]_.
    References
    ----------
    .. [1] Mormann, F. et al. (2006). Seizure prediction: the long and
           winding road. Brain, 130(2), 314-333.
    .. [2] Teixeira, C. A. et al. (2011). EPILAB: A software package for
           studies on the prediction of epileptic seizures. Journal of
           Neuroscience Methods, 200(2), 257-271.
    """
    _psd_params = _psd_params_checker(psd_params)
    psd, freqs = power_spectrum(sfreq, data, psd_method=psd_method,
                                **_psd_params)
    w_freqs = np.power(freqs, 2)
    mobility = np.sum(np.multiply(psd, w_freqs), axis=-1)
    if normalize:
        mobility = np.divide(mobility, np.sum(psd, axis=-1))
    return mobility


def compute_hjorth_complexity_spect(sfreq, data, normalize=False,
                                    psd_method='welch', psd_params=None):
    """Hjorth complexity (per channel).
    Hjorth complexity parameter computed from the Power Spectrum of the data.
    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.
    data : ndarray, shape (n_channels, n_times)
    normalize : bool (default: False)
        Normalize the result by the total power.
    psd_method : str (default: 'welch')
        Method used for the estimation of the Power Spectral Density (PSD).
        Valid methods are: ``'welch'``, ``'multitaper'`` or ``'fft'``.
    psd_params : dict or None (default: None)
        If not None, dict with optional parameters (`welch_n_fft`,
        `welch_n_per_seg`, `welch_n_overlap`) to be passed to
        :func:`mne_features.utils.power_spectrum`. If None, default parameters
        are used (see doc for :func:`mne_features.utils.power_spectrum`).
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **hjorth_complexity_spect**. See [1]_ and
    [2]_.
    References
    ----------
    .. [1] Mormann, F. et al. (2006). Seizure prediction: the long and
           winding road. Brain, 130(2), 314-333.
    .. [2] Teixeira, C. A. et al. (2011). EPILAB: A software package for
           studies on the prediction of epileptic seizures. Journal of
           Neuroscience Methods, 200(2), 257-271.
    """
    _psd_params = _psd_params_checker(psd_params)
    psd, freqs = power_spectrum(sfreq, data, psd_method=psd_method,
                                **_psd_params)
    w_freqs = np.power(freqs, 4)
    complexity = np.sum(np.multiply(psd, w_freqs), axis=-1)
    if normalize:
        complexity = np.divide(complexity, np.sum(psd, axis=-1))
    return complexity


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
################################################### Pyrem #####################################################################

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
    result=[]
    first_deriv = np.diff(a)
    second_deriv = np.diff(a,2)

    var_zero = np.mean(a ** 2)
    var_d1 = np.mean(first_deriv ** 2)
    var_d2 = np.mean(second_deriv ** 2)

    activity = var_zero
    morbidity = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / morbidity
    result.append(activity)
    result.append(morbidity)
    result.append(complexity)

    return activity, morbidity, complexity
    #return result
############################## pyeeg #########################################
def hurst(X):
    """ Compute the Hurst exponent of X. If the output H=0.5,the behavior
    of the time-series is similar to random walk. If H<0.5, the time-series
    cover less "distance" than a random walk, vice verse.

    Parameters
    ----------

    X

        list

        a time series

    Returns
    -------
    H

        float

        Hurst exponent

    Notes
    --------
    Author of this function is Xin Liu

    Examples
    --------

    >>> import pyeeg
    >>> from numpy.random import randn
    >>> a = randn(4096)
    >>> pyeeg.hurst(a)
    0.5057444

    """
    X = np.array(X)
    N = X.size
    T = np.arange(1, N + 1)
    Y = np.cumsum(X)
    Ave_T = Y / T

    S_T = np.zeros(N)
    R_T = np.zeros(N)

    for i in range(N):
        S_T[i] = np.std(X[:i + 1])
        X_T = Y - T * Ave_T[i]
        R_T[i] = np.ptp(X_T[:i + 1])

    R_S = R_T / S_T
    R_S = np.log(R_S)[1:]
    n = np.log(T)[1:]
    A = np.column_stack((n, np.ones(n.size)))
    [m, c] = np.linalg.lstsq(A, R_S)[0]
    H = m
    return H



def hjorth_mob(X, D=None):
    """ Compute Hjorth mobility and complexity of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, a first order differential sequence of X (if D is provided,
           recommended to speed up)

    In case 1, D is computed using Numpy's Difference function.

    Notes
    -----
    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.

    Parameters
    ----------

    X
        list

        a time series

    D
        list

        first order differential sequence of a time series

    Returns
    -------

    As indicated in return line

    Hjorth mobility and complexity

    """

    if D is None:
        D = np.diff(X)
        D = D.tolist()

    D.insert(0, X[0])  # pad the first difference
    D = np.array(D)

    n = len(X)

    M2 = float(sum(D ** 2)) / n
    TP = sum(np.array(X) ** 2)
    M4 = 0
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n
    mobility = np.sqrt(M2 / TP)
    complexity = np.sqrt(float(M4) * TP / M2 / M2)
    return mobility, complexity

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

def largest_lyapunov_exponent(x):
    n= 2 # embedding dimension
    tau = 0  # time delay / embedding lag
    fs =128
    T = np.mean(x)  # mean of the signal
    return pyeeg.LLE(x, n, tau, fs, T)
def information_based_similarity(x,y,n):

    # return pyeeg.IBS(x,y,n)
    return pyeeg.information_based_similarity(x,y,n)

########## EEGExract ########################################
# Lyapunov exponent
def lyapunov(eegData):
    return np.mean(np.log(np.abs(np.gradient(eegData,axis=1))),axis=1)




##########################################################" pyrem ###############################################################

def hurst_x(signal):
    """
    **Experimental**/untested implementation taken from:
    http://drtomstarke.com/index.php/calculation-of-the-hurst-exponent-to-test-for-trend-and-mean-reversion/

    Use at your own risks.
    """
    tau = []; lagvec = []

    #  Step through the different lags
    for lag in range(2,20):

    #  produce price difference with lag
        pp = np.subtract(signal[lag:],signal[:-lag])

    #  Write the different lags into a vector
        lagvec.append(lag)

    #  Calculate the variance of the difference vector
        tau.append(np.std(pp))

    #  linear fit to double-log graph (gives power)
    m = np.polyfit(np.log10(lagvec),np.log10(tau),1)

    # calculate hurst
    hurst = m[0]

    return hurst
def pfd(a):
    r"""
    Compute Petrosian Fractal Dimension of a time series [PET95]_.


    It is defined by:

    .. math::

        \frac{log(N)}{log(N) + log(\frac{N}{N+0.4N_{\delta}})}

    .. note::
        **Difference with PyEEG:**

        Results is different from [PYEEG]_ which implemented an apparently erroneous formulae:

        .. math::

            \frac{log(N)}{log(N) + log(\frac{N}{N}+0.4N_{\delta})}



    Where:

    :math:`N` is the length of the time series, and

    :math:`N_{\delta}` is the number of sign changes.


    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :return: the Petrosian Fractal Dimension; a scalar.
    :rtype: float

    """

    diff = np.diff(a)
    # x[i] * x[i-1] for i in t0 -> tmax
    prod = diff[1:-1] * diff[0:-2]

    # Number of sign changes in derivative of the signal
    N_delta = np.sum(prod < 0)
    n = len(a)

    return np.log(n)/(np.log(n)+np.log(n/(n+0.4*N_delta)))
def hfd(a, k_max):

    r"""
    Compute Higuchi Fractal Dimension of a time series.
    Vectorised version of the eponymous [PYEEG]_ function.

    .. note::

        **Difference with PyEEG:**

        Results is different from [PYEEG]_ which appears to have implemented an erroneous formulae.
        [HIG88]_ defines the normalisation factor as:

        .. math::

            \frac{N-1}{[\frac{N-m}{k} ]\dot{} k}

        [PYEEG]_ implementation uses:

        .. math::

            \frac{N-1}{[\frac{N-m}{k}]}

        The latter does *not* give the expected fractal dimension of approximately `1.50` for brownian motion (see example bellow).



    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :param k_max: the maximal value of k
    :type k_max: int

    :return: Higuchi's fractal dimension; a scalar
    :rtype: float

    """

    L = []
    x = []
    N = a.size


    # TODO this could be used to pregenerate k and m idxs ... but memory pblem?
    # km_idxs = np.triu_indices(k_max - 1)
    # km_idxs = k_max - np.flipud(np.column_stack(km_idxs)) -1
    # km_idxs[:,1] -= 1
    #

    for k in range(1,k_max):
        Lk = 0
        for m in range(0,k):
            #we pregenerate all idxs
            idxs = np.arange(1,int(np.floor((N-m)/k)),dtype=np.int32)

            Lmk = np.sum(np.abs(a[m+idxs*k] - a[m+k*(idxs-1)]))
            Lmk = (Lmk*(N - 1)/(((N - m)/ k)* k)) / k
            Lk += Lmk


        L.append(np.log(Lk/(m+1)))
        x.append([np.log(1.0/ k), 1])

    (p, r1, r2, s)=np.linalg.lstsq(x, L)
    return p[0]
def HFD(X, Kmax):
    """ Compute Higuchi Fractal Dimension of a time series X. kmax
     is an HFD parameter
    """
    L = []
    x = []
    N = len(X)
    for k in range(1, Kmax):
        Lk = []
        for m in range(0, k):
            Lmk = 0
            for i in range(1, int(np.floor((N - m) / k))):
                Lmk += abs(X[m + i * k] - X[m + i * k - k])
            Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k
            Lk.append(Lmk)
        L.append(np.log(np.mean(Lk)))
        x.append([np.log(float(1) / k), 1])

    (p, _, _, _) = np.linalg.lstsq(x, L)
    return p[0]



class FeatureContext(TypedDict):
    freq_signal: Optional[float]
    freq_filter_range: Optional[Tuple[float, float]]
    epoch_id: Optional[int]


class FeatureExtractorFeatureContextError(Exception):
    pass


class FeatureExtractorFeatureInvalidArgument(Exception):
    pass


class FeatureExtractor:
    def __init__(self, selected_feature_names: List[str]):
        self._set_mappers()
        self.signal = None
        self.freq = None
        filtered_features = filter(lambda pair: pair[0] in selected_feature_names, self._name_to_function_mapper.items())
        self.selected_features_functions = list(map(lambda pair: pair[1], filtered_features))

    def _set_mappers(self):
        self._name_to_function_mapper = {
            # 4 features used in the paper




            #"CEN":self.feature_cond_entropy,
            "SUE": self.feature_sure_entropy,

            "delta": self.feature_delta_power,
            "theta": self.feature_theta_power,
            "alpha": self.feature_alpha_power,
            "beta": self.feature_beta_power,
            "alpha_ratio": self.feature_alpha_power_ratio,
            "beta_ratio": self.feature_beta_power_ratio,
            "delta_ratio": self.feature_delta_power_ratio,
            "theta_ratio": self.feature_theta_power_ratio,
            "theta_beta_ratio": self.feature_theta_beta_power_ratio,



        }
        self._feature_function_to_mapper_mapper = {v: k for k, v in self._name_to_function_mapper.items()}

    def _validate_feature_context(self, key, context: FeatureContext):
        if key not in context:
            raise FeatureExtractorFeatureContextError("Missing key '{}' in the context.".format(key))

    def fit(self, signal, freq=None):
        self.signal = signal
        self.freq = freq


    def feature_sure_entropy(self, df: DataFrame, **kwargs):
        return df.apply(func=lambda x: sure_entropy(x.to_numpy()), axis=0)
    def feature_delta_power(self, df: DataFrame, **kwargs):
        return df.apply(func=lambda x: bin_power(x.to_numpy())[0][0], axis=0)
    def feature_theta_power(self, df: DataFrame, **kwargs):
        return df.apply(func=lambda x: bin_power(x.to_numpy())[0][1], axis=0)
    def feature_alpha_power(self, df: DataFrame, **kwargs):
        return df.apply(func=lambda x: bin_power(x.to_numpy())[0][2], axis=0)
    def feature_beta_power(self, df: DataFrame, **kwargs):
        return df.apply(func=lambda x: bin_power(x.to_numpy())[0][3], axis=0)

    def feature_delta_power_ratio(self, df: DataFrame, **kwargs):
        return df.apply(func=lambda x: bin_power(x.to_numpy())[1][0], axis=0)
    def feature_theta_power_ratio(self, df: DataFrame, **kwargs):
        return df.apply(func=lambda x: bin_power(x.to_numpy())[1][1], axis=0)
    def feature_alpha_power_ratio(self, df: DataFrame, **kwargs):
        return df.apply(func=lambda x: bin_power(x.to_numpy())[1][2], axis=0)
    def feature_beta_power_ratio(self, df: DataFrame, **kwargs):
        return df.apply(func=lambda x: bin_power(x.to_numpy())[1][3], axis=0)

    def feature_theta_beta_power_ratio(self, df: DataFrame, **kwargs):
        return df.apply(func=lambda x: bin_power(x.to_numpy())[2], axis=0)


    def get_features(self, df: DataFrame, **kwargs: FeatureContext) -> Dict:
        features = {}
        for feature_function in self.selected_features_functions:
            feature_name = self.function_to_name(feature_function)
            features[feature_name] = feature_function(df, **kwargs)
        return features

    def name_to_function(self, feature_name):
        return self._name_to_function_mapper[feature_name]

    def function_to_name(self, feature_function):
        return self._feature_function_to_mapper_mapper[feature_function]

    def get_feature_names(self):
        return list(map(lambda feature_function: self.function_to_name(feature_function), self.selected_features_functions))


if __name__ == "__main__":
    pass