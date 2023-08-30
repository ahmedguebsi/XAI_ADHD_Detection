import mne
from load_data import load_mat_data
import os
from pathlib import Path
import scipy.io as sio
import numpy as np
from mne.io.base import BaseRaw
from mne.io import RawArray
from mne import compute_raw_covariance
from mne.minimum_norm import make_inverse_operator, apply_inverse


import pyxdf

PATH_DATASET_MAT= r"C:\Users\Ahmed Guebsi\Downloads\ADHD_part1"

def import_data(fname_raw):

    streams, header = pyxdf.load_xdf(fname_raw)

    data = streams[0]["time_series"].T

    assert data.shape[0] == 19

    info = mne.create_info(19, 128, "eeg")

    raw = RawArray(data, info)

    return raw


def get_mat_file_name(file_path):
    file_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name)[0]
    return file_name_without_extension

def get_mat_filename(i_child: int, state: str):
    return "{i_child}_{state}.mat".format(i_child=i_child, state=state)

def covariance_matrix(raw):

    cov = compute_raw_covariance(raw,scalings='auto')
    # compute_raw_covariance(raw, tmin=0, tmax=None, tstep=0.2, reject=None, flat=None, picks=None, method='empirical',
    #                        method_params=None, cv=3, scalings=None, n_jobs=1, return_estimators=False,
    #                        reject_by_annotation=True, rank=None, verbose=None)

    cov.plot(raw.info, proj=True, block=True)

    return cov

def psd(raw):    #raw.plot_psd(fmin=0, fmax=50, estimate='power', dB=True, average=False, line_alpha=1, spatial_colors=True)  # sphere=(0, 0.015, 0, 0.085) sphere=(0, -0.0075, 0, 0.12)
    spectrum = raw.compute_psd(method="welch")

    fig = spectrum.plot()
    return fig


def inverse_solution(raw, fwd, noise_cov, inv_method):

    snr = 3.0  # use smaller SNR for raw data
    # inv_method = 'sLORETA'  # sLORETA, MNE, dSPM
    parc = 'aparc'  # the parcellation to use, e.g., 'aparc' 'aparc.a2009s'
    loose = dict(surface=0.2, volume=1.)

    lambda2 = 1.0 / snr ** 2

    inverse_operator = make_inverse_operator(
        raw.info, fwd, noise_cov, depth=None, loose=loose, verbose=True)
    del fwd

    stc = apply_inverse(raw, inverse_operator, lambda2, inv_method,
                        pick_ori=None)

    src = inverse_operator['src']

    return stc, src

def bin_power(X, Band, Fs):
    """Compute power in each frequency bin specified by Band from FFT result of
    X. By default, X is a real signal.

    Note
    -----
    A real signal can be synthesized, thus not real.

    Parameters
    -----------

    Band
        list

        boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.
        [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.
        You can also use range() function of Python to generate equal bins and
        pass the generated list to this function.

        Each element of Band is a physical frequency and shall not exceed the
        Nyquist frequency, i.e., half of sampling frequency.

     X
        list

        a 1-D real time series.

    Fs
        integer

        the sampling rate in physical frequency

    Returns
    -------

    Power
        list

        spectral power in each frequency bin.

    Power_ratio
        list

        spectral power in each frequency bin normalized by total power in ALL
        frequency bins.

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
    return Power, Power_Ratio


# accepts PSD of all sensors, returns band power for all sensors
def get_brain_waves_power(psd_welch, freqs):

	brain_waves = OrderedDict({
		"delta" : [1.0, 4.0],
		"theta": [4.0, 7.5],
		"alpha": [7.5, 13.0],
		"lower_beta": [13.0, 16.0],
		"higher_beta": [16.0, 30.0],
		"gamma": [30.0, 40.0]
	})

	# create new variable you want to "fill": n_brain_wave_bands
	band_powers = np.zeros((psd_welch.shape[0], 6))

	for wave_idx, wave in enumerate(brain_waves.keys()):
		# identify freq indices of the wave band
		if wave_idx == 0:
			band_freqs_idx = np.argwhere((freqs <= brain_waves[wave][1]))
		else:
			band_freqs_idx = np.argwhere((freqs >= brain_waves[wave][0]) & (freqs <= brain_waves[wave][1]))

		# extract the psd values for those freq indices
		band_psd = psd_welch[:, band_freqs_idx.ravel()]

		# sum the band psd data to get total band power
		total_band_power = np.sum(band_psd, axis=1)

		# set power in band for all sensors
		band_powers[:, wave_idx] = total_band_power

	return band_powers




def calculate_asymmetry_ch(df_psd_band,left_ch,right_ch):
    """
    Calculate asymmetry between brain hemispheres.

    Parameters
    ----------
    df_psd_band: A dataframe with PSD values (for each region/channel) per subject for one band
    left_ch: A string for the left channel (or region)
    right_ch: A string for the right channel (or region)

    Returns
    -------
    df_asymmetry: A dataframe for calculated asymmetry for all the subjects
    """
    df_asymmetry = (df_psd_band[left_ch] - df_psd_band[right_ch])/(df_psd_band[left_ch] + df_psd_band[right_ch])*100

    return df_asymmetry

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    #print(b,a)
    y = lfilter(b, a, data)
    return y
def calc_bands_power(x, dt, bands):
    from scipy.signal import welch
    f, psd = welch(x, fs=1. / dt)
    power = {band: np.mean(psd[np.where((f >= lf) & (f <= hf))]) for band, (lf, hf) in bands.items()}
    return power
for i in np.arange(n):
    alpha1 = butter_bandpass_filter(fft1[i, :], 8.1, 12.0, 256)
    beta1 = butter_bandpass_filter(fft1[i, :], 16.0, 36.0, 256)
    gamma1 = butter_bandpass_filter(fft1[i, :], 36.1, 80, 256)
    delta1 = butter_bandpass_filter(fft1[i, :], 0.0, 4.0, 256)
    sigma1 = butter_bandpass_filter(fft1[i, :], 12.1, 16.0, 256)
    theta1 = butter_bandpass_filter(fft1[i, :], 4.1, 8.0, 256)
    sumalpha1 = sum(abs(alpha1))
    sumbeta1 = sum(abs(beta1))
    sumgamma1 = sum(abs(gamma1))
    sumdelta1 = sum(abs(delta1))
    sumsigma1 = sum(abs(sigma1))
    sumtheta1 = sum(abs(theta1))
    objects = [sumalpha1, sumbeta1, sumgamma1, sumdelta1, sumsigma1, sumtheta1]
    N = len(objects)
    ra = range(N)
    plt.title(signal_labels[i])
    plt.autoscale
    somestuffneeded = np.arange(6)
    ticks = ['alpha','beta','gamma','delta','sigma','theta']
    plt.xticks(somestuffneeded, ticks)
    plt.bar(ra, objects)
    plt.show()
if __name__ == "__main__" :
    signal_filepath = str(Path(PATH_DATASET_MAT, get_mat_filename( 1, "normal")))
    print(signal_filepath)
    #raw, info, ch_names, eeg_signal =load_mat_data(signal_filepath)
    #psd(raw)