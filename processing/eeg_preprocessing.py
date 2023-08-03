from mne.io.base import BaseRaw
import mne
from autoreject import (get_rejection_threshold, AutoReject)
import matplotlib.pyplot as plt

def filter_raw_data(raw, filter_design, line_remove=None, eog_channels=None,
                    plot_filt=False, savefig=False, verbose=True):
    """
    Apply FIR bandpass filter and remove EOG noise.

    Parameters
    ----------
    raw: Raw-type (MNE-Python) EEG file
    filter_design: A dictionary of all the filter parameters (see MNE raw.filter or create_filter functions)
    line_remove (optional): A boolean whether to remove power-line noise (50Hz) with a Notch filter or not
    eog_channels (optional): A boolean whether to remove EOG noise or not, requires list of EOG channels
    plot_filter (optional): A boolean whether to plot the band-pass filter
    savefig (optional): A boolean whether to save the filter design

    Returns
    -------
    filt: Raw-type (MNE-Python) EEG file
    """

    if verbose == True: print('---\nAPPLYING FILTER\n')
    filt = raw.copy().load_data().filter(**filter_design, verbose=verbose)

    if plot_filt == True:
        filter_params = mne.filter.create_filter(raw.get_data(), raw.info['sfreq'], **filter_design)

        freq_ideal = [0, filter_design['l_freq'], filter_design['l_freq'],
                      filter_design['h_freq'], filter_design['h_freq'], raw.info['sfreq'] / 2]
        gain_ideal = [0, 0, 1, 1, 0, 0]

        fig, axs = plt.subplots(nrows=3, figsize=(8, 8), layout='tight', dpi=100)
        mne.viz.misc.plot_filter(filter_params, raw.info['sfreq'], freq=freq_ideal, gain=gain_ideal,
                                 fscale='log', flim=(0.01, 80), dlim=(0, 6), axes=axs, show=False)
        if savefig == True:
            plt.savefig(fname='Data/filter_design.png', dpi=300)
        plt.show()

    if line_remove != None:
        if verbose == True: print('---\nAPPLYING NOTCH FILTER\n')
        filt = filt.notch_filter([line_remove])

    if eog_channels != None or eog_channels != False:
        if verbose == True: print('---\nAPPLYING SSP FOR EOG-REMOVAL\n')
        eog_projs, _ = mne.preprocessing.compute_proj_eog(filt, n_grad=0, n_mag=0, n_eeg=1, reject=None,
                                                          no_proj=True, ch_name=eog_channels, verbose=verbose)
        filt.add_proj(eog_projs, remove_existing=True)
        filt.apply_proj()
        filt.drop_channels(eog_channels)

    return filt


def artefact_rejection(filt, subjectname, epo_duration=5, verbose=True):
    """
    Convert Raw file to Epochs and conduct artefact rejection/augmentation on the signal.

    Parameters
    ----------
    filt: Raw-type (MNE-Python) EEG file
    subjectname: A string for subject's name
    epo_duration (optional): An integer for the duration for epochs

    Returns
    -------
    epochs: Epochs-type (MNE-Python) EEG file
    """
    if verbose == True: print('---\nDIVIDING INTO EPOCHS\n')
    epochs = mne.make_fixed_length_epochs(filt, duration=epo_duration, preload=True, verbose=verbose)

    if verbose == True: print('---\nEPOCHS BEFORE AR\n')
    epochs.average().plot()
    epochs.plot_image(title="GFP without AR ({})".format(subjectname))

    if verbose == True: print('---\nAPPLYING GLOBAL AR\n')
    reject_criteria = get_rejection_threshold(epochs)
    print('Dropping epochs with rejection threshold:', reject_criteria)
    epochs.drop_bad(reject=reject_criteria, verbose=verbose)

    if verbose == True: print('---\nAPPLYING LOCAL AR\n')
    ar = AutoReject(thresh_method='random_search', random_state=1)
    ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    reject_log.plot('horizontal')

    if verbose == True: print('---\nEPOCHS AFTER AR\n')
    epochs_ar.average().plot()
    epochs_ar.plot_image(title="GFP with AR ({})".format(subjectname))

    return epochs_ar
def load_clean_cnt(filename: str, channels: List[str]):
    cnt = read_raw_cnt(filename, preload=True, verbose=False)
    print("Bad channels found by MNE:", cnt.info["bads"])
    cnt.interpolate_bads()
    cnt.pick_channels(channels)
    return cnt


def signal_crop(signal: BaseRaw, freq: float, signal_offset: float, signal_duration_wanted: float):
    signal_total_duration = floor(len(signal) / freq)
    start = signal_total_duration - signal_duration_wanted + signal_offset
    end = signal_total_duration + signal_offset
    return signal.crop(tmin=start, tmax=end)


def signal_filter_notch(signal: BaseRaw, filter_hz):
    return signal.copy().notch_filter(np.arange(filter_hz, (filter_hz * 5) + 1, filter_hz))


def low_high_pass_filter(signal: BaseRaw, l_freq, h_freq):
    return signal.copy().filter(l_freq=l_freq, h_freq=h_freq)