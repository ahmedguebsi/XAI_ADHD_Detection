import numpy as np
import pandas as pd
from scipy import stats, signal, integrate
from statsmodels import tsa
import statsmodels.api as sm
import itertools
import bisect
from sklearn.metrics.cluster import normalized_mutual_info_score as normed_mutual_info
from sklearn.metrics import mutual_info_score

def filt_data(eegData, lowcut, highcut, fs, order=7):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filt_eegData = signal.lfilter(b, a, eegData, axis = 1)
    return filt_eegData


##########
# compute the bandpower (area under segment (from fband[0] to fband[1] in Hz)
# of curve in freqency domain) of data, at sampling frequency of Fs (100 ussually)
def bandpower(data, fs, fband):
    freqs, powers = periodogram(data, fs)
    idx_min = np.argmax(freqs > fband[0]) - 1
    idx_max = np.argmax(freqs > fband[1]) - 1
    idx_delta = np.zeros(dtype=bool, shape=freqs.shape)
    idx_delta[idx_min:idx_max] = True
    bpower = simps(powers[idx_delta], freqs[idx_delta])
    return bpower


##########
# computes the same thing as vecbandpower but with a loop
def pfvecbandpower(data, fs, fband):
    bpowers = np.zeros((data.shape[0], data.shape[2]))
    for i in range(data.shape[0]):
        freqs, powers = periodogram(data[i, :, :], fs, axis=0)
        idx_min = np.argmax(freqs > fband[0]) - 1
        idx_max = np.argmax(freqs > fband[1]) - 1
        idx_delta = np.zeros(dtype=bool, shape=freqs.shape)
        idx_delta[idx_min:idx_max] = True

        bpower = simps(powers[idx_delta, :], freqs[idx_delta], axis=0)
        bpowers[i, :] = bpower

    return bpowers

##########
# Cepstrum Coefficients (n=2)
def mfcc(eegData,fs,order=2):
    H = np.zeros((eegData.shape[0], eegData.shape[2],order))
    for chan in range(H.shape[0]):
        for epoch in range(H.shape[1]):
            H[chan, epoch, : ] = librosa.feature.mfcc(np.asfortranarray(eegData[chan,:,epoch]), sr=fs)[0:order].T
    return H
##########
# Coherence in the Delta Band
def CoherenceDelta(eegData, i, j, fs=100):
    nfft=eegData.shape[1]
    f, Cxy = signal.coherence(eegData[i,:,:], eegData[j,:,:], fs=fs, nfft=nfft, axis=0)#, window=np.hanning(nfft))
    out = np.mean(Cxy[np.all([f >= 0.5, f<=4], axis=0)], axis=0)
    return out

##########
# Mutual information
def calculate2Chan_MI(eegData,ii,jj,bin_min=-200, bin_max=200, binWidth=2):
    H = np.zeros(eegData.shape[2])
    bins = np.arange(bin_min+1, bin_max, binWidth)
    for epoch in range(eegData.shape[2]):
        c_xy = np.histogram2d(eegData[ii,:,epoch],eegData[jj,:,epoch],bins)[0]
        H[epoch] = mutual_info_score(None, None, contingency=c_xy)
    return H

##########
# Granger causality
def calcGrangerCausality(eegData,ii,jj):
    H = np.zeros(eegData.shape[2])
    for epoch in range(eegData.shape[2]):
        X = np.vstack([eegData[ii,:,epoch],eegData[jj,:,epoch]]).T
        H[epoch] = tsa.stattools.grangercausalitytests(X, 1, addconst=True, verbose=False)[1][0]['ssr_ftest'][0]
    return H


##########
# correlation across channels
def PhaseLagIndex(eegData, i, j):
    hxi = ss.hilbert(eegData[i,:,:])
    hxj = ss.hilbert(eegData[j,:,:])
    # calculating the INSTANTANEOUS PHASE
    inst_phasei = np.arctan(np.angle(hxi))
    inst_phasej = np.arctan(np.angle(hxj))

    out = np.abs(np.mean(np.sign(inst_phasej - inst_phasei), axis=0))
    return out

##########
# ARMA coefficients
def arma(eegData,order=2):
    H = np.zeros((eegData.shape[0], eegData.shape[2],order))
    for chan in range(H.shape[0]):
        for epoch in range(H.shape[1]):
            arma_mod = sm.tsa.ARMA(eegData[chan,:,epoch], order=(order,order))
            arma_res = arma_mod.fit(trend='nc', disp=-1)
            H[chan, epoch, : ] = arma_res.arparams
    return H
##########
# Cross-correlation Magnitude
def crossCorrMag(eegData,ii,jj):
	crossCorr_res = []
	for ii, jj in itertools.combinations(range(eegData.shape[0]), 2):
		crossCorr_res.append(crossCorrelation(eegData, ii, jj))
	crossCorr_res = np.array(crossCorr_res)
	return crossCorr_res

##########
# Cross-correlation Lag
def corrCorrLag(eegData,ii,jj,fs=100):
	crossCorrLag_res = []
	for ii, jj in itertools.combinations(range(eegData.shape[0]), 2):
		crossCorrLag_res.append(corrCorrLag(eegData, ii, jj, fs))
	crossCorrLag_res = np.array(crossCorrLag_res)
	return crossCorrLag_res
def SlopeLagIndex(x):
    return 0