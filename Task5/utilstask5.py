# coding: utf-8
import numpy as np
from scipy import integrate
from biosppy.signals import eeg
from scipy.signal import find_peaks,peak_prominences,peak_widths,periodogram
from scipy.stats import kurtosis,skew
from sklearn.decomposition import PCA 

def StatFeature(arrays):
    mean = np.mean(arrays)
    std = np.std(arrays)
    maxv = np.max(arrays)
    minv = np.min(arrays)
    return [mean,std,maxv, minv]


def findLFHF(psd, w):
    VLFpsd = VLFw = LFpsd = LFw = HFpsd = HFw = np.empty(0)
    m = w.shape[0]

    for i in range(0, m):
        if w[i] <= 0.05:
            VLFpsd = np.append(VLFpsd, psd[i])
            VLFw = np.append(VLFw, w[i])
        if w[i] > 0.05 and w[i] <= 0.15:
            LFpsd = np.append(LFpsd, psd[i])
            LFw = np.append(LFw, w[i])
        if w[i] > 0.15 and w[i] <= 0.4:
            HFpsd = np.append(HFpsd, psd[i])
            HFw = np.append(HFw, w[i])

    LF = integrate.trapz(LFw, LFpsd) / (integrate.trapz(w, psd) - integrate.trapz(VLFw, VLFpsd))
    HF = integrate.trapz(HFw, HFpsd) / (integrate.trapz(w, psd) - integrate.trapz(VLFw, VLFpsd))
    LFHFratio = LF / HF
    inter = LF / (LF + HF)
    if HFpsd.size:
        [maxHFD, maxIndex] = max((v, i) for i, v in enumerate(HFpsd))
        FreqmaxP = HFw[maxIndex]
    else:
        maxHFD = 0
        FreqmaxP = 0
    return (LF, HF, FreqmaxP, maxHFD, LFHFratio, inter)


def EMGFeatures(raw_signal, fs=128):
    # Statistical Features
    [_,std,maxv,minv] = StatFeature(raw_signal)
    
    # Power Spectrum
    w = np.hamming(len(raw_signal))
    w, psd = periodogram(raw_signal, window=w, detrend=False)
    _, _, _, maxHFD, _, _ = findLFHF(psd, w)
    
    # Time Series
    kurt = kurtosis(raw_signal)
    sk = skew(raw_signal)
        
    # Peak Features
    [peaks,_] = find_peaks(raw_signal)
    pprom = peak_prominences(raw_signal,peaks)[0]
    contour_heights = raw_signal[peaks] - pprom
    pwid = peak_widths(raw_signal,peaks,rel_height=0.4)[0]
    [ppmean,ppstd,_,ppmin] = StatFeature(pprom)
    [pwmean,pwstd,pwmax,pwmin] = StatFeature(pwid)
 
    return np.array([std,maxv,minv,maxHFD, kurt,sk,ppmean,ppstd,ppmin,pwmean,pwstd,pwmax,pwmin])


def EEGFeatures(raw_signal,signal_1,signal_2, fs=128):
    bio_signal = np.transpose(raw_signal)
        
    # Statistical Features
    [_,std1,maxv1,minv1] = StatFeature(signal_1)
    [_,std2,maxv2,minv2] = StatFeature(signal_2)    
        
    # Power Features
    [_, theta, alpha_low,alpha_high,beta, gamma]= eeg.get_power_features(signal=bio_signal, sampling_rate=fs)
        
    [theta1_mean,theta1_std,theta1_max,theta1_min] = StatFeature(theta[:,0])
    [theta2_mean,theta2_std,theta2_max,theta2_min] = StatFeature(theta[:,1])
    [alpha_low1_mean,alpha_low1_std,alpha_low1_max,alpha_low1_min] = StatFeature(alpha_low[:,0])
    [alpha_low2_mean,alpha_low2_std,alpha_low2_max,alpha_low2_min] = StatFeature(alpha_low[:,1])
    [alpha_high1_mean,alpha_high1_std,alpha_high1_max,alpha_high1_min] = StatFeature(alpha_high[:,0])
    [alpha_high2_mean,alpha_high2_std,alpha_high2_max,alpha_high2_min] = StatFeature(alpha_high[:,1])
    [beta1_mean,beta1_std,beta1_max,beta1_min] = StatFeature(beta[:,0])
    [beta2_mean,beta2_std,beta2_max,beta2_min] = StatFeature(beta[:,1])
    [gamma1_mean,gamma1_std,gamma1_max,gamma1_min] = StatFeature(gamma[:,0])
    [gamma2_mean,gamma2_std,gamma2_max,gamma2_min] = StatFeature(gamma[:,1])
        
        
    # Power Spectrum
    w = np.hamming(len(signal_1))
    w, psd = periodogram(signal_1, window=w, detrend=False)
    _, _, FreqmaxP1, _, _, _ = findLFHF(psd, w)
    w = np.hamming(len(signal_2))
    w, psd = periodogram(signal_2, window=w, detrend=False)
    _, _, FreqmaxP2, _, _, _ = findLFHF(psd, w)
        
    # Time Series
    kurt1 = kurtosis(signal_1)
    skew1 = skew(signal_1)
    kurt2 = kurtosis(signal_2)
    skew2 = skew(signal_2)
        
    # Peak Features
    [peaks1,_] = find_peaks(signal_1)
    pprom1 = peak_prominences(signal_1,peaks1)[0]
    contour_heights1 = signal_1[peaks1] - pprom1
    pwid1 = peak_widths(signal_1,peaks1,rel_height=0.4)[0]
    [ppmean1,ppstd1,_,ppmin1] = StatFeature(pprom1)
    [pwmean1,pwstd1,pwmax1,pwmin1] = StatFeature(pwid1)
        
    [peaks2,_] = find_peaks(signal_2)
    pprom2 = peak_prominences(signal_2,peaks2)[0]
    contour_heights2 = signal_2[peaks2] - pprom2
    pwid2 = peak_widths(signal_2,peaks2,rel_height=0.4)[0]
    [ppmean2,ppstd2,_,ppmin2] = StatFeature(pprom2)
    [pwmean2,pwstd2,pwmax2,pwmin2] = StatFeature(pwid2)
 
    return np.array([theta1_mean,theta1_std,theta1_max,theta1_min, theta2_mean,theta2_std,theta2_max,theta2_min, alpha_low1_mean,alpha_low1_std,alpha_low1_max,alpha_low1_min,alpha_low2_mean,alpha_low2_std,alpha_low2_max,alpha_low2_min,alpha_high1_mean,alpha_high1_std,alpha_high1_max,alpha_high1_min,alpha_high2_mean,alpha_high2_std,alpha_high2_max,alpha_high2_min,beta1_mean,beta1_std,beta1_max,beta1_min,beta2_mean,beta2_std,beta2_max,beta2_min,gamma1_mean,gamma1_std,gamma1_max,gamma1_min,gamma2_mean,gamma2_std,gamma2_max,gamma2_min,FreqmaxP1,kurt1,skew1,ppmean1,ppstd1,ppmin1,pwmean1,pwstd1,pwmax1,pwmin1,FreqmaxP2,kurt2,skew2,ppmean2,ppstd2,ppmin2,pwmean2,pwstd2,pwmax2,pwmin2,std1,maxv1,minv1,std2,maxv2,minv2])


def ExtractFeatures(eeg1,eeg2,emg, fs=128):
    raw_signal = np.concatenate(([eeg1],[eeg2]))
    feature1 = EEGFeatures(raw_signal,eeg1,eeg2,fs=fs)
    feature2 = EMGFeatures(emg)
    return np.concatenate((feature1, feature2))
    #return feature1

