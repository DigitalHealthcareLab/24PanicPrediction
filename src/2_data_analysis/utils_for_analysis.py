from pandas import DataFrame
from scipy.signal import welch
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
from CosinorPy import file_parser, cosinor, cosinor1
import pandas as pd 
import numpy as np 


def bandpower(data, sf, band, method='welch', window_sec=None, relative=False):
    band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        freqs, psd = welch(data, sf, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

def check_bandpower_value_a(col_index, col_HR):
    sampling_frequency =  1/60
    temp = pd.DataFrame()
    temp['index'] = col_index
    temp['HR'] = col_HR
    temp = temp.fillna(method='ffill')
    temp = temp.fillna(method='backfill')
    data = temp.HR
    power_rel = bandpower(data, sampling_frequency, [0.0005, 0.001], 'multitaper', relative=True)
    return power_rel

def check_bandpower_value_b(col_index, col_HR):
    sampling_frequency =  1/60
    temp = pd.DataFrame()
    temp['index'] = col_index
    temp['HR'] = col_HR
    temp = temp.fillna(method='ffill')
    temp = temp.fillna(method='backfill')
    data = temp.HR
    power_rel = bandpower(data, sampling_frequency, [0.0001, 0.0005], 'multitaper', relative=True)
    return power_rel

def check_bandpower_value_c(col_index, col_HR):
    sampling_frequency =  1/60
    temp = pd.DataFrame()
    temp['index'] = col_index
    temp['HR'] = col_HR
    temp = temp.fillna(method='ffill')
    temp = temp.fillna(method='backfill')
    data = temp.HR
    power_rel = bandpower(data, sampling_frequency, [0.00005, 0.0001], 'multitaper', relative=True)
    return power_rel

def check_bandpower_value_d(col_index, col_HR):
    sampling_frequency =  1/60
    temp = pd.DataFrame()
    temp['index'] = col_index
    temp['HR'] = col_HR
    temp = temp.fillna(method='ffill')
    temp = temp.fillna(method='backfill')
    data = temp.HR
    power_rel = bandpower(data, sampling_frequency, [0.00001, 0.00005], 'multitaper', relative=True)
    return power_rel

def mesor(col_index, col_HR):
    MESOR = cosinor1.fit_cosinor(col_index, col_HR, period= 1440, plot_on=False)[3]['values'][0]
    return MESOR

def amplitude(col_index, col_HR):
    amp = cosinor1.fit_cosinor(col_index , col_HR, period= 1440, plot_on=False)[3]['values'][1]
    return amp

def acrophase(col_index, col_HR):
    acr = cosinor1.fit_cosinor(col_index, col_HR, period= 1440, plot_on=False)[3]['values'][2]
    acrophase_min = cosinor.acrophase_to_hours(acr)
    acrophase = (acrophase_min +1440)/60
    return acrophase_min