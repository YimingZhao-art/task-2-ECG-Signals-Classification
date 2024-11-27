import csv
import os
import sys 

import biosppy.signals.ecg as ecg
import biosppy
import neurokit2 as nk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from scipy.fftpack import fft
import scipy.fftpack as sf

import hrvanalysis
import heartpy as hp

from multiprocessing import Pool

from tqdm import tqdm

freq_array = [
    0.46981001,
    0.94989355,
    1.34722766,
    1.92037653,
    2.45140581,
    3.58897684,
    5.61198292,
    7.85416366,
    11.77742192,
    19.56926069,
]

def extract_fft_heartbeat(heartbeat, n = 10, freq_array = freq_array):

    fourier_specture = np.abs(fft(heartbeat))
    freqs = sf.fftfreq(len(fourier_specture), 1. / 300.)
    fourier_specture = fourier_specture[freqs >= 0]
    freqs = freqs[freqs >= 0]
    
    # cut even more base on the freq_array
    fourier_specture = fourier_specture[freqs <= freq_array[-1]]
    freqs = freqs[freqs <= freq_array[-1]]
    
    # compute the sums of frequency bands
    sums = []
    sums.append(np.sum(fourier_specture[freqs <= freq_array[0]]))
    for i in range(len(freq_array) - 1):
        sum = np.sum(fourier_specture[np.logical_and(freqs > freq_array[i], freqs <= freq_array[i+1])])
        sums.append(sum)
        
    return sums

def extract_fft_feature(clean_signal):
    _, info = nk.ecg_peaks(ecg_cleaned=clean_signal, sampling_rate=300)
    fft_features = []
    n_peaks = 10
    peaks = info['ECG_R_Peaks']
    beats = biosppy.signals.ecg.extract_heartbeats(signal=clean_signal, rpeaks=peaks,sampling_rate=300)["templates"]
    n_beats = len(beats)
    for i in range(n_beats):
        fft_features.append(np.array(extract_fft_heartbeat(beats[i], n_peaks)))
    
    fft_features = list(np.array(fft_features).T)

    return fft_features

empty = np.empty(1)
empty[:] = np.nan

def get_pqrst_peaks(data):
    p_peaks = []
    q_peaks = []
    r_peaks_loc = []
    s_peaks = []
    t_peaks = []
    p_amp = []
    q_amp = []
    r_amp = []
    s_amp = []
    t_amp = []
    templates = ecg.ecg(signal=data, sampling_rate=300, show=False)["templates"]
    for template in templates:
        # calculate the locations
        try:
            # get local maximas and minimas with signal library
            loc_max = np.array(sp.signal.argrelextrema(template, np.greater))
            loc_min = np.array(sp.signal.argrelextrema(template, np.less))
            # find the maximum, but cut the search area to the first half to avoid
            # finding peaks at the wrong place
            r = np.argmax(template[: int(len(template) / 2)])
            
            # q and s are the first minima after and before the r value
            q = loc_min[loc_min < r][-1]
            s = loc_min[loc_min > r][0]
            # p and t are the first maxima after and before the r value
            p = loc_max[loc_max < r][-1]
            t = loc_max[loc_max > r][0]
            
            r_peaks_loc.append(r)
            q_peaks.append(q)
            s_peaks.append(s)
            p_peaks.append(p)
            t_peaks.append(t)
        except:
            ex = 0
            r_peaks_loc.append(ex)
            q_peaks.append(ex)
            s_peaks.append(ex)
            p_peaks.append(ex)
            t_peaks.append(ex)
        try:
            r_a = template[r]
            p_a = template[p]
            q_a = template[q]
            s_a = template[s]
            t_a = template[t]
        
            t_amp.append[t_a]
            p_amp.append[p_a]
            q_amp.append[q_a]
            s_amp.append[s_a]
            r_amp.append[r_a]
        except:
            ex = 0
            r_amp.append(ex)
            q_amp.append(ex)
            s_amp.append(ex)
            p_amp.append(ex)
            t_amp.append(ex)
        
    peaks = [np.array(p_peaks), np.array(q_peaks),np.array(r_peaks_loc), np.array(s_peaks), np.array(t_peaks)]
    amps = [ np.array(p_amp), np.array(q_amp),np.array(r_amp), np.array(s_amp), np.array(t_amp) ]

    return peaks, amps

# very bad almost never works
def get_pqrst_peaks_biosppy(data):
    try:
        ecg_ret = ecg.ecg(signal=data, sampling_rate=300, show=False)
        p_peaks = ecg.getPPositions(ecg_proc=ecg_ret)
        q_peaks = ecg.getQPositions(ecg_proc=ecg_ret)
        r_peaks = ecg_ret["rpeaks"]
        s_peaks = ecg.getSPositions(ecg_proc=ecg_ret)
        t_peaks = ecg.getTPositions(ecg_proc=ecg_ret)
    except:
        return empty, empty, empty, empty, empty

    return p_peaks, q_peaks, s_peaks, t_peaks, r_peaks


def extract_r_peaks(signal):
    _, r_peaks_nk = nk.ecg_peaks(signal, sampling_rate=300)
    s = len(r_peaks_nk['ECG_R_Peaks'])
    return np.array(r_peaks_nk['ECG_R_Peaks'])

def delineate(signal, r_peaks):
    try:
        _, waves_peak = nk.ecg_delineate(signal, r_peaks, sampling_rate=300, method='dwt')
    except:
        try:
            _, waves_peak = nk.ecg_delineate(signal, r_peaks, sampling_rate=300, method='peak')
        except:
            _, waves_peak = nk.ecg_delineate(signal, r_peaks, sampling_rate=300, method='cwt')
            
    return waves_peak

def extract_other_peaks(signal, r_peaks):
    
    if (len(r_peaks) >= 1): 
        try:
            waves_peak = delineate(signal, r_peaks)
        except:
            return empty, empty, empty, empty, empty

        p_peaks = np.array(waves_peak['ECG_P_Peaks'])       
        q_peaks = np.array(waves_peak['ECG_Q_Peaks'])
        s_peaks = np.array(waves_peak['ECG_S_Peaks'])
        t_peaks = np.array(waves_peak['ECG_T_Peaks'])

        return p_peaks, q_peaks, s_peaks, t_peaks, r_peaks
    else:
        return empty, empty, empty, empty, r_peaks

def extract_amp(signal, peaks):
    amps = []
    for p in peaks:
        mask = ~np.isnan(p)
        p_no_nan = p[mask].astype(int)
        amps.append(signal[p_no_nan])
    return amps

# not working
def extract_amp_loc(heartbeats, peaks):
    amps = []
    for p in peaks:
        peak_amps = []
        for i, beat in enumerate(heartbeats):
            if np.isnan(p[i]):
                continue
            peak_amps.append(beat[p[i]])
        amps.append(np.array(peak_amps))
    return amps

def extract_relative_pos(signal, left, right):
    if right.shape[0] == left.shape[0]:
        mask = ~np.logical_or(np.isnan(left), np.isnan(right))
        return right[mask] - left[mask]
    elif right.shape[0] < left.shape[0]:
        return extract_relative_pos(signal, left[1:], right)
    else:
        return extract_relative_pos(signal, left, right[1:])
        
def extract_loc(signal, peaks):
    q_loc = extract_relative_pos(signal, peaks[0], peaks[1])
    r_loc = extract_relative_pos(signal, peaks[0], peaks[2])
    s_loc = extract_relative_pos(signal, peaks[0], peaks[3])
    t_loc = extract_relative_pos(signal, peaks[0], peaks[4])
    return [q_loc, r_loc, s_loc, t_loc]
    
def extract_loc_hand(peaks):
    # assumes that non erroneous values are zero
    q_loc = peaks[1] - peaks[0]
    r_loc = peaks[2] - peaks[0]
    s_loc = peaks[3] - peaks[0]
    t_loc = peaks[4] - peaks[0]
    return [q_loc, r_loc, s_loc, t_loc]

def extract_dur(signal, peaks):
    pq_dur  = extract_relative_pos(signal, peaks[0], peaks[1])
    qrs_dur = extract_relative_pos(signal, peaks[1], peaks[3])
    st_dur  = extract_relative_pos(signal, peaks[3], peaks[4])
    return [pq_dur, qrs_dur, st_dur]

def extract_dur_and(peaks):
    pq_dur  = peaks[1] - peaks[0]
    qrs_dur = peaks[3] - peaks[1] 
    st_dur  = peaks[4] - peaks[3]
    return [pq_dur, qrs_dur, st_dur]

def extract_abs_diff_dur(durs):
    diffs_durs = []
    for dur in durs:
        np_dur = np.array(dur)
        n = len(np_dur)
        diffs_durs.append(np.abs(np_dur[1:n] - np_dur[0:n-1]))
    return diffs_durs

def extract_int(signal, peaks):
    if(len(peaks) > 1):
        rr_int = extract_relative_pos(signal, peaks[2][:-1], peaks[2][1:])
        pp_int = extract_relative_pos(signal, peaks[0][:-1], peaks[0][1:])
        tt_int = extract_relative_pos(signal, peaks[4][:-1], peaks[4][1:])
        return [pp_int, rr_int, tt_int]
    else:
        rr_int = extract_relative_pos(signal, peaks[0][:-1], peaks[0][1:])
        return [rr_int]

def extract_qrs_complex(signal, peaks):
    
    if (peaks[1].shape[0] != peaks[2].shape[0] or peaks[2].shape[0] != peaks[3].shape[0]):
        return [empty, empty, empty, empty]
    
    mask = ~np.logical_or.reduce(np.array([np.isnan(peaks[1]), np.isnan(peaks[2]), np.isnan(peaks[3])]))
    q = peaks[1][mask].astype(int)
    r = peaks[2][mask].astype(int)
    s = peaks[3][mask].astype(int)
    q_amp = signal[q]
    r_amp = signal[r]
    s_amp = signal[s]
    qr_amp = q_amp + r_amp
    qrs_wave = np.divide(q_amp, qr_amp)
    qr_wave = np.divide(q_amp, r_amp)
    rs_wave = np.divide(s_amp, r_amp)
    return [qr_amp, qrs_wave, qr_wave, rs_wave]

def extract_qrs_hand(amps):

    # peaks always have the same lengh
    #if (peaks[1].shape[0] != peaks[2].shape[0] or peaks[2].shape[0] != peaks[3].shape[0]):
    #    return [empty, empty, empty, empty]
    
    q_amp = amps[1]
    r_amp = amps[2]
    s_amp = amps[3]
    qr_amp = q_amp + r_amp
    qrs_wave = np.divide(q_amp, qr_amp, out=np.zeros_like(qr_amp).astype(float), where=(qr_amp!=0))
    qr_wave = np.divide(q_amp, r_amp, out=np.zeros_like(r_amp).astype(float), where=(r_amp!=0))
    rs_wave = np.divide(r_amp, s_amp, out=np.zeros_like(s_amp).astype(float), where=(s_amp!=0))
    return [qr_amp, qrs_wave, qr_wave, rs_wave]
    

def get_phases(signal, r_peaks):
    pass

def extract_ecg_data(signal, peak_meth = "hand"):
    
    #Check if signal is inverted and correct it if necessary
    
    signal, is_inverted = nk.ecg_invert(signal, sampling_rate=300, show=False)
    
    # Variable with additional values
    ind = []
    
    # Extract r peak
    r_peaks = extract_r_peaks(signal)
    
    # Extract other peaks
    if peak_meth == "bio":
        p_peaks, q_peaks, s_peaks, t_peaks, r_peaks = get_pqrst_peaks_biosppy(signal)
        peaks = [p_peaks, q_peaks, r_peaks, s_peaks, t_peaks]
        # Extract amplitudes
        amps = extract_amp(signal, peaks)
    elif peak_meth == "nk":
        p_peaks, q_peaks, s_peaks, t_peaks, _ = extract_other_peaks(signal, r_peaks=r_peaks)
        peaks = [p_peaks, q_peaks, r_peaks, s_peaks, t_peaks]
        # Extract amplitudes
        amps = extract_amp(signal, peaks)
        # Extract qrs complex
        qrs_complex = extract_qrs_complex(signal, peaks)
        # Extract locations
        locs = extract_loc(signal, peaks)
        # Extract durations
        durs = extract_dur(signal, peaks)
        # Extract intervals
        ints = extract_int(signal, peaks)
    else:
        peaks, amps = get_pqrst_peaks(signal)
        # Extract qrs complex
        qrs_complex = extract_qrs_hand(amps)
        # Extract locations
        locs = extract_loc_hand(peaks)
        # Extract durations
        durs = extract_dur_and(peaks)
        # Extract intervals
        ints = extract_int(signal, [r_peaks])
    
    
    # Diffs in the peaksof
    diff_amps = extract_abs_diff_dur(amps)
    
    # Extract difference of durations
    diff_dur = extract_abs_diff_dur(durs)

    # Extract interval differences
    diff_ints = extract_abs_diff_dur(ints)
    
    # ECG-Rate 
    ecg_rate = [np.array(nk.signal_rate(r_peaks, sampling_rate=300, desired_length=None))]
    
    data = amps + locs + durs + ints + qrs_complex + ecg_rate + diff_ints + diff_dur + diff_amps
    return data

def extract_hrv_features(r_peaks):
    tdf_names = [
        "mean_nni",
        "sdnn",
        "sdsd",
        "nni_50",
        "pnni_50",
        "nni_20",
        "pnni_20",
        "rmssd",
        "median_nni",
        "range_nni",
        "cvsd",
        "cvnni",
        "mean_hr",
        "max_hr",
        "min_hr",
        "std_hr",
    ]

    gf_names = ["triangular_index"]

    fdf_names = ["lf", "hf", "lf_hf_ratio", "lfnu", "hfnu", "total_power", "vlf"]

    cscv_names = [
        "csi",
        "cvi",
        "Modified_csi",
    ]

    pcp_names = ["sd1", "sd2", "ratio_sd2_sd1"]
    features = np.ndarray((len(tdf_names) + len(gf_names) + len(fdf_names) + len(cscv_names) + len(pcp_names),))
    features[:] = 0
    features = list(features)
    
    try:
        tdf = hrvanalysis.get_time_domain_features(r_peaks)
        gf = hrvanalysis.get_geometrical_features(r_peaks)
        fdf = hrvanalysis.get_frequency_domain_features(r_peaks)
        cscv = hrvanalysis.get_csi_cvi_features(r_peaks)
        pcp = hrvanalysis.get_poincare_plot_features(r_peaks)
        samp = hrvanalysis.get_sampen(r_peaks)
    except:
        return []

    for name in tdf_names:
        features.append(tdf[name])

    for name in gf_names:
        features.append(gf[name])

    for name in fdf_names:
        features.append(fdf[name])

    for name in cscv_names:
        features.append(cscv[name])

    for name in pcp_names:
        features.append(pcp[name])

    features.append(samp["sampen"])

    return features

def extract_hp_features(signal):
    try:
        _, measures = hp.process(signal, sample_rate=300)
    except:
        try:
            _, measures = hp.process(hp.flip_signal(signal), sample_rate=300)
        except:
            print("hp fail")
            return [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

    features = []
    features.append(measures["pnn50"])
    features.append(measures["pnn20"])
    features.append(measures["sd1"])
    features.append(measures["sd2"])
    features.append(measures["s"])
    features.append(np.log10(measures["sd1/sd2"] ** 2))
    return features

def extract_template_features(signal):
    templates = ecg.ecg(signal=signal, sampling_rate=300, show=False)["templates"]

    med_template = np.median(templates, axis=0)
    med_std = np.std(med_template)
    med_mean = np.mean(med_template)
    med_med = np.median(med_template)

    mean_template = np.mean(templates, axis=0)
    mean_std = np.std(mean_template)
    mean_mean = np.mean(mean_template)
    mean_med = np.median(mean_template)

    std_template = np.std(templates, axis = 0)
    std_std = np.std(std_template)
    std_mean = np.mean(std_template)
    std_med = np.median(std_template)

    return [med_std, med_mean, med_med, mean_std, mean_mean, mean_med, std_std, std_mean, std_med]

def s_to_noise_dB(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    return [20 * np.log10(abs(np.where(std == 0, 0, mean / std)))]

def s_over_thresh(signal, r_peaks, thresh):
    thres = np.max(signal) * thresh
    thresh_rate = sum(signal > thres) / len(signal)
    thresh_over_peak = thresh_rate / len(r_peaks)
    return [thresh_rate, thresh_over_peak]

def create_features(ecg_data):
    features = []
    for x in ecg_data:
        if len(x) > 0:
            mean = np.mean(x)
            std = np.std(x)
            median = np.median(x)
            min = np.min(x)
            max = np.max(x)
            skew = sp.stats.skew(x)
            kurtosis = sp.stats.kurtosis(x)
            variation = sp.stats.variation(x)
            iqr = sp.stats.iqr(x) # difference between the 0.75 and 0.25 quantile
            slope = x[0] - x[-1] #not really, but w/e
            new_features = [mean, std, median, min, max, skew, kurtosis, variation, iqr, slope] 
            features += new_features
        else:
            with open('filename.txt', 'a') as f:
                original_stdout = sys.stdout # Save a reference to the original standard output
                sys.stdout = f # Change the standard output to the file we created.
                print(f'No data for ecgdata ')
                sys.stdout = original_stdout

            features += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
    return np.array(features)

def get_features(df_raw_signals):
    
    features = []
    
    for i in tqdm(range(0, df_raw_signals.shape[0])):
        signal = df_raw_signals.iloc[i].dropna().to_numpy(dtype='float32')
        meth = "nk"
        try:
            cleaned_signal = nk.ecg_clean(signal, sampling_rate=300, method='neurokit')
            ecg_data = extract_ecg_data(cleaned_signal, meth)
        except:
            try:
                cleaned_signal = nk.ecg_clean(signal, sampling_rate=300, method='hamilton2002')
                ecg_data = extract_ecg_data(cleaned_signal, meth)
            except:
                try:
                    cleaned_signal = nk.ecg_clean(signal, sampling_rate=300, method='elgendi2010')
                    ecg_data = extract_ecg_data(cleaned_signal, meth)
                except:
                    print('really bad data point', i)
                    #exit(-1)
        fft_data = extract_fft_feature(cleaned_signal)
        r_peaks = extract_r_peaks(cleaned_signal)
        hrv_features = extract_hrv_features(r_peaks)
        hp_fetures = extract_hp_features(cleaned_signal)
        s_over_features = s_over_thresh(cleaned_signal, r_peaks, 0.7)
        template_features = extract_template_features(cleaned_signal)
        s_to_noise_feature = s_to_noise_dB(cleaned_signal)
        

        f = list(create_features(ecg_data)) + list(create_features(fft_data)) + hrv_features + hp_fetures + s_over_features + template_features + s_to_noise_feature
        features.append(f)
        
    
    df = pd.DataFrame(features)
    return df

def sub_features(arg_tuple):
    df_raw, idx = arg_tuple
    df_processed = get_features(df_raw)
    return idx, df_processed

def multi_features(df_raw_signals, n_cores=128):
    ids = df_raw_signals.index.to_list()
    split = np.array_split(ids, n_cores)
    
    chunks = []
    for l, i in zip(split, range(len(split))):
        start = l[0]
        end = l[-1]
        chunks.append((df_raw_signals.iloc[start:end+1], i))
    
    my_pool = Pool(n_cores)
    result = my_pool.map(sub_features, chunks)
    result = sorted(result, key=lambda tup: tup[0])
    
    df_list = [item[1] for item in result]
    df_final = pd.concat(df_list)
    df_final = df_final.reset_index(drop=True)
    
    return df_final


    