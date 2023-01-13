import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Binary
from glob import glob
from scipy import signal
import os, mea
import warnings
import scipy.signal as spsig
import matplotlib.patches as patches
from detect_peaks import detect_peaks
import matplotlib.cm as cmap
from scipy.signal import butter, sosfilt, sosfreqz
from tqdm.auto import tqdm

    

def load_openephys(folder,fs, window, start
                 , LFP=1):

    """Load chunk of open_ephys recording.

    Parameters
    ----------
    folder : str or path
        path to the open_ephys folder.
    fsd : positive integer, sampling rate of the signal (recording).
    window : positive number, optional (default = 300)
        window in [s] for the duration of the chunk. 
    start : positive number, optional (default = 0)
        start time of the chunk of the recording to extract.
    end : positive number, optional (default = 300)
        end time of the chunk of the recording to extract. need to be the same
    LFP : positive number, optional (default = 1)
        channel of the recording node where the signal is store on the open_ephys folder.
    Returns
    -------
    recording : 1D array_like
        numpy array of a channel.
    timestamp : 1D array_like
        timestamp of the event.
    """

    # Sample rate and desired cutoff frequencies (in Hz).
    from open_ephys.analysis import Session

    session = Session(folder)
    recordnode = session.recordnodes[0]
    recording = session.recordnodes[0].recordings[0]
    timing = recording.continuous[LFP].timestamps
    sample = np.size(timing)
    times = np.zeros(int(window*fs))

    for i in range(int(window*fs)):
        times[i] = start+(i/fs)
    st = (start*fs)
    end = (start*fs)+window*fs
    data = recording.continuous[LFP].get_samples(start_sample_index= st, end_sample_index= end)

    recording = data
    timestamp = times
    return recording, timestamp


def preprocessing(recording):

    """CMR.

    Parameters
    ----------
    recording : 1D array_like
        numpy array of a channel.
    Returns
    -------
    recording_preprocessed : 1D array_like
        numpy array of the preprocessed recording.
    """
    dataT = recording.transpose()
    car = np.median(dataT, axis=0) # calculate the Common Average Reference across all the trials for each channel
    recording_preprocessed = dataT - car
    return recording_preprocessed


def average_over_channel(recording, channel):
    """ average across channel.

    Parameters
    ----------
    recording : 1D array_like
        numpy array of a channel.
    chanbel : positive number, optional (default = 0)
        Integer to perform average accross channel. 
    Returns
    -------
    recording_preprocessed : 1D array_like
        numpy array of the preprocessed recording.
    """
    recording_preprocessed = np.mean(recording[channel*24:24*(channel+1)],axis=0)
    return recording_preprocessed


def detection(recording, timestamp ,fsd, std_factor=4, filt= (2,20)
                 , show=True, spike=(False,0)):

    """Detect events in a neuropixel dataset based on their amplitude and other features.

    Parameters
    ----------
    recording : 1D numpy array from neuropixel recording
        (Array over 1 channel or average across several channel).
    timestamp : 1D numpy array of timestamp related to the recording.
    fsd : positive integer, sampling rate of the signal (recording).
    std_factor : positive number, optional (default = 4)
        Standart deviation factor for the detection of peak or  valley.
    filt : tuple of positive number, optional (default = (2,20))
        Frequency for the lower part of the bandpass filter.
    show : bool, optional (default = True)
        if True (1), plot data in matplotlib figure.
    spike : Tuple, optional (default = (False,0))
        if True (1), also plot raster align with the data in matplotlib figure,
        second index is the pandas dataframe with the spike cluster[Time,Cluster]

    Returns
    -------
    ripple_duration : 1D array_like
        duration of the event.
    ripple_start_idx : 1D array_like
        Start index of the event.
    ripple_end_idx : 1D array_like
        End index of the event.
    ripple_peak_idx: 1D array_like
        Index of the peak of the event.
    ripple_start_time : 1D array_like
        Start timestamps of the event.
    ripple_end_time : 1D array_like
        End timestamps of the event.
    ripple_peak_time : 1D array_like
        Peak timestamp of the event.
    ripple_peak_amp : 1D array_like
        Peak's amplitude of the event.
    """
    
    warnings.filterwarnings('ignore')

    # load the data
    eeg_sig = recording
    # load the timestamp
    time = timestamp

    # load the filtered ripple data
    f_ripple = filt
    filt_rip_sig = mea.get_bandpass_filter_signal(eeg_sig, fsd, f_ripple)
    #        mea.plot_lfp(eeg_sig[:fsd*1000], filt_rip_sig[:fsd*1000], time[:fsd*1000])
    # calculate the envelope of filtered data
    filt_rip_env = abs(spsig.hilbert(filt_rip_sig))
    filt_rip_env_zscore = mea.zscore(filt_rip_env)

    # Root mean square (RMS) ripple power calculation
    ripple_power = mea.window_rms(filt_rip_sig, 10)
    ripple_power_env_zscore = mea.window_rms(filt_rip_env_zscore, 10)

    # calculate mean and standard deviation of the ripple power
    mean_rms = np.nanmean(ripple_power)
    std_rms = np.nanstd(ripple_power)
    mean_rms_env_zscore = np.nanmean(ripple_power_env_zscore)
    std_rms_env_zscore = np.nanstd(ripple_power_env_zscore)

    minThreshTime = .02 # minimum duration threshold in seconds
    maxThreshTime = .2 # maximum duration threshold in seconds
    ripplePowerThresh = mean_rms + std_factor*std_rms #peak power threshold
    falloffThresh = mean_rms + 0.5*std_rms # falloffthresh = mean + .5*std_rms

    ripplePowerThresh_env_zscore = mean_rms_env_zscore + std_factor*std_rms_env_zscore #peak power threshold
    falloffThresh_env_zscore = mean_rms_env_zscore + 0.5*std_rms_env_zscore # falloffthresh = mean + .5*std_rms

    # data to hold the variables
    ripple_duration = []
    ripple_start_time = []
    ripple_end_time = []
    ripple_start_idx = []
    ripple_end_idx = []
    ripple_peak_time = []
    ripple_peak_idx = []
    ripple_peak_amp = []

    # data to hold the variables
    ripple_durationv2 = []
    ripple_start_timev2 = []
    ripple_end_timev2 = []
    ripple_start_idxv2 = []
    ripple_end_idxv2 = []
    ripple_peak_timev2 = []
    ripple_peak_idxv2 = []
    ripple_peak_ampv2 = []

    # ripple peak detection using ripple power threshold
    # find peaks using height as power thershold, min ripple distance = 200ms
    idx_peak = detect_peaks(-filt_rip_sig, mph=ripplePowerThresh, mpd=0.3*fsd)
    # iterate over each peak
    for idx in idx_peak:
        # nice trick: no point looking beyond +/- 300ms of the peak
        # since the ripple cannot be longer than that
        idx_max = idx + int(maxThreshTime*fsd)
        idx_min = idx - int(maxThreshTime*fsd)
        # boundary check
        if idx_min<0:
            idx_min = 0
        # find the left and right falloff points for individual ripple
        ripple_power_sub = ripple_power[idx_min:idx_max]
        idx_falloff = np.where(ripple_power_sub<=falloffThresh)[0]
        idx_falloff += idx_min
        # find the start and end index for individual ripple
        _, startidx = mea.find_le(idx_falloff, idx)
        _, endidx = mea.find_ge(idx_falloff, idx)
        # accounting for some boundary conditions
        if startidx is None:
            th = (maxThreshTime + minThreshTime)//2
            startidx = idx - int(th*fsd)
        if endidx is None:
            endidx = 2*idx - startidx
            if endidx > len(time):
                endidx = len(time)-1
        #duration CHECK!
        dur = time[endidx-1]-time[startidx]
    #            print(time[idx], time[endidx], time[startidx], dur, dur>=minThreshTime and dur<=maxThreshTime)
        # add the ripple to saved data if it passes duration threshold
        if dur>=minThreshTime and dur<=maxThreshTime:
            ripple_duration.append(dur)
            ripple_start_idx.append(startidx)
            ripple_end_idx.append(endidx)
            ripple_peak_idx.append(idx)
            ripple_start_time.append(time[startidx])
            ripple_end_time.append(time[endidx])
            ripple_peak_time.append(time[idx])
            ripple_peak_amp.append(ripple_power[idx])

    # ripple peak detection using z-score threshold
    # find peaks using height as power thershold, min ripple distance = 200ms
    idx_peak2 = detect_peaks(filt_rip_env_zscore, mph=ripplePowerThresh_env_zscore, mpd=.3*fsd)
    # iterate over each peak
    for idx in idx_peak2:
        # nice trick: no point looking beyond +/- 300ms of the peak
        # since the ripple cannot be longer than that
        idx_max = idx + int(maxThreshTime*fsd)
        idx_min = idx - int(maxThreshTime*fsd)
        # boundary check
        if idx_min<0:
            idx_min = 0
        # find the left and right falloff points for individual ripple
        filt_rip_env_zscore_sub = filt_rip_env_zscore[idx_min:idx_max]
        idx_falloff = np.where(filt_rip_env_zscore_sub<=0.15)[0]
        idx_falloff += idx_min
        # find the start and end index for individual ripple
        _, startidx = mea.find_le(idx_falloff, idx)
        _, endidx = mea.find_ge(idx_falloff, idx)
        # accounting for some boundary conditions
        if startidx is None:
            th = (maxThreshTime + minThreshTime)//2
            startidx = idx - int(th*fsd)
        if endidx is None:
            endidx = 2*idx - startidx
            if endidx > len(time):
                endidx = len(time)-1
        # duration CHECK!
        dur = time[endidx-1]-time[startidx]
    #            print(time[idx], time[endidx], time[startidx], dur, dur>=minThreshTime and dur<=maxThreshTime)
        # add the ripple to saved data if it passes duration threshold
        if dur>=minThreshTime and dur<=maxThreshTime:
            ripple_durationv2.append(dur)
            ripple_start_idxv2.append(startidx)
            ripple_end_idxv2.append(endidx)
            ripple_peak_idxv2.append(idx)
            ripple_start_timev2.append(time[startidx])
            ripple_end_timev2.append(time[endidx])
            ripple_peak_timev2.append(time[idx])
            ripple_peak_ampv2.append(filt_rip_env_zscore[idx])

    # plot the processed data
    if show:
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4, ncols=1, sharex=True)
        plt.rcParams['figure.figsize'] = [10, 10]
        ax1.plot(time, eeg_sig, c='k')
        for ind_ in range(len(ripple_peak_time)):
            rect = patches.Rectangle((ripple_start_time[ind_],np.nanmin(eeg_sig)),
                                     ripple_end_time[ind_] - ripple_start_time[ind_],
                                     np.nanmax(eeg_sig) - np.nanmin(eeg_sig),
                                     linewidth=1,edgecolor='none',alpha=0.5,facecolor='y')
            ax1.add_patch(rect)
        ax1.set_ylabel('Raw signal', fontsize=16)
        ax2.plot(time, filt_rip_sig, c='r')
        for ind_ in range(len(ripple_peak_time)):
            rect = patches.Rectangle((ripple_start_time[ind_],np.nanmin(filt_rip_sig)),
                                     ripple_end_time[ind_] - ripple_start_time[ind_],
                                     np.nanmax(filt_rip_sig) - np.nanmin(filt_rip_sig),
                                     linewidth=1,edgecolor='none',alpha=0.5,facecolor='y')
            ax2.add_patch(rect)
        ax2.set_ylabel('Filtered signal', fontsize=16)
        ax3.plot(time, filt_rip_env_zscore, c='m')
        for ind_ in range(len(ripple_peak_timev2)):
            rect = patches.Rectangle((ripple_start_timev2[ind_],np.nanmin(filt_rip_env_zscore)),
                                     ripple_end_timev2[ind_] - ripple_start_timev2[ind_],
                                     np.nanmax(filt_rip_env_zscore) - np.nanmin(filt_rip_env_zscore),
                                     linewidth=1,edgecolor='none',alpha=0.5,facecolor='gray')
            ax3.add_patch(rect)
        ax3.set_ylabel('zscored filtered envelope', fontsize=16)
        ax4.plot(time, ripple_power, c='g')
        ax4.plot(time[ripple_peak_idx], ripple_power[ripple_peak_idx], 'b.')
        for ind_ in range(len(ripple_peak_time)):
            rect = patches.Rectangle((ripple_start_time[ind_],0),ripple_end_time[ind_] - ripple_start_time[ind_],np.nanmax(ripple_power),
                                     linewidth=1,edgecolor='none',alpha=0.5,facecolor='y')
            ax4.add_patch(rect)
        
        ax4.set_ylabel("Event power", fontsize=16)
        #plt.savefig('/camp/home/combadk/working/raw_data/raw_NP/NP_221003_A/LFP_Fig/downsampled/event_detection_example4.eps')
        plt.show()
    elif spike[0]:
        fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5, ncols=1, sharex=True)
        plt.rcParams['figure.figsize'] = [10, 10]
        ax1.plot(time, eeg_sig, c='k')
        for ind_ in range(len(ripple_peak_time)):
            rect = patches.Rectangle((ripple_start_time[ind_],np.nanmin(eeg_sig)),
                                     ripple_end_time[ind_] - ripple_start_time[ind_],
                                     np.nanmax(eeg_sig) - np.nanmin(eeg_sig),
                                     linewidth=1,edgecolor='none',alpha=0.5,facecolor='y')
            ax1.add_patch(rect)
        ax1.set_ylabel('Raw signal', fontsize=16)
        ax2.plot(time, filt_rip_sig, c='r')
        for ind_ in range(len(ripple_peak_time)):
            rect = patches.Rectangle((ripple_start_time[ind_],np.nanmin(filt_rip_sig)),
                                     ripple_end_time[ind_] - ripple_start_time[ind_],
                                     np.nanmax(filt_rip_sig) - np.nanmin(filt_rip_sig),
                                     linewidth=1,edgecolor='none',alpha=0.5,facecolor='y')
            ax2.add_patch(rect)
        ax2.set_ylabel('Filtered signal', fontsize=16)
        ax3.plot(time, filt_rip_env_zscore, c='m')
        for ind_ in range(len(ripple_peak_timev2)):
            rect = patches.Rectangle((ripple_start_timev2[ind_],np.nanmin(filt_rip_env_zscore)),
                                     ripple_end_timev2[ind_] - ripple_start_timev2[ind_],
                                     np.nanmax(filt_rip_env_zscore) - np.nanmin(filt_rip_env_zscore),
                                     linewidth=1,edgecolor='none',alpha=0.5,facecolor='gray')
            ax3.add_patch(rect)
        ax3.set_ylabel('zscored filtered envelope', fontsize=16)
        ax4.plot(time, ripple_power, c='g')
        ax4.plot(time[ripple_peak_idx], ripple_power[ripple_peak_idx], 'b.')
        for ind_ in range(len(ripple_peak_time)):
            rect = patches.Rectangle((ripple_start_time[ind_],0),ripple_end_time[ind_] - ripple_start_time[ind_],np.nanmax(ripple_power),
                                     linewidth=1,edgecolor='none',alpha=0.5,facecolor='y')
            ax4.add_patch(rect)
        
        ax4.set_ylabel("Event power", fontsize=16)
        ax5.set_ylabel("Cluster", fontsize=16)

        ax5.scatter(spike[1]['time_x'],spike[1]['new_cluster'],marker = '|')
        #plt.savefig('/camp/home/combadk/working/raw_data/raw_NP/NP_221003_A/LFP_Fig/downsampled/event_detection_example4.eps')
        plt.show()

    return ripple_duration, ripple_start_idx, ripple_end_idx, ripple_peak_idx, ripple_start_time, ripple_end_time, ripple_peak_time, ripple_peak_amp