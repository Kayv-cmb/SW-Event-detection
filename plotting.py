import os, mea
import warnings
import numpy as np
import pandas as pd
import scipy.signal as spsig
import neuralynxio as nlxio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from detect_peaks import detect_peaks
import matplotlib.cm as cmap
from open_ephys.analysis import Session
import event_detection
import seaborn as sns
import spike_LFP
from tqdm.auto import tqdm


def heatmap(folder,fs, window, std_fac, filt= (2,20), LFP=1):
    """Heatmap of the event detected over 24 channel.

    Parameters
    ----------
    folder : str or path to the neuropixel recording
        (Array over 1 channel or average across several channel).
    fs : positive integer, sampling rate of the signal (recording).
    window : positive number
       window in [s] to create a chunk of the recording.
    filt : tuple of positive number, optional (default = (2,20))
        Frequency for the lower part of the bandpass filter.
    std_fac : positive number, optional (default = 4)
       Standart deviation factor for the detection of event.
    LFP : positive number, optional (default = 1)
       channel of the recording node where the signal is store on the open_ephys folder.

    Returns
    -------
    Heatmap of the event
    """
    session = Session(folder)
    recordnode = session.recordnodes[0]
    recording = session.recordnodes[0].recordings[0]
    data = recording.continuous[LFP]
    timing = recording.continuous[LFP].timestamps
    sample = np.size(timing)
    
    arr = np.empty((0,int(384/24)), int)
    array_event_time = np.empty((0,int(384/24)), int)
    for j in tqdm(range(0,int(sample/fs),window)):
        nb_peak_ch = []
        start = j
        rec, timestamp = event_detection.load_openephys(folder ,fs, window, start, LFP=LFP)
        rec_pre = event_detection.preprocessing(rec)

        for k in range(int(384/24)):
            rec_av = event_detection.average_over_channel(rec_pre,k)

            # load the data
            duration, start_idx, end_idx, peak_idx, start_time,end_time, peak_time, peak_amp = event_detection.detection(rec_av, timestamp ,fs,std_factor = std_fac ,filt = filt,show = False)
            nb_peak_ch.append(np.size(peak_time))
        a = np.array([nb_peak_ch])
        arr = np.append(arr,a, axis = 0)
    df_norm_col=(arr-arr.mean())/arr.std()
    arrT = arr.transpose()
    ax = sns.heatmap(arrT, cmap='viridis')

    ax.invert_yaxis()
    
    
def hist(folder, fs, window, LFP=1):
    """Histogram of the event detected over the first 24 channel.

    Parameters
    ----------
    folder : str or path to the neuropixel recording
        (Array over 1 channel or average across several channel).
    fs : positive integer, sampling rate of the signal (recording).
    window : positive number
       window in [s] to create a chunk of the recording.
    LFP : positive number, optional (default = 1)
       channel of the recording node where the signal is store on the open_ephys folder.


    Returns
    -------
    Histogramme of the event
    """
    session = Session(folder)
    recordnode = session.recordnodes[0]
    recording = session.recordnodes[0].recordings[0]
    data = recording.continuous[LFP]
    timing = recording.continuous[LFP].timestamps
    sample = np.size(timing)
    rg = sample/(fs*window)

    nb_peak = []
    time_peak = []
    event = []
    for j in tqdm(range(0,int(sample/fs),window)):

        start = j
        rec, timestamp = event_detection.load_openephys(folder ,fs, window, start, LFP=LFP)

        rec = event_detection.preprocessing(rec)
        rec = event_detection.average_over_channel(rec, 0)

        # load the data
        duration, start_idx, end_idx,  peak_idx,  start_time,  end_time,  peak_time,  peak_amp = event_detection.detection(rec, timestamp ,fs, show = False)

        nb_peak.append(np.size( peak_time))
    tim= np.arange(round(rg+1))*5
    som = np.sum(nb_peak)
    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=False)
    ax1.bar(tim,nb_peak, width = 5, color='k')

def template(folder,channel = 0, LFP=1):
    """Template of the event as an average of all the event detected across the recording.

    Parameters
    ----------
    folder : str or path to the neuropixel recording
        (Array over 1 channel or average across several channel).
    fs : positive integer, sampling rate of the signal (recording).
    window : positive number
       window in [s] to create a chunk of the recording.
    LFP : positive number, optional (default = 1)
       channel of the recording node where the signal is store on the open_ephys folder.

    Returns
    -------
    plot the template
    """
    
    #Extraction of the recording
    nb_temp_piri = []
    time_peak = []
    event = []
    decay_piri = []


    for j in tqdm(range(0,int(sample/fs),window)):

        start = j
        rec, timestamp = event_detection.load_openephys(folder ,fs, window, start, LFP=LFP)

        rec = event_detection.preprocessing(rec)
        rec = event_detection.average_over_channel(rec, channel)

        # load the data
        duration, start_idx, end_idx,  peak_idx,  start_time,  end_time,  peak_time,  peak_amp = event_detection.detection(rec, timestamp ,fs, show = False)

        nb_peak.append(np.size( peak_time))

    template_piri = [0]
    for i in peak_time:
        start = int((i-0.2)*fs)
        end = int((i+0.3)*fs)
        rec, timestamp = event_detection.load_openephys(folder ,fs, window, start, LFP=LFP)
        rec = event_detection.preprocessing(rec)
        average_piri = event_detection.average_over_channel(rec, 0)
        template_piri = template_piri + average_piri
    print(len(template_piri))
    size_temp_piri = len(nb_temp_piri)
    temp_piri = template_piri/(size_temp_piri+1)
    wind_tm = fs*0.5
    times = np.zeros(int(wind_tm))
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    plt.rcParams['figure.figsize'] = [10, 10]
    ax1.plot(times, temp_piri, c='k')
    plt.show()

    
def event_spike_freq(folder ,fs ,fsd,window, phy_path,chunk_wake, chunk_sleep, LFP=LFP):

    """Comparaison of spike frequency during wake, sleep and event.

    Parameters
    ----------
    folder : str or path to the neuropixel recording
        (Array over 1 channel or average across several channel).
    fs : positive integer, sampling rate of the signal LFP(recording).
    fsd : positive integer, sampling rate of the signal (recording).
    window : positive number
       window in [s] to create a chunk of the recording.
    phy_path : pandas DataFrame,
        of the spike during wake
    chunk_wake : tuple of q bool and 2 positive integer, optional (default = (False,0,0))
        The boolean is to know if we kneed to chunk,
        the first interger is the start of the chunk,
        the second is the end of the chunk.
    chunk_sleep : tuple of q bool and 2 positive integer, optional (default = (False,0,0))
        The boolean is to know if we kneed to chunk,
        the first interger is the start of the chunk,
        the second is the end of the chunk.
    LFP : positive number, optional (default = 1)
       channel of the recording node where the signal is store on the open_ephys folder.


    Returns
    -------
    Parralel plot of the spike during wake, sleep, event.
    """
    #Extraction of the recording

    nb_peak = []
    time_peak = []
    event = []
    df_count_event_av = pd.DataFrame()
    spike_wake = spike_LFP.spike_phy(phy_path, fs= fsd, chunk=chunk_wake)
    df_count_wake = spike_wake['cluster'].value_counts() # change to recovery of spike
    df_count_wake = df_count_wake/(2*(chunk_wake[2]-chunk_wake[1])) # Need to get the wake duration
    df_count_wake = df_count_wake.sort_index()

    spike_sleep = spike_LFP.spike_phy(phy_path,fs=fsd, chunk=chunk_sleep)
    df_count_sleep = spike_sleep['cluster'].value_counts() #change to recovery of spike
    df_count_sleep = df_count_sleep/(2*(chunk_sleep[2]-chunk_sleep[1])) # need to get the sleep duration
    df_count_sleep = df_count_sleep.sort_index()

    session = Session(folder)
    recordnode = session.recordnodes[0]
    recording = session.recordnodes[0].recordings[0]
    data = recording.continuous[LFP]
    timing = recording.continuous[LFP].timestamps
    sample = np.size(timing)

    for j in tqdm(range(0,int(sample/fs),window)):

        start = j
        rec, timestamp = event_detection.load_openephys(folder ,fs, window, start, LFP=LFP)

        rec = event_detection.preprocessing(rec)
        rec = event_detection.average_over_channel(rec, 0)

        # load the data
        duration,  start_idx,  end_idx,  peak_idx,  start_time,  end_time,  peak_time,  peak_amp = event_detection.detection(rec, timestamp ,fs, show = False)

        nb_peak.append(np.size( peak_time))
        for time_idx in  peak_time:
            spike_sleep = spike_LFP.spike_phy(phy_path,fs=fsd, chunk=(True,time_idx-0.2,time_idx+0.3))
            df_count_event = spike_sleep['cluster'].value_counts()
            df_count_event = df_count_event.sort_index()
            df_count_event_av = pd.merge(df_count_event_av, df_count_event, how = 'outer',right_index = True, left_index = True )

    # Initialize plot
    df_event_count_sum = df_count_event_av.sum(axis = 1)
    nb = sum(nb_peak)
    df_event_count_freq = df_event_count_sum / nb
    from pandas.plotting import parallel_coordinates
    df_plot = pd.merge(df_count_wake, df_count_sleep, how = 'inner',right_index = True, left_index = True,suffixes = ('_wake',  '_sleep') )
    df_plot['Cluster_event']= df_event_count_freq
    df_plot['Index']= df_plot.index


    #df_plot = pd.merge(df_plot, df_event_count_freq, how = 'inner',right_index = True, left_index = True , suffixes = ('',  '_event'))
    parallel_coordinates(df_plot,'Index', colormap=plt.get_cmap("Set2"), axvlines=True)
    plt.show()