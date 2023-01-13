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

def spike_phy(phy_path, channel_treshold=384 ,fs= 30000, chunk=(False,0,0)):

    """Import phy_output from a kilosort spikesorting.

    Parameters
    ----------
    phy_path : str or path to the phy_output for Kilosort
        (Array over 1 channel or average across several channel).
    channel_treshold : positive integer, optionnal (default = 384)
        channel treshold to look below.
    fs : positive integer, optional (default = 30 000)
        frequency of the recording where the spikesorting where perform.
    chunk : tuple of q bool and 2 positive integer, optional (default = (False,0,0))
        The boolean is to know if we kneed to chunk,
        the first interger is the start of the chunk,
        the second is the end of the chunk.
    end : bool, optional (default = True)
        if True (1), plot data in matplotlib figure.

    Returns
    -------
    df_event : pandas dataframe
        spike
    """

    spike_cluster = np.load(phy_path+'/spike_clusters.npy')
    spike_time = np.load(phy_path+'/spike_times.npy')

    save_path = phy_path+"/cluster_info.tsv"

    df = pd.read_csv(save_path, sep="\t")   # read dummy .tsv file into memory
    df = df.where(df['KSLabel']=='good')
    df = df.where(df['ch']<channel_treshold)

    df = df.dropna(how='all')
    df_spike_cluster = pd.DataFrame(spike_cluster, columns=['Cluster'])
    df_spike_time = pd.DataFrame(spike_time, columns=['time'])
    df_spike = pd.DataFrame()
    df_spike['Cluster']=df_spike_cluster
    df_spike['time']=df_spike_time/fs
    df_merge = pd.merge(df_spike, df,left_on='Cluster',right_on='cluster_id')
    if chunk[0]:
        df_merge = df_merge.where(df_merge['time']>chunk[1])
        df_merge = df_merge.dropna(how='all')
        df_merge = df_merge.where(df_merge['time']<chunk[2])
        df_merge = df_merge.dropna(how='all')
   
    df_event = pd.DataFrame()
    df_event['time'] = df_merge['time']
    df_event['cluster'] = df_merge['Cluster']

    return df_event


def raster(df_event):

    """Rearrange pandas dataframe for raster plot

    Parameters
    ----------
    df_event : pandas dataframe
        spike
    Returns
    -------
    df_event : pandas dataframe
        rearrange pandas dataframe for the raster plot
    """
    df_event.sort_values(by = ['cluster'])
    dfgroup = df_event.groupby(['cluster']).mean()
    for l,(index,row) in zip(range(dfgroup['time'].size), dfgroup.iterrows()):
        dfgroup.loc[index,'new_cluster']=l
    df_raster = pd.merge(df_event, dfgroup,how='inner',left_on='cluster',right_on='cluster')
    df_count_event = df_raster['cluster'].value_counts()
    return df_raster

def import_ttl(TTL_path):

    """Import ttl for the open_ephys recording.

    Parameters
    ----------
    TTL_path : str or path to the TTL 

    Returns
    -------
    df_odor : pandas dataframe
        Odor_stim according to neuropixel timestamp.
    """

    fullwords = np.load(TTL_path+'/full_words.npy')
    sample_numbers = np.load(TTL_path+'/sample_numbers.npy')
    states = np.load(TTL_path+'/states.npy')
    timestamps = np.load(TTL_path+'/timestamps.npy')

    df_odor = pd.DataFrame(sample_numbers/fsd, columns=['timestamps'])
    df_odor['states'] = fullwords
    #df_odor = df_odor.where(df_odor['timestamps']>st)
    #df_odor = df_odor.dropna(how='all')
    #df_odor = df_odor.where(df_odor['timestamps']<(st+(wind/fs)))
    #df_odor = df_odor.dropna(how='all')

    print(df_odor)
