#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:58:29 2022
Data manipulation with bandpass filtering, etc... to pull out cleaner spikes
@author: hannahgermaine
"""
import numpy as np
from scipy.signal import butter, filtfilt
import tqdm


def butter_lowpass(cutoff, fs, order=5):
	"""Function per https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units"""
	return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
	"""Function per https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units"""
	b, a = butter_lowpass(cutoff, fs, order=order)
	y = filtfilt(b, a, data)
	return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    low = lowcut
    high = highcut
    b, a = butter(order, [low, high], btype='bandpass', fs=fs, output='ba')
    return b, a

def bandpass_filter(data_segment, lowcut, highcut, fs, order=5):
	"""Function to bandpass filter. Calls butter_bandpass function.
	Copied from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
	Scipy documentation for bandpass filter
	data_segment = data to be filtered
	lowcut = low frequency
	highcut = high frequency
	fs = sampling rate of data"""
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = filtfilt(b, a, data_segment)
	return y

def signal_averaging(data):
	"""Function to increase the signal-to-noise ratio by removing common signals
	across electrodes"""
	
	total_time = len(data[0])
	chunk = int(np.ceil(total_time/10000))
	start_times = np.arange(stop = total_time,step = chunk)
	cleaned_data = np.zeros(np.shape(data))
	for t in tqdm.tqdm(range(len(start_times))):
		s_t = start_times[t]
		data_chunk = data[:,s_t:s_t+chunk]
		#Median Subtraction
		#med = np.median(data_chunk,0)
		#cleaned_chunk = data_chunk - med
		#Mean Subtraction
		mean = np.mean(data_chunk,0)
		cleaned_chunk = data_chunk - mean
		cleaned_data[:,s_t:s_t+chunk] = cleaned_chunk
	
	return cleaned_data

