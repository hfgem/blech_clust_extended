#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:16:18 2022

@author: hannahgermaine

This code is written to perform template-matching
"""

import os, time
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from numba import jit
import scipy.stats as ss
from tqdm import tqdm
from random import sample

def spike_template_sort(all_spikes,sort_type,sampling_rate,num_pts_left,num_pts_right,
						cut_percentile,unit_dir,
						chunk_size = 100000):
	"""This function performs template-matching to pull out potential spikes.
	INPUTS:
		- all_spikes
		- sort_type
		- sampling_rate
		- num_pts_left
		- num_pts_right
		- cut_percentile
		- unit_dir - directory of unit's storage data
		- chunk_size - data chunking for processing
	OUTPUTS:
		- potential_spikes
		- good_ind
	"""
	template_dir = unit_dir + '/template_matching/'
	if os.path.isdir(template_dir) == False:
		os.mkdir(template_dir)
	
	#Grab templates
	print("\t Preparing Data for Template Matching")
	num_spikes = len(all_spikes)
	spike_inds = np.arange(num_spikes)
	#Break up into chunks due to large data size
	chunk_inds = np.arange(0,num_spikes,chunk_size)
	num_chunks = len(chunk_inds)
	norm_spikes = np.zeros(np.shape(all_spikes)) #Storage matrix
	num_peaks = np.zeros(num_spikes) #Storage vector
	for c_i in tqdm(range(num_chunks)):
		#Normalize data and grab number of peaks
		chunk_start = chunk_inds[c_i]
		chunk_end = min(chunk_start + chunk_size,num_spikes)
		norm_spikes[chunk_start:chunk_end,:],num_peaks[chunk_start:chunk_end] = norm_and_num_mat(all_spikes[chunk_start:chunk_end,:],num_pts_left)
	
	#Grab templates of spikes
	print("\t Creating new templates.")
	spike_template = generate_templates(sampling_rate,sort_type,num_pts_left,num_pts_right)
	#Template distance scores
	spike_mat = np.multiply(np.ones(np.shape(norm_spikes)),spike_template)
	dist = np.sqrt(np.sum(np.square(np.subtract(norm_spikes,spike_mat)),1))
	score = dist*num_peaks
	percentile = np.percentile(score,cut_percentile)
	cutoff_value = percentile
	new_template_waveform_ind = list(spike_inds[list(np.where(score < cutoff_value)[0])])
	new_template = np.mean(norm_spikes[new_template_waveform_ind,:],axis=0)
	#Plot a histogram of the scores and save to the template_matching dir
	print("\t Plotting and pulling best spikes.")
	fig = plt.figure(figsize=(20,20))
	#Calculate new template distance scores
	spike_mat = np.multiply(np.ones(np.shape(norm_spikes)),new_template)
	dist_2 = np.sqrt(np.sum(np.square(np.subtract(norm_spikes,spike_mat)),1))
	score_2 = dist_2*num_peaks
	percentile = np.percentile(score_2,cut_percentile)
	#Create subplot to plot histogram and percentile cutoff
	plt.subplot(2,1,1)
	plt.hist(score_2,150,label='Mean Template Similarity Scores')
	plt.axvline(percentile,color = 'r', linestyle = '--', label='Cutoff Threshold')
	plt.legend()
	plt.xlabel('Score = distance*peak_count')
	plt.ylabel('Number of occurrences')
	plt.title('Scores in comparison to template')
	plt.subplot(2,1,2)
	plt.plot(new_template)
	plt.title('Template waveform')
	good_ind = list(spike_inds[list(np.where(score_2 < percentile)[0])])
	fig.savefig(template_dir + '/template_matching_results_' + sort_type + '.png',dpi=100)
	plt.close(fig)
	potential_spikes = np.array([all_spikes[g_i] for g_i in good_ind])

	return potential_spikes, good_ind
	
@jit(forceobj=True)
def generate_templates(sampling_rate,sort_type,num_pts_left,num_pts_right):
	"""This function generates 3 template vectors of neurons with a peak 
	centered between num_pts_left and num_pts_right."""
	
	x_points = np.arange(-num_pts_left,num_pts_right)
	
	fast_spike_width = sampling_rate*(1/1000)
	sd = fast_spike_width/20
	
	if sort_type=='min':
		reg_spike_bit = ss.gamma.pdf(np.arange(fast_spike_width-1),5)
		peak_reg = find_peaks(reg_spike_bit)[0][0]
		reg_spike = np.concatenate((np.zeros(num_pts_left-peak_reg),-1*reg_spike_bit),axis=0)
		reg_spike = np.concatenate((reg_spike,np.zeros(len(x_points) - len(reg_spike))),axis=0)
		max_reg_spike = max(abs(reg_spike))
		reg_spike = reg_spike/max_reg_spike
		template = reg_spike
	elif sort_type=='max':
		pos_spike = ss.norm.pdf(x_points, 0, sd)
		max_pos_spike = max(abs(pos_spike))
		pos_spike = pos_spike/max_pos_spike
		template = pos_spike
	else:
		print('Incorrect sort type passed to template-making code.')
	
	return template


@jit(forceobj=True)
def norm_and_num_mat(mat,peak_ind):
	norm_spike_chunk =  norm_spikes_func(mat,peak_ind) #Normalize the data
	num_peak_chunk = np.apply_along_axis(num_peaks_func,1,norm_spike_chunk)
	return norm_spike_chunk, num_peak_chunk
	

@jit(forceobj=True)
def num_peaks_func(vec):
	return len(find_peaks(vec,0.3)[0]) + len(find_peaks(-1*vec,0.3)[0])


@jit(forceobj=True)
def norm_spikes_func(mat,peak_ind):
	peak_val = np.expand_dims(np.abs(mat[:,peak_ind]),1)
	norm_spike_chunk = np.divide(mat,peak_val) #Normalize the data
	return norm_spike_chunk