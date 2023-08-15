#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:49:13 2023

@author: Hannah Germaine
This python script is meant to be run after initial clustering to pull out 
clustered waveforms from the taste interval and use them to separate out more
from the rest of the recording.
"""

import os, argparse, tables, sys
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import pylab as plt
from utils import memory_monitor as mm
from utils.blech_utils import (
    imp_metadata,
)
import utils.blech_process_utils as bpu
import utils.spike_template as st
import utils.blech_spike_features as bsf

# Set seed to allow inter-run reliability
# Also allows reusing the same sorting sheets across runs
np.random.seed(0)

parser = argparse.ArgumentParser(description = 'Spike extraction from full recording')
parser.add_argument('--dir-name',  '-d', help = 'Directory containing data files')
args = parser.parse_args()

############################################################
# Load Data
############################################################

if args.dir_name is not None: 
    metadata_handler = imp_metadata([[],args.dir_name])
else:
    metadata_handler = imp_metadata([])
dir_name = metadata_handler.dir_name
params_dict = metadata_handler.params_dict
#dir_name = easygui.diropenbox()
os.chdir(dir_name)
file_list = metadata_handler.file_list
hdf5_name = metadata_handler.hdf5_name
# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# Read the hdf5 file for sorted units and the electrode numbers they are on
unit_descriptor = hf5.root.unit_descriptor
num_units = len(unit_descriptor)
electrode_nums = [unit_descriptor[u_i][1] for u_i in range(num_units)]
hf5.close()
############################################################
# Template match waveforms
############################################################

#Grab waveform properties
sampling_rate = params_dict['sampling_rate']
spike_snapshot_before = params_dict['spike_snapshot_before'] #in ms
samples_before = int(sampling_rate/1000*spike_snapshot_before)

for u_i in range(num_units):
	print('\nFinding all waveforms for clustered unit #' + str(u_i+1))
	#Grab sorted unit information
	unit_number =  u_i
	electrode_num = electrode_nums[u_i]
	print('\tElectrode Number = ' + str(electrode_num))
	
	dir_list = [f'./Plots/unit_{unit_number:02}',
	            f'./spike_waveforms/unit_{unit_number:02}',
	            f'./spike_times/unit_{unit_number:02}',
	            f'./clustering_results/unit_{unit_number:02}']
	for this_dir in dir_list:
	    bpu.ifisdir_rmdir(this_dir)
	    os.makedirs(this_dir)
	
	#Load electrode
	electrode = bpu.electrode_handler(
	                  metadata_handler.hdf5_name,
	                  electrode_num,
	                  params_dict,
					  taste=False,
					  unit_num=unit_number)

	electrode.filter_electrode()

	# Calculate the 3 voltage parameters
	electrode.cut_to_int_seconds()
	electrode.calc_recording_cutoff()

	# Dump a plot showing where the recording was cut off at
	electrode.make_cutoff_plot()

	# Then cut the recording accordingly
	electrode.cutoff_electrode()
	
	# Extract spike times and waveforms from filtered data
	spike_set = bpu.spike_handler(metadata_handler.hdf5_name,
								  electrode.filt_el, 
	                              params_dict, dir_name, 
								  electrode_num, taste=False,
								  unit_num=unit_number)
	spike_set.extract_waveforms()
	
	# Extract windows from filt_el and plot with threshold overlayed
	window_len= 0.2  # sec
	window_count= 10
	fig= bpu.gen_window_plots(
	    electrode.filt_el,
	    window_len,
	    window_count,
	    params_dict['sampling_rate'],
	    spike_set.spike_times,
	    spike_set.mean_val,
	    spike_set.threshold,
	)
	fig.savefig(f'./Plots/unit_{unit_number:02}/bandpass_trace_snippets.png',
		            bbox_inches='tight', dpi=300)
	plt.close(fig)
	
	# Delete filtered electrode from memory
	del electrode
	
	# Dejitter these spike waveforms, and get their maximum amplitudes
	# Slices are returned sorted by amplitude polarity
	spike_set.dejitter_spikes()
	
	spike_set.extract_amplitudes()
	
	spike_set.extract_features(
            bsf.feature_pipeline,
            bsf.feature_names,
            fitted_transformer=False,
            )
	
	spike_set.write_out_spike_data()
	
	for cluster_num in range(params_dict['min_clusters'], params_dict['max_clusters']+1):
	    cluster_handler = bpu.cluster_handler(
	            params_dict, 
	            dir_name, 
	            electrode_num,
	            cluster_num,
	            spike_set,
				taste=False,
				unit_num=unit_number)
	    cluster_handler.perform_prediction()
	    cluster_handler.remove_outliers(params_dict)
	    cluster_handler.save_cluster_labels()
	    cluster_handler.create_output_plots( 
	                            params_dict)

	# Make file for dumping info about memory usage
	f= open(f'./memory_monitor_clustering/{unit_number:02}.txt', 'w')
	print(mm.memory_usage_resource(), file=f)
	f.close()
	print(f'Unit {unit_number} complete.\n')

# Now delete the sorted taste units
try:
    hf5.remove_node('/sorted_units', recursive = 1)
except:
	print('No taste interval sorted units to remove')
