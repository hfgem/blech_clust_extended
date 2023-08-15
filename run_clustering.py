#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:50:42 2023

@author: Hannah Germaine
This python script is written for Mac to be able to run through spike sorting 
of long recordings with time outside of taste delivery. It runs through 
pre-processing, common-average-referencing, clustering of just the taste 
delivery interval, template matching clustered spikes from the taste delivery
interval, and then final post-processing of spikes pulled from the full recording.
If stopped for any reason, the program will pick up where it left off using the state_tracker.csv file.
"""

import os, subprocess, easygui, csv
import multiprocessing

def cont_func(next_step):
	ask_loop = 1
	while ask_loop == 1:
		print('Spike sorting is at a good stopping point.')
		print('You can continue or quit here and resume at the same spot by running run_clustering.py.')
		keep_going = input('Would you like to continue to ' + next_step + ' [y/n]? ')
		keep_going = keep_going.lower()
		try:
			if (keep_going == 'y') or (keep_going == 'n'):
				ask_loop = 0
			else:
				print("Please try again. Incorrect entry.")
		except:
			print("Please try again. Incorrect entry.")
	return keep_going

def run_blech_process(e_i):
	process_e = 'python blech_process.py ' + str(e_i)
	os.system(process_e)

if __name__ == '__main__':
	# Ask the user for the directory to save the video files in
	print('Select the directory of raw recording files.')
	directory = easygui.diropenbox(msg='Select the directory of raw recording files.', title='Select directory')
	try:
		with open(os.path.join(directory, 'state_tracker.csv'), newline='') as f:
			reader = csv.reader(f)
			state_list = list(reader)
		state_val = int(state_list[0][0])
	except:
		state_val = 0
		with open(os.path.join(directory, 'state_tracker.csv'), 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows([[state_val]])
	
	if state_val == 0:
		#Run experiment info code
		print('Running blech_exp_info.py')
		run_blech_exp_info = 'python blech_exp_info.py ' + directory
		os.system(run_blech_exp_info)
		state_val += 1
		with open(os.path.join(directory, 'state_tracker.csv'), 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows([[state_val]])
		keep_going = cont_func('set directories and import data to .h5 file')
		if keep_going == 'n':
			quit()
			
	if state_val == 1:
		#Run blech_clust.py
		print('Running blech_clust.py')
		run_blech_clust = 'python blech_clust.py ' + directory
		os.system(run_blech_clust)
		state_val += 1
		with open(os.path.join(directory, 'state_tracker.csv'), 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows([[state_val]])
		keep_going = cont_func('run common average referencing (maybe)')
		if keep_going == 'n':
			quit()
			
	if state_val == 2:
		#Run common average reference (if desired)
		ask_loop = 1
		while ask_loop == 1:
			run_car = input('Would you like to run common average referencing on this dataset [y/n]? ')
			run_car = run_car.lower()
			try:
				if (run_car == 'y') or (run_car == 'n'):
					ask_loop = 0
				else:
					print("Please try again. Incorrect entry.")
			except:
				print("Please try again. Incorrect entry.")
		if run_car == 'y':
			print('Running CAR')
			run_car = 'python blech_common_avg_reference.py ' + directory
			os.system(run_car)
		state_val += 1
		with open(os.path.join(directory, 'state_tracker.csv'), 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows([[state_val]])
		keep_going = cont_func('cluster the taste interval')
		if keep_going == 'n':
			quit()
			
	if state_val == 3:
		#Run clustering of taste interval
		with open(os.path.join(directory, 'electrode_list.csv'), newline='') as f:
			reader = csv.reader(f)
			e_list = list(reader)
		e_list = e_list[0][0].split(' ')
		electrode_inds = [int(e_list[e_i]) for e_i in range(len(e_list))]
		print('Now beginning taste interval clustering of electrodes: ')
		print(electrode_inds)
		pool = multiprocessing.Pool(4)
		pool.map(run_blech_process,electrode_inds)
	# 	for e_i in electrode_inds:
	# 		process_e = 'python blech_process.py ' + str(e_i)
	# 		os.system(process_e)
		state_val += 1
		with open(os.path.join(directory, 'state_tracker.csv'), 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows([[state_val]])
		keep_going = cont_func('pull out good taste spikes')
		if keep_going == 'n':
			quit()
			
	if state_val == 4:
		#Selecting good taste interval clusters
		print('Time to select good taste interval waveforms!')
		run_post_process = 'python blech_post_process.py -d ' + directory + ' -t True'
		os.system(run_post_process)
		state_val += 1
		with open(os.path.join(directory, 'state_tracker.csv'), 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows([[state_val]])
		keep_going = cont_func('find spikes in full recording')
		if keep_going == 'n':
			quit()
			
	if state_val == 5:
		#Clustering the full recordings of those waveforms
		#NOTE: this will overwrite the taste interval clustering results (plots, .npy files, etc...)
		print('Now clustering full recordings')
		run_process_full = 'python blech_process_full_recording.py -d ' + directory
		os.system(run_process_full)
		state_val += 1
		with open(os.path.join(directory, 'state_tracker.csv'), 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows([[state_val]])
		keep_going = cont_func('pull out good spikes')
		if keep_going == 'n':
			quit()
			
	if state_val == 6:
		#Finally clustering full recordings
		print('Time to select good full recording waveforms!')
		run_post_process = 'python blech_post_process.py -d ' + directory + ' -t False'
		os.system(run_post_process)
		state_val += 1
		with open(os.path.join(directory, 'state_tracker.csv'), 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows([[state_val]])
		keep_going = cont_func('test unit similarity')
		if keep_going == 'n':
			quit()
			
	if state_val == 7:
		#Test unit similarity
		print("Testing unit similarity")
		run_unit_similarity = 'python blech_units_similarity.py ' + directory
		os.system(run_unit_similarity)
		print("Go take a look at the similar units and determine if you need to clean up the file.")
		ask_loop = 1
		while ask_loop == 1:
			reorg = input('Did you remove units and need to reorganize [y/n]? ')
			reorg = reorg.lower()
			try:
				if (reorg == 'y') or (reorg == 'n'):
					ask_loop = 0
				else:
					print("Please try again. Incorrect entry.")
			except:
				print("Please try again. Incorrect entry.")
		if reorg == 'y':
			organize_units = 'python blech_units_organize.py ' + directory
			os.system(organize_units)
		state_val += 1
		with open(os.path.join(directory, 'state_tracker.csv'), 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows([[state_val]])
		keep_going = cont_func('plot final units')
		if keep_going == 'n':
			quit()
			
	if state_val == 8:
		#Plot the units
		print("Now plotting final units.")
		plot_units = 'python blech_units_plot.py ' + directory
		os.system(plot_units)
		state_val += 1
		with open(os.path.join(directory, 'state_tracker.csv'), 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows([[state_val]])
		keep_going = cont_func('make arrays and plot PSTHs')
		if keep_going == 'n':
			quit()
			
	if state_val == 9:
		#Make arrays and psths
		print("Now making arrays and plotting PSTHs")
		make_arrays = 'python blech_make_arrays.py ' + directory
		os.system(make_arrays)
		make_psths = 'python blech_make_psth.py ' + directory
		os.system(make_psths)
		state_val += 1
		with open(os.path.join(directory, 'state_tracker.csv'), 'w') as f:
			# using csv.writer method from CSV package
			write = csv.writer(f)
			write.writerows([[state_val]])
		print("All done!")