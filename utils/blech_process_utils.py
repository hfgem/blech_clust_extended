
import utils.clustering as clust
from utils import blech_waveforms_datashader
# import subprocess
from joblib import load
from sklearn.mixture import GaussianMixture as gmm
from sklearn.cluster import KMeans
import subprocess
from scipy.stats import zscore
import pylab as plt
import json
# import sys
import numpy as np
import tables
import os
import shutil
import matplotlib
import pandas as pd
matplotlib.use('Agg')

############################################################
# Define Functions
############################################################


class path_handler():
	
	def __init__(self):
		self.home_dir = os.getenv('HOME')
		file_path = os.path.abspath(__file__)
		blech_clust_dir =  ('/').join(file_path.split('/')[:-2])
		self.blech_clust_dir = blech_clust_dir

class cluster_handler():
	"""
	Class to handle clustering steps
	"""

	def __init__(self, params_dict,
				 data_dir, electrode_num, cluster_num,
				 spike_set, taste=False, unit_num=-1):
		self.params_dict = params_dict
		self.dat_thresh = 10e3
		self.data_dir = data_dir
		self.electrode_num = electrode_num
		self.cluster_num = cluster_num
		self.spike_set = spike_set
		self.taste = taste
		self.unit_num = unit_num
		self.create_output_dir()

	def return_training_set(self, data):
		"""
		Return training set for clustering
		"""
		train_set = data[
			np.random.choice(np.arange(data.shape[0]),
							 int(np.min((data.shape[0], self.dat_thresh))))]
		return train_set

	def fit_model(self, train_set, clusters):
		"""
		Cluster waveforms
		"""
		
		if self.params_dict['clust_type'] == 'kmeans':
			model = KMeans(n_clusters=clusters, 
						   random_state=np.random.randint(100),
						   n_init= self.params_dict['num_restarts'],
						   max_iter=self.params_dict['num_iter'],
						   tol=self.params_dict['thresh']).fit(train_set)
		else:
			model = gmm(
				n_components=clusters,
				max_iter=self.params_dict['num_iter'],
				n_init=self.params_dict['num_restarts'],
				tol=self.params_dict['thresh']).fit(train_set)
		
		return model

	def get_cluster_labels(self, data, model):
		"""
		Get cluster labels
		"""
		return model.predict(data)

	def perform_prediction(self):
		full_data = self.spike_set.spike_features
		train_set = self.return_training_set(full_data)
		model = self.fit_model(train_set, self.cluster_num)
		labels = self.get_cluster_labels(full_data, model)
		self.labels = labels

	def remove_outliers(self, params_dict):
		"""
		Clear large waveforms
		"""
		# Sometimes large amplitude noise waveforms cluster with the
		# spike waveforms because the amplitude has been factored out of
		# the scaled slices.
		# Run through the clusters and find the waveforms that are more than
		# wf_amplitude_sd_cutoff larger than the cluster mean.
		# Set predictions = -1 at these points so that they aren't
		# picked up by blech_post_process
		wf_amplitude_sd_cutoff = params_dict['wf_amplitude_sd_cutoff']
		for cluster in np.unique(self.labels):
			cluster_points = np.where(self.labels[:] == cluster)[0]
			this_cluster = remove_too_large_waveforms(
				cluster_points,
				self.spike_set.amplitudes,
				self.labels,
				wf_amplitude_sd_cutoff)
			self.labels[cluster_points] = this_cluster

	def save_cluster_labels(self):
		np.save(
			os.path.join(
				self.clust_results_dir, 'predictions.npy'),
			self.labels)

	def create_output_dir(self):
		if self.taste == True:
			# Make folder for results of i+2 clusters, and store results there
			clust_results_dir = os.path.join(
				self.data_dir,
				'clustering_results_taste',
				f'electrode{self.electrode_num:02}',
				f'clusters{self.cluster_num}'
			)
			clust_plot_dir = os.path.join(
				self.data_dir,
				'Plots_taste',
				f'{self.electrode_num:02}',
				f'clusters{self.cluster_num}'
			)
		else:
			# Make folder for results of i+2 clusters, and store results there
			clust_results_dir = os.path.join(
				self.data_dir,
				'clustering_results',
				f'unit_{self.unit_num:02}',
				f'clusters{self.cluster_num}'
			)
			clust_plot_dir = os.path.join(
				self.data_dir,
				'Plots',
				f'unit_{self.unit_num:02}',
				f'clusters{self.cluster_num}'
			)
		ifisdir_rmdir(clust_results_dir)
		ifisdir_rmdir(clust_plot_dir)
		os.makedirs(clust_results_dir)
		os.makedirs(clust_plot_dir)
		self.clust_results_dir = clust_results_dir
		self.clust_plot_dir = clust_plot_dir

	def create_output_plots(self,
							params_dict):

		slices_dejittered = self.spike_set.slices_dejittered
		times_dejittered = self.spike_set.times_dejittered
		standard_data = self.spike_set.spike_features
		feature_names = self.spike_set.feature_names
		threshold = self.spike_set.threshold
		# Create file, and plot spike waveforms for the different clusters.
		# Plot 10 times downsampled dejittered/smoothed waveforms.
		# Additionally plot the ISI distribution of each cluster
		x = np.arange(len(slices_dejittered[0])) + 1
		for cluster in np.unique(self.labels):
			cluster_points = np.where(self.labels == cluster)[0]

			if len(cluster_points) > 0:
				# downsample = False, Prevents waveforms_datashader
				# from FURTHER downsampling the given waveforms for plotting
				# Because in the previous version they were upsampled for clustering

				# Create waveform datashader plot
				#############################
				fig, ax = gen_datashader_plot(
					slices_dejittered,
					cluster_points,
					x,
					threshold,
					self.electrode_num,
					params_dict['sampling_rate'],
					cluster,
					self.taste
				)
				fig.savefig(os.path.join(
					self.clust_plot_dir, f'Cluster{cluster}_waveforms'))
				plt.close("all")

				# Create ISI distribution plot
				#############################
				fig = gen_isi_hist(
					times_dejittered,
					cluster_points,
				)
				fig.savefig(os.path.join(
					self.clust_plot_dir, f'Cluster{cluster}_ISIs'))
				plt.close("all")

				# Create features timeseries plot
				# And plot histogram of spiketimes
				#############################
				fig, ax = feature_timeseries_plot(
					standard_data,
					times_dejittered,
					feature_names,
					cluster_points
				)
				fig.suptitle(f'Cluster {cluster} features')
				fig.savefig(os.path.join(
					self.clust_plot_dir, f'Cluster{cluster}_features'))
				plt.close(fig)

			else:
				# Write out file that somehow there are no spikes
				#############################
				file_path = os.path.join(
					self.clust_plot_dir, f'no_spikes_Cluster{cluster}')
				with open(file_path, 'w') as file_connect:
					file_connect.write('')

class electrode_handler():
	"""
	Class to handle electrode data
	"""

	def __init__(self, hdf5_path, electrode_num, params_dict, taste=False, unit_num=-1):
		self.hdf5_path = hdf5_path
		self.electrode_num = electrode_num
		self.params_dict = params_dict
		self.taste = taste
		self.unit_num = unit_num

		hf5 = tables.open_file(hdf5_path, 'r')
		if self.taste==True:
			el_path = f'/raw_taste/electrode{electrode_num:02}'
		else:
			el_path = f'/raw/electrode{electrode_num:02}'
		if el_path in hf5:
			self.raw_el = hf5.get_node(el_path)[:]
		else:
			raise Exception(f'{el_path} not in HDF5')
		hf5.close()

	def filter_electrode(self):
		self.filt_el = clust.get_filtered_electrode(
			self.raw_el,
			freq=[self.params_dict['bandpass_lower_cutoff'],
				  self.params_dict['bandpass_upper_cutoff']],
			sampling_rate=self.params_dict['sampling_rate'],)
		
		# Delete raw electrode recording from memory to save space
		del self.raw_el

	def adjust_to_sampling_rate(data, sampling_rate):
		return

	def cut_to_int_seconds(self):
		"""
		Cut data to have integer number of seconds

		data: numpy array
		sampling_rate: int
		"""
		data = self.filt_el
		sampling_rate = self.params_dict['sampling_rate']
		self.filt_el = data[:int(sampling_rate)*int(len(data)/sampling_rate)]

	def calc_recording_cutoff(self):
		keywords = (
			'filt_el',
			'sampling_rate',
			'voltage_cutoff',
			'max_breach_rate',
			'max_secs_above_cutoff',
			'max_mean_breach_rate_persec'
		)
		values = (
			self.filt_el,
			self.params_dict['sampling_rate'],
			self.params_dict['voltage_cutoff'],
			self.params_dict['max_breach_rate'],
			self.params_dict['max_secs_above_cutoff'],
			self.params_dict['max_mean_breach_rate_persec'],
		)
		kwarg_dict = dict(zip(keywords, values))
		(
			breach_rate,
			breaches_per_sec,
			secs_above_cutoff,
			mean_breach_rate_persec,
			recording_cutoff
		) = return_cutoff_values(**kwarg_dict)

		self.recording_cutoff = recording_cutoff

	def make_cutoff_plot(self):
		"""
		Makes a plot showing where the recording was cut off at

		filt_el: numpy array
		recording_cutoff: int
		"""
		fig = plt.figure()
		second_data = np.reshape(
			self.filt_el,
			(-1, self.params_dict['sampling_rate']))
		plt.plot(np.mean(second_data, axis=1))
		plt.axvline(self.recording_cutoff,
					color='k', linewidth=4.0, linestyle='--')
		plt.xlabel('Recording time (secs)')
		plt.ylabel('Average voltage recorded per sec (microvolts)')
		plt.title(f'Recording length : {len(second_data)}s' + '\n' +
				  f'Cutoff time : {self.recording_cutoff}s')
		if self.taste == True:
			fig.savefig(
				f'./Plots_taste/{self.electrode_num:02}/cutoff_time.png',
				bbox_inches='tight')
		else:
			fig.savefig(
				f'./Plots/unit_{self.unit_num:02}/cutoff_time.png',
				bbox_inches='tight')
		plt.close("all")

	def cutoff_electrode(self):
		# TODO: Add warning if recording cutoff before the end
		# Warning should be printed out to file AND printed
		self.filt_el = self.filt_el[:self.recording_cutoff *
									self.params_dict['sampling_rate']]


class spike_handler():
	"""
	Class to handler processing of spikes
	"""

	def __init__(self, hdf5_path, filt_el, params_dict, dir_name, electrode_num, taste=False, unit_num=None):
		self.hdf5_path = hdf5_path
		self.filt_el = filt_el
		self.params_dict = params_dict
		self.dir_name = dir_name
		self.electrode_num = electrode_num
		self.taste = taste
		self.unit_num = unit_num

	def extract_waveforms(self):
		"""
		Extract waveforms from filtered electrode
		"""
		if self.taste == True:
			waveform_dir = f'{self.dir_name}/spike_waveforms_taste/electrode{self.electrode_num:02}'
			slices, spike_times, polarity, mean_val, threshold = \
					clust.extract_waveforms_hannah(
							self.filt_el,
							dir_name=waveform_dir,
							spike_snapshot=[self.params_dict['spike_snapshot_before'],
										 self.params_dict['spike_snapshot_after']],
							sampling_rate=self.params_dict['sampling_rate'],
							threshold_mult=self.params_dict['waveform_threshold'])
		else:
			waveform_dir = f'{self.dir_name}/spike_waveforms/unit_{self.unit_num:02}'
			#Grab relevant parameters
			sampling_rate = self.params_dict['sampling_rate']
			spike_snapshot_before = self.params_dict['spike_snapshot_before'] #in ms
			samples_before = int(sampling_rate/1000*spike_snapshot_before)
			#Grab waveforms from pre-clustered taste interval
			hf5 = tables.open_file(self.hdf5_path, 'r+')
			unit_descriptor = hf5.root.unit_descriptor
			sorted_nodes = hf5.list_nodes('/sorted_units')
			waveforms = sorted_nodes[self.unit_num].waveforms[:]
			hf5.close()
			#Pull out template waveform and threshold values
			mean_waveform = np.mean(waveforms[:],0)
			std_waveform = np.std(waveforms[:],0)
			mean_peak = mean_waveform[samples_before]
			std_peak = std_waveform[samples_before]
			if mean_peak < 0:
				thresh_min = mean_peak - 3*std_peak
				thresh_max = mean_peak + std_peak
			elif mean_peak > 0:
				thresh_min = mean_peak - std_peak
				thresh_max = mean_peak + 3*std_peak
			slices, spike_times, polarity = \
				clust.template_match_waveforms(self.filt_el,
							spike_template=mean_waveform,
							dir_name=waveform_dir,
							spike_snapshot=[self.params_dict['spike_snapshot_before'],
										 self.params_dict['spike_snapshot_after']],
							sampling_rate=self.params_dict['sampling_rate'],
							thresh_min = thresh_min,
							thresh_max = thresh_max,
							cut_percentile=self.params_dict['template_percentile'])
			mean_val = 0
			threshold = max([np.abs(thresh_max),np.abs(thresh_min)])
		self.slices = slices
		self.spike_times = spike_times
		self.polarity = polarity
		self.mean_val = mean_val
		self.threshold = threshold

	def dejitter_spikes(self):
		"""
		Dejitter spikes
		"""
		slices_dejittered = self.slices
		times_dejittered = self.spike_times
		
		# Sort data by time
		spike_order = np.argsort(times_dejittered)
		times_dejittered = times_dejittered[spike_order]
		slices_dejittered = slices_dejittered[spike_order]
		polarity = self.polarity[spike_order]

		self.slices_dejittered = slices_dejittered
		self.times_dejittered = times_dejittered
		self.polarity = polarity
		del self.slices
		del self.spike_times

	def extract_amplitudes(self):
		"""
		Extract amplitudes from dejittered spikes
		"""
		zero_ind = self.params_dict['spike_snapshot_before'] *\
			self.params_dict['sampling_rate']/1000
		zero_ind = int(zero_ind)
		self.amplitudes = np.array([self.slices_dejittered[i, zero_ind] for i in range(len(self.slices_dejittered))])

	def extract_features(self,
						 feature_transformer,
						 feature_names,
						 fitted_transformer = True):

		self.feature_names = feature_names
		if fitted_transformer:
			self.spike_features = feature_transformer.transform(
				self.slices_dejittered, self.dir_name)
		else:
			self.spike_features = feature_transformer.fit_transform(
				self.slices_dejittered, self.dir_name)

	def write_out_spike_data(self):
		"""
		Save the pca_slices, energy and amplitudes to the
		spike_waveforms folder for this electrode
		Save slices/spike waveforms and their times to their respective folders
		"""
		to_be_saved = ['slices_dejittered',
					   'pca_slices',
					   'energy',
					   'amplitude',
					   'times_dejittered']

		slices_dejittered = self.slices_dejittered
		times_dejittered = self.times_dejittered
		pca_inds = [i for i, x in enumerate(self.feature_names) if 'pca' in x]
		pca_slices = self.spike_features[:, pca_inds]
		energy_inds = [i for i, x in enumerate(self.feature_names) if 'energy' in x]
		energy = self.spike_features[:, energy_inds]
		amp_inds = [i for i, x in enumerate(self.feature_names) if 'amplitude' in x]
		amplitude = self.spike_features[:,amp_inds]
		
		if self.taste == True:
			waveform_dir = f'{self.dir_name}/spike_waveforms_taste/electrode{self.electrode_num:02}'
			spiketime_dir = f'{self.dir_name}/spike_times_taste/electrode{self.electrode_num:02}'
			save_paths = [f'{waveform_dir}/spike_waveforms.npy',
						  f'{waveform_dir}/pca_waveforms.npy',
						  f'{waveform_dir}/energy.npy',
						  f'{waveform_dir}/spike_amplitudes.npy',
						  f'{spiketime_dir}/spike_times.npy',
						  ]
		else:
			waveform_dir = f'{self.dir_name}/spike_waveforms/unit_{self.unit_num:02}'
			spiketime_dir = f'{self.dir_name}/spike_times/unit_{self.unit_num:02}'
			save_paths = [f'{waveform_dir}/spike_waveforms.npy',
						  f'{waveform_dir}/pca_waveforms.npy',
						  f'{waveform_dir}/energy.npy',
						  f'{waveform_dir}/spike_amplitudes.npy',
						  f'{spiketime_dir}/spike_times.npy',
						  ]

		for key, path in zip(to_be_saved, save_paths):
			np.save(path, locals()[key])

def ifisdir_rmdir(dir_name):
	if os.path.isdir(dir_name):
		shutil.rmtree(dir_name)


def return_cutoff_values(
	filt_el,
	sampling_rate,
	voltage_cutoff,
	max_breach_rate,
	max_secs_above_cutoff,
	max_mean_breach_rate_persec
):

	breach_rate = float(len(np.where(filt_el > voltage_cutoff)[0])
						* int(sampling_rate))/len(filt_el)
	#test_el is #seconds x #samples per seconds reshaping of filtered data
	test_el = np.reshape(filt_el, (-1, sampling_rate)) #-1 is an "unspecified value" which is inferred
	breaches_per_sec = (test_el > voltage_cutoff).sum(axis=-1)
	secs_above_cutoff = (breaches_per_sec > 0).sum()
	if secs_above_cutoff == 0:
		mean_breach_rate_persec = 0
	else:
		mean_breach_rate_persec = np.mean(breaches_per_sec[
			breaches_per_sec > 0])

	# And if they all exceed the cutoffs,
	# assume that the headstage fell off mid-experiment
	recording_cutoff = int(len(filt_el)/sampling_rate)
	if breach_rate >= max_breach_rate and \
			secs_above_cutoff >= max_secs_above_cutoff and \
			mean_breach_rate_persec >= max_mean_breach_rate_persec:
		# Find the first 1 second epoch where the number of cutoff breaches
		# is higher than the maximum allowed mean breach rate
		recording_cutoff = np.where(breaches_per_sec >
									max_mean_breach_rate_persec)[0][0]

	return (breach_rate, breaches_per_sec, secs_above_cutoff,
			mean_breach_rate_persec, recording_cutoff)


def gen_window_plots(
	filt_el,
	window_len,
	window_count,
	sampling_rate,
	spike_times,
	mean_val,
	threshold,
):
	windows_in_data = len(filt_el) // (window_len * sampling_rate)
	window_markers = np.linspace(0,
								 int(windows_in_data*(window_len * sampling_rate)),
								 int(windows_in_data))
	window_markers = np.array([int(x) for x in window_markers])
	chosen_window_inds = np.vectorize(np.int)(np.sort(np.random.choice(
		np.arange(windows_in_data), window_count)))
	chosen_window_markers = [(window_markers[x-1], window_markers[x])
							 for x in chosen_window_inds]
	chosen_windows = [filt_el[start:end]
					  for (start, end) in chosen_window_markers]
	# For each window, extract detected spikes
	chosen_window_spikes = [np.array(spike_times)
							[(spike_times > start)*(spike_times < end)] - start
							for (start, end) in chosen_window_markers]

	fig, ax = plt.subplots(len(chosen_windows), 1,
						   sharex=True, sharey=True, figsize=(10, 10))
	for dat, spikes, this_ax in zip(chosen_windows, chosen_window_spikes, ax):
		this_ax.plot(dat, linewidth=0.5)
		this_ax.hlines(mean_val + threshold, 0, len(dat))
		this_ax.hlines(mean_val - threshold, 0, len(dat))
		if len(spikes) > 0:
			this_ax.scatter(spikes, np.repeat(
				mean_val, len(spikes)), s=5, c='red')
		this_ax.set_ylim((mean_val - 1.5*threshold,
						  mean_val + 1.5*threshold))
	return fig


def gen_datashader_plot(
		slices_dejittered,
		cluster_points,
		x,
		threshold,
		electrode_num,
		sampling_rate,
		cluster,
		taste
):
	if taste == True:
		fig, ax = blech_waveforms_datashader.waveforms_datashader(
			slices_dejittered[cluster_points, :],
			x,
			downsample=False,
			threshold=threshold,
			dir_name="Plots_taste/" + "datashader_temp_el" + str(electrode_num))
	else:
		fig, ax = blech_waveforms_datashader.waveforms_datashader(
			slices_dejittered[cluster_points, :],
			x,
			downsample=False,
			threshold=threshold,
			dir_name="Plots/" + "datashader_temp_el" + str(electrode_num))

	ax.set_xlabel('Sample ({:d} samples per ms)'.
				  format(int(sampling_rate/1000)))
	ax.set_ylabel('Voltage (microvolts)')
	ax.set_title('Cluster%i' % cluster)
	return fig, ax


def gen_isi_hist(
		times_dejittered,
		cluster_points,
):
	fig = plt.figure()
	cluster_times = times_dejittered[cluster_points]
	ISIs = np.ediff1d(np.sort(cluster_times))
	ISIs = ISIs/30.0
	max_ISI_val = 20
	bin_count = 100
	neg_pos_ISI = np.concatenate((-1*ISIs, ISIs), axis=-1)
	hist_obj = plt.hist(
		neg_pos_ISI,
		bins=np.linspace(-max_ISI_val, max_ISI_val, bin_count))
	plt.xlim([-max_ISI_val, max_ISI_val])
	# Scale y-lims by all but the last value
	upper_lim = np.max(hist_obj[0][:-1])
	if upper_lim:
		plt.ylim([0, upper_lim])
	plt.title("2ms ISI violations = %.1f percent (%i/%i)"
			  % ((float(len(np.where(ISIs < 2.0)[0])) /
				  float(len(cluster_times)))*100.0,
				 len(np.where(ISIs < 2.0)[0]),
				 len(cluster_times)) + '\n' +
			  "1ms ISI violations = %.1f percent (%i/%i)"
			  % ((float(len(np.where(ISIs < 1.0)[0])) /
				  float(len(cluster_times)))*100.0,
				 len(np.where(ISIs < 1.0)[0]), len(cluster_times)))
	return fig


def remove_too_large_waveforms(
		cluster_points,
		amplitudes,
		predictions,
		wf_amplitude_sd_cutoff
):
	this_cluster = predictions[cluster_points]
	cluster_amplitudes = amplitudes[cluster_points]
	cluster_amplitude_mean = np.mean(cluster_amplitudes)
	cluster_amplitude_sd = np.std(cluster_amplitudes)
	reject_wf = np.where(cluster_amplitudes <= cluster_amplitude_mean
						 - wf_amplitude_sd_cutoff*cluster_amplitude_sd)[0]
	this_cluster[reject_wf] = -1
	return this_cluster


def feature_timeseries_plot(
		standard_data,
		times_dejittered,
		feature_names,
		cluster_points
):
	this_standard_data = standard_data[cluster_points]
	this_spiketimes = times_dejittered[cluster_points]
	fig, ax = plt.subplots(this_standard_data.shape[1] + 1, 1,
						   figsize=(7, 9), sharex=True)
	for this_label, this_dat, this_ax in \
			zip(feature_names, this_standard_data.T, ax[:-1]):
		this_ax.scatter(this_spiketimes, this_dat,
						s=0.5, alpha=0.5)
		this_ax.set_ylabel(this_label)
	ax[-1].hist(this_spiketimes, bins=50)
	ax[-1].set_ylabel('Spiketime' + '\n' + 'Histogram')
	return fig, ax
