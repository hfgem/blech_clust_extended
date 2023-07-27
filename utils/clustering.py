import numpy as np
from scipy.signal import butter, filtfilt, fftconvolve, find_peaks
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
import pylab as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import utils.spike_template as st
from tqdm import tqdm
import random

def get_filtered_electrode(data, freq = [300.0, 3000.0], sampling_rate = 30000.0):
		el = 0.195*(data) #convert to microvolts
		m, n = butter(5, freq, btype='bandpass', fs=sampling_rate, output='ba')
		#m, n = butter(2, [2.0*freq[0]/sampling_rate, 2.0*freq[1]/sampling_rate], btype = 'bandpass') 
		filt_el = filtfilt(m, n, el)
		return filt_el

def extract_waveforms_abu(filt_el, spike_snapshot = [0.5, 1.0], 
								    sampling_rate = 30000.0,
								    threshold_mult = 5.0):

		m = np.mean(filt_el)
		th = threshold_mult*np.median(np.abs(filt_el)/0.6745)

		negative = np.where(filt_el <= m-th)[0] 
		positive = np.where(filt_el >= m+th)[0] 
		# Marking breaks in detected threshold crossings 
		neg_changes = np.concatenate(([0],np.where(np.diff(negative) > 1)[0]+1))
		pos_changes = np.concatenate(([0],np.where(np.diff(positive) > 1)[0]+1))
		
		# Mark indices to be extracted
		neg_inds = [(negative[neg_changes[x]],negative[neg_changes[x+1]-1]) \
				for x in range(len(neg_changes)-1)]
		pos_inds = [(positive[pos_changes[x]],positive[pos_changes[x+1]-1]) \
				for x in range(len(pos_changes)-1)]

		# Mark the extremum of every threshold crossing
		minima = [np.argmin(filt_el[start:(end+1)]) + start \
				for start,end in neg_inds]
		maxima = [np.argmax(filt_el[start:(end+1)]) + start \
				for start,end in pos_inds]

		polarity = np.concatenate(([-1]*len(minima),[1]*len(maxima)))

		spike_times = np.concatenate((minima,maxima))

		needed_before = int((spike_snapshot[0] + 0.1)*(sampling_rate/1000.0))
		needed_after = int((spike_snapshot[1]+ 0.1)*(sampling_rate/1000.0))
		before_inds = spike_times - needed_before
		after_inds = spike_times + needed_after

		# Make sure event has required window around it
		relevant_inds = (before_inds > 0) * (after_inds < len(filt_el))
		before_inds = before_inds[relevant_inds]
		after_inds = after_inds[relevant_inds]
		slices = np.array([filt_el[start:end] \
				for start,end in zip(before_inds,after_inds)])

		return slices, spike_times[relevant_inds], polarity[relevant_inds], m, th

def extract_waveforms_hannah(filt_el, dir_name, spike_snapshot = [0.5, 1.0], 
								    sampling_rate = 30000.0,
								    threshold_mult = 5.0,
									cut_percentile =50):
		print('Pulling peak times.')
		#Sliding thresholding
		print('\t Calculating Threshold')
		len_filt_el = len(filt_el)
		sec_samples = int(60*5*sampling_rate) #5 minutes in samples
		start_times = np.arange(0,len_filt_el-sec_samples,sec_samples)
		threshold_vals = []
		mean_vals = []
		for s_i in tqdm(range(len(start_times))):
			s_t = start_times[s_i]
			filt_el_clip = np.array(filt_el)[max(s_t,0):min(s_t+sec_samples,len_filt_el)]
			#If the threshold is being calculated as resistant to outliers, so 
			#should the mean, so we should be calculating median instead.
			m_clip = np.median(filt_el_clip)
			mean_vals.extend([m_clip])
			#th_clip = threshold_mult*np.median(np.abs(filt_el_clip)/0.6745)
			#The above calculation comes from 'Robust Statistics' by B.D.Ripley
			#http://web.archive.org/web/20120410072907/http://www.stats.ox.ac.uk/pub/StatMeth/Robust.pdf
			#It's actually incorrectly written and should be:
			th_clip = threshold_mult*np.median(np.abs(filt_el_clip-m_clip))/0.6745
			threshold_vals.extend([th_clip])
		#Percentile mean and threshold values
		m = np.median(mean_vals)
		th = np.median(threshold_vals)
		print('\t Selected mean = ' + str(round(m,3)) + '; Selected thresh = ' + str(round(th,3)))
		#Fin/d peaks crossing threshold in either direction and combine
		all_peaks = np.array(find_peaks(np.abs(filt_el-m),height=th,distance=(1/1000)*sampling_rate)[0])
		abs_peak_heights = np.array([filt_el[all_peaks[i]] for i in range(len(all_peaks))])
		abs_peak_max_cutoff = np.percentile(abs_peak_heights,90)
		
		minima = np.array(find_peaks(-1*(filt_el-m),height=th)[0]) #indices of - peaks
		min_peak_heights = np.array([filt_el[minima[i]] for i in range(len(minima))])
		min_peak_max_cutoff = np.percentile(min_peak_heights,10)
		#Keep only the majority height range
		relevant_inds = np.find(min_peak_heights >= min_peak_max_cutoff)
		minima = minima[relevant_inds]
		
		maxima = np.setdiff1d(all_peaks,minima) #This ensures the maxima are not too close to the minima
		max_peak_heights = np.array([filt_el[maxima[i]] for i in range(len(maxima))])
		max_peak_max_cutoff = np.percentile(max_peak_heights,90)
		#Keep only the majority height range
		relevant_inds = np.find(max_peak_heights <= max_peak_max_cutoff)
		maxima = maxima[relevant_inds]
		
		#Plot peak heights and 90th percentile cutoff
		fig = plt.figure(figsize=(10,20))
		plt.subplot(3,1,1)
		plt.hist(abs_peak_heights)
		plt.axvline(abs_peak_max_cutoff)
		plt.title('Absolute Value of Signal')
		plt.subplot(3,1,2)
		plt.hist(min_peak_heights)
		plt.axvline(min_peak_max_cutoff)
		plt.title('Negative Deflections')
		plt.subplot(3,1,3)
		plt.hist(max_peak_heights)
		plt.axvline(max_peak_max_cutoff)
		plt.title('Positive Deflections')
		fig.savefig(dir_name + '/peak_height_distribution.png',bbox_inches='tight')
		plt.close('all')
		
		#Separately template match minima
		print('\t Sorting negative spikes')
		#Set snippet parameters
		needed_before = int((spike_snapshot[0] + 0.1)*(sampling_rate/1000.0))
		needed_after = int((spike_snapshot[1]+ 0.1)*(sampling_rate/1000.0))
		#Grab positive and negative spike waveforms and 'polarity'
		before_inds = minima - needed_before
		after_inds = minima + needed_after
		relevant_inds = (before_inds > 0) * (after_inds < len(filt_el))
		before_inds = before_inds[relevant_inds]
		after_inds = after_inds[relevant_inds]
		minima = minima[relevant_inds]
		minima_slices = np.array([filt_el[before_inds[m_i]:after_inds[m_i]] for m_i in range(len(before_inds))])
# 		#Returns only the template-match thresholded slices
# 		minima_slices, relevant_inds = st.spike_template_sort(minima_slices,'min',
# 														sampling_rate,needed_before,needed_after,
# 														cut_percentile,dir_name)
# 		irrelevant_inds = list(np.setdiff1d(np.arange(len(minima)),relevant_inds))
# 		#Plot example good and bad waveforms
# 		num_plot = 10
# 		keep_plot = random.sample(relevant_inds,num_plot)
# 		not_keep_plot = random.sample(irrelevant_inds,num_plot)
# 		fig = plt.figure(figsize=(10,20))
# 		for p_i in range(num_plot):
# 			plt.subplot(num_plot,2,p_i*2+1)
# 			plt.plot(filt_el[minima[keep_plot[p_i]]-25:minima[keep_plot[p_i]]+25])
# 			plt.title('Good')
# 		for p_i in range(num_plot):
# 			plt.subplot(num_plot,2,p_i*2+2)
# 			plt.plot(filt_el[minima[not_keep_plot[p_i]]-25:minima[not_keep_plot[p_i]]+25])
# 			plt.title('Bad')
# 		fig.savefig(dir_name + '/good_bad_min_waveforms.png',bbox_inches='tight')
# 		plt.close('all')
# 		#Keep only relevant waveforms
# 		minima = minima[relevant_inds]
# 		
 		#Separately threshold maxima
		print('\t Sorting positive spikes')
		#Set snippet parameters
		needed_before = int((spike_snapshot[0] + 0.1)*(sampling_rate/1000.0))
		needed_after = int((spike_snapshot[1]+ 0.1)*(sampling_rate/1000.0))
		#Grab positive and negative spike waveforms and 'polarity'
		before_inds = maxima - needed_before
		after_inds = maxima + needed_after
		relevant_inds = (before_inds > 0) * (after_inds < len(filt_el))
		before_inds = before_inds[relevant_inds]
		after_inds = after_inds[relevant_inds]
		maxima = maxima[relevant_inds]
		maxima_slices = np.array([filt_el[before_inds[m_i]:after_inds[m_i]] for m_i in range(len(before_inds))])
		#Returns only the template-match thresholded slices
# 		maxima_slices, relevant_inds = st.spike_template_sort(maxima_slices,'max',
# 														sampling_rate,needed_before,needed_after,
# 														cut_percentile,dir_name)
# 		irrelevant_inds = list(np.setdiff1d(np.arange(len(maxima)),relevant_inds))
# 		#Plot example good and bad waveforms
# 		num_plot = 10
# 		keep_plot = random.sample(relevant_inds,num_plot)
# 		not_keep_plot = random.sample(irrelevant_inds,num_plot)
# 		fig = plt.figure(figsize=(10,20))
# 		for p_i in range(num_plot):
# 			plt.subplot(num_plot,2,p_i*2+1)
# 			plt.plot(filt_el[maxima[keep_plot[p_i]]-25:maxima[keep_plot[p_i]]+25])
# 			plt.title('Good')
# 		for p_i in range(num_plot):
# 			plt.subplot(num_plot,2,p_i*2+2)
# 			plt.plot(filt_el[maxima[not_keep_plot[p_i]]-25:maxima[not_keep_plot[p_i]]+25])
# 			plt.title('Bad')
# 		fig.savefig(dir_name + '/good_bad_max_waveforms.png',bbox_inches='tight')
# 		plt.close('all')
# 		maxima = maxima[relevant_inds]
# 		
		#Combine thresholded results
		slices = np.concatenate([minima_slices,maxima_slices])
		spike_times = np.concatenate((minima,maxima))
		polarity = np.concatenate(([-1]*len(minima),[1]*len(maxima)))
		print('\t Total number of peaks = ' + str(len(spike_times)))

		return slices, spike_times, polarity, m, th

def extract_waveforms(filt_el, spike_snapshot = [0.5, 1.0], sampling_rate = 30000.0):
		m = np.mean(filt_el)
		th = 5.0*np.median(np.abs(filt_el)/0.6745)
		#pos = np.where(filt_el <= m-th)[0]
		pos = np.where( (filt_el <= m-th) | (filt_el > m+th) )[0]
		
		changes = []
		for i in range(len(pos)-1):
				if pos[i+1] - pos[i] > 1:
						changes.append(i+1)

		# slices = np.zeros((len(changes)-1,150))

		slices = []
		spike_times = []
		for i in range(len(changes) - 1):
				minimum = np.where(filt_el[pos[changes[i]:changes[i+1]]] == \
						np.min(filt_el[pos[changes[i]:changes[i+1]]]))[0]

				#print minimum, len(slices), len(changes), len(filt_el)
				# try slicing out the putative waveform, 
				# only do this if there are 10ms of data points 
				# (waveform is not too close to the start or end of the recording)
				if pos[minimum[0]+changes[i]] \
						- int((spike_snapshot[0] + 0.1)*(sampling_rate/1000.0))> 0 \
						and pos[minimum[0]+changes[i]] + int((spike_snapshot[1]+ 0.1)\
								    *(sampling_rate/1000.0)) < len(filt_el):
						slices.append(filt_el[pos[minimum[0]+changes[i]] - \
								int((spike_snapshot[0] + 0.1)*(sampling_rate/1000.0)) : \
										    pos[minimum[0]+changes[i]] + int((spike_snapshot[1] \
												    + 0.1)*(sampling_rate/1000.0))])
						spike_times.append(pos[minimum[0]+changes[i]])

		return np.array(slices), spike_times, m, th

def dejitter(slices, spike_times, spike_snapshot = [0.5, 1.0], sampling_rate = 30000.0):
		x = np.arange(0,len(slices[0]),1)
		# Support vector for 10x interpolation
		xnew = np.arange(0,len(slices[0])-1,0.1)

		# Calculate the number of samples to be sliced out around each spike's minimum
		before = int((sampling_rate/1000.0)*(spike_snapshot[0]))
		after = int((sampling_rate/1000.0)*(spike_snapshot[1]))
		
		#slices_dejittered = []
		slices_dejittered = np.zeros((slices.shape[0],(before+after)*10))
		spike_times_dejittered = []
		for i in range(len(slices)):
				f = interp1d(x, slices[i])
				# 10-fold interpolated spike
				ynew = f(xnew)
				# Find minimum only around the center of the waveform
				# Since we're slicing around the waveform as well
				minimum = np.where(ynew == np.min(ynew))[0][0]
				# Only accept spikes if the interpolated minimum has 
				# shifted by less than 1/10th of a ms 
				# (3 samples for a 30kHz recording, 
				# 30 samples after interpolation)
				# If minimum hasn't shifted at all, 
				# then minimum - 5ms should be equal to zero 
				# (because we sliced out 5 ms before the minimum 
				# in extract_waveforms())
				# We use this property in the if statement below
				cond1 = np.abs(minimum - \
						int((spike_snapshot[0] +0.1)\
						*(sampling_rate/100.0))) \
						    <= int(10.0*(sampling_rate/10000.0))
				y_temp = ynew[minimum - before*10 : minimum + after*10]
				# Or slices which (SOMEHOW) don't have the expected size
				cond2 = len(y_temp) == slices_dejittered.shape[-1] 
				if cond1 and cond2:
						#slices_dejittered.append(ynew[minimum - before*10 : minimum + after*10])
						slices_dejittered[i] = y_temp 
						spike_times_dejittered.append(spike_times[i])

		# Remove placeholder for slices which didnt meet the condition criteria
		slices_dejittered = \
		slices_dejittered[np.sum(slices_dejittered,axis=-1) != 0]

		return slices_dejittered, np.array(spike_times_dejittered)

def dejitter_abu3(slices, 
				spike_times, 
				polarity,
				spike_snapshot = [0.5, 1.0], 
				sampling_rate = 30000.0):

		"""
		Dejitter without interpolation and see what breaks :P
		"""
		# Calculate the number of samples to be sliced 
		#out around each spike's minimum
		before = int((sampling_rate/1000.0)*(spike_snapshot[0]))
		after = int((sampling_rate/1000.0)*(spike_snapshot[1]))

		# Determine positive or negative spike and flip
		# positive spikes so everything is aligned by minimum
		# Then flip positive spikes back to being positive
		flipped_slices = np.copy(slices)
		flipped_slices[polarity > 0] *= -1

		#interp_slices = np.array([interp1d(x, this_slice)(xnew) \
		#		for this_slice in flipped_slices])

		# Cut out part around focus of spike snapshot to use
		# for finding minima
		# 3 bins (0.1 ms) is the wiggle room we gave ourselves 
		# when extracting spikes, therefore, each spike is organized as
		#   0.1 ms |-| Before |-| Minimum |-| After |-| 0.1 ms
		# We will use 0.1ms around the minimum to dejitter the spike 
		cut_radius = 3
		cut_tuple = (int((before) + (cut_radius/2)), 
				int(flipped_slices.shape[1] - (after) - (cut_radius/2)))
		# minima will tell us how much each spike needs to be shifted
		minima = np.argmin(flipped_slices[:,cut_tuple[0]:cut_tuple[1]],
						axis=-1) + (before) + (cut_radius/2) 

		# Extract windows AROUND minima
		slices_dejittered = np.array([this_slice[\
		    int(this_min - (before)) : int(this_min + (after))] \
		    for this_slice, this_min in zip(flipped_slices, minima)])

		# Flip positive slices
		slices_dejittered[polarity > 0] *= -1

		return slices_dejittered, spike_times

def scale_waveforms(slices_dejittered):
		energy = np.sqrt(np.sum(slices_dejittered**2, axis = 1))/len(slices_dejittered[0])
		scaled_slices = np.zeros((len(slices_dejittered),len(slices_dejittered[0])))
		for i in range(len(slices_dejittered)):
				scaled_slices[i] = slices_dejittered[i]/energy[i]

		return scaled_slices, energy

def implement_pca(scaled_slices):
		pca = PCA()
		pca_slices = pca.fit_transform(scaled_slices)   
		return pca_slices, pca.explained_variance_ratio_
