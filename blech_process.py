"""
Changes to be made:
    1) Add flag for throwing out waveforms at blech_clust step
    2) Convert pred_spikes + pred_noice plots to datashader
    3) Number of noise clusters = 7
    4) Number of spikes clusters = 3
    5) Save predictions so they can be used in UMAP plot

Model monitoring:
    1) Monitoring input and output data distributions

Steps:
    1) Load Data
    2) Preprocess
        a) Bandpass filter
        b) Extract and dejitter spikes
        c) Extract amplitude (accounting for polarity)
        d) Extract energy and scale waveforms by energy
        e) Perform PCA
        f) Scale all features using StandardScaler
    3) Perform clustering
"""

############################################################
# Imports
############################################################

import os, json, sys
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import pylab as plt
import numpy as np
from utils.blech_utils import (
    imp_metadata,
)
import utils.blech_process_utils as bpu
from utils import memory_monitor as mm
import utils.blech_spike_features as bsf

# Set seed to allow inter-run reliability
# Also allows reusing the same sorting sheets across runs
np.random.seed(0)

############################################################
# Load Data
############################################################

# cd to passed directory of data
electrode_num = int(sys.argv[1])
data_dir_name = sys.argv[2]

metadata_handler = imp_metadata([[], data_dir_name])
os.chdir(metadata_handler.dir_name)

params_dict = metadata_handler.params_dict

# Check if the directories for this electrode number exist -
# if they do, delete them (existence of the directories indicates a
# job restart on the cluster, so restart afresh)
dir_list = [f'./Plots_taste/{electrode_num:02}',
            f'./spike_waveforms_taste/electrode{electrode_num:02}',
            f'./spike_times_taste/electrode{electrode_num:02}',
            f'./clustering_results_taste/electrode{electrode_num:02}']
for this_dir in dir_list:
    bpu.ifisdir_rmdir(this_dir)
    os.makedirs(this_dir)

############################################################
# Preprocessing
############################################################
# Open up hdf5 file, and load this electrode number
electrode = bpu.electrode_handler(
                  metadata_handler.hdf5_name,
                  electrode_num,
                  params_dict,
				  taste=True)

electrode.filter_electrode()

# Calculate the 3 voltage parameters
electrode.cut_to_int_seconds()
electrode.calc_recording_cutoff()

# Dump a plot showing where the recording was cut off at
electrode.make_cutoff_plot()

# Then cut the recording accordingly
electrode.cutoff_electrode()

#############################################################
# Process Spikes
#############################################################

# Extract spike times and waveforms from filtered data
spike_set = bpu.spike_handler(metadata_handler.hdf5_name,
							  electrode.filt_el, 
                              params_dict, data_dir_name, 
							  electrode_num, taste=True)
spike_set.extract_waveforms()

############################################################
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
fig.savefig(f'./Plots_taste/{electrode_num:02}/bandpass_trace_snippets.png',
            bbox_inches='tight', dpi=300)
plt.close(fig)
############################################################

# Delete filtered electrode from memory
del electrode

# Dejitter these spike waveforms, and get their maximum amplitudes
# Slices are returned sorted by amplitude polaity
spike_set.dejitter_spikes()

spike_set.extract_amplitudes()
num_spikes = len(spike_set.amplitudes)
cluster_num = int(max(min(np.ceil(num_spikes/5000),25),2))

spike_set.extract_features(
            bsf.feature_pipeline,
            bsf.feature_names,
            fitted_transformer=False,
            )

spike_set.write_out_spike_data()

# Set a threshold on how many datapoints are used to FIT the gmm
# Run GMM, from 2 to max_clusters
#for cluster_num in range(params_dict['min_clusters'], params_dict['max_clusters']+1):
cluster_handler = bpu.cluster_handler(
            params_dict, 
            data_dir_name, 
            electrode_num,
            cluster_num,
            spike_set,
			taste=True,
			unit_num=-1)
cluster_handler.perform_prediction()
cluster_handler.remove_outliers(params_dict)
cluster_handler.save_cluster_labels()
cluster_handler.create_output_plots( 
                            params_dict)

# Make file for dumping info about memory usage
f= open(f'./memory_monitor_clustering_taste/{electrode_num:02}.txt', 'w')
print(mm.memory_usage_resource(), file=f)
f.close()
print(f'Electrode {electrode_num} complete.')
