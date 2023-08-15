# Necessary python modules
import os
import tables
import sys
import numpy as np
import multiprocessing
import json
import glob
import pandas as pd
import shutil

# Necessary blech_clust modules
from utils import read_file
from utils.blech_utils import entry_checker, imp_metadata

# Get blech_clust path
script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(script_path)

############################################################

metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name
print(f'Processing : {dir_name}')
os.chdir(dir_name)

info_dict = metadata_handler.info_dict
file_list = metadata_handler.file_list


# Get the type of data files (.rhd or .dat)
if 'auxiliary.dat' in file_list:
    file_type = ['one file per signal type']
else:
    file_type = ['one file per channel']

# Create hdf5 file, and make groups for raw data, raw emgs,
# digital outputs and digital inputs, and close

# Grab directory name to create the name of the hdf5 file
# If HDF5 present, use that, otherwise, create new one
h5_search = glob.glob('*.h5')
if len(h5_search):
    hdf5_name = h5_search[0]
    print(f'HDF5 file found...Using file {hdf5_name}')
    hf5 = tables.open_file(hdf5_name, 'r+')
else:
	if dir_name[-1] != '/':
		dir_name = dir_name + '/' #On Mac the directory pulled doesn't always end in /
	hdf5_name = str(os.path.dirname(dir_name)).split('/')[-1]+'.h5'
	print(f'No HDF5 found...Creating file {hdf5_name}')
	hf5 = tables.open_file(hdf5_name, 'w', title=hdf5_name[-1])

group_list = ['raw', 'raw_taste', 'raw_emg', 'raw_emg_taste', 'digital_in', 'digital_out']
for this_group in group_list:
    if '/'+this_group in hf5:
        hf5.remove_node('/', this_group, recursive=True)
    hf5.create_group('/', this_group)
hf5.close()
print('Created nodes in HF5')

# Create directories to store waveforms, spike times, clustering results, and plots
# And a directory for dumping files talking about memory usage in blech_process.py
# Check if dirs are already present, if they are, ask to delete and continue
# or abort
dir_list = ['spike_waveforms',
            'spike_times',
            'clustering_results',
            'Plots',
            'memory_monitor_clustering',
			'spike_waveforms_taste',
			'spike_times_taste',
			'clustering_results_taste',
			'Plots_taste',
			'memory_monitor_clustering_taste']
dir_exists = [x for x in dir_list if os.path.exists(x)]
recreate_msg = f'Following dirs are present :' + '\n' + f'{dir_exists}' + \
    '\n' + 'Overwrite dirs? (yes/y/n/no) ::: '

# If dirs exist, check with user
if len(dir_exists) > 0:
    recreate_str, continue_bool = entry_checker(
        msg=recreate_msg,
        check_func=lambda x: x in ['y', 'yes', 'n', 'no'],
        fail_response='Please enter (yes/y/n/no)')
# Otherwise, create all of them
else:
    continue_bool = True
    recreate_str = 'y'

# Break if user said n/no or gave exit signal
if continue_bool:
    if recreate_str in ['y', 'yes']:
        for x in dir_list:
            if os.path.exists(x):
                shutil.rmtree(x)
            os.makedirs(x)
else:
    quit()

print('Created dirs in data folder')

# Get lists of amplifier and digital input files
if file_type == ['one file per signal type']:
    electrodes_list = ['amplifier.dat']
    dig_in_list = ['digitalin.dat']
elif file_type == ['one file per channel']:
    electrodes_list = [name for name in file_list if name.startswith('amp-')]
    dig_in_list = [name for name in file_list if name.startswith('board-DI')]

# Use info file for port list calculation
info_file = np.fromfile(dir_name + '/info.rhd', dtype=np.dtype('float32'))
sampling_rate = int(info_file[2])

# Read the time.dat file for use in separating out the one file per signal type data
num_recorded_samples = len(np.fromfile(
    dir_name + '/' + 'time.dat', dtype=np.dtype('float32')))
total_recording_time = num_recorded_samples/sampling_rate  # In seconds

check_str = f'Amplifier files: {electrodes_list} \nSampling rate: {sampling_rate} Hz'\
    f'\nDigital input files: {dig_in_list} \n ---------- \n \n'
print(check_str)

ports = info_dict['ports']

if file_type == ['one file per channel']:
    print("\tOne file per CHANNEL Detected")

    # Read dig-in data
    # Pull out the digital input channels used,
    # and convert them to integers
    dig_in = [x.split('-')[-1].split('.')[0] for x in dig_in_list]
    dig_in = sorted([int(x) for x in dig_in])

elif file_type == ['one file per signal type']:

    print("\tOne file per SIGNAL Detected")
    dig_in = np.arange(info_dict['dig_ins']['count'])

check_str = f'ports used: {ports} \n sampling rate: {sampling_rate} Hz'\
            f'\n digital inputs on intan board: {dig_in}'

print(check_str)

all_car_group_vals = []
for region_name, region_elecs in info_dict['electrode_layout'].items():
    if not region_name == 'emg':
        for group in region_elecs:
            if len(group) > 0:
                all_car_group_vals.append(group)
all_electrodes = [electrode for region in all_car_group_vals
                  for electrode in region]

emg_info = info_dict['emg']
emg_port = emg_info['port']
emg_channels = sorted(emg_info['electrodes'])


layout_path = glob.glob(os.path.join(dir_name, "*layout.csv"))[0]
electrode_layout_frame = pd.read_csv(layout_path)


# Read data files, and append to electrode arrays
if file_type == ['one file per channel']:
    min_time, max_time = read_file.read_digins(hdf5_name, dig_in, dig_in_list, sampling_rate)
    read_file.read_electrode_channels(hdf5_name, electrode_layout_frame, min_time, max_time)
    if len(emg_channels) > 0:
        read_file.read_emg_channels(hdf5_name, electrode_layout_frame, min_time, max_time)
elif file_type == ['one file per signal type']:
    min_time, max_time = read_file.read_digins_single_file(hdf5_name, dig_in, dig_in_list, sampling_rate)
    # This next line takes care of both electrodes and emgs
    read_file.read_electrode_emg_channels_single_file(
        hdf5_name, electrode_layout_frame, electrodes_list, num_recorded_samples, emg_channels, min_time, max_time)

# Write out template params file to directory if not present
print(blech_clust_dir)
params_template_path = os.path.join(
    blech_clust_dir,
    'params/sorting_params_template.json')
params_template = json.load(open(params_template_path, 'r'))
# Info on taste digins and laser should be in exp_info file
all_params_dict = params_template.copy()
all_params_dict['sampling_rate'] = sampling_rate

params_out_path = hdf5_name.split('.')[0] + '.params'
if not os.path.exists(params_out_path):
    print('No params file found...Creating new params file')
    with open(params_out_path, 'w') as params_file:
        json.dump(all_params_dict, params_file, indent=4)
else:
    print("Params file already present...not writing a new one")

# Dump shell file(s) for running GNU parallel job on the user's blech_clust folder on the desktop
# First get number of CPUs - parallel be asked to run num_cpu-1 threads in parallel
num_cpu = multiprocessing.cpu_count()

electrode_bool = electrode_layout_frame.loc[
    electrode_layout_frame.electrode_ind.isin(all_electrodes)]
not_none_bool = electrode_bool.loc[~electrode_bool.CAR_group.isin(
    ["none", "None", 'na'])]
not_emg_bool = not_none_bool.loc[
    ~not_none_bool.CAR_group.str.contains('emg')
]
bash_electrode_list = not_emg_bool.electrode_ind.values
job_count = np.min((len(bash_electrode_list), int(num_cpu-2)))

#Output a .csv of the non_emg_bool vals as well
f = open(os.path.join(dir_name, 'electrode_list.csv'), 'w')
print(f"{' '.join([str(x) for x in bash_electrode_list])}", file=f)
f.close()

# Dump the directory name where blech_process has to cd
f = open(os.path.join(blech_clust_dir, 'blech.dir'), 'w')
print(dir_name, file=f)
f.close()

print('blech_clust.py complete \n')
print('*** Please check params file to make sure all is good ***\n')
