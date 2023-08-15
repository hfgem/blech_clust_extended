# blech_clust_extended

Python and R based code for clustering and sorting electrophysiology data
recorded using the Intan RHD2132 chips.  Originally written for cortical
multi-electrode recordings in Don Katz's lab at Brandeis. Visit the Katz lab
website at https://sites.google.com/a/brandeis.edu/katzlab/

### Setup
```
cd <path_to_blech_clust>/requirements               # Move into blech_clust folder with requirements files
conda clean --all                                   # Removes unused packages and caches
conda create --name blech_clust python=3.8.13       # Create "blech_clust" environment with conda requirements
conda activate blech_clust                          # Activate blech_clust environment
bash conda_requirements_base.sh                     # Install main packages using conda/mamba
bash install_gnu_parallel.sh                        # Install GNU Parallel
pip install -r pip_requirements_base.txt            # Install pip requirements (not covered by conda)
```
### Running
run_clustering.py is a master script to run through each step of clustering (below), allowing user input to continue or stop as needed. The script automatically stores the status of the clustering in the data directory for use next time it is run. Note, as of 08/15/2023 you cannot run clustering on multiple datasets in parallel unless they are past the 'blech_process.py' stage, as this stage currently uses a directory stored in the blech_clust_extended GitHub folder for reference. This will be fixed in a future change.

### Order of operations
The indices below align with outputs to state_tracker.csv from run_clustering.py. When a state is complete (like blech_exp_info.py), that number will be stored in state_tracker.csv (when blech_exp_info.py has been run, the csv will contain a '1').

1. python blech_exp_info.py  
    - Pre-clustering step. Annotate recorded channels and save experimental parameters  
    - Takes template for info and electrode layout as argument
2. python blech_clust.py  
    - Setup directories and define clustering parameters  
3. python blech_common_avg_reference.py  [optional - will ask user if desired]
    - Perform common average referencing to remove large artifacts  
4. python blech_process.py [will run for each neuron individually, parallelizing 4 at a time]
    - Embarrasingly parallel spike extraction and clustering  
5. python blech_post_process.py - taste=True
    - Add selected units to HDF5 file for further processing
6. python blech_process_full_recording.py
    - Use the selected units from the taste interval to create spike templates and template match the full recording.
7. python blech_post_process.py - taste=False
    - Add the final selected units from the full recording to the HDF5 file for further processing
8. python blech_units_similarity.py  [optional - will ask user if needed]
    - Check for collisions of spiketimes to assess double-counting of waveforms in clustering
9. python blech_units_plot.py  
    - Plot waveforms of selected spikes  
10. python blech_make_arrays.py and python blech_make_psth.py
    - Generate spike-train arrays and plot PSTHs and rasters for all units.

### Test Dataset
We are grateful to Brandeis University Google Filestream for hosting this dataset <br>
Data to test workflow available at:<br>
https://drive.google.com/drive/folders/1ne5SNU3Vxf74tbbWvOYbYOE1mSBkJ3u3?usp=sharing

