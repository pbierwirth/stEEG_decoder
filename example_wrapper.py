"""
@author: philipp bierwirth
"""

from stEEG_decoder.pipeline import decode_temporal_generalization
from stEEG_decoder.cross_validation_eeg import CrossValidator
from stEEG_decoder.viz import plot_decoding_results
from stEEG_decoder.helper import subaverage # remove # if you want to aggregate trials

import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% Run pipleine across all subjects 

# 1. Change directory to get all files
files = glob.glob('./eeg_example_data/*pkl')

# 2. Initialize lists 
group_diagonals = []
group_tp_matrices = []
group_activations = []
time = None 

# 3. Main Subject Loop
for file in tqdm(files, desc='Processing subjects'):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        
    x, y = data['x'], data['y']
    
    # Remove the # from the line below to use subaverage for aggregation of trials 
    # x,y = subaverage(x,y, samples_per_group = 4) # Please note that you also need to import the subaverage function
    
    if time is None:
        time = data['time'] # Grab the time array once
        
    cv = CrossValidator(y, n_splits=5)
    
    # Temporary lists for this specific subject's folds
    fold_diags, fold_tps, fold_acts = [], [], []
    
    # Inner Fold Loop
    for fold in tqdm(range(len(cv)), desc='Computing folds', leave=False):
        x_train, x_test, y_train, y_test = cv.split_data(x, fold_idx=fold)
        
        # Run pipeline
        res = decode_temporal_generalization(x_train, x_test, y_train, y_test)
        
        # Store fold results directly
        fold_diags.append(res['diagonal'])
        fold_tps.append(res['tp_matrix'])
        fold_acts.append(res['activation_map'])
        
    # Average across the 5 folds (axis=0) and store in the main group lists
    group_diagonals.append(np.mean(fold_diags, axis=0))
    group_tp_matrices.append(np.mean(fold_tps, axis=0))
    group_activations.append(np.mean(fold_acts, axis=0))

# 4. Convert lists to 3D NumPy arrays for easy plotting
diagonal = np.array(group_diagonals)
tp_matrix = np.array(group_tp_matrices)
activation = np.array(group_activations)

print("Pipeline finished successfully!")

#%% Plot grand-average results of pipeline

# load channel coordinates
fpath = 'eeg_example_data/channel_coordinates.npy'
with open(fpath, 'rb') as f:
    coords = np.load(f)


# Plot results
fig = plot_decoding_results(
    diagonal, 
    tp_matrix, 
    activation, 
    time, 
    coords,
    topo_time_range=(165, 175) # Average between 165-175ms
)

