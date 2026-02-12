"""
@author: Philipp Bierwirth

"helper.py" contains helper functions that are used by the NeuroDecode pipeline

"""

import numpy as np
from numba import njit

@njit
def fast_auc(y_true, y_scores_2d):
    """
    Numba-optimized computation of the Area Under the Curve (AUC) across time.
    
    Input
    ----------
    y_true : np.ndarray
        True binary labels for each trial (must be 0 and 1). Shape: (n_trials,)
    y_scores_2d : np.ndarray
        Decision function scores or probabilities. Shape: (n_trials, n_times)

    Output
    -------
    aucs : np.ndarray
        The computed AUC score for each time point. Shape: (n_times,)
  
    """

    n_trials, n_times = y_scores_2d.shape
    aucs = np.empty(n_times)
    
    for t in range(n_times):
        scores = y_scores_2d[:, t]
        
        # argsort scores descending
        order = np.argsort(-scores)
        y_sorted = y_true[order]
        
        cum_pos = 0
        cum_neg = 0
        auc = 0.0
        
        for i in range(n_trials):
            if y_sorted[i] == 1:
                cum_pos += 1
            else:
                auc += cum_pos
                cum_neg += 1
        
        if cum_pos == 0 or cum_neg == 0:
            aucs[t] = np.nan 
        else:
            aucs[t] = auc / (cum_pos * cum_neg)
    
    return aucs


def subaverage(X, y, samples_per_group,  random_state=None):
    """
    Sub-averages EEG trials within each class label.
    
    Input:
    -----------
    X : np.ndarray
        EEG data, shape: (electrodes, timepoints, trials)
    y : np.ndarray
        Labels for each trial, shape: (trials,)
    samples_per_group : int
        Number of trials to average together
    random_state : int or None
        Random seed for reproducibility

    Output:
    --------
    new_X : np.ndarray
        Averaged EEG data, shape: (electrodes, timepoints, new_trials)
    new_y : np.ndarray
        Corresponding labels, shape: (new_trials,)
    """
    rng = np.random.default_rng(random_state)

    new_X = []
    new_y = []
    unique_labels = np.unique(y)

    for label in unique_labels:
        idx = np.where(y == label)[0]
        n_available = len(idx)


        rng.shuffle(idx)
        for i in range(0, n_available, samples_per_group):
            chunk = idx[i:i + samples_per_group]
            if len(chunk) >= 1:
                avg = X[:, :, chunk].mean(axis=2)
                new_X.append(avg)
                new_y.append(label)

    new_X = np.stack(new_X, axis=2)
    new_y = np.array(new_y)
    return new_X, new_y

