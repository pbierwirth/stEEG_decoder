"""
@author: philipp bierwirth
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from .helper import fast_auc


def decode_temporal_generalization(x_train, x_test, y_train, y_test):
    """
    Performs temporal generalization decoding with PCA, a linear SVM, and Haufe-transformed weights for each channel.
    Note: This function currently supports only binary classification.

    Input:
    -----------
    x_train : np.ndarray
        EEG data for training (elecs × time × trials)
    x_test : np.ndarray
        EEG data for testing (elecs × time × trials)
    y_train : np.ndarray
        Class labels for training trials
    y_test : np.ndarray
        Class labels for testing trials

    Please note that inputs have to be provided as numpy arrays.

    Output:
    --------
    Dictionary with the following numpy arrays:

        'diagonal': 
            AUC score for t_train-t_test decoding (n_timepoints)
        'tp_matrix': 
            Temporal generalization matrix (train_time × test_time)
        'activation_map':  
            Haufe-transformed weights (train_time × electrodes)
        
    Example Usage:
    --------
    >>> results = (decode_temporal_generalization(x_train, x_test, y_train, y_test))
    """

    # Safety Check: Enforce Binary Classes 
    unique_labels = np.unique(y_train)
    if len(unique_labels) != 2:
        raise ValueError(f"Pipeline requires exactly 2 classes. Found {len(unique_labels)}: {unique_labels}")
    
    # Safety Check: Enforce 0 and 1 Integers 
    if not np.array_equal(unique_labels, [0, 1]):
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
    
    # Reshape test data to shape: (trials × time, elecs)
    x_test_reshaped = x_test.transpose(
        2, 1, 0).reshape(-1, x_train.shape[0])   
    
    # initalize output list
    score = []
    activation_map = []
    
    for t in range(x_train.shape[1]):
        # get training data for single time point and transpose to shape: (trials x elec)
        x_train_t = x_train[:, t, :].T
    
        # ------------------------------------------------------------------------
        # Preprocessing: Standardization and PCA 
        # ------------------------------------------------------------------------
        
        scaler = StandardScaler()
        x_train_t = scaler.fit_transform(x_train_t)
        
        # calculate covariance matrix between electrodes for Haufe transformation 
        covmat = np.cov(x_train_t.T)
        
        # reduce dimensionality of the data via PCA
        pca = PCA(n_components=0.95)
        x_train_pca = pca.fit_transform(x_train_t)
           
                  # apply training-based standaridzation and PCA to test data
        x_test_t = scaler.transform(x_test_reshaped)
        x_test_pca = pca.transform(x_test_t)

        # ------------------------------------------------------------------------
        # Train the classifier
        # ------------------------------------------------------------------------
        
        # initialize classifier
        clf = LinearSVC()
        # fit classifier on training data
        clf.fit(x_train_pca, y_train)
 
        # ------------------------------------------------------------------------
        # Temporal generalization
        # ------------------------------------------------------------------------
        
        # get scores/estiamtions of the decision function across trials and time points
        y_scores = clf.decision_function(x_test_pca)
        # reshape scores into 2d matrix (trials x time points)
        y_scores = np.reshape(y_scores,
                              (len(y_test), int(len(y_scores)/len(y_test)))
                              )
        
        # call fast_auc and save AUC score in list
        score.append(fast_auc(y_test, y_scores))
        
        # ------------------------------------------------------------------------
        # Weight projection
        # ------------------------------------------------------------------------
        
        # project classifier weights from component space into electrode space 
        projected_weights = clf.coef_ @ pca.components_ 
        
        # Apply Haufe transformation to get interpretable activation maps and 
        # stroe maps in list
        activation_map.append(projected_weights @ covmat)
        
    
    # transform lists to numpy array for further processing
    score = np.stack(score)  # shape: (train_time, test_time)
    activation_map = np.stack(np.squeeze(activation_map))  # shape: (train_time, electrodes)

    results = {
        'diagonal': np.diag(score),
        'tp_matrix': score,
        'activation_map': activation_map
        }
    return results
    
    
    
    
    
