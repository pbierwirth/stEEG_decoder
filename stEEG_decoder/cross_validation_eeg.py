#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:27:11 2025

@author: philipp
"""

from sklearn.model_selection import StratifiedKFold
import numpy as np

class CrossValidator:
    def __init__(self, y, n_splits=5, shuffle=True, random_state=42):
        self.y = y
        self.n_trials = len(y)
        self.splits = []
        self.kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self._create_splits()

    def _create_splits(self):
        dummy_X = np.zeros((self.n_trials, 1))
        self.splits = list(self.kf.split(dummy_X, self.y))

    def get_split_indices(self, fold_idx):
        return self.splits[fold_idx]

    def split_data(self, data, fold_idx):
        """
        Split full data and labels using stored indices for a given fold.

        Parameters
        ----------
        data : np.ndarray
            EEG data of shape (electrodes, time, trials)
        fold_idx : int
            Index of the CV fold

        Returns
        -------
        x_train, x_test, y_train, y_test
        """
        train_idx, test_idx = self.get_split_indices(fold_idx)
        x_train = data[:, :, train_idx]
        x_test = data[:, :, test_idx]
        y_train = self.y[train_idx]
        y_test = self.y[test_idx]
        return x_train, x_test, y_train, y_test

    def __len__(self):
        return len(self.splits)
