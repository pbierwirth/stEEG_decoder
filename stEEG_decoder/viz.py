#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: philipp bierwirth
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf


def topoplot( topo_data, coords, ax = None, levels=200, 
             elec = False, contour = False):
    """
    Rough replication of the EGLAB topoplot style in Matplotlib.
    Please note that this function is only for quick illustration purposes 
    it shouldn't be used to create publication-grade figures
    
    Input
    ----------
    topo_data: np.ndarray (n_electrodes,)
    coords: np.ndarray of x,y,and z channel positions 

    Output
    -------
    ax: matplotlib axis object depicting 2d scalp topography
  
    
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Azimuthal Equidistant Projection
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arccos(z / r)
    theta = np.arctan2(y, x)
    
    # Scale so the furthest electrodes are at radius 0.9
    phi = (phi / phi.max()) * 0.9 
    x2d = phi * np.cos(theta)
    y2d = phi * np.sin(theta)

    # High-resolution grid
    grid_res = 200
    xi = np.linspace(-1.05, 1.05, grid_res)
    yi = np.linspace(-1.05, 1.05, grid_res)
    XI, YI = np.meshgrid(xi, yi)

    # EEGLAB's 'v4' Interpolation equivalent (thin_plate)
    rbf = Rbf(x2d, y2d, topo_data, function='thin_plate')
    ZI = rbf(XI, YI)

    # Calculate the distance from the center for every point in the grid
    radius_grid = np.sqrt(XI**2 + YI**2)
    # Set everything outside the head radius (1.0) to NaN
    ZI[radius_grid > 1.0] = np.nan 

    # Plotting Colors and Contours
    v_limit = np.max(np.abs(topo_data))
    
    # The smooth color fill
    ax.contourf(XI, YI, ZI, levels=levels, cmap='RdBu_r', 
                     vmin=-v_limit, vmax=v_limit)
    
    # Add a white ring around the head to smooth edges
    white_ring = plt.Circle((0, 0), radius=1.0, color='white', 
                            fill=False, linewidth=4, zorder=3)
    ax.add_patch(white_ring)
    
    if contour == True:
        # Add contour lines
        ax.contour(XI, YI, ZI, levels=6, colors='black', linewidths=0.5, alpha=0.5)

    # Draw the head
    head_radius = 0.75
    linewidth = 1

    # Head circle
    head_x = head_radius * np.cos(np.linspace(0, 2*np.pi, 100))
    head_y = head_radius * np.sin(np.linspace(0, 2*np.pi, 100))
    ax.plot(head_x, head_y, color='black', linewidth=linewidth)

    # Nose
    nose_x = [-0.1, 0, 0.1]
    nose_y = [head_radius - 0.01, head_radius + 0.08, head_radius - 0.01]
    ax.plot(nose_x, nose_y, color='black', linewidth=linewidth)

    # Left Ear
    ear_t = np.linspace(-np.pi/2, np.pi/2, 50)
    ear_x_left = -head_radius - 0.05 * np.cos(ear_t)
    ear_y = 0.15 * np.sin(ear_t)
    ax.plot(ear_x_left, ear_y, color='black', linewidth=linewidth)

    # Right Ear
    ear_x_right = head_radius + 0.05 * np.cos(ear_t)
    ax.plot(ear_x_right, ear_y, color='black', linewidth=linewidth)
    
    if elec == True:
        # Draw electrodes
        ax.scatter(x2d, y2d, c='black', s=10, edgecolors='black', linewidth=0.5, zorder=5)
    
    # Formatting
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Explicitly lock axes boundaries
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    # Add Colorbar
    # cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label('Activation Intensity', rotation=90, labelpad=10)
    
    return ax

def plot_decoding_results(diagonal, tp_matrix, activation_map, time, channel_coords, 
                          topo_time_range=(0.165, 0.175), figsize=(10, 4)):
    """
    Generates a summary figure containing:
      1. Diagonal decoding performance with SEM shading
      2. Topoplot of activation patterns
      3. Temporal Generalization Matrix

    Input
    ----------
    diagonal : np.ndarray
        Shape (n_subjects, n_times). Diagonal decoding scores.
    tp_matrix : np.ndarray
        Shape (n_subjects, n_times, n_times). Temporal generalization matrices.
    activation_map : np.ndarray
        Shape (n_subjects, n_times, n_channels). Haufe-transformed weights.
    time : np.ndarray
        Time vector in seconds or ms.
    channel_coords : np.ndarray
        Shape (n_channels, 3). x, y, z coordinates of electrodes.
    topo_time_range : tuple
        (start, end) time window to average for the topoplot inset (e.g., 0.170).
    figsize : tuple
        Figure dimensions.

    Output
    -------
    matplotlib figure
    """
    
    # Prepare Data
    # Calculate Mean and SEM for diagonal
    dia_mean = np.mean(diagonal, axis=0)
    dia_sem = np.std(diagonal, axis=0) / np.sqrt(diagonal.shape[0])
    
    # Calculate Mean TGM
    tgm_mean = np.mean(tp_matrix, axis=0)
    
    # Calculate Topo Data (Average over subjects and time window)
    # Find indices for the time window
    t_start = np.argmin(np.abs(time - topo_time_range[0]))
    t_end = np.argmin(np.abs(time - topo_time_range[1]))
    
    # Shape: (n_subjects, n_times, n_chans) 
    topo_data = np.mean(activation_map[:, t_start:t_end, :], axis=(0, 1))

    # Setup Figure
    fig = plt.figure(figsize=figsize)
    
    # Define axes using custom layout

    ax_dia = fig.add_axes([0.08, 0.2, 0.38, 0.7])
    ax_tg  = fig.add_axes([0.55, 0.2, 0.38, 0.7]) 
    # Topoplot 
    ax_top = fig.add_axes([0.25, 0.55, 0.15, 0.3]) 

    # Plot Diagonal Decoding
    ax_dia.axhline(0.5, linestyle='--', color='black', linewidth=0.5)
    ax_dia.axvline(0, linestyle='--', color='black', linewidth=0.5)
    
    # Plot mean and shading
    ax_dia.plot(time, dia_mean, color='#2b8cbe', linewidth=1, label='Decoding')
    ax_dia.fill_between(time, dia_mean - dia_sem, dia_mean + dia_sem, 
                        color='#2b8cbe', alpha=0.2)
    
    ax_dia.set_xlabel('Time (s)')
    ax_dia.set_ylabel('AUC Score')
    ax_dia.set_title('Diagonal Decoding', fontsize=10, pad=10)
    ax_dia.spines[['top', 'right']].set_visible(False)

    # Plot Inset Topoplot
    topoplot(topo_data, channel_coords, ax=ax_top, elec=False)
    ax_top.set_title(f'{np.mean(topo_time_range):.0f} ms', fontsize=8, pad=2)

    #v_max = np.max(np.abs(tgm_mean - 0.5)) + 0.5 
    
    pcm = ax_tg.imshow(tgm_mean, extent=[time[0], time[-1], time[0], time[-1]], 
                       origin='lower', cmap='RdBu_r', vmin=0.3, vmax=0.7,
                       interpolation='nearest')
    
    ax_tg.axhline(0, linestyle='--', color='black', linewidth=0.5)
    ax_tg.axvline(0, linestyle='--', color='black', linewidth=0.5)
    ax_tg.set_xlabel('Test Time (s)')
    ax_tg.set_ylabel('Train Time (s)')
    ax_tg.set_title('Temporal Generalization', fontsize=10, pad=10)

    # Colorbar for TGM
    cbar = plt.colorbar(pcm, ax=ax_tg)
    cbar.set_label('AUC')

    return fig