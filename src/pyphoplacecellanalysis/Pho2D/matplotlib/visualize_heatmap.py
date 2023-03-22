"""
title: visualize_heatmap.py
date: 2023-03-22 15:11:25
tags: python, matplotlib, heatmap, visualization
"""
import numpy as np
import matplotlib.pyplot as plt


def visualize_heatmap(data, show_value_labels=False, title="Simple Heatmap", show_xticks=False, show_yticks=False, show_colorbar=False):
    """
    Creates a simple heatmap visualization of the given 2D numpy array data.

	from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap
	data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
	fig, ax, im = visualize_heatmap(data)

    """
    # Reshape the 1D array into a 2D array with a single row
    if data.ndim == 1:
        data = np.reshape(data, (1, -1))
    
    fig, ax = plt.subplots(figsize=(20, 8))
    im = ax.imshow(data)

    if show_colorbar:
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)

    if show_xticks:
        # Set x-ticks and tick labels
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels(np.arange(data.shape[1]))
        # Rotate the x-tick labels and set their alignment
        plt.xticks(rotation=45, ha="right")
        plt.setp(ax.get_xticklabels(), fontsize=8)
    else:
        # Hide x-ticks and tick labels
        ax.set_xticks([])
        ax.set_xticklabels([])

    if show_yticks:
        # Set y-ticks and tick labels
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_yticklabels(np.arange(data.shape[0]))
    else:
        # Hide y-ticks and tick labels
        ax.set_yticks([])
        ax.set_yticklabels([])

    # Loop over data dimensions and create text annotations.
    if show_value_labels:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text = ax.text(j, i, data[i, j], ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    
    return fig, ax, im