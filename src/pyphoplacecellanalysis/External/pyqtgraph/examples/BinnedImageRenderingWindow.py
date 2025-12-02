"""
Example demonstrating BasicBinnedImageRenderingWindow - a window for displaying 
binned 2D data as heatmaps with customizable colormaps, crosshairs, and color bars.

This example shows how to use BasicBinnedImageRenderingWindow to display 
2D binned data matrices with custom bin labels and interactive features.
"""

import sys
import os

# Add the parent directory to the path to allow importing from pyphoplacecellanalysis
examples_dir = os.path.dirname(os.path.abspath(__file__))
pypho_root = os.path.abspath(os.path.join(examples_dir, '../../..'))
if pypho_root not in sys.path:
    sys.path.insert(0, pypho_root)

import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability

app = pg.mkQApp("BasicBinnedImageRenderingWindow Example")

if __name__ == '__main__':
    # Create example binned data - a 2D heatmap
    n_xbins = 20
    n_ybins = 20
    
    # Create bin labels
    xbins = np.linspace(0, 100, n_xbins)
    ybins = np.linspace(0, 100, n_ybins)
    
    # Create example matrix data - a 2D Gaussian-like pattern
    x_center, y_center = 50, 50
    x_coords = np.linspace(0, 100, n_xbins)
    y_coords = np.linspace(0, 100, n_ybins)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Create a Gaussian-like pattern
    sigma = 15
    matrix = np.exp(-((X - x_center)**2 + (Y - y_center)**2) / (2 * sigma**2))
    
    # Add some noise
    matrix += 0.3 * np.sin(X / 10) * np.cos(Y / 10)
    matrix += np.random.normal(0, 0.1, size=matrix.shape)
    matrix = np.clip(matrix, 0, 1)
    
    # Create the window with the first plot
    window = BasicBinnedImageRenderingWindow(
        matrix=matrix,
        xbins=xbins,
        ybins=ybins,
        name='example_heatmap',
        title="Example Heatmap",
        variable_label='Intensity',
        scrollability_mode=LayoutScrollability.NON_SCROLLABLE,
        wants_crosshairs=True,
        color_bar_mode='each',
        grid_opacity=0.4
    )
    
    # Add additional example plots
    matrix2 = np.zeros((n_xbins, n_ybins))
    for i in range(n_xbins):
        for j in range(n_ybins):
            matrix2[i, j] = ((i // 3) + (j // 3)) % 2
    
    window.add_data(
        row=1, col=0,
        matrix=matrix2,
        xbins=xbins,
        ybins=ybins,
        name='checkerboard',
        title='Checkerboard Pattern',
        variable_label='Pattern Value'
    )
    
    # Create a third pattern
    matrix3 = np.sqrt((X - x_center)**2 + (Y - y_center)**2) / 50
    matrix3 = np.clip(matrix3, 0, 1)
    
    window.add_data(
        row=2, col=0,
        matrix=matrix3,
        xbins=xbins,
        ybins=ybins,
        name='radial_gradient',
        title='Radial Gradient',
        variable_label='Distance'
    )
    
    pg.exec()
