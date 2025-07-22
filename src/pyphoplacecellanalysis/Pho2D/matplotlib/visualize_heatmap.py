"""
title: visualize_heatmap.py
date: 2023-03-22 15:11:25
tags: python, matplotlib, heatmap, visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from pyphocorehelpers.function_helpers import function_attributes

@function_attributes(short_name='visualize_heatmap', tags=['display','matplotlib','heatmap'], input_requires=[], output_provides=[], uses=[], used_by=['build_callout_subgraphic'], creation_date='2023-03-22 15:11')
def visualize_heatmap(data, ax=None, show_value_labels=False, title="Simple Heatmap", show_xticks=False, show_yticks=False, show_colorbar=False, defer_show:bool = False, layout=None, **imshow_kwargs):
    """
    A MATPLOTLIB-based simple heatmap plot of the given 2D numpy array data.

	from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap
	data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
	fig, ax, im = visualize_heatmap(data)

    """
    # Reshape the 1D array into a 2D array with a single row
    if data.ndim == 1:
        data = np.reshape(data, (1, -1))
    
    if ax is None:
        # Create a new figure and axes for output:
        fig, ax = plt.subplots(figsize=(20, 8))
    else:
        # already provided an axes to plot into:
        fig = ax.get_figure()
        # fig.set_size_inches([23, 9.7])


    # Perform the plot:
    im = ax.imshow(data, **imshow_kwargs)
    # ax.set_xlim((xmin, xmax))
    # ax.set_ylim((ymin, ymax))

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
    if (layout is not None) and (layout == 'tight'):
        fig.tight_layout()
    
    if not defer_show:
        plt.show()
    
    return fig, ax, im


from typing import Tuple
import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphoplacecellanalysis.External.pyqtgraph_extensions.PlotWidget.CustomPlotWidget import CustomPlotWidget

@function_attributes(short_name='heatmap_pyqtgraph', tags=['pyqtgraph', 'heatmap', 'app', 'window'], input_requires=[], output_provides=[], uses=[], used_by=['plot_kourosh_activity_style_figure'], creation_date='2023-06-21 15:27', related_items=[])
def visualize_heatmap_pyqtgraph(data, win=None, show_value_labels=False, title="Simple Heatmap", show_xticks=False, show_yticks=False, show_colorbar=False, defer_show:bool = False) -> Tuple[CustomPlotWidget, pg.ImageItem]:
    """
    Creates a simple heatmap visualization of the given 2D numpy array data.

    data: a 2D numpy array
    win: a pyqtgraph PlotWidget object to plot into. If not provided, a new PlotWidget will be created.
    show_value_labels: if True, display the values in each cell
    title: the title of the plot
    show_xticks: if True, display the x-tick labels
    show_yticks: if True, display the y-tick labels
    show_colorbar: if True, display the colorbar
    defer_show: if True, do not show the plot immediately. Instead, return the plot object.

    Returns: the PlotWidget object
    
    
    Usage:
    
        from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap_pyqtgraph
        
    """

    # Reshape the 1D array into a 2D array with a single row
    if data.ndim == 1:
        data = np.reshape(data, (1, -1))

    # Create a new PlotWidget if win is not provided
    if win is None:
        app = pg.mkQApp()
        # win = pg.PlotWidget()
        win = CustomPlotWidget()
        win.setWindowTitle(title)
        if not defer_show:
            win.show() # is this needed for something?
        did_create_win = True
    else:
        did_create_win = False

    # Create an image item to display the heatmap
    img = pg.ImageItem(data)

    # Add the image item to the PlotWidget
    win.addItem(img)

    if show_colorbar:
        # Add a colorbar to the PlotWidget
        cbar = pg.GradientWidget(orientation='right')
        cbar.setColorMap(pg.ColorMap(*pg.colorTuple('bwr')))
        win.addItem(cbar)

        # Link the range of the colorbar with the image item
        cbar.linkedViewChanged(img)

    # Set the tick labels
    if show_xticks:
        ax = win.getAxis('bottom')
        ax.setTicks([[(i, str(i)) for i in range(data.shape[1])]])
        ax.setLabel(text='X Axis')
    else:
        win.hideAxis('bottom')

    if show_yticks:
        ax = win.getAxis('left')
        ax.setTicks([[(i, str(i)) for i in range(data.shape[0])]])
        ax.setLabel(text='Y Axis')
    else:
        win.hideAxis('left')

    # Display the value labels
    if show_value_labels:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                text = pg.TextItem(f"{val:.2f}")
                text.setPos(j, i)
                win.addItem(text)

    # Set the title of the plot
    win.setTitle(title)

    if did_create_win and (not defer_show):
        app.exec_()

    return win, img

