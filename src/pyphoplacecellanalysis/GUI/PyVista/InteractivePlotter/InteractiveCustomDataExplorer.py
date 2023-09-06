#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho


"""
import numpy as np
import pyvista as pv
from pyvistaqt.plotting import MultiPlotter

from pyphoplacecellanalysis.Pho3D.PyVista.spikeAndPositions import perform_plot_flat_arena
from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.InteractiveDataExplorerBase import InteractiveDataExplorerBase

# from neuropy.utils.debug_helpers import safely_accepts_kwargs

class InteractiveCustomDataExplorer(InteractiveDataExplorerBase):
    """ This class is the minimal concrete implementation of the abstract InteractiveDataExplorerBase.
    
    This can be used as a very simple class to be extended with custom data.
    
    """
    def __init__(self, active_config, active_session, extant_plotter=None):
        super(InteractiveCustomDataExplorer, self).__init__(active_config, active_session, extant_plotter, data_explorer_name='CustomDataExplorer')
        self._setup()
    
    def _setup_variables(self):
        pass

    def _setup_visualization(self):
        pass


    ######################
    # General Plotting Method:
   
    def plot(self, pActivePlotter=None, default_plotting=True):
        ################################################
        ### Build Appropriate Plotter and set it up:
        #####################
        # Only Create a new BackgroundPlotter if it's needed:
        self.p = InteractiveCustomDataExplorer.build_new_plotter_if_needed(pActivePlotter, shape=self.active_config.plotting_config.subplots_shape, title=self.data_explorer_name, plotter_type=self.active_config.plotting_config.plotter_type)
        # p.background_color = 'black'
        
        # Plot the flat arena
        if default_plotting:
            if isinstance(self.p, MultiPlotter):
                # for p in self.p:
                p = self.p[0,0] # the first plotter
                    
            else:
                p = self.p
                
        self.plots['maze_bg'] = perform_plot_flat_arena(p, self.x, self.y, bShowSequenceTraversalGradient=False, smoothing=self.active_config.plotting_config.get('use_smoothed_maze_rendering', True))

        p.hide_axes()
        # self.p.camera_position = 'xy' # Overhead (top) view
        # apply_close_overhead_zoomed_camera_view(self.p)
        # apply_close_perspective_camera_view(self.p)
        p.render() # manually render when needed
        
        return self.p
