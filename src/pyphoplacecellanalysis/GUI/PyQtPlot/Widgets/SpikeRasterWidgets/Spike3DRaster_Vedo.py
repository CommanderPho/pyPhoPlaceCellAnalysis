import time
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, to_hex # for neuron colors to_hex

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets # pyqtgraph is only currently used for its Qt imports
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from vedo import Mesh, Cone, Plotter, printc, Glyph
from vedo import Volume, ProgressBar, show, settings

from pyphocorehelpers.plotting.vedo_qt_helpers import MainVedoPlottingWindow

from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
# import qdarkstyle

from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.SpikeRasterBase import SpikeRasterBase


class Spike3DRaster_Vedo(SpikeRasterBase):
    """ **Vedo version** - Displays a 3D version of a raster plot with the spikes occuring along a plane. 
    
    TODO: CURRENTLY UNIMPLEMENTED I THINK. Switched back to Spike3DRaster as it works well and good enough.
    
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster_Vedo import Spike3DRaster_Vedo
        curr_epoch_name = 'maze1'
        curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
        curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
        curr_spikes_df = curr_sess.spikes_df
        spike_raster_plt = Spike3DRaster_Vedo(curr_spikes_df, window_duration=4.0, window_start_time=30.0)
    """
    
    temporal_mapping_changed = QtCore.pyqtSignal() # signal emitted when the mapping from the temporal window to the spatial layout is changed
    close_signal = QtCore.pyqtSignal() # Called when the window is closing. 
    
    SpeedBurstPlaybackRate = 16.0
    PlaybackUpdateFrequency = 0.04 # in seconds
     # GUI Configuration Options:
    WantsRenderWindowControls = False
    WantsPlaybackControls = False
    

    @property
    def overlay_text_lines_dict(self):
        """The lines of text to be displayed in the overlay."""    
        af = QtCore.Qt.AlignmentFlag

        lines_dict = dict()
        
        lines_dict[af.AlignTop | af.AlignLeft] = ['TL']
        lines_dict[af.AlignTop | af.AlignRight] = ['TR', 
                                                   f"n_cells : {self.n_cells}",
                                                   f'render_window_duration: {self.render_window_duration}',
                                                   f'animation_time_step: {self.animation_time_step}',
                                                   f'temporal_axis_length: {self.temporal_axis_length}',
                                                   f'temporal_zoom_factor: {self.temporal_zoom_factor}']
        lines_dict[af.AlignBottom | af.AlignLeft] = ['BL', 
                                                   f'active_time_window: {self.spikes_window.active_time_window}',
                                                   f'playback_rate_multiplier: {self.playback_rate_multiplier}'
                                                   ]
        lines_dict[af.AlignBottom | af.AlignRight] = ['BR']    
        return lines_dict
    
    
    ######  Get/Set Properties ######:

    @property
    def axes_walls_z_height(self):
        """The axes_walls_z_height property."""
        return self._axes_walls_z_height
    
    @property
    def z_floor(self):
        """The offset of the floor in the z-axis."""
        return -10
    
    @property
    def y_backwall(self):
        """The y position location of the green back (Y=0) axes wall plane."""
        return self.n_half_cells
    

    def __init__(self, spikes_df, *args, window_duration=15.0, window_start_time=0.0, neuron_colors=None, neuron_sort_order=None, **kwargs):
        super(Spike3DRaster_Vedo, self).__init__(spikes_df, *args, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, **kwargs)
        # SpikeRasterBase.__init__(spikes_df, *args, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, **kwargs)
        # Initialize member variables:
        
        # Helper container variables
        # self.enable_debug_print = False
        self.enable_debug_widgets = False
        
        self.enable_debug_print = True
        
       
        if 'cell_idx' not in self.spikes_df.columns:
            # self.spikes_df['cell_idx'] = self.spikes_df['unit_id'].copy() # TODO: this is bad! The self.get_neuron_id_and_idx(...) function doesn't work!
            # note that this is very slow, but works:
            print(f'cell_idx column missing. rebuilding (this might take a minute or two)...')
            included_cell_INDEXES = np.array([self.get_neuron_id_and_idx(neuron_id=an_included_cell_ID)[0] for an_included_cell_ID in self.spikes_df['aclu'].to_numpy()]) # get the indexes from the cellIDs
            self.spikes_df['cell_idx'] = included_cell_INDEXES.copy()

        if 'visualization_raster_y_location' not in self.spikes_df.columns:
            print(f'visualization_raster_y_location column missing. rebuilding (this might take a minute or two)...')
            # Compute the y for all windows, not just the current one:
            y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode='zero_centered', bin_position_mode='bin_center', side_bin_margins = self.params.side_bin_margins)
            all_y = [y[a_cell_id] for a_cell_id in self.spikes_df['cell_idx'].to_numpy()]
            self.spikes_df['visualization_raster_y_location'] = all_y # adds as a column to the dataframe. Only needs to be updated when the number of active units changes
                    
                    
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        
        # build the UI components:
        # self.buildUI()


    def buildUI(self):
        """ for QGridLayout
            addWidget(widget, row, column, rowSpan, columnSpan, Qt.Alignment alignment = 0)
        """
        self.ui = PhoUIContainer()

        self.ui.frame = QtWidgets.QFrame()
        self.ui.frame_layout = QtWidgets.QVBoxLayout()
        
        self.ui.layout = QtWidgets.QGridLayout()
        self.ui.layout.setContentsMargins(0, 0, 0, 0)
        self.ui.layout.setVerticalSpacing(0)
        self.ui.layout.setHorizontalSpacing(0)
        self.setStyleSheet("background : #1B1B1B; color : #727272")
        
        
        # Set-up the rest of the Qt window
        button = QtWidgets.QPushButton("My Button makes the cone red")
        button.setToolTip('This is an example button')
        button.clicked.connect(self.onClick)
 
        #### Build Graphics Objects #####
        self._buildGraphics()
        
        # setup self.ui.frame_layout:
        # self.ui.frame_layout.addWidget(self.ui.vtkWidget)
        self.ui.frame_layout.addWidget(button)
        self.ui.frame.setLayout(self.ui.frame_layout)
        
        # Add the frame to the root layout
        self.ui.layout.addWidget(self.ui.frame, 0, 0)
        
        # #### Build Graphics Objects #####
        # self._buildGraphics()
        
        # if self.params.wantsPlaybackControls:
        #     # Build the bottom playback controls bar:
        #     self.setup_render_playback_controls()

        # if self.params.wantsRenderWindowControls:
        #     # Build the right controls bar:
        #     self.setup_render_window_controls() # creates self.ui.right_controls_panel

                
        # # addWidget(widget, row, column, rowSpan, columnSpan, Qt.Alignment alignment = 0)
         
        # Set the root (self) layout properties
        self.setLayout(self.ui.layout)
        self.resize(1920, 900)
        self.setWindowTitle('Spike3DRaster_Vedo')
        # Connect window update signals
        # self.spikes_window.spike_dataframe_changed_signal.connect(self.on_spikes_df_changed)
        # self.spikes_window.window_duration_changed_signal.connect(self.on_window_duration_changed)
        # self.spikes_window.window_changed_signal.connect(self.on_window_changed)
        self.spikes_window.window_updated_signal.connect(self.on_window_changed)


        self.ui.plt.show()                  # <--- show the vedo rendering
        self.show()                     # <--- show the Qt Window

    def setup(self):
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        
    
        self.app = pg.mkQApp("Spike3DRaster_Vedo")
        
        # Configure vedo settings:
        settings.allowInteraction = True
        # "depth peeling" may improve the rendering of transparent objects
        settings.useDepthPeeling = True
        settings.multiSamples = 2  # needed on OSX vtk9
            
        
        
    def _buildGraphics(self):
        """ Implementors must override this method to build the main graphics object and add it at layout position (0, 0)"""
        # vedo_qt_main_window = MainVedoPlottingWindow() # Create the main window with the vedo plotter
        self.ui.vtkWidget = QVTKRenderWindowInteractor(self.ui.frame)
        # Create renderer and add the vedo objects and callbacks
        self.ui.plt = Plotter(qtWidget=self.ui.vtkWidget)
        self.id1 = self.ui.plt.addCallback("mouse click", self.onMouseClick)
        self.id2 = self.ui.plt.addCallback("key press",   self.onKeypress)
        self.ui.plt += Cone().rotateX(20)
        # self.ui.plt.show()                  # <--- show the vedo rendering

        # setup self.ui.frame_layout:
        self.ui.frame_layout.addWidget(self.ui.vtkWidget)
        # raise NotImplementedError
    
    
    
    # def on_window_changed(self):
    #     # called when the window is updated
    #     if self.enable_debug_print:
    #         print(f'Spike3DRaster_Vedo.on_window_changed()')
    #     self._update_plots()
        
            
    def _update_plots(self):
        if self.enable_debug_print:
            print(f'Spike3DRaster_Vedo._update_plots()')
        # build the position range for each unit along the y-axis:
        # y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode='zero_centered', bin_position_mode='bin_center', side_bin_margins = self.params.side_bin_margins)
        
        # All series at once approach:
        # curr_spike_t = self.active_windowed_df[self.active_windowed_df.spikes.time_variable_name].to_numpy() # this will map
        # curr_unit_n_spikes = len(curr_spike_t)
        
        # if self.glyph is None:        
        #     # Create a mesh to be used like a symbol (a "glyph") to be attached to each point
        #     self.cone = Cone().scale(0.3) # make it smaller and orient tip to positive x
        #     # .rotateY(90) # orient tip to positive x
        #     self.glyph = Glyph(self.active_spike_render_points, self.cone)
        #     # glyph = Glyph(pts, cone, vecs, scaleByVectorSize=True, colorByVectorSize=True)
        #     self.glyph.lighting('ambient') # .cmap('Blues').addScalarBar(title='wind speed')
        # else:
        #     # already have self.glyph created, just need to update its points
        #     self.glyph.points(self.active_spike_render_points)
        pass
        
        # show with:
        # plt = show(glyph, __doc__, axes=True).close()
        
        
        # # Plot each unit one at a time:
        # for i, cell_id in enumerate(self.unit_ids):    
        #     # Filter the dataframe using that column and value from the list
        #     curr_cell_df = self.active_windowed_df[self.active_windowed_df['unit_id']==cell_id]
        #     curr_spike_t = curr_cell_df[curr_cell_df.spikes.time_variable_name].to_numpy() # this will map
        #     # efficiently get curr_spike_t by filtering for unit and column at the same time
        #     # curr_spike_t = self.active_windowed_df.loc[self.active_windowed_df.spikes.time_variable_name, (self.active_windowed_df['unit_id']==cell_id)].values # .to_numpy()
            
        #     curr_unit_n_spikes = len(curr_spike_t)
            
        #     yi = y[i] # get the correct y-position for all spikes of this cell
        #     # map the current spike times back onto the range of the window's (-half_render_window_duration, +half_render_window_duration) so they represent the x coordinate
        #     # curr_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (-self.half_render_window_duration, +self.half_render_window_duration))
        #     curr_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (-self.half_temporal_axis_length, +self.half_temporal_axis_length))
        #     # curr_paired_x = np.squeeze(interleave_elements(np.atleast_2d(curr_x).T, np.atleast_2d(curr_x).T))        
        #     curr_paired_x = curr_x.repeat(2)
            
        #     # Z-positions:
        #     # spike_bottom_zs = np.full_like(curr_x, self.params.spike_start_z)
        #     # spike_top_zs = np.full_like(curr_x, self.params.spike_end_z)
        #     # curr_paired_spike_zs = np.squeeze(interleave_elements(np.atleast_2d(spike_bottom_zs).T, np.atleast_2d(spike_top_zs).T)) # alternating top and bottom z-positions
        #     curr_paired_spike_zs = np.squeeze(np.tile(np.array([self.params.spike_start_z, self.params.spike_end_z]), curr_unit_n_spikes)) # repeat pair of z values once for each spike
        
        #     # Build lines:
        #     pts = np.column_stack([curr_paired_x, np.full_like(curr_paired_x, yi), curr_paired_spike_zs]) # the middle coordinate is the size of the x array with the value given by yi. yi must be the scalar for this cell.
        #     # plt = gl.GLLinePlotItem(pos=pts, color=curr_color, width=0.5, antialias=True, mode='lines') # mode='lines' means that each pair of vertexes draws a single line segement
        #     self.ui.gl_line_plots[i].setData(pos=pts, mode='lines') # update the current data
            
        #     # self.ui.main_gl_widget.addItem(plt)
        #     # self.ui.gl_line_plots.append(plt) # append to the gl_line_plots array
            
    
        # Update the additional display lines information on the overlay:
        # self.ui.viewport_overlay.additional_overlay_text_lines = self.overlay_text_lines
        

    def onMouseClick(self, evt):
        printc("You have clicked your mouse button. Event info:\n", evt, c='y')

    def onKeypress(self, evt):
        printc("You have pressed key:", evt.keyPressed, c='b')

    @QtCore.pyqtSlot()
    def onClick(self):
        printc("..calling onClick")
        self.ui.plt.actors[0].color('red').rotateZ(40)
        self.ui.plt.interactor.Render()

    def onClose(self):
        #Disable the interactor before closing to prevent it
        #from trying to act on already deleted items
        printc("..calling onClose")
        self.ui.vtkWidget.close()
        
        
# josfd