from io import StringIO
import time
import sys
from copy import deepcopy

import numpy as np
import pandas as pd

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets # pyqtgraph is only currently used for its Qt imports
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import vedo
from vedo import Mesh, Cone, Cross3D, Plotter, printc, Glyph
from vedo import Rectangle, Lines, Plane, Axes, merge, colorMap # for StaticVedo_3DRasterHelper
from vedo import Volume, ProgressBar, show, settings, printc

# from pyphocorehelpers.plotting.vedo_qt_helpers import MainVedoPlottingWindow

from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.gui.Vedo.vedo_helpers import VedoHelpers # for vedo_get_camera_debug_info

# import qdarkstyle


from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
from pyphoplacecellanalysis.General.Mixins.DisplayHelpers import debug_print_axes_locations
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.SpikeRasterBase import SpikeRasterBase
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphoplacecellanalysis.GUI.Vedo.VedoMeshManipulatable import VedoPlotterHelpers
from pyphoplacecellanalysis.GUI.Vedo.Vedo3DStatic import StaticVedo_3DRasterHelper

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.RenderTimeCurvesMixin import VedoSpecificTimeCurvesMixin


# class Spike3DRaster_Vedo(SimplePlayPauseWithExternalAppMixin, SpikeRasterBottomFrameControlsMixin, VedoSpecificTimeCurvesMixin, SpikeRasterBase):
class Spike3DRaster_Vedo(VedoSpecificTimeCurvesMixin, SpikeRasterBase):
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
    
    # Application/Window Configuration Options:
    applicationName = 'Spike3DRaster_Vedo'
    windowName = 'Spike3DRaster_Vedo'
    
    SpeedBurstPlaybackRate = 16.0
    PlaybackUpdateFrequency = 0.04 # in seconds
    
     # GUI Configuration Options:
    WantsRenderWindowControls = False
    WantsPlaybackControls = False    

    af = QtCore.Qt.AlignmentFlag
    # a dict that maps from QtCore.Qt.AlignmentFlag to the strings that Vedo's Text2D function accepts to position text
    qt_to_vedo_alignment_dict = {(af.AlignTop | af.AlignLeft):'top-left', 
                                (af.AlignTop | af.AlignRight):'top-right', 
                                (af.AlignBottom | af.AlignLeft):'bottom-left', 
                                (af.AlignBottom | af.AlignRight):'bottom-right'}
    
        
    @property
    def overlay_text_lines_dict(self):
        """The lines of text to be displayed in the overlay."""    
        af = QtCore.Qt.AlignmentFlag

        lines_dict = dict()
        
        camera_debug_text = VedoHelpers.vedo_get_camera_debug_info(self.ui.plt.camera)
            
        lines_dict[af.AlignTop | af.AlignLeft] = ['TL',
                                                  camera_debug_text]
        lines_dict[af.AlignTop | af.AlignRight] = ['TR', 
                                                   f"n_cells : {self.n_cells}",
                                                   f'render_window_duration: {self.render_window_duration} [sec]',
                                                #    f'animation_time_step: {self.animation_time_step}',
                                                   f'temporal_axis_length: {self.temporal_axis_length}',
                                                   f'total_data_duration: {self.total_data_duration} [sec]',
                                                   f'total_data_temporal_axis_length: {self.total_data_temporal_axis_length}',
                                                   f'temporal_zoom_factor: {self.temporal_zoom_factor}']
        lines_dict[af.AlignBottom | af.AlignLeft] = ['BL', 
                                                   f'active_time_window: {self.spikes_window.active_time_window}',
                                                   f'total_df_start_end_times: {self.spikes_window.total_df_start_end_times}',
                                                #    f'playback_rate_multiplier: {self.playback_rate_multiplier}'
                                                   ]
        lines_dict[af.AlignBottom | af.AlignRight] = ['BR']    
        return lines_dict
    
    
    @property
    def overlay_vedo_text_lines_dict(self):
        """The overlay_vedo_text_lines_dict property."""
        return {self.qt_to_vedo_alignment_dict[k]:v for (k,v) in self.overlay_text_lines_dict.items()}
    
    
    @property
    def total_data_duration(self):
        """ The duration (in seconds) of all data in self.spikes_window."""
        return (self.spikes_window.total_data_end_time - self.spikes_window.total_data_start_time)
    
    @property
    def total_data_temporal_axis_length(self):
        """The equivalent of self.temporal_axis_length but for all data instead of just the active window."""
        return self.temporal_zoom_factor * self.total_data_duration
    
    
    ######  Get/Set Properties ######:

    # @property
    # def axes_walls_z_height(self):
    #     """The axes_walls_z_height property."""
    #     return self._axes_walls_z_height
    
    @property
    def floor_z(self):
        """The offset of the floor in the z-axis."""
        # return -10
        return 0
    
    @property
    def back_wall_y(self):
        """The y position location of the green back (Y=0) axes wall plane."""
        return self.n_half_cells
    
    @property
    def plt(self):
        """The plt property."""
        return self.ui.plt
    @plt.setter
    def plt(self, value):
        raise NotImplementedError # currently property should be read-only via this accessor
        self.ui.plt = value

    def __init__(self, params=None, spikes_window=None, playback_controller=None, neuron_colors=None, neuron_sort_order=None, application_name=None, **kwargs):
        super(Spike3DRaster_Vedo, self).__init__(params=params, spikes_window=spikes_window, playback_controller=playback_controller, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=application_name, **kwargs)
        # Initialize member variables:
        
        # Helper container variables
        # self.enable_debug_print = False
        self.enable_debug_widgets = False
        self.enable_debug_print = False
        
        # Helper Mixins: INIT:
        
        # Init the TimeCurvesViewMixin for 3D Line plots:
        ### No plots will actually be added until self.add_3D_time_curves(plot_dataframe) is called with a valid dataframe.
        self.TimeCurvesViewMixin_on_init()
        
        if self.params.wantsPlaybackControls:
            self.SpikeRasterBottomFrameControlsMixin_on_init()
                    
        # Connect signals:
        self.temporal_mapping_changed.connect(self.on_adjust_temporal_spatial_mapping)
        self.spikes_window.timeWindow.window_duration_changed_signal.connect(self.on_adjust_temporal_spatial_mapping) # this signal isn't working
        self.unit_sort_order_changed_signal.connect(self.on_unit_sort_order_changed)

        # Initialize and start vedo update timer:
        self.initialize_timer()

    def setup(self):
        """ setup() is called before self.buildUI(), etc.
            self.plots
        
        """
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        
        self.app = pg.mkQApp(self.applicationName)
        
        # Configure vedo settings:
        settings.allowInteraction = True
        # "depth peeling" may improve the rendering of transparent objects
        settings.useDepthPeeling = True
        settings.multiSamples = 2  # needed on OSX vtk9
            
        # Custom Member Variables:
        self.enable_epoch_rectangle_meshes = False
        self.enable_debug_print = False
        self.enable_debug_widgets = True
        
        
        # Config
        """ Adds required params to self.params:
            spike_height_z (default 4.0): the z height of the spikes
            center_mode (allowed: ['starting_at_zero', 'zero_centered']), (default 'starting_at_zero'): I believe how everything is layed out relative to the origin
            bin_position_mode (allowed: ['bin_center', 'left_edges']), (default 'bin_center'): specifies how the bins are positioned??
        """
        self.params.setdefault('spike_height_z', 4.0)
        # self.params.setdefault('spike_start_z', self.floor_z) # spike_start_z should be overriden with computed value.
        # self.params.setdefault('spike_end_z', -6.0) # spike_end_z should be overriden with computed value.
        self.params.setdefault('center_mode', 'starting_at_zero')
        self.params.setdefault('bin_position_mode', 'bin_center')
        
        # self.params.spike_height_z = 4.0
        self.params.spike_start_z = self.floor_z # self.floor_z
        self.params.spike_end_z = self.params.spike_start_z + self.params.spike_height_z
        
        # self.params.max_y_pos = 50.0
        # self.params.max_z_pos = 10.0
        
        # max_y_all_data = self.spikes_df['visualization_raster_y_location'].nanmax()
        
        self.params.max_y_pos = 10.0
        self.params.max_z_pos = max(self.params.spike_end_z, (self.floor_z + 1.0))
        
        # self.params.center_mode = 'zero_centered'
        # self.params.center_mode = 'starting_at_zero'
        # self.params.bin_position_mode = 'bin_center'
        # self.params.bin_position_mode = 'left_edges'
        
        # by default we want the time axis to approximately span -20 to 20. So we set the temporal_zoom_factor to 
        # self.params.temporal_zoom_factor = 1.0
        self.params.temporal_zoom_factor = 40.0 / float(self.render_window_duration)
        # self.params.temporal_zoom_factor = 1000.0      
        
        self.params.enable_epoch_rectangle_meshes = self.enable_epoch_rectangle_meshes
        self.params.active_cell_colormap_name = 'rainbow'
        
        # Plots Structures:
        self.plots.meshes = dict()
                
        # TODO: Setup self.epochs_df:
        if not self.enable_epoch_rectangle_meshes:
            self.epochs_df = None
        else:
            raise NotImplementedError
        
        if 'neuron_IDX' not in self.spikes_df.columns:
            # self.spikes_df['neuron_IDX'] = self.spikes_df['fragile_linear_neuron_IDX'].copy() # TODO: this is bad! The self.get_neuron_id_and_idx(...) function doesn't work!
            # note that this is very slow, but works:
            print(f'neuron_IDX column missing. rebuilding (this might take a minute or two)...')
            included_cell_INDEXES = np.array([self.get_neuron_id_and_idx(neuron_id=an_included_cell_ID)[0] for an_included_cell_ID in self.spikes_df['aclu'].to_numpy()]) # get the indexes from the cellIDs
            self.spikes_df['neuron_IDX'] = included_cell_INDEXES.copy()

        # Determine the y-values corresponding to the series identity
        self._series_identity_y_values = None
        self.update_series_identity_y_values()
        
        if 'visualization_raster_y_location' not in self.spikes_df.columns:
            print(f'visualization_raster_y_location column missing. rebuilding (this might take a minute or two)...')
            # Compute the y for all windows, not just the current one:
            # y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='bin_center', side_bin_margins = self.params.side_bin_margins)
            # all_y = [y[a_cell_id] for a_cell_id in self.spikes_df['neuron_IDX'].to_numpy()]
            all_y = [self.fragile_linear_neuron_IDX_to_spatial(self.cell_id_to_fragile_linear_neuron_IDX_map[a_cell_id]) for a_cell_id in self.spikes_df['neuron_IDX'].to_numpy()]
            
            self.spikes_df['visualization_raster_y_location'] = all_y # adds as a column to the dataframe. Only needs to be updated when the number of active units changes
            # max_y_all_data = np.nanmax(all_y) # self.spikes_df['visualization_raster_y_location'] 

        max_y_all_data = np.nanmax(self.spikes_df['visualization_raster_y_location'].to_numpy()) # self.spikes_df['visualization_raster_y_location'] 
        self.params.max_y_pos = max(10.0, max_y_all_data)
        
        # Vedo-specific Timer Variables:
        self.params.dt = None # update every dt milliseconds
        self.timerId = None
        self.timer_tick_counter = 0
        self.timerevt = None

        # Vedo Plotter Interaction Mode:
        self.params.interaction_mode = "TrackballCamera"
        # self.params.interaction_mode = "TrackballActor"
        # self.params.interaction_mode = "JoystickCamera"
        
        # How the camera is updated when the active window changes:
        self.params.camera_update_mode = 'None'
        # self.params.camera_update_mode = 'ResetCamera' # calls resetCamera() which resets the camera to include only the current items with useBounds == True
        # self.params.camera_update_mode = 'CenterToActiveWindow' # centers the camera's x-position to the current active window, leaving the other coords constant. 
        
        # Helper Mixins: SETUP:
        if self.params.wantsPlaybackControls:
            self.SpikeRasterBottomFrameControlsMixin_on_setup()
        
        
    def buildUI(self):
        """ for QGridLayout
            addWidget(widget, row, column, rowSpan, columnSpan, Qt.Alignment alignment = 0)
        """
        self.ui = PhoUIContainer()

        self.ui.frame = QtWidgets.QFrame()
        self.ui.frame.setObjectName('root_frame')
        self.ui.frame_layout = QtWidgets.QVBoxLayout()
        self.ui.frame_layout.setObjectName('root_frame_layout')
        
        self.ui.layout = QtWidgets.QGridLayout()
        self.ui.layout.setObjectName('root_layout')
        self.ui.layout.setContentsMargins(0, 0, 0, 0)
        self.ui.layout.setVerticalSpacing(0)
        self.ui.layout.setHorizontalSpacing(0)
        self.setStyleSheet("background : #1B1B1B; color : #727272")
        
        
        # Set-up the rest of the Qt window
        # button = QtWidgets.QPushButton("My Button makes the cone red")
        # button.setToolTip('This is an example button')
        # button.clicked.connect(self.onClick)
 
        #### Build Graphics Objects #####
        self._buildGraphics()
        
        # Helper Mixins: buildUI:
        if self.params.wantsPlaybackControls:
            self.ui.bottom_controls_frame, self.ui.bottom_controls_layout = self.SpikeRasterBottomFrameControlsMixin_on_buildUI()
            self.ui.frame_layout.addWidget(self.ui.bottom_controls_frame) # add the button controls
            
        # TODO: Register Functions:
        # self.ui.bottom_controls_frame.
        
        # setup self.ui.frame_layout:
        # self.ui.frame_layout.addWidget(self.ui.vtkWidget)
        # self.ui.frame_layout.addWidget(button)

        
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
        self.setWindowTitle(self.windowName)
        # Connect window update signals
        ## NOTE: this doesn't need to be done because the base class does it!

        self.ui.plt.show(mode=self.params.interaction_mode) # , axes=1                  # <--- show the vedo rendering
        
        if self.enable_show_on_init:
            self.show() # <--- show the Qt Window

        
    def _update_spike_raster_lines_mesh(self):
        """ requires that the lines raster mesh (all_spike_lines) already exists, in which case it just updates its points without recreating it. """
        all_spike_lines = self.plots.meshes.get('all_spike_lines', None)
        if all_spike_lines is not None:
            all_spike_t = self.spikes_df[self.spikes_df.spikes.time_variable_name].to_numpy() # this will map
            # all_spike_x = DataSeriesToSpatial.temporal_to_spatial_map(all_spike_t, self.spikes_window.total_data_start_time, self.spikes_window.total_data_end_time, self.temporal_axis_length, center_mode=self.params.center_mode)
            all_spike_x = DataSeriesToSpatial.temporal_to_spatial_map(all_spike_t, self.spikes_window.total_data_start_time, self.spikes_window.total_data_end_time, self.total_data_temporal_axis_length, center_mode=self.params.center_mode)
            curr_spike_points = all_spike_lines.points() # get all the points x-points
            curr_spike_points[:, 0] = all_spike_x.repeat(2) # repeat each element twice so that it's of the correct form for .points()
            all_spike_lines.points(curr_spike_points) # update the points
            
        else:
            raise NotImplementedError        
        
        
    def _buildGraphics(self, local_enable_debug_print=False):
        """ Implementors must override this method to build the main graphics object and add it at layout position (0, 0)"""
        # vedo_qt_main_window = MainVedoPlottingWindow() # Create the main window with the vedo plotter
        self.ui.vtkWidget = QVTKRenderWindowInteractor(self.ui.frame)
        # Create renderer and add the vedo objects and callbacks
        self.ui.plt = Plotter(qtWidget=self.ui.vtkWidget, title='Pho Vedo MainVedoPlottingWindow Test', bg='#111111', bg2='#222222')
        self.id1 = self.ui.plt.addCallback("mouse click", self.onMouseClick)
        self.id2 = self.ui.plt.addCallback("key press",   self.onKeypress)

        # Build All Meshes:
        ## CRITICAL: for some reason removing this one line that adds the cone mesh makes the whole thing break, throwing some error about not being able to assign to tuples.
        # self.ui.plt += Cone() # add test cone mesh
        self.ui.plt += Cross3D()
        
        """ Have:
        self.params.spike_start_z
        self.params.spike_end_z
        
        """
        if self.enable_epoch_rectangle_meshes:
            rect_meshes = StaticVedo_3DRasterHelper.plot_epoch_rects_vedo(self.epochs_df, max_y_pos=self.params.max_y_pos, max_z_pos=self.params.max_z_pos, should_save=False)
            rect_meshes.useBounds(False) # Says to ignore the bounds of the rect_meshes
            rect_meshes.color(1).lighting('glossy')
        else:
            rect_meshes = None
            
        self.plots.meshes['rect_meshes'] = rect_meshes
        
        # rebuild the position range for each unit along the y-axis:
        self.update_series_identity_y_values()
        ## TODO: note that this doesn't currently affect self.spikes_df['visualization_raster_y_location'], which is what determines where spikes are placed.
        
        # replaces StaticVedo_3DRasterHelper.build_spikes_lines(...) with a version optimized for Spike3DRaster_Vedo:
        all_spike_t = self.spikes_df[self.spikes_df.spikes.time_variable_name].to_numpy() # this will map
        
        
        # all_spike_x = DataSeriesToSpatial.temporal_to_spatial_map(all_spike_t, self.spikes_window.total_data_start_time, self.spikes_window.total_data_end_time, self.temporal_axis_length, center_mode=self.params.center_mode)
        all_spike_x = DataSeriesToSpatial.temporal_to_spatial_map(all_spike_t, self.spikes_window.total_data_start_time, self.spikes_window.total_data_end_time, self.total_data_temporal_axis_length, center_mode=self.params.center_mode)
        curr_spike_y = self.spikes_df['visualization_raster_y_location'].to_numpy() # this will map

        # t-mode:
        # startPoints = np.vstack((curr_spike_t, curr_spike_y, np.full_like(curr_spike_t, self.params.spike_start_z))).T
        # endPoints = np.vstack((curr_spike_t, curr_spike_y, np.full_like(curr_spike_t, self.params.spike_end_z))).T
        
        # x-mode:
        startPoints = np.vstack((all_spike_x, curr_spike_y, np.full_like(all_spike_x, self.params.spike_start_z))).T
        endPoints = np.vstack((all_spike_x, curr_spike_y, np.full_like(all_spike_x, self.params.spike_end_z))).T
        
        all_spike_lines = Lines(startPoints, endPoints=endPoints, c='k', alpha=0.8, lw=1, dotted=False, scale=1, res=1) # curr_spike_alphas
        # let the scalar be the y coordinate of the mesh vertices
        spike_color_ids = curr_spike_y.copy() # one per spike
        spike_point_color_ids = all_spike_lines.points()[:, 1]
        curr_spike_cmap, curr_spike_alphas, spike_point_color_ids = StaticVedo_3DRasterHelper._build_spikes_colormap(spike_point_color_ids)

        all_spike_lines.useBounds(False)
        
        y_cells = np.unique(spike_color_ids)
        n_cells = len(y_cells)
        # n_cells # 40
        
        # Builds correct colors for every spike point (specified by spike_point_color_ids) using self.params.active_cell_colormap_name
        spike_rgba_colors, spike_rgb_colors = StaticVedo_3DRasterHelper.build_spike_rgb_colors(spike_color_ids, active_cell_colormap_name=self.params.active_cell_colormap_name)
        
        all_spike_lines.lighting('default')
        ## Set Colors using explicitly computed spike_rgba_colors:
        all_spike_lines.cellIndividualColors(spike_rgba_colors*255)
        self.plots.meshes['all_spike_lines'] = all_spike_lines
        
        """ 
        # self.spikes_window.total_data_start_time
        # self.spikes_window.total_data_end_time
        
        """
        
        active_t_start, active_t_end = (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time)
        active_window_t_duration = self.spikes_window.window_duration
        if self.enable_debug_print and local_enable_debug_print:
            printc('debug_print_axes_locations(...): Active Window/Local Properties:')
            printc(f'\t(active_t_start: {active_t_start}, active_t_end: {active_t_end}), active_window_t_duration: {active_window_t_duration}')
        active_x_start, active_x_end = DataSeriesToSpatial.temporal_to_spatial_map((active_t_start, active_t_end), self.spikes_window.total_data_start_time, self.spikes_window.total_data_end_time,
                                                                                self.total_data_temporal_axis_length,
                                                                                center_mode=self.params.center_mode)
        if self.enable_debug_print:
            printc(f'\t(active_x_start: {active_x_start}, active_x_end: {active_x_end}), active_x_length: {active_x_end - active_x_start}')
        
        active_ids = self.update_active_spikes_window(x_start=active_x_start, x_end=active_x_end, max_y_pos=self.params.max_y_pos, max_z_pos=self.params.max_z_pos)
        
        if rect_meshes is not None:
            active_mesh_args = (all_spike_lines, rect_meshes)
        else:
            active_mesh_args = (all_spike_lines)

        # Builds the axes objects:
        self._build_axes_objects()
        
        # Add the meshes to the plotter:
        self.ui.plt += active_mesh_args

        # setup self.ui.frame_layout:
        self.ui.frame_layout.addWidget(self.ui.vtkWidget)
        
        ## Setup Viewport Overlay Text:
        self.ui.viewport_overlay  = vedo.CornerAnnotation().color('white').alpha(0.85)#.font("Kanopus")
        self.ui.plt += self.ui.viewport_overlay
        # self.ui.viewport_overlay.text(vedo.getColorName(self.counter), "top-center")
        # self.ui.viewport_overlay.text("..press q to quit", "bottom-right")
        for vedo_pos_key, values in self.overlay_vedo_text_lines_dict.items():
            # print(f'a_key: {a_key}, values: {values}')
            self.ui.viewport_overlay.text('\n'.join(values), vedo_pos_key)
        
        
        self.update_camera()
        


    def _build_axes_objects(self):
         # New Way of building the axes for all data (displaying evenly-spaced ticks along the x-axis with labels reflecting the corresponding t-value time:
        
        #  xValuesAndLabels: list of custom tick positions and labels [(pos1, label1), …]
        # Want to add a tick/label at the x-values corresponding to each minute.
        (active_t_start, active_t_end, active_window_t_duration), (global_start_t, global_end_t, global_total_data_duration), (active_x_start, active_x_end, active_x_duration), (global_x_start, global_x_end, global_x_duration) = debug_print_axes_locations(self)
        new_axes_x_to_time_labels = DataSeriesToSpatial.build_minute_x_tick_labels(self)
        
        if self.enable_debug_print:
            printc(f'new_axes_x_to_time_labels: {new_axes_x_to_time_labels}, global_x_start: {global_x_start}, global_x_end: {global_x_end}')

        
        all_data_axes = Axes(self.plots.meshes['all_spike_lines'], xrange=[0, self.total_data_temporal_axis_length], c='white', textScale=0.1, gridLineWidth=0.1, axesLineWidth=0.1, xTickLength=0.005*0.1, xTickThickness=0.0025*0.1,
                                xValuesAndLabels = new_axes_x_to_time_labels, useGlobal=True)
        
        VedoHelpers.recurrsively_apply_use_bounds(all_data_axes, False)
        
        active_window_bounding_box = self.plots.meshes.get('active_window_bounding_box', None)
        
        # add the axes meshes to the meshes array and to the plotter if needed:
        all_data_axes = VedoPlotterHelpers.vedo_create_if_needed(self, 'all_data_axes', all_data_axes, defer_render=True)
        # active_window_only_axes = VedoPlotterHelpers.vedo_create_if_needed(self, 'active_window_only_axes', active_window_only_axes, defer_render=True)
        
        # Set the visibility, useBounds, etc properties                
        # active_window_only_axes.SetVisibility(True)
        all_data_axes.SetVisibility(False)
        

                    
    def _update_plots(self):
        if self.enable_debug_print:
            printc(f'Spike3DRaster_Vedo._update_plots()')
        # build the position range for each unit along the y-axis:
        # y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='bin_center', side_bin_margins = self.params.side_bin_margins)
        
        all_spike_lines = self.plots.meshes.get('all_spike_lines', None)
        start_bound_plane = self.plots.meshes.get('start_bound_plane', None)
        end_bound_plane = self.plots.meshes.get('end_bound_plane', None)
        # active_window_only_axes = self.plots.meshes.get('active_window_only_axes', None)
        
        active_t_start, active_t_end = (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time)
        active_window_t_duration = self.spikes_window.window_duration
        if self.enable_debug_print:
            printc('debug_print_axes_locations(...): Active Window/Local Properties:')
            printc(f'\t(active_t_start: {active_t_start}, active_t_end: {active_t_end}), active_window_t_duration: {active_window_t_duration}')
        active_x_start, active_x_end = DataSeriesToSpatial.temporal_to_spatial_map((active_t_start, active_t_end),
                                                                        self.spikes_window.total_data_start_time, self.spikes_window.total_data_end_time,
                                                                        self.total_data_temporal_axis_length,
                                                                        center_mode=self.params.center_mode)
        
        if self.enable_debug_print:
            printc(f'\t(active_x_start: {active_x_start}, active_x_end: {active_x_end}), active_x_length: {active_x_end - active_x_start}')
        
        active_ids = self.update_active_spikes_window(x_start=active_x_start, x_end=active_x_end, max_y_pos=self.params.max_y_pos, max_z_pos=self.params.max_z_pos)
        
        ## Update the TimeCurves:
        self.TimeCurvesViewMixin_on_window_update()
        
        # Updates the camera
        self.update_camera()
        
        self.ui.plt.render()

        
    def rebuild_active_spikes_window(self):
        """ called on resize to rebuild the meshes 
        The planes don't need to be removed because update works even after window resize. The active_window_only_axes and active_window_bounding_box on the other hand do need to be removed and re-added.
            These removed meshes will be re-added on the next call to self.update_active_spikes_window(...)
        """
        VedoPlotterHelpers.vedo_remove_if_exists(self, 'active_window_only_axes', defer_render=True)
        # VedoPlotterHelpers.vedo_remove_if_exists(self, 'start_bound_plane', defer_render=True)
        VedoPlotterHelpers.vedo_remove_if_exists(self, 'active_window_bounding_box', defer_render=True)
        # VedoPlotterHelpers.vedo_remove_if_exists(self, 'end_bound_plane', defer_render=True)
        self._update_plots()
        
    def update_active_spikes_window(self, x_start=0.0, x_end=10.0, max_y_pos = 50.0, max_z_pos = 10.0, debug_print=False):
        active_spikes_lines_mesh = self.plots.meshes.get('all_spike_lines', None)
        
        # X-version:
        active_ids = active_spikes_lines_mesh.findCellsWithin(xbounds=(x_start, x_end))
        
        if debug_print:
            print(f'update_active_spikes_window(...): active_ids: {active_ids}')
        
        ## Get Colors from the celldata
        curr_cell_rgba_colors = active_spikes_lines_mesh.celldata['CellIndividualColors'] # note that the cell colors have components out of 0-255 (not 0.0-1.0)
        # set opacity component to zero for all non-window spikes
        curr_cell_rgba_colors[:,3] = 0.05*255 # np.full((spike_rgb_colors.shape[0], 1), 1.0) # Nearly invisible, but very faintly present
        # curr_cell_rgba_colors[:,3] = 0*255 # np.full((spike_rgb_colors.shape[0], 1), 1.0)
        
        if len(active_ids) > 0:
            curr_cell_rgba_colors[active_ids,3] = 1.0*255 # set alpha for active_ids to an opaque 1.0
        
        active_spikes_lines_mesh.cellIndividualColors(curr_cell_rgba_colors) # needed?
        
        # Build or update the start/end bounding planes and bounding box
        active_window_x_length = np.abs((x_end - x_start))
        active_window_x_half_length = active_window_x_length / 2.0
        active_x_center = x_start + active_window_x_half_length
        # y_depth = (max_y_pos/2.0)
        # z_height = (max_z_pos/2.0)
        plane_padding = 4.0
        y_depth = max_y_pos + plane_padding
        z_height = max_z_pos + plane_padding
        # y_center = (max_y_pos/2.0)
        # z_center = (max_z_pos/2.0)
        y_center = (y_depth/2.0)
        z_center = (z_height/2.0)
        
        # Active Window Start bounding plane:
        start_bound_plane = self.plots.meshes.get('start_bound_plane', None)        
        # Store the start_bound_plane's previous x position before updating (if it has one). This allows us to compute the delta between frames and apply relative transforms to active_window_only_axes if that's needed.
        prev_x_position = None
        if start_bound_plane is None:
            start_bound_plane = Plane(pos=(x_start, y_center, z_center), normal=(1,0,0), sx=z_height, sy=y_depth, alpha=0.5).lw(2.0).lineColor('#CCFFCC') #.x(x_start) # s is the plane size
            start_bound_plane = VedoPlotterHelpers.vedo_create_if_needed(self, 'start_bound_plane', start_bound_plane, defer_render=True)
            prev_x_position = None
        else:
            # Backup the previous x_position from the start_bound_plane before adjusting it.
            prev_x_position = start_bound_plane.x()
            
            # just update the extant one
            start_bound_plane.x(x_start)
        
        
        # Active Window Region Bounding Box:
        active_window_bounding_box = self.plots.meshes.get('active_window_bounding_box', None)
        if active_window_bounding_box is None:
            # self.params.active_window_fill_color = 'g4'
            self.params.active_window_fill_color = 'white'
            self.params.active_window_line_color = '#CCFFCC'
            self.params.active_window_line_width = 2.0
            
            
            active_window_bounding_box = vedo.Box(size=(x_start, x_end, 0.0, y_depth, 0.0, z_height), c=self.params.active_window_fill_color, alpha=0.2).lw(self.params.active_window_line_width).lineColor(self.params.active_window_line_color)
            active_window_bounding_box = VedoPlotterHelpers.vedo_create_if_needed(self, 'active_window_bounding_box', active_window_bounding_box, defer_render=True)
        else:
            # just update the extant one
            active_window_bounding_box.x(active_x_center)
            
        # Active Window Region Axes Object:
        active_window_only_axes = self.plots.meshes.get('active_window_only_axes', None)
        if active_window_only_axes is None:
            ## The axes only for the active window:
            active_window_only_axes = vedo.Axes([active_window_bounding_box],  # build axes for this set of objects
                        xtitle="window t",
                        ytitle="Cell ID",
                        ztitle="",
                        hTitleColor='red',
                        zHighlightZero=True,
                        xyFrameLine=2, yzFrameLine=1, zxFrameLine=1,
                        xyFrameColor='red',
                        # xyShift=1.05, # move xy 5% above the top of z-range
                        yzGrid=True,
                        zxGrid=True,
                        yMinorTicks=self.n_cells,
                        yLineColor='red',
                        xrange=(x_start, x_end),
                        yrange=(0.0, self.params.max_y_pos),
                        zrange=(0.0, self.params.max_z_pos)
            ) # .x(active_x_center)
            # Add the axis if needed:
            active_window_only_axes = VedoPlotterHelpers.vedo_create_if_needed(self, 'active_window_only_axes', active_window_only_axes, defer_render=True)
        else:
            # just update the extant one
            if prev_x_position is not None:
                delta_x = start_bound_plane.x() - prev_x_position
                prev_x_pos = active_window_only_axes.x() # get its old x() position
                active_window_only_axes.x(prev_x_pos + delta_x) # works for positioning but doesn't update numbers
            else:
                # there was no previous position to change from, so just skip the positioning for now.
                pass
            # active_window_only_axes.x(x_start)
            
        # Active Window End bounding plane:
        end_bound_plane = self.plots.meshes.get('end_bound_plane', None)
        if end_bound_plane is None:
            end_bound_plane = start_bound_plane.clone().lineColor('#FFCCCC').x(x_end)
            end_bound_plane = VedoPlotterHelpers.vedo_create_if_needed(self, 'end_bound_plane', end_bound_plane, defer_render=True)
        else:
            # just update the extant one
            end_bound_plane.x(x_end)
            
        # active_window_only_axes = self.plots.meshes.get('active_window_only_axes', None)
        
        return active_ids #, start_bound_plane, active_window_bounding_box, end_bound_plane
    
    

    ## Camera Position Updating Functions:
    def update_camera(self):
        """ called to update the camera's position when the active window or data is changed. 
                
        # Requires:
            self.params.camera_update_mode
        
        # camera_update_mode: How the camera is updated when the active window changes:
            'None'
            'ResetCamera' # calls resetCamera() which resets the camera to include only the current items with useBounds == True
            'CenterToActiveWindow' # centers the camera's x-position to the current active window, leaving the other coords constant. 
            
        """
        if self.params.camera_update_mode == 'ResetCamera':
            ## resetCamera() method:
            self.ui.plt.resetCamera() # resetCamera() updates the camera's position given the ignored components
            # This limits the meshes to just the active window's meshes: [start_bound_plane, end_bound_plane, active_window_only_axes]
        elif self.params.camera_update_mode == 'CenterToActiveWindow':
            ## center_camera_on_active_timewindow(): tries to compute the explicit center of the time window
            self.center_camera_on_active_timewindow()
        elif self.params.camera_update_mode == 'None':
            pass # do no automatic adjustment of the camera
        else:
            raise NotImplementedError 

    def center_camera_on_active_timewindow(self, debug_print = False):
        """ centers the camera on the current time window (in the x-position only) """
        (active_t_start, active_t_end, active_window_t_duration), (global_start_t, global_end_t, global_total_data_duration), (active_x_start, active_x_end, active_x_duration), (global_x_start, global_x_end, global_x_duration) = debug_print_axes_locations(self)
        if debug_print:
            print((active_t_start, active_t_end, active_window_t_duration), (global_start_t, global_end_t, global_total_data_duration), (active_x_start, active_x_end, active_x_duration), (global_x_start, global_x_end, global_x_duration))
        active_x_center = active_x_start + (active_x_duration/2.0)
        if debug_print:
            print(f'active_x_center: {active_x_center}')
        prev_cam_pos = self.ui.plt.camera.GetPosition() # (1793.4129435152863, 26.484467923780887, 399.31668579161686)
        if debug_print:
            print(f'previous camera position: {prev_cam_pos}')
        self.ui.plt.camera.SetPosition(active_x_center, prev_cam_pos[1], prev_cam_pos[2]) # update the camera's x position to the active_x_center, keep the other positions intact.
        if debug_print:
            print(f'updated camera position: {self.ui.plt.camera.GetPosition()}')


    ###################################
    #### EVENT HANDLERS
    ##################################
    
    def initialize_timer(self, timer_update_duration_milliseconds = 100):
        """ initializes the vedo timer """        
        self.params.dt = timer_update_duration_milliseconds # update every dt milliseconds
        self.timerId = None
        # self.isplaying = False
        self.timer_tick_counter = 0 # frame counter
        # Destroy any extant timer:
        if self.timerId is not None:
            print('destroying existing timer.')
            self.ui.plt.timerCallback("destroy", self.timerId)
        # Build the new timer:
        print('building new timer')
        self.timerevt = self.ui.plt.addCallback('timer', self.handle_timer)
        # Start the new timer:
        if self.enable_debug_print:
            print('starting new timer')
        self.timerId = self.ui.plt.timerCallback("create", dt=self.params.dt)

    def handle_timer(self, event):
        #####################################################################
        ### Animate your stuff here                                       ###
        #####################################################################
        #print(event)               # info about what was clicked and more
        #print(self.plotter.actors) # to access object from the internal list
        
        # Update the additional display lines information on the overlay:
        for vedo_pos_key, values in self.overlay_vedo_text_lines_dict.items():
            # print(f'a_key: {a_key}, values: {values}')
            self.ui.viewport_overlay.text('\n'.join(values), vedo_pos_key)
            
        self.plt.render() # is this needed?
        self.timer_tick_counter += 1
        

    @pyqtExceptionPrintingSlot()
    def on_adjust_temporal_spatial_mapping(self):
        """ called when the spatio-temporal mapping property is changed.
        
        Should change whenever any of the following change:
            self.temporal_zoom_factor
            self.render_window_duration
            
        """
        self.update_series_identity_y_values()
        self.rebuild_active_spikes_window()
        self._update_spike_raster_lines_mesh() # updates the extant spikes raster lines mesh
        self._build_axes_objects() # rebuild the axes objects
        self._update_plots()

    def onMouseClick(self, evt):
        printc("You have clicked your mouse button. Event info:\n", evt, c='y')

    def onKeypress(self, evt):
        printc("You have pressed key:", evt.keyPressed, c='b')

    @pyqtExceptionPrintingSlot()
    def onClick(self):
        printc("..calling onClick")
        self.ui.plt.actors[0].color('red').rotateZ(40)
        self.ui.plt.interactor.Render()


    # Overrides the superclasses' onClose(self) function:
    def onClose(self):
        #Disable the interactor before closing to prevent it
        #from trying to act on already deleted items
        printc(f'vedo override - onClose()')
        self.debug_print_instance_info()
        printc("..calling onClose")
        
        # Deleting any timer
        if self.timerId is not None:
            print('deleting extant timer...')
            self.plt.timerCallback("destroy", self.timerId)
            print('\t done.')
        # Close widget:
        self.ui.vtkWidget.close()
        
        # Emit the close signal:
        self.close_signal.emit() # emit to indicate that we're closing this window


    # Overrides for DataSeriesToSpatialTransformingMixin:
    
    ## Series X positions:
    def temporal_to_spatial(self, temporal_data):
        """ transforms the times in temporal_data to a spatial offset (such as the x-positions for a 3D raster plot) """
        return DataSeriesToSpatial.temporal_to_spatial_map(temporal_data, self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time, self.total_data_temporal_axis_length, center_mode=self.params.center_mode)


    ## Series Y positions:
    
    @property
    def series_identity_y_values(self):
        """The series_identity_y_values property."""
        return self._series_identity_y_values
    
    def update_series_identity_y_values(self):
        """ updates the fixed self._series_identity_y_values using the DataSeriesToSpatial.build_series_identity_axis(...) function.
        
        Should be called whenever:
        self.n_cells, 
        self.params.center_mode,
        self.params.bin_position_mode
        self.params.side_bin_margins
        
        values change.
        """
        self._series_identity_y_values = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)
        
    
    ## Required for DataSeriesToSpatialTransformingMixin
    # TODO: convert all instances of self.y[i], etc into using self.fragile_linear_neuron_IDX_to_spatial(...)
    def fragile_linear_neuron_IDX_to_spatial(self, fragile_linear_neuron_IDXs):
        """ transforms the fragile_linear_neuron_IDXs in fragile_linear_neuron_IDXs to a spatial offset (such as the y-positions for a 3D raster plot) """
        # build the position range for each unit along the y-axis:
        # rebuild the position range for each unit along the y-axis:
        if self.series_identity_y_values is None:
            # rebuild self.series_identity_y_values
            self.update_series_identity_y_values()
    
        fragile_linear_neuron_IDX_series_indicies = self.unit_sort_order[fragile_linear_neuron_IDXs] # get the appropriate series index for each fragile_linear_neuron_IDX given their sort order
        return self.series_identity_y_values[fragile_linear_neuron_IDX_series_indicies]
    

    def update(self, sort_changed=True, colors_changed=True):
        """ refreshes the raster when the colors or sort change. 
        
        NOTE: Unlike Spike2DRaster, sorts seem to be able to be updated independent of colors.
        
        """
        if sort_changed:
            # rebuild the position range for each unit along the y-axis:
            self.update_series_identity_y_values()
            self._update_plots()
                        
        if colors_changed:
            # TODO: Impplement for Vedo version:
            # self.rebuild_main_gl_line_plots_if_needed()
            # for i, a_fragile_linear_neuron_IDX in enumerate(self.fragile_linear_neuron_IDXs):
            #     # color= (N,4) array of floats (0.0-1.0) or tuple of floats specifying a single color for the entire item.
            #     curr_color = self.params.neuron_qcolors[a_fragile_linear_neuron_IDX] # get the pre-build color
            #     self.ui.gl_line_plots[i].setData(color=curr_color) # update the current data        
            self._update_plots()
            
    
    @pyqtExceptionPrintingSlot(object)
    def on_unit_sort_order_changed(self, new_sort_order):
        print(f'unit_sort_order_changed_signal(new_sort_order: {new_sort_order})')        
        self.update(sort_changed=True, colors_changed=False)
        

    @pyqtExceptionPrintingSlot(object)
    def on_neuron_colors_changed(self, neuron_id_color_update_dict):
        """ Called when the neuron colors have finished changing (changed) to update the rendered elements.
        """
        print(f'Spike3DRaster_Vedo.neuron_id_color_update_dict: {neuron_id_color_update_dict}')
        self.update(sort_changed=False, colors_changed=True)
        
        
# josfd