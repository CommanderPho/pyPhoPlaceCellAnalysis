from io import StringIO
import time
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, to_hex # for neuron colors to_hex

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets # pyqtgraph is only currently used for its Qt imports
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import vedo
from vedo import Mesh, Cone, Plotter, printc, Glyph
from vedo import Rectangle, Lines, Plane, Axes, merge, colorMap # for StaticVedo_3DRasterHelper
from vedo import Volume, ProgressBar, show, settings, printc

# from pyphocorehelpers.plotting.vedo_qt_helpers import MainVedoPlottingWindow

from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.gui.Vedo.vedo_helpers import VedoHelpers # for vedo_get_camera_debug_info

# import qdarkstyle

from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
from  pyphoplacecellanalysis.General.Mixins.DisplayHelpers import debug_print_axes_locations
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.SpikeRasterBase import SpikeRasterBase

from pyphoplacecellanalysis.GUI.Vedo.Vedo3DStatic import StaticVedo_3DRasterHelper

from pyphoplacecellanalysis.GUI.Qt.PlaybackControls.Spike3DRasterBottomPlaybackControlBarWidget import Spike3DRasterBottomPlaybackControlBar


class SimplePlayPauseWithExternalAppMixin:
    
    # @property
    # def animationThread(self):
    #     """The animationThread property."""
    #     return self.playback_controller
    
    @property
    def animationThread(self):
        """The animationThread property."""
        return self.playback_controller.animationThread
    
    
class Spike3DRasterBottomFrameControlsMixin:
    """ renders the UI controls for the Spike3DRaster_Vedo class 
        Follows Conventions outlined in ModelViewMixin Conventions.md
    """
    
    @QtCore.pyqtSlot()
    def Spike3DRasterBottomFrameControlsMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        pass

    @QtCore.pyqtSlot()
    def Spike3DRasterBottomFrameControlsMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass
    
    @QtCore.pyqtSlot()
    def Spike3DRasterBottomFrameControlsMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        # CALLED:
        
        # controls_frame = QtWidgets.QFrame()
        # controls_layout = QtWidgets.QHBoxLayout() # H-box layout
        
        # # controls_layout = QtWidgets.QGridLayout()
        # # controls_layout.setContentsMargins(0, 0, 0, 0)
        # # controls_layout.setVerticalSpacing(0)
        # # controls_layout.setHorizontalSpacing(0)
        # # controls_layout.setStyleSheet("background : #1B1B1B; color : #727272")
        
        # # Set-up the rest of the Qt window
        # button = QtWidgets.QPushButton("My Button makes the cone red")
        # button.setToolTip('This is an example button')
        # button.clicked.connect(self.onClick)
        # controls_layout.addWidget(button)
        
        # button2 = QtWidgets.QPushButton("<")
        # button2.setToolTip('<')
        # # button2.clicked.connect(self.onClick)
        # controls_layout.addWidget(button2)
        
        # button3 = QtWidgets.QPushButton(">")
        # button3.setToolTip('>')
        # controls_layout.addWidget(button3)
        
        # # Set Final Layouts:
        # controls_frame.setLayout(controls_layout)
        
        controls_frame = Spike3DRasterBottomPlaybackControlBar() # Initialize new controls class from the Spike3DRasterBottomPlaybackControlBar class.
        controls_layout = controls_frame.layout() # Get the layout
        
        controls_frame.play_pause_toggled.connect(self.play_pause)
        controls_frame.jump_left.connect(self.on_jump_left)
        controls_frame.jump_right.connect(self.on_jump_right)
        controls_frame.reverse_toggled.connect(self.on_reverse_held)
        
        return controls_frame, controls_layout


    @QtCore.pyqtSlot()
    def Spike3DRasterBottomFrameControlsMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        # TODO: NOT CALLED
        pass

    @QtCore.pyqtSlot(float, float)
    def Spike3DRasterBottomFrameControlsMixin_on_window_update(self, new_start=None, new_end=None):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc. """
        # TODO: NOT CALLED
        pass
    
    
    ## Update Functions:
    @QtCore.pyqtSlot(bool)
    def play_pause(self, is_playing):
        print(f'Spike3DRasterBottomFrameControlsMixin.play_pause(is_playing: {is_playing})')
        if (not is_playing):
            self.animationThread.start()
        else:
            self.animationThread.terminate()

    @QtCore.pyqtSlot()
    def on_jump_left(self):
        # Skip back some frames
        print(f'Spike3DRasterBottomFrameControlsMixin.on_jump_left()')
        self.shift_animation_frame_val(-5)
        
    @QtCore.pyqtSlot()
    def on_jump_right(self):
        # Skip forward some frames
        print(f'Spike3DRasterBottomFrameControlsMixin.on_jump_right()')
        self.shift_animation_frame_val(5)
        

    @QtCore.pyqtSlot(bool)
    def on_reverse_held(self, is_reversed):
        print(f'Spike3DRasterBottomFrameControlsMixin.on_reverse_held(is_reversed: {is_reversed})')
        pass
        
    
class Spike3DRaster_Vedo(SimplePlayPauseWithExternalAppMixin, Spike3DRasterBottomFrameControlsMixin, SpikeRasterBase):
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
                                                   f'render_window_duration: {self.render_window_duration}',
                                                #    f'animation_time_step: {self.animation_time_step}',
                                                   f'temporal_axis_length: {self.temporal_axis_length}',
                                                   f'temporal_zoom_factor: {self.temporal_zoom_factor}']
        lines_dict[af.AlignBottom | af.AlignLeft] = ['BL', 
                                                   f'active_time_window: {self.spikes_window.active_time_window}',
                                                #    f'playback_rate_multiplier: {self.playback_rate_multiplier}'
                                                   ]
        lines_dict[af.AlignBottom | af.AlignRight] = ['BR']    
        return lines_dict
    
    
    @property
    def overlay_vedo_text_lines_dict(self):
        """The overlay_vedo_text_lines_dict property."""
        return {self.qt_to_vedo_alignment_dict[k]:v for (k,v) in self.overlay_text_lines_dict.items()}
    
    ######  Get/Set Properties ######:

    # @property
    # def axes_walls_z_height(self):
    #     """The axes_walls_z_height property."""
    #     return self._axes_walls_z_height
    
    @property
    def z_floor(self):
        """The offset of the floor in the z-axis."""
        # return -10
        return 0
    
    @property
    def y_backwall(self):
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

    def __init__(self, params=None, spikes_window=None, playback_controller=None, neuron_colors=None, neuron_sort_order=None, **kwargs):
        super(Spike3DRaster_Vedo, self).__init__(params=params, spikes_window=spikes_window, playback_controller=playback_controller, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, **kwargs)
        # Initialize member variables:
        
        # Helper container variables
        # self.enable_debug_print = False
        self.enable_debug_widgets = False
        self.enable_debug_print = True
        
        # Helper Mixins: INIT:
        self.Spike3DRasterBottomFrameControlsMixin_on_init()
                    
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        
        # build the UI components:
        # self.buildUI()


    def setup(self):
        """ setup() is called before self.buildUI(), etc.
            self.plots
        
        """
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        
        # self.app = pg.mkQApp("Spike3DRaster_Vedo")
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
        self.params.spike_height_z = 4.0
        self.params.spike_start_z = self.z_floor # self.z_floor
        self.params.spike_end_z = self.params.spike_start_z + self.params.spike_height_z
        
        # self.params.max_y_pos = 50.0
        # self.params.max_z_pos = 10.0
        
        # max_y_all_data = self.spikes_df['visualization_raster_y_location'].nanmax()
        
        self.params.max_y_pos = 10.0
        self.params.max_z_pos = max(self.params.spike_end_z, (self.z_floor + 1.0))
        
        
        # self.params.center_mode = 'zero_centered'
        self.params.center_mode = 'starting_at_zero'
        self.params.bin_position_mode = 'bin_center'
        # self.params.bin_position_mode = 'left_edges'
        
        # by default we want the time axis to approximately span -20 to 20. So we set the temporal_zoom_factor to 
        # self.params.temporal_zoom_factor = 1.0
        self.params.temporal_zoom_factor = 1000.0      
        
        self.params.enable_epoch_rectangle_meshes = self.enable_epoch_rectangle_meshes
        self.params.active_cell_colormap_name = 'rainbow'
        
        # Plots Structures:
        self.plots.meshes = dict()
                
        # TODO: Setup self.epochs_df:
        if not self.enable_epoch_rectangle_meshes:
            self.epochs_df = None
        else:
            raise NotImplementedError
        
        if 'cell_idx' not in self.spikes_df.columns:
            # self.spikes_df['cell_idx'] = self.spikes_df['unit_id'].copy() # TODO: this is bad! The self.get_neuron_id_and_idx(...) function doesn't work!
            # note that this is very slow, but works:
            print(f'cell_idx column missing. rebuilding (this might take a minute or two)...')
            included_cell_INDEXES = np.array([self.get_neuron_id_and_idx(neuron_id=an_included_cell_ID)[0] for an_included_cell_ID in self.spikes_df['aclu'].to_numpy()]) # get the indexes from the cellIDs
            self.spikes_df['cell_idx'] = included_cell_INDEXES.copy()

        if 'visualization_raster_y_location' not in self.spikes_df.columns:
            print(f'visualization_raster_y_location column missing. rebuilding (this might take a minute or two)...')
            # Compute the y for all windows, not just the current one:
            y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='bin_center', side_bin_margins = self.params.side_bin_margins)
            all_y = [y[a_cell_id] for a_cell_id in self.spikes_df['cell_idx'].to_numpy()]
            self.spikes_df['visualization_raster_y_location'] = all_y # adds as a column to the dataframe. Only needs to be updated when the number of active units changes
            # max_y_all_data = np.nanmax(all_y) # self.spikes_df['visualization_raster_y_location'] 

        max_y_all_data = np.nanmax(self.spikes_df['visualization_raster_y_location'].to_numpy()) # self.spikes_df['visualization_raster_y_location'] 
        self.params.max_y_pos = max(10.0, max_y_all_data)
        
        # Helper Mixins: SETUP:
        self.Spike3DRasterBottomFrameControlsMixin_on_setup()
        
        
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
        # button = QtWidgets.QPushButton("My Button makes the cone red")
        # button.setToolTip('This is an example button')
        # button.clicked.connect(self.onClick)
 
               
        #### Build Graphics Objects #####
        self._buildGraphics()
 
        
        # Helper Mixins: buildUI:
        self.ui.bottom_controls_frame, self.ui.bottom_controls_layout = self.Spike3DRasterBottomFrameControlsMixin_on_buildUI()
        
        
        
        # TODO: Register Functions:
        # self.ui.bottom_controls_frame.
        
        # setup self.ui.frame_layout:
        # self.ui.frame_layout.addWidget(self.ui.vtkWidget)
        # self.ui.frame_layout.addWidget(button)

        self.ui.frame_layout.addWidget(self.ui.bottom_controls_frame) # add the button controls
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
        # self.spikes_window.spike_dataframe_changed_signal.connect(self.on_spikes_df_changed)
        # self.spikes_window.window_duration_changed_signal.connect(self.on_window_duration_changed)
        # self.spikes_window.window_changed_signal.connect(self.on_window_changed)
        self.spikes_window.window_updated_signal.connect(self.on_window_changed)



        

        self.ui.plt.show()                  # <--- show the vedo rendering
        self.show()                     # <--- show the Qt Window

    def _buildGraphics(self):
        """ Implementors must override this method to build the main graphics object and add it at layout position (0, 0)"""
        # vedo_qt_main_window = MainVedoPlottingWindow() # Create the main window with the vedo plotter
        self.ui.vtkWidget = QVTKRenderWindowInteractor(self.ui.frame)
        # Create renderer and add the vedo objects and callbacks
        self.ui.plt = Plotter(qtWidget=self.ui.vtkWidget, title='Pho Vedo MainVedoPlottingWindow Test', bg='black')
        self.id1 = self.ui.plt.addCallback("mouse click", self.onMouseClick)
        self.id2 = self.ui.plt.addCallback("key press",   self.onKeypress)

        self.ui.plt += Cone().rotateX(20)
        # self.ui.plt.show()                  # <--- show the vedo rendering

        # Build All Meshes:
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
            
            
        # replaces StaticVedo_3DRasterHelper.build_spikes_lines(...) with a version optimized for Spike3DRaster_Vedo:
        all_spike_t = self.spikes_df[self.spikes_df.spikes.time_variable_name].to_numpy() # this will map
        # all_spike_x = DataSeriesToSpatial.temporal_to_spatial_map(all_spike_t, self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time, self.temporal_axis_length, center_mode=self.params.center_mode)
        all_spike_x = DataSeriesToSpatial.temporal_to_spatial_map(all_spike_t, self.spikes_window.total_data_start_time, self.spikes_window.total_data_end_time, self.temporal_axis_length, center_mode=self.params.center_mode)
        curr_spike_y = self.spikes_df['visualization_raster_y_location'].to_numpy() # this will map

        # t-mode:
        # startPoints = np.vstack((curr_spike_t, curr_spike_y, np.full_like(curr_spike_t, self.params.spike_start_z))).T
        # endPoints = np.vstack((curr_spike_t, curr_spike_y, np.full_like(curr_spike_t, self.params.spike_end_z))).T
        
        # x-mode:
        startPoints = np.vstack((all_spike_x, curr_spike_y, np.full_like(all_spike_x, self.params.spike_start_z))).T
        endPoints = np.vstack((all_spike_x, curr_spike_y, np.full_like(all_spike_x, self.params.spike_end_z))).T
        
        all_spike_lines = Lines(startPoints, endPoints=endPoints, c='k', alpha=0.8, lw=1.0, dotted=False, scale=1, res=1) # curr_spike_alphas
        # let the scalar be the y coordinate of the mesh vertices
        spike_color_ids = curr_spike_y.copy() # one per spike
        spike_point_color_ids = all_spike_lines.points()[:, 1]
        curr_spike_cmap, curr_spike_alphas, spike_point_color_ids = StaticVedo_3DRasterHelper._build_spikes_colormap(spike_point_color_ids)
        
        # Uses the old version from StaticVedo_3DRasterHelper.build_spikes_lines: 
        # all_spike_lines, curr_spike_cmap, curr_spike_alphas, spike_point_color_ids, spike_color_ids = StaticVedo_3DRasterHelper.build_spikes_lines(self.spikes_df, spike_start_z = self.params.spike_start_z, spike_end_z = self.params.spike_end_z)
        all_spike_lines.useBounds(False)
        
        y_cells = np.unique(spike_color_ids)
        n_cells = len(y_cells)
        # n_cells # 40
        
        # Builds correct colors for every spike point (specified by spike_point_color_ids) using self.params.active_cell_colormap_name
        spike_rgba_colors, spike_rgb_colors = StaticVedo_3DRasterHelper.build_spike_rgb_colors(spike_color_ids, active_cell_colormap_name=self.params.active_cell_colormap_name)
        
        all_spike_lines.lighting('default')
        ## Set Colors using explicitly computed spike_rgba_colors:
        all_spike_lines.cellIndividualColors(spike_rgba_colors*255)
        # ## Get Colors
        # curr_cell_rgba_colors = all_spike_lines.celldata['CellIndividualColors']
        # print(f'curr_cell_rgba_colors: {curr_cell_rgba_colors}')
        # # set opacity component to zero for all non-window spikes
        # curr_cell_rgba_colors[:,3] = int(0.3*255) # np.full((spike_rgb_colors.shape[0], 1), 1.0)
        # curr_cell_rgba_colors[active_ids,3] = int(1.0*255) # set alpha for active_ids to an opaque 1.0
        # all_spike_lines.cellIndividualColors(curr_cell_rgba_colors) # needed?

        
        """ 
        # self.spikes_window.total_data_start_time
        # self.spikes_window.total_data_end_time
        
        """
        
        active_t_start, active_t_end = (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time)
        active_window_t_duration = self.spikes_window.window_duration
        if self.enable_debug_print:
            printc('debug_print_axes_locations(...): Active Window/Local Properties:')
            printc(f'\t(active_t_start: {active_t_start}, active_t_end: {active_t_end}), active_window_t_duration: {active_window_t_duration}')
        active_x_start, active_x_end = DataSeriesToSpatial.temporal_to_spatial_map((active_t_start, active_t_end),
                                                                                self.spikes_window.total_data_start_time, self.spikes_window.total_data_end_time,
                                                                                self.temporal_axis_length,
                                                                                center_mode=self.params.center_mode)
        if self.enable_debug_print:
            printc(f'\t(active_x_start: {active_x_start}, active_x_end: {active_x_end}), active_x_length: {active_x_end - active_x_start}')
        
        # active_x_start, active_x_end = DataSeriesToSpatial.temporal_to_spatial_map((active_t_start, active_t_end), self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time, self.temporal_axis_length, center_mode=self.params.center_mode)
        # (active_t_start: 30.0, active_t_end: 45.0)
        # (active_x_start: -20.0, active_x_end: 20.0)

        # Bounding planes:
        # active_ids, start_bound_plane, end_bound_plane = StaticVedo_3DRasterHelper.update_active_spikes_window(all_spike_lines, x_start=active_t_start, x_end=active_t_end, max_y_pos=self.params.max_y_pos, max_z_pos=self.params.max_z_pos)
        active_ids, start_bound_plane, end_bound_plane = StaticVedo_3DRasterHelper.update_active_spikes_window(all_spike_lines, x_start=active_x_start, x_end=active_x_end, max_y_pos=self.params.max_y_pos, max_z_pos=self.params.max_z_pos)
                
        if rect_meshes is not None:
            active_mesh_args = (all_spike_lines, rect_meshes, start_bound_plane, end_bound_plane)
        else:
            active_mesh_args = (all_spike_lines, start_bound_plane, end_bound_plane)

        # New Way of building the axes for all data (displaying evenly-spaced ticks along the x-axis with labels reflecting the corresponding t-value time:
        

        # Old Way of building the axes for all data:
        # all_data_axes = vedo.Axes([all_spike_lines, rect_meshes, start_bound_plane, end_bound_plane],  # build axes for this set of objects
        # all_data_axes = vedo.Axes(active_mesh_args,  # build axes for this set of objects
        #             xtitle="timestamp (t)",
        #             ytitle="Cell ID",
        #             ztitle="Z",
        #             hTitleColor='white',
        #             zHighlightZero=True,
        #             xyFrameLine=2, yzFrameLine=1, zxFrameLine=1,
        #             xyFrameColor='white',
        #             # xyShift=1.05, # move xy 5% above the top of z-range
        #             yzGrid=True,
        #             zxGrid=True,
        #             yMinorTicks=n_cells,
        #             yLineColor='white',
        #             # xrange=(active_x_start, active_x_end),
        #             # yrange=(0.0, max_y_pos),
        #             # zrange=(0.0, max_z_pos)
        # )
        
        #  xValuesAndLabels: list of custom tick positions and labels [(pos1, label1), …]
        # Want to add a tick/label at the x-values corresponding to each minute.
        (active_t_start, active_t_end, active_window_t_duration), (global_start_t, global_end_t, global_total_data_duration), (active_x_start, active_x_end, active_x_duration), (global_x_start, global_x_end, global_x_duration) = debug_print_axes_locations(self)
        new_axes_x_to_time_labels = DataSeriesToSpatial.build_minute_x_tick_labels(self)
        
        if self.enable_debug_print:
            printc(f'new_axes_x_to_time_labels: {new_axes_x_to_time_labels}, global_x_start: {global_x_start}, global_x_end: {global_x_end}')

        all_data_axes = Axes(all_spike_lines, xrange=[0, 15000], c='white', textScale=0.1, gridLineWidth=0.1, axesLineWidth=0.1, xTickLength=0.005*0.1, xTickThickness=0.0025*0.1,
                                xValuesAndLabels = new_axes_x_to_time_labels, useGlobal=True)
        
        all_data_axes.useBounds(False)
        
        
        ## The axes only for the active window:
        active_window_only_axes = vedo.Axes([start_bound_plane, end_bound_plane],  # build axes for this set of objects
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
                    yMinorTicks=n_cells,
                    yLineColor='red',
                    xrange=(active_x_start, active_x_end),
                    yrange=(0.0, self.params.max_y_pos),
                    zrange=(0.0, self.params.max_z_pos)
        )
        

        self.ui.plt += active_mesh_args
        self.ui.plt += all_data_axes
        self.ui.plt += active_window_only_axes
                
        active_window_only_axes.SetVisibility(False)
        all_data_axes.SetVisibility(True)
        
        # Set meshes to self.plots.meshes:
        self.plots.meshes['rect_meshes'] = rect_meshes
        self.plots.meshes['all_spike_lines'] = all_spike_lines
        self.plots.meshes['start_bound_plane'] = start_bound_plane
        self.plots.meshes['end_bound_plane'] = end_bound_plane
        self.plots.meshes['all_data_axes'] = all_data_axes
        self.plots.meshes['active_window_only_axes'] = active_window_only_axes
        
        # setup self.ui.frame_layout:
        self.ui.frame_layout.addWidget(self.ui.vtkWidget)
        # raise NotImplementedError
        
        
        ## Setup Viewport Overlay Text:
        self.ui.viewport_overlay  = vedo.CornerAnnotation().color('white').alpha(0.85)#.font("Kanopus")
        self.ui.plt += self.ui.viewport_overlay
        # self.ui.viewport_overlay.text(vedo.getColorName(self.counter), "top-center")
        # self.ui.viewport_overlay.text("..press q to quit", "bottom-right")
        for vedo_pos_key, values in self.overlay_vedo_text_lines_dict.items():
            # print(f'a_key: {a_key}, values: {values}')
            self.ui.viewport_overlay.text('\n'.join(values), vedo_pos_key)
        
    
        self.ui.plt.resetCamera() # resetCamera() updates the camera's position given the ignored components
        # This limits the meshes to just the active window's meshes: [start_bound_plane, end_bound_plane, active_window_only_axes]

    
    
    # def on_window_changed(self):
    #     # called when the window is updated
    #     if self.enable_debug_print:
    #         print(f'Spike3DRaster_Vedo.on_window_changed()')
    #     self._update_plots()
        
            
    def _update_plots(self):
        if self.enable_debug_print:
            printc(f'Spike3DRaster_Vedo._update_plots()')
        # build the position range for each unit along the y-axis:
        # y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode='bin_center', side_bin_margins = self.params.side_bin_margins)
        
        
        all_spike_lines = self.plots.meshes.get('all_spike_lines', None)
        start_bound_plane = self.plots.meshes.get('start_bound_plane', None)
        end_bound_plane = self.plots.meshes.get('end_bound_plane', None)
        active_window_only_axes = self.plots.meshes.get('active_window_only_axes', None)
        
        prev_x_position = start_bound_plane.x()
        
        active_t_start, active_t_end = (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time)
        active_window_t_duration = self.spikes_window.window_duration
        if self.enable_debug_print:
            printc('debug_print_axes_locations(...): Active Window/Local Properties:')
            printc(f'\t(active_t_start: {active_t_start}, active_t_end: {active_t_end}), active_window_t_duration: {active_window_t_duration}')
        # active_x_start, active_x_end = DataSeriesToSpatial.temporal_to_spatial_map((active_t_start, active_t_end), self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time, self.temporal_axis_length, center_mode=self.params.center_mode)
        active_x_start, active_x_end = DataSeriesToSpatial.temporal_to_spatial_map((active_t_start, active_t_end),
                                                                                self.spikes_window.total_data_start_time, self.spikes_window.total_data_end_time,
                                                                                self.temporal_axis_length,
                                                                                center_mode=self.params.center_mode)
        if self.enable_debug_print:
            printc(f'\t(active_x_start: {active_x_start}, active_x_end: {active_x_end}), active_x_length: {active_x_end - active_x_start}')
            
        
        # print(f'(active_t_start: {active_t_start}, active_t_end: {active_t_end})')
        # print(f'(active_x_start: {active_x_start}, active_x_end: {active_x_end})')
        
        # active_ids, start_bound_plane, end_bound_plane = StaticVedo_3DRasterHelper.update_active_spikes_window(all_spike_lines, x_start=active_t_start, x_end=active_t_end, max_y_pos=self.params.max_y_pos, max_z_pos=self.params.max_z_pos, start_bound_plane=start_bound_plane, end_bound_plane=end_bound_plane)
        active_ids, start_bound_plane, end_bound_plane = StaticVedo_3DRasterHelper.update_active_spikes_window(all_spike_lines, x_start=active_x_start, x_end=active_x_end, max_y_pos=self.params.max_y_pos, max_z_pos=self.params.max_z_pos, start_bound_plane=start_bound_plane, end_bound_plane=end_bound_plane)
        
        delta_x = start_bound_plane.x() - prev_x_position
        
        prev_x_pos = active_window_only_axes.x()
        active_window_only_axes.x(prev_x_pos + delta_x) # works for positioning but doesn't update numbers
        
        
        # Update the additional display lines information on the overlay:
        for vedo_pos_key, values in self.overlay_vedo_text_lines_dict.items():
            # print(f'a_key: {a_key}, values: {values}')
            self.ui.viewport_overlay.text('\n'.join(values), vedo_pos_key)
        
        
        
        self.ui.plt.resetCamera() # resetCamera() updates the camera's position
        self.ui.plt.render()

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
        
        
        

    def onMouseClick(self, evt):
        printc("You have clicked your mouse button. Event info:\n", evt, c='y')

    def onKeypress(self, evt):
        printc("You have pressed key:", evt.keyPressed, c='b')

    @QtCore.pyqtSlot()
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
        self.ui.vtkWidget.close()
        
        # Emit the close signal:
        self.close_signal.emit() # emit to indicate that we're closing this window


# josfd