from copy import deepcopy
import time
import sys
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyphoplacecellanalysis.External.pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np
from pyphocorehelpers.indexing_helpers import interleave_elements, partition

# import qdarkstyle
from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GLGraphicsItems.GLDebugAxisItem import GLDebugAxisItem
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GLGraphicsItems.GLViewportOverlayPainterItem import GLViewportOverlayPainterItem

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.SpikeRasterBase import SpikeRasterBase
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.RenderTimeEpoch3DMeshesMixin import RenderTimeEpoch3DMeshesMixin

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.RenderTimeCurvesMixin import PyQtGraphSpecificTimeCurvesMixin

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContextMenuGLViewWidget import ContextMenuGLViewWidget




""" Windowed Spiking Datasource Features

See ** DataSeriesToSpatial ** class which performs the mapping from the temporal axis to space.

Transforming the events into either 2D or 3D representations for visualization should NOT be part of this class' responsibilities.
Separate 2D and 3D event visualization functions should be made to transform events from this class into appropriate point/datastructure representations for the visualization framework being used.

# Local window properties
Get (window_start, window_end) times

# Global data properties
Get (earliest_datapoint_time, latest_datapoint_time) # globally, for the entire timeseries


# Note that in addition to the above-mentioned mapping, there's an additional mapping that must be performed due to 'temporal_zoom_factor', a visualization property belonging to the RasterPlot class.
Note that it fires a signal 'temporal_mapping_changed which indicates a change in this scale value

Internally it also performs the on_adjust_temporal_spatial_mapping() function to update anything that needs to be updated.


"""

class Spike3DRaster(PyQtGraphSpecificTimeCurvesMixin, RenderTimeEpoch3DMeshesMixin, SpikeRasterBase):
    """ Displays a 3D version of a raster plot with the spikes occuring along a plane. 
    
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster import Spike3DRaster
        curr_epoch_name = 'maze1'
        curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
        curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
        curr_spikes_df = curr_sess.spikes_df
        spike_raster_plt = Spike3DRaster(curr_spikes_df, window_duration=4.0, window_start_time=30.0)
    """
    
    # GUI Configuration Options:
    WantsRenderWindowControls = False
    WantsPlaybackControls = False
    
    # Application/Window Configuration Options:
    applicationName = 'Spike3DRaster'
    windowName = 'Spike3DRaster'
    
    
    @property
    def overlay_text_lines_dict(self):
        """The lines of text to be displayed in the overlay."""    
        af = QtCore.Qt.AlignmentFlag

        lines_dict = dict()
        
        lines_dict[af.AlignTop | af.AlignLeft] = ['TL']
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
    
    
    ######  Get/Set Properties ######:

    @property
    def axes_walls_z_height(self):
        """The axes_walls_z_height property."""
        return self.params.axes_walls_z_height
    
    @property
    def floor_z(self):
        """The offset of the floor in the z-axis."""
        return -10
    
    @property
    def back_wall_y(self):
        """The y position location of the green back (Y=0) axes wall plane."""
        return self.n_half_cells
    
    
    @property
    def side_wall_x(self):
        """The x position of the nearest wall (the side wall, not the back wall) """
        return -self.half_temporal_axis_length
    
    
    @property
    def axes_walls_floor_alignment_offset_z(self):
        """The z offset to align the bottom of the two walls with the floor """
        # To determine the translation required to align the axes walls with the floor:
        pre_translated_wall_z_bottom = -(self.params.axes_walls_z_height/2.0)
        # print(f'pre_translated_wall_z_bottom: {pre_translated_wall_z_bottom}, floor_z: {self.floor_z}')
        return self.floor_z - pre_translated_wall_z_bottom # required z-shift to align the bottom of the walls with the floors
    
    
    @property
    def series_identity_y_values(self):
        """The series_identity_y_values property."""
        return self._series_identity_y_values
    
    
    
    
    @property
    def interval_rendering_plots(self):
        """ returns the list of child subplots/graphics (usually PlotItems) that participate in rendering intervals """
        return [self.ui.main_gl_widget]
    
    
    def __init__(self, params=None, spikes_window=None, playback_controller=None, neuron_colors=None, neuron_sort_order=None, application_name=None, **kwargs):
        super(Spike3DRaster, self).__init__(params=params, spikes_window=spikes_window, playback_controller=playback_controller, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=application_name, **kwargs)
        
        # Setup Specific Member Variables:
        self.params.render_epochs = None
        
        # Init the TimeCurvesViewMixin for 3D Line plots:
        ### No plots will actually be added until self.add_3D_time_curves(plot_dataframe) is called with a valid dataframe.
        self.TimeCurvesViewMixin_on_init()
        self.RenderTimeEpoch3DMeshesMixin_on_init()
        
        # Setup Signals:
        self.temporal_mapping_changed.connect(self.on_adjust_temporal_spatial_mapping)
        self.spikes_window.timeWindow.window_duration_changed_signal.connect(self.on_adjust_temporal_spatial_mapping)
        # self.on_window_duration_changed.connect(self.on_adjust_temporal_spatial_mapping)
        # self.spikes_window.timeWindow.window_changed_signal.connect(self.TimeCurvesViewMixin_on_window_update) # TODO: this is for TimeCurvesViewMixin but currently just call manually.
        self.unit_sort_order_changed_signal.connect(self.on_unit_sort_order_changed)
        
        if self.enable_show_on_init:
            self.show()
    
        


    def setup(self):
        # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
        
        # self.app = pg.mkQApp("Spike3DRaster")
        self.app = pg.mkQApp(self.applicationName)
        
        # Configure pyqtgraph config:
        try:
            import OpenGL
            pg.setConfigOption('useOpenGL', True)
            pg.setConfigOption('enableExperimental', True)
        except Exception as e:
            print(f"Enabling OpenGL failed with {e}. Will result in slow rendering. Try installing PyOpenGL.")
            
        pg.setConfigOptions(antialias = True)
        pg.setConfigOption('background', "#1B1B1B")
        pg.setConfigOption('foreground', "#727272")
    
        # Config        
        """ Adds required params to self.params:
            spike_start_z (default -10.0): the z-offset of the start of the spikes
            spike_end_z (default -6.0): the z-offset of the end of the spikes
            center_mode (allowed: ['starting_at_zero', 'zero_centered']), (default 'zero_centered'): I believe how everything is layed out relative to the origin
            bin_position_mode (allowed: ['bin_center', 'left_edges']), (default 'left_edges'): specifies how the bins are positioned??
            
            axes_walls_z_height (default 20.0): the z-height of the axes plane box that frames the data
            axes_planes_floor_fixed_y_spacing (default 10.0): the spacing of grid lines along the y-axis that subdivide the floor axes plane (blue z-plane)
        
        """
        self.params.setdefault('spike_start_z', -10.0)
        self.params.setdefault('spike_end_z', -6.0)
        self.params.setdefault('center_mode', 'zero_centered')
        self.params.setdefault('bin_position_mode', 'left_edges')
        # Axes Planes Variables:
        self.params.setdefault('axes_walls_z_height', 20.0)
        self.params.setdefault('axes_planes_floor_fixed_y_spacing', 10.0)

        # by default we want the time axis to approximately span -20 to 20. So we set the temporal_zoom_factor to 
        self.params.temporal_zoom_factor = 40.0 / float(self.render_window_duration)        
                
        self.enable_debug_print = False
        self.enable_debug_widgets = False # Default: False, If True displays a white wireframe sphere at the origin and some helper x,y,z axes arrows to see the coordinate system.
        
        # Determine the y-values corresponding to the series identity
        self._series_identity_y_values = None
        self.update_series_identity_y_values()
        
        self.RenderTimeEpoch3DMeshesMixin_on_setup()
        
   
    def _buildGraphics(self):
        ##### Main Raster Plot Content Top ##########
        # self.ui.main_gl_widget = gl.GLViewWidget()
        self.ui.main_gl_widget = ContextMenuGLViewWidget()

        self.ui.main_gl_widget.resize(1000,600)
        # self.ui.main_gl_widget.setWindowTitle('pyqtgraph: 3D Raster Spikes Plotting')
        self.ui.main_gl_widget.setCameraPosition(distance=40)
        
        # Add the main widget to the layout in the (0, 0) location:
        self.ui.layout.addWidget(self.ui.main_gl_widget, 0, 0) # add the GLViewWidget to the layout at 0, 0
        
        # self.ui.main_gl_widget.clicked.connect(self.play_pause)
        # self.ui.main_gl_widget.doubleClicked.connect(self.toggle_full_screen)
        # self.ui.main_gl_widget.wheel.connect(self.wheel_handler)
        # self.ui.main_gl_widget.keyPressed.connect(self.key_handler)
        
        #### Build Graphics Objects ##### 
        # Add debugging widget:
        
        # Adds a helper widget that displays the x/y/z vector at the origin:
        if self.enable_debug_widgets:
            self.ui.ref_axes_indicator = GLDebugAxisItem()
            self.ui.ref_axes_indicator.setObjectName('debug_ref_axes_indicator')
            self.ui.ref_axes_indicator.setSize(x=15.0, y=10.0, z=5.0)
            self.ui.main_gl_widget.addItem(self.ui.ref_axes_indicator)
            
            self.ui.gl_test_points = []
            md = gl.MeshData.sphere(rows=10, cols=20)
            m1 = gl.GLMeshItem(meshdata=md, smooth=False, drawFaces=False, drawEdges=True, edgeColor=(1,1,1,1))
            m1.setObjectName('debug_test_points_origin_sphere')
            # m1.translate(5, 0, 0)
            m1.setGLOptions('additive')
            self.ui.main_gl_widget.addItem(m1)
            self.ui.gl_test_points.append(m1)
            
        # The 2D viewport overlay that contains text:
        self.ui.viewport_overlay = GLViewportOverlayPainterItem()
        self.ui.viewport_overlay.setObjectName('viewport_overlay')
        self.ui.main_gl_widget.addItem(self.ui.viewport_overlay)
        # Update the additional display lines information on the overlay:
        # self.ui.viewport_overlay.additional_overlay_text_lines = self.overlay_text_lines
        self.ui.viewport_overlay.additional_overlay_text_dict = self.overlay_text_lines_dict

                
        # Add axes planes:
        self._build_axes_plane_graphics()
        
        
        # Custom 3D raster plot:
        
        # TODO: EFFICIENCY: For a potentially considerable speedup, could compute the "curr_x" values for all cells at once and add as a column to the dataframe since it only depends on the current window parameters (changes when window changes).
            ## OH, but the window changes every frame update (as that's what it means to animate the spikes as a function of time). Maybe not a big speedup.
        
        self.ui.gl_line_plots = [] # create an empty array for each GLLinePlotItem, of which there will be one for each unit.
        
        # build the position range for each unit along the y-axis:
        # rebuild the position range for each unit along the y-axis:
        self.update_series_identity_y_values()
        # self.series_identity_y_values = self.series_identity_y_values[self.unit_sort_order] # re-sort the y-values by the unit_sort_order
        # TODO: convert to using self.fragile_linear_neuron_IDX_to_spatial(...)
        
        self._build_neuron_id_graphics(self.ui.main_gl_widget, self.series_identity_y_values)
        
        # Plot each unit one at a time:
        for i, a_fragile_linear_neuron_IDX in enumerate(self.fragile_linear_neuron_IDXs):
            # curr_color = pg.mkColor((i, self.n_cells*1.3))
            curr_color = self.params.neuron_qcolors_map[a_fragile_linear_neuron_IDX]
            curr_color.setAlphaF(0.5)
            # print(f'cell_id: {cell_id}, curr_color: {curr_color.alpha()}')
            
            # Filter the dataframe using that column and value from the list
            curr_cell_df = self.active_windowed_df[self.active_windowed_df['fragile_linear_neuron_IDX']==a_fragile_linear_neuron_IDX].copy() # is .copy() needed here since nothing is updated???
            # curr_fragile_linear_neuron_IDX = curr_cell_df['fragile_linear_neuron_IDX'].to_numpy() # this will map to the y position
            curr_spike_t = curr_cell_df[curr_cell_df.spikes.time_variable_name].to_numpy() # this will map 
            
            # yi = self.series_identity_y_values[i] # get the correct y-position for all spikes of this cell
            yi = self.fragile_linear_neuron_IDX_to_spatial(a_fragile_linear_neuron_IDX)
            # self.fragile_linear_neuron_IDXs
            # print(f'cell_id: {cell_id}, yi: {yi}')
            
            # map the current spike times back onto the range of the window's (-half_render_window_duration, +half_render_window_duration) so they represent the x coordinate
            curr_x = DataSeriesToSpatial.temporal_to_spatial_map(curr_spike_t, self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time, self.temporal_axis_length, center_mode='zero_centered')
            curr_paired_x = np.squeeze(interleave_elements(np.atleast_2d(curr_x).T, np.atleast_2d(curr_x).T))
            
            # Z-positions:
            # z = curr_spike_t[np.arange(100)] # get the first 20 spikes for each
            spike_bottom_zs = np.full_like(curr_x, self.params.spike_start_z)
            spike_top_zs = np.full_like(curr_x, self.params.spike_end_z)
            curr_paired_spike_zs = np.squeeze(interleave_elements(np.atleast_2d(spike_bottom_zs).T, np.atleast_2d(spike_top_zs).T)) # alternating top and bottom z-positions
        
            # sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
            # sp1.translate(5,5,0)
            # w.addItem(sp1)
            
            # Build lines:
            pts = np.column_stack([curr_paired_x, np.full_like(curr_paired_x, yi), curr_paired_spike_zs]) # the middle coordinate is the size of the x array with the value given by yi. yi must be the scalar for this cell.
            # pts = np.column_stack([x, np.full_like(x, yi), z]) # the middle coordinate is the size of the x array with the value given by yi. yi must be the scalar for this cell.
            # plt = gl.GLLinePlotItem(pos=pts, color=pg.mkColor((cell_id,n*1.3)), width=(cell_id+1)/10., antialias=True)
            plt = gl.GLLinePlotItem(pos=pts, color=curr_color, width=1.0, antialias=True, mode='lines') # mode='lines' means that each pair of vertexes draws a single line segement

            # plt.setYRange((-self.n_half_cells - self.side_bin_margins), (self.n_half_cells + self.side_bin_margins))
            # plt.setXRange(-self.half_render_window_duration, +self.half_render_window_duration)
            
            self.ui.main_gl_widget.addItem(plt)
            self.ui.gl_line_plots.append(plt)
            
        self.RenderTimeEpoch3DMeshesMixin_on_buildUI()

     
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


    def _build_axes_plane_graphics(self):
        # Add axes planes:
        # X-plane:
        x_color = (255, 155, 155, 76.5)
        self.ui.gx = gl.GLGridItem(color=x_color) # 'x' plane, red
        self.ui.gx.setObjectName('axes_plane_gx')
        self.ui.gx.rotate(90, 0, 1, 0)
        self.ui.gx.translate(self.side_wall_x, 0, self.axes_walls_floor_alignment_offset_z) # shift backwards
        self.ui.gx.setSize(self.params.axes_walls_z_height, self.n_full_cell_grid) # (z-axis, x-axis) # std size in z-dir, n_cell size across
        self.ui.gx.setSpacing((self.params.axes_walls_z_height/2.0), 1) # (z-axis, x-axis) # want two subidivisions in the non-axis direction (z-axis), and one line per neuron in the axis dimenion (x-axis)
        self.ui.main_gl_widget.addItem(self.ui.gx)
        self.ui.x_txtitem = gl.GLTextItem(pos=(self.side_wall_x, self.n_half_cells, self.axes_walls_floor_alignment_offset_z), text='x', color=x_color) # The axis label text 
        self.ui.x_txtitem.setObjectName('axes_plane_x_txtitem')
        self.ui.main_gl_widget.addItem(self.ui.x_txtitem)

        # Y-plane:
        y_color = (155, 255, 155, 76.5)
        self.ui.gy = gl.GLGridItem(color=y_color) # 'y' plane, green
        self.ui.gy.setObjectName('axes_plane_gy')
        self.ui.gy.rotate(90, 1, 0, 0)
        self.ui.gy.translate(0, -self.n_half_cells, self.axes_walls_floor_alignment_offset_z) # offset by half the number of units in the -y direction
        self.ui.gy.setSize(self.temporal_axis_length, self.params.axes_walls_z_height) # (y-axis, z-axis) # temporal_axis_length along its main direction (y-axis), 20 along its secondary (z-axis)
        self.ui.gy.setSpacing(1, (self.params.axes_walls_z_height/2.0)) # (y-axis, z-axis) # unit along the (y-axis) itself, only one subdivision along the (z-axis)
        self.ui.main_gl_widget.addItem(self.ui.gy)
        self.ui.y_txtitem = gl.GLTextItem(pos=(self.half_temporal_axis_length+0.5, -self.n_half_cells, self.axes_walls_floor_alignment_offset_z), text='y', color=y_color) # The axis label text 
        self.ui.y_txtitem.setObjectName('axes_plane_y_txtitem')
        self.ui.main_gl_widget.addItem(self.ui.y_txtitem)
        
        # XY-plane (with normal in z-dir):
        z_color = (155, 155, 255, 76.5)
        self.ui.gz = gl.GLGridItem(color=z_color) # 'z' plane, blue
        self.ui.gz.setObjectName('axes_plane_gz')
        self.ui.gz.translate(0, 0, self.floor_z) # Shift down by 10 units in the z-dir
        self.ui.gz.setSize(self.temporal_axis_length, self.n_full_cell_grid) # (y-axis, x-axis)
        self.ui.gz.setSpacing(self.params.axes_planes_floor_fixed_y_spacing, 1) # (y-axis, x-axis)
        # gz.setSize(n_full_cell_grid, n_full_cell_grid)
        self.ui.main_gl_widget.addItem(self.ui.gz)
        self.ui.z_txtitem = gl.GLTextItem(pos=(self.side_wall_x, -self.n_half_cells, (self.floor_z + 0.5)), text='z', color=z_color)  # The axis label text 
        self.ui.z_txtitem.setObjectName('axes_plane_z_txtitem')
        self.ui.main_gl_widget.addItem(self.ui.z_txtitem)

    def _update_axes_plane_graphics(self):
        """ updates the axes planes 
        Requires:
            self.ui.gx
            self.ui.gy
            self.ui.gz
            
            self.ui.x_txtitem
            self.ui.y_txtitem
            self.ui.z_txtitem
        """
         # Adjust the three axes planes:
        self.ui.gx.resetTransform()
        self.ui.gx.rotate(90, 0, 1, 0)
        self.ui.gx.translate(self.side_wall_x, 0, self.axes_walls_floor_alignment_offset_z) # shift backwards
        self.ui.gx.setSize(self.params.axes_walls_z_height, self.n_full_cell_grid) # (z-axis, x-axis) # std size in z-dir, n_cell size across
        self.ui.x_txtitem.setData(pos=(self.side_wall_x, self.n_half_cells, self.axes_walls_floor_alignment_offset_z))
        
        self.ui.gy.resetTransform()
        self.ui.gy.rotate(90, 1, 0, 0)
        self.ui.gy.translate(0, -self.n_half_cells, self.axes_walls_floor_alignment_offset_z) # offset by half the number of units in the -y direction
        self.ui.gy.setSize(self.temporal_axis_length, self.params.axes_walls_z_height) # (y-axis, z-axis) # temporal_axis_length along its main direction (y-axis), 20 along its secondary (z-axis)
        self.ui.y_txtitem.setData(pos=(self.half_temporal_axis_length+0.5, -self.n_half_cells, self.axes_walls_floor_alignment_offset_z))
        
        self.ui.gz.resetTransform()
        self.ui.gz.translate(0, 0, self.floor_z) # Shift down by 10 units in the z-dir
        self.ui.gz.setSize(self.temporal_axis_length, self.n_full_cell_grid) # (y-axis, x-axis)
        self.ui.z_txtitem.setData(pos=(self.side_wall_x, -self.n_half_cells, (self.floor_z + 0.5)))


    def _build_neuron_id_graphics(self, w, y_pos):
        """ builds the text items to indicate the neuron ID for each neuron in the df. """
        all_cell_ids = self.cell_ids
        # all_fragile_linear_neuron_IDXs = [self.fragile_linear_neuron_IDX_to_cell_id_map[a_cell_id] for a_cell_id in all_cell_ids] # get the list of all fragile_linear_neuron_IDXs
        
        cell_id_text_item_font = QtGui.QFont('Helvetica', 12)
        
        self.ui.glCellIdTextItems = []
        for i, cell_id in enumerate(all_cell_ids):
            a_fragile_linear_neuron_IDX = self.cell_id_to_fragile_linear_neuron_IDX_map[cell_id]
            # curr_color = pg.mkColor((i, self.n_cells*1.3))
            try:
                curr_color = self.params.neuron_qcolors_map[a_fragile_linear_neuron_IDX]
            except KeyError as e:
                print(f'_build_neuron_id_graphics(...): key error: {e}! i: {i}, cell_id: {cell_id}, a_fragile_linear_neuron_IDX: {a_fragile_linear_neuron_IDX} not found in {self.params.neuron_qcolors_map.keys()}')
                curr_color = self.params.neuron_qcolors[i]
            except Exception as e:
                raise e
            
            # curr_color = self.params.neuron_qcolors[i]
            curr_color.setAlphaF(1.0)
            # print(f'cell_id: {cell_id}, curr_color: {curr_color.alpha()}')
            curr_id_txtitem = gl.GLTextItem(pos=(self.side_wall_x, y_pos[i], (self.floor_z - 0.5)), text=f'{cell_id}', color=curr_color, font=cell_id_text_item_font)
            curr_id_txtitem.setObjectName(f'neuron_id_txtitem_fragile_IDX_{a_fragile_linear_neuron_IDX}_aclu_{cell_id}')
            w.addItem(curr_id_txtitem) # add to the current widget
            # add to the cell_ids array
            self.ui.glCellIdTextItems.append(curr_id_txtitem)
           
                
    def _update_neuron_id_graphics(self):
        """ updates the text items to indicate the neuron ID for each neuron in the df. """
        all_cell_ids = self.cell_ids
        assert len(self.ui.glCellIdTextItems) == len(all_cell_ids), f"we should already have correct number of neuron ID text items, but len(self.ui.glCellIdTextItems): {len(self.ui.glCellIdTextItems)} and len(all_cell_ids): {len(all_cell_ids)}!"
        assert len(self.ui.glCellIdTextItems) == len(self.series_identity_y_values), f"we should already have correct number of neuron ID text items, but len(self.ui.glCellIdTextItems): {len(self.ui.glCellIdTextItems)} and len(self.y): {len(self.series_identity_y_values)}!"
        # all_fragile_linear_neuron_IDXs = [self.fragile_linear_neuron_IDX_to_cell_id_map[a_cell_id] for a_cell_id in all_cell_ids] # get the list of all fragile_linear_neuron_IDXs
        
        for i, cell_id in enumerate(all_cell_ids):
        # for i, a_fragile_linear_neuron_IDX in enumerate(all_fragile_linear_neuron_IDXs):
            a_fragile_linear_neuron_IDX = self.cell_id_to_fragile_linear_neuron_IDX_map[cell_id]
            # curr_color = self.params.neuron_qcolors[i]
            try:
                curr_color = self.params.neuron_qcolors_map[a_fragile_linear_neuron_IDX]
            except KeyError as e:
                print(f'_build_neuron_id_graphics(...): key error: {e}! i: {i}, cell_id: {cell_id}, a_fragile_linear_neuron_IDX: {a_fragile_linear_neuron_IDX} not found in {self.params.neuron_qcolors_map.keys()}')
                curr_color = self.params.neuron_qcolors[i]
            except Exception as e:
                raise e
            
            curr_color.setAlphaF(1.0)
            curr_id_txtitem = self.ui.glCellIdTextItems[i]
            curr_id_txtitem.setObjectName(f'neuron_id_txtitem_fragile_IDX_{a_fragile_linear_neuron_IDX}_aclu_{cell_id}')
            curr_id_txtitem.setData(pos=(self.side_wall_x, self.fragile_linear_neuron_IDX_to_spatial(a_fragile_linear_neuron_IDX), (self.floor_z - 0.5)), color=curr_color) # TODO: could update color as well
            


    ###################################
    #### EVENT HANDLERS
    ##################################
    
    @pyqtExceptionPrintingSlot()
    def on_adjust_temporal_spatial_mapping(self):
        """ called when the spatio-temporal mapping property is changed.
        
        Should change whenever any of the following change:
            self.temporal_zoom_factor
            self.render_window_duration
            
        """
        self._update_axes_plane_graphics() # Adjust the three axes planes: 
        self.update_series_identity_y_values()
        self._update_neuron_id_graphics()


    def _update_plots(self):
        """ performance went:
        FROM:
            > Entering Spike3DRaster.on_window_changed
            Finished calling _update_plots(): 1179.6892 ms
            < Exiting Spike3DRaster.on_window_changed, total time: 1179.7600 ms

        TO:
            > Entering Spike3DRaster.on_window_changed
            Finished calling _update_plots(): 203.8840 ms
            < Exiting Spike3DRaster.on_window_changed, total time: 203.9544 ms

        Just by removing the lines that initialized the color. Conclusion is that pg.mkColor((cell_id, self.n_cells*1.3)) must be VERY slow.
    
        """
        if self.enable_debug_print:
            print(f'Spike3DRaster._update_plots()')
        assert (len(self.ui.gl_line_plots) == self.n_cells), f"after all operations the length of the plots array should be the same as the n_cells, but len(self.ui.gl_line_plots): {len(self.ui.gl_line_plots)} and self.n_cells: {self.n_cells}!"
        # build the position range for each unit along the y-axis:
        # y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode='zero_centered', bin_position_mode='bin_center', side_bin_margins = self.params.side_bin_margins)
        # self.y = DataSeriesToSpatial.build_series_identity_axis(self.n_cells, center_mode=self.params.center_mode, bin_position_mode=self.params.bin_position_mode, side_bin_margins = self.params.side_bin_margins)
        
        # Plot each unit one at a time:
        for i, a_fragile_linear_neuron_IDX in enumerate(self.fragile_linear_neuron_IDXs):    
            # Filter the dataframe using that column and value from the list
            curr_cell_df = self.active_windowed_df[self.active_windowed_df['fragile_linear_neuron_IDX']==a_fragile_linear_neuron_IDX]
            curr_spike_t = curr_cell_df[curr_cell_df.spikes.time_variable_name].to_numpy() # this will map
            # efficiently get curr_spike_t by filtering for unit and column at the same time
            # curr_spike_t = self.active_windowed_df.loc[self.active_windowed_df.spikes.time_variable_name, (self.active_windowed_df['fragile_linear_neuron_IDX']==cell_id)].values # .to_numpy()
            
            curr_unit_n_spikes = len(curr_spike_t)
            
            yi = self.fragile_linear_neuron_IDX_to_spatial(a_fragile_linear_neuron_IDX)
            # yi = self.series_identity_y_values[i] # get the correct y-position for all spikes of this cell
            # map the current spike times back onto the range of the window's (-half_render_window_duration, +half_render_window_duration) so they represent the x coordinate
            curr_x = DataSeriesToSpatial.temporal_to_spatial_map(curr_spike_t, self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time, self.temporal_axis_length, center_mode='zero_centered')
            # curr_paired_x = np.squeeze(interleave_elements(np.atleast_2d(curr_x).T, np.atleast_2d(curr_x).T))
            curr_paired_x = curr_x.repeat(2)
            
            # Z-positions:
            curr_paired_spike_zs = np.squeeze(np.tile(np.array([self.params.spike_start_z, self.params.spike_end_z]), curr_unit_n_spikes)) # repeat pair of z values once for each spike
        
            # Build lines:
            pts = np.column_stack([curr_paired_x, np.full_like(curr_paired_x, yi), curr_paired_spike_zs]) # the middle coordinate is the size of the x array with the value given by yi. yi must be the scalar for this cell.
            # plt = gl.GLLinePlotItem(pos=pts, color=curr_color, width=0.5, antialias=True, mode='lines') # mode='lines' means that each pair of vertexes draws a single line segement
            self.ui.gl_line_plots[i].setData(pos=pts, mode='lines') # update the current data
            self.ui.gl_line_plots[i].setObjectName(f'main_line_plots_fragile_IDX_{a_fragile_linear_neuron_IDX}') # shouldn't be needed
           
    
        # Update the additional display lines information on the overlay:
        # self.ui.viewport_overlay.additional_overlay_text_lines = self.overlay_text_lines
        self.ui.viewport_overlay.additional_overlay_text_dict = self.overlay_text_lines_dict
        
        # Update the epochs if we have them:
        self.RenderTimeEpoch3DMeshesMixin_on_window_update()
        # Update 3D Curves if we have them:
        self.TimeCurvesViewMixin_on_window_update()
        
        
    def rebuild_main_gl_line_plots_if_needed(self, debug_print=True):
        """ adds or removes GLLinePlotItems to self.ui.gl_line_plots based on the current number of cells. """
        n_extant_plts = len(self.ui.gl_line_plots)
        if (n_extant_plts < self.n_cells):
            # need to create new plots for the difference
            if debug_print:
                print(f'!! Spike3DRaster.rebuild_main_gl_line_plots_if_needed(): building additional plots: n_extant_plts: {n_extant_plts}, self.n_cells: {self.n_cells}')
            for new_unit_i in np.arange(n_extant_plts-1, self.n_cells, 1):
                cell_id = self.fragile_linear_neuron_IDXs[new_unit_i]
                # curr_color = pg.mkColor((cell_id, self.n_cells*1.3))
                # curr_color.setAlphaF(0.5)
                curr_color = self.params.neuron_qcolors[cell_id] # get the pre-build color
                plt = gl.GLLinePlotItem(pos=[], color=curr_color, width=1.0, antialias=True, mode='lines') # mode='lines' means that each pair of vertexes draws a single line segement
                plt.setObjectName(f'main_line_plots_fragile_IDX_{cell_id}')
                # plt.setYRange((-self.n_half_cells - self.side_bin_margins), (self.n_half_cells + self.side_bin_margins))
                # plt.setXRange(-self.half_render_window_duration, +self.half_render_window_duration)
                self.ui.main_gl_widget.addItem(plt)
                self.ui.gl_line_plots.append(plt) # append to the gl_line_plots array
                
        elif (n_extant_plts > self.n_cells):
            # excess plots, need to remove (or at least hide) them:              
            if debug_print:
                print(f'!! Spike3DRaster.rebuild_main_gl_line_plots_if_needed(): removing excess plots: n_extant_plts: {n_extant_plts}, self.n_cells: {self.n_cells}')
            for extra_unit_i in np.arange(n_extant_plts, self.n_cells, 1):
                plt = self.ui.gl_line_plots[extra_unit_i] # get the unit to be removed 
                self.ui.main_gl_widget.removeItem(plt)
            # remove from the array
            del self.ui.gl_line_plots[n_extant_plts:] # from n_extant_plts up to the end of the list
        else:
            return # the correct number of items are already in the list
        
        assert (len(self.ui.gl_line_plots) == self.n_cells), f"after all operations the length of the plots array should be the same as the n_cells, but len(self.ui.gl_line_plots): {len(self.ui.gl_line_plots)} and self.n_cells: {self.n_cells}!"

            
    def dynamic_add_widget(self, widget):
        """ adds a widget dynamically to the viewport """
        
        # Adds the widget with addItem:
        self.ui.main_gl_widget.addItem(widget)


    def update(self, sort_changed=True, colors_changed=True):
        """ refreshes the raster when the colors or sort change. 
        
        NOTE: Unlike Spike2DRaster, sorts seem to be able to be updated independent of colors.
        
        """
        if sort_changed:
            # rebuild the position range for each unit along the y-axis:
            self.update_series_identity_y_values()
            self._update_neuron_id_graphics() # rebuild the text labels
            self._update_plots()
                        
        if colors_changed:
            ## Spike3DRaster:
            # ._update_neuron_id_graphics()
            # .rebuild_main_gl_line_plots_if_needed()
            self._update_neuron_id_graphics() # rebuild the text labels
            self.rebuild_main_gl_line_plots_if_needed()
            for i, a_fragile_linear_neuron_IDX in enumerate(self.fragile_linear_neuron_IDXs):
                # color= (N,4) array of floats (0.0-1.0) or tuple of floats specifying a single color for the entire item.
                curr_color = self.params.neuron_qcolors[a_fragile_linear_neuron_IDX] # get the pre-build color
                self.ui.gl_line_plots[i].setData(color=curr_color) # update the current data
            self._update_plots()
            

        
    # unit_sort_order_changed_signal
    @pyqtExceptionPrintingSlot(object)
    def on_unit_sort_order_changed(self, new_sort_order):
        print(f'unit_sort_order_changed_signal(new_sort_order: {new_sort_order})')        
        self.update(sort_changed=True, colors_changed=False)


    @pyqtExceptionPrintingSlot(object)
    def on_neuron_colors_changed(self, neuron_id_color_update_dict):
        """ Called when the neuron colors have finished changing (changed) to update the rendered elements.
        """
        print(f'Spike3DRaster.neuron_id_color_update_dict: {neuron_id_color_update_dict}')
        self.update(sort_changed=False, colors_changed=True)
        
        
        


# Start Qt event loop unless running in interactive mode.
# if __name__ == '__main__':
#     # v = Visualizer()
#     v = Spike3DRaster()
#     v.animation()
# sfs

