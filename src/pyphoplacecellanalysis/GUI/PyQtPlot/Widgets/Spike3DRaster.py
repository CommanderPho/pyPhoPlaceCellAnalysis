import time
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np

from pyphocorehelpers.indexing_helpers import interleave_elements, partition
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

import qdarkstyle
import numpy as np
import time

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GLDebugAxisItem import GLDebugAxisItem
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GLViewportOverlayPainterItem import GLViewportOverlayPainterItem

""" For threading info see:
https://stackoverflow.com/questions/41526832/pyqt5-qthread-signal-not-working-gui-freeze


"""

def trap_exc_during_debug(*args):
    # when app raises uncaught exception, print info
    print(args)


# install exception hook: without this, uncaught exception would cause application to exit
sys.excepthook = trap_exc_during_debug




class SpikesDataframeWindow(QtCore.QObject):
    """ a zoomable (variable sized) window into a dataframe with a time axis 
    
    active_window_start_time can be adjusted to set the location of the current window.

    Usage:
        render_window_duration = 60.0
        curr_spikes_df_window = SpikesDataframeWindow(curr_spikes_df, window_duration=render_window_duration)
        curr_spikes_df_window

    """
    spike_dataframe_changed_signal = QtCore.pyqtSignal() # signal emitted when the spike dataframe is changed, which might change the number of units, number of spikes, and other properties.
    window_duration_changed_signal = QtCore.pyqtSignal() # more conservitive singal that only changes when the duration of the window changes
    window_changed_signal = QtCore.pyqtSignal()
    
    @property
    def active_windowed_df(self):
        """The dataframe sliced to the current time window (active_time_window)"""
        return self.df[self.df[self.df.spikes.time_variable_name].between(self.active_time_window[0], self.active_time_window[1])]

    @property
    def active_time_window(self):
        """ a 2-element time window [start_time, end_time]"""
        return [self.active_window_start_time, self.active_window_end_time]
        
    @property
    def active_window_end_time(self):
        """The active_window_end_time property."""
        return (self.active_window_start_time + self.window_duration)
        
    @property
    def active_window_num_spikes(self):
        """The number of spikes (across all units) in the active window."""
        return self.active_windowed_df.shape[0] 
    
    @property
    def total_df_start_end_times(self):
        """[earliest_df_time, latest_df_time]: The earliest and latest spiketimes in the total df """
        earliest_df_time = np.nanmin(self.df[self.df.spikes.time_variable_name])
        latest_df_time = np.nanmax(self.df[self.df.spikes.time_variable_name])
        
        df_timestamps = self.df[self.df.spikes.time_variable_name].to_numpy()
        earliest_df_time = df_timestamps[0]
        latest_df_time = df_timestamps[-1]
        return [earliest_df_time, latest_df_time]
            
    ##### Get/Set Properties ####:
    @property
    def df(self):
        """The df property."""
        return self._df
    @df.setter
    def df(self, value):
        self._df = value
        self.spike_dataframe_changed_signal.emit()
        
    @property
    def window_duration(self):
        """The window_duration property."""
        return self._window_duration
    @window_duration.setter
    def window_duration(self, value):
        self._window_duration = value
        self.window_duration_changed_signal.emit() # emit window duration changed signal
        self.window_changed_signal.emit() # emit window changed signal
        
    @property
    def active_window_start_time(self):
        """The current start time of the sliding time window"""
        return self._active_window_start_time
    @active_window_start_time.setter
    def active_window_start_time(self, value):
        self._active_window_start_time = value
        self.window_changed_signal.emit() # emit window changed signal
    
    def __init__(self, spikes_df, window_duration=15.0, window_start_time=0.0):
        QtCore.QObject.__init__(self)
        self._df = spikes_df
        self._window_duration = window_duration
        self._active_window_start_time = window_start_time
        self.window_changed_signal.connect(self.on_window_changed)
        
    @QtCore.pyqtSlot(float)
    def update_window_start(self, new_value):
        self.active_window_start_time = new_value

        
    def on_window_changed(self):
        print(f'SpikesDataframeWindow.on_window_changed(): window_changed_signal emitted. self.active_time_window: {self.active_time_window}')
        
        
        
        
class SliderRunner(QtCore.QThread):
    update_signal = QtCore.pyqtSignal()

    def __init__(self):
        QtCore.QThread.__init__(self)

    def run(self):
        while(True):
            self.update_signal.emit()
            time.sleep(.03) # probably do a different form of rate limiting instead (like use SignalProxy)? Actually this might be okay because it's on a different thread.
            
                

class Spike3DRaster(QtWidgets.QWidget):
    """docstring for 3d_raster_app."""
    
    @property
    def spikes_window(self):
        """The spikes_window property."""
        return self._spikes_window

    @property
    def active_windowed_df(self):
        """ """
        return self.spikes_window.active_windowed_df
    
    @property
    def unit_ids(self):
        """The unit_ids from the whole df (not just the current window)"""
        return np.unique(self.spikes_window.df['unit_id'].to_numpy())
    
    @property
    def n_cells(self):
        """The number_units property."""
        return len(self.unit_ids)
    
    @property
    def n_half_cells(self):
        """ """
        return np.ceil(float(self.n_cells)/2.0)
    
    @property
    def n_full_cell_grid(self):
        """ """
        return 2.0 * self.n_half_cells # could be one more than n
    
    @property
    def render_window_duration(self):
        """ """
        return float(self.spikes_window.window_duration)
    
    @property
    def half_render_window_duration(self):
        """ """
        return np.ceil(float(self.spikes_window.window_duration)/2.0) # 10 by default 

    @property
    def temporal_axis_length(self):
        """The temporal_axis_length property."""
        return self.temporal_zoom_factor * self.render_window_duration
    @property
    def half_temporal_axis_length(self):
        """The temporal_axis_length property."""
        return self.temporal_axis_length / 2.0
    

    @property
    def temporal_zoom_factor(self):
        """The time dilation factor that maps spikes in the current window to x-positions along the time axis multiplicatively.
            Increasing this factor will result in a more spatially expanded time axis while leaving the visible window unchanged.
        """
        return self._temporal_zoom_factor
    @temporal_zoom_factor.setter
    def temporal_zoom_factor(self, value):
        self._temporal_zoom_factor = value


    def __init__(self, spikes_df, *args, window_duration=15.0, window_start_time=0.0, **kwargs):
        super(Spike3DRaster, self).__init__(*args, **kwargs)
        # Initialize member variables:
        self.slidebar_val = 0
        self._spikes_window = SpikesDataframeWindow(spikes_df, window_duration=window_duration, window_start_time=window_start_time)
        self.spike_start_z = -10.0
        self.spike_end_z = 0.1
        self.side_bin_margins = 1.0 # space to sides of the first and last cell on the y-axis
        # by default we want the time axis to approximately span -20 to 20. So we set the temporal_zoom_factor to 
        self._temporal_zoom_factor = 40.0 / float(self.render_window_duration)
        
        self.enable_debug_print = True
        
        
        self.app = pg.mkQApp("Spike3DRaster")
        
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
        # build the UI components:
        self.buildUI()
        

        
    def buildUI(self):
        self.ui = PhoUIContainer()
        
        # widget = QWidget()
        # widget.setLayout(layout)
        # self.setCentralWidget(widget)
        
        self.ui.layout = QtWidgets.QGridLayout()
        self.ui.layout.setContentsMargins(0, 0, 0, 0)
        self.ui.layout.setVerticalSpacing(0)
        self.ui.layout.setHorizontalSpacing(0)
        self.setStyleSheet("background : #1B1B1B; color : #727272")
        
        ##### Main Raster Plot Content Top ##########
        self.ui.main_gl_widget = gl.GLViewWidget()
        # self.ui.main_gl_widget.show()
        self.ui.main_gl_widget.resize(1000,600)
        # self.ui.main_gl_widget.setWindowTitle('pyqtgraph: 3D Raster Spikes Plotting')
        self.ui.main_gl_widget.setCameraPosition(distance=40)
        self.ui.layout.addWidget(self.ui.main_gl_widget, 0, 0) # add the GLViewWidget to the layout
        
        #### Build Graphics Objects #####
        self._buildGraphics(self.ui.main_gl_widget) # pass the GLViewWidget
        
        ####    Slide Bar Left #######
        self.ui.panel_slide_bar = QtWidgets.QWidget()
        self.ui.panel_slide_bar.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        self.ui.panel_slide_bar.setMaximumHeight(50.0)
        self.ui.layout_slide_bar = QtWidgets.QHBoxLayout()
        self.ui.layout_slide_bar.setContentsMargins(6, 3, 4, 4)

        self.ui.panel_slide_bar.setLayout(self.ui.layout_slide_bar)

        self.ui.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ui.slider.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        # self.ui.slider.setFocusPolicy(Qt.NoFocus) # removes ugly focus rectangle frm around the slider
        self.ui.slider.setRange(0, 100)
        self.ui.slider.setSingleStep(1)
        # self.ui.slider.setSingleStep(2)
        self.ui.slider.setValue(0)
        self.ui.slider.valueChanged.connect(self.slider_val_changed)
        # sliderMoved vs valueChanged? vs sliderChange?
        

        self.ui.layout_slide_bar.addWidget(self.ui.slider)

        self.ui.btn_slide_run = QtWidgets.QPushButton(">")
        self.ui.btn_slide_run.setMinimumHeight(25)
        self.ui.btn_slide_run.setMinimumWidth(30)
        self.ui.btn_slide_run.tag = "paused"
        self.ui.btn_slide_run.clicked.connect(self.btn_slide_run_clicked)

        self.ui.layout_slide_bar.addWidget(self.ui.btn_slide_run)
        self.ui.layout.addWidget(self.ui.panel_slide_bar, 1, 0) 
        
        self.setLayout(self.ui.layout)
        self.resize(1920, 900)
        self.setWindowTitle('Spike3DRaster')
        # Connect window update signals
        # self.spikes_window.spike_dataframe_changed_signal.connect(self.on_spikes_df_changed)
        # self.spikes_window.window_duration_changed_signal.connect(self.on_window_duration_changed)
        self.spikes_window.window_changed_signal.connect(self.on_window_changed)

        # Slider update thread:        
        self.sliderThread = SliderRunner()
        self.sliderThread.update_signal.connect(self.increase_slider_val)
        self.show()
      
    def _buildGraphics(self, w):
        # Add debugging widget:
        
        # Adds a helper widget that displays the x/y/z vector at the origin:
        self.ui.ref_axes_indicator = GLDebugAxisItem()
        self.ui.ref_axes_indicator.setSize(x=15.0, y=10.0, z=5.0)
        w.addItem(self.ui.ref_axes_indicator)
        
        self.ui.viewport_overlay = GLViewportOverlayPainterItem()
        w.addItem(self.ui.viewport_overlay)
        
        # Add axes planes:
        # X-plane:
        x_color = (255, 155, 155, 76.5)
        self.ui.gx = gl.GLGridItem(color=x_color) # 'x' plane, red
        self.ui.gx.rotate(90, 0, 1, 0)
        self.ui.gx.translate(-self.half_temporal_axis_length, 0, 0) # shift backwards
        self.ui.gx.setSize(20, self.n_full_cell_grid) # std size in z-dir, n_cell size across
        self.ui.gx.setSpacing(10.0, 1) 
        w.addItem(self.ui.gx)
        
        self.ui.x_txtitem = gl.GLTextItem(pos=(-self.half_temporal_axis_length, self.n_half_cells, 0.0), text='x', color=x_color) # position label 
        w.addItem(self.ui.x_txtitem)

        # Y-plane:
        y_color = (155, 255, 155, 76.5)
        self.ui.gy = gl.GLGridItem(color=y_color) # 'y' plane, green
        self.ui.gy.rotate(90, 1, 0, 0)
        # gy.translate(0, -10, 0)
        self.ui.gy.translate(0, -self.n_half_cells, 0) # offset by half the number of units in the -y direction
        self.ui.gy.setSize(self.temporal_axis_length, 20)
        self.ui.gy.setSpacing(1, 10.0) # unit along the y axis itself, only one subdivision along the z-axis
        w.addItem(self.ui.gy)
        
        self.ui.y_txtitem = gl.GLTextItem(pos=(self.half_temporal_axis_length+0.5, -self.n_half_cells, 0.0), text='y', color=y_color)
        w.addItem(self.ui.y_txtitem)
        
        # XY-plane (with normal in z-dir):
        z_color = (155, 155, 255, 76.5)
        self.ui.gz = gl.GLGridItem(color=z_color) # 'z' plane, blue
        self.ui.gz.translate(0, 0, -10) # Shift down by 10 units in the z-dir
        self.ui.gz.setSize(self.temporal_axis_length, self.n_full_cell_grid)
        self.ui.gz.setSpacing(20.0, 1)
        # gz.setSize(n_full_cell_grid, n_full_cell_grid)
        w.addItem(self.ui.gz)
        
        self.ui.z_txtitem = gl.GLTextItem(pos=(-self.half_temporal_axis_length, -self.n_half_cells, 10.5), text='z', color=z_color)
        w.addItem(self.ui.z_txtitem)
        
        # Custom 3D raster plot:
        
        # TODO: EFFICIENCY: For a potentially considerable speedup, could compute the "curr_x" values for all cells at once and add as a column to the dataframe since it only depends on the current window parameters (changes when window changes).
            ## OH, but the window changes every frame update (as that's what it means to animate the spikes as a function of time). Maybe not a big speedup.
        
        self.ui.gl_line_plots = [] # create an empty array for each GLLinePlotItem, of which there will be one for each unit.
        
        
                    
        y = np.linspace(-self.n_half_cells, self.n_half_cells, self.n_cells) + 0.5 # add 0.5 so they're centered
        # Plot each unit one at a time:
        for cell_id in self.unit_ids:
            curr_color = pg.mkColor((cell_id, self.n_cells*1.3))
            # curr_color.SetAlpha(120) # alpha should be between 0-255
            curr_color.setAlphaF(0.5)
            # print(f'cell_id: {cell_id}, curr_color: {curr_color.alpha()}')
            
            # Filter the dataframe using that column and value from the list
            curr_cell_df = self.active_windowed_df[self.active_windowed_df['unit_id']==cell_id].copy() # is .copy() needed here since nothing is updated???
            # curr_unit_id = curr_cell_df['unit_id'].to_numpy() # this will map to the y position
            curr_spike_t = curr_cell_df[curr_cell_df.spikes.time_variable_name].to_numpy() # this will map 
            yi = y[cell_id] # get the correct y-position for all spikes of this cell
            # print(f'cell_id: {cell_id}, yi: {yi}')
            # map the current spike times back onto the range of the window's (-half_render_window_duration, +half_render_window_duration) so they represent the x coordinate
            curr_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (-self.half_temporal_axis_length, +self.half_temporal_axis_length))
            curr_paired_x = np.squeeze(interleave_elements(np.atleast_2d(curr_x).T, np.atleast_2d(curr_x).T))        
            
            # Z-positions:
            # z = curr_spike_t[np.arange(100)] # get the first 20 spikes for each
            spike_bottom_zs = np.full_like(curr_x, self.spike_start_z)
            spike_top_zs = np.full_like(curr_x, self.spike_end_z)
            curr_paired_spike_zs = np.squeeze(interleave_elements(np.atleast_2d(spike_bottom_zs).T, np.atleast_2d(spike_top_zs).T)) # alternating top and bottom z-positions
        
            # sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
            # sp1.translate(5,5,0)
            # w.addItem(sp1)
            
            # Build lines:
            pts = np.column_stack([curr_paired_x, np.full_like(curr_paired_x, yi), curr_paired_spike_zs]) # the middle coordinate is the size of the x array with the value given by yi. yi must be the scalar for this cell.
            # pts = np.column_stack([x, np.full_like(x, yi), z]) # the middle coordinate is the size of the x array with the value given by yi. yi must be the scalar for this cell.
            # plt = gl.GLLinePlotItem(pos=pts, color=pg.mkColor((cell_id,n*1.3)), width=(cell_id+1)/10., antialias=True)
            plt = gl.GLLinePlotItem(pos=pts, color=curr_color, width=0.5, antialias=True, mode='lines') # mode='lines' means that each pair of vertexes draws a single line segement

            # plt.setYRange((-self.n_half_cells - self.side_bin_margins), (self.n_half_cells + self.side_bin_margins))
            # plt.setXRange(-self.half_render_window_duration, +self.half_render_window_duration)
            
            w.addItem(plt)
            self.ui.gl_line_plots.append(plt)
            
            # # Adds a helper widget that displays the x/y/z vector at the origin:
            # ref_axes_indicator = gl.GLAxisItem()
            # ref_axes_indicator.setSize(x=10.0, y=10.0, z=5.0)
            # w.addItem(ref_axes_indicator)

    def on_spikes_df_changed(self):
        """ changes:
            self.unit_ids
            self.n_full_cell_grid
        """
        debug_print=True
        if debug_print:
            print(f'Spike3DRaster.on_spikes_df_changed()')
        # TODO: these '.translate(...)' instructions might not be right if they're relative to the original transform. May need to translate back to by the inverse of the old value, and then do the fresh transform with the new value. Or compute the difference between the old and new.
        self.ui.gx.setSize(20, self.n_full_cell_grid) # std size in z-dir, n_cell size across
        self.ui.gy.translate(0, -self.n_half_cells, 0) # offset by half the number of units in the -y direction
        self.ui.gz.setSize(self.temporal_axis_length, self.n_full_cell_grid)
        self.rebuild_main_gl_line_plots_if_needed()
        

    def on_window_duration_changed(self):
        """ changes self.half_render_window_duration """
        print(f'Spike3DRaster.on_window_duration_changed()')
        self.ui.gx.translate(-self.half_temporal_axis_length, 0, 0) # shift backwards
        self.ui.gy.setSize(self.temporal_axis_length, 20)
        self.ui.gz.setSize(self.temporal_axis_length, self.n_full_cell_grid)
        # update grids. on_window_changed should be triggered separately        
        
    def on_window_changed(self):
        # called when the window is updated
        if self.enable_debug_print:
            print(f'Spike3DRaster.on_window_changed()')
        profiler = pg.debug.Profiler(disabled=False, delayed=False)
        self._update_plots()
        profiler('Finished calling _update_plots()')
        
            
    def _update_plots(self):
        if self.enable_debug_print:
            print(f'Spike3DRaster._update_plots()')
        assert (len(self.ui.gl_line_plots) == self.n_cells), f"after all operations the length of the plots array should be the same as the n_cells, but len(self.ui.gl_line_plots): {len(self.ui.gl_line_plots)} and self.n_cells: {self.n_cells}!"
        y = np.linspace(-self.n_half_cells, self.n_half_cells, self.n_cells) + 0.5 # add 0.5 so they're centered
        
        # Plot each unit one at a time:
        for cell_id in self.unit_ids:
            curr_color = pg.mkColor((cell_id, self.n_cells*1.3))
            curr_color.setAlphaF(0.5)            
            # Filter the dataframe using that column and value from the list
            curr_cell_df = self.active_windowed_df[self.active_windowed_df['unit_id']==cell_id].copy() # is .copy() needed here since nothing is updated???
            curr_spike_t = curr_cell_df[curr_cell_df.spikes.time_variable_name].to_numpy() # this will map 
            yi = y[cell_id] # get the correct y-position for all spikes of this cell
            # map the current spike times back onto the range of the window's (-half_render_window_duration, +half_render_window_duration) so they represent the x coordinate
            # curr_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (-self.half_render_window_duration, +self.half_render_window_duration))
            curr_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (-self.half_temporal_axis_length, +self.half_temporal_axis_length))
            curr_paired_x = np.squeeze(interleave_elements(np.atleast_2d(curr_x).T, np.atleast_2d(curr_x).T))        
            
            # Z-positions:
            spike_bottom_zs = np.full_like(curr_x, self.spike_start_z)
            spike_top_zs = np.full_like(curr_x, self.spike_end_z)
            curr_paired_spike_zs = np.squeeze(interleave_elements(np.atleast_2d(spike_bottom_zs).T, np.atleast_2d(spike_top_zs).T)) # alternating top and bottom z-positions
        
            # sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
            # sp1.translate(5,5,0)
            # w.addItem(sp1)
            
            # Build lines:
            pts = np.column_stack([curr_paired_x, np.full_like(curr_paired_x, yi), curr_paired_spike_zs]) # the middle coordinate is the size of the x array with the value given by yi. yi must be the scalar for this cell.
            # plt = gl.GLLinePlotItem(pos=pts, color=curr_color, width=0.5, antialias=True, mode='lines') # mode='lines' means that each pair of vertexes draws a single line segement
            self.ui.gl_line_plots[cell_id].setData(pos=pts, mode='lines') # update the current data
            
            # self.ui.main_gl_widget.addItem(plt)
            # self.ui.gl_line_plots.append(plt) # append to the gl_line_plots array
            
    def rebuild_main_gl_line_plots_if_needed(self, debug_print=True):
        """ adds or removes GLLinePlotItems to self.ui.gl_line_plots based on the current number of cells. """
        n_extant_plts = len(self.ui.gl_line_plots)
        if (n_extant_plts < self.n_cells):
            # need to create new plots for the difference
            if debug_print:
                print(f'!! Spike3DRaster.rebuild_main_gl_line_plots_if_needed(): building additional plots: n_extant_plts: {n_extant_plts}, self.n_cells: {self.n_cells}')
            for new_unit_i in np.arange(n_extant_plts-1, self.n_cells, 1):
                cell_id = self.unit_ids[new_unit_i]
                curr_color = pg.mkColor((cell_id, self.n_cells*1.3))
                curr_color.setAlphaF(0.5)
                plt = gl.GLLinePlotItem(pos=[], color=curr_color, width=0.5, antialias=True, mode='lines') # mode='lines' means that each pair of vertexes draws a single line segement
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

    def _compute_window_transform(self, relative_offset):
        """ computes the transform from 0.0-1.0 as the slider would provide to the offset given the current information. """
        earliest_t, latest_t = self.spikes_window.total_df_start_end_times
        total_spikes_df_duration = latest_t - earliest_t # get the duration of the entire spikes df
        render_window_offset = (total_spikes_df_duration * relative_offset) + earliest_t
        return render_window_offset
    
    
    def btn_slide_run_clicked(self):
        if self.ui.btn_slide_run.tag == "paused" or self.slidebar_val == 1:
            if self.slidebar_val == 1:
                self.ui.slider.setValue(0)
            
            self.ui.btn_slide_run.setText("||")
            self.ui.btn_slide_run.tag = "running"
            self.sliderThread.start()

        elif self.ui.btn_slide_run.tag == "running":
            self.ui.btn_slide_run.setText(">")
            self.ui.btn_slide_run.tag = "paused"
            self.sliderThread.terminate()

    def increase_slider_val(self):
        slider_val = self.ui.slider.value()
        if self.enable_debug_print:
            print(f'Spike3DRaster.increase_slider_val(): slider_val: {slider_val}')
        if slider_val < 100:
            self.ui.slider.setValue(slider_val + 1)
        else:
            print("thread ended..")
            self.ui.btn_slide_run.setText(">")
            self.ui.btn_slide_run.tag = "paused"
            self.sliderThread.terminate()

    def slider_val_changed(self, val):
        self.slidebar_val = val / 100
        # Gets the transform from relative (0.0 - 1.0) to absolute timestamp offset
        curr_t = self._compute_window_transform(self.slidebar_val)
        
        if self.enable_debug_print:
            print(f'Spike3DRaster.slider_val_changed(): self.slidebar_val: {self.slidebar_val}, curr_t: {curr_t}')
            print(f'BEFORE: self.spikes_window.active_time_window: {self.spikes_window.active_time_window}')
         # set the start time which will trigger the update cascade and result in on_window_changed(...) being called
        self.spikes_window.update_window_start(curr_t)
        if self.enable_debug_print:
            print(f'AFTER: self.spikes_window.active_time_window: {self.spikes_window.active_time_window}')
    
        
    # def computeTransform(self, x, y, t = None):
    #     if t == None:
    #         v1_x = (1 * (1 - self.slidebar_val)) + (self.v1_x * self.slidebar_val)
    #         v1_y = (0 * (1 - self.slidebar_val)) + (self.v1_y * self.slidebar_val)

    #         v2_y = (1 * (1 - self.slidebar_val)) + (self.v2_y * self.slidebar_val)
    #         v2_x = (0 * (1 - self.slidebar_val)) + (self.v2_x * self.slidebar_val)
    #     else:
    #         v1_x = self.v1_x
    #         v1_y = self.v1_y
    #         v2_x = self.v2_x
    #         v2_y = self.v2_y
    #     return ((v1_x * x) + (v2_x * y), (v1_y * x) + (v2_y * y))

    # def set_spikes_data(self, curr_spikes_df):
    #     unit_ids = np.unique(curr_spikes_df['unit_id'].to_numpy())
    #     self.n_cells = len(unit_ids)
    #     render_window_duration = 60.0 # in seconds, 1minute by default
    #     spike_start_z = -10.0
    #     spike_end_z = 0.1
        
    #     n_half_cells = np.ceil(float(self.n_cells)/2.0)
    #     n_full_cell_grid = 2.0 * n_half_cells # could be one more than n
    #     half_render_window_duration = np.ceil(float(render_window_duration)/2.0) # 10 by default
        
    #     print(f'plot_3d_raster_plot(...): unit_ids: {unit_ids}, n: {self.n_cells}')
    
