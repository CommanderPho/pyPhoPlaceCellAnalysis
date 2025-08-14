# AnimalTrajectoryPlottingMixin
import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
from qtpy import QtCore, QtWidgets

class AnimalTrajectoryPlottingMixin:
    """ Implementors render a trajectory through space
    
    Used by TimeSynchronizedOccupancyPlotter and TimeSynchronizedPlacefieldsPlotter
    """
    
    @property
    def curr_recent_trajectory(self):
        """The animal's most recent trajectory preceding self.active_time_dependent_placefields.last_t"""
        # Fixed time ago backward:
        earliest_trajectory_start_time = self.last_t - self.params.recent_position_trajectory_max_seconds_ago # gets the earliest start time for the current trajectory to display
        return self.active_time_dependent_placefields.all_time_filtered_pos_df.position.time_sliced(earliest_trajectory_start_time, self.last_t)[['t','x','y']] # Get all rows within the most recent time
    
    
    @property
    def curr_position(self):
        # .iloc[-1:] gets the last row of a dataframe as another dataframe
        return self.active_time_dependent_placefields.filtered_pos_df.iloc[-1:][['t','x','y']] # Get only the most recent row


    @QtCore.Slot()
    def AnimalTrajectoryPlottingMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        pass

    @QtCore.Slot()
    def AnimalTrajectoryPlottingMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        self.params.recent_position_trajectory_max_seconds_ago = 7.0
        
        self.params.trajectory_path_current_position_marker_size = 25.0
        # self.params.trajectory_path_marker_max_fill_opacity = 255
        self.params.trajectory_path_marker_max_fill_opacity = 200 # out of 255
        
        ### Path Settings:
        self.params.recent_position_trajectory_path_pen = pg.mkPen({'color': [255, 255, 255, 10], 'width': 1}) # White
        # path_shadow_pen = pg.mkPen({'color': [0, 0, 0, 100], 'width': 20})
        self.params.recent_position_trajectory_path_shadow_pen = None
        
        ### Marker Settings:
        # self.params.recent_position_trajectory_symbol_pen = pg.mkPen({'color': [255, 255, 255, 10], 'width': 1}) # White
        self.params.recent_position_trajectory_symbol_pen = pg.mkPen({'color': [20, 20, 20, 255], 'width': 1}) # Black
        self.params.trajectory_path_current_position_marker_brush = pg.mkBrush(0, 255, 0, 200)
  
  

    @QtCore.Slot()
    def AnimalTrajectoryPlottingMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        ## Optional Animal Trajectory Path Plot:            
        # Note that pg.PlotDataItem is a combination of pg.PlotCurveItem and pg.ScatterPlotItem
        self.ui.trajectory_curve = pg.PlotDataItem(pen=self.params.recent_position_trajectory_path_pen, shadowPen=self.params.recent_position_trajectory_path_shadow_pen,
                                                   symbol='o', symbolBrush=(50,50,50), pxMode=True, symbolSize=6.0, symbolPen=self.params.recent_position_trajectory_symbol_pen,
                                                   antialias=True, name='recent trajectory') #downsample=20, downsampleMethod='peak', autoDownsample=True, skipFiniteCheck=True, clipToView=True
        
        # curr_occupancy_plotter.ui.trajectory_curve = pg.PlotCurveItem(pen=({'color': 'white', 'width': 3}), skipFiniteCheck=True)
        self.ui.root_plot.addItem(self.ui.trajectory_curve)
        
        

    @QtCore.Slot()
    def AnimalTrajectoryPlottingMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        pass

    @QtCore.Slot(float, float)
    def AnimalTrajectoryPlottingMixin_on_window_update(self, new_start=None, new_end=None):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc. """
        pass
    
    
    def AnimalTrajectoryPlottingMixin_update_plots(self):
        """ 
        requires: self.curr_recent_trajectory
        
        """
        # Update most recent trajectory plot:
        curr_trajectory_rows = self.curr_recent_trajectory
        curr_num_points = np.shape(curr_trajectory_rows)[0]
        
        ## Build Current Visual Settings:
        if curr_num_points > 0:
            # Fixed size for all points:
            curr_desired_sizes = np.full((curr_num_points,), 20.0) # build an array of all ones of the same size as the number of points in the current path
            # Decaying size over time
            # decay_sizes = (curr_trajectory_rows['t'].to_numpy() - curr_occupancy_plotter.last_t) + curr_occupancy_plotter.params.recent_position_trajectory_max_seconds_ago
            # curr_desired_sizes = decay_sizes

            # Map onto a range of sizes from 0-20
            # curr_desired_sizes = np.interp(curr_desired_sizes, [0, curr_occupancy_plotter.params.recent_position_trajectory_max_seconds_ago], [0,20]) # map onto a size range from 0-20

            # Fading Color over time:
            desired_symbol_brushes = [pg.mkBrush(255, 255, 255, np.interp(i, [0,(curr_num_points-1)], [0,self.params.trajectory_path_marker_max_fill_opacity])) for i in np.arange(curr_num_points-1)] # -1 for the special last symbol
            # Fixed Color for all but the last point
            # fading_brush_color = pg.mkBrush(50, 50, 50, 100) # pg.mkBrush(R, G, B, A)
            # desired_symbol_brushes = [fading_brush_color] * (curr_num_points - 1) # -1 for the special last symbol

            ### Current Symbol:
            curr_desired_sizes[-1] = self.params.trajectory_path_current_position_marker_size # only current point is big
            desired_symbol_brushes.append(self.params.trajectory_path_current_position_marker_brush)

            self.ui.trajectory_curve.setData(x=curr_trajectory_rows.x.to_numpy(), y=curr_trajectory_rows.y.to_numpy(), symbolSize=list(curr_desired_sizes), symbolBrush=desired_symbol_brushes) 
        else:
            # curr_num_points == 0, path is empty
            self.ui.trajectory_curve.setData(x=None, y=None) 