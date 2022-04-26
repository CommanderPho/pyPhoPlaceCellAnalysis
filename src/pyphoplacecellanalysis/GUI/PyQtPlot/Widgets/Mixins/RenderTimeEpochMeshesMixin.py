import pyphoplacecellanalysis
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyphoplacecellanalysis.External.pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np

from pyphocorehelpers.general_helpers import OrderedMeta
from pyphocorehelpers.print_helpers import SimplePrintable, PrettyPrintable
from pyphocorehelpers.geometry_helpers import find_ranges_in_window

from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial



class RenderEpochs(PrettyPrintable, SimplePrintable, metaclass=OrderedMeta):
    def __init__(self, name) -> None:
        # super(RenderEpochs, self).__init__(**kwargs)
        self.name = name
        # self.__dict__ = (self.__dict__ | kwargs)
        
    # def __init__(self, name, **kwargs) -> None:
    #     # super(VisualizationParameters, self).__init__(**kwargs)
    #     self.name = name
    #     # self.__dict__ = (self.__dict__ | kwargs)
    
    

class RenderTimeEpochMeshesMixin:
    """ 
        RenderTimeEpochMeshes
        
        Renders rectangular meshes to represent periods of time in the current 3D plot
        
        
    """
    
    @classmethod
    def _build_cube_mesh_data(cls):
        vertexes = np.array([[1, 0, 0], #0
                            [0, 0, 0], #1
                            [0, 1, 0], #2
                            [0, 0, 1], #3
                            [1, 1, 0], #4
                            [1, 1, 1], #5
                            [0, 1, 1], #6
                            [1, 0, 1]])#7
        faces = np.array([[1,0,7], [1,3,7],
                        [1,2,4], [1,0,4],
                        [1,2,6], [1,3,6],
                        [0,4,5], [0,7,5],
                        [2,4,5], [2,6,5],
                        [3,6,5], [3,7,5]])
        colors = np.array([[1,0,0,1] for i in range(12)])
        md = gl.MeshData(vertexes=vertexes, faces=faces, edges=None, vertexColors=None, faceColors=colors)
        return md


    def add_render_epochs(self, starts_t, durations, epoch_type_name='PBE'):
        """ adds the render epochs to be displayed. Stores them internally"""
        self.params.render_epochs = RenderEpochs(epoch_type_name)
        self.params.render_epochs.epoch_type_name = epoch_type_name
        self.params.render_epochs.starts_t = starts_t
        self.params.render_epochs.durations = durations
        self._build_epoch_meshes(self.params.render_epochs.starts_t, self.params.render_epochs.durations)
        
        

    # def _temporal_to_spatial(self, starts_t):
    #     """ currently this constrains all epochs outside the active window to be aligned with the endpoints of the window, meaning there are a ton of stacked 0-width windows rendered at both endpoints since only a few epochs are visible at a time."""
    #     return DataSeriesToSpatial.temporal_to_spatial_map(starts_t, self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time, self.temporal_axis_length, center_mode='zero_centered')
        
    def _temporal_to_spatial(self, epoch_start_times, epoch_durations):
        """ epoch_window_relative_start_x_positions, epoch_spatial_durations = self._temporal_to_spatial()
        
        """
        return DataSeriesToSpatial.temporal_to_spatial_transform_computation(epoch_start_times, epoch_durations, self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time, self.temporal_axis_length, center_mode='zero_centered')
        

    def _build_epoch_meshes(self, starts_t, durations):
        """ 
        # find center of pbe periods (as this is where the mesh will be positioned.
        # pbe_half_durations = curr_sess.pbe.durations / 2.0
        # pbe_t_centers = curr_sess.pbe.starts + pbe_half_durations

        Usage:  
            curr_sess.pbe.durations
        
        """
        # stops_t = starts_t + durations
        # # Compute spatial positions/durations:
        # starts_x = self._temporal_to_spatial(starts_t)
        # stops_x = self._temporal_to_spatial(stops_t)
        # durations_spatial_widths = stops_x - starts_x
        # half_durations_spatial_widths = durations_spatial_widths / 2.0
        # x_centers = starts_x + half_durations_spatial_widths
        
        
        centers_t = starts_t + (durations / 2.0)
        x_centers, duration_spatial_widths = self._temporal_to_spatial(centers_t, durations) # actually compute the centers of each epoch rect, not the start
        
        
        
        # The transform needs to be done here to match the temporal_scale_Factor:
        # curr_x = np.interp(curr_spike_t, (self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time), (-self.half_temporal_axis_length, +self.half_temporal_axis_length))
        
        # pg.gl.GLViewWidget()
        # self.ui.parent_epoch_container_item = gl.GLGraphicsItem.GLGraphicsItem()
        # self.ui.parent_epoch_container_item = pg.GraphicsObject()
        # self.ui.parent_epoch_container_item.setObjectName('parent_epoch_container')
        # # self.ui.parent_epoch_container_item.translate(0, 0, 0)
        # # self.ui.parent_epoch_container_item.scale(1, 1, 1)
        # self.ui.main_gl_widget.addItem(self.ui.parent_epoch_container_item)
        # gl.GLBoxItem()
                
        self.ui.new_cube_objects = []
        for i in np.arange(len(x_centers)):
            curr_md = RenderTimeEpochMeshesMixin._build_cube_mesh_data()
            curr_cube = gl.GLMeshItem(meshdata=curr_md, smooth=True, color=(1, 0, 0, 0.2), shader='balloon', glOptions='additive') # , drawEdges=True, edgeColor=(0, 0, 0, 1)
            # new_cube = gl.GLMeshItem(vertexes=vertexes, faces=faces, faceColors=colors, drawEdges=True, edgeColor=(0, 0, 0, 1))
            curr_cube.translate(x_centers[i], -self.n_half_cells, self.z_floor)
            curr_cube.scale(duration_spatial_widths[i], self.n_full_cell_grid, 0.25)
            # curr_cube.setParentItem(self.ui.parent_epoch_container_item)
            self.ui.main_gl_widget.addItem(curr_cube) # add directly
            # self.ui.parent_epoch_container_item.addItem(curr_cube)
            self.ui.new_cube_objects.append(curr_cube)


        # self.ui.main_gl_widget.addItem(self.ui.parent_epoch_container_item)
        
        
    def _remove_epoch_meshes(self):
        for (i, aCube) in enumerate(self.ui.new_cube_objects):
            aCube.setParent(None)
            aCube.deleteLater()
        # self.ui.main_gl_widget.
        self.ui.new_cube_objects.clear()
        
        


        
    def update_epoch_meshes(self, starts_t, durations):
        # stops_t = starts_t + durations
        # # Compute spatial positions/durations:
        # starts_x = self._temporal_to_spatial(starts_t)
        # stops_x = self._temporal_to_spatial(stops_t)
        # duration_spatial_widths = stops_x - starts_x
        # half_durations_spatial_widths = duration_spatial_widths / 2.0
        # x_centers = starts_x + half_durations_spatial_widths
        
        centers_t = starts_t + (durations / 2.0)
        x_shifted_centers, duration_spatial_widths = self._temporal_to_spatial(centers_t, durations) # actually compute the centers of each epoch rect, not the start
        
        # t_shifted_centers = t_centers - self.spikes_window.active_time_window[0] # offset by the start of the current window
        # x_shifted_centers = x_centers
        for (i, aCube) in enumerate(self.ui.new_cube_objects):
            # aCube.setPos(x_centers[i], self.n_half_cells, 0)
            aCube.resetTransform()
            aCube.translate(x_shifted_centers[i], -self.n_half_cells, self.z_floor)
            aCube.scale(duration_spatial_widths[i], self.n_full_cell_grid, 0.25)
            # aCube.setData(pos=(x_centers[i], self.n_half_cells, 0))
            # aCube.setParent(None)
            # aCube.deleteLater()
            
    @QtCore.pyqtSlot()
    def RenderTimeEpochMeshesMixin_on_update_window(self):
        """ called when the window is updated to update the mesh locations. """
        if self.params.render_epochs is not None:
            self.update_epoch_meshes(self.params.render_epochs.starts_t, self.params.render_epochs.durations)
            
        