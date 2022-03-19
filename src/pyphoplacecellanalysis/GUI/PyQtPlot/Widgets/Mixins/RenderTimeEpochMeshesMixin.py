import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np


from pyphocorehelpers.geometry_helpers import find_ranges_in_window


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


    def _build_epoch_meshes(self, starts_t, durations):
        """ 
        # find center of pbe periods (as this is where the mesh will be positioned.
        # pbe_half_durations = curr_sess.pbe.durations / 2.0
        # pbe_t_centers = curr_sess.pbe.starts + pbe_half_durations

        Usage:  
            curr_sess.pbe.durations
        
        """
        
        half_durations = durations / 2.0
        t_centers = starts_t + half_durations
        
        
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
        for i in np.arange(len(t_centers)):
            curr_md = RenderTimeEpochMeshesMixin._build_cube_mesh_data()
            curr_cube = gl.GLMeshItem(meshdata=curr_md, smooth=True, color=(1, 0, 0, 0.2), shader='balloon', glOptions='additive') # , drawEdges=True, edgeColor=(0, 0, 0, 1)
            # new_cube = gl.GLMeshItem(vertexes=vertexes, faces=faces, faceColors=colors, drawEdges=True, edgeColor=(0, 0, 0, 1))
            curr_cube.translate(t_centers[i], -self.n_half_cells, self.z_floor)
            curr_cube.scale(durations[i], self.n_full_cell_grid, 0.25)
            # curr_cube.setParentItem(self.ui.parent_epoch_container_item)
            self.ui.main_gl_widget.addItem(curr_cube) # add directly
            # self.ui.parent_epoch_container_item.addItem(curr_cube)
            self.ui.new_cube_objects.append(curr_cube)


        # self.ui.main_gl_widget.addItem(self.ui.parent_epoch_container_item)
        
    def update_epoch_meshes(self, starts_t, durations):
        half_durations = durations / 2.0
        t_centers = starts_t + half_durations
        # t_shifted_centers = t_centers - self.spikes_window.active_time_window[0] # offset by the start of the current window
        t_shifted_centers = t_centers
        for (i, aCube) in enumerate(self.ui.new_cube_objects):
            # aCube.setPos(t_centers[i], self.n_half_cells, 0)
            aCube.resetTransform()
            aCube.translate(t_shifted_centers[i], -self.n_half_cells, self.z_floor)
            aCube.scale(durations[i], self.n_full_cell_grid, 0.25)
            # aCube.setData(pos=(t_centers[i], self.n_half_cells, 0))
            # aCube.setParent(None)
            # aCube.deleteLater()