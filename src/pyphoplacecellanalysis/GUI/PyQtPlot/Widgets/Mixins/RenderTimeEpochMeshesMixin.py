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
        
        Requires Implementors have:
        
            Functions:
                ._temporal_to_spatial(...)
                
            Signals:
                .window_scrolled
                .close_signal
                
            Variables:
                .n_half_cells 
                .floor_z
                .n_full_cell_grid
                .ui.main_gl_widget
                .plots
        
        
        Provides:
            .params.render_epochs
            .plots.new_cube_objects
            .epoch_connection
            
        
            .add_render_epochs(starts_t, durations, epoch_type_name='PBE')
            .update_epoch_meshes(starts_t, durations)
            
            @QtCore.pyqtSlot()
            def RenderTimeEpochMeshesMixin_on_update_window(self)
    
    """
    
    @property
    def has_render_epoch_meshes(self):
        """ True if epoch meshes to render have been added. """
        if self.params.render_epochs is None:
            return False        
        if self.plots.new_cube_objects is None:
            return False
        else:
            return (len(self.plots.new_cube_objects) > 0)
        

    @QtCore.pyqtSlot()
    def RenderTimeEpochMeshesMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        self.params.setdefault('render_epochs', None)        

    @QtCore.pyqtSlot()
    def RenderTimeEpochMeshesMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        self.close_signal.connect(self.RenderTimeEpochMeshesMixin_on_destroy) # Connect the *_on_destroy function to the close_signal
        # self.epoch_connection = pg.SignalProxy(self.window_scrolled, delay=0.2, rateLimit=60, slot=self.RenderTimeEpochMeshesMixin_on_window_update_rate_limited)
        # self.epoch_connection.blockSignals(True) # block signals by default so it isn't calling update needlessly

    @QtCore.pyqtSlot()
    def RenderTimeEpochMeshesMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        self.plots.setdefault('new_cube_objects', None)

    @QtCore.pyqtSlot()
    def RenderTimeEpochMeshesMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        # self.epoch_connection.disconnect()
        # self.epoch_connection = None
        pass

    @QtCore.pyqtSlot(float, float)
    def RenderTimeEpochMeshesMixin_on_window_update(self, new_start=None, new_end=None):
        """ called when the window is updated to update the mesh locations. """
        if self.has_render_epoch_meshes is not None:
            self.update_epoch_meshes(self.params.render_epochs.starts_t, self.params.render_epochs.durations)

    ############### Rate-Limited SLots ###############:
    ##################################################
    ## For use with pg.SignalProxy
    # using signal proxy turns original arguments into a tuple
    @QtCore.pyqtSlot(object)
    def RenderTimeEpochMeshesMixin_on_window_update_rate_limited(self, evt):
        self.RenderTimeEpochMeshesMixin_on_window_update(*evt)

    ############### Internal Methods #################:
    ##################################################
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
        centers_t = starts_t + (durations / 2.0)
        x_centers, duration_spatial_widths = self._temporal_to_spatial(centers_t, durations) # actually compute the centers of each epoch rect, not the start
                
        self.plots.new_cube_objects = []
        for i in np.arange(len(x_centers)):
            curr_md = RenderTimeEpochMeshesMixin._build_cube_mesh_data()
            curr_cube = gl.GLMeshItem(meshdata=curr_md, smooth=True, color=(1, 0, 0, 0.2), shader='balloon', glOptions='additive') # , drawEdges=True, edgeColor=(0, 0, 0, 1)
            curr_cube.translate(x_centers[i], -self.n_half_cells, self.floor_z)
            curr_cube.scale(duration_spatial_widths[i], self.n_full_cell_grid, 0.25)
            # curr_cube.setParentItem(self.ui.parent_epoch_container_item)
            self.ui.main_gl_widget.addItem(curr_cube) # add directly
            self.plots.new_cube_objects.append(curr_cube)

    ############### Public Methods ###################:
    ##################################################
    def add_render_epochs(self, starts_t, durations, epoch_type_name='PBE'):
        """ adds the render epochs to be displayed. Stores them internally"""
        self.params.render_epochs = RenderEpochs(epoch_type_name)
        self.params.render_epochs.epoch_type_name = epoch_type_name
        self.params.render_epochs.starts_t = starts_t
        self.params.render_epochs.durations = durations
        self._build_epoch_meshes(self.params.render_epochs.starts_t, self.params.render_epochs.durations)
        # self.epoch_connection.blockSignals(False) # Disabling blocking the signals so it can update
        
    def update_epoch_meshes(self, starts_t, durations):
        """ Modifies both the position and scale of the existing self.plots.new_cube_objects
        Requires Implementors:
        
        Functions:
           ._temporal_to_spatial(...)
        Variables:
            self.n_half_cells 
            self.floor_z
            self.n_full_cell_grid
        """
        centers_t = starts_t + (durations / 2.0)
        x_shifted_centers, duration_spatial_widths = self._temporal_to_spatial(centers_t, durations) # actually compute the centers of each epoch rect, not the start
        
        for (i, aCube) in enumerate(self.plots.new_cube_objects):
            aCube.resetTransform()
            aCube.translate(x_shifted_centers[i], -self.n_half_cells, self.floor_z)
            aCube.scale(duration_spatial_widths[i], self.n_full_cell_grid, 0.25)
           
            
    def remove_epoch_meshes(self):
        for (i, aCube) in enumerate(self.plots.new_cube_objects):
            aCube.setParent(None) # Set parent None is just as good as removing from self.ui.main_gl_widget I think
            aCube.deleteLater()
        self.plots.new_cube_objects.clear()
        # if not self.has_render_epoch_meshes:
        #     # if there are no epoch meshes left to render, block the update signal.
        #     self.epoch_connection.blockSignals(True)
        

