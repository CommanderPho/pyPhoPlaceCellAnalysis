import pyphoplacecellanalysis
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyphoplacecellanalysis.External.pyqtgraph.opengl as gl # for 3D raster plot

import numpy as np

from pyphocorehelpers.general_helpers import OrderedMeta
from pyphocorehelpers.print_helpers import SimplePrintable, PrettyPrintable
from pyphocorehelpers.geometry_helpers import find_ranges_in_window

from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.EpochRenderingMixin import EpochRenderingMixin, RenderedEpochsItemsContainer
from pyphoplacecellanalysis.General.Model.Datasources.IntervalDatasource import IntervalsDatasource

class RenderEpochs(PrettyPrintable, SimplePrintable, metaclass=OrderedMeta):
    def __init__(self, name) -> None:
        # super(RenderEpochs, self).__init__(**kwargs)
        self.name = name
        # self.__dict__ = (self.__dict__ | kwargs)
        
    # def __init__(self, name, **kwargs) -> None:
    #     # super(VisualizationParameters, self).__init__(**kwargs)
    #     self.name = name
    #     # self.__dict__ = (self.__dict__ | kwargs)
    
    

class RenderTimeEpochMeshesMixin(EpochRenderingMixin):
    """  Extends EpochRenderingMixin for 3D OpenGL Plots such as Spike3DRaster, enabling it to render intervals in an OpenGL view
    
        It does this by overriding EpochRenderingMixin's add_rendered_intervals(...) function
        
        
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
    
    
    
    REFACTORING to EpochRenderingMixin inheritances:
    
    
    Removing:
        params.render_epochs
        plots.new_cube_objects
    
    Adding:

    
    """
    
    #############################################################################
    ################## EpochRenderingMixin Required Conformances
    #############################################################################
    @property
    def interval_rendering_plots(self):
        """ returns the list of child subplots/graphics (usually PlotItems) that participate in rendering intervals """
        # return [self.ui.main_gl_widget]
        raise NotImplementedError # MUST OVERRIDE in child
    
    @property
    def rendered_epoch_series_names(self):
        """The rendered_epoch_names property."""
        return [a_name for a_name in self.rendered_epochs.keys() if a_name != 'name']


    @property
    def has_render_epoch_meshes(self):
        """ True if epoch meshes to render have been added. """
        if self.rendered_epochs is None:
            return False
        # Find at least one plot:
        # curr_rendered_epoch_names = [a_name for a_name in self.rendered_epochs.keys() if a_name != 'name'] # done to prevent problems with dict changing size during iteration
        curr_rendered_epoch_names = self.rendered_epoch_series_names
        child_plots = self.interval_rendering_plots
        # See if we have at least one set of non-empty rendered rects
        for a_name in curr_rendered_epoch_names:
            if a_name != 'name':
                for a_plot in child_plots:
                    extant_rect_plot_item_meshes = self.rendered_epochs[a_name][a_plot]
                    if len(extant_rect_plot_item_meshes) > 0:
                        # has at least one set of non-empty rendered rects
                        return True
        # Otherwise we found no non-empty mesh rects and should return false
        return False


    @QtCore.pyqtSlot()
    def RenderTimeEpochMeshesMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        # self.params.setdefault('render_epochs', None)
        self.EpochRenderingMixin_on_init()
        

    @QtCore.pyqtSlot()
    def RenderTimeEpochMeshesMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        self.EpochRenderingMixin_on_setup()
        self.close_signal.connect(self.RenderTimeEpochMeshesMixin_on_destroy) # Connect the *_on_destroy function to the close_signal

    @QtCore.pyqtSlot()
    def RenderTimeEpochMeshesMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        self.EpochRenderingMixin_on_buildUI()
        # self.plots.setdefault('new_cube_objects', None) # need to keep: unique
        
    @QtCore.pyqtSlot()
    def RenderTimeEpochMeshesMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        # self.epoch_connection.disconnect()
        # self.epoch_connection = None
        pass


    @QtCore.pyqtSlot(float, float)
    def RenderTimeEpochMeshesMixin_on_window_update(self, new_start=None, new_end=None):
        """ called when the window is updated to update the mesh locations. """
        if self.has_render_epoch_meshes:
            self.update_all_epoch_meshes()
            
    ############### Rate-Limited SLots ###############:
    ##################################################
    ## For use with pg.SignalProxy
    # using signal proxy turns original arguments into a tuple
    @QtCore.pyqtSlot(object)
    def RenderTimeEpochMeshesMixin_on_window_update_rate_limited(self, evt):
        self.RenderTimeEpochMeshesMixin_on_window_update(*evt)


    
    ############### Public Methods ###################:
    ##################################################
    
    

    def add_rendered_intervals(self, interval_datasource, name=None, child_plots=None, debug_print=True):
        """ adds the intervals specified by the interval_datasource to the plots 
        
        Inputs: 
            interval_datasource: IntervalDatasource
            name: str, an optional but highly recommended string identifier like 'Laps'
            child_plots: an optional list of plots to add the intervals to. If None are specified, the defaults are used (defined by the implementor)
            
        Returns:
            returned_rect_items: a dictionary of tuples containing the newly created rect items and the plots they were added to.
            
            
        Usage:
            active_pbe_interval_rects_item = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_interval_datasource(interval_datasources.PBEs)
            
        """
        
        ######### DATASOURCE:
        assert isinstance(interval_datasource, IntervalsDatasource), f"interval_datasource: must be an IntervalsDatasource object but instead is of type: {type(interval_datasource)}"
        if child_plots is None:
            child_plots = self.interval_rendering_plots

        num_plot_items = len(child_plots)
        if debug_print:
            print(f'num_plot_items: {num_plot_items}')
            

        if name is None:
            print(f'WARNING: no name provided for rendered intervals. Defaulting to datasource name: "{interval_datasource.custom_datasource_name}"')
            name = interval_datasource.custom_datasource_name
        
        extant_datasource = self.interval_datasources.get(name, None)
        if extant_datasource is None:
            # no extant datasource with this name, create it:
            self.interval_datasources[name] = interval_datasource # add new datasource.

        else:
            # extant_datasource exists!
            print(f'WARNING: extant_datasource with the name ({name}) already exists. Attempting to update.')
            if extant_datasource == interval_datasource:
                # already the same datasource
                print(f'\t already the same datasource!')
                return
            else:
                # Otherwise the datasource should be replaced:
                print(f'\t replacing extant datasource.')
                # TODO: remove plots associated with replaced datasource
                self.interval_datasources[name] = interval_datasource
                        
        
        
        ######### PLOTS:
        returned_mesh_list_items = {}
        
        extant_rects_plot_items_container = self.rendered_epochs.get(name, None)
        if extant_rects_plot_items_container is not None:
            # extant plot exists!
            print(f'WARNING: extant_rects_plot_item with the name ({name}) already exists. removing.')
            assert isinstance(extant_rects_plot_items_container, RenderedEpochsItemsContainer), f"extant_rects_plot_item must be RenderedEpochsItemsContainer but type(extant_rects_plot_item): {type(extant_rects_plot_items_container)}"
            
            for a_plot in child_plots:
                if a_plot in extant_rects_plot_items_container:
                    # the plot is already here: remove and re-add it
                    # extant_rect_plot_item_meshes = extant_rects_plot_items_container[a_plot] # this is done inside self.update_epoch_meshes                                        
                    self._perform_update_epoch_meshes(name, self.interval_datasources[name].time_column_values.t_start.to_numpy(),
                                             self.interval_datasources[name].time_column_values.t_duration.to_numpy(), child_plots=None)
                        
                else:
                    # Only if child plot doesn't yet exist:
                    new_mesh_objects = self._build_epoch_meshes(self.interval_datasources[name].time_column_values.t_start.to_numpy(), self.interval_datasources[name].time_column_values.t_duration.to_numpy())
                    extant_rects_plot_items_container[a_plot] = new_mesh_objects
                    
                    ## Can't do:
                    self._perform_add_render_item(a_plot, extant_rects_plot_items_container[a_plot])
                    returned_mesh_list_items[a_plot.objectName()] = dict(plot=a_plot, rect_item=extant_rects_plot_items_container[a_plot])
                
                    
        else:
            # Need to create a new RenderedEpochsItemsContainer with the items:
            # Equiv to new_interval_rects_item:
            new_mesh_objects = self._build_epoch_meshes(self.interval_datasources[name].time_column_values.t_start.to_numpy(), self.interval_datasources[name].time_column_values.t_duration.to_numpy())
            self.rendered_epochs[name] = RenderedEpochsItemsContainer(new_mesh_objects, child_plots) # set the plot item
            for a_plot, a_rect_item_meshes in self.rendered_epochs[name].items():
                if not isinstance(a_rect_item_meshes, str):
                    if debug_print:
                        print(f'plotting item')
                        
                    self._perform_add_render_item(a_plot, a_rect_item_meshes)
                    returned_mesh_list_items[a_plot.objectName()] = dict(plot=a_plot, rect_item=a_rect_item_meshes)                                                
                                                
        return returned_mesh_list_items 



    ## TODO: IMPLEMENT            
    def add_render_epochs(self, starts_t, durations, epoch_type_name='PBE'):
        
        """ adds the render epochs to be displayed. Stores them internally"""
        self.params.render_epochs = RenderEpochs(epoch_type_name)
        self.params.render_epochs.epoch_type_name = epoch_type_name
        self.params.render_epochs.starts_t = starts_t
        self.params.render_epochs.durations = durations
        self._build_epoch_meshes(self.params.render_epochs.starts_t, self.params.render_epochs.durations)
        # self.epoch_connection.blockSignals(False) # Disabling blocking the signals so it can update
    
        
    
    ## TODO: IMPLEMENT            
    # def remove_epoch_meshes(self):
    #     for (i, aCube) in enumerate(self.plots.new_cube_objects):
    #         # aCube.setParent(None) # Set parent None is just as good as removing from self.ui.main_gl_widget I think
    #         self.ui.main_gl_widget.removeItem(aCube)
    #         aCube.deleteLater()
    #     self.plots.new_cube_objects.clear()
    #     # if not self.has_render_epoch_meshes:
    #     #     # if there are no epoch meshes left to render, block the update signal.
    #     #     self.epoch_connection.blockSignals(True)
        

    
    def update_all_epoch_meshes(self):
        """ Modifies both the position and scale of the existing self.plots.new_cube_objects
        Requires Implementors:
        
        Functions:
           ._temporal_to_spatial(...)
        Variables:
            self.n_half_cells 
            self.floor_z
            self.n_full_cell_grid
        """
        curr_rendered_epoch_names = list(self.rendered_epochs.keys()) # done to prevent problems with dict changing size during iteration
        for a_name in curr_rendered_epoch_names:
            if a_name != 'name':
                self._perform_update_epoch_meshes(a_name, self.interval_datasources[a_name].time_column_values.t_start.to_numpy(),
                                             self.interval_datasources[a_name].time_column_values.t_duration.to_numpy(), child_plots=None)

    def get_all_epoch_meshes(self):
        """ returns a flat list of all epoch meshes """
        curr_rendered_epoch_names = list(self.rendered_epochs.keys()) # done to prevent problems with dict changing size during iteration
        child_plots = self.interval_rendering_plots
        flat_mesh_list = []
        for a_name in curr_rendered_epoch_names:
            if a_name != 'name':
                for a_plot in child_plots:
                    extant_rect_plot_item_meshes = self.rendered_epochs[a_name][a_plot]
                    flat_mesh_list.extend(extant_rect_plot_item_meshes) # append these meshes with those of previous names/plots
        return flat_mesh_list
    

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
        """ 
            This is why the rectangle positions are updated when the window is scrolled (on update_epoch_meshes)
            
        Usage:
            epoch_window_relative_start_x_positions, epoch_spatial_durations = self._temporal_to_spatial()
        """
        return DataSeriesToSpatial.temporal_to_spatial_transform_computation(epoch_start_times, epoch_durations, self.spikes_window.active_window_start_time, self.spikes_window.active_window_end_time, self.temporal_axis_length, center_mode='zero_centered')
        

    def _build_epoch_meshes(self, starts_t, durations):
        """ builds, but does not add, the list of gl.GLMeshItem to be added to the OpenGL viewport. There is one per interval.
        
        Input:
            a_plot: self.ui.main_gl_widget
            
        Usage:  
            curr_sess.pbe.durations
        
        """        
        centers_t = starts_t + (durations / 2.0)
        x_centers, duration_spatial_widths = self._temporal_to_spatial(centers_t, durations) # actually compute the centers of each epoch rect, not the start
                
        new_mesh_objects = []
        # self.plots.new_cube_objects = []
        for i in np.arange(len(x_centers)):
            curr_md = RenderTimeEpochMeshesMixin._build_cube_mesh_data()
            curr_cube = gl.GLMeshItem(meshdata=curr_md, smooth=True, color=(1, 0, 0, 0.2), shader='balloon', glOptions='additive') # , drawEdges=True, edgeColor=(0, 0, 0, 1)
            curr_cube.translate(x_centers[i], -self.n_half_cells, self.floor_z)
            curr_cube.scale(duration_spatial_widths[i], self.n_full_cell_grid, 0.25)
            # curr_cube.setParentItem(self.ui.parent_epoch_container_item)
            new_mesh_objects.append(curr_cube)

        return new_mesh_objects
    
          
    def _perform_update_epoch_meshes(self, name, starts_t, durations, child_plots=None):
        """ Modifies both the position and scale of the existing self.rendered_epochs
        Requires Implementors:
        
        Functions:
           ._temporal_to_spatial(...)
        Variables:
            self.n_half_cells 
            self.floor_z
            self.n_full_cell_grid
        """
        if child_plots is None:
            child_plots = self.interval_rendering_plots
        
        centers_t = starts_t + (durations / 2.0)
        x_shifted_centers, duration_spatial_widths = self._temporal_to_spatial(centers_t, durations) # actually compute the centers of each epoch rect, not the start
        
        for a_plot in child_plots:
            assert a_plot in self.rendered_epochs[name], f"a_plot must be in self.rendered_epochs[name]"
            # the plot is already here: remove and re-add it
            extant_rect_plot_item_meshes = self.rendered_epochs[name][a_plot]
            # Update the meshes for this item:
            for (i, aCube) in enumerate(extant_rect_plot_item_meshes):
                aCube.resetTransform()
                aCube.translate(x_shifted_centers[i], -self.n_half_cells, self.floor_z)
                aCube.scale(duration_spatial_widths[i], self.n_full_cell_grid, 0.25)
           



    
    ######################################################
    # EpochRenderingMixin Convencince methods:
    #####################################################
    def _perform_add_render_item(self, a_plot, a_render_item):
        """Performs the operation of adding the render item from the plot specified

        Args:
            a_render_item (list): a list of a_rect_item_meshes
            a_plot (_type_): _description_
        """
        for curr_cube in a_render_item:
            a_plot.addItem(curr_cube) # add directly
        
        
    def _perform_remove_render_item(self, a_plot, a_render_item):
        """Performs the operation of removing the render item from the plot specified

        Args:
            a_render_item (IntervalRectsItem): _description_
            a_plot (PlotItem): _description_
        """
        for curr_cube in a_render_item:
            a_plot.removeItem(curr_cube) # add directly

    #######################################################################################################################################
    
