from copy import copy, deepcopy
from neuropy.core import Epoch
import numpy as np
import pandas as pd

from qtpy import QtCore

from pyphocorehelpers.print_helpers import SimplePrintable, PrettyPrintable, iPythonKeyCompletingMixin
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

from pyphocorehelpers.DataStructure.general_parameter_containers import DebugHelper, VisualizationParameters, RenderPlots, RenderPlotsData
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.gui.Qt.connections_container import ConnectionsContainer


from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem import IntervalRectsItem, RectangleRenderTupleHelpers
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Render2DEventRectanglesHelper import Render2DEventRectanglesHelper

from pyphoplacecellanalysis.General.Model.Datasources.IntervalDatasource import IntervalsDatasource
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs # required for `update_interval_visualization_properties(...)`


class RenderedEpochsItemsContainer(iPythonKeyCompletingMixin, DynamicParameters):
    """ Wraps a list of plots and their rendered_rects_item for a given datasource/name
    
        Note that the plots are only given by self.dynamically_added_attributes since the 'name' key exists.
    
    """
    def __init__(self, rendered_rects_item, target_plots_list):
        super(RenderedEpochsItemsContainer, self).__init__()
        if len(target_plots_list) == 1:
            a_plot = target_plots_list[0]
            self[a_plot] = rendered_rects_item # no conflict, so can just return the original rendered_rects_item

        else:
            for a_plot in target_plots_list:
                # make an independent copy of the rendered_rects_item for each plot
                independent_data_copy = RectangleRenderTupleHelpers.copy_data(rendered_rects_item.data)
                self[a_plot] = IntervalRectsItem(data=independent_data_copy)

   
class EpochRenderingMixin:
    """ Implementors render Epochs/Intervals
    
    Requires:
        self.plots
        self.plots_data
        
    Provides:
        self.plots_data['interval_datasources']: RenderPlotsData
        self.plots.rendered_epochs: RenderPlots
        self.ui
        self.ui.connections
    
    Known Conformances:
        RasterPlot2D: to render laps, PBEs, and more on the 2D plots

    Usage:
        ## Build a PBEs datasource:
        laps_interval_datasource = Specific2DRenderTimeEpochsHelper.build_Laps_render_time_epochs_datasource(curr_sess=sess, series_vertical_offset=42.0, series_height=1.0)
        new_PBEs_interval_datasource = Specific2DRenderTimeEpochsHelper.build_PBEs_render_time_epochs_datasource(curr_sess=sess, series_vertical_offset=43.0, series_height=1.0) # new_PBEs_interval_datasource
        
        ## General Adding:
            active_2d_plot.add_rendered_intervals(new_PBEs_interval_datasource, name='PBEs', child_plots=[background_static_scroll_plot_widget, main_plot_widget], debug_print=True)
            active_2d_plot.add_rendered_intervals(laps_interval_datasource, name='Laps', child_plots=[background_static_scroll_plot_widget, main_plot_widget], debug_print=True)
        
        ## Selectively Adding:
            # Tests adding PBEs to just a single child plot (main_plot_widget):
            active_2d_plot.add_rendered_intervals(new_PBEs_interval_datasource, name='PBEs', child_plots=[main_plot_widget], debug_print=True)

        ## Selectively Removing:
            active_2d_plot.remove_rendered_intervals(name='PBEs', child_plots_removal_list=[main_plot_widget]) # Tests removing a single series from a single plot (main_plot_widget)
            active_2d_plot.remove_rendered_intervals(name='PBEs') # Tests removing a single series ('PBEs') from all plots it's on
            
        ## Clearing:
            active_2d_plot.clear_all_rendered_intervals()
        
    """


    sigRenderedIntervalsListChanged = QtCore.Signal(object) # signal emitted whenever the list of rendered intervals changed (add/remove)



    @property
    def interval_rendering_plots(self):
        """ returns the list of child subplots/graphics (usually PlotItems) that participate in rendering intervals """
        raise NotImplementedError # MUST OVERRIDE in child
        # return [self.plots.background_static_scroll_window_plot, self.plots.main_plot_widget] # for spike_raster_plt_2d
    
    @property
    def interval_datasources(self):
        """The interval_datasources property. A RenderPlotsData object """
        return self.plots_data['interval_datasources']


    @property
    def interval_datasource_names(self):
        """The interval_datasources property."""
        return list(self.interval_datasources.dynamically_added_attributes) # ['CustomPBEs', 'PBEs', 'Ripples', 'Laps', 'Replays', 'SessionEpochs']

    @property
    def interval_datasource_updating_connections(self):
        """The interval_datasource_updating_connections property. A ConnectionsContainer object """
        return self.ui.connections
    
    @property
    def rendered_epochs(self):
        """The interval_datasources property."""
        return self.plots.rendered_epochs
    
    @property
    def rendered_epoch_series_names(self):
        """The rendered_epoch_names property."""
        return [a_name for a_name in self.rendered_epochs.keys() if ((a_name != 'name') and (a_name != 'context'))]


    #######################################################################################################################################
    
    @QtCore.Slot()
    def EpochRenderingMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        self.plots_data['interval_datasources'] = RenderPlotsData('EpochRenderingMixin')
        
        

        # self.plots_data['interval_datasource_updating_connections'] = ConnectionsContainer('EpochRenderingMixin')
    
    @QtCore.Slot()
    def EpochRenderingMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        self.plots.rendered_epochs = RenderPlots('EpochRenderingMixin') # the container to hold the time rectangles

    @QtCore.Slot()
    def EpochRenderingMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        # Adds the self.ui and self.ui.connections if they don't exist
        if not hasattr(self, 'ui'):
            # if the window has no .ui property, create one:
            setattr(self, 'ui', PhoUIContainer())
            
        if isinstance(self.ui, DynamicParameters):            
            # Need this workaround because hasattr fails for DynamicParameters/PhoUIContainer right now:
            self.ui.setdefault('connections', ConnectionsContainer())
        else:
            if not hasattr(self.ui, 'connections'):
                self.ui.connections = ConnectionsContainer()

    @QtCore.Slot()
    def EpochRenderingMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        # TODO: REGISTER AND IMPLEMENT
        raise NotImplementedError
        pass

    @QtCore.Slot(float, float)
    def EpochRenderingMixin_on_window_update(self, new_start=None, new_end=None):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc. """
        raise NotImplementedError
        pass

    ############### Rate-Limited SLots ###############:
    ##################################################
    ## For use with pg.SignalProxy
    # using signal proxy turns original arguments into a tuple
    @QtCore.Slot(object)
    def EpochRenderingMixin_on_window_update_rate_limited(self, evt):
        self.EpochRenderingMixin_on_window_update(*evt)
        

    
    #######################################################################################################################################
    
    @QtCore.Slot(object)
    def EpochRenderingMixin_on_interval_datasource_changed(self, datasource):
        """ emit our own custom signal when the general datasource update method returns """
        # print(f'datasource: {datasource.custom_datasource_name}')
        self.add_rendered_intervals(datasource, name=datasource.custom_datasource_name, debug_print=False) # updates the rendered intervals on the change
        
        
    def add_rendered_intervals(self, interval_datasource, name=None, child_plots=None, debug_print=False):
        """ adds or updates the intervals specified by the interval_datasource to the plots 
        
        Inputs: 
            interval_datasource: IntervalDatasource
            name: str, an optional but highly recommended string identifier like 'Laps'
            child_plots: an optional list of plots to add the intervals to. If None are specified, the defaults are used (defined by the implementor)
            
        Returns:
            returned_rect_items: a dictionary of tuples containing the newly created rect items and the plots they were added to.
            
            
        Example:
            active_pbe_interval_rects_item = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_interval_datasource(interval_datasources.PBEs)
            
        Usages:
            Used in the EpochRenderingMixin Convencince methods in Spike2DRaster:
                .add_laps_intervals(...)
                .add_PBEs_intervals(...)
        
        """
        assert isinstance(interval_datasource, IntervalsDatasource), f"interval_datasource: must be an IntervalsDatasource object but instead is of type: {type(interval_datasource)}"
        if name is None:
            print(f'WARNING: no name provided for rendered intervals. Defaulting to datasource name: "{interval_datasource.custom_datasource_name}"')
            name = interval_datasource.custom_datasource_name
            
        # Update the custom datasource name with the provided name
        interval_datasource.custom_datasource_name = name
        
        rendered_intervals_list_did_change = False
        extant_datasource = self.interval_datasources.get(name, None)
        if extant_datasource is None:
            # no extant datasource with this name, create it:
            self.interval_datasources[name] = interval_datasource # add new datasource.
            # Connect the source_data_changed_signal to handle changes to the datasource:
            
            self.interval_datasources[name].source_data_changed_signal.connect(self.EpochRenderingMixin_on_interval_datasource_changed)
            rendered_intervals_list_did_change = True

        else:
            # extant_datasource exists!
            if debug_print:
                print(f'WARNING: extant_datasource with the name ({name}) already exists. Attempting to update.')
            if extant_datasource == interval_datasource:
                # already the same datasource
                if debug_print:
                    print(f'\t already the same datasource. Continuing to try and update.')
            else:
                # Otherwise the datasource should be replaced:
                if debug_print:
                    print(f'\t replacing extant datasource.')
                # TODO: remove plots associated with replaced datasource? DONE: as long as name doesn't change, this is done below
                # TODO: disconnect the previous datasource from the update signal?
                self.interval_datasources[name] = interval_datasource
                # Connect the source_data_changed_signal to handle changes to the datasource:
                self.interval_datasources[name].source_data_changed_signal.connect(self.EpochRenderingMixin_on_interval_datasource_changed)
                        
        
        returned_rect_items = {}
        
        # Build the rendered interval item:
        new_interval_rects_item = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_interval_datasource(interval_datasource)
        new_interval_rects_item.setToolTip(name) # The tooltip is set generically here to 'PBEs', 'Replays' or whatever the dataseries name is
        
        ######### PLOTS:
        if child_plots is None:
            child_plots = self.interval_rendering_plots
        num_plot_items = len(child_plots)
        if debug_print:
            print(f'num_plot_items: {num_plot_items}')
        
        extant_rects_plot_items_container = self.rendered_epochs.get(name, None)
        if extant_rects_plot_items_container is not None:
            # extant plot exists!
            if debug_print:
                print(f'WARNING: extant_rects_plot_item with the name ({name}) already exists. removing.')
            assert isinstance(extant_rects_plot_items_container, RenderedEpochsItemsContainer), f"extant_rects_plot_item must be RenderedEpochsItemsContainer but type(extant_rects_plot_item): {type(extant_rects_plot_items_container)}"
            
            for a_plot in child_plots:
                if a_plot in extant_rects_plot_items_container:
                    # the plot is already here: remove and re-add it
                    extant_rect_plot_item = extant_rects_plot_items_container[a_plot]
                    self._perform_remove_render_item(a_plot, extant_rect_plot_item)
                                        
                    # TODO: update the item's data instead of replacing it
                    # # add the new one:
                    # extant_rects_plot_items_container[a_plot] = new_interval_rects_item.copy()
                    # a_plot.addItem(extant_rects_plot_items_container[a_plot])
                
                independent_data_copy = RectangleRenderTupleHelpers.copy_data(new_interval_rects_item.data)
                extant_rects_plot_items_container[a_plot] = IntervalRectsItem(data=independent_data_copy)
                extant_rects_plot_items_container[a_plot].setToolTip(name)
                self._perform_add_render_item(a_plot, extant_rects_plot_items_container[a_plot])
                returned_rect_items[a_plot.objectName()] = dict(plot=a_plot, rect_item=extant_rects_plot_items_container[a_plot])
                # Adjust the bounds to fit any children:
                EpochRenderingMixin.compute_bounds_adjustment_for_rect_item(a_plot, extant_rects_plot_items_container[a_plot])
                
                    
        else:
            # Need to create a new RenderedEpochsItemsContainer with the items:
            self.rendered_epochs[name] = RenderedEpochsItemsContainer(new_interval_rects_item, child_plots) # set the plot item
            for a_plot, a_rect_item in self.rendered_epochs[name].items():
                if not isinstance(a_rect_item, str):
                    if debug_print:
                        print(f'plotting item')
                    self._perform_remove_render_item(a_plot, a_rect_item)
                    self._perform_add_render_item(a_plot, a_rect_item)
                    returned_rect_items[a_plot.objectName()] = dict(plot=a_plot, rect_item=a_rect_item)
                    
                    # Adjust the bounds to fit any children:
                    EpochRenderingMixin.compute_bounds_adjustment_for_rect_item(a_plot, a_rect_item)


        if rendered_intervals_list_did_change:
            self.sigRenderedIntervalsListChanged.emit(self) # Emit the intervals list changed signal when a truely new item is added

        return returned_rect_items 

    def remove_rendered_intervals(self, name, child_plots_removal_list=None, debug_print=False):
        """ removes the intervals specified by the interval_datasource to the plots

        Inputs:
            name: the name of the rendered_repochs to remove.
            child_plots_removal_list: is not-None, a list of child plots can be specified and rects will only be removed from those plots.
        
        Returns:
            a list of removed items
        """
        extant_rects_plot_item = self.rendered_epochs[name]
        items_to_remove_from_rendered_epochs = []
        for a_plot, a_rect_item in extant_rects_plot_item.items():
            if not isinstance(a_plot, str):
                if child_plots_removal_list is not None:
                    if a_plot in child_plots_removal_list:
                        # only remove if the plot is in the child plots:
                        self._perform_remove_render_item(a_plot, a_rect_item)
                        items_to_remove_from_rendered_epochs.append(a_plot)
                    else:
                        pass # continue
                else:
                    # otherwise remove all
                    self._perform_remove_render_item(a_plot, a_rect_item)
                    items_to_remove_from_rendered_epochs.append(a_plot)
                
        ## remove the items from the list:
        for a_key_to_remove in items_to_remove_from_rendered_epochs:
            del extant_rects_plot_item[a_key_to_remove] # remove the key from the RenderedEpochsItemsContainer
        
        if len(self.rendered_epochs[name]) == 0:
            # if the item is now empty, remove it and its and paired datasource
            if debug_print:
                print(f'self.rendered_epochs[{name}] now empty. Removing it and its datasource...')
            del self.rendered_epochs[name]
            del self.interval_datasources[name]
            self.sigRenderedIntervalsListChanged.emit(self) # Emit the intervals list changed signal when the item is removed
    
        return items_to_remove_from_rendered_epochs

    def clear_all_rendered_intervals(self, child_plots_removal_list=None, debug_print=False):
        """ removes all rendered rects - a batch version of removed_rendered_intervals(...) """
        # curr_rendered_epoch_names = list(self.rendered_epochs.keys()) # done to prevent problems with dict changing size during iteration
        curr_rendered_epoch_names = self.rendered_epoch_series_names
        # the `self.rendered_epochs` is of type RenderPlots, and it has a 'name' and 'context' property that don't correspond to real outputs
        for a_name in curr_rendered_epoch_names:
            if (a_name != 'name') and (a_name != 'context'):
                if debug_print:
                    print(f'removing {a_name}...')
                self.remove_rendered_intervals(a_name, child_plots_removal_list=child_plots_removal_list, debug_print=debug_print)
      
      
    def list_all_rendered_intervals(self, debug_print=True):
        """ Returns a dictionary containing the hierarchy of all the members. Can optionally also print. 
        
        Example:
            interval_info = active_2d_plot.list_all_rendered_intervals()
            >>> CONSOLE OUTPUT >>>        
                rendered_epoch_names: ['PBEs', 'Laps']
                    name: PBEs - 0 plots:
                    name: Laps - 2 plots:
                        background_static_scroll_window_plot: plot[42 intervals]
                        main_plot_widget: plot[42 intervals]
                out_dict: {'PBEs': {}, 'Laps': {'background_static_scroll_window_plot': 'plot[42 intervals]', 'main_plot_widget': 'plot[42 intervals]'}}
            <<<
        
            interval_info
                {'PBEs': {},
                'Laps': {'background_static_scroll_window_plot': 'plot[42 intervals]',
                'main_plot_widget': 'plot[42 intervals]'}}
        """
        out_dict = {}
        rendered_epoch_names = self.interval_datasource_names
        if debug_print:
            print(f'rendered_epoch_names: {rendered_epoch_names}')
        for a_name in rendered_epoch_names:
            out_dict[a_name] = {}
            a_render_container = self.rendered_epochs[a_name]
            render_container_items = {key:value for key, value in a_render_container.items() if (not isinstance(key, str))}
            if debug_print:
                print(f'\tname: {a_name} - {len(render_container_items)} plots:')
                # print(f'\t\ta_render_container: {a_render_container}')
            curr_plots_dict = {}
            
            for a_plot, a_rect_item in render_container_items.items():
                if isinstance(a_plot, str):
                    ## This is still happening due to the '__class__' item!
                    print(f'WARNING: there was an item in a_render_container of type string: (a_plot: {a_plot} <{type(a_plot)}>, a_rect_item: {type(a_rect_item)}')
                    # pass 
                else:
                    if isinstance(a_rect_item, IntervalRectsItem):
                        num_intervals = len(a_rect_item.data)
                    else:
                        num_intervals = len(a_rect_item) # for 3D plots, for example, we have a list of meshes which we will use len(...) to get the number of
                        
                    if debug_print:
                        print(f'\t\t{a_plot.objectName()}: plot[{num_intervals} intervals]')
                    curr_plots_dict[a_plot.objectName()] = f'plot[{num_intervals} intervals]'
            out_dict[a_name] = curr_plots_dict
            
        if debug_print:
            print(f'out_dict: {out_dict}')
    
        return out_dict

    def update_rendered_intervals_visualization_properties(self, update_dict):
        """ Updates the interval datasources from the provided update_dict

        Args:
            update_dict (_type_): _description_

        Usage:

            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.EpochRenderingMixin import EpochRenderingMixin

            rendered_interval_keys = ['_', 'SessionEpochs', 'Laps', '_', 'PBEs', 'Ripples', 'Replays'] # '_' indicates a vertical spacer

            desired_interval_height_ratios = [2.0, 2.0, 1.0, 0.1, 1.0, 1.0, 1.0] # ratio of heights to each interval
            required_vertical_offsets, required_interval_heights = EpochRenderingMixin.build_stacked_epoch_layout(desired_interval_height_ratios, epoch_render_stack_height=20.0, interval_stack_location='below')
            stacked_epoch_layout_dict = {interval_key:dict(y_location=y_location, height=height) for interval_key, y_location, height in zip(rendered_interval_keys, required_vertical_offsets, required_interval_heights)}

        """
        ## Inline Concise: Position Replays, PBEs, and Ripples all below the scatter:
        for interval_key, interval_update_kwargs in update_dict.items():
            if interval_key in self.interval_datasources:
                if not isinstance(interval_update_kwargs, dict):
                    interval_update_kwargs = interval_update_kwargs.to_dict() # deal with EpochDisplayConfig 
                self.interval_datasources[interval_key].update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, **(interval_update_kwargs | kwargs))) ## Fully inline
            else:
                print(f"WARNING: interval_key '{interval_key}' was not found in self.interval_datasources. Skipping update for unknown item.")



    # Interval Positioning Helpers _______________________________________________________________________________________ #

    def get_render_intervals_plot_range(self, debug_print=False):
        """ Gets the most extreme range of all the interval plots 
            Internally calls `self.get_plot_view_range(a_plot)` on each `self.interval_rendering_plots` to determine the absolute ('x_min', 'x_max', 'y_min','y_max') among all these plots.

        Usage:
            curr_x_min, curr_x_max, curr_y_min, curr_y_max = active_2d_plot.get_render_intervals_plot_range()
            (curr_x_min, curr_x_max, curr_y_min, curr_y_max) # (22.3668519082712, 2093.8524703475414, -21.0, 72.85886744622752)
        
        """
        extrema_tuples = []
        for a_plot in self.interval_rendering_plots:
            extrema_tuples.append(self.get_plot_view_range(a_plot, debug_print=debug_print))
        extrema_df = pd.DataFrame(np.array(extrema_tuples), columns=['x_min', 'x_max', 'y_min','y_max'])
        return (extrema_df['x_min'].min(), extrema_df['x_max'].max(), extrema_df['y_min'].min(), extrema_df['y_max'].max())


    def recover_interval_datasources_positioning_properties(self, debug_print=False):
        """ Tries to recover the positioning properties from each of the interval_datasources of active_2d_plot
        
        Usage:

            all_series_positioning_dfs, all_series_compressed_positioning_dfs, all_series_compressed_positioning_update_dicts = active_2d_plot.recover_interval_datasources_positioning_properties()
            # all_series_positioning_dfs
            all_series_compressed_positioning_dfs

        all_series_compressed_positioning_dfs: {'PBEs': {'y_location': -11.666666666666668, 'height': 4.166666666666667},
        'Ripples': {'y_location': -15.833333333333336, 'height': 4.166666666666667},
        'Replays': {'y_location': -20.000000000000004, 'height': 4.166666666666667},
        'Laps': {'y_location': -7.083333333333334, 'height': 4.166666666666667},
        'SessionEpochs': {'y_location': -2.916666666666667, 'height': 2.0833333333333335}}


        >> Can restore with:

            all_series_compressed_positioning_update_dicts = { 'SessionEpochs': {'y_location': -2.916666666666667, 'height': 2.0833333333333335},
            'Laps': {'y_location': -7.083333333333334, 'height': 4.166666666666667},
            'PBEs': {'y_location': -11.666666666666668, 'height': 4.166666666666667},
            'Ripples': {'y_location': -15.833333333333336, 'height': 4.166666666666667},
            'Replays': {'y_location': -20.000000000000004, 'height': 4.166666666666667}}
            active_2d_plot.update_rendered_intervals_visualization_properties(all_series_compressed_positioning_update_dicts)


        """
        all_series_positioning_dfs = {}
        all_series_compressed_positioning_dfs = {}
        all_series_compressed_positioning_update_dicts = {}
        for a_name, a_ds in self.interval_datasources.items():
            # print(a_name, a_ds)
            if isinstance(a_ds, IntervalsDatasource):
                # all_series_positioning_dfs[a_name], a_compressed_series_positioning_df, series_compressed_positioning_update_dict = a_ds.recover_positioning_properties()
                all_series_positioning_dfs[a_name], all_series_compressed_positioning_dfs[a_name], series_compressed_positioning_update_dict = a_ds.recover_positioning_properties()
                if series_compressed_positioning_update_dict is not None:
                    # only one entry, to be expected
                    all_series_compressed_positioning_update_dicts[a_name] = series_compressed_positioning_update_dict
                else:
                    print(f'ERROR: series_compressed_positioning_update_dict is None for {a_name}. it will not be represented in the output dict.')            
            else:
                if debug_print:
                    print(f'weird a_name, a_ds: {a_name}, {a_ds}, type(a_ds): {type(a_ds)}')
                pass

        return all_series_positioning_dfs, all_series_compressed_positioning_dfs, all_series_compressed_positioning_update_dicts




    def extract_interval_bottom_top_area(self, debug_print=False):
        """ Computes the REQUIRED display rectangles for each of the `self.interval_datasources`. Does NOT take into account the active raster data.

        TODO: Unused, and obsoleted by `self.recover_interval_datasources_positioning_properties(...)
        Usage:
            upper_extreme_vertical_offset, lower_extreme_vertical_offsets = active_2d_plot.extract_interval_bottom_top_area()
            (upper_extreme_vertical_offset, lower_extreme_vertical_offsets) # (-24.16666666666667, -5.0)
        """
        all_series_positioning_dfs = {}
        all_series_vertical_offsets = []
        all_series_heights = []

        for a_name, a_ds in self.interval_datasources.items():
            if isinstance(a_ds, IntervalsDatasource):
                curr_df = a_ds.df[['series_vertical_offset', 'series_height']].copy()
                # all_series_positioning_dfs.append(curr_df)
                all_series_positioning_dfs[a_name] = curr_df
                all_series_vertical_offsets.extend(curr_df.series_vertical_offset.values)
                all_series_heights.extend(curr_df.series_height.values)
            else:
                if debug_print:
                    print(f'weird a_name, a_ds: {a_name}, {a_ds}, type(a_ds): {type(a_ds)}')
                pass

        # Convert to a numpy array:
        all_series_vertical_offsets = np.array(all_series_vertical_offsets)
        all_series_heights = np.array(all_series_heights)

        # series can render either 'above' or 'below':
        is_series_below = (all_series_vertical_offsets <= 0.0) # all elements less than or equal to zero indicate that it's below the plot, and its height will be added negatively to find the max-y value

        _temp_active_effective_series_heights = all_series_heights.copy()
        _temp_active_effective_series_heights[is_series_below] = -1.0 * _temp_active_effective_series_heights[is_series_below] # effective heights are negative for series below the y-axis
        _temp_active_effective_series_extreme_vertical_offsets = all_series_vertical_offsets + _temp_active_effective_series_heights

        return ( _temp_active_effective_series_extreme_vertical_offsets.min(), _temp_active_effective_series_extreme_vertical_offsets.max()), all_series_positioning_dfs # (-24.16666666666667, -5.0)



    def apply_stacked_epoch_layout(self, rendered_interval_keys, desired_interval_height_ratios, epoch_render_stack_height=20.0, interval_stack_location='below', debug_print=True):
        """ Builds and applies a stacked layout for the list of specified epochs

            rendered_interval_keys = ['_', 'SessionEpochs', 'Laps', '_', 'PBEs', 'Ripples', 'Replays'] # '_' indicates a vertical spacer
            rendered_interval_heights = [0.2, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0] # ratio of heights to each interval
            vertical_spacer_height = 0.2
            epoch_render_stack_height = 40.0 # the height of the entire stack containing all rendered epochs:
            interval_stack_location = 'below' # 'below' or 'above'

        Usage:
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs

            rendered_interval_keys = ['_', 'SessionEpochs', 'Laps', '_', 'PBEs', 'Ripples', 'Replays'] # '_' indicates a vertical spacer
            desired_interval_height_ratios = [0.2, 0.5, 1.0, 0.1, 1.0, 1.0, 1.0] # ratio of heights to each interval (and the vertical spacers)
            stacked_epoch_layout_dict = active_2d_plot.apply_stacked_epoch_layout(rendered_interval_keys, desired_interval_height_ratios, epoch_render_stack_height=20.0, interval_stack_location='below')
            stacked_epoch_layout_dict


            ## Inline Concise: Position Replays, PBEs, and Ripples all below the scatter:
            for interval_key, y_location, height in zip(rendered_interval_keys, required_vertical_offsets, required_interval_heights):
                if interval_key in active_2d_plot.interval_datasources:
                    active_2d_plot.interval_datasources[interval_key].update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, y_location=y_location, height=height, **kwargs)) ## Fully inline
        """
        assert len(rendered_interval_keys) == len(desired_interval_height_ratios), f"len(rendered_interval_keys): {len(rendered_interval_keys)} != len(desired_interval_height_ratios): {len(desired_interval_height_ratios)}"
        required_vertical_offsets, required_interval_heights = self.build_stacked_epoch_layout(desired_interval_height_ratios, epoch_render_stack_height=epoch_render_stack_height, interval_stack_location=interval_stack_location)
        
        if interval_stack_location == 'below':
            # required_vertical_offsets = required_vertical_offsets * -1.0 # make offsets negative if it's below the plot
            pass
        elif interval_stack_location == 'above':
            # if it's to be placed above the plot, we need to add the top of the plot to each of the offsets:
            curr_x_min, curr_x_max, curr_y_min, curr_y_max = self.get_render_intervals_plot_range()
            required_vertical_offsets = required_vertical_offsets + curr_y_max # TODO: get top of plot
        else:
            print(f"interval_stack_location: str must be either ('below' or 'above') but was {interval_stack_location}")
            raise NotImplementedError
        
        # Build update dict:
        stacked_epoch_layout_dict = {interval_key:dict(y_location=y_location, height=height) for interval_key, y_location, height in zip(rendered_interval_keys, required_vertical_offsets, required_interval_heights)} # Build a stacked_epoch_layout_dict to update the display
        self.update_rendered_intervals_visualization_properties(stacked_epoch_layout_dict)

        return stacked_epoch_layout_dict


    # Separator Lines ____________________________________________________________________________________________________ #

    def add_raster_spikes_and_epochs_separator_line(self):
        """ adds a thick separator line between the spikes and the epochs. """
        _out_lines_dict = {}
        for a_dest_plot in self.interval_rendering_plots:
            _out_lines_dict[a_dest_plot.objectName()] = a_dest_plot.addLine(x=None, y=0.0, pen={'color':'w', 'width':3.0}, name='EpochDividerLine') # pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.InfiniteLine.InfiniteLine
        return _out_lines_dict



    # 2023-10-16 - Interval `EpochDisplayConfig` extraction from datasources: ____________________________________________ #
    def extract_interval_display_config_lists(self):
        """ Build the EpochDisplayConfig lists for each interval datasource

        
        Usage:
        
        import panel as pn
        pn.extension()

        out_configs_dict = active_2d_plot.extract_interval_display_config_lists()
        pn.Row(*[pn.Column(*[pn.Param(a_sub_v) for a_sub_v in v]) for k,v in out_configs_dict.items()])

        """
        from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.epochs_plotting_mixins import EpochDisplayConfig
        out_configs_dict = {}
        rendered_epoch_names = self.interval_datasource_names
        for a_name in rendered_epoch_names:
            a_ds = self.interval_datasources[a_name]
            result = EpochDisplayConfig.init_configs_list_from_interval_datasource_df(a_name, a_ds)
            out_configs_dict[a_name] = result

        return out_configs_dict
    

    # ---------------------------------------------------------------------------- #
    #                          Private Implementor Methods                         #
    # ---------------------------------------------------------------------------- #
    def _perform_add_render_item(self, a_plot, a_render_item):
        """Performs the operation of adding the render item from the plot specified

        Args:
            a_render_item (_type_): _description_
            a_plot (_type_): _description_
        """
        raise NotImplementedError  # Needs to be overriden for the specific plot type in the implementor
        
        
    def _perform_remove_render_item(self, a_plot, a_render_item):
        """Performs the operation of removing the render item from the plot specified

        Args:
            a_render_item (IntervalRectsItem): _description_
            a_plot (PlotItem): _description_
        """
        raise NotImplementedError  # Needs to be overriden for the specific plot type in the implementor
    
    
    # ---------------------------------------------------------------------------- #
    #                                 Class Methods                                #
    # ---------------------------------------------------------------------------- #
    @classmethod
    def compute_bounds_adjustment_for_rect_item(cls, a_plot, a_rect_item, should_apply_adjustment:bool=True, debug_print=False):
        """ 
        NOTE: 2D Only
        
        Inputs:
            a_plot: PlotItem or equivalent
            a_rect_item: 
            should_apply_adjustment: bool - If True, the adjustment is actually applied
        Returns:
            adjustment_needed: a float representing the difference of adjustment after adjusting or NONE if no changes needed
            
        Usage:
            Called in add_rendered_intervals(...) above, but not sure if it's working or helping.
            The rects that do work are the BurstIntervals which rely on *.y_fragile_linear_neuron_IDX_map instead.
        """
        adjustment_needed = None
        curr_x_min, curr_x_max, curr_y_min, curr_y_max = cls.get_plot_view_range(a_plot, debug_print=False) # curr_x_min: 22.30206346133491, curr_x_max: 1739.1355703625595, curr_y_min: 0.5, curr_y_max: 39.5        
        if debug_print:
            print(f'compute_bounds_adjustment_for_rect_item(a_plot, a_rect_item):')
            print(f'\ta_plot.y: {curr_y_min}, {curr_y_max}')
            
        new_min_y_range, new_max_y_range = cls.get_added_rect_item_required_y_value(a_rect_item, debug_print=debug_print)
        if (new_max_y_range > curr_y_max):
            # needs adjustment
            adjustment_needed = (new_max_y_range - curr_y_max)
            if debug_print:
                print(f'\t needs adjustment: a_rect_item requested new y_max: {new_max_y_range}')
                    
        final_y_max = max(new_max_y_range, curr_y_max)
        
        if (new_min_y_range < curr_y_min):
            # needs adjustment
            if adjustment_needed is None:
                adjustment_needed = 0
            adjustment_needed = adjustment_needed + (new_min_y_range - curr_y_min)
            if debug_print:
                print(f'\t needs adjustment: a_rect_item requested new new_min_y_range: {new_min_y_range}')            
        else:
            adjusted_y_min_range = new_min_y_range
    
        final_y_min = min(new_min_y_range, curr_y_min)
    
        if (adjustment_needed and should_apply_adjustment):
            a_plot.setYRange(final_y_min, final_y_max, padding=0)
    
        return adjustment_needed
    
    
    @staticmethod
    def get_added_rect_item_required_y_value(a_rect_item, debug_print=False):
        """  
        NOTE: 2D Only
            curr_rect.top() # 43.0
            curr_rect.bottom() # 45.0 (why is bottom() greater than top()?
            # curr_rect.y()
            
         Usage:
            Only known to be used by .compute_bounds_adjustment_for_rect_item(...) above
        """
        curr_rect = a_rect_item.boundingRect() # PyQt5.QtCore.QRectF(29.0, 43.0, 1683.0, 2.0)
        new_min_y_range = min(curr_rect.top(), curr_rect.bottom())
        new_max_y_range = max(curr_rect.top(), curr_rect.bottom())
        if debug_print:
            print(f'new_min_y_range: {new_min_y_range}')
            print(f'new_max_y_range: {new_max_y_range}')
        return new_min_y_range, new_max_y_range

    
    @staticmethod
    def get_plot_view_range(a_plot, debug_print=True):
        """ gets the current viewRange for the passed in plot
        NOTE: 2D Only
      
        Inputs:
            a_plot: PlotItem
        Returns:
            (curr_x_min, curr_x_max, curr_y_min, curr_y_max)

        Usage:
            Only known to be used by .compute_bounds_adjustment_for_rect_item(...) above
            
        Examples:
            curr_x_min, curr_x_max, curr_y_min, curr_y_max = get_plot_view_range(main_plot_widget, debug_print=True)
            curr_x_min, curr_x_max, curr_y_min, curr_y_max = get_plot_view_range(background_static_scroll_plot_widget, debug_print=True)
        """
        curr_x_range, curr_y_range = a_plot.viewRange() # [[30.0, 45.0], [-1.359252049028905, 41.3592520490289]]
        if debug_print:
            print(f'curr_x_range: {curr_x_range}, curr_y_range: {curr_y_range}')
        curr_x_min, curr_x_max = curr_x_range
        curr_y_min, curr_y_max = curr_y_range
        # curr_x_min, curr_x_max, curr_y_min, curr_y_max = main_plot_widget.viewRange()
        if debug_print:
            print(f'curr_x_min: {curr_x_min}, curr_x_max: {curr_x_max}, curr_y_min: {curr_y_min}, curr_y_max: {curr_y_max}')
        return (curr_x_min, curr_x_max, curr_y_min, curr_y_max)


    @classmethod
    def build_stacked_epoch_layout(cls, rendered_interval_heights, epoch_render_stack_height=40.0, interval_stack_location='below', debug_print=True):
        """ Builds a stack layout for the list of specified epochs

            rendered_interval_keys = ['_', 'SessionEpochs', 'Laps', '_', 'PBEs', 'Ripples', 'Replays'] # '_' indicates a vertical spacer
            rendered_interval_heights = [0.2, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0] # ratio of heights to each interval
            vertical_spacer_height = 0.2
            epoch_render_stack_height = 40.0 # the height of the entire stack containing all rendered epochs:
            interval_stack_location = 'below' # 'below' or 'above'

        Usage:
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs

            rendered_interval_keys = ['_', 'SessionEpochs', 'Laps', '_', 'PBEs', 'Ripples', 'Replays'] # '_' indicates a vertical spacer
            desired_interval_height_ratios = [0.2, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0] # ratio of heights to each interval
            required_vertical_offsets, required_interval_heights = build_stacked_epoch_layout(desired_interval_height_ratios, epoch_render_stack_height=40.0, interval_stack_location='below')


            ## Inline Concise: Position Replays, PBEs, and Ripples all below the scatter:
            for interval_key, y_location, height in zip(rendered_interval_keys, required_vertical_offsets, required_interval_heights):
                if interval_key in active_2d_plot.interval_datasources:
                    active_2d_plot.interval_datasources[interval_key].update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, y_location=y_location, height=height, **kwargs)) ## Fully inline
        """
        normalized_interval_heights = rendered_interval_heights/np.sum(rendered_interval_heights) # array([0.2, 0.2, 0.2, 0.2, 0.2])
        required_interval_heights = normalized_interval_heights * epoch_render_stack_height # array([3.2, 3.2, 3.2, 3.2, 3.2])
        required_vertical_offsets = np.cumsum(required_interval_heights) # array([ 3.2  6.4  9.6 12.8 16.])
        if interval_stack_location == 'below':
            required_vertical_offsets = required_vertical_offsets * -1.0 # make offsets negative if it's below the plot
        elif interval_stack_location == 'above':
            # if it's to be placed above the plot, we need to add the top of the plot to each of the offsets:
            required_vertical_offsets = required_vertical_offsets + 0.0 # TODO: get top of plot
        else:
            print(f"interval_stack_location: str must be either ('below' or 'above') but was {interval_stack_location}")
            raise NotImplementedError
        if debug_print:
            print(f'required_interval_heights: {required_interval_heights}, required_vertical_offsets: {required_vertical_offsets}')

        return required_vertical_offsets, required_interval_heights

