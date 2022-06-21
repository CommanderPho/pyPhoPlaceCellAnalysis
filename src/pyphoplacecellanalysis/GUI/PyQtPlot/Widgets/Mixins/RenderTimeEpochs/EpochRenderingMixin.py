from copy import copy, deepcopy
from neuropy.core import Epoch

from qtpy import QtCore

from pyphocorehelpers.print_helpers import SimplePrintable, PrettyPrintable, iPythonKeyCompletingMixin
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

from pyphocorehelpers.DataStructure.general_parameter_containers import DebugHelper, VisualizationParameters, RenderPlots, RenderPlotsData
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem import IntervalRectsItem, RectangleRenderTupleHelpers
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import Specific2DRenderTimeEpochsHelper
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Render2DEventRectanglesHelper import Render2DEventRectanglesHelper

from pyphoplacecellanalysis.General.Model.Datasources.IntervalDatasource import IntervalsDatasource



class RenderedEpochsItemsContainer(iPythonKeyCompletingMixin, DynamicParameters):
    """ Wraps a list of plots and their rendered_rects_item for a given datasource/name
    
        Note that the plots are only given by self.dynamically_added_attributes since the 'name' key exists.
    
    """
    def __init__(self, rendered_rects_item, target_plots_list):
        super(RenderedEpochsItemsContainer, self).__init__()
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
    @property
    def interval_rendering_plots(self):
        """ returns the list of child subplots/graphics (usually PlotItems) that participate in rendering intervals """
        raise NotImplementedError # MUST OVERRIDE in child
        # return [self.plots.background_static_scroll_window_plot, self.plots.main_plot_widget] # for spike_raster_plt_2d
    
    @property
    def interval_datasources(self):
        """The interval_datasources property."""
        return self.plots_data['interval_datasources']
 
    @property
    def rendered_epochs(self):
        """The interval_datasources property."""
        return self.plots.rendered_epochs
    
    #######################################################################################################################################
    
    @QtCore.Slot()
    def EpochRenderingMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        self.plots_data['interval_datasources'] = RenderPlotsData('EpochRenderingMixin')
    
    @QtCore.Slot()
    def EpochRenderingMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        self.plots.rendered_epochs = RenderPlots('EpochRenderingMixin') # the container to hold the time rectangles

    @QtCore.Slot()
    def EpochRenderingMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass

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
    def add_rendered_intervals(self, interval_datasource, name=None, child_plots=None, debug_print=True):
        """ adds the intervals specified by the interval_datasource to the plots 
        
        Inputs: 
            interval_datasource: IntervalDatasource
            name: str, an optional but highly recommended string identifier like 'Laps'
            child_plots: an optional list of plots to add the intervals to. If None are specified, the defaults are used (defined by the implementor)
            
        Usage:
            active_pbe_interval_rects_item = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_interval_datasource(interval_datasources.PBEs)
            
        """
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
                        
        
        # Build the rendered interval item:
        new_interval_rects_item = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_interval_datasource(interval_datasource)
        new_interval_rects_item.setToolTip(name)
        
        ######### PLOTS:
        
        # TODO: store the IntervalRectsItem somewhere? Probably in Plots?
        # self.plots.rendered_epochs[]
        
        extant_rects_plot_items_container = self.rendered_epochs.get(name, None)
        if extant_rects_plot_items_container is not None:
            # extant plot exists!
            print(f'WARNING: extant_rects_plot_item with the name ({name}) already exists. removing.')
            assert isinstance(extant_rects_plot_items_container, RenderedEpochsItemsContainer), f"extant_rects_plot_item must be RenderedEpochsItemsContainer but type(extant_rects_plot_item): {type(extant_rects_plot_items_container)}"
            
            ## TODO: should I actually update this one instead?
            # extant_rects_plot_item.data = 
            ## TODO: remove!
            for a_plot in child_plots:
                if a_plot in extant_rects_plot_items_container:
                    # the plot is already here: remove and re-add it
                    extant_rect_plot_item = extant_rects_plot_items_container[a_plot]
                    a_plot.removeItem(extant_rect_plot_item) # Remove it from the plot
                    
                    
                    # TODO: update the item's data instead of replacing it
                    # # add the new one:
                    # extant_rects_plot_items_container[a_plot] = new_interval_rects_item.copy()
                    # a_plot.addItem(extant_rects_plot_items_container[a_plot])
                # else:
                #     # Otherwise it isn't in there, copy it and insert it
                #     pass
                # Need to duplicate the rect item for each child plot (need unique instance per plot):
                # extant_rects_plot_items_container[a_plot] = IntervalRectsItem(data=deepcopy(new_interval_rects_item.data))
                
                independent_data_copy = RectangleRenderTupleHelpers.copy_data(new_interval_rects_item.data)
                extant_rects_plot_items_container[a_plot] = IntervalRectsItem(data=independent_data_copy)
                extant_rects_plot_items_container[a_plot].setToolTip(name)
                # extant_rects_plot_items_container[a_plot] = new_interval_rects_item.copy()
                a_plot.addItem(extant_rects_plot_items_container[a_plot])
                
        else:
            # Need to create a new RenderedEpochsItemsContainer with the items:
            self.rendered_epochs[name] = RenderedEpochsItemsContainer(new_interval_rects_item, child_plots) # set the plot item
            for a_plot, a_rect_item in self.rendered_epochs[name].items():
                if not isinstance(a_rect_item, str):
                    if debug_print:
                        print(f'plotting item')
                    a_plot.removeItem(a_rect_item)
                    a_plot.addItem(a_rect_item)

       

    def remove_rendered_intervals(self, name, child_plots_removal_list=None, debug_print=True):
        """ removes the intervals specified by the interval_datasource to the plots
        name: the name of the rendered_repochs to remove.
        child_plots_removal_list: is not-None, a list of child plots can be specified and rects will only be removed from those plots.
        """
        extant_rects_plot_item = self.rendered_epochs[name]
        items_to_remove_from_rendered_epochs = []
        for a_plot, a_rect_item in extant_rects_plot_item.items():
            if not isinstance(a_rect_item, str):                
                if child_plots_removal_list is not None:
                    if a_plot in child_plots_removal_list:
                        # only remove if the plot is in the child plots:
                        a_plot.removeItem(a_rect_item)
                        items_to_remove_from_rendered_epochs.append(a_plot)
                    else:
                        pass # continue
                else:
                    # otherwise remove all
                    a_plot.removeItem(a_rect_item)
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
    

        
    def clear_all_rendered_intervals(self, child_plots_removal_list=None, debug_print=True):
        curr_rendered_epoch_names = list(self.rendered_epochs.keys()) # done to prevent problems with dict changing size during iteration
        for a_name in curr_rendered_epoch_names:
            if a_name != 'name':
                if debug_print:
                    print(f'removing {a_name}...')
                self.remove_rendered_intervals(a_name, child_plots_removal_list=child_plots_removal_list, debug_print=debug_print)
        
    
    @staticmethod
    def get_plot_view_range(a_plot, debug_print=True):
        """ gets the current viewRange for the passed in plot
        Inputs:
            a_plot: PlotItem
        Returns:
            (curr_x_min, curr_x_max, curr_y_min, curr_y_max)

        Usage:
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

