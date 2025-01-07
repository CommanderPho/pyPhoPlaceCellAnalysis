from copy import deepcopy
import numpy as np
import pandas as pd
from attrs import define, field, fields, asdict, astuple

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomLinearRegionItem import CustomLinearRegionItem

from pyphoplacecellanalysis.General.Model.Datasources.Datasources import DataframeDatasource


@define(frozen=True)
class ScatterItemData:
    t: float = field(alias='t_rel_seconds')
    aclu: int = field() # alias='neuron_ID'
    neuron_IDX: int = field(alias='fragile_linear_neuron_IDX')
    visualization_raster_y_location: float = field(default=np.nan)


class Render2DScrollWindowPlotMixin:
    """ Adds a LinearRegionItem to the plot that represents the entire data timerange which defines a user-adjustable window into the data. Finally, also adds a plot that shows only the zoomed-in data within the window. 
    
    Known Uses:
        Implemented by Spike2DRaster
    
    Requires:
        a Datasource to fetch the spiking data from.
        TimeWindow: a Active Window to synchronize the LinearRegionItem (2D Scroll Widget) with.
    
    
    Provides:
        window_scrolled (float, float) signal is emitted on updating the 2D sliding window, where the first argument is the new start value and the 2nd is the new end value
    """
    
    ## Scrollable Window Signals
    window_scrolled = QtCore.pyqtSignal(float, float) # signal is emitted on updating the 2D sliding window, where the first argument is the new start value and the 2nd is the new end value
    
    
    def ScrollRasterPreviewWindow_on_BuildUI(self, background_static_scroll_window_plot):
        """ Note that this doesn't need to update because the background is static (it shows all time) 
        
        Inputs:
        
        background_static_scroll_window_plot: the plot to add to. For example created with `graphics_layout_widget.addPlot(row=layout_row, col=layout_col)`
         
        Requires:
        
            self.plots
            self.ui
            self.params.scroll_window_plot_downsampling_rate
            
            self.spikes_window.total_df_start_end_times # to get the current start/end times to set the linear region to
            
        Creates:
            self.plots_data.all_spots # data for all spikes to be rendered on a scatter plot
            self.plots_data.all_spots_downsampled # temporally downsampled data for all spikes to be rendered on a scatter plot
            self.ui.scroll_window_region # a pg.LinearRegionItem                        
            self.plots.preview_overview_scatter_plot # a pg.ScatterPlotItem
        
        Usage:
            self.plots.background_static_scroll_window_plot = self.ui.main_graphics_layout_widget.addPlot(row=2, col=0)
            self.plots.background_static_scroll_window_plot = self.ScrollRasterPreviewWindow_on_BuildUI(self.plots.background_static_scroll_window_plot)
        
        
        """
        # Common Tick Label
        vtick = QtGui.QPainterPath()
        vtick.moveTo(0, -0.5)
        vtick.lineTo(0, 0.5)
        
        #############################
        ## Bottom Windowed Scroll Plot/Widget:
        scroll_window_plot_downsampling_rate: int = self.params.setdefault('scroll_window_plot_downsampling_rate', 100)
        
        # ALL Spikes in the preview window:
        # curr_spike_x, curr_spike_y, curr_spike_pens, _all_scatterplot_tooltips_kwargs, self.plots_data.all_spots, curr_n = self._build_all_spikes_data_values(should_return_data_tooltips_kwargs=False, downsampling_rate=scroll_window_plot_downsampling_rate) #TODO 2023-06-28 21:18: - [ ] Could use returned tooltips to set the spike hover text
        
        curr_spike_x, _, curr_spike_pens, _all_scatterplot_tooltips_kwargs, self.plots_data.all_spots_downsampled, curr_n = self._build_all_spikes_data_values(should_return_data_tooltips_kwargs=False, downsampling_rate=scroll_window_plot_downsampling_rate) # downsampled all_spots
        curr_spike_x, curr_spike_y, curr_spike_pens, _all_scatterplot_tooltips_kwargs, self.plots_data.all_spots, curr_n = self._build_all_spikes_data_values(should_return_data_tooltips_kwargs=False, downsampling_rate=1) ## all spots


        self.plots.preview_overview_scatter_plot = pg.ScatterPlotItem(name='spikeRasterOverviewWindowScatterPlotItem', pxMode=True, symbol=vtick, size=5, pen={'color': 'w', 'width': 1}, hoverable=False, )
        self.plots.preview_overview_scatter_plot.setObjectName('preview_overview_scatter_plot') # this seems necissary, the 'name' parameter in addPlot(...) seems to only change some internal property related to the legend AND drastically slows down the plotting
        self.plots.preview_overview_scatter_plot.opts['useCache'] = True
        # self.plots.preview_overview_scatter_plot.addPoints(self.plots_data.all_spots) # , hoverable=True
        self.plots.preview_overview_scatter_plot.addPoints(self.plots_data.all_spots_downsampled)
        background_static_scroll_window_plot.addItem(self.plots.preview_overview_scatter_plot)
        
        # Add the linear region overlay:
        self.ui.scroll_window_region = CustomLinearRegionItem(pen=pg.mkPen('#fff'), brush=pg.mkBrush('#f004'), hoverBrush=pg.mkBrush('#fff4'), hoverPen=pg.mkPen('#f00'), clipItem=self.plots.preview_overview_scatter_plot) # bound the LinearRegionItem to the plotted data
        self.ui.scroll_window_region.setObjectName('scroll_window_region')
        self.ui.scroll_window_region.setZValue(10)
        # Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this item when doing auto-range calculations.
        background_static_scroll_window_plot.addItem(self.ui.scroll_window_region, ignoreBounds=True)
        self.ui.scroll_window_region.sigRegionChanged.connect(self._Render2DScrollWindowPlot_on_linear_region_item_update)

        
        # Setup axes bounds for the bottom windowed plot:
        background_static_scroll_window_plot.hideAxis('left')
        background_static_scroll_window_plot.hideAxis('bottom')
        # background_static_scroll_window_plot.setLabel('bottom', 'Time', units='s')
        background_static_scroll_window_plot.setMouseEnabled(x=False, y=False)
        background_static_scroll_window_plot.disableAutoRange('xy')
        # background_static_scroll_window_plot.enableAutoRange(x=False, y=False)
        background_static_scroll_window_plot.setAutoVisible(x=False, y=False)
        background_static_scroll_window_plot.setAutoPan(x=False, y=False)
        
        # Setup range for plot:
        earliest_t, latest_t = self.spikes_window.total_df_start_end_times
        background_static_scroll_window_plot.setXRange(earliest_t, latest_t, padding=0)
        background_static_scroll_window_plot.setYRange(np.nanmin(curr_spike_y), np.nanmax(curr_spike_y), padding=0)
        
        return background_static_scroll_window_plot

    
    def _fix_initial_linearRegionLocation(self, debug_print=False):
        """ Hopefully finally resolves the issue where the linear scroll region window was always cut-off initially. 
        Note that when this is called, self.temporal_axis_length and self.spikes_window.window_duration were both observed to be 0.0, which is why they couldn't be used.
        """
        confirmed_valid_window_start_t = self.spikes_window.total_data_start_time
        if (self.spikes_window.window_duration == 0.0):
            # invalid window length, just choose something reasonable the user can grab, say 5% of the total window data
            total_data_duration = self.spikes_window.total_data_end_time - self.spikes_window.total_data_start_time
            reasonable_active_window_duration = float(total_data_duration) * 0.05 # 5%
            ## UGHH, it works but please note that the final window is actually going to be MORE than 5% of the total data duration because of the temporal_zoom_factor > 1.0. 
        else:
            reasonable_active_window_duration = float(self.spikes_window.window_duration)
            
        # Compute the final reasonable window end_t:
        confirmed_valid_window_end_t = confirmed_valid_window_start_t + reasonable_active_window_duration
        if debug_print:
            print(f'_fix_initial_linearRegionLocation():')
            print(f'\tconfirmed_valid_window: (start_t: {confirmed_valid_window_start_t}, end_t: {confirmed_valid_window_end_t})')
        ## THIS SHOULD FIX THE INITIAL SCROLLWINDOW ISSUE, preventing it from being outside the window it's rendered on top of, unless the active window is set wrong.
        # self.update_scroll_window_region(confirmed_valid_window_start_t, confirmed_valid_window_end_t, block_signals=False)
        self.Render2DScrollWindowPlot_on_window_update(confirmed_valid_window_start_t, confirmed_valid_window_end_t)
        ## TODO: attach the event forwarder 2025-01-02
        # self.ui.main_graphics_layout_widget.set_target_event_forwarding_child(self.ui.scroll_window_region) #TODO 2025-01-02 07:55: - [ ] Did not work
        

    def update_rasters(self):
        """ updates all rasters (which are scatter plots) from the self.plot_data.all_spots variable """
        # Update preview_overview_scatter_plot
        self.plots.preview_overview_scatter_plot.setData(self.plots_data.all_spots)
        if self.Includes2DActiveWindowScatter:
            self.plots.scatter_plot.setData(self.plots_data.all_spots)
            
            

    @QtCore.pyqtSlot()
    def _Render2DScrollWindowPlot_on_linear_region_item_update(self) -> None:
        """self when the region moves.zoom_Change plotter area"""
        # self.ui.scroll_window_region.setZValue(10) # bring to the front
        min_x, max_x = self.ui.scroll_window_region.getRegion() # get the current region
        self.window_scrolled.emit(min_x, max_x) # emit this mixin's own window_scrolled function
        
        

    @QtCore.pyqtSlot()
    def Render2DScrollWindowPlot_on_init():
        """ perform any parameters setting/checking during init """
        pass

    @QtCore.pyqtSlot()
    def Render2DScrollWindowPlot_on_setup():
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        # Connect the signals for the zoom region and the LinearRegionItem
        # self.ui.scroll_window_region.sigRegionChanged.connect(self.update_zoom_plotter)
        pass

    @QtCore.pyqtSlot()
    def Render2DScrollWindowPlot_on_destroy():
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        pass

    @QtCore.pyqtSlot(float, float)
    def Render2DScrollWindowPlot_on_window_update(self, new_start=None, new_end=None):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc. """
        # Make sure that the scroller isn't too tiny to grab.
        self.ui.scroll_window_region.setRegion([new_start, new_end]) # adjust scroll control
    
    
    # ==================================================================================================================== #
    # Private/Internal Methods                                                                                             #
    # ==================================================================================================================== #
    
    def _build_all_spikes_data_values(self, is_included_indicies=None, **kwargs):
        """ build global spikes for entire dataframe (not just the current window) 
        
        Called ONLY by ScrollRasterPreviewWindow_on_BuildUI(self, background_static_scroll_window_plot)

        Implementation:
            Uses the df['visualization_raster_y_location'] field added to the spikes dataframe to get the y-value for the spike
            Note that the colors are built using the self.config_fragile_linear_neuron_IDX_map property
            is_included_indicies: Optional np.array of bools indicating whether each spike is included in the generated points
            Internally calls `cls.build_spikes_data_values_from_df(...)
            
        """
        # All units at once approach:
        return Render2DScrollWindowPlotMixin.build_spikes_data_values_from_df(self.spikes_window.df, self.config_fragile_linear_neuron_IDX_map, is_spike_included=is_included_indicies, **kwargs)
        
    def _build_all_spikes_all_spots(self, is_included_indicies=None, **kwargs):
        """ build the all_spots from the global spikes for entire dataframe (not just the current window) 
        

        Called by:
            on_unit_sort_order_changed
            on_neuron_colors_changed
            reset_spike_emphasis
            update_spike_emphasis


        Example:
            ## Rebuild Raster Plot Points:
            self._build_cell_configs()

            # ALL Spikes in the preview window:
            self.plots_data.all_spots = self._build_all_spikes_all_spots()
            # Update preview_overview_scatter_plot
            self.plots.preview_overview_scatter_plot.setData(self.plots_data.all_spots)
            if self.Includes2DActiveWindowScatter:
                self.plots.scatter_plot.setData(self.plots_data.all_spots)     
        """
        return Render2DScrollWindowPlotMixin.build_spikes_all_spots_from_df(self.spikes_window.df, self.config_fragile_linear_neuron_IDX_map, is_spike_included=is_included_indicies, **kwargs)
    


    
    # ==================================================================================================================== #
    # Class/Static Methods                                                                                                 #
    # ==================================================================================================================== #
    
    @classmethod
    def _build_spike_data_tuples_from_spikes_df(cls, spikes_df: pd.DataFrame, generate_debug_tuples=False) -> dict:
        """ generates a list of tuples uniquely identifying each spike in the spikes_df as requested by the pg.ScatterPlotItem's `data` argument.
                data: a list of python objects used to uniquely identify each spot.
                tip: A string-valued function of a spot's (x, y, data) values. Set to None to prevent a tool tip from being shown.
        """
        # from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
        if generate_debug_tuples:
            # debug_datapoint_column_names = [spikes_df.spikes.time_variable_name, 'shank', 'cluster', 'aclu', 'qclu', 'x', 'y', 'speed', 'traj', 'lap', 'maze_relative_lap', 'maze_id', 'neuron_type', 'flat_spike_idx', 'x_loaded', 'y_loaded', 'lin_pos', 'fragile_linear_neuron_IDX', 'PBE_id', 'scISI', 'neuron_IDX', 'replay_epoch_id', 'visualization_raster_y_location', 'visualization_raster_emphasis_state']
            debug_datapoint_column_names = [spikes_df.spikes.time_variable_name, 'aclu', 'fragile_linear_neuron_IDX', 'visualization_raster_y_location'] # a subset I'm actually interested in for debugging
            active_datapoint_column_names = debug_datapoint_column_names # all values for the purpose of debugging
        else:
            default_datapoint_column_names = [spikes_df.spikes.time_variable_name, 'aclu', 'fragile_linear_neuron_IDX']
            active_datapoint_column_names = default_datapoint_column_names
            
        def _tip_fn(x, y, data):
            """ the function required by pg.ScatterPlotItem's `tip` argument to print the tooltip for each spike. """
            # data_string:str = '\n'.join([f"{k}:\t{str(v)}" for k, v in zip(active_datapoint_column_names, data)])
            # data_string:str = '\n'.join([f"{k}:\t{str(v)}" for k, v in asdict(data).items()])
            data_string:str = '|'.join([f"{k}: {str(v)}" for k, v in asdict(data).items()])
            print(f'_tip_fn(...): data_string: {data_string}')
            return f"spike: (x={x:.3f}, y={y:.2f})\n{data_string}"

        # spikes_data = spikes_df[active_datapoint_column_names].to_records(index=False).tolist() # list of tuples
        spikes_data = spikes_df[active_datapoint_column_names].to_dict('records') # list of dicts
        spikes_data = [ScatterItemData(**v) for v in spikes_data] 
        
        # spikes_data = [DynamicParameters.init_from_dict(v) for v in spikes_data] # convert to list of DynamicParameters objects
        return dict(data=spikes_data, tip=_tip_fn)


    @classmethod
    def build_spikes_data_values_from_df(cls, spikes_df: pd.DataFrame, config_fragile_linear_neuron_IDX_map, is_spike_included=None, should_return_data_tooltips_kwargs:bool=False, downsampling_rate: int = 10, **kwargs):
        """ build global spikes for entire dataframe (not just the current window) 
        
        Called by:
            self.build_spikes_all_spots_from_df(...)
            cls.build_spikes_all_spots_from_df(...)
            
            
        Needs to be called whenever:
            spikes_df['visualization_raster_y_location']
            spikes_df['visualization_raster_emphasis_state']
            spikes_df['fragile_linear_neuron_IDX']
        Changes.

        
        Uses the df['visualization_raster_y_location'] field added to the spikes dataframe to get the y-value for the spike
        
        Note that the colors are built using the pens contained in self.config_fragile_linear_neuron_IDX_map property
        
        config_fragile_linear_neuron_IDX_map: a map from fragile_linear_neuron_IDX to config (tuple) values
        is_included_indicies: Optional np.array of bools indicating whether each spike is included in the generated points
        
        
        2023-12-06 `config_fragile_linear_neuron_IDX_map` comes in mostly empty except for Pens and Brushes for each state
        """
        if (downsampling_rate is not None) and (downsampling_rate > 1):
            active_spikes_df = deepcopy(spikes_df).iloc[::downsampling_rate]  # Take every 10th row
        else:
            active_spikes_df = deepcopy(spikes_df)
            

        # All units at once approach:
        active_time_variable_name = active_spikes_df.spikes.time_variable_name
        # Copy only the relevent columns so filtering is easier:
        filtered_spikes_df = active_spikes_df[[active_time_variable_name, 'visualization_raster_y_location',  'visualization_raster_emphasis_state', 'fragile_linear_neuron_IDX']].copy()
        
        spike_emphasis_states = kwargs.get('spike_emphasis_state', None)
        if spike_emphasis_states is not None:
            assert len(spike_emphasis_states) == np.shape(active_spikes_df)[0], f"if specified, spike_emphasis_states must be the same length as the number of spikes but np.shape(spikes_df)[0]: {np.shape(active_spikes_df)[0]} and len(is_included_indicies): {len(spike_emphasis_states)}"
            # Can set it on the dataframe:
            # 'visualization_raster_y_location'
        
        if is_spike_included is not None:
            assert len(is_spike_included) == np.shape(active_spikes_df)[0], f"if specified, is_included_indicies must be the same length as the number of spikes but np.shape(spikes_df)[0]: {np.shape(active_spikes_df)[0]} and len(is_included_indicies): {len(is_spike_included)}"
            ## filter them by the is_included_indicies:
            filtered_spikes_df = filtered_spikes_df[is_spike_included]
        
        # Filter the dataframe using that column and value from the list
        curr_spike_t = filtered_spikes_df[active_time_variable_name].to_numpy() # this will map
        curr_spike_y = filtered_spikes_df['visualization_raster_y_location'].to_numpy() # this will map
        
        # Build the "tooltips" for each spike:
        # curr_spike_data_tooltips = [f"{an_aclu}" for an_aclu in spikes_df['aclu'].to_numpy()]
        if should_return_data_tooltips_kwargs:
            # #TODO 2023-12-06 03:35: - [ ] This doesn't look like it can sort the tooltips at all, right? Or does this not matter?
            all_scatterplot_tooltips_kwargs = cls._build_spike_data_tuples_from_spikes_df(active_spikes_df, generate_debug_tuples=True) # need the full spikes_df, not the filtered one
            assert len(all_scatterplot_tooltips_kwargs['data']) == np.shape(active_spikes_df)[0], f"if specified, all_scatterplot_tooltips_kwargs must be the same length as the number of spikes but np.shape(spikes_df)[0]: {np.shape(active_spikes_df)[0]} and len((all_scatterplot_tooltips_kwargs['data']): {len(all_scatterplot_tooltips_kwargs['data'])}"
        else:
            all_scatterplot_tooltips_kwargs = None
            
        # config_fragile_linear_neuron_IDX_map values are of the form: (i, fragile_linear_neuron_IDX, curr_pen, self._series_identity_lower_y_values[i], self._series_identity_upper_y_values[i])
        # Emphasis/Deemphasis-Dependent Pens:
        curr_spike_pens = [config_fragile_linear_neuron_IDX_map[a_fragile_linear_neuron_IDX][2][a_spike_emphasis_state] for a_fragile_linear_neuron_IDX, a_spike_emphasis_state in zip(filtered_spikes_df['fragile_linear_neuron_IDX'].to_numpy(), filtered_spikes_df['visualization_raster_emphasis_state'].to_numpy())] # get the pens for each spike from the configs map
        
        has_brushes: bool = np.all([(len(config_fragile_linear_neuron_IDX_map[a_fragile_linear_neuron_IDX])>=6) for a_fragile_linear_neuron_IDX in filtered_spikes_df['fragile_linear_neuron_IDX'].to_numpy()])
        if has_brushes:
            curr_spikes_brushes = [config_fragile_linear_neuron_IDX_map[a_fragile_linear_neuron_IDX][-1][a_spike_emphasis_state] for a_fragile_linear_neuron_IDX, a_spike_emphasis_state in zip(filtered_spikes_df['fragile_linear_neuron_IDX'].to_numpy(), filtered_spikes_df['visualization_raster_emphasis_state'].to_numpy())] # get the pens for each spike from the configs map
        else:
            curr_spikes_brushes = [] # scared to modify/use the brushes here out of fear of breaking the spike_emphasis_states
        
        curr_n = len(curr_spike_t) # curr number of spikes
        # builds the 'all_spots' tuples suitable for setting self.plots_data.all_spots from ALL Spikes
        pos = np.vstack((curr_spike_t, curr_spike_y))
        if has_brushes:
            all_spots = [{'pos': pos[:,i], 'data': i, 'pen': curr_spike_pens[i], 'brush': curr_spikes_brushes[i]} for i in range(curr_n)] # returned spikes {'pos','data','pen'}
        else:
            all_spots = [{'pos': pos[:,i], 'data': i, 'pen': curr_spike_pens[i]} for i in range(curr_n)] # returned spikes {'pos','data','pen'}
        return curr_spike_t, curr_spike_y, curr_spike_pens, all_scatterplot_tooltips_kwargs, all_spots, curr_n
    

    @classmethod
    def build_spikes_all_spots_from_df(cls, spikes_df: pd.DataFrame, config_fragile_linear_neuron_IDX_map, is_spike_included=None, should_return_data_tooltips_kwargs:bool=False, **kwargs):
        """ builds the 'all_spots' tuples suitable for setting self.plots_data.all_spots from ALL Spikes 
        Internally calls `cls.build_spikes_data_values_from_df(...)
        """
        curr_spike_x, curr_spike_y, curr_spike_pens, all_scatterplot_tooltips_kwargs, all_spots, curr_n = cls.build_spikes_data_values_from_df(spikes_df, config_fragile_linear_neuron_IDX_map, is_spike_included=is_spike_included, should_return_data_tooltips_kwargs=should_return_data_tooltips_kwargs, **kwargs)
        if should_return_data_tooltips_kwargs:
            return all_spots, all_scatterplot_tooltips_kwargs
        else:
            return all_spots



@function_attributes(short_name=None, tags=['spikes', 'spots', 'raster', 'scatterplot', 'pyqtgraph', 'unused'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-06 14:02', related_items=[])
def independent_build_spikes_all_spots_from_df(spikes_df: pd.DataFrame, config_fragile_linear_neuron_IDX_map, is_spike_included=None, should_return_data_tooltips_kwargs:bool=False, generate_debug_tuples=False, **kwargs):
    """ Made completely independent from the class methods in `Render2DScrollWindowPlotMixin`. Builds the 'all_spots' tuples suitable for setting self.plots_data.all_spots from ALL Spikes 
        Needs to be called whenever:
            spikes_df['visualization_raster_y_location']
            spikes_df['visualization_raster_emphasis_state']
            spikes_df['fragile_linear_neuron_IDX']
        Changes.
        
        
        History: Flattened on 2023-12-06 but otherwise unmodified
        
    """
    
    # INLINEING `build_spikes_data_values_from_df`: ______________________________________________________________________ #
    # curr_spike_x, curr_spike_y, curr_spike_pens, all_scatterplot_tooltips_kwargs, all_spots, curr_n = cls.build_spikes_data_values_from_df(spikes_df, config_fragile_linear_neuron_IDX_map, is_spike_included=is_spike_included, should_return_data_tooltips_kwargs=should_return_data_tooltips_kwargs, **kwargs)
    # All units at once approach:
    active_time_variable_name = spikes_df.spikes.time_variable_name
    # Copy only the relevent columns so filtering is easier:
    filtered_spikes_df = spikes_df[[active_time_variable_name, 'visualization_raster_y_location',  'visualization_raster_emphasis_state', 'fragile_linear_neuron_IDX']].copy()
    
    spike_emphasis_states = kwargs.get('spike_emphasis_state', None)
    if spike_emphasis_states is not None:
        assert len(spike_emphasis_states) == np.shape(spikes_df)[0], f"if specified, spike_emphasis_states must be the same length as the number of spikes but np.shape(spikes_df)[0]: {np.shape(spikes_df)[0]} and len(is_included_indicies): {len(spike_emphasis_states)}"
        # Can set it on the dataframe:
        # 'visualization_raster_y_location'
    
    if is_spike_included is not None:
        assert len(is_spike_included) == np.shape(spikes_df)[0], f"if specified, is_included_indicies must be the same length as the number of spikes but np.shape(spikes_df)[0]: {np.shape(spikes_df)[0]} and len(is_included_indicies): {len(is_spike_included)}"
        ## filter them by the is_included_indicies:
        filtered_spikes_df = filtered_spikes_df[is_spike_included]
    
    # Filter the dataframe using that column and value from the list
    curr_spike_t = filtered_spikes_df[active_time_variable_name].to_numpy() # this will map
    curr_spike_y = filtered_spikes_df['visualization_raster_y_location'].to_numpy() # this will map
    
    # Build the "tooltips" for each spike:
    # curr_spike_data_tooltips = [f"{an_aclu}" for an_aclu in spikes_df['aclu'].to_numpy()]
    if should_return_data_tooltips_kwargs:
        # #TODO 2023-12-06 03:35: - [ ] This doesn't look like it can sort the tooltips at all, right? Or does this not matter?
        # all_scatterplot_tooltips_kwargs = cls._build_spike_data_tuples_from_spikes_df(spikes_df, generate_debug_tuples=True) # need the full spikes_df, not the filtered one
        # INLINING: _build_spike_data_tuples_from_spikes_df __________________________________________________________________ #

        if generate_debug_tuples:
            # debug_datapoint_column_names = [spikes_df.spikes.time_variable_name, 'shank', 'cluster', 'aclu', 'qclu', 'x', 'y', 'speed', 'traj', 'lap', 'maze_relative_lap', 'maze_id', 'neuron_type', 'flat_spike_idx', 'x_loaded', 'y_loaded', 'lin_pos', 'fragile_linear_neuron_IDX', 'PBE_id', 'scISI', 'neuron_IDX', 'replay_epoch_id', 'visualization_raster_y_location', 'visualization_raster_emphasis_state']
            debug_datapoint_column_names = [spikes_df.spikes.time_variable_name, 'aclu', 'fragile_linear_neuron_IDX', 'visualization_raster_y_location'] # a subset I'm actually interested in for debugging
            active_datapoint_column_names = debug_datapoint_column_names # all values for the purpose of debugging
        else:
            default_datapoint_column_names = [spikes_df.spikes.time_variable_name, 'aclu', 'fragile_linear_neuron_IDX']
            active_datapoint_column_names = default_datapoint_column_names
            
        def _tip_fn(x, y, data):
            """ the function required by pg.ScatterPlotItem's `tip` argument to print the tooltip for each spike. """
            # data_string:str = '\n'.join([f"{k}:\t{str(v)}" for k, v in zip(active_datapoint_column_names, data)])
            # data_string:str = '\n'.join([f"{k}:\t{str(v)}" for k, v in asdict(data).items()])
            data_string:str = '|'.join([f"{k}: {str(v)}" for k, v in asdict(data).items()])
            print(f'_tip_fn(...): data_string: {data_string}')
            return f"spike: (x={x:.3f}, y={y:.2f})\n{data_string}"

        # spikes_data = spikes_df[active_datapoint_column_names].to_records(index=False).tolist() # list of tuples
        spikes_data = spikes_df[active_datapoint_column_names].to_dict('records') # list of dicts
        spikes_data = [ScatterItemData(**v) for v in spikes_data] 
        all_scatterplot_tooltips_kwargs = dict(data=spikes_data, tip=_tip_fn)
        assert len(all_scatterplot_tooltips_kwargs['data']) == np.shape(spikes_df)[0], f"if specified, all_scatterplot_tooltips_kwargs must be the same length as the number of spikes but np.shape(spikes_df)[0]: {np.shape(spikes_df)[0]} and len((all_scatterplot_tooltips_kwargs['data']): {len(all_scatterplot_tooltips_kwargs['data'])}"
    else:
        all_scatterplot_tooltips_kwargs = None
        
    # config_fragile_linear_neuron_IDX_map values are of the form: (i, fragile_linear_neuron_IDX, curr_pen, self._series_identity_lower_y_values[i], self._series_identity_upper_y_values[i])
    # Emphasis/Deemphasis-Dependent Pens:
    curr_spike_pens = [config_fragile_linear_neuron_IDX_map[a_fragile_linear_neuron_IDX][2][a_spike_emphasis_state] for a_fragile_linear_neuron_IDX, a_spike_emphasis_state in zip(filtered_spikes_df['fragile_linear_neuron_IDX'].to_numpy(), filtered_spikes_df['visualization_raster_emphasis_state'].to_numpy())] # get the pens for each spike from the configs map
    
    has_brushes: bool = np.all([(len(config_fragile_linear_neuron_IDX_map[a_fragile_linear_neuron_IDX])>=6) for a_fragile_linear_neuron_IDX in filtered_spikes_df['fragile_linear_neuron_IDX'].to_numpy()])
    if has_brushes:
        curr_spikes_brushes = [config_fragile_linear_neuron_IDX_map[a_fragile_linear_neuron_IDX][-1][a_spike_emphasis_state] for a_fragile_linear_neuron_IDX, a_spike_emphasis_state in zip(filtered_spikes_df['fragile_linear_neuron_IDX'].to_numpy(), filtered_spikes_df['visualization_raster_emphasis_state'].to_numpy())] # get the pens for each spike from the configs map
    else:
        curr_spikes_brushes = [] # scared to modify/use the brushes here out of fear of breaking the spike_emphasis_states
    
    curr_n = len(curr_spike_t) # curr number of spikes
    # builds the 'all_spots' tuples suitable for setting self.plots_data.all_spots from ALL Spikes
    pos = np.vstack((curr_spike_t, curr_spike_y))
    if has_brushes:
        all_spots = [{'pos': pos[:,i], 'data': i, 'pen': curr_spike_pens[i], 'brush': curr_spikes_brushes[i]} for i in range(curr_n)] # returned spikes {'pos','data','pen'}
    else:
        all_spots = [{'pos': pos[:,i], 'data': i, 'pen': curr_spike_pens[i]} for i in range(curr_n)] # returned spikes {'pos','data','pen'}
        
    if should_return_data_tooltips_kwargs:
        return all_spots, all_scatterplot_tooltips_kwargs
    else:
        return all_spots

