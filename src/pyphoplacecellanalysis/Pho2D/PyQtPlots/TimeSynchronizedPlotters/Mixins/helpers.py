import pyphoplacecellanalysis.External.pyqtgraph as pg # required for _temp_pyqtplot_plot_image_array


# ==================================================================================================================== #
# 🔜👁️‍🗨️ Merging TimeSynchronized Plotters:                                                                         #
# ==================================================================================================================== #
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedOccupancyPlotter import TimeSynchronizedOccupancyPlotter
from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPlacefieldsPlotter import TimeSynchronizedPlacefieldsPlotter
from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPositionDecoderPlotter import TimeSynchronizedPositionDecoderPlotter
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper



def build_combined_time_synchronized_plotters_window(active_pf_2D_dt, fixed_window_duration = 15.0, controlling_widget=None, create_new_controlling_widget=True):
    """ Builds a single window with time_synchronized (time-dependent placefield) plotters controlled by an internal 2DRasterPlot widget.
    
    Usage:
        active_pf_2D_dt.reset()
        active_pf_2D_dt.update(t=45.0, start_relative_t=True)
        all_plotters, root_dockAreaWindow, app = build_combined_time_synchronized_plotters_window(active_pf_2D_dt, fixed_window_duration = 15.0)
    """
    def _merge_plotters(a_controlling_widget, curr_sync_occupancy_plotter, curr_placefields_plotter, is_controlling_widget_external=False):
        # root_dockAreaWindow, app = DockAreaWrapper.wrap_with_dockAreaWindow(curr_sync_occupancy_plotter, spike_raster_plt_2d, title='All Time Synchronized Plotters')
        # curr_placefields_plotter, dDisplayItem = root_dockAreaWindow.add_display_dock(identifier='Time Dependent Placefields', widget=curr_placefields_plotter, dockAddLocationOpts=['left'])
        root_dockAreaWindow, app = DockAreaWrapper.wrap_with_dockAreaWindow(curr_sync_occupancy_plotter, curr_placefields_plotter, title='All Time Synchronized Plotters')
        
        if a_controlling_widget is not None:
            if not is_controlling_widget_external:
                a_controlling_widget, dDisplayItem = root_dockAreaWindow.add_display_dock(identifier='Time Dependent Placefields', widget=a_controlling_widget, dockAddLocationOpts=['bottom'])
        
        ## Register the children items as drivables/drivers:
        root_dockAreaWindow.connection_man.register_drivable(curr_sync_occupancy_plotter)
        root_dockAreaWindow.connection_man.register_drivable(curr_placefields_plotter)
        
        if a_controlling_widget is not None:
            root_dockAreaWindow.connection_man.register_driver(a_controlling_widget)
            # Wire up signals such that time-synchronized plotters are controlled by the RasterPlot2D:
            occupancy_raster_window_sync_connection = root_dockAreaWindow.connection_man.connect_drivable_to_driver(drivable=curr_sync_occupancy_plotter, driver=a_controlling_widget,
                                                                custom_connect_function=(lambda driver, drivable: pg.SignalProxy(driver.window_scrolled, delay=0.2, rateLimit=60, slot=drivable.on_window_changed_rate_limited)))
            placefields_raster_window_sync_connection = root_dockAreaWindow.connection_man.connect_drivable_to_driver(drivable=curr_placefields_plotter, driver=a_controlling_widget,
                                                                custom_connect_function=(lambda driver, drivable: pg.SignalProxy(driver.window_scrolled, delay=0.2, rateLimit=60, slot=drivable.on_window_changed_rate_limited)))
            
        return root_dockAreaWindow, app
    
    # Build the 2D Raster Plotter using a fixed window duration
    current_window_start_time = active_pf_2D_dt.last_t - fixed_window_duration
    
    if (controlling_widget is None):
        if create_new_controlling_widget:
            spike_raster_plt_2d = Spike2DRaster.init_from_independent_data(active_pf_2D_dt.all_time_filtered_spikes_df, window_duration=fixed_window_duration, window_start_time=current_window_start_time,
                                                                        neuron_colors=None, neuron_sort_order=None, application_name='TimeSynchronizedPlotterControlSpikeRaster2D',
                                                                        enable_independent_playback_controller=False, should_show=False, parent=None) # setting , parent=spike_raster_plt_3d makes a single window
            spike_raster_plt_2d.setWindowTitle('2D Raster Control Window')
            # Update the 2D Scroll Region to the initial value:
            spike_raster_plt_2d.update_scroll_window_region(current_window_start_time, active_pf_2D_dt.last_t, block_signals=False)
            controlling_widget = spike_raster_plt_2d
            is_controlling_widget_external = False
        else:
            print(f'WARNING: build_combined_time_synchronized_plotters_window(...) called with (controlling_widget == None) and (create_new_controlling_widget == False)')
            controlling_widget = None # no controlling widget
            is_controlling_widget_external = True
    else:
        # otherwise we have a controlling widget already
        controlling_widget = controlling_widget
        is_controlling_widget_external = True # external to window being created        
        
    curr_sync_occupancy_plotter = TimeSynchronizedOccupancyPlotter(active_pf_2D_dt)
    curr_placefields_plotter = TimeSynchronizedPlacefieldsPlotter(active_pf_2D_dt)
    
    root_dockAreaWindow, app = _merge_plotters(controlling_widget, curr_sync_occupancy_plotter, curr_placefields_plotter, is_controlling_widget_external=is_controlling_widget_external)
    return (controlling_widget, curr_sync_occupancy_plotter, curr_placefields_plotter), root_dockAreaWindow, app



def connect_time_synchronized_plotter(curr_plotter, sync_driver):
    # Control Placefields Plotter by spike_raster_window:
    return sync_driver.connection_man.connect_drivable_to_driver(drivable=curr_plotter, driver=sync_driver.spike_raster_plt_2d,
                                                       custom_connect_function=(lambda driver, drivable: pg.SignalProxy(driver.window_scrolled, delay=0.2, rateLimit=60, slot=drivable.on_window_changed_rate_limited)))
 
 
 
# def build_connected_time_synchronized_plotters_window(active_pf_2D_dt, sync_driver):
def build_connected_time_synchronized_occupancy_plotter(active_pf_2D_dt, sync_driver=None, should_defer_show=False):
    """ 
    sync_driver: spike_raster_window, 2DRaster, etc
    """
    curr_occupancy_plotter = TimeSynchronizedOccupancyPlotter(active_pf_2D_dt)
    if not should_defer_show:
        curr_occupancy_plotter.show()
    # Control Plotter by spike_raster_window:
    if sync_driver is not None:
        occupancy_raster_window_sync_connection = connect_time_synchronized_plotter(curr_occupancy_plotter, sync_driver)
    else:
        occupancy_raster_window_sync_connection = None
        
    return curr_occupancy_plotter, occupancy_raster_window_sync_connection

def build_connected_time_synchronized_placefields_plotter(active_pf_2D_dt, sync_driver=None, should_defer_show=False):
    """ 
    sync_driver: spike_raster_window, 2DRaster, etc
    """
    curr_placefields_plotter = TimeSynchronizedPlacefieldsPlotter(active_pf_2D_dt)
    if not should_defer_show:
        curr_placefields_plotter.show()
    # Control Plotter by spike_raster_window:
    if sync_driver is not None:
        # placefields_raster_window_sync_connection = sync_driver.connection_man.connect_drivable_to_driver(drivable=curr_placefields_plotter, driver=sync_driver.spike_raster_plt_2d,
                                                    #    custom_connect_function=(lambda driver, drivable: pg.SignalProxy(driver.window_scrolled, delay=0.2, rateLimit=60, slot=drivable.on_window_changed_rate_limited)))
        placefields_raster_window_sync_connection = connect_time_synchronized_plotter(curr_placefields_plotter, sync_driver)
    else:
        placefields_raster_window_sync_connection = None
        
    return curr_placefields_plotter, placefields_raster_window_sync_connection

        
        
def build_connected_time_synchronized_decoder_plotter(active_pf_2D_dt, sync_driver=None, should_defer_show=False):
    """ 
    sync_driver: spike_raster_window, 2DRaster, etc
    """
    curr_position_decoder_plotter = TimeSynchronizedPositionDecoderPlotter(active_one_step_decoder, active_two_step_decoder)
    if not should_defer_show:
        curr_position_decoder_plotter.show()
    # Control Plotter by spike_raster_window:
    if sync_driver is not None:
        decoder_raster_window_sync_connection = connect_time_synchronized_plotter(curr_position_decoder_plotter, sync_driver)
    else:
        decoder_raster_window_sync_connection = None
        
    return curr_position_decoder_plotter, decoder_raster_window_sync_connection

        