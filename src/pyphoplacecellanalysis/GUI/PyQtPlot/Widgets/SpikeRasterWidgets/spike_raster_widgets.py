# .SpikeRasterWidgets
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import numpy as np
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyphocorehelpers.gui.Qt.widget_positioning_helpers import WidgetPositioningHelpers
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.SpikeRasterBase import SpikeRasterBase
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster import Spike3DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster_Vedo import Spike3DRaster_Vedo

# for reandering in a single window:
from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget


# """ 
# Each separate call to Spikes3DRaster, Spikes2DRaster, etc shouldn't nec. create a whole new app. We want the ability for data such as the spikes_window to be shared between these windows.

# TimeWindowPlaybackController

# """
# class UnifiedSpikeRasterApp(TimeWindowPlaybackControllerActionsMixin, TimeWindowPlaybackPropertiesMixin, QtCore.QObject):
#     """ An attempt to make a singleton global app instance to hold the main window and synchronized playback controls and other global properties.
#         Currently Unused!
#     """
    
#     # TimeWindowPlaybackPropertiesMixin requirement:
#     @property
#     def animation_active_time_window(self):
#         """The accessor for the TimeWindowPlaybackPropertiesMixin class for the main active time window that it will animate."""
#         return self._spikes_window
    
#     # Get/Set Properties:
#     @property
#     def spikes_window(self):
#         """The spikes_window property."""
#         return self._spikes_window
#     @spikes_window.setter
#     def spikes_window(self, value):
#         self._spikes_window = value
    
#     def __init__(self, curr_spikes_df, core_app_name='UnifiedSpikeRasterApp', window_duration=15.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None):
#         super(UnifiedSpikeRasterApp, self).__init__() # QtCore.QObject.__init__(self)
        
#         # Set app name
#         self.name = core_app_name
        
#         self.params = VisualizationParameters('')
#         self._spikes_window = SpikesDataframeWindow(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time)
#         self.playback_controller = TimeWindowPlaybackController()
#         self.playback_controller.setup(self) # pass self to have properties set
        
        

def _get_required_static_layout_height(active_2d_plot) -> float:
    """ 
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.spike_raster_widgets import _get_required_static_layout_height
    
        required_static_children_bounding_rect_height: float = _get_required_static_layout_height(active_2d_plot=active_2d_plot)
        main_graphics_layout_widget.setMaximumHeight(required_static_children_bounding_rect_height)
        
    """
    import pyphoplacecellanalysis.External.pyqtgraph as pg
    root_layout: pg.GraphicsLayout = active_2d_plot.plots.background_static_scroll_window_plot.parentWidget()
    static_children_bounding_rect = root_layout.childrenBoundingRect() # QRectF
    required_static_children_bounding_rect_height: float = static_children_bounding_rect.height()
    return required_static_children_bounding_rect_height

    
@function_attributes(short_name=None, tags=['resize'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-06-02 14:29', related_items=[])
def _post_hoc_layout_resize(active_2d_plot, desired_static_area_height: Optional[float]=None):
    """ resizes the dynamic tracks to static area heights
    
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.spike_raster_widgets import _post_hoc_layout_resize
    
    _post_hoc_layout_resize(active_2d_plot=active_2d_plot, desired_static_area_height=249)
    
    """

    if active_2d_plot.ui.has_attr('main_content_splitter'):
        ## requires main_content_splitter, which is only used when `self.params.use_docked_pyqtgraph_plots` == False
        main_content_splitter = active_2d_plot.ui.main_content_splitter # QSplitter
        if active_2d_plot.ui.main_content_splitter is not None:
            if desired_static_area_height is None:
                required_static_children_bounding_rect_height: float = _get_required_static_layout_height(active_2d_plot=active_2d_plot)
                desired_static_area_height = required_static_children_bounding_rect_height
            else:
                ## use user provided
                print(f'desired_static_area_height: {desired_static_area_height}')
                
            ## INPUTS: main_content_splitter, desired_static_area_height
            original_sizes = np.array(main_content_splitter.sizes())
            extra_v_height = (original_sizes[-1] - desired_static_area_height)
            desired_sizes = deepcopy(original_sizes)
            desired_sizes[-1] = desired_static_area_height
            desired_sizes[0] = desired_sizes[0] + extra_v_height

            assert np.sum(desired_sizes) == np.sum(original_sizes), f"np.sum(desired_sizes): {np.sum(desired_sizes)} != np.sum(original_sizes): {np.sum(original_sizes)}"

            main_content_splitter.setSizes(desired_sizes.tolist())


@function_attributes(short_name=None, tags=['2024-12-18', 'ACTIVE', 'gui', 'debugging', 'continuous'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-18 19:29', related_items=[])
def _setup_spike_raster_window_for_debugging(spike_raster_window, wants_docked_raster_window_track:bool=False, enable_interval_overview_track:bool=False, allow_replace_hardcoded_main_plots_with_tracks: bool = False, debug_print=False, additional_post_hoc_fcns: Dict[str, Any]=None):
    """ Called to setup a specific `spike_raster_window` instance for 2024-12-18 style debugging.
    
    
    ['AddMatplotlibPlot.DecodedPosition', 'AddMatplotlibPlot.Custom',
     'AddTimeCurves.Position', 'AddTimeCurves.Velocity', 'AddTimeCurves.Random', 'AddTimeCurves.RelativeEntropySurprise', 'AddTimeCurves.Custom',
     'AddTimeIntervals.Laps', 'AddTimeIntervals.PBEs', 'AddTimeIntervals.SessionEpochs', 'AddTimeIntervals.Ripples', 'AddTimeIntervals.Replays', 'AddTimeIntervals.Bursts', 'AddTimeIntervals.Custom',
     'CreateNewConnectedWidget.NewConnected2DRaster', 'CreateNewConnectedWidget.NewConnected3DRaster.PyQtGraph', 'CreateNewConnectedWidget.NewConnected3DRaster.Vedo', 'CreateNewConnectedWidget.NewConnectedDataExplorer.ipc', 'CreateNewConnectedWidget.NewConnectedDataExplorer.ipspikes', 'CreateNewConnectedWidget.AddMatplotlibPlot.DecodedPosition', 'CreateNewConnectedWidget.Decoded_Epoch_Slices.Laps', 'CreateNewConnectedWidget.Decoded_Epoch_Slices.PBEs', 'CreateNewConnectedWidget.Decoded_Epoch_Slices.Ripple', 'CreateNewConnectedWidget.Decoded_Epoch_Slices.Replay', 'CreateNewConnectedWidget.Decoded_Epoch_Slices.Custom', 'CreateNewConnectedWidget.MenuCreateNewConnectedWidget', 'CreateNewConnectedWidget.MenuCreateNewConnectedDecodedEpochSlices',
     'Debug.MenuDebug', 'Debug.MenuDebugMenuActiveDrivers', 'Debug.MenuDebugMenuActiveDrivables', 'Debug.MenuDebugMenuActiveConnections',

     'DockedWidgets.NewDockedMatplotlibView', 'DockedWidgets.NewDockedContextNested', 'DockedWidgets.LongShortDecodedEpochsDockedMatplotlibView', 'DockedWidgets.DirectionalDecodedEpochsDockedMatplotlibView', 'DockedWidgets.Pseudo2DDecodedEpochsDockedMatplotlibView', 'DockedWidgets.ContinuousPseudo2DDecodedMarginalsDockedMatplotlibView', 'DockedWidgets.NewDockedCustom', 'DockedWidgets.AddDockedWidget']
     ['AddMatplotlibPlot.DecodedPosition', 'AddMatplotlibPlot.Custom', 'AddTimeCurves.Position', 'AddTimeCurves.Velocity', 'AddTimeCurves.Random', 'AddTimeCurves.RelativeEntropySurprise', 'AddTimeCurves.Custom', 'AddTimeIntervals.Laps', 'AddTimeIntervals.PBEs', 'AddTimeIntervals.SessionEpochs', 'AddTimeIntervals.Ripples', 'AddTimeIntervals.Replays', 'AddTimeIntervals.Bursts', 'AddTimeIntervals.Custom', 'CreateNewConnectedWidget.NewConnected2DRaster', 'CreateNewConnectedWidget.NewConnected3DRaster.PyQtGraph', 'CreateNewConnectedWidget.NewConnected3DRaster.Vedo', 'CreateNewConnectedWidget.NewConnectedDataExplorer.ipc', 'CreateNewConnectedWidget.NewConnectedDataExplorer.ipspikes', 'CreateNewConnectedWidget.AddMatplotlibPlot.DecodedPosition', 'CreateNewConnectedWidget.Decoded_Epoch_Slices.Laps', 'CreateNewConnectedWidget.Decoded_Epoch_Slices.PBEs', 'CreateNewConnectedWidget.Decoded_Epoch_Slices.Ripple', 'CreateNewConnectedWidget.Decoded_Epoch_Slices.Replay', 'CreateNewConnectedWidget.Decoded_Epoch_Slices.Custom', 'CreateNewConnectedWidget.MenuCreateNewConnectedWidget', 'CreateNewConnectedWidget.MenuCreateNewConnectedDecodedEpochSlices', 'Debug.MenuDebug', 'Debug.MenuDebugMenuActiveDrivers', 'Debug.MenuDebugMenuActiveDrivables', 'Debug.MenuDebugMenuActiveConnections', 'DockedWidgets.NewDockedMatplotlibView', 'DockedWidgets.LongShortDecodedEpochsDockedMatplotlibView', 'DockedWidgets.TrackTemplatesDecodedEpochsDockedMatplotlibView', 'DockedWidgets.DirectionalDecodedEpochsDockedMatplotlibView', 'DockedWidgets.Pseudo2DDecodedEpochsDockedMatplotlibView', 'DockedWidgets.ContinuousPseudo2DDecodedMarginalsDockedMatplotlibView', 'DockedWidgets.NewDockedCustom', 'DockedWidgets.MenuDockedWidgets', 'DockedWidgets.AddDockedWidget'
     uSAGE:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.spike_raster_widgets import _setup_spike_raster_window_for_debugging

        all_global_menus_actionsDict, global_flat_action_dict = _setup_spike_raster_window_for_debugging(spike_raster_window)

    """
    import pyphoplacecellanalysis.External.pyqtgraph as pg
    from PyQt5.QtWidgets import QAbstractScrollArea
    from PyQt5.QtWidgets import QSizePolicy

    is_docked_pyqtgraph_plots_mode: bool = spike_raster_window.params.use_docked_pyqtgraph_plots

    if additional_post_hoc_fcns is None:
        additional_post_hoc_fcns = {}

    _all_outputs_dict = {}
    
    omit_menu_item_names = ['Debug.MenuDebug', 'DockedWidgets.MenuDockedWidgets', ] # maybe , 'CreateNewConnectedWidget.MenuCreateNewConnectedWidget'
    all_global_menus_actionsDict, global_flat_action_dict = spike_raster_window.build_all_menus_actions_dict()
    if debug_print:
        print(list(global_flat_action_dict.keys()))


    ## extract the components so the `background_static_scroll_window_plot` scroll bar is the right size:
    active_2d_plot = spike_raster_window.spike_raster_plt_2d
    wrapper_layout: pg.QtWidgets.QVBoxLayout = active_2d_plot.ui.wrapper_layout
    layout = active_2d_plot.ui.layout
    
    _all_outputs_dict.update(**dict(#preview_overview_scatter_plot=preview_overview_scatter_plot, 
                                    #main_graphics_layout_widget=main_graphics_layout_widget, main_content_splitter=main_content_splitter, active_window_container_layout=active_window_container_layout,
                                    wrapper_layout=wrapper_layout, layout=layout))
    


    # Non-common (many of these used only in `is_docked_pyqtgraph_plots_mode == True` mode _______________________________________________________________________________________________________________________________________________________________________________________________ #
    
    # main_content_splitter = active_2d_plot.ui.main_content_splitter # QSplitter
    main_content_splitter = active_2d_plot.ui.get('main_content_splitter', None) # QSplitter
    if main_content_splitter is not None:
        _all_outputs_dict['main_content_splitter'] = main_content_splitter

    # main_graphics_layout_widget: pg.GraphicsLayoutWidget = active_2d_plot.ui.main_graphics_layout_widget
    main_graphics_layout_widget = active_2d_plot.ui.get('main_graphics_layout_widget', None) # GraphicsLayoutWidget
    if main_graphics_layout_widget is not None:
        _all_outputs_dict['main_graphics_layout_widget'] = main_graphics_layout_widget
        main_graphics_layout_widget.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContentsOnFirstShow)

    # active_window_container_layout = active_2d_plot.ui.active_window_container_layout
    active_window_container_layout = active_2d_plot.ui.get('active_window_container_layout', None) # QSplitter
    if active_window_container_layout is not None:
        _all_outputs_dict['active_window_container_layout'] = active_window_container_layout

    main_plot_widget = active_2d_plot.plots.get('main_plot_widget', None) # PlotItem
    if main_plot_widget is not None:
        _all_outputs_dict['main_plot_widget'] = main_plot_widget

    # preview_overview_scatter_plot: pg.ScatterPlotItem  = active_2d_plot.plots.preview_overview_scatter_plot # ScatterPlotItem
    # preview_overview_scatter_plot.setDownsampling(auto=True, method='subsample', dsRate=10)
    preview_overview_scatter_plot = active_2d_plot.plots.get('preview_overview_scatter_plot', None) # ScatterPlotItem
    if preview_overview_scatter_plot is not None:
        _all_outputs_dict['preview_overview_scatter_plot'] = preview_overview_scatter_plot
            

    background_static_scroll_plot_widget = active_2d_plot.plots.get('background_static_scroll_plot_widget', None) # PlotItem
    if background_static_scroll_plot_widget is not None:
        _all_outputs_dict['background_static_scroll_plot_widget'] = background_static_scroll_plot_widget
    
    background_static_scroll_window_plot = active_2d_plot.plots.get('background_static_scroll_window_plot', None) # PlotItem
    if background_static_scroll_window_plot is not None:
        _all_outputs_dict['background_static_scroll_window_plot'] = background_static_scroll_window_plot


    should_replace_hardcoded_main_plots_with_tracks: bool = False
    if (enable_interval_overview_track and wants_docked_raster_window_track and allow_replace_hardcoded_main_plots_with_tracks):
        should_replace_hardcoded_main_plots_with_tracks = True
        print(f'should_replace_hardcoded_main_plots_with_tracks: {should_replace_hardcoded_main_plots_with_tracks}')


    has_main_raster_plot: bool = (active_2d_plot.plots.main_plot_widget is not None)
    if has_main_raster_plot:
        main_plot_widget = active_2d_plot.plots.main_plot_widget # PlotItem
        if not should_replace_hardcoded_main_plots_with_tracks:
            main_plot_widget.setMinimumHeight(20.0)
        _all_outputs_dict['main_plot_widget'] = main_plot_widget
        
        if main_graphics_layout_widget is not None:
            main_graphics_layout_widget.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContentsOnFirstShow)                    
            required_static_children_bounding_rect_height: float = _get_required_static_layout_height(active_2d_plot=active_2d_plot)
            print(f'required_static_children_bounding_rect_height: {required_static_children_bounding_rect_height}')
            # main_graphics_layout_widget.setMaximumHeight(required_static_children_bounding_rect_height)

    else:
        active_window_container_layout.setVisible(False)


    # background_static_scroll_window_plot = active_2d_plot.plots.background_static_scroll_window_plot # PlotItem
    if background_static_scroll_window_plot:
        background_static_scroll_window_plot.setMinimumHeight(50.0)
        # background_static_scroll_window_plot.setMaximumHeight(75.0)
        # # background_static_scroll_window_plot.setFixedHeight(50.0)
        # background_static_scroll_window_plot.setMaximumHeight(144)
    # _all_outputs_dict['background_static_scroll_window_plot'] = background_static_scroll_window_plot

    if should_replace_hardcoded_main_plots_with_tracks:
        if active_window_container_layout:
            active_window_container_layout.setVisible(False) ## hide the container that contains the main_plot_widget
        if background_static_scroll_window_plot:
            background_static_scroll_window_plot.setMaximumHeight(144)


    # # Set stretch factors to control priority
    # main_graphics_layout_widget.ci.layout.setRowStretchFactor(0, 3)  # Plot1: lowest priority
    # main_graphics_layout_widget.ci.layout.setRowStretchFactor(1, 2)  # Plot2: mid priority
    # main_graphics_layout_widget.ci.layout.setRowStretchFactor(2, 2)  # Plot3: highest priority
    # main_graphics_layout_widget.ci.layout.setRowStretchFactor(3, 2)  # Plot3: highest priority

    _interval_tracks_out_dict = active_2d_plot.prepare_pyqtgraph_intervalPlot_tracks(enable_interval_overview_track=enable_interval_overview_track, should_link_to_main_plot_widget=has_main_raster_plot)
    _all_outputs_dict['_interval_tracks_out_dict'] = _interval_tracks_out_dict

    # interval_window_dock_config, interval_dock_item, intervals_time_sync_pyqtgraph_widget, intervals_root_graphics_layout_widget, intervals_plot_item = _interval_tracks_out_dict['intervals']
    # dock_config, interval_overview_dock_item, intervals_overview_time_sync_pyqtgraph_widget, intervals_overview_root_graphics_layout_widget, intervals_overview_plot_item = _interval_tracks_out_dict['interval_overview']

    if wants_docked_raster_window_track:
        _raster_tracks_out_dict = active_2d_plot.prepare_pyqtgraph_rasterPlot_track(name_modifier_suffix='raster_window', should_link_to_main_plot_widget=has_main_raster_plot)
        _all_outputs_dict['_raster_tracks_out_dict'] = _raster_tracks_out_dict




    # Add Renderables ____________________________________________________________________________________________________ #
    # add_renderables_menu = active_2d_plot.ui.menus.custom_context_menus.add_renderables[0].programmatic_actions_dict
    menu_commands = ['AddTimeIntervals.Replays', 'AddTimeIntervals.Laps', 'AddTimeIntervals.SessionEpochs', 'AddTimeIntervals.PBEs'] # , 'AddTimeIntervals.SessionEpochs', 'AddTimeIntervals.PBEs', 'AddTimeIntervals.Ripples',
    for a_command in menu_commands:
        assert a_command in global_flat_action_dict, f"a_command: '{a_command}' is not present in global_flat_action_dict: {list(global_flat_action_dict.keys())}"
        # add_renderables_menu[a_command].trigger()
        global_flat_action_dict[a_command].trigger()

    # # active_2d_plot.activeMenuReference
    # # active_2d_plot.ui.menus # .global_window_menus.docked_widgets.actions_dict

    active_2d_plot.params.enable_non_marginalized_raw_result = False
    active_2d_plot.params.enable_marginal_over_direction = False
    active_2d_plot.params.enable_marginal_over_track_ID = True


    menu_commands = [
        'AddTimeCurves.Position', ## 2025-03-11 02:32 Running this too soon after launching the window causes weird black bars on the top and bottom of the window
        'AddTimeCurves.ThetaPhase',
        # 'DockedWidgets.LongShortDecodedEpochsDockedMatplotlibView',
        # 'DockedWidgets.DirectionalDecodedEpochsDockedMatplotlibView',
        # 'DockedWidgets.TrackTemplatesDecodedEpochsDockedMatplotlibView',
        # 'DockedWidgets.Pseudo2DDecodedEpochsDockedMatplotlibView', # [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Qt/Menus/SpecificMenus/DockedWidgets_MenuProvider.py:141](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/Qt/Menus/SpecificMenus/DockedWidgets_MenuProvider.py:141)`'actionPseudo2DDecodedEpochsDockedMatplotlibView': AddNewDecodedPosteriors_MatplotlibPlotCommand`
        #  'DockedWidgets.ContinuousPseudo2DDecodedMarginalsDockedMatplotlibView',

    ]
    # menu_commands = ['actionPseudo2DDecodedEpochsDockedMatplotlibView', 'actionContinuousPseudo2DDecodedMarginalsDockedMatplotlibView'] # , 'AddTimeIntervals.SessionEpochs'
    # Run after a 0.5 second delay
    from PyQt5.QtCore import QTimer
    def trigger_commands():
        """ capturees: additional_post_hoc_fcns 
        """
        for a_command in menu_commands:
            # all_global_menus_actionsDict[a_command].trigger()
            global_flat_action_dict[a_command].trigger()

        for a_fn_name, a_fn in additional_post_hoc_fcns.items():
            print(f'running post-hoc fn: {a_fn_name}...')
            try:
                a_fn()
            except Exception as e:
                print(f'\terror {e} occurred while running "{a_fn_name}"')
                raise
    

    QTimer.singleShot(800, trigger_commands)
    # ## add the right sidebar
    # visible_intervals_info_widget_container, visible_intervals_ctrl_layout_widget =  spike_raster_window._perform_build_attached_visible_interval_info_widget() # builds the tables
    
    # spike_raster_window.build_epoch_intervals_visual_configs_widget()
    

    # ## Dock all Grouped results from `'DockedWidgets.Pseudo2DDecodedEpochsDockedMatplotlibView'`
    # ## INPUTS: active_2d_plot
    # nested_dock_items, nested_dynamic_docked_widget_container_widgets = active_2d_plot.ui.dynamic_docked_widget_container.layout_dockGroups()
    # grouped_dock_items_dict = active_2d_plot.ui.dynamic_docked_widget_container.get_dockGroup_dock_dict()
    # ## OUTPUTS: nested_dock_items, nested_dynamic_docked_widget_container_widgets


    _post_hoc_layout_resize(active_2d_plot=active_2d_plot, desired_static_area_height=144)


    return all_global_menus_actionsDict, global_flat_action_dict, _all_outputs_dict # , (_raster_tracks_out_dict, _raster_tracks_out_dict, _raster_tracks_out_dict)

        

        
def build_spike_3d_raster_with_2d_controls(curr_spikes_df, window_duration=15.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None, separate_windows=False, application_name=None):
    """ builds a 3D Raster plot for spikes with 2D controls in a separate window
    
    Inputs:
        separate_windows: bool - If True, the 3d plotter and its 2d controls are rendered in separate windows. Otherwise they're rendered in a single Spike3DRasterWindowWidget
    
    Usage:
    
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.spike_raster_widgets import build_spike_3d_raster_with_2d_controls
        curr_epoch_name = 'maze1'
        curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
        curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
        curr_spikes_df = curr_sess.spikes_df
        curr_computations_results = curr_active_pipeline.computation_results[curr_epoch_name]
        # Build the output widget:
        spike_raster_plt_3d, spike_raster_plt_2d, spike_3d_to_2d_window_connection = build_spike_3d_raster_with_2d_controls(curr_spikes_df)
        
    """
    if separate_windows:
        spike_raster_plt_3d = Spike3DRaster.init_from_independent_data(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=application_name)
        # Connect the 2D window scrolled signal to the 3D plot's spikes_window.update_window_start_end function
        spike_raster_plt_2d = Spike2DRaster.init_from_independent_data(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=application_name, parent=None) # setting , parent=spike_raster_plt_3d makes a single window
        spike_raster_plt_2d.setWindowTitle('2D Raster Control Window')
        spike_3d_to_2d_window_connection = spike_raster_plt_2d.window_scrolled.connect(spike_raster_plt_3d.spikes_window.update_window_start_end)
        spike_raster_plt_3d.disable_render_window_controls()
        # spike_raster_plt_3d.setWindowTitle('3D Raster with 2D Control Window')
        spike_raster_plt_3d.setWindowTitle('Main 3D Raster Window')
        WidgetPositioningHelpers.move_widget_to_top_left_corner(spike_raster_plt_3d, debug_print=False)
        WidgetPositioningHelpers.align_3d_and_2d_windows(spike_raster_plt_3d, spike_raster_plt_2d) # Align the two windows
        spike_raster_plt_2d.update_scroll_window_region(window_start_time, window_start_time+window_duration, block_signals=False) # TODO: ERROR: I know that these numbers are wrong, they should be in the 1000's. See spike_raster_plt_2d._fix_initial_linearRegionLocation(debug_print=True)
        spike_raster_window = None
    else:
        spike_raster_window = Spike3DRasterWindowWidget(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=application_name, type_of_3d_plotter='pyqtgraph')
        spike_raster_plt_2d = spike_raster_window.spike_raster_plt_2d
        spike_raster_plt_3d = spike_raster_window.spike_raster_plt_3d
        spike_3d_to_2d_window_connection = spike_raster_window.spike_3d_to_2d_window_connection
    
    return spike_raster_plt_3d, spike_raster_plt_2d, spike_3d_to_2d_window_connection, spike_raster_window



def build_spike_3d_raster_vedo_with_2d_controls(curr_spikes_df, window_duration=15.0, window_start_time=30.0, neuron_colors=None, neuron_sort_order=None, extant_spike_raster_plt_3d_vedo = None, separate_windows=False, application_name=None):
    """ builds a vedo-based 3D Raster plot for spikes with 2D controls in a separate window

    # NOTE: It appears this only works if the 2D Raster plot (pyqtgraph-based) is created before the Spike3DRaster_Vedo (Vedo-based). This is probably due to the pyqtgraph's instancing of the QtApplication.
    
    Usage:
    
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.spike_raster_widgets import build_spike_3d_raster_with_2d_controls
        curr_epoch_name = 'maze1'
        curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
        curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
        curr_spikes_df = curr_sess.spikes_df
        curr_computations_results = curr_active_pipeline.computation_results[curr_epoch_name]
        # Build the output widget:
        spike_raster_plt_3d_vedo, spike_raster_plt_2d, spike_3d_to_2d_window_connection = build_spike_3d_raster_vedo_with_2d_controls(curr_spikes_df)
    
    """
    # Build the 2D Raster Plotter
    spike_raster_plt_2d = Spike2DRaster.init_from_independent_data(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=application_name,  parent=None) # setting , parent=spike_raster_plt_3d makes a single window
    spike_raster_plt_2d.setWindowTitle('2D Raster Control Window')
    # Update the 2D Scroll Region to the initial value:
    spike_raster_plt_2d.update_scroll_window_region(window_start_time, window_start_time+window_duration, block_signals=False)
    

    # Build the 3D Vedo Raster plotter
    spike_raster_plt_3d_vedo = Spike3DRaster_Vedo.init_from_independent_data(curr_spikes_df, window_duration=window_duration, window_start_time=window_start_time, neuron_colors=neuron_colors, neuron_sort_order=neuron_sort_order, application_name=application_name)
    spike_raster_plt_3d_vedo.setWindowTitle('Main 3D (Vedo) Raster Window')
    spike_raster_plt_3d_vedo.disable_render_window_controls()
    # spike_raster_plt_3d.setWindowTitle('3D Raster with 2D Control Window')
    
    # Set the 3D Vedo plots' window to the current values of the 2d plot:
    spike_raster_plt_3d_vedo.spikes_window.update_window_start_end(spike_raster_plt_2d.spikes_window.active_time_window[0], spike_raster_plt_2d.spikes_window.active_time_window[1])

    # Connect the 2D window scrolled signal to the 3D plot's spikes_window.update_window_start_end function
    spike_3d_to_2d_window_connection = spike_raster_plt_2d.window_scrolled.connect(spike_raster_plt_3d_vedo.spikes_window.update_window_start_end)
    
    # Position the Windows As a Stack in the top-left corner:
    WidgetPositioningHelpers.move_widget_to_top_left_corner(spike_raster_plt_3d_vedo, debug_print=False)
    WidgetPositioningHelpers.align_3d_and_2d_windows(spike_raster_plt_3d_vedo, spike_raster_plt_2d) # Align the two windows
    
    # Update the scroll position programmatically with block_signals=False to ensure the 3D plot is synced:
    spike_raster_plt_2d.update_scroll_window_region(window_start_time, window_start_time+window_duration, block_signals=False)
    
    # Stand-in for future return value:
    spike_raster_window = None
    return spike_raster_plt_3d_vedo, spike_raster_plt_2d, spike_3d_to_2d_window_connection, spike_raster_window



# fd

