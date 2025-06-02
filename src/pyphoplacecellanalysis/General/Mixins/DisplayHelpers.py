from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.print_helpers import WrappingMessagePrinter
from pyphocorehelpers.plotting.mixins.figure_param_text_box import add_figure_text_box # for _display_add_computation_param_text_box

from pyphoplacecellanalysis.General.DataSeriesToSpatial import DataSeriesToSpatial # required for debug_print_axes_locations(...)

# Used by _display_2d_placefield_result_plot_ratemaps_2D
@function_attributes(short_name=None, tags=['save','figure','if_needed'], input_requires=[], output_provides=[], uses=[], used_by=['_display_2d_placefield_result_plot_ratemaps_2D'], creation_date='2023-05-11 19:55', related_items=[])
def _save_displayed_figure_if_needed(plotting_config, plot_type_name='plot', active_variant_name=None, active_figures=list(), debug_print=False):
    if active_variant_name is not None:
        active_plot_filename = '-'.join([plot_type_name, active_variant_name])
    else:
        active_plot_filename = plot_type_name
    active_plot_filepath = plotting_config.get_figure_save_path(active_plot_filename).with_suffix('.png')
    if debug_print:
        print(f'active_plot_filepath: {active_plot_filepath}')
    with WrappingMessagePrinter('Saving 2D Placefield image out to "{}"...'.format(active_plot_filepath), begin_line_ending='...', finished_message='done.'):
        for aFig in active_figures:
            aFig.savefig(active_plot_filepath)
    
    
# Post plotting figure helpers:
@function_attributes(short_name=None, tags=['save','figure','param','text','text_box'], input_requires=[], output_provides=[], uses=[], used_by=['_display_2d_placefield_result_plot_ratemaps_2D'], creation_date='2023-05-11 19:55', related_items=[])
def _display_add_computation_param_text_box(fig, computation_config, **kwargs):
    """ Adds a small box containing the computation parmaters to the matplotlib figure. 
    Usage:
        _display_add_computation_param_text_box(plt.gcf(), active_session_computation_config)
    """
    if fig is None:
        fig = plt.gcf()
    render_text = computation_config.str_for_attributes_list_display(key_val_sep_char=':', **kwargs)
    return add_figure_text_box(fig, render_text=render_text)
# used by _display_2d_placefield_result_plot_ratemaps_2D

""" 
TODO: EXPLORE: REVIEW: thse debug_print_* functions seem very useful and I didn't know they were here

"""


# ==================================================================================================================== #
# General                                                                                                              #
# ==================================================================================================================== #
@function_attributes(short_name=None, tags=['debug_print','debug','print','QRect'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-11 19:55', related_items=[])
def debug_print_QRect(rect, prefix_string='rect: ', indent_string = '\t', include_edge_positions=False):
    """ Prints QRectF in a more readible format
    By default printing QRectF objects results in output like 'PyQt5.QtCore.QRectF(57.847549828567, -0.007193522045074202, 15.76451934295443, 1.0150365839255244)'
        Which is in the format (x, y, width, height)

    Input:
        rect: QRectF
        include_edge_positions: Bool - If True, prints the relative edge positions like .left(), .right(), .top(), .bottom()
    Output:
        QRectF(x: 57.847549828567, y: -0.007193522045074202, width: 15.76451934295443, height: 1.0150365839255244)  
    """
    print(f'{indent_string}{prefix_string}QRectF(x: {rect.x()}, y: {rect.y()}, width: {rect.width()}, height: {rect.height()})') # Concise
    if include_edge_positions:
        print(f'{indent_string}{indent_string}left: {rect.left()}\t right: {rect.right()}')    
        print(f'{indent_string}{indent_string}top: {rect.top()}\t bottom: {rect.bottom()}')
    

def debug_widget_size_policy(a_size_policy):
    """Format a QSizePolicy with relevant information.
    
    Args:
        obj: QSizePolicy object
        p: IPython printer
        
    Returns:
        Formatted string representation
    """
    from PyQt5.QtWidgets import QSizePolicy

    general_policies = {QSizePolicy.Fixed: "Fixed", QSizePolicy.Minimum: "Minimum",
                QSizePolicy.Maximum: "Maximum", QSizePolicy.Preferred: "Preferred",
                QSizePolicy.Expanding: "Expanding", QSizePolicy.MinimumExpanding: "MinimumExpanding",
                QSizePolicy.Ignored: "Ignored"}

    h_policy_idx = a_size_policy.horizontalPolicy()
    v_policy_idx = a_size_policy.verticalPolicy()
    
    h_policy = general_policies.get(h_policy_idx, f"Unknown({h_policy_idx})")
    v_policy = general_policies.get(v_policy_idx, f"Unknown({v_policy_idx})")
                            
    print(f"QSizePolicy(horizontal={h_policy}, vertical={v_policy}, " \
            f"h_stretch={a_size_policy.horizontalStretch()}, v_stretch={a_size_policy.verticalStretch()})")
    


def debug_widget_size_adjust_policy(a_size_adjust_policy):
    """Format a QSizePolicy with relevant information.
    
    Args:
        obj: QSizePolicy object
        p: IPython printer
        
    Returns:
        Formatted string representation
    """
    from PyQt5.QtWidgets import QAbstractScrollArea

    

    general_policies = {QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored: "Ignored",
                        QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents: "ToContents",
                        QAbstractScrollArea.SizeAdjustPolicy.AdjustToContentsOnFirstShow: "ToContentsOnFirstShow",
    }   
    
    size_adjust_policy_str: str = general_policies.get(a_size_adjust_policy, f"Unknown({a_size_adjust_policy})")
    print(f"QAbstractScrollArea.SizeAdjustPolicy({size_adjust_policy_str})")
    

    

@function_attributes(short_name=None, tags=['debug_print', 'debug', 'print', 'pyqt5', 'pyqt', 'qwidget'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-04-04 06:45', related_items=[])
def debug_widget_geometry(a_widget, widget_name: Optional[str]=None, single_line_print: bool=False):
    """Print comprehensive debug information about a DockLabel to diagnose layout issues.
    
    from pyphoplacecellanalysis.General.Mixins.DisplayHelpers import debug_widget_geometry
    
    
    """
    def _perform_print(*args, **kwargs):
        """ captures single_line_print """
        if not single_line_print:
            print(*args, **kwargs)
        else:
            print(*args, kwargs.pop('end', '\t'), **kwargs)
        
    widget_type = type(a_widget)
    widget_object_name: Optional[str] = a_widget.objectName()
    if not widget_object_name:
        widget_object_name = None
        
    if widget_name is None:
        ## no name provided
        undefined_widget_name: str = 'Unnamed'
        widget_name = (widget_object_name or undefined_widget_name)
        
    if single_line_print:
        _perform_print('(')
        _perform_print(f"Geometry: {a_widget.geometry()}")
        _perform_print(f"Content rect: {a_widget.rect()}")
        _perform_print(f"Size hint: {a_widget.sizeHint()}")
        _perform_print(f"Minimum size hint: {a_widget.minimumSizeHint()}")
        
        # Size constraints
        _perform_print(f"Minimum size: {a_widget.minimumSize()}")
        _perform_print(f"Maximum size: {a_widget.maximumSize()}")
        _perform_print(f"Minimum width/height: {a_widget.minimumWidth()}, {a_widget.minimumHeight()}")
        _perform_print(f"Maximum width/height: {a_widget.maximumWidth()}, {a_widget.maximumHeight()}")
        
        # Layout related
        _perform_print(f"Base size: {a_widget.baseSize()}")
        _perform_print(f"Size increment: {a_widget.sizeIncrement()}")
        try:
            _perform_print(f"Contents rect: {a_widget.contentsRect()}")
            margins = a_widget.contentsMargins()
            _perform_print(f"Contents margins: left={margins.left()}, top={margins.top()}, right={margins.right()}, bottom={margins.bottom()}")
        except AttributeError:
            _perform_print("Contents rect/margins: Not available")
        
        # _perform_print(f"Layout direction: {a_widget.layoutDirection()}")
        
        # # Frame related
        # try:
        #     _perform_print(f"Frame geometry: {a_widget.frameGeometry()}")
        #     _perform_print(f"Frame size: {a_widget.frameSize()}")
        #     _perform_print(f"Frame width: {a_widget.frameWidth()}")
        # except AttributeError:
        #     _perform_print("Frame properties: Not available")
        
        # # Window related
        # try:
        #     _perform_print(f"Window frame geometry: {a_widget.windowFrameGeometry()}")
        #     _perform_print(f"Normal geometry: {a_widget.normalGeometry()}")
        # except AttributeError:
        #     _perform_print("Window frame properties: Not available")
            
        # Orientation and layout issues
        # _perform_print(f"Orientation: {a_widget.orientation}")
        # _perform_print(f"Dim state: {a_widget.dim}")
        _perform_print(f"Size policy:", end='\t') # {a_widget.sizePolicy().horizontalPolicy()}, {a_widget.sizePolicy().verticalPolicy()}
        debug_widget_size_policy(a_size_policy=a_widget.sizePolicy())
        

        try:
            a_size_adjust_policy = a_widget.sizeAdjustPolicy()
            _perform_print(f"Size Adjust policy:", end='\t')
            if a_size_adjust_policy is not None:
                debug_widget_size_adjust_policy(a_size_adjust_policy=a_size_adjust_policy)                
        except (AttributeError, KeyError) as e:
            # AttributeError: 'NestedDockAreaWidget' object has no attribute 'sizeAdjustPolicy'
            pass
        except Exception as e:
            raise
        
        
        # Visibility and enablement
        if not a_widget.isEnabled():
            _perform_print(f"Is enabled: {a_widget.isEnabled()}")
        if not a_widget.isVisible():
            _perform_print(f"Is visible: {a_widget.isVisible()}")
        
        _perform_print(')')
        
    else:
        ## multi-line output
        _widget_desc_str: str = f"\n--- {widget_type} Debug: '{widget_name}' ---"
        if (widget_object_name is not None) and (widget_object_name != widget_name):
            _widget_desc_str = f"{_widget_desc_str}\n.objectName(): '{widget_object_name}'"
            
        _perform_print(_widget_desc_str)
        # Basic geometry info
        _perform_print(f"Position: ({a_widget.x()}, {a_widget.y()})")
        _perform_print(f"Size: {a_widget.width()} × {a_widget.height()}")
        _perform_print(f"Geometry: {a_widget.geometry()}")
        _perform_print(f"Content rect: {a_widget.rect()}")
        _perform_print(f"Size hint: {a_widget.sizeHint()}")
        _perform_print(f"Minimum size hint: {a_widget.minimumSizeHint()}")
        
        # Size constraints
        _perform_print(f"Minimum size: {a_widget.minimumSize()}")
        _perform_print(f"Maximum size: {a_widget.maximumSize()}")
        _perform_print(f"Minimum width/height: {a_widget.minimumWidth()}, {a_widget.minimumHeight()}")
        _perform_print(f"Maximum width/height: {a_widget.maximumWidth()}, {a_widget.maximumHeight()}")
        
        # Layout related
        _perform_print(f"Base size: {a_widget.baseSize()}")
        _perform_print(f"Size increment: {a_widget.sizeIncrement()}")
        try:
            _perform_print(f"Contents rect: {a_widget.contentsRect()}")
            margins = a_widget.contentsMargins()
            _perform_print(f"Contents margins: left={margins.left()}, top={margins.top()}, right={margins.right()}, bottom={margins.bottom()}")
        except AttributeError:
            _perform_print("Contents rect/margins: Not available")
        
        _perform_print(f"Layout direction: {a_widget.layoutDirection()}")
        
        # Frame related
        try:
            _perform_print(f"Frame geometry: {a_widget.frameGeometry()}")
            _perform_print(f"Frame size: {a_widget.frameSize()}")
            _perform_print(f"Frame width: {a_widget.frameWidth()}")
        except AttributeError:
            _perform_print("Frame properties: Not available")
        
        # Window related
        try:
            _perform_print(f"Window frame geometry: {a_widget.windowFrameGeometry()}")
            _perform_print(f"Normal geometry: {a_widget.normalGeometry()}")
        except AttributeError:
            _perform_print("Window frame properties: Not available")
            
        # Orientation and layout issues
        # _perform_print(f"Orientation: {a_widget.orientation}")
        # _perform_print(f"Dim state: {a_widget.dim}")
        _perform_print(f"Size policy:", end='\t') # {a_widget.sizePolicy().horizontalPolicy()}, {a_widget.sizePolicy().verticalPolicy()}
        debug_widget_size_policy(a_size_policy=a_widget.sizePolicy())


        try:
            a_size_adjust_policy = a_widget.sizeAdjustPolicy()
            _perform_print(f"Size Adjust policy:", end='\t')
            if a_size_adjust_policy is not None:
                debug_widget_size_adjust_policy(a_size_adjust_policy=a_size_adjust_policy)                
        except (AttributeError, KeyError) as e:
            # AttributeError: 'NestedDockAreaWidget' object has no attribute 'sizeAdjustPolicy'
            pass
        except Exception as e:
            raise
        
        
        # Visibility and enablement
        _perform_print(f"Is visible: {a_widget.isVisible()}")
        _perform_print(f"Is enabled: {a_widget.isEnabled()}")
        _perform_print(f"Is shown: {not a_widget.isHidden()}")
        
        # Parent and layout context
        _perform_print(f"Parent type: {type(a_widget.parent()).__name__}")
        
        _perform_print("--- End Debug Info ---\n")
    



# ==================================================================================================================== #
# Specific                                                                                                             #
# ==================================================================================================================== #



@function_attributes(short_name=None, tags=['debug_print','debug','print','session'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-11 19:55', related_items=[])
def debug_print_identity_properties_from_session(curr_sess, debug_print=True):
    """ 
    Usage:
        curr_epoch_name = 'maze1'
        curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
        curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
        curr_spikes_df = curr_sess.spikes_df
        curr_computations_results = curr_active_pipeline.computation_results[curr_epoch_name]
        debug_print_identity_properties_from_session(curr_sess)
        
        >> OUTPUT >>:
            debug_print_identity_properties_from_session(curr_sess, ...): n_cells=40
                curr_map_keys: [ 2  3  4  5  7  8  9 10 11 12 14 17 18 21 22 23 24 25 26 27 28 29 33 34 38 39 42 44 45 46 47 48 53 55 57 58 61 62 63 64]
                curr_map_values: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
        

    """
    curr_map = curr_sess.neurons.reverse_cellID_index_map
    curr_map_keys = np.array(list(curr_map.keys()))
    curr_map_values = np.array(list(curr_map.values()))
    # print(len(curr_sess.neurons.reverse_cellID_index_map))
    # print(curr_sess.neurons.reverse_cellID_index_map)
    n_cells = len(curr_map_keys)
    if debug_print:
        print(f'debug_print_identity_properties_from_session(curr_sess, ...): n_cells={n_cells}')
        print(f'\t\t session: {curr_sess.get_description()}')
        print(f'\t\t curr_map_keys: {curr_map_keys}\n \t\t curr_map_values: {curr_map_values}')
    return n_cells, curr_map_keys, curr_map_values

@function_attributes(short_name=None, tags=['debug_print','debug','print','spikes', 'spikes_df'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-11 19:55', related_items=[])
def debug_print_identity_properties(spikes_df, debug_print=True):
    """ 
    Usage:
        curr_epoch_name = 'maze1'
        curr_epoch = curr_active_pipeline.filtered_epochs[curr_epoch_name] # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
        curr_sess = curr_active_pipeline.filtered_sessions[curr_epoch_name]
        curr_spikes_df = curr_sess.spikes_df
        curr_computations_results = curr_active_pipeline.computation_results[curr_epoch_name]
        debug_print_identity_properties(curr_spikes_df)
        
        >> OUTPUT >>:
            debug_print_identity_properties(spikes_df, ...): n_cells=40
                fragile_linear_neuron_IDXs: [ 0  1  2  3  5  6  7  8  9 10 12 15 16 19 20 21 22 23 24 25 26 27 31 32 36 37 40 42 43 44 45 46 51 53 55 56 59 60 61 62]
                cell_ids: [ 2  3  4  5  7  8  9 10 11 12 14 17 18 21 22 23 24 25 26 27 28 29 33 34 38 39 42 44 45 46 47 48 53 55 57 58 61 62 63 64]
            
    """
    
    fragile_linear_neuron_IDXs = np.unique(spikes_df['fragile_linear_neuron_IDX'].to_numpy())
    cell_ids = np.unique(spikes_df['aclu'].to_numpy())
    n_cells = len(fragile_linear_neuron_IDXs)
    if debug_print:
        print(f'debug_print_identity_properties(spikes_df, ...): n_cells={n_cells}')
        print(f'\t\t fragile_linear_neuron_IDXs: {fragile_linear_neuron_IDXs}\n \t\t cell_ids: {cell_ids}')
    return n_cells, fragile_linear_neuron_IDXs, cell_ids

@function_attributes(short_name=None, tags=['debug_print','debug','print','spike_raster_plt', 'axes'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-11 19:55', related_items=[])
def debug_print_axes_locations(spike_raster_plt):
    """ debugs the active and global (data) windows. 
    
    Requires the passed plotter (spike_raster_plt) has:
        spike_raster_plt.spikes_window
        spike_raster_plt.temporal_axis_length
        spike_raster_plt.params.center_mode
    
    Example Output:
        debug_print_axes_locations(...): Active Window/Local Properties:
            (active_t_start: 30.0, active_t_end: 45.0), active_window_t_duration: 15.0
            (active_x_start: 67.25698654867858, active_x_end: 198.3122106548942), active_x_length: 131.0552241062156
        debug_print_axes_locations(...): Global Data Properties:
            (global_start_t: 22.30206346133491, global_end_t: 1739.1355703625595), global_total_data_duration: 1716.8335069012246 (seconds)
            total_data_duration_minutes: 28.0
            (global_x_start: 0.0, global_x_end: 15000.0), global_total_x_length: 15000.0
        (30.0, 45.0, 15.0) (22.30206346133491, 1739.1355703625595, 1716.8335069012246) (67.25698654867858, 198.3122106548942, 131.0552241062156) (0.0, 15000.0, 15000.0)

            
    Example with assigning return values:
        (active_t_start, active_t_end, active_window_t_duration), (global_start_t, global_end_t, global_total_data_duration), (active_x_start, active_x_end, active_x_duration), (global_x_start, global_x_end, global_x_duration) = debug_print_axes_locations(spike_raster_plt_vedo)
        print((active_t_start, active_t_end, active_window_t_duration), (global_start_t, global_end_t, global_total_data_duration), (active_x_start, active_x_end, active_x_duration), (global_x_start, global_x_end, global_x_duration))

    """
    active_t_start, active_t_end = (spike_raster_plt.spikes_window.active_window_start_time, spike_raster_plt.spikes_window.active_window_end_time)
    active_window_t_duration = spike_raster_plt.spikes_window.window_duration
    print('debug_print_axes_locations(...): Active Window/Local Properties:')
    print(f'\t(active_t_start: {active_t_start}, active_t_end: {active_t_end}), active_window_t_duration: {active_window_t_duration}')
    active_x_start, active_x_end = DataSeriesToSpatial.temporal_to_spatial_map((active_t_start, active_t_end),
                                                                               spike_raster_plt.spikes_window.total_data_start_time, spike_raster_plt.spikes_window.total_data_end_time,
                                                                               spike_raster_plt.temporal_axis_length,
                                                                               center_mode=spike_raster_plt.params.center_mode)
    print(f'\t(active_x_start: {active_x_start}, active_x_end: {active_x_end}), active_x_length: {active_x_end - active_x_start}')

    # Global (all data)
    print('debug_print_axes_locations(...): Global (all data) Data Properties:')
    global_start_t, global_end_t = spike_raster_plt.spikes_window.total_df_start_end_times
    global_total_data_duration = global_end_t - global_start_t
    print(f'\t(global_start_t: {global_start_t}, global_end_t: {global_end_t}), global_total_data_duration: {global_total_data_duration} (seconds)')

    global_total_data_duration_minutes = np.floor_divide(global_total_data_duration, 60.0)
    print(f'\ttotal_data_duration_minutes: {global_total_data_duration_minutes}') # 28.0

    global_x_start, global_x_end = DataSeriesToSpatial.temporal_to_spatial_map((global_start_t, global_end_t),
                                                                               spike_raster_plt.spikes_window.total_data_start_time, spike_raster_plt.spikes_window.total_data_end_time, # spike_raster_plt_vedo.spikes_window.active_window_start_time, spike_raster_plt_vedo.spikes_window.active_window_end_time,
                                                                               spike_raster_plt.temporal_axis_length,
                                                                               center_mode=spike_raster_plt.params.center_mode)
    print(f'\t(global_x_start: {global_x_start}, global_x_end: {global_x_end}), global_total_x_length: {global_x_end - global_x_start}')
    # Return this complicated but exhaustive tuple of values:
    return ((active_t_start, active_t_end, active_window_t_duration), (global_start_t, global_end_t, global_total_data_duration), 
            (active_x_start, active_x_end, (active_x_end - active_x_start)), (global_x_start, global_x_end, (global_x_end - global_x_start)))

@function_attributes(short_name=None, tags=['debug_print','debug','print','app', 'info', 'qt', 'QApp'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-11 19:55', related_items=[])
def debug_print_app_info(qt_app):
    """ prints the informationa bout the active QtApp. 
    
    Usage:
        debug_print_app_info(spike_raster_plt_3d.app)
    
    """
    print(f'.app - memory address: {hex(id(qt_app))}\n.applicationName(): {qt_app.applicationName()}')
    if qt_app.objectName() != '':
        print(f'.objectName(): {qt_app.objectName()}')

@function_attributes(short_name=None, tags=['debug_print','debug','print','raster_plotter', 'info'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-11 19:55', related_items=[])
def debug_print_base_raster_plotter_info(raster_plotter):
    print(f'raster_plotter:\nmemory address: {hex(id(raster_plotter))}')
    debug_print_app_info(raster_plotter.app)
    print(f'.spikes_window address: {hex(id(raster_plotter.spikes_window))}')

@function_attributes(short_name=None, tags=['debug_print','debug','print','spikes', 'animation'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-11 19:55', related_items=[])
def _debug_print_spike_raster_window_animation_properties(spike_raster_window):
    """ dumps debug properties related to animation for a spike_raster_window
    Usage:
        from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget
        display_output = display_output | curr_active_pipeline.display('_display_spike_rasters_window', active_config_name, active_config_name=active_config_name)
        spike_raster_window = display_output['spike_raster_window']
        _debug_print_spike_raster_window_animation_properties(spike_raster_window) 
    
    Example Output:
        animation_active_time_window.window_duration: 0.0
        animation_time_step: 0.04
        animation_active_time_window.active_window_start_time: 7412.0642973107815
        render_window_duration: 0.0
        spike_raster_plt_2d.spikes_window.active_time_window: (7412.0642973107815, 7412.0642973107815)
        spike_raster_plt_2d.spikes_window.window_duration: 0.0
        spike_raster_plt_2d.ui.scroll_window_region
        min_x: 7412.0642973107815, max_x: 7412.0642973107815, x_duration: 0.0

    """
    print(f'\tanimation_active_time_window.window_duration: {spike_raster_window.animation_active_time_window.window_duration}')
    print(f'\tanimation_time_step: {spike_raster_window.animation_time_step}')
    print(f'\tanimation_active_time_window.active_window_start_time: {spike_raster_window.animation_active_time_window.active_window_start_time}')
    print(f'\trender_window_duration: {spike_raster_window.render_window_duration}')
    print(f'\tspike_raster_plt_2d.spikes_window.active_time_window: {spike_raster_window.spike_raster_plt_2d.spikes_window.active_time_window}') # (7455.820603311667, 7470.820603311667) start_t matches, but end_t does not! 
    print(f'\tspike_raster_plt_2d.spikes_window.window_duration: {spike_raster_window.spike_raster_plt_2d.spikes_window.window_duration}') # 15.0 -- This on the other hand, is not right... 
    min_x, max_x = spike_raster_window.spike_raster_plt_2d.ui.scroll_window_region.getRegion()
    x_duration = max_x - min_x
    print(f'\tspike_raster_plt_2d.ui.scroll_window_region\n\t\tmin_x: {min_x}, max_x: {max_x}, x_duration: {x_duration}') # min_x: 7455.820603311667, max_x: 7532.52160713601, x_duration: 76.70100382434339 -- NOTE: these are the real seconds!

@function_attributes(short_name=None, tags=['debug_print','debug','print','firing_rate', 'overview'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-11 19:55', related_items=[])
def debug_print_active_firing_rate_trends_result_overview(active_firing_rate_trends):
    """ 
    
    Usage:
        from pyphoplacecellanalysis.General.Mixins.DisplayHelpers import debug_print_active_firing_rate_trends_result_overview
        debug_print_active_firing_rate_trends_result_overview(active_firing_rate_trends)
    
    """
    def _print_single_result(a_trends_set):
        # a_trends_set['time_window_edges']
        # a_trends_set['time_window_edges_binning_info']
        binned_rates = a_trends_set['time_binned_unit_specific_binned_spike_rate']
        mins = a_trends_set['min_spike_rates'].to_numpy()
        means = binned_rates.mean().to_numpy()
        medians = a_trends_set['median_spike_rates'].to_numpy()
        maxs = a_trends_set['max_spike_rates'].to_numpy()
        # print(f"\t\tmins: {mins}") # all zero, which is reasonable I suppose
        print(f"\t\tmeans: {means}") # non-zero, which is good.
        # print(f"\t\tmedians: {medians}") # also all zero, which doesn't seem super reasonable
        print(f"\t\tmaxs: {maxs}")
    print(f"time_bin_size_seconds: {active_firing_rate_trends['time_bin_size_seconds']}")
    print(f"\tall_session_spikes: ")
    _print_single_result(active_firing_rate_trends['all_session_spikes'])
    print(f"\tpf_included_spikes_only: ")
    _print_single_result(active_firing_rate_trends['pf_included_spikes_only'])

@function_attributes(short_name=None, tags=['debug_print','debug','print','spikes', 'spikes_df'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-11 19:55', related_items=[])
def debug_print_spikes_df_column_info(spikes_df):
    """ Prints info about the spikes_df (spikes dataframe)
    Usage:
        from pyphoplacecellanalysis.General.Mixins.DisplayHelpers import debug_print_spikes_df_column_info
        _debug_print_spikes_df_column_info(curr_active_pipeline.sess.spikes_df)
        _debug_print_spikes_df_column_info(curr_active_pipeline.filtered_sessions['track'].spikes_df)
    """
    print(f'.columns: {list(spikes_df.columns)}')
    print(f'.spikes.time_variable_name: {spikes_df.spikes.time_variable_name}')    
    
    
    
# ==================================================================================================================== #
# Raster Plot Specific                                                                                                 #
# ==================================================================================================================== #

## NEED to figure out
    
@function_attributes(short_name=None, tags=['debug_print','debug','print','raster_plot', 'temporal','info'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-05-11 19:55', related_items=[])
def debug_print_temporal_info(active_2d_plot, prefix_string='active_2d_plot.', indent_string = '\t'):
    """ NOTE: used by Spike2DRaster's debug_print_spike_raster_timeline_alignments(...) function """
    print(f'{indent_string}{prefix_string}temporal_axis_length: {active_2d_plot.temporal_axis_length}')
    print(f'{indent_string}{prefix_string}temporal_zoom_factor: {active_2d_plot.temporal_zoom_factor}')
    print(f'{indent_string}{prefix_string}render_window_duration: {active_2d_plot.render_window_duration}')
    
    

