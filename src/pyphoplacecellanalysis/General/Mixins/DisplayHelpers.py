import matplotlib.pyplot as plt
import numpy as np

from pyphocorehelpers.print_helpers import WrappingMessagePrinter
from pyphocorehelpers.plotting.mixins.figure_param_text_box import add_figure_text_box # for _display_add_computation_param_text_box


# Used by _display_2d_placefield_result_plot_ratemaps_2D
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
def _display_add_computation_param_text_box(fig, computation_config):
    """ Adds a small box containing the computation parmaters to the matplotlib figure. 
    Usage:
        _display_add_computation_param_text_box(plt.gcf(), active_session_computation_config)
    """
    if fig is None:
        fig = plt.gcf()
    render_text = computation_config.str_for_attributes_list_display(key_val_sep_char=':')
    return add_figure_text_box(fig, render_text=render_text)
# used by _display_2d_placefield_result_plot_ratemaps_2D



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
        print(f'\t\t curr_map_keys: {curr_map_keys}\n \t\t curr_map_values: {curr_map_values}')
    return n_cells, curr_map_keys, curr_map_values

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
                unit_ids: [ 0  1  2  3  5  6  7  8  9 10 12 15 16 19 20 21 22 23 24 25 26 27 31 32 36 37 40 42 43 44 45 46 51 53 55 56 59 60 61 62]
                cell_ids: [ 2  3  4  5  7  8  9 10 11 12 14 17 18 21 22 23 24 25 26 27 28 29 33 34 38 39 42 44 45 46 47 48 53 55 57 58 61 62 63 64]
            
    """
    
    unit_ids = np.unique(spikes_df['unit_id'].to_numpy())
    cell_ids = np.unique(spikes_df['aclu'].to_numpy())
    n_cells = len(unit_ids)
    if debug_print:
        print(f'debug_print_identity_properties(spikes_df, ...): n_cells={n_cells}')
        print(f'\t\t unit_ids: {unit_ids}\n \t\t cell_ids: {cell_ids}')
    return n_cells, unit_ids, cell_ids
    
    
    