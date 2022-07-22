from copy import deepcopy
from pathlib import Path
import pandas as pd
import numpy as np
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

# ==================================================================================================================== #
# FIGURE/GRAPHICS EXPORT                                                                                               #
# ==================================================================================================================== #

import pyphoplacecellanalysis.External.pyqtgraph as pg
import pyphoplacecellanalysis.External.pyqtgraph.exporters
# import pyphoplacecellanalysis.External.pyqtgraph.widgets.GraphicsLayoutWidget
from pyphoplacecellanalysis.External.pyqtgraph.widgets.GraphicsView import GraphicsView

def export_pyqtgraph_plot(a_plot, debug_print=True):
    # create an exporter instance, as an argument give it
    # the item you wish to export    
    if isinstance(a_plot, GraphicsView):
        a_plot = a_plot.scene()
    else:
        a_plot = a_plot.plotItem
    exporter = pg.exporters.ImageExporter(a_plot)
    # set export parameters if needed
    # exporter.parameters()['width'] = 100   # (note this also affects height parameter)
    # save to file
    export_filepath = 'fileName.png'
    exporter.export(export_filepath)
    if debug_print:
        print(f'exported plot to {export_filepath}')


# ==================================================================================================================== #
# DATA EXPORT                                                                                                          #
# ==================================================================================================================== #

def get_default_pipeline_data_keys(active_config_name):
    if active_config_name is None:
        active_config_name = 'sess' # the default keys for no filter config are '/sess/spikes_df'
        
    return {'spikes_df': f'/filtered_sessions/{active_config_name}/spikes_df',
            'positions_df': f'/filtered_sessions/{active_config_name}/pos_df'
        }

        
def save_some_pipeline_data_to_h5(active_pipeline, included_session_identifiers=None, custom_key_prefix=None, finalized_output_cache_file='./pipeline_cache_store.h5', debug_print=False):
    """ 
    
    Inputs:
        included_session_identifiers: [] -  a list of session names to include in the output (e.g. ['maze','maze1','maze2']
       finalized_output_cache_file: str - a string specifying the prefix to prepend to each h5 key created, or None to use the default
        
    # Saves out ['/spikes_df', '/sess/spikes_df', '/filtered_sessions/maze2/spikes_df', '/filtered_sessions/maze1/spikes_df', '/filtered_sessions/maze/spikes_df'] to a .h5 file which can be loaded with
    # with pd.HDFStore(finalized_spike_df_cache_file) as store:
        # print(store.keys())
        # reread = pd.read_hdf(finalized_spike_df_cache_file, key='spikes_df')
        # reread
    Usage:
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import _test_save_pipeline_data_to_h5, get_h5_data_keys, save_some_pipeline_data_to_h5, load_pipeline_data_from_h5  #ExportHelpers
        finalized_output_cache_file='data/pipeline_cache_store.h5'
        output_save_result = save_some_pipeline_data_to_h5(curr_active_pipeline, finalized_output_cache_file=finalized_output_cache_file)
        output_save_result
        
        >> DynamicParameters({'finalized_output_cache_file': 'data/pipeline_cache_store.h5', 'sess': {'spikes_df': 'sess/spikes_df', 'pos_df': 'sess/pos_df'}, 'filtered_sessions/maze1': {'spikes_df': 'filtered_sessions/maze1/spikes_df', 'pos_df': 'filtered_sessions/maze1/pos_df'}, 'filtered_sessions/maze2': {'spikes_df': 'filtered_sessions/maze2/spikes_df', 'pos_df': 'filtered_sessions/maze2/pos_df'}, 'filtered_sessions/maze': {'spikes_df': 'filtered_sessions/maze/spikes_df', 'pos_df': 'filtered_sessions/maze/pos_df'}})

    
    Example: Loading Saved Dataframe:
        # Load the saved .h5 spikes dataframe for testing:
        finalized_spike_df_cache_file='./pipeline_cache_store.h5'
        desired_spikes_df_key = '/filtered_sessions/maze1/spikes_df'
        spikes_df = pd.read_hdf(finalized_spike_df_cache_file, key=desired_spikes_df_key)
        spikes_df
    """
    def _perform_save_cache_pipeline_data_to_h5(spikes_df, pos_df, sess_identifier_key='sess', finalized_output_cache_file='./pipeline_cache_store.h5'):
        """ 
            sess_identifier_key: str: like 'sess' or 'filtered_sessions/maze1'
        
        """
        # local_output_structure = output_structure.setdefault(sess_identifier_key, {})
        local_output_keys = get_default_pipeline_data_keys(sess_identifier_key)
        spikes_df.to_hdf(finalized_output_cache_file, key=f'{sess_identifier_key}/spikes_df')
        pos_df.to_hdf(finalized_output_cache_file, key=f'{sess_identifier_key}/pos_df', format='table')
        return sess_identifier_key, local_output_keys

    output_structure = DynamicParameters(finalized_output_cache_file=finalized_output_cache_file)
    
    if included_session_identifiers is None:
        included_session_identifiers = ['sess'] + active_pipeline.filtered_session_names
                            
        
    # Save out the non-filtered (sess) if desired:
    if 'sess' in included_session_identifiers:
        if custom_key_prefix is not None:
            curr_sess_identifier_key = '/'.join([custom_key_prefix, 'sess'])
        else:
            curr_sess_identifier_key = 'sess'
        local_sess_identifier_key, local_output_keys = _perform_save_cache_pipeline_data_to_h5(active_pipeline.sess.spikes_df, active_pipeline.sess.position.to_dataframe(), sess_identifier_key=curr_sess_identifier_key, finalized_output_cache_file=finalized_output_cache_file)
        output_structure[local_sess_identifier_key] = local_output_keys
    else:
        if debug_print:
            print("skipping 'sess' because it is not included in included_session_identifiers")
    
    for (a_key, a_filtered_session) in active_pipeline.filtered_sessions.items():
        if a_key in included_session_identifiers:
            if debug_print:
                print(f'a_filtered_session: {a_filtered_session}')
                
            # curr_sess_identifier_key = f'filtered_sessions/{a_key}'
            if custom_key_prefix is not None:
                curr_sess_identifier_key = '/'.join([custom_key_prefix, 'filtered_sessions', a_key])
            else:
                curr_sess_identifier_key = '/'.join(['filtered_sessions', a_key])
            
            local_sess_identifier_key, local_output_keys = _perform_save_cache_pipeline_data_to_h5(a_filtered_session.spikes_df, a_filtered_session.position.to_dataframe(), sess_identifier_key=curr_sess_identifier_key, finalized_output_cache_file=finalized_output_cache_file)
            output_structure[local_sess_identifier_key] = local_output_keys 
        else:
            if debug_print:
                print(f'skipping {a_key} because it is not included in included_session_identifiers')
    return output_structure


def load_pipeline_data_from_h5(finalized_output_cache_file, desired_spikes_df_key, desired_positions_df_key):
    """  Load the saved .h5 spikes dataframe for testing:
    
    Usage:
        desired_spikes_df_key = f'/filtered_sessions/{active_config_name}/spikes_df'
        desired_positions_df_key = f'/filtered_sessions/{active_config_name}/pos_df'    
        spikes_df, pos_df = load_pipeline_data_from_h5(finalized_output_cache_file=finalized_output_cache_file, desired_spikes_df_key=desired_spikes_df_key, desired_positions_df_key=desired_positions_df_key)

        spikes_df
        pos_df    
    """
    # Load the saved .h5 spikes dataframe for testing:
    spikes_df = pd.read_hdf(finalized_output_cache_file, key=desired_spikes_df_key)
    pos_df = pd.read_hdf(finalized_output_cache_file, key=desired_positions_df_key)
    # spikes_df
    # pos_df
    return spikes_df, pos_df



def get_h5_data_keys(finalized_output_cache_file, enable_debug_print=False):
    """ Returns the keys (variables) in the .h5 file
    Usage:
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import _test_save_pipeline_data_to_h5, get_h5_data_keys, save_some_pipeline_data_to_h5, load_pipeline_data_from_h5  #ExportHelpers
        finalized_output_cache_file='data/pipeline_cache_store.h5'
        out_keys = get_h5_data_keys(finalized_output_cache_file=finalized_output_cache_file)
        print(out_keys)
        >>> ['/spikes_df', '/sess/pos_df', '/sess/spikes_df', '/filtered_sessions/maze2/pos_df', '/filtered_sessions/maze2/spikes_df', '/filtered_sessions/maze2/pos_df/meta/values_block_2/meta', '/filtered_sessions/maze2/pos_df/meta/values_block_1/meta', '/filtered_sessions/maze1/pos_df', '/filtered_sessions/maze1/spikes_df', '/filtered_sessions/maze1/pos_df/meta/values_block_2/meta', '/filtered_sessions/maze1/pos_df/meta/values_block_1/meta', '/filtered_sessions/maze/pos_df', '/filtered_sessions/maze/spikes_df', '/filtered_sessions/maze/pos_df/meta/values_block_2/meta', '/filtered_sessions/maze/pos_df/meta/values_block_1/meta']
        
    """
    out_keys = None
    with pd.HDFStore(finalized_output_cache_file) as store:
        out_keys = store.keys()
        if enable_debug_print:
            print(out_keys)
    return out_keys


def _test_save_pipeline_data_to_h5(curr_active_pipeline, finalized_output_cache_file=None, data_output_directory=None, enable_dry_run=True, enable_debug_print=True):
    """ 
    
    Usage:
        finalized_output_cache_file = _test_save_pipeline_data_to_h5(curr_active_pipeline, finalized_output_cache_file=finalized_output_cache_file, enable_dry_run=False, enable_debug_print=True)
        finalized_output_cache_file

    """
    # Define Saving/Loading Directory and paths:
    if finalized_output_cache_file is None:
        if data_output_directory is None:
            data_output_directory = Path('./data')
        finalized_output_cache_file = data_output_directory.joinpath('pipeline_cache_store.h5') # '../../data/pipeline_cache_store.h5'

    if enable_debug_print:
        print(f'finalized_output_cache_file: "{str(finalized_output_cache_file)}"')

    curr_epoch_labels = list(curr_active_pipeline.sess.epochs.labels) # ['pre', 'maze1', 'post1', 'maze2', 'post2']
    curr_named_timeranges = [curr_active_pipeline.sess.epochs.get_named_timerange(a_label) for a_label in curr_epoch_labels]

    if enable_debug_print:
        print(f'curr_named_timeranges: {curr_named_timeranges}')
    

    all_filters_list = list(curr_active_pipeline.filtered_sessions.keys())
    if enable_debug_print:
        print(f'all_filters_list: {all_filters_list}')
        
    active_config_name = 'maze1'
    # active_config_name = 'maze'

    desired_spikes_df_key = f'/filtered_sessions/{active_config_name}/spikes_df'
    desired_positions_df_key = f'/filtered_sessions/{active_config_name}/pos_df'
    # desired_spikes_df_key = f'/filtered_sessions/{active_config_name}/spikes_df'

    if enable_debug_print:
        print(f'desired_spikes_df_key: "{desired_spikes_df_key}"')
        print(f'desired_positions_df_key: "{desired_positions_df_key}"')


    # Get relevant variables:
    # curr_active_pipeline is set above, and usable here
    sess = curr_active_pipeline.filtered_sessions[active_config_name]
    active_computed_data = curr_active_pipeline.computation_results[active_config_name].computed_data
    if enable_debug_print:
        print(f'active_computed_data.keys(): {active_computed_data.keys()}')
    
    pf = curr_active_pipeline.computation_results[active_config_name].computed_data['pf1D']
    active_one_step_decoder = curr_active_pipeline.computation_results[active_config_name].computed_data['pf2D_Decoder']
    active_two_step_decoder = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_TwoStepDecoder', None)
    active_measured_positions = curr_active_pipeline.computation_results[active_config_name].sess.position.to_dataframe()

    if not enable_dry_run:
        save_some_pipeline_data_to_h5(curr_active_pipeline, finalized_output_cache_file=finalized_output_cache_file)
    else:
        print(f'dry run only because enable_dry_run == True. No changes will be made.')
        print(f'final command would have been: save_some_pipeline_data_to_h5(curr_active_pipeline, finalized_output_cache_file="{finalized_output_cache_file}")')
    # save_spikes_data_to_h5(curr_active_pipeline, finalized_output_cache_file=finalized_output_cache_file)
    
    return finalized_output_cache_file