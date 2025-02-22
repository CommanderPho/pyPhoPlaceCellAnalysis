# 2024-01-29 - A version of "PendingNotebookCode" that is inside the pyphoplacecellanalysis library so that it can be imported from notebook that are not in the root of Spike3D
## This file serves as overflow from active Jupyter-lab notebooks, to eventually be refactored.
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re
from typing import List, Optional, Dict, Tuple, Any, Union
from matplotlib import cm, pyplot as plt
from matplotlib.gridspec import GridSpec
from neuropy.core import Laps, Position
from neuropy.core.user_annotations import UserAnnotationsManager
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.indexing_helpers import union_of_arrays
from neuropy.utils.result_context import IdentifyingContext
from nptyping import NDArray
import attrs
import matplotlib as mpl
import napari
from neuropy.core.epoch import Epoch, ensure_dataframe
from neuropy.analyses.placefields import HDF_SerializationMixin, PfND
import numpy as np
import pandas as pd
from attrs import asdict, astuple, define, field, Factory

from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import metadata_attributes

from functools import wraps, partial
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult


# from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing import NewType
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types
decoder_name: TypeAlias = str # a string that describes a decoder, such as 'LongLR' or 'ShortRL'
epoch_split_key: TypeAlias = str # a string that describes a split epoch, such as 'train' or 'test'
DecoderName = NewType('DecoderName', str)

import matplotlib.pyplot as plt
import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity

from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

from benedict import benedict
from neuropy.utils.mixins.indexing_helpers import get_dict_subset
from neuropy.utils.indexing_helpers import flatten_dict
from attrs import define, field, Factory, asdict # used for `ComputedResult`

from neuropy.utils.indexing_helpers import get_values_from_keypaths, set_value_by_keypath, update_nested_dict


# ==================================================================================================================== #
# 2025-02-14 - Drawing Final 2D Time Snapshots on Track                                                                #
# ==================================================================================================================== #




# ==================================================================================================================== #
# 2025-02-13 - Misc                                                                                                    #
# ==================================================================================================================== #

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, BayesianPlacemapPositionDecoder, DecodedFilterEpochsResult, Zhang_Two_Step
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsResult, TrackTemplates, TrainTestSplitResult
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin
from neuropy.utils.mixins.indexing_helpers import UnpackableMixin
from neuropy.utils.indexing_helpers import PandasHelpers

@function_attributes(short_name=None, tags=['UNFINISHED', 'plotting', 'computing'], input_requires=[], output_provides=[], uses=['_perform_plot_multi_decoder_meas_pred_position_track'], used_by=[], creation_date='2025-02-13 14:58', related_items=['_perform_plot_multi_decoder_meas_pred_position_track'])
def add_continuous_decoded_posterior(spike_raster_window, curr_active_pipeline, desired_time_bin_size: float, debug_print=True):
    """ computes the continuously decoded position posteriors (if needed) using the pipeline, then adds them as a new track to the SpikeRaster2D 
    
    """
    # ==================================================================================================================== #
    # COMPUTING                                                                                                            #
    # ==================================================================================================================== #
    
    # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'], computation_kwargs_list=[{'time_bin_size': 0.058}], #computation_kwargs_list=[{'time_bin_size': 0.025}], 
    #                                                   enabled_filter_names=None, fail_on_exception=True, debug_print=False)
    # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['merged_directional_placefields', 'directional_decoders_decode_continuous'], computation_kwargs_list=[{'laps_decoding_time_bin_size': 0.058}, {'time_bin_size': 0.058, 'should_disable_cache':False}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
    curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'], computation_kwargs_list=[{'time_bin_size': desired_time_bin_size, 'should_disable_cache': False}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
    

    # ==================================================================================================================== #
    # PLOTTING                                                                                                             #
    # ==================================================================================================================== #
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import AddNewDecodedPosteriors_MatplotlibPlotCommand

    display_output = {}
    AddNewDecodedPosteriors_MatplotlibPlotCommand(spike_raster_window, curr_active_pipeline, active_config_name=active_config_name, active_context=active_context, display_output=display_output, action_identifier='actionPseudo2DDecodedEpochsDockedMatplotlibView')

    all_global_menus_actionsDict, global_flat_action_dict = spike_raster_window.build_all_menus_actions_dict()
    if debug_print:
        print(list(global_flat_action_dict.keys()))


    ## extract the components so the `background_static_scroll_window_plot` scroll bar is the right size:
    active_2d_plot = spike_raster_window.spike_raster_plt_2d
    
    active_2d_plot.params.enable_non_marginalized_raw_result = False
    active_2d_plot.params.enable_marginal_over_direction = False
    active_2d_plot.params.enable_marginal_over_track_ID = True


    menu_commands = [
        'AddTimeCurves.Position',
        # 'DockedWidgets.LongShortDecodedEpochsDockedMatplotlibView',
        # 'DockedWidgets.DirectionalDecodedEpochsDockedMatplotlibView',
        # 'DockedWidgets.TrackTemplatesDecodedEpochsDockedMatplotlibView',
        'DockedWidgets.Pseudo2DDecodedEpochsDockedMatplotlibView',
        #  'DockedWidgets.ContinuousPseudo2DDecodedMarginalsDockedMatplotlibView',

    ]
    # menu_commands = ['actionPseudo2DDecodedEpochsDockedMatplotlibView', 'actionContinuousPseudo2DDecodedMarginalsDockedMatplotlibView'] # , 'AddTimeIntervals.SessionEpochs'
    for a_command in menu_commands:
        # all_global_menus_actionsDict[a_command].trigger()
        global_flat_action_dict[a_command].trigger()


    ## Dock all Grouped results from `'DockedWidgets.Pseudo2DDecodedEpochsDockedMatplotlibView'`
    ## INPUTS: active_2d_plot
    grouped_dock_items_dict = active_2d_plot.ui.dynamic_docked_widget_container.get_dockGroup_dock_dict()
    nested_dock_items = {}
    nested_dynamic_docked_widget_container_widgets = {}
    for dock_group_name, flat_group_dockitems_list in grouped_dock_items_dict.items():
        dDisplayItem, nested_dynamic_docked_widget_container = active_2d_plot.ui.dynamic_docked_widget_container.build_wrapping_nested_dock_area(flat_group_dockitems_list, dock_group_name=dock_group_name)
        nested_dock_items[dock_group_name] = dDisplayItem
        nested_dynamic_docked_widget_container_widgets[dock_group_name] = nested_dynamic_docked_widget_container

    ## OUTPUTS: nested_dock_items, nested_dynamic_docked_widget_container_widgets


    pass




# ==================================================================================================================== #
# 2025-02-11 - Subdivided Epochs                                                                                       #
# ==================================================================================================================== #
from neuropy.core.epoch import subdivide_epochs, ensure_dataframe, ensure_Epoch





# ==================================================================================================================== #
# 2025-01-27 - New Train/Test Splitting Results                                                                        #
# ==================================================================================================================== #

    

def _single_compute_train_test_split_epochs_decoders(a_decoder: BasePositionDecoder, a_config: Any, an_epoch_training_df: pd.DataFrame, an_epoch_test_df: pd.DataFrame, a_modern_name: str, training_test_suffixes = ['_train', '_test'], debug_print: bool = False): # , debug_output_hdf5_file_path=None, debug_plot: bool = False
    """ Replaces the config and updates/recomputes the computation epochs
    
    
        
    
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_train_test_split_epochs_decoders, _single_compute_train_test_split_epochs_decoders
    
    (a_training_test_split_epochs_df_dict, a_training_test_split_epochs_epoch_obj_dict), a_training_test_split_epochs_epoch_obj_dict, (an_epoch_period_description, a_config_copy, epoch_filtered_curr_pf1D, a_sliced_pf1D_Decoder) = _single_compute_train_test_split_epochs_decoders(a_1D_decoder, a_config, an_epoch_training_df, an_epoch_test_df, a_modern_name=a_modern_name, debug_print=debug_print)
    
    train_test_split_epochs_df_dict.update(a_training_test_split_epochs_df_dict)
    train_test_split_epoch_obj_dict.update(a_training_test_split_epochs_epoch_obj_dict)
    
    split_train_test_epoch_specific_configs[an_epoch_period_description] = a_config_copy
    split_train_test_epoch_specific_pf1D_dict[an_epoch_period_description] = lap_filtered_curr_pf1D
    split_train_test_epoch_specific_pf1D_Decoder_dict[an_epoch_period_description] = a_sliced_pf1D_Decoder
    
    """
    from nptyping import NDArray
    from neuropy.core.epoch import Epoch, ensure_dataframe
    from neuropy.analyses.placefields import PfND
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder

    ## INPUTS: training_data_portion, a_new_training_df, a_new_test_df
    # if debug_output_hdf5_file_path is not None:
    #     # Write out to HDF5 file:
    #     a_possible_hdf5_file_output_prefix: str = 'provided'
    #     a_prev_computation_epochs_df.to_hdf(debug_output_hdf5_file_path, f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/laps_df', format='table')
    #     an_epoch_training_df.to_hdf(debug_output_hdf5_file_path, f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/train_df', format='table')
    #     an_epoch_test_df.to_hdf(debug_output_hdf5_file_path, f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/test_df', format='table')

    #     _written_HDF5_manifest_keys.extend([f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/laps_df', f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/train_df', f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/test_df'])


    a_training_test_names = [f"{a_modern_name}{a_suffix}" for a_suffix in training_test_suffixes] # ['long_LR_train', 'long_LR_test']
    a_train_epoch_name: str = a_training_test_names[0] # just the train epochs, like 'long_LR_train'
    a_test_epoch_name: str = a_training_test_names[1] # just the test epochs, like 'long_LR_test'
    
    a_training_test_split_epochs_df_dict: Dict[str, pd.DataFrame] = dict(zip(a_training_test_names, (an_epoch_training_df, an_epoch_test_df))) # analagoues to `directional_laps_results.split_directional_laps_dict`

    # _temp_a_training_test_split_laps_valid_epoch_df_dict: Dict[str,Epoch] = {k:deepcopy(v).get_non_overlapping() for k, v in a_training_test_split_laps_df_dict.items()} ## NOTE: these lose the associated extra columns like 'lap_id', 'lap_dir', etc.
    a_training_test_split_epochs_epoch_obj_dict: Dict[str, Epoch] = {k:Epoch(deepcopy(v)).get_non_overlapping() for k, v in a_training_test_split_epochs_df_dict.items()} ## NOTE: these lose the associated extra columns like 'lap_id', 'lap_dir', etc.

    # a_valid_laps_training_df, a_valid_laps_test_df = ensure_dataframe(a_training_test_split_epochs_epoch_obj_dict[a_training_test_names[0]]), ensure_dataframe(a_training_test_split_epochs_epoch_obj_dict[a_training_test_names[1]])

    # if debug_output_hdf5_file_path is not None:
    #     # Write out to HDF5 file:
    #     a_possible_hdf5_file_output_prefix: str = 'valid'
    #     # a_laps_df.to_hdf(debug_output_hdf5_file_path, f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/laps_df', format='table')
    #     a_valid_laps_training_df.to_hdf(debug_output_hdf5_file_path, f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/train_df', format='table')
    #     a_valid_laps_test_df.to_hdf(debug_output_hdf5_file_path, f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/test_df', format='table')

    #     _written_HDF5_manifest_keys.extend([f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/train_df', f'{a_possible_hdf5_file_output_prefix}/{a_modern_name}/test_df'])


    # uses `a_modern_name`
    an_epoch_period_description: str = a_train_epoch_name
    curr_epoch_period_epoch_obj: Epoch = a_training_test_split_epochs_epoch_obj_dict[a_train_epoch_name]

    if a_config is not None:
        a_config_copy = deepcopy(a_config)
        a_config_copy['pf_params'].computation_epochs = curr_epoch_period_epoch_obj
    else:
        a_config_copy = None

    # curr_pf1D = a_1D_decoder.pf
    ## Restrict the PfNDs:
    # epoch_filtered_curr_pf1D: PfND = curr_pf1D.replacing_computation_epochs(deepcopy(curr_epoch_period_epoch_obj))

    ## apply the lap_filtered_curr_pf1D to the decoder:
    # a_sliced_pf1D_Decoder: BasePositionDecoder = BasePositionDecoder(pf=epoch_filtered_curr_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)    
    a_sliced_pf1D_Decoder: BasePositionDecoder = deepcopy(a_decoder).replacing_computation_epochs(epochs=deepcopy(curr_epoch_period_epoch_obj)) # BasePositionDecoder(pf=epoch_filtered_curr_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)
    epoch_filtered_curr_pf1D: PfND = a_sliced_pf1D_Decoder.pf
    
    ## ENDFOR a_modern_name in modern_names_list
    return (a_training_test_split_epochs_df_dict, a_training_test_split_epochs_epoch_obj_dict), (a_train_epoch_name, a_test_epoch_name), (an_epoch_period_description, a_config_copy, epoch_filtered_curr_pf1D, a_sliced_pf1D_Decoder)



@function_attributes(short_name=None, tags=['split', 'train-test'], input_requires=[], output_provides=[], uses=['split_laps_training_and_test'], used_by=['_split_train_test_laps_data'], creation_date='2025-01-27 22:14', related_items=[])
def compute_train_test_split_epochs_decoders(directional_laps_results: DirectionalLapsResult, track_templates: TrackTemplates, training_data_portion: float=5.0/6.0, debug_output_hdf5_file_path=None, debug_plot: bool = False, debug_print: bool = False) -> TrainTestSplitResult:
    """
    
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_train_test_split_epochs_decoders, _single_compute_train_test_split_epochs_decoders
    
    """
    from nptyping import NDArray
    from neuropy.core.epoch import Epoch, ensure_dataframe
    from neuropy.analyses.placefields import PfND
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrainTestLapsSplitting

    ## INPUTS: training_data_portion, a_new_training_df, a_new_test_df

    ## INPUTS: directional_laps_results, track_templates, directional_laps_results, debug_plot=False
    test_data_portion: float = 1.0 - training_data_portion # test data portion is 1/6 of the total duration

    if debug_print:
        print(f'training_data_portion: {training_data_portion}, test_data_portion: {test_data_portion}')


    decoders_dict = deepcopy(track_templates.get_decoders_dict())


    # Converting between decoder names and filtered epoch names:
    # {'long':'maze1', 'short':'maze2'}
    # {'LR':'odd', 'RL':'even'}
    long_LR_name, short_LR_name, long_RL_name, short_RL_name = ['maze1_odd', 'maze2_odd', 'maze1_even', 'maze2_even']
    decoder_name_to_session_context_name: Dict[str,str] = dict(zip(track_templates.get_decoder_names(), (long_LR_name, long_RL_name, short_LR_name, short_RL_name))) # {'long_LR': 'maze1_odd', 'long_RL': 'maze1_even', 'short_LR': 'maze2_odd', 'short_RL': 'maze2_even'}
    # session_context_name_to_decoder_name: Dict[str,str] = dict(zip((long_LR_name, long_RL_name, short_LR_name, short_RL_name), track_templates.get_decoder_names())) # {'maze1_odd': 'long_LR', 'maze1_even': 'long_RL', 'maze2_odd': 'short_LR', 'maze2_even': 'short_RL'}
    old_directional_names = list(directional_laps_results.directional_lap_specific_configs.keys()) #['maze1_odd', 'maze1_even', 'maze2_odd', 'maze2_even']
    modern_names_list = list(decoders_dict.keys()) # ['long_LR', 'long_RL', 'short_LR', 'short_RL']
    assert len(old_directional_names) == len(modern_names_list), f"old_directional_names: {old_directional_names} length is not equal to modern_names_list: {modern_names_list}"

    # lap_dir_keys = ['LR', 'RL']
    # maze_id_keys = ['long', 'short']
    training_test_suffixes = ['_train', '_test'] ## used in loop

    _written_HDF5_manifest_keys = []

    train_test_split_epochs_df_dict: Dict[str, pd.DataFrame] = {} # analagoues to `directional_laps_results.split_directional_laps_dict`
    train_test_split_epoch_obj_dict: Dict[str, Epoch] = {}

    ## Per-Period Outputs
    split_train_test_epoch_specific_configs = {}
    split_train_test_epoch_specific_pfND_dict = {} # analagous to `all_directional_decoder_dict` (despite `all_directional_decoder_dict` having an incorrect name, it's actually pfs)
    split_train_test_epoch_specific_pfND_Decoder_dict = {}

    for a_modern_name in modern_names_list:
        ## Loop through each decoder:
        old_directional_lap_name: str = decoder_name_to_session_context_name[a_modern_name] # e.g. 'maze1_even'
        if debug_print:
            print(f'a_modern_name: {a_modern_name}, old_directional_lap_name: {old_directional_lap_name}')
        a_1D_decoder: BasePositionDecoder = deepcopy(decoders_dict[a_modern_name])

        # directional_laps_results # DirectionalLapsResult
        a_config = deepcopy(directional_laps_results.directional_lap_specific_configs[old_directional_lap_name])
        # type(a_config) # DynamicContainer

        # type(a_config['pf_params'].computation_epochs) # Epoch
        # a_config['pf_params'].computation_epochs
        a_prev_computation_epochs_df: pd.DataFrame = ensure_dataframe(deepcopy(a_config['pf_params'].computation_epochs))
        # ensure non-overlapping first:
        an_epoch_training_df, an_epoch_test_df = a_prev_computation_epochs_df.epochs.split_into_training_and_test(training_data_portion=training_data_portion, group_column_name ='lap_id', additional_epoch_identity_column_names=['label', 'lap_id', 'lap_dir'], skip_get_non_overlapping=False, debug_print=False) # a_laps_training_df, a_laps_test_df both comeback good here.
        

        (a_training_test_split_epochs_df_dict, a_training_test_split_epochs_epoch_obj_dict), a_training_test_split_epochs_epoch_obj_dict, (an_epoch_period_description, a_config_copy, epoch_filtered_curr_pf1D, a_sliced_pf1D_Decoder) = _single_compute_train_test_split_epochs_decoders(a_decoder=a_1D_decoder, a_config=a_config,
                                                                                                                                                                                                                                                                                            an_epoch_training_df=an_epoch_training_df, an_epoch_test_df=an_epoch_test_df,
                                                                                                                                                                                                                                                                                             a_modern_name=a_modern_name, debug_print=debug_print)
        train_test_split_epochs_df_dict.update(a_training_test_split_epochs_df_dict)
        train_test_split_epoch_obj_dict.update(a_training_test_split_epochs_epoch_obj_dict)
        
        split_train_test_epoch_specific_configs[an_epoch_period_description] = a_config_copy
        split_train_test_epoch_specific_pfND_dict[an_epoch_period_description] = epoch_filtered_curr_pf1D
        split_train_test_epoch_specific_pfND_Decoder_dict[an_epoch_period_description] = a_sliced_pf1D_Decoder
        


    ## ENDFOR a_modern_name in modern_names_list
        
    if debug_print:
        print(list(split_train_test_epoch_specific_pfND_Decoder_dict.keys())) # ['long_LR_train', 'long_RL_train', 'short_LR_train', 'short_RL_train']

    ## OUTPUTS: (train_test_split_laps_df_dict, train_test_split_laps_epoch_obj_dict), (split_train_test_lap_specific_pf1D_Decoder_dict, split_train_test_lap_specific_pf1D_dict, split_train_test_lap_specific_configs)

    ## Get test epochs:
    train_epoch_names: List[str] = [k for k in train_test_split_epochs_df_dict.keys() if k.endswith('_train')] # ['long_LR_train', 'long_RL_train', 'short_LR_train', 'short_RL_train']
    test_epoch_names: List[str] = [k for k in train_test_split_epochs_df_dict.keys() if k.endswith('_test')] # ['long_LR_test', 'long_RL_test', 'short_LR_test', 'short_RL_test']

    ## train_test_split_laps_df_dict['long_LR_test'] != train_test_split_laps_df_dict['long_LR_train'], which is correct
    ## Only the decoders built with the training epochs make any sense:
    train_lap_specific_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = {k.split('_train', maxsplit=1)[0]:split_train_test_epoch_specific_pfND_Decoder_dict[k] for k in train_epoch_names} # the `k.split('_train', maxsplit=1)[0]` part just gets the original key like 'long_LR'

    # DF mode so they don't lose the associated info:
    test_epochs_dict: Dict[str, pd.DataFrame] = {k.split('_test', maxsplit=1)[0]:v for k,v in train_test_split_epochs_df_dict.items() if k.endswith('_test')} # the `k.split('_test', maxsplit=1)[0]` part just gets the original key like 'long_LR'
    train_epochs_dict: Dict[str, pd.DataFrame] = {k.split('_train', maxsplit=1)[0]:v for k,v in train_test_split_epochs_df_dict.items() if k.endswith('_train')} # the `k.split('_train', maxsplit=1)[0]` part just gets the original key like 'long_LR'

    ## Now decode the test epochs using the new decoders:
    # ## INPUTS: global_spikes_df, train_lap_specific_pf1D_Decoder_dict, test_epochs_dict, laps_decoding_time_bin_size
    # global_spikes_df = get_proper_global_spikes_df(curr_active_pipeline)
    # test_laps_decoder_results_dict = decode_using_new_decoders(global_spikes_df, train_lap_specific_pf1D_Decoder_dict, test_epochs_dict, laps_decoding_time_bin_size)
    # test_laps_decoder_results_dict
        
    a_train_test_result: TrainTestSplitResult = TrainTestSplitResult(is_global=True, training_data_portion=training_data_portion, test_data_portion=test_data_portion,
                            test_epochs_dict=test_epochs_dict, train_epochs_dict=train_epochs_dict,
                            train_lap_specific_pf1D_Decoder_dict=train_lap_specific_pf1D_Decoder_dict)
    return a_train_test_result




# ==================================================================================================================== #
# 2025-01-21 - Bin-by-bin decoding examples                                                                            #
# ==================================================================================================================== #
import numpy as np
import pyqtgraph as pg
from copy import deepcopy
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.print_helpers import strip_type_str_to_classname
from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult, TrainTestSplitResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df

# @define(slots=False, eq=False)
@metadata_attributes(short_name=None, tags=['pyqtgraph'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-24 17:22', related_items=[])
class BinByBinDecodingDebugger:
    """ handles displaying the process of debugging decoding for each time bin
    
    Usage:    
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import BinByBinDecodingDebugger    
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import LayoutScrollability, pyqtplot_build_image_bounds_extent
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import BaseTemplateDebuggingMixin, build_pf1D_heatmap_with_labels_and_peaks, TrackTemplates
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import BinByBinDecodingDebugger

        # Example usage:
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping() 
        global_laps_epochs_df = global_laps.to_dataframe()
        global_laps_epochs_df

        ## INPUTS: 
        time_bin_size: float = 0.250
        a_lap_id: int = 9
        a_decoder_name = 'long_LR'

        ## COMPUTED: 
        a_decoder_idx: int = track_templates.get_decoder_names().index(a_decoder_name)
        a_decoder = deepcopy(track_templates.long_LR_decoder)
        (_out_decoded_time_bin_edges, _out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists, _out_decoded_active_p_x_given_n, _out_decoded_active_plots_data) = BinByBinDecodingDebugger.build_spike_counts_and_decoder_outputs(track_templates=track_templates, global_laps_epochs_df=global_laps_epochs_df, spikes_df=global_spikes_df, a_decoder_name=a_decoder_name, time_bin_size=time_bin_size)
        win, out_pf1D_decoder_template_objects, (_out_decoded_active_plots, _out_decoded_active_plots_data) = BinByBinDecodingDebugger.build_time_binned_decoder_debug_plots(a_decoder=a_decoder, a_lap_id=a_lap_id, _out_decoded_time_bin_edges=_out_decoded_time_bin_edges, _out_decoded_active_p_x_given_n=_out_decoded_active_p_x_given_n, _out_decoded_unit_specific_time_binned_spike_counts=_out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists=_out_decoded_active_unit_lists, _out_decoded_active_plots_data=_out_decoded_active_plots_data, debug_print=True)
        print(f"Returned window: {win}")
        print(f"Returned decoder objects: {out_pf1D_decoder_template_objects}")


    """
    plots: RenderPlots = field(repr=keys_only_repr)
    plots_data: RenderPlotsData = field(repr=keys_only_repr)
    ui: PhoUIContainer = field(repr=keys_only_repr)
    params: VisualizationParameters = field(repr=keys_only_repr)

    # time_bin_size: float = field(default=0.500) # 500ms
    # spikes_df: pd.DataFrame = field()
    # global_laps_epochs_df: pd.DataFrame = field()
    @classmethod
    def build_spike_counts_and_decoder_outputs(cls, track_templates, global_laps_epochs_df, spikes_df, a_decoder_name='long_LR', time_bin_size=0.500, debug_print=False):
        """
            a_decoder_name: types.DecoderName = 'long_LR'
            
        """
        ## Get a specific decoder
        
        a_decoder_idx: int = track_templates.get_decoder_names().index(a_decoder_name)
        a_decoder = deepcopy(track_templates.long_LR_decoder)

        neuron_IDs = deepcopy(a_decoder.neuron_IDs)
        spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(neuron_IDs)
        unique_units = np.unique(spikes_df['aclu'])
        _out_decoded_time_bin_edges = {}
        _out_decoded_unit_specific_time_binned_spike_counts = {}
        _out_decoded_active_unit_lists = {}
        _out_decoded_active_p_x_given_n = {}
        _out_decoded_active_plots_data: Dict[str, RenderPlotsData]  = {}

        for a_row in global_laps_epochs_df.itertuples():
            t_start = a_row.start
            t_end = a_row.stop
            time_bin_edges = np.arange(t_start, (t_end + time_bin_size), time_bin_size)
            n_time_bins = len(time_bin_edges) - 1
            assert n_time_bins > 0
            _out_decoded_time_bin_edges[a_row.lap_id] = time_bin_edges
            unit_specific_time_binned_spike_counts = np.array([
                np.histogram(spikes_df.loc[spikes_df['aclu'] == unit, 't_rel_seconds'], bins=time_bin_edges)[0]
                for unit in unique_units
            ])
            all_lap_active_units_list = []
            active_units_list = []
            for a_time_bin_idx in np.arange(n_time_bins):
                unit_spike_counts = np.squeeze(unit_specific_time_binned_spike_counts[:, a_time_bin_idx])
                normalized_unit_spike_counts = (unit_spike_counts / np.sum(unit_spike_counts))
                active_unit_idxs = np.where(unit_spike_counts > 0)[0]
                active_units = neuron_IDs[active_unit_idxs]
                active_aclu_spike_counts_dict = dict(zip(active_units, unit_spike_counts[active_unit_idxs]))
                active_units_list.append(active_aclu_spike_counts_dict)
                all_lap_active_units_list.extend(active_units)

            _out_decoded_active_unit_lists[a_row.lap_id] = active_units_list
            _out_decoded_unit_specific_time_binned_spike_counts[a_row.lap_id] = unit_specific_time_binned_spike_counts

            # all_lap_active_units_list = np.unique(list(Set(all_lap_active_units_list)))
            all_lap_active_units_list = np.unique(all_lap_active_units_list)
            if debug_print:
                print(f'all_lap_active_units_list: {all_lap_active_units_list}')
            lap_specific_spikes_df = deepcopy(spikes_df).spikes.time_sliced(t_start=t_start, t_stop=t_end).spikes.sliced_by_neuron_id(all_lap_active_units_list)
            lap_specific_spikes_df, neuron_id_to_new_IDX_map = lap_specific_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()  # rebuild the fragile indicies afterwards
            _decoded_pos_outputs = a_decoder.decode(unit_specific_time_binned_spike_counts=unit_specific_time_binned_spike_counts, time_bin_size=time_bin_size, output_flat_versions=True, debug_print=False)
            _out_decoded_active_p_x_given_n[a_row.lap_id] = _decoded_pos_outputs
            _out_decoded_active_plots_data[a_row.lap_id] = RenderPlotsData(name=f'lap[{a_row.lap_id}]', spikes_df=lap_specific_spikes_df, active_aclus=all_lap_active_units_list)

        return (_out_decoded_time_bin_edges, _out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists, _out_decoded_active_p_x_given_n, _out_decoded_active_plots_data)


    @classmethod
    def build_time_binned_decoder_debug_plots(cls, a_decoder, a_lap_id, _out_decoded_time_bin_edges, _out_decoded_active_p_x_given_n, _out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists, _out_decoded_active_plots_data, debug_print=False):
        """ builds the plots 
        """
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import pyqtplot_build_image_bounds_extent
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import BaseTemplateDebuggingMixin
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import new_plot_raster_plot, NewSimpleRaster

        def _simply_plot_posterior_in_pyqtgraph_plotitem(curr_plot, image, xbin_edges, ybin_edges):
            pg.setConfigOptions(imageAxisOrder='row-major')
            pg.setConfigOptions(antialias=True)
            cmap = pg.colormap.get('jet','matplotlib')
            image_bounds_extent, x_range, y_range = pyqtplot_build_image_bounds_extent(xbin_edges, ybin_edges, margin=0.0, debug_print=debug_print)
            curr_plot.hideButtons()
            img_item = pg.ImageItem(image=image, levels=(0,1))
            curr_plot.addItem(img_item, defaultPadding=0.0)
            img_item.setImage(image, rect=image_bounds_extent, autoLevels=False)
            img_item.setLookupTable(cmap.getLookupTable(nPts=256), update=False)
            curr_plot.setRange(xRange=x_range, yRange=y_range, padding=0.0, update=False, disableAutoRange=True)
            curr_plot.setLimits(xMin=x_range[0], xMax=x_range[-1], yMin=y_range[0], yMax=y_range[-1])
            return img_item

        time_bin_edges = _out_decoded_time_bin_edges[a_lap_id]
        n_epoch_time_bins = len(time_bin_edges) - 1
        if debug_print:
            print(f'a_lap_id: {a_lap_id}, n_epoch_time_bins: {n_epoch_time_bins}')

        win = pg.GraphicsLayoutWidget()
        plots = []
        out_pf1D_decoder_template_objects = []
        _out_decoded_active_plots = {}

        plots_data = _out_decoded_active_plots_data[a_lap_id]
        plots_container = RenderPlots(name=a_lap_id, root_plot=None) # [a_lap_id]

        # Epoch Active Spikes, takes up a row _______________________________________________________________ #
        spanning_spikes_raster_plot = win.addPlot(title="spikes_raster Plot", row=0, rowspan=1, col=0, colspan=n_epoch_time_bins)
        spanning_spikes_raster_plot.setTitle("spikes_raster Plot")
        plots_container.root_plot = spanning_spikes_raster_plot
        app, raster_win, plots_container, plots_data = new_plot_raster_plot(plots_data.spikes_df, plots_data.active_aclus, scatter_plot_kwargs=None, win=spanning_spikes_raster_plot, plots_data=plots_data, plots=plots_container,
                                                            scatter_app_name=f'lap_specific_spike_raster', defer_show=True, active_context=None, add_debug_header_label=False)



        _out_decoded_active_plots[a_lap_id] = plots_container
        win.nextRow()

        # Decoded Epoch Posterior (bin-by-bin), takes up a row _______________________________________________________________ #
        spanning_posterior_plot = win.addPlot(title="P_x_given_n Plot", row=1, rowspan=1, col=0, colspan=n_epoch_time_bins)
        spanning_posterior_plot.setTitle("P_x_given_n Plot - Decoded over lap")

        most_likely_positions, p_x_given_n, most_likely_position_indicies, flat_outputs_container = _out_decoded_active_p_x_given_n[a_lap_id]
        flat_p_x_given_n = deepcopy(p_x_given_n)
        _simply_plot_posterior_in_pyqtgraph_plotitem(
            curr_plot=spanning_posterior_plot,
            image=flat_p_x_given_n,
            xbin_edges=np.arange(n_epoch_time_bins+1),
            ybin_edges=deepcopy(a_decoder.xbin)
        )
        win.nextRow()

        # Bin-by-bin active spike templates/pf1D fields ______________________________________________________________________ #
        active_bin_unit_specific_time_binned_spike_counts = _out_decoded_unit_specific_time_binned_spike_counts[a_lap_id]
        active_lap_active_aclu_spike_counts_list = _out_decoded_active_unit_lists[a_lap_id]

        for a_time_bin_idx in np.arange(n_epoch_time_bins):
            active_bin_active_aclu_spike_counts_dict = active_lap_active_aclu_spike_counts_list[a_time_bin_idx]
            active_bin_active_aclu_spike_count_values = np.array(list(active_bin_active_aclu_spike_counts_dict.values()))
            active_bin_active_aclu_bin_normalized_spike_count_values = active_bin_active_aclu_spike_count_values / np.sum(
                active_bin_active_aclu_spike_count_values)

            aclu_override_alpha_weights = 0.8 + (0.2 * active_bin_active_aclu_bin_normalized_spike_count_values)
            active_bin_aclus = np.array(list(active_bin_active_aclu_spike_counts_dict.keys()))
            active_solo_override_num_spikes_weights = dict(zip(active_bin_aclus, active_bin_active_aclu_bin_normalized_spike_count_values))
            active_aclu_override_alpha_weights_dict = dict(zip(active_bin_aclus, aclu_override_alpha_weights))
            if debug_print:
                print(f'a_time_bin_idx: {a_time_bin_idx}/{n_epoch_time_bins} - active_bin_aclus: {active_bin_aclus}')

            plot = win.addPlot(title=f"Plot {a_time_bin_idx+1}", row=2, rowspan=1, col=a_time_bin_idx, colspan=1)
            plot.getViewBox().setBorder(color=(200, 200, 200), width=1)
            spanning_posterior_plot.showGrid(x=True, y=True)
            x_axis = spanning_posterior_plot.getAxis('bottom')
            x_axis.setTickSpacing(major=5, minor=1)

            plots.append(plot)
            _obj = BaseTemplateDebuggingMixin.init_from_decoder(a_decoder=a_decoder, win=plot, title_str=f't={a_time_bin_idx}')
            _obj.update_base_decoder_debugger_data(
                included_neuron_ids=active_bin_aclus,
                solo_override_alpha_weights=active_aclu_override_alpha_weights_dict,
                solo_override_num_spikes_weights=active_solo_override_num_spikes_weights
            )
            out_pf1D_decoder_template_objects.append(_obj)

        win.nextRow()
        win.show()
        return win, out_pf1D_decoder_template_objects, (_out_decoded_active_plots, _out_decoded_active_plots_data)


    @classmethod
    def plot_bin_by_bin_decoding_example(cls, curr_active_pipeline, track_templates, time_bin_size: float = 0.250, a_lap_id: int = 9, a_decoder_name = 'long_LR'):
        """
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_bin_by_bin_decoding_example

                ## INPUTS: time_bin_size
        time_bin_size: float = 0.500 # 500ms

        ## any (generic) directionald decoder
        # neuron_IDs = deepcopy(track_templates.any_decoder_neuron_IDs) # array([  2,   5,   8,  10,  14,  15,  23,  24,  25,  26,  31,  32,  33,  41,  49,  50,  51,  55,  58,  64,  69,  70,  73,  74,  75,  76,  78,  82,  83,  85,  86,  90,  92,  93,  96, 109])

        ## Get a specific decoder
        a_decoder_name: types.DecoderName = 'long_LR'
        a_decoder_idx: int = track_templates.get_decoder_names().index(a_decoder_name)
        a_decoder = deepcopy(track_templates.long_LR_decoder)

        
        
        """
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import LayoutScrollability, pyqtplot_build_image_bounds_extent
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import BaseTemplateDebuggingMixin, build_pf1D_heatmap_with_labels_and_peaks, TrackTemplates
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import BinByBinDecodingDebugger

        # Example usage:
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping() 
        global_laps_epochs_df = global_laps.to_dataframe()
        global_laps_epochs_df



        ## COMPUTED: 
        a_decoder_idx: int = track_templates.get_decoder_names().index(a_decoder_name)
        a_decoder = deepcopy(track_templates.long_LR_decoder)
        (_out_decoded_time_bin_edges, _out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists, _out_decoded_active_p_x_given_n, _out_decoded_active_plots_data) = BinByBinDecodingDebugger.build_spike_counts_and_decoder_outputs(track_templates=track_templates, global_laps_epochs_df=global_laps_epochs_df, spikes_df=global_spikes_df, a_decoder_name=a_decoder_name, time_bin_size=time_bin_size)
        win, out_pf1D_decoder_template_objects, (_out_decoded_active_plots, _out_decoded_active_plots_data) = BinByBinDecodingDebugger.build_time_binned_decoder_debug_plots(a_decoder=a_decoder, a_lap_id=a_lap_id, _out_decoded_time_bin_edges=_out_decoded_time_bin_edges, _out_decoded_active_p_x_given_n=_out_decoded_active_p_x_given_n, _out_decoded_unit_specific_time_binned_spike_counts=_out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists=_out_decoded_active_unit_lists, _out_decoded_active_plots_data=_out_decoded_active_plots_data, debug_print=True)
        print(f"Returned window: {win}")
        print(f"Returned decoder objects: {out_pf1D_decoder_template_objects}")

        return win, out_pf1D_decoder_template_objects, (_out_decoded_active_plots, _out_decoded_active_plots_data)
    



    ## OUTPUTS: _out_decoded_time_bin_edges, _out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists, _out_decoded_active_p_x_given_n


# ==================================================================================================================== #
# 2025-01-20 - Easy Decoding                                                                                           #
# ==================================================================================================================== #
@function_attributes(short_name=None, tags=['decoding', 'ACTIVE', 'useful'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-20 15:26', related_items=[])
def easy_independent_decoding(long_LR_decoder: BasePositionDecoder, spikes_df: pd.DataFrame, time_bin_size: float = 0.025, t_start: float = 0.0, t_end: float = 2093.8978568242164):
    """ Uses the provieded decoder, spikes, and time binning parameters to decode the neural activity for the specified epoch.

    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import easy_independent_decoding


    time_bin_size: float = 0.025
    t_start = 0.0
    t_end = 2093.8978568242164
    _decoded_pos_outputs, (unit_specific_time_binned_spike_counts, time_bin_edges, spikes_df) = easy_independent_decoding(long_LR_decoder, spikes_df=spikes_df, time_bin_size=time_bin_size, t_start=t_start, t_end=t_end)



    """
    neuron_IDs = deepcopy(long_LR_decoder.neuron_IDs) # array([  2,   5,   8,  10,  14,  15,  23,  24,  25,  26,  31,  32,  33,  41,  49,  50,  51,  55,  58,  64,  69,  70,  73,  74,  75,  76,  78,  82,  83,  85,  86,  90,  92,  93,  96, 109])
    spikes_df: pd.DataFrame = deepcopy(spikes_df).spikes.sliced_by_neuron_id(neuron_IDs) ## filter everything down
    time_bin_edges: NDArray = np.arange(t_start, t_end + time_bin_size, time_bin_size)
    unique_units = np.unique(spikes_df['aclu']) # sorted
    unit_specific_time_binned_spike_counts: NDArray = np.array([
        np.histogram(spikes_df.loc[spikes_df['aclu'] == unit, 't_rel_seconds'], bins=time_bin_edges)[0]
        for unit in unique_units
    ])

    ## OUTPUT: time_bin_edges, unit_specific_time_binned_spike_counts
    _decoded_pos_outputs = long_LR_decoder.decode(unit_specific_time_binned_spike_counts=unit_specific_time_binned_spike_counts, time_bin_size=time_bin_size, output_flat_versions=True, debug_print=True)
    # _decoded_pos_outputs = all_directional_pf1D_Decoder.decode(unit_specific_time_binned_spike_counts=unit_specific_time_binned_spike_counts, time_bin_size=0.020, output_flat_versions=True, debug_print=True)
    return _decoded_pos_outputs, (unit_specific_time_binned_spike_counts, time_bin_edges, spikes_df)


# ==================================================================================================================== #
# Older                                                                                                                #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['save', 'split'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-17 16:14', related_items=[])
def perform_split_save_dictlike_result(split_save_folder: Path, active_computed_data, include_includelist=None, continue_after_pickling_errors: bool=True, debug_print:bool=True):
    """ Custom saves any dict-like object


    split_folder = curr_active_pipeline.get_output_path().joinpath('split')
    split_folder.mkdir(exist_ok=True)

    ['loaded_data', '']

    # active_computed_data = self.global_computation_results.computed_data
    # include_includelist = list(self.global_computation_results.computed_data.keys())
    # split_save_folder_name: str = f'{global_computation_results_pickle_path.stem}_split'
    # split_save_folder: Path = global_computation_results_pickle_path.parent.joinpath(split_save_folder_name).resolve()


    # ==================================================================================================================== #
    # 'computation_results' (local computations)                                                                           #
    # ==================================================================================================================== #
    # split_computation_results_dir = split_folder.joinpath('computation_results')
    # split_computation_results_dir.mkdir(exist_ok=True)
    # split_save_folder, split_save_paths, split_save_output_types, failed_keys = perform_split_save_dictlike_result(split_save_folder=split_computation_results_dir, active_computed_data=curr_active_pipeline.computation_results)


    # ==================================================================================================================== #
    # 'filtered_sessions'                                                                                                  #
    # ==================================================================================================================== #
    # split_filtered_sessions_dir = split_folder.joinpath('filtered_sessions')
    # split_filtered_sessions_dir.mkdir(exist_ok=True)
    # split_save_folder, split_save_paths, split_save_output_types, failed_keys = perform_split_save_dictlike_result(split_save_folder=split_filtered_sessions_dir, active_computed_data=curr_active_pipeline.filtered_sessions)


    # ==================================================================================================================== #
    # 'global_computation_results' (global computations)                                                                   #
    # ==================================================================================================================== #
    split_global_computation_results_dir = split_folder.joinpath('global_computation_results')
    split_global_computation_results_dir.mkdir(exist_ok=True)
    split_save_folder, split_save_paths, split_save_output_types, failed_keys = perform_split_save_dictlike_result(split_save_folder=split_global_computation_results_dir, active_computed_data=curr_active_pipeline.global_computation_results.computed_data) # .__dict__


    """
    from pyphocorehelpers.print_helpers import print_filesystem_file_size, print_object_memory_usage
    from pickle import PicklingError
    from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData # used for `save_global_computation_results`
    from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult

    ## In split save, we save each result separately in a folder
    if debug_print:
        print(f'split_save_folder: {split_save_folder}')
    # make if doesn't exist
    split_save_folder.mkdir(exist_ok=True)

    if include_includelist is None:
        ## include all keys if none are specified
        include_includelist = list(active_computed_data.keys()) ## all keys by default

    split_save_paths = {}
    split_save_output_types = {}
    failed_keys = []
    skipped_keys = []
    for k, v in active_computed_data.items():
        if k in include_includelist:
            curr_split_result_pickle_path = split_save_folder.joinpath(f'Split_{k}.pkl').resolve()
            if debug_print:
                print(f'k: {k} -- size_MB: {print_object_memory_usage(v, enable_print=False)}')
                print(f'\tcurr_split_result_pickle_path: {curr_split_result_pickle_path}')
            was_save_success = False
            curr_item_type = type(v)
            try:
                ## try get as dict
                v_dict = v.__dict__ #__getstate__()
                # saveData(curr_split_result_pickle_path, (v_dict))
                saveData(curr_split_result_pickle_path, (v_dict, str(curr_item_type.__module__), str(curr_item_type.__name__)))
                was_save_success = True
            except KeyError as e:
                print(f'\t{k} encountered {e} while trying to save {k}. Skipping')
                pass
            except PicklingError as e:
                if not continue_after_pickling_errors:
                    raise
                else:
                    print(f'\t{k} encountered {e} while trying to save {k}. Skipping')
                    pass

            if was_save_success:
                split_save_paths[k] = curr_split_result_pickle_path
                split_save_output_types[k] = curr_item_type
                if debug_print:
                    print(f'\tfile_size_MB: {print_filesystem_file_size(curr_split_result_pickle_path, enable_print=False)} MB')
            else:
                failed_keys.append(k)
        else:
            if debug_print:
                print(f'\tskipping key "{k}" because it is not included in include_includelist: {include_includelist}')
            skipped_keys.append(k)

    if len(failed_keys) > 0:
        print(f'WARNING: failed_keys: {failed_keys} did not save for global results! They HAVE NOT BEEN SAVED!')
    return split_save_folder, split_save_paths, split_save_output_types, failed_keys



# ==================================================================================================================== #
# 2025-01-15 Plotting Decoding Performance on Track                                                                    #
# ==================================================================================================================== #

from neuropy.utils.mixins.time_slicing import TimePointEventAccessor
from neuropy.core.position import PositionAccessor
from neuropy.core.flattened_spiketrains import SpikesAccessor
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import SynchronizedPlotMode
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DecoderIdentityColors

@function_attributes(short_name=None, tags=['decoder', 'matplotlib', 'plot', 'track', 'performance'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-15 17:44', related_items=[])
def _perform_plot_multi_decoder_meas_pred_position_track(curr_active_pipeline, fig, ax_list, desired_time_bin_size: Optional[float]=None, enable_flat_line_drawing: bool = False, debug_print = False): # , pos_df: pd.DataFrame, laps_df: pd.DataFrame
    """ Plots a new matplotlib-based track that displays the measured and most-likely decoded decoded position LINES (for all four decoders) **on the same axes**. The "correct" (ground-truth) decoder is highlighted (higher opacity and thicker line) compared to the wrong decoders' estimates.

    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import SynchronizedPlotMode
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DecoderIdentityColors
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _perform_plot_multi_decoder_meas_pred_position_track

        ## Compute continuous first
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'], computation_kwargs_list=[{'time_bin_size': 0.025}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'], computation_kwargs_list=[{'time_bin_size': 0.050}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'], computation_kwargs_list=[{'time_bin_size': 0.075}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'], computation_kwargs_list=[{'time_bin_size': 0.100}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'], computation_kwargs_list=[{'time_bin_size': 0.250}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)

        ## Build the new dock track:
        dock_identifier: str = 'Continuous Decoding Performance'
        ts_widget, fig, ax_list = active_2d_plot.add_new_matplotlib_render_plot_widget(name=dock_identifier)
        ## Get the needed data:
        directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
        all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
        previously_decoded_keys: List[float] = list(continuously_decoded_result_cache_dict.keys()) # [0.03333]
        print(F'previously_decoded time_bin_sizes: {previously_decoded_keys}')

        time_bin_size: float = directional_decoders_decode_result.most_recent_decoding_time_bin_size
        print(f'time_bin_size: {time_bin_size}')
        continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = directional_decoders_decode_result.most_recent_continuously_decoded_dict
        all_directional_continuously_decoded_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = {k:v for k, v in (continuously_decoded_dict or {}).items() if k in TrackTemplates.get_decoder_names()} ## what is plotted in the `f'{a_decoder_name}_ContinuousDecode'` rows by `AddNewDirectionalDecodedEpochs_MatplotlibPlotCommand`
        ## OUT: all_directional_continuously_decoded_dict
        ## Draw the position meas/decoded on the plot widget
        ## INPUT: fig, ax_list, all_directional_continuously_decoded_dict, track_templates

        _out_artists =  _perform_plot_multi_decoder_meas_pred_position_track(curr_active_pipeline, fig, ax_list, all_directional_continuously_decoded_dict, track_templates, enable_flat_line_drawing=True)


        ## sync up the widgets
        active_2d_plot.sync_matplotlib_render_plot_widget(dock_identifier, sync_mode=SynchronizedPlotMode.TO_WINDOW)

    """
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult

    ## Get the needed data:

    ## get from parameters:
    minimum_inclusion_fr_Hz: float = curr_active_pipeline.global_computation_results.computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz
    included_qclu_values: List[int] = curr_active_pipeline.global_computation_results.computation_config.rank_order_shuffle_analysis.included_qclu_values

    directional_laps_results: DirectionalLapsResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']
    track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only -- !! Is minimum_inclusion_fr_Hz=None the issue/difference?
    print(f'minimum_inclusion_fr_Hz: {minimum_inclusion_fr_Hz}')
    print(f'included_qclu_values: {included_qclu_values}')

    directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
    # all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
    previously_decoded_keys: List[float] = list(directional_decoders_decode_result.continuously_decoded_result_cache_dict.keys()) # [0.03333]
    if debug_print:
        print(F'previously_decoded time_bin_sizes: {previously_decoded_keys}')

    if desired_time_bin_size is None:
        time_bin_size: float = directional_decoders_decode_result.most_recent_decoding_time_bin_size
        continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = directional_decoders_decode_result.most_recent_continuously_decoded_dict
    else:
        if desired_time_bin_size not in previously_decoded_keys:
            print(f'desired_time_bin_size: {desired_time_bin_size} is missing from previously decoded continuous cache. Must recompute.')
            curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'], computation_kwargs_list=[{'time_bin_size': desired_time_bin_size}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
            directional_decoders_decode_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded'] ## update the result
            print(f'\tcalculation complete.')

        time_bin_size: float =  desired_time_bin_size
        continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = directional_decoders_decode_result.continuously_decoded_result_cache_dict[time_bin_size]


    print(f'time_bin_size: {time_bin_size}')
    # continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = directional_decoders_decode_result.most_recent_continuously_decoded_dict
    all_directional_continuously_decoded_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = {k:v for k, v in (continuously_decoded_dict or {}).items() if k in TrackTemplates.get_decoder_names()} ## what is plotted in the `f'{a_decoder_name}_ContinuousDecode'` rows by `AddNewDirectionalDecodedEpochs_MatplotlibPlotCommand`
    ## OUT: all_directional_continuously_decoded_dict

    ## INPUT: fig, ax_list, all_directional_continuously_decoded_dict, track_templates

    ## Laps
    global_laps_obj: Laps = deepcopy(curr_active_pipeline.sess.laps)
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    laps_df = global_laps_obj.adding_true_decoder_identifier(t_start, t_delta, t_end) ## ensures ['maze_id', 'lap_dir', 'is_LR_dir', 'truth_decoder_name']
    ## Positions:
    pos_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.position).to_dataframe().position.adding_lap_info(laps_df=laps_df, inplace=False)
    pos_df = pos_df.time_point_event.adding_true_decoder_identifier(t_start=t_start, t_delta=t_delta, t_end=t_end) ## ensures ['maze_id', 'is_LR_dir']
    # pos_df = pos_df.position.add_binned_time_column(time_window_edges=time_bin_containers.edges, time_window_edges_binning_info=time_bin_containers.edge_info) # 'binned_time' refers to which time bins these are

    ## OUTPUTS: laps_df, pos_df


    ## Draw the position meas/decoded on the plot widget
    kwargs = {}
    # decoded_pos_line_kwargs = dict(lw=1.0, color='gray', alpha=0.8, marker='+', markersize=6, animated=False)
    # lw=1.0, color='#00ff7f99', alpha=0.6
    # two_step_options_dict = { 'color':'#00ff7f99', 'face_color':'#55ff0099', 'edge_color':'#00aa0099' }
    inactive_decoded_pos_line_kwargs = dict(lw=0.3, alpha=0.2, marker='.', markersize=2, animated=False)
    active_decoded_pos_line_kwargs = dict(lw=1.0, alpha=0.8, marker='+', markersize=6, animated=False)

    decoder_color_dict: Dict[types.DecoderName, str] = DecoderIdentityColors.build_decoder_color_dict()
    # pos_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.position.to_dataframe())

    # ax = ax_list[0]
    ax_list[0].clear() ## clear any existing artists just to be sure
    _out_data = {}
    _out_data_plot_kwargs = {}
    _out_artists = {}
    # curr_active_pipeline.global_computation_results.t
    for i, (a_decoder_name, a_decoder) in enumerate(track_templates.get_decoders_dict().items()):
        is_first_iteration: bool = (i == 0)
        a_continuously_decoded_result = all_directional_continuously_decoded_dict[a_decoder_name]
        a_decoder_color = decoder_color_dict[a_decoder_name]

        assert len(a_continuously_decoded_result.p_x_given_n_list) == 1
        p_x_given_n = a_continuously_decoded_result.p_x_given_n_list[0]
        # p_x_given_n = a_continuously_decoded_result.p_x_given_n_list[0]['p_x_given_n']
        time_bin_containers = a_continuously_decoded_result.time_bin_containers[0]
        time_window_centers = time_bin_containers.centers
        # p_x_given_n.shape # (62, 4, 209389)
        a_marginal_x = a_continuously_decoded_result.marginal_x_list[0]
        # active_time_window_variable = a_decoder.active_time_window_centers
        active_time_window_variable = time_window_centers
        active_most_likely_positions_x = a_marginal_x['most_likely_positions_1D'] # a_decoder.most_likely_positions[:,0].T


        ## Plot general laps only on the first iteration. Needs: pos_df
        if is_first_iteration:
            pos_df = pos_df.position.add_binned_time_column(time_window_edges=time_bin_containers.edges, time_window_edges_binning_info=time_bin_containers.edge_info) # 'binned_time' refers to which time bins these are
            # Plot the measured position X:
            _, ax = plot_1D_most_likely_position_comparsions(pos_df, variable_name='x', time_window_centers=None, xbin=None, posterior=None, active_most_likely_positions_1D=None, ax=ax_list[0],
                                                            enable_flat_line_drawing=enable_flat_line_drawing, debug_print=debug_print, **kwargs)
        ## END if is_first_iteration...

        _out_data[a_decoder_name] = pd.DataFrame({'t': time_window_centers, 'x': active_most_likely_positions_x, 'binned_time': np.arange(len(time_window_centers))})
        _out_data[a_decoder_name] = _out_data[a_decoder_name].position.adding_lap_info(laps_df=laps_df, inplace=False)
        _out_data[a_decoder_name] = _out_data[a_decoder_name].time_point_event.adding_true_decoder_identifier(t_start=t_start, t_delta=t_delta, t_end=t_end) ## ensures ['maze_id', 'is_LR_dir']
        _out_data[a_decoder_name]['is_active_decoder_time'] = (_out_data[a_decoder_name]['truth_decoder_name'].fillna('', inplace=False) == a_decoder_name)

        # is_active_decoder_time = (_out_data[a_decoder_name]['truth_decoder_name'] == a_decoder_name)
        active_decoder_time_points = _out_data[a_decoder_name][_out_data[a_decoder_name]['truth_decoder_name'] == a_decoder_name]['t'].to_numpy()
        active_decoder_most_likely_positions_x = _out_data[a_decoder_name][_out_data[a_decoder_name]['truth_decoder_name'] == a_decoder_name]['x'].to_numpy()
        active_decoder_inactive_time_points = _out_data[a_decoder_name][_out_data[a_decoder_name]['truth_decoder_name'] != a_decoder_name]['t'].to_numpy()
        active_decoder_inactive_most_likely_positions_x = _out_data[a_decoder_name][_out_data[a_decoder_name]['truth_decoder_name'] != a_decoder_name]['x'].to_numpy()
        ## could fill y with np.nan instead of getting shorter?
        # _out_data_plot_kwargs[a_decoder_name] = (dict(x=active_decoder_time_points, y=active_decoder_most_likely_positions_x, color=a_decoder_color, **active_decoded_pos_line_kwargs), dict(x=active_decoder_inactive_time_points, y=active_decoder_inactive_most_likely_positions_x, color=a_decoder_color, **inactive_decoded_pos_line_kwargs))

        _out_data_plot_kwargs[a_decoder_name] = dict(active=dict(x=active_decoder_time_points, y=active_decoder_most_likely_positions_x, color=a_decoder_color, **active_decoded_pos_line_kwargs), inactive=dict(x=active_decoder_inactive_time_points, y=active_decoder_inactive_most_likely_positions_x, color=a_decoder_color, **inactive_decoded_pos_line_kwargs))
        _out_artists[a_decoder_name] = {}
        for a_plot_name, a_plot_kwargs in _out_data_plot_kwargs[a_decoder_name].items():
            # _out_artists[a_decoder_name][a_plot_name] = ax.plot(**a_plot_kwargs, label=f'Most-likely {a_decoder_name} ({a_plot_name})') # (Num windows x 2)
            _out_artists[a_decoder_name][a_plot_name] = ax.plot(a_plot_kwargs.pop('x'), a_plot_kwargs.pop('y'), **a_plot_kwargs, label=f'Most-likely {a_decoder_name} ({a_plot_name})')

    ## OUTPUT: _out_artists
    return _out_artists




# ==================================================================================================================== #
# 2025-01-13 - Lap Overriding/qclu filtering                                                                           #
# ==================================================================================================================== #
from neuropy.core.user_annotations import UserAnnotationsManager


@function_attributes(short_name=None, tags=['laps', 'update', 'session', 'TODO', 'UNFINISHED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-13 15:45', related_items=[])
def override_laps(curr_active_pipeline, override_laps_df: pd.DataFrame, debug_print=False):
    """
    overrides the laps

    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import override_laps

        override_laps_df: Optional[pd.DataFrame] = UserAnnotationsManager.get_hardcoded_laps_override_dict().get(curr_active_pipeline.get_session_context(), None)
        if override_laps_df is not None:
            print(f'overriding laps....')
            display(override_laps_df)
            override_laps(curr_active_pipeline, override_laps_df=override_laps_df)



    """
    from neuropy.core.laps import Laps

    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()

    ## Load the custom laps
    if debug_print:
        print(f'override_laps_df: {override_laps_df}')
    if 'lap_id' not in override_laps_df:
        override_laps_df['lap_id'] = override_laps_df.index.astype('int') ## set lap_id from the index
    else:
        override_laps_df[['lap_id']] = override_laps_df[['lap_id']].astype('int')

    # Either way, ensure that the lap_dir is an 'int' column.
    override_laps_df['lap_dir'] = override_laps_df['lap_dir'].astype('int')
    # override_laps_df['lap_dir'] = override_laps_df['lap_dir'].astype('int')
    # override_laps_df['is_LR_dir'] = (override_laps_df['lap_dir'] < 1.0)

    if 'label' not in override_laps_df:
        override_laps_df['label'] = override_laps_df['lap_id'].astype('str') # add the string "label" column
    else:
        override_laps_df['label'] = override_laps_df['label'].astype('str')

    # curr_laps_df = Laps._compute_lap_dir_from_smoothed_velocity(laps_df=curr_laps_df, global_session=global_session, replace_existing=True)

    override_laps_df = Laps._compute_lap_dir_from_smoothed_velocity(laps_df=override_laps_df, global_session=curr_active_pipeline.sess, replace_existing=True)
    override_laps_df = Laps._update_dataframe_computed_vars(laps_df=override_laps_df, t_start=t_start, t_delta=t_delta, t_end=t_end, global_session=curr_active_pipeline.sess, replace_existing=False)
    override_laps_obj = Laps(laps=override_laps_df, metadata=None).filter_to_valid()
    ## OUTPUTS: override_laps_obj

    curr_active_pipeline.sess.laps = deepcopy(override_laps_obj)

    # curr_active_pipeline.sess.laps_df = override_laps_df
    curr_active_pipeline.sess.compute_position_laps()

    # curr_active_pipeline.sess = curr_active_pipeline.sess
    # a_pf1D_dt.replacing_computation_epochs(epochs=override_laps_df)

    for a_filtered_context_name, a_filtered_context in curr_active_pipeline.filtered_contexts.items():
        ## current session
        a_filtered_epoch = curr_active_pipeline.filtered_epochs[a_filtered_context_name]
        filtered_sess = curr_active_pipeline.filtered_sessions[a_filtered_context_name]
        if debug_print:
            print(f'a_filtered_context_name: {a_filtered_context_name}, a_filtered_context: {a_filtered_context}')
            display(a_filtered_context)
        # override_laps_obj.filter_to_valid()
        a_filtered_override_laps_obj: Laps = deepcopy(override_laps_obj).time_slice(t_start=filtered_sess.t_start, t_stop=filtered_sess.t_stop)
        # a_filtered_context.lap_dir
        filtered_sess.laps = deepcopy(a_filtered_override_laps_obj)
        filtered_sess.compute_position_laps()
        if debug_print:
            print(f'\tupdated.')




# ==================================================================================================================== #
# @ 2025-01-01 - Better Aggregation of Probabilities across bins                                                       #
# ==================================================================================================================== #
@metadata_attributes(short_name=None, tags=['aggregation', 'UNFINSHED', 'integration', 'confidence'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-01 00:00', related_items=[])
class TimeBinAggregation:
    """ Methods of aggregating over many time bins

    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import TimeBinAggregation, ParticleFilter


    """
    @function_attributes(short_name=None, tags=['NDArray', 'streak', 'sequence', 'probability', 'likelihood'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-01 14:00', related_items=[])
    @classmethod
    def compute_epoch_p_long_with_streaks(cls, p_long: List[float], min_probability_threshold: float = 0.5) -> float:
        """ Prior to `compute_streak_weighted_p_long` version, which operates on dataframes


        Computes the overall P_Long for an epoch, giving higher weight to uninterrupted streaks.

        Args:
            p_long (List[float]): Probabilities of being in the "Long" state for each time bin.
            threshold (float): Minimum probability to consider a bin as part of a streak. Default is 0.5.

        Returns:
            float: The overall P_Long for the epoch.

        Usage:
            # Example usage
            p_long = a_lap_df[a_var_name].to_numpy() # [0.8, 0.7, 0.2, 0.9, 0.95, 0.3, 0.85, 0.87, 0.86]
            min_probability_threshold = 0.5  # Minimum probability to count as "Long"
            overall_p_long = compute_epoch_p_long_with_streaks(p_long, min_probability_threshold)
            print(f"Overall P_Long with streak weighting: {overall_p_long}")

        """
        streaks = []
        current_streak = []

        # Identify streaks
        for i, prob in enumerate(p_long):
            if prob >= min_probability_threshold:
                current_streak.append(i)
            else:
                if current_streak:
                    streaks.append(current_streak)
                    current_streak = []
        if current_streak:  # Add the last streak if it ends at the last bin
            streaks.append(current_streak)

        # Assign weights based on streak length (linearly)
        weights = [0] * len(p_long)
        for streak in streaks:
            streak_length = len(streak)
            for idx in streak:
                weights[idx] = streak_length

        # Compute weighted average
        weighted_sum = np.sum(w * p for w, p in zip(weights, p_long))
        total_weight = np.sum(weights)
        return weighted_sum / total_weight if total_weight > 0 else 0.0



    @function_attributes(short_name=None, tags=['streak', 'sequence', 'dataframe', 'probability', 'likelihood'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-01 14:11', related_items=[])
    @classmethod
    def compute_streak_weighted_p_long(cls, df: pd.DataFrame, column: str, threshold: float = 0.5) -> float:
        """
        Computes the streak-weighted P_Long for a given DataFrame, giving higher weight to longer sequences of adjacent bins.

        Args:
            df (pd.DataFrame): Input DataFrame containing the P_Long column.
            column (str): The column name for P_Long.
            threshold (float): Minimum probability to consider a bin as part of a streak. Default is 0.5.

        Returns:
            float: The streak-weighted P_Long for the DataFrame.
        """
        p_long = df[column].values
        streaks = []
        current_streak = []

        # Identify streaks
        for i, prob in enumerate(p_long):
            if prob >= threshold:
                current_streak.append(i)
            else:
                if current_streak:
                    streaks.append(current_streak)
                    current_streak = []
        if current_streak:  # Add the last streak if it ends at the last bin
            streaks.append(current_streak)

        # Assign weights based on streak length
        weights = [0] * len(p_long)
        for streak in streaks:
            streak_length = len(streak)
            for idx in streak:
                weights[idx] = streak_length

        # Compute weighted average
        weighted_sum = sum(w * p for w, p in zip(weights, p_long))
        total_weight = sum(weights)
        return weighted_sum / total_weight if total_weight > 0 else 0.0


    @classmethod
    def peak_rolling_avg(cls, df: pd.DataFrame, column: str, window: int=3, *args, **kwargs) -> float:
        """
        Computes the streak-weighted P_Long for a given DataFrame, giving higher weight to longer sequences of adjacent bins.

        Args:
            df (pd.DataFrame): Input DataFrame containing the P_Long column.
            column (str): The column name for P_Long.
            threshold (float): Minimum probability to consider a bin as part of a streak. Default is 0.5.

        Returns:
            float: The streak-weighted P_Long for the DataFrame.
        """
        return df[column].rolling(window, *args, **kwargs).max().max()

@metadata_attributes(short_name=None, tags=['UNUSED', 'ChatGPT', 'aggregation'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-01 00:00', related_items=['TimeBinAggregation'])
class ParticleFilter:
    """ Example: Generated by ChatGPT, does something about aggregating time bins

        ## TEST

        # Example usage
        def state_transition_func(particles):
            # Example state transition function: simple linear motion.
            return particles + 1  # Each state increases by 1

        def measurement_func(state):
            # Example measurement function: identity mapping.
            return state



        num_particles = 1000
        state_dim = 1
        process_noise = 1.0
        measurement_noise = 2.0

        pf = ParticleFilter(num_particles, state_dim, process_noise, measurement_noise)

        # Simulate a series of observations
        # observations = np.array([5, 6, 7, 8, 9])
        observations = a_lap_df[a_var_name].to_numpy()

        for obs in observations:
            pf.predict(state_transition_func)
            pf.update(obs, measurement_func)
            pf.resample()

            estimated_state = pf.estimate()
            print(f"Estimated State: {estimated_state}")


    """
    def __init__(self, num_particles: int, state_dim: int, process_noise: float, measurement_noise: float):
        """
        Initialize the Particle Filter.

        :param num_particles: Number of particles to use.
        :param state_dim: Dimension of the state space.
        :param process_noise: Standard deviation of process noise.
        :param measurement_noise: Standard deviation of measurement noise.
        """
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Initialize particles randomly within the state space
        self.particles = np.random.rand(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, state_transition_func):
        """
        Predict the next state of each particle using the state transition function.

        :param state_transition_func: A function to apply the state transition.
        """
        noise = np.random.normal(0, self.process_noise, size=(self.num_particles, self.state_dim))
        self.particles = state_transition_func(self.particles) + noise

    def update(self, observation: np.ndarray, measurement_func):
        """
        Update the particle weights based on the observation.

        :param observation: The observed data.
        :param measurement_func: A function to calculate the expected observation from a particle state.
        """
        for i in range(self.num_particles):
            predicted_obs = measurement_func(self.particles[i])
            error = observation - predicted_obs
            likelihood = np.exp(-0.5 * np.sum(error**2) / self.measurement_noise**2)
            self.weights[i] = likelihood

        # Normalize weights to sum to 1
        self.weights /= np.sum(self.weights)

    def resample(self):
        """
        Resample particles based on their weights.
        """
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self) -> np.ndarray:
        """
        Estimate the state as the weighted mean of the particles.

        :return: The estimated state.
        """
        return np.average(self.particles, weights=self.weights, axis=0)



# ==================================================================================================================== #
# 2024-12-31 - Decoder ID x Position                                                                                   #
# ==================================================================================================================== #
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import CustomDecodeEpochsResult, MeasuredDecodedPositionComparison, DecodedFilterEpochsResult

@metadata_attributes(short_name=None, tags=['validation', 'plot', 'figure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-02 09:10', related_items=[])
class EstimationCorrectnessPlots:
    """ Compares ground truth to the decoded positions during laps

    EstimationCorrectnessPlots.plot_estimation_correctness_vertical_stack(
        _out_subset_decode_dfs_dict, 'binned_x_meas', 'estimation_correctness_track_ID'
    )

    # Example usage
    # EstimationCorrectnessPlots.plot_estimation_correctness_bean_plot(
    #     _out_subset_decode_dfs_dict, 'binned_x_meas', 'estimation_correctness_track_ID'
    # )

    """
    @function_attributes(short_name=None, tags=['performance'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-31 12:32', related_items=['build_lap_bin_by_bin_performance_analysis_df'])
    def plot_estimation_correctness_with_raw_data(epochs_df: pd.DataFrame, x_col: str, y_col: str, extra_info_str: str=''):
        """
        Plots a bar plot with error bars for the mean and variability of a metric across bins,
        overlayed with a swarm-like plot showing raw data points, ensuring proper alignment.

        Args:
            epochs_df (pd.DataFrame): DataFrame containing the data.
            x_col (str): Column name for the x-axis (binned variable).
            y_col (str): Column name for the y-axis (metric to visualize).

        Usage:
            # Example usage
            EstimationCorrectnessPlots.plot_estimation_correctness_with_raw_data(epochs_track_identity_marginal_df, 'binned_x_meas', 'estimation_correctness_track_ID')


        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Clip the values to the range [0, 1]
        epochs_df[y_col] = epochs_df[y_col].clip(lower=0, upper=1)

        # Ensure x_col is treated consistently as a categorical variable
        epochs_df[x_col] = pd.Categorical(epochs_df[x_col], ordered=True)
        grouped = epochs_df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()

        # Plotting
        plt.figure(figsize=(10, 6))

        # Bar plot with error bars
        plt.bar(grouped[x_col].cat.codes, grouped['mean'], yerr=grouped['std'], capsize=5, color='skyblue', alpha=0.7, label='Mean ± Std')

        # Overlay raw data as a strip plot
        sns.stripplot(data=epochs_df, x=x_col, y=y_col, color='black', alpha=0.6, jitter=True, size=5)

        # Manually add legend for raw data once
        plt.scatter([], [], color='black', alpha=0.6, label='Raw Data')

        # Align x-axis ticks and labels
        plt.xticks(ticks=range(len(grouped[x_col].cat.categories)), labels=grouped[x_col].cat.categories)

        plt.title(f'Estimation Correctness Across Binned X Measurements: {extra_info_str}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    @function_attributes(short_name=None, tags=['performance'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-31 12:32', related_items=['build_lap_bin_by_bin_performance_analysis_df'])
    def plot_estimation_correctness_vertical_stack(epochs_dict: dict, x_col: str, y_col: str):
        """
        Plots a vertical stack of estimation correctness across multiple time bin sizes.
        Each time bin size is rendered in a separate subplot with consistent y-limits [0, 1].

        Args:
            epochs_dict (dict): Dictionary where keys are time bin sizes and values are DataFrames containing the data.
            x_col (str): Column name for the x-axis (binned variable).
            y_col (str): Column name for the y-axis (metric to visualize).

        Usage:
            # Example usage
            plot_estimation_correctness_vertical_stack(
                _out_subset_decode_dict, 'binned_x_meas', 'estimation_correctness_track_ID'
            )
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        n_bins = len(epochs_dict)
        fig, axes = plt.subplots(n_bins, 1, figsize=(10, 5 * n_bins), sharex=True)
        color_palette = sns.color_palette("Set2", n_bins)

        for i, (time_bin, df) in enumerate(epochs_dict.items()):
            ax = axes[i] if n_bins > 1 else axes

            # Clip the values to the range [0, 1]
            df[y_col] = df[y_col].clip(lower=0, upper=1)

            # Ensure x_col is treated consistently as a categorical variable
            df[x_col] = pd.Categorical(df[x_col], ordered=True)
            grouped = df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()

            # Bar plot with error bars
            ax.bar(
                grouped[x_col].cat.codes,
                grouped['mean'],
                yerr=grouped['std'],
                capsize=5,
                color=color_palette[i],
                alpha=0.7,
                label=f'Time Bin: {time_bin}'
            )

            # Overlay raw data as a strip plot
            sns.stripplot(
                data=df,
                x=pd.Categorical(df[x_col]).codes,
                y=y_col,
                color='black',
                alpha=0.6,
                jitter=True,
                size=4,
                ax=ax
            )

            # Adjust y-axis and title
            ax.set_ylim(0, 1)
            ax.set_title(f'Estimation Correctness (Time Bin: {time_bin})')
            ax.set_ylabel(y_col)
            ax.grid(True, linestyle='--', alpha=0.7)

        # X-axis adjustments (shared)
        grouped_x_labels = grouped[x_col].cat.categories
        ax.set_xticks(range(len(grouped_x_labels)))
        ax.set_xticklabels(grouped_x_labels)
        ax.set_xlabel(x_col)

        plt.tight_layout()
        plt.show()


    @function_attributes(short_name=None, tags=['performance'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-31 12:32', related_items=['build_lap_bin_by_bin_performance_analysis_df'])
    def plot_estimation_correctness_bean_plot(epochs_dict: dict, x_col: str, y_col: str):
        """
        Plots a vertical stack of bean plots for estimation correctness across multiple time bin sizes.
        Each time bin size is rendered in a separate subplot with consistent y-limits [0, 1].

        Args:
            epochs_dict (dict): Dictionary where keys are time bin sizes and values are DataFrames containing the data.
            x_col (str): Column name for the x-axis (binned variable).
            y_col (str): Column name for the y-axis (metric to visualize).

        Usage:
            # Example usage
            plot_estimation_correctness_bean_plot(
                _out_subset_decode_dict, 'binned_x_meas', 'estimation_correctness_track_ID'
            )
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        n_bins = len(epochs_dict)
        fig, axes = plt.subplots(n_bins, 1, figsize=(10, 5 * n_bins), sharex=True)

        for i, (time_bin, df) in enumerate(epochs_dict.items()):
            ax = axes[i] if n_bins > 1 else axes

            # Clip the values to the range [0, 1]
            df[y_col] = df[y_col].clip(lower=0, upper=1)

            # Ensure x_col is treated consistently as a categorical variable
            df[x_col] = pd.Categorical(df[x_col], ordered=True)

            # Violin plot (density estimates)
            sns.violinplot(
                data=df,
                x=x_col,
                y=y_col,
                scale='width',
                inner=None,  # Remove internal bars to focus on raw points
                bw=0.2,  # Bandwidth adjustment for smoother density
                cut=0,  # Restrict to data range
                linewidth=1,
                color='skyblue',
                ax=ax
            )

            # Overlay raw data points
            sns.stripplot(
                data=df,
                x=x_col,
                y=y_col,
                color='black',
                alpha=0.6,
                jitter=True,
                size=4,
                ax=ax
            )

            # Adjust y-axis and title
            ax.set_ylim(0, 1)
            ax.set_title(f'Estimation Correctness (Time Bin: {time_bin})')
            ax.set_ylabel(y_col)
            ax.grid(True, linestyle='--', alpha=0.7)

        # Shared X-axis adjustments
        ax.set_xlabel(x_col)
        plt.tight_layout()
        plt.show()


@function_attributes(short_name=None, tags=['transition_matrix', 'position', 'decoder_id', '2D'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-31 10:05', related_items=[])
def build_position_by_decoder_transition_matrix(p_x_given_n, debug_print=False):
    """
    given a decoder that gives a probability that the generating process is one of two possibilities, what methods are available to estimate the probability for a contiguous epoch made of many time bins?
    Note: there is most certainly temporal dependence, how should I go about dealing with this?

    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import build_position_by_decoder_transition_matrix, plot_blocked_transition_matrix

        ## INPUTS: p_x_given_n
        n_position_bins, n_decoding_models, n_time_bins = p_x_given_n.shape
        A_position, A_model, A_big = build_position_by_decoder_transition_matrix(p_x_given_n)

        ## Plotting:
        import matplotlib.pyplot as plt; import seaborn as sns

        # plt.figure(figsize=(8,6)); sns.heatmap(A_big, cmap='viridis'); plt.title("Transition Matrix A_big"); plt.show()
        plt.figure(figsize=(8,6)); sns.heatmap(A_position, cmap='viridis'); plt.title("Transition Matrix A_position"); plt.show()
        plt.figure(figsize=(8,6)); sns.heatmap(A_model, cmap='viridis'); plt.title("Transition Matrix A_model"); plt.show()

        plot_blocked_transition_matrix(A_big, n_position_bins, n_decoding_models)


    """
    # Assume p_x_given_n is already loaded with shape (57, 4, 29951).
    # We'll demonstrate by generating random data:
    # p_x_given_n = np.random.rand(57, 4, 29951)

    n_position_bins, n_decoding_models, n_time_bins = p_x_given_n.shape

    # 1. Determine the most likely model for each time bin
    sum_over_positions = p_x_given_n.sum(axis=0)  # (n_decoding_models, n_time_bins)
    best_model_each_bin = sum_over_positions.argmax(axis=0)  # (n_time_bins,)

    # 2. Determine the most likely position for each time bin (conditional on chosen model)
    best_position_each_bin = np.array([
        p_x_given_n[:, best_model_each_bin[t], t].argmax()
        for t in range(n_time_bins)
    ])

    # 3. Build position transition matrix
    A_position_counts = np.zeros((n_position_bins, n_position_bins))
    for t in range(n_time_bins - 1):
        A_position_counts[best_position_each_bin[t], best_position_each_bin[t+1]] += 1
    A_position = A_position_counts / A_position_counts.sum(axis=1, keepdims=True)
    A_position = np.nan_to_num(A_position)  # handle rows with zero counts

    # 4. Build model transition matrix
    A_model_counts = np.zeros((n_decoding_models, n_decoding_models))
    for t in range(n_time_bins - 1):
        A_model_counts[best_model_each_bin[t], best_model_each_bin[t+1]] += 1
    A_model = A_model_counts / A_model_counts.sum(axis=1, keepdims=True)
    A_model = np.nan_to_num(A_model)

    # 5. Construct combined transition matrix (Kronecker product)
    A_big = np.kron(A_position, A_model)
    if debug_print:
        print("A_position:", A_position)
        print("A_model:", A_model)
        print("A_big shape:", A_big.shape)
    return A_position, A_model, A_big



def plot_blocked_transition_matrix(A_big, n_position_bins, n_decoding_models, tick_labels=('long_LR', 'long_RL', 'short_LR', 'short_RL'), should_show_marginals:bool=True, extra_title_suffix:str=''):
    """

    plot_blocked_transition_matrix(A_big, n_position_bins, n_decoding_models)


    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.gridspec as gridspec
    from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle

    if should_show_marginals:
        fig = plt.figure(figsize=(9, 9))
        gs = gridspec.GridSpec(2, 2, width_ratios=[10, 1], height_ratios=[1, 10])

        ax_heatmap = fig.add_subplot(gs[1, 0])
        ax_row_sums = fig.add_subplot(gs[1, 1], sharey=ax_heatmap)
        ax_col_sums = fig.add_subplot(gs[0, 0], sharex=ax_heatmap)

        # Hide tick labels on margin plots
        plt.setp(ax_row_sums.get_yticklabels(), visible=False)
        plt.setp(ax_col_sums.get_xticklabels(), visible=False)

        # Main heatmap
        sns.heatmap(A_big, cmap='viridis', ax=ax_heatmap, cbar=False)

        # Draw lines separating decoder blocks
        for i in range(1, n_decoding_models):
            ax_heatmap.axhline(i * n_position_bins, color='white')
            ax_heatmap.axvline(i * n_position_bins, color='white')

        # Row sums (marginal over columns)
        row_sums = A_big.sum(axis=1)
        ax_row_sums.barh(np.arange(len(row_sums)), row_sums, color='gray')
        ax_row_sums.invert_xaxis()

        # Column sums (marginal over rows)
        col_sums = A_big.sum(axis=0)
        ax_col_sums.bar(np.arange(len(col_sums)), col_sums, color='gray')

        # Tick positions (centered in each block)
        tick_locs = [i * n_position_bins + n_position_bins / 2 for i in range(n_decoding_models)]
        if tick_labels is not None:
            assert len(tick_labels) == n_decoding_models, f"n_decoding_models: {n_decoding_models}, len(tick_labels): {len(tick_labels)}"
            tick_labels = list(tick_labels)
        else:
            tick_labels = [f'Decoder {i}' for i in range(n_decoding_models)]

        # Apply block-centered labels
        ax_heatmap.set_xticks(tick_locs)
        ax_heatmap.set_xticklabels(tick_labels, rotation=90)
        ax_heatmap.set_yticks(tick_locs)
        ax_heatmap.set_yticklabels(tick_labels)

        plt.tight_layout()
        title_text =  "Transition Matrix Blocks by Decoder w/ Marginals"

    else:
        fig = plt.figure(figsize=(8,8))
        ax_heatmap = sns.heatmap(A_big, cmap='viridis')

        for i in range(1, n_decoding_models):
            plt.axhline(i * n_position_bins, color='white')
            plt.axvline(i * n_position_bins, color='white')

        tick_locs = [i * n_position_bins + n_position_bins / 2 for i in range(n_decoding_models)]
        if tick_labels is not None:
            assert len(tick_labels) == n_decoding_models, f"n_decoding_models: {n_decoding_models}, len(tick_labels): {len(tick_labels)}"
            tick_labels = list(tick_labels)
        else:
            tick_labels = [f'Decoder {i}' for i in range(n_decoding_models)]

        plt.xticks(tick_locs, tick_labels, rotation=90)
        plt.yticks(tick_locs, tick_labels, rotation=0)
        title_text = "Transition Matrix Blocks by Decoder"


    perform_update_title_subtitle(fig=fig, ax=None, title_string=f"plot_blocked_transition_matrix()- {title_text}{extra_title_suffix}", subtitle_string=None)
    plt.show()
    return fig, ax_heatmap


# ==================================================================================================================== #
# 2024-12-20 - Heuristicy Wisticky                                                                                     #
# ==================================================================================================================== #
from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import SubsequencesPartitioningResult
from neuropy.utils.indexing_helpers import PandasHelpers
from pyphoplacecellanalysis.Pho2D.track_shape_drawing import LinearTrackInstance
# from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import SubsequenceDetectionSamples, GroundTruthData




import matplotlib.pyplot as plt
import numpy as np

@metadata_attributes(short_name=None, tags=['matplotlib', 'interactive'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-09 14:00', related_items=[])
class InteractivePlot:
    """ 2024-12-23 - Add bin selection to a matplotlib plot to allow selecting the desired main sequence position bins for heuristic analysis

    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import InteractivePlot

    _out = subsequence_partitioning_result.plot_time_bins_multiple()
    # Pass the existing ax to the InteractivePlot
    interactive_plot = InteractivePlot(_out.axes)
    # plt.show()

    """
    # Computed Properties ________________________________________________________________________________________________ #
    # @property
    # def n_pos_bins(self) -> int:
    #     "the total number of unique position bins along the track, unrelated to the number of *positions* in `flat_positions`"
    #     return len(self.pos_bin_edges)-1


    # @property
    # def n_diff_bins(self) -> int:
    #     return len(self.first_order_diff_lst)

    @property
    def selected_indicies(self) -> List[int]:
        return np.unique(list(self.selected_bins.keys())).tolist()


    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self.fig = ax.figure
        self.selected_bins = {}
        self.crosshair = self.ax.axvline(x=0, color='r', linestyle='--')
        self.rects = []

        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)

    def on_mouse_move(self, event):
        if event.inaxes == self.ax:
            self.crosshair.set_xdata(event.xdata)
            self.fig.canvas.draw_idle()

    def on_mouse_click(self, event):
        if event.inaxes == self.ax and event.button == 1:  # Left mouse button
            time_bin = int(event.xdata)
            if time_bin in self.selected_bins:
                self.deselect_bin(time_bin)
            else:
                self.select_bin(time_bin)
            print(f"Selected time bins: {list(self.selected_bins.keys())}")

    def select_bin(self, bin_index):
        rect = self.ax.axvspan((bin_index - 0.0), (bin_index + 1.0), color='yellow', alpha=0.3)
        self.selected_bins[bin_index] = rect
        self.fig.canvas.draw_idle()

    def deselect_bin(self, bin_index):
        rect = self.selected_bins.pop(bin_index, None)
        if rect:
            rect.remove()
            self.fig.canvas.draw_idle()

    @classmethod
    def draw_bins(cls, ax, bin_index):
        rect = ax.axvspan((bin_index - 0.0), (bin_index + 1.0), color='yellow', alpha=0.3)
        # selected_bins[bin_index] = rect
        fig = ax.figure
        fig.canvas.draw_idle()
        return rect




@function_attributes(short_name=None, tags=['endcap', 'track_identity'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-20 19:08', related_items=[])
def classify_pos_bins(x: NDArray):
    """	classifies the pos_bin_edges as being either endcaps/on the main straightaway, stc and returns a dataframe

    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import classify_pos_bins

        pos_bin_edges = deepcopy(track_templates.get_decoders_dict()['long_LR'].xbin_centers)
        pos_classification_df = classify_pos_bins(x=pos_bin_edges)
        pos_classification_df

    """
    long_track_inst, short_track_inst = LinearTrackInstance.init_tracks_from_session_config(curr_active_pipeline.sess.config)
    ## test xbins
    pos_bin_classification = [long_track_inst.classify_x_position(x) for x in pos_bin_edges]
    is_pos_bin_endcap = [long_track_inst.classify_x_position(x).is_endcap for x in pos_bin_edges]
    is_pos_bin_on_maze = [long_track_inst.classify_x_position(x).is_on_maze for x in pos_bin_edges]
    # is_pos_bin_endcap
    # is_pos_bin_on_maze

    # Create long track classification DataFrame
    long_data = pd.DataFrame({
        'is_endcap': [long_track_inst.classify_x_position(x).is_endcap for x in pos_bin_edges],
        'is_track_straightaway': [long_track_inst.classify_x_position(x).is_track_straightaway for x in pos_bin_edges],
        'is_off_track': [(not long_track_inst.classify_x_position(x).is_on_maze) for x in pos_bin_edges],
    })

    # Create short track classification DataFrame
    short_data = pd.DataFrame({
        'is_endcap': [short_track_inst.classify_x_position(x).is_endcap for x in pos_bin_edges],
        'is_track_straightaway': [short_track_inst.classify_x_position(x).is_track_straightaway for x in pos_bin_edges],
        'is_off_track': [(not short_track_inst.classify_x_position(x).is_on_maze) for x in pos_bin_edges],
    })

    # Combine into a multi-level column DataFrame
    pos_classification_df = pd.concat(
        [pd.DataFrame({'x': pos_bin_edges, 'flat_index': range(len(pos_bin_edges))}),
        pd.concat({'long': long_data, 'short': short_data}, axis=1)],
        axis=1
    )

    # Ensure columns are correctly nested
    pos_classification_df.columns = pd.MultiIndex.from_tuples(
        [(col if isinstance(col, str) else col[0], '' if isinstance(col, str) else col[1]) for col in pos_classification_df.columns]
    )

    # combined_df['long']
    return pos_classification_df


# ==================================================================================================================== #
# 2024-12-18 Heuristic Evaluation in the continuous timeline                                                           #
# ==================================================================================================================== #


# ==================================================================================================================== #
# 2024-12-17 Heuristic Evaluation and Filtering Helpers                                                                #
# ==================================================================================================================== #
@function_attributes(short_name=None, tags=['heuristic_filter', 'heuristic', 'plotting'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-12-17 09:51', related_items=[])
def _plot_heuristic_evaluation_epochs(curr_active_pipeline, track_templates, filtered_decoder_filter_epochs_decoder_result_dict, ripple_merged_complete_epoch_stats_df: pd.DataFrame):
    """ Plots two GUI Windows: one with the high-heuristic-score epochs, and the other with the lows


    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _plot_heuristic_evaluation_epochs

    app, (high_heuristic_paginated_multi_decoder_decoded_epochs_window, high_heuristic_pagination_controller_dict), (low_heuristic_paginated_multi_decoder_decoded_epochs_window, low_heuristic_pagination_controller_dict) = _plot_heuristic_evaluation_epochs(curr_active_pipeline, track_templates, filtered_decoder_filter_epochs_decoder_result_dict, ripple_merged_complete_epoch_stats_df=ripple_merged_complete_epoch_stats_df)

    """
    from neuropy.utils.indexing_helpers import flatten, NumpyHelpers, PandasHelpers
    from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicThresholdFiltering
    from neuropy.utils.matplotlib_helpers import get_heatmap_cmap
    from pyphocorehelpers.gui.Qt.color_helpers import ColormapHelpers, ColorFormatConverter
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import FixedCustomColormaps
    from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import PhoPaginatedMultiDecoderDecodedEpochsWindow, DecodedEpochSlicesPaginatedFigureController, EpochSelectionsObject, ClickActionCallbacks
    from pyphoplacecellanalysis.GUI.Qt.Widgets.ThinButtonBar.ThinButtonBarWidget import ThinButtonBarWidget
    from pyphoplacecellanalysis.GUI.Qt.Widgets.PaginationCtrl.PaginationControlWidget import PaginationControlWidget, PaginationControlWidgetState
    from neuropy.core.user_annotations import UserAnnotationsManager
    from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons, silx_resources_rc

    # active_cmap = FixedCustomColormaps.get_custom_orange_with_low_values_dropped_cmap()
    # active_cmap = FixedCustomColormaps.get_custom_black_with_low_values_dropped_cmap(low_value_cutoff=0.05)
    # active_cmap = ColormapHelpers.create_colormap_transparent_below_value(active_cmap, low_value_cuttoff=0.1)
    active_cmap = FixedCustomColormaps.get_custom_greyscale_with_low_values_dropped_cmap(low_value_cutoff=0.05, full_opacity_threshold=0.4)

    ## filter by 'is_valid_epoch' first:
    ripple_merged_complete_epoch_stats_df = ripple_merged_complete_epoch_stats_df[ripple_merged_complete_epoch_stats_df['is_valid_epoch']] ## 136, 71 included requiring both

    ## filter by `included_epoch_indicies`
    # filter_thresholds_dict = {'mseq_len_ignoring_intrusions': 5, 'mseq_tcov': 0.35}
    # df_is_included_criteria_fn = lambda df: NumpyHelpers.logical_and(*[(df[f'overall_best_{a_col_name}'] >= a_thresh) for a_col_name, a_thresh in filter_thresholds_dict.items()])
    # included_heuristic_ripple_start_times = ripple_merged_complete_epoch_stats_df[df_is_included_criteria_fn(ripple_merged_complete_epoch_stats_df)]['ripple_start_t'].values
    # high_heuristic_only_filtered_decoder_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = {a_name:a_result.filtered_by_epoch_times(included_heuristic_ripple_start_times) for a_name, a_result in filtered_decoder_filter_epochs_decoder_result_dict.items()} # working filtered
    ripple_merged_complete_epoch_stats_df, (included_heuristic_ripple_start_times, excluded_heuristic_ripple_start_times) = HeuristicThresholdFiltering.add_columns(df=ripple_merged_complete_epoch_stats_df)
    high_heuristic_only_filtered_decoder_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = {a_name:a_result.filtered_by_epoch_times(included_heuristic_ripple_start_times) for a_name, a_result in filtered_decoder_filter_epochs_decoder_result_dict.items()} # working filtered
    low_heuristic_only_filtered_decoder_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = {a_name:a_result.filtered_by_epoch_times(excluded_heuristic_ripple_start_times) for a_name, a_result in filtered_decoder_filter_epochs_decoder_result_dict.items()} # working filtered

    example_decoder_name = 'long_LR'
    all_epoch_result: DecodedFilterEpochsResult = deepcopy(filtered_decoder_filter_epochs_decoder_result_dict[example_decoder_name])
    all_filter_epochs_df: pd.DataFrame = deepcopy(all_epoch_result.filter_epochs)

    included_filter_epoch_result: DecodedFilterEpochsResult = deepcopy(high_heuristic_only_filtered_decoder_filter_epochs_decoder_result_dict[example_decoder_name])
    # included_filter_epoch_result: DecodedFilterEpochsResult = deepcopy(low_heuristic_only_filtered_decoder_filter_epochs_decoder_result_dict[example_decoder_name])

    included_filter_epochs_df: pd.DataFrame = deepcopy(included_filter_epoch_result.filter_epochs)
    # included_filter_epoch_times = included_filter_epochs_df[['start', 'stop']].to_numpy() # Both 'start', 'stop' column matching
    included_filter_epoch_times = included_filter_epochs_df['start'].to_numpy() # Both 'start', 'stop' column matching

    # included_filter_epoch_times_to_all_epoch_index_map = included_filter_epoch_result.find_epoch_times_to_data_indicies_map(epoch_times=included_filter_epoch_times)
    included_filter_epoch_times_to_all_epoch_index_arr: NDArray = included_filter_epoch_result.find_data_indicies_from_epoch_times(epoch_times=included_filter_epoch_times)

    ## OUTPUTS: all_filter_epochs_df, all_filter_epochs_df
    ## OUTPUTS: included_filter_epoch_times_to_all_epoch_index_arr
    common_data_overlay_included_columns=['P_decoder', #'ratio_jump_valid_bins',
                    #    'wcorr',
    #'avg_jump_cm', 'max_jump_cm',
        'mseq_len', 'mseq_len_ignoring_intrusions', 'mseq_tcov', 'mseq_tdist', # , 'mseq_len_ratio_ignoring_intrusions_and_repeats', 'mseq_len_ignoring_intrusions_and_repeats'
    ]

    common_params_kwargs={'enable_per_epoch_action_buttons': False,
            'skip_plotting_most_likely_positions': True, 'skip_plotting_measured_positions': True,
            'enable_decoded_most_likely_position_curve': False,
            'enable_decoded_sequence_and_heuristics_curve': True, 'show_pre_merged_debug_sequences': True,
                'enable_radon_transform_info': False, 'enable_weighted_correlation_info': True, 'enable_weighted_corr_data_provider_modify_axes_rect': False,
            # 'enable_radon_transform_info': False, 'enable_weighted_correlation_info': False,
            # 'disable_y_label': True,
            'isPaginatorControlWidgetBackedMode': True,
            'enable_update_window_title_on_page_change': False, 'build_internal_callbacks': True,
            # 'debug_print': True,
            'max_subplots_per_page': 9,
            # 'scrollable_figure': False,
            'scrollable_figure': True,
            # 'posterior_heatmap_imshow_kwargs': dict(vmin=0.0075),
            'use_AnchoredCustomText': False,
            'should_suppress_callback_exceptions': False,
            # 'build_fn': 'insets_view',
            'track_length_cm_dict': deepcopy(track_templates.get_track_length_dict()),
            'posterior_heatmap_imshow_kwargs': dict(cmap=active_cmap), # , vmin=0.1, vmax=1.0
    }

    app, high_heuristic_paginated_multi_decoder_decoded_epochs_window, high_heuristic_pagination_controller_dict = PhoPaginatedMultiDecoderDecodedEpochsWindow.init_from_track_templates(curr_active_pipeline, track_templates,
        # decoder_decoded_epochs_result_dict=decoder_ripple_filter_epochs_decoder_result_dict, epochs_name='ripple',
        # decoder_decoded_epochs_result_dict=filtered_decoder_filter_epochs_decoder_result_dict, epochs_name='ripple', title='High-sequence Score Ripples Only',
        decoder_decoded_epochs_result_dict=high_heuristic_only_filtered_decoder_filter_epochs_decoder_result_dict, epochs_name='ripple', title='High-Heuristic Score Ripples Only', ## RIPPLE
        included_epoch_indicies=None, ## NO FILTERING
        # included_epoch_indicies=included_filter_epoch_times_to_all_epoch_index_arr, ## unsorted
        # decoder_decoded_epochs_result_dict=sorted_filtered_decoder_filter_epochs_decoder_result_dict, epochs_name='ripple',  ## SORTED
        # included_epoch_indicies=sorted_included_filter_epoch_times_to_all_epoch_index_arr, ## SORTED
        debug_print=False,
        params_kwargs=common_params_kwargs)
    high_heuristic_paginated_multi_decoder_decoded_epochs_window.add_data_overlays(included_columns=common_data_overlay_included_columns, defer_refresh=False)
    high_heuristic_paginated_multi_decoder_decoded_epochs_window.setWindowTitle('High-Heuristic Score DecodedEpochs Only')


    app, low_heuristic_paginated_multi_decoder_decoded_epochs_window, low_heuristic_pagination_controller_dict = PhoPaginatedMultiDecoderDecodedEpochsWindow.init_from_track_templates(curr_active_pipeline, track_templates,
        decoder_decoded_epochs_result_dict=low_heuristic_only_filtered_decoder_filter_epochs_decoder_result_dict, epochs_name='ripple', title='Low-Heuristic Score Ripples Only', ## RIPPLE
        included_epoch_indicies=None, ## NO FILTERING
        debug_print=False,
        params_kwargs=common_params_kwargs)
    low_heuristic_paginated_multi_decoder_decoded_epochs_window.add_data_overlays(included_columns=common_data_overlay_included_columns, defer_refresh=False)
    low_heuristic_paginated_multi_decoder_decoded_epochs_window.setWindowTitle('LOW-Heuristic Score DecodedEpochs Only')

    return app, (high_heuristic_paginated_multi_decoder_decoded_epochs_window, high_heuristic_pagination_controller_dict), (low_heuristic_paginated_multi_decoder_decoded_epochs_window, low_heuristic_pagination_controller_dict)






# ==================================================================================================================== #
# 2024-11-25 - Save/Load Heuristic Helpers                                                                             #
# ==================================================================================================================== #
from pyphocorehelpers.programming_helpers import function_attributes

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates

class SerializationHelperBaseClass:
    @classmethod
    def save(cls, *args, **kwargs):
        raise NotImplementedError(f'Implementors must override')

    @classmethod
    def load(cls, load_path: Path):
        raise NotImplementedError(f'Implementors must override')



class SerializationHelper_CustomDecodingResults(SerializationHelperBaseClass):
    @function_attributes(short_name=None, tags=['save', 'export'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-25 12:58', related_items=[])
    @classmethod
    def save(cls, a_directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult, long_pf2D, save_path, debug_print=False):
        """ Used for "2024-08-01 - Heuristic Analysis.ipynb"
        Usage:
            directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = deepcopy(curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersEpochsEvaluations']) ## GENERAL
            a_decoded_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = deepcopy(directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict)
            a_decoded_filter_epochs_decoder_result_dict, _out_new_scores = HeuristicReplayScoring.compute_all_heuristic_scores(track_templates=track_templates, a_decoded_filter_epochs_decoder_result_dict=a_decoded_filter_epochs_decoder_result_dict)
            save_path = curr_active_pipeline.get_output_path().joinpath(f"{DAY_DATE_TO_USE}_CustomDecodingResults.pkl").resolve()
            save_path = save_CustomDecodingResults(a_directional_decoders_epochs_decode_result=directional_decoders_epochs_decode_result, long_pf2D=long_pf2D,
                                                    save_path=save_path)
            save_path


        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData

        xbin = deepcopy(long_pf2D.xbin)
        xbin_centers = deepcopy(long_pf2D.xbin_centers)
        ybin = deepcopy(long_pf2D.ybin)
        ybin_centers = deepcopy(long_pf2D.ybin_centers)

        if debug_print:
            print(xbin_centers)
        save_dict = {
        'directional_decoders_epochs_decode_result': a_directional_decoders_epochs_decode_result.__getstate__(),
        'xbin': xbin, 'xbin_centers': xbin_centers}

        saveData(save_path, save_dict)
        if debug_print:
            print(f'save_path: {save_path}')
        return save_path


    @function_attributes(short_name=None, tags=['load', 'import'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-25 12:58', related_items=[])
    @classmethod
    def load(cls, load_path: Path):
        """ Used for "2024-08-01 - Heuristic Analysis.ipynb"
        Usage:
            directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = deepcopy(curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersEpochsEvaluations']) ## GENERAL
            a_decoded_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = deepcopy(directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict)
            a_decoded_filter_epochs_decoder_result_dict, _out_new_scores = HeuristicReplayScoring.compute_all_heuristic_scores(track_templates=track_templates, a_decoded_filter_epochs_decoder_result_dict=a_decoded_filter_epochs_decoder_result_dict)
            save_path = curr_active_pipeline.get_output_path().joinpath(f"{DAY_DATE_TO_USE}_CustomDecodingResults.pkl").resolve()
            save_path = save_CustomDecodingResults(a_directional_decoders_epochs_decode_result=directional_decoders_epochs_decode_result, long_pf2D=long_pf2D,
                                                    save_path=save_path)
            save_path


        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import loadData
        assert load_path.exists()
        # loaded_dict = loadData(load_path, debug_print=False)
        # print_keys_if_possible('loaded_dict', loaded_dict)

        base_loaded_dict = loadData(load_path, debug_print=False)
        xbin = base_loaded_dict.pop('xbin', None)
        xbin_centers = base_loaded_dict.pop('xbin_centers', None)
        # ybin = deepcopy(long_pf2D.ybin)
        # ybin_centers = deepcopy(long_pf2D.ybin_centers)
        print(f"xbin_centers: {xbin_centers}")

        loaded_dict = base_loaded_dict['directional_decoders_epochs_decode_result']

        ## UNPACK HERE:
        pos_bin_size: float = loaded_dict['pos_bin_size'] # 3.8632841399651463
        ripple_decoding_time_bin_size = loaded_dict['ripple_decoding_time_bin_size']
        laps_decoding_time_bin_size = loaded_dict['laps_decoding_time_bin_size']
        decoder_laps_filter_epochs_decoder_result_dict = loaded_dict['decoder_laps_filter_epochs_decoder_result_dict']
        decoder_ripple_filter_epochs_decoder_result_dict = loaded_dict['decoder_ripple_filter_epochs_decoder_result_dict']
        decoder_laps_radon_transform_df_dict = loaded_dict['decoder_laps_radon_transform_df_dict']
        decoder_ripple_radon_transform_df_dict = loaded_dict['decoder_ripple_radon_transform_df_dict']
        ## New 2024-02-14 - Noon:
        decoder_laps_radon_transform_extras_dict = loaded_dict['decoder_laps_radon_transform_extras_dict']
        decoder_ripple_radon_transform_extras_dict = loaded_dict['decoder_ripple_radon_transform_extras_dict']

        laps_weighted_corr_merged_df = loaded_dict['laps_weighted_corr_merged_df']
        ripple_weighted_corr_merged_df = loaded_dict['ripple_weighted_corr_merged_df']
        laps_simple_pf_pearson_merged_df = loaded_dict['laps_simple_pf_pearson_merged_df']
        ripple_simple_pf_pearson_merged_df = loaded_dict['ripple_simple_pf_pearson_merged_df']

        _VersionedResultMixin_version = loaded_dict.pop('_VersionedResultMixin_version', None)

        directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = DecoderDecodedEpochsResult(**loaded_dict)
        # {'ripple_decoding_time_bin_size':ripple_decoding_time_bin_size, 'laps_decoding_time_bin_size':laps_decoding_time_bin_size, 'decoder_laps_filter_epochs_decoder_result_dict':decoder_laps_filter_epochs_decoder_result_dict, 'decoder_ripple_filter_epochs_decoder_result_dict':decoder_ripple_filter_epochs_decoder_result_dict, 'decoder_laps_radon_transform_df_dict':decoder_laps_radon_transform_df_dict, 'decoder_ripple_radon_transform_df_dict':decoder_ripple_radon_transform_df_dict}

        return directional_decoders_epochs_decode_result, xbin, xbin_centers



class SerializationHelper_AllCustomDecodingResults(SerializationHelperBaseClass):
    """

    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import SerializationHelper_AllCustomDecodingResults, SerializationHelper_CustomDecodingResults
    load_path = Path("W:/Data/KDIBA/gor01/one/2006-6-09_1-22-43/output/2024-11-25_AllCustomDecodingResults.pkl")
    track_templates, directional_decoders_epochs_decode_result, xbin, xbin_centers =  SerializationHelper_AllCustomDecodingResults.load(load_path=load_path)
    pos_bin_size = directional_decoders_epochs_decode_result.pos_bin_size

    """
    @function_attributes(short_name=None, tags=['save', 'export'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-25 12:58', related_items=[])
    @classmethod
    def save(cls, track_templates: TrackTemplates, a_directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult,
                                    #    a_decoded_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult],
                                    save_path: Path, **kwargs):
        """ Used for "2024-08-01 - Heuristic Analysis.ipynb"

        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData
        xbin = deepcopy(track_templates.long_LR_decoder.xbin)
        xbin_centers = deepcopy(track_templates.long_LR_decoder.xbin_centers)
        save_dict = {
            'track_templates': deepcopy(track_templates),
            'directional_decoders_epochs_decode_result': a_directional_decoders_epochs_decode_result.__getstate__(),
            # 'directional_decoders_epochs_decode_result': {k:a_directional_decoders_epochs_decode_result.__getstate__() for k, a_directional_decoders_epochs_decode_result in a_decoded_filter_epochs_decoder_result_dict.items()},
            'xbin': xbin, 'xbin_centers': xbin_centers,
            **kwargs
        }
        saveData(save_path, save_dict)
        print(f'save_path: {save_path}')
        return save_path

    @function_attributes(short_name=None, tags=['load', 'import'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-25 12:58', related_items=[])
    @classmethod
    def load(cls, load_path: Path):
        """ Used for "2024-08-01 - Heuristic Analysis.ipynb"
        Usage:
            load_path = Path("W:/Data/KDIBA/gor01/one/2006-6-09_1-22-43/output/2024-11-25_AllCustomDecodingResults.pkl")
            track_templates, directional_decoders_epochs_decode_result, xbin, xbin_centers =  SerializationHelper_AllCustomDecodingResults.load(load_path=load_path)
            pos_bin_size = directional_decoders_epochs_decode_result.pos_bin_size

        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import loadData
        assert load_path.exists()
        # loaded_dict = loadData(load_path, debug_print=False)
        # print_keys_if_possible('loaded_dict', loaded_dict)

        base_loaded_dict = loadData(load_path, debug_print=False)
        xbin = base_loaded_dict.pop('xbin', None)
        xbin_centers = base_loaded_dict.pop('xbin_centers', None)
        # ybin = deepcopy(long_pf2D.ybin)
        # ybin_centers = deepcopy(long_pf2D.ybin_centers)
        print(f"xbin_centers: {xbin_centers}")

        loaded_dict = base_loaded_dict['directional_decoders_epochs_decode_result']

        track_templates = base_loaded_dict['track_templates']

        ## UNPACK HERE:
        pos_bin_size: float = loaded_dict['pos_bin_size'] # 3.8632841399651463
        ripple_decoding_time_bin_size = loaded_dict['ripple_decoding_time_bin_size']
        laps_decoding_time_bin_size = loaded_dict['laps_decoding_time_bin_size']
        decoder_laps_filter_epochs_decoder_result_dict = loaded_dict['decoder_laps_filter_epochs_decoder_result_dict']
        decoder_ripple_filter_epochs_decoder_result_dict = loaded_dict['decoder_ripple_filter_epochs_decoder_result_dict']
        decoder_laps_radon_transform_df_dict = loaded_dict['decoder_laps_radon_transform_df_dict']
        decoder_ripple_radon_transform_df_dict = loaded_dict['decoder_ripple_radon_transform_df_dict']
        ## New 2024-02-14 - Noon:
        decoder_laps_radon_transform_extras_dict = loaded_dict['decoder_laps_radon_transform_extras_dict']
        decoder_ripple_radon_transform_extras_dict = loaded_dict['decoder_ripple_radon_transform_extras_dict']

        laps_weighted_corr_merged_df = loaded_dict['laps_weighted_corr_merged_df']
        ripple_weighted_corr_merged_df = loaded_dict['ripple_weighted_corr_merged_df']
        laps_simple_pf_pearson_merged_df = loaded_dict['laps_simple_pf_pearson_merged_df']
        ripple_simple_pf_pearson_merged_df = loaded_dict['ripple_simple_pf_pearson_merged_df']

        _VersionedResultMixin_version = loaded_dict.pop('_VersionedResultMixin_version', None)

        directional_decoders_epochs_decode_result: DecoderDecodedEpochsResult = DecoderDecodedEpochsResult(**loaded_dict)
        # {'ripple_decoding_time_bin_size':ripple_decoding_time_bin_size, 'laps_decoding_time_bin_size':laps_decoding_time_bin_size, 'decoder_laps_filter_epochs_decoder_result_dict':decoder_laps_filter_epochs_decoder_result_dict, 'decoder_ripple_filter_epochs_decoder_result_dict':decoder_ripple_filter_epochs_decoder_result_dict, 'decoder_laps_radon_transform_df_dict':decoder_laps_radon_transform_df_dict, 'decoder_ripple_radon_transform_df_dict':decoder_ripple_radon_transform_df_dict}

        return track_templates, directional_decoders_epochs_decode_result, xbin, xbin_centers

# ==================================================================================================================== #
# 2024-11-07 - PhoJonathan first-spike indicator lines                                                                 #
# ==================================================================================================================== #
import neuropy.utils.type_aliases as types

# DecoderName = NewType('DecoderName', str)

def add_time_indicator_lines(active_figures_dict, later_lap_appearing_aclus_times_dict: Dict[types.aclu_index, Dict[str, float]], time_point_formatting_kwargs_dict=None, defer_draw: bool=False):
    """
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import add_time_indicator_lines
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import CellsFirstSpikeTimes
        from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
        from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import BatchPhoJonathanFiguresHelper

        ## INPUTS: cells_first_spike_times

        later_lap_appearing_aclus = [32, 33,34, 35, 62, 67]
        filtered_cells_first_spike_times: CellsFirstSpikeTimes = cells_first_spike_times.sliced_by_neuron_id(later_lap_appearing_aclus)

        later_lap_appearing_aclus_df = filtered_cells_first_spike_times.all_cells_first_spike_time_df ## find ones that appear only on later laps
        later_lap_appearing_aclus = later_lap_appearing_aclus_df['aclu'].to_numpy() ## get the aclus that only appear on later laps

        ## plot each aclu in a separate figures
        later_lap_appearing_figures_dict: Dict[IdentifyingContext, MatplotlibRenderPlots] = BatchPhoJonathanFiguresHelper.perform_run(curr_active_pipeline, shared_aclus=later_lap_appearing_aclus, n_max_page_rows=1, disable_top_row=True,
                                                                                                                                    #    progress_print=True, write_png=True, write_vector_format=True,
                                                                                                                                    )
        ## Inputs: later_lap_appearing_aclus_df
        time_point_formatting_kwargs_dict = {'lap': dict(color='orange', alpha=0.8), 'PBE': dict(color='purple', alpha=0.8)}
        later_lap_appearing_aclus_times_dict: Dict[types.aclu_index, Dict[str, float]] = {aclu_tuple.aclu:{'lap': aclu_tuple.first_spike_lap, 'PBE': aclu_tuple.first_spike_PBE} for aclu_tuple in later_lap_appearing_aclus_df.itertuples(index=False)}

        # ## add the lines:
        add_time_indicator_lines(later_lap_appearing_figures_dict, later_lap_appearing_aclus_times_dict=later_lap_appearing_aclus_times_dict, time_point_formatting_kwargs_dict=time_point_formatting_kwargs_dict, defer_draw=False)

    """
    ## INPUTS: later_lap_appearing_figures_dict, later_lap_appearing_aclus_df
    if time_point_formatting_kwargs_dict is None:
        time_point_formatting_kwargs_dict = {'lap': dict(color='orange', alpha=0.8), 'PBE': dict(color='purple', alpha=0.8)}

    #
    # _out_dict = {}
    modified_figures_dict = {}

    for fig_page_context, container in active_figures_dict.items():
        ## Add the first-spike time point indicator lines to each of the aclus:
        # container: MatplotlibRenderPlots = list(later_lap_appearing_figures_dict.values())[0]
        container.plot_data['first_spike_indicator_lines'] = {} # empty
        ## for a single container/figure, parse back into the real aclu value from the axes name
        lap_spikes_axs = [v['lap_spikes'] for v in container.axes]
        laps_spikes_aclu_ax_dict = {int(ax.get_label().removeprefix('ax_lap_spikes[').removesuffix(']')):ax for ax in lap_spikes_axs}
        # aclu_first_lap_spike_time_dict = dict(zip(later_lap_appearing_aclus_df['aclu'].values, later_lap_appearing_aclus_df['first_spike_lap'].values))
        ## OUTPUT: laps_spikes_aclu_ax_dict

        for aclu, ax in laps_spikes_aclu_ax_dict.items():
            lap_first_spike_lines = {}
            # _temp_df = later_lap_appearing_aclus_df[later_lap_appearing_aclus_df['aclu'] == aclu][['first_spike_lap', 'first_spike_PBE']]
            # lap_time_point = _temp_df['first_spike_lap'].to_numpy()[0]
            # pbe_time_point = _temp_df['first_spike_PBE'].to_numpy()[0]
            # time_point_dict = {'lap': lap_time_point, 'PBE': pbe_time_point}
            time_point_dict = later_lap_appearing_aclus_times_dict[aclu]
            ylims = deepcopy(laps_spikes_aclu_ax_dict[aclu].get_ylim())
            # print(f'time_point: {time_point}')
            for name, time_point in time_point_dict.items():
                lap_first_spike_lines[name] = {}
                common_formatting_kwargs = time_point_formatting_kwargs_dict[name] # could do .get(name, dict(color='black', alpha=1.0)) to provide defaults
                # Draw vertical line
                lap_first_spike_lines[name]['vline'] = ax.axvline(x=time_point, linewidth=1, **common_formatting_kwargs)
                lap_first_spike_lines[name]['triangle_marker'] = ax.plot(time_point, ylims[-1]-10, marker='v', markersize=10, **common_formatting_kwargs)  # 'v' for downward triangle

            ax.set_ybound(*ylims)
            # if not defer_render:
            #     fig = ax.get_figure().get_figure() # For SubFigure
            #     fig.canvas.draw()


            # _out_dict[aclu] = lap_first_spike_lines
            container.plot_data['first_spike_indicator_lines'][aclu] = lap_first_spike_lines
        ## end for aclu, ax
        # container.plot_data['first_spike_indicator_lines'] = _out_dict
        ## redraw all figures in this container
        for fig in container.figures:
            fig.canvas.draw()   # Redraw the current figure



        modified_context = fig_page_context.overwriting_context(modification='first_spike')
        modified_figures_dict[modified_context] = container # deepcopy(container)
    # end for fig_page_context, container

    if not defer_draw:
        plt.draw()

    return modified_figures_dict #container.plot_data['first_spike_indicator_lines']



# ==================================================================================================================== #
# 2024-11-07 - Spike Stationarity Testing                                                                              #
# ==================================================================================================================== #
from statsmodels.tsa.stattools import adfuller, kpss

@metadata_attributes(short_name=None, tags=['stationary', 'stationarity', 'time-series', 'statistics'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-07 13:45', related_items=['perform_timeseries_stationarity_tests'])
@define
class ADFResult:
    """ Augmented Dickey-Fuller (ADF) Test for non-stationarity of a timeseries"""
    adf_statistic: float
    pvalue: float
    usedlag: int
    nobs: int
    critical_values: dict
    icbest: float

    @classmethod
    def from_tuple(cls, result_tuple):
        return cls(*result_tuple)

    def print_summary(self):
        """Prints the ADF test results and interpretation."""
        # Print the results
        print(f'ADF Statistic: {self.adf_statistic}')
        print(f'p-value: {self.pvalue}')
        print('Critical Values:')
        for key, value in self.critical_values.items():
            print(f'\t{key}: {value:.3f}')

        # Interpretation
        adf_critical_value_5perc = self.critical_values['5%']
        if (self.adf_statistic < adf_critical_value_5perc) or (self.pvalue < 0.05):
            print('ADF Test Conclusion:')
            print('\tReject the null hypothesis (series is stationary).')
        else:
            print('ADF Test Conclusion:')
            print('\tFail to reject the null hypothesis (series is non-stationary).')

@metadata_attributes(short_name=None, tags=['stationary', 'stationarity', 'time-series', 'statistics'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-07 13:44', related_items=['perform_timeseries_stationarity_tests'])
@define
class KPSSResult:
    """ Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test for non-stationarity of a timeseries"""
    kpss_statistic: float
    pvalue: float
    nlags: int
    critical_values: dict

    @classmethod
    def from_tuple(cls, result_tuple):
        return cls(*result_tuple)

    def print_summary(self):
        """Prints the KPSS test results and interpretation."""
        # Print the results
        print(f'KPSS Statistic: {self.kpss_statistic}')
        print(f'p-value: {self.pvalue}')
        print('Critical Values:')
        for key, value in self.critical_values.items():
            print(f'\t{key}: {value:.3f}')

        # Interpretation
        kpss_critical_value_5perc = float(self.critical_values['5%'])
        if (self.kpss_statistic < kpss_critical_value_5perc) and (self.pvalue > 0.05):
            print('KPSS Test Conclusion:')
            print('\tFail to reject the null hypothesis (series is stationary).')
        else:
            print('KPSS Test Conclusion:')
            print('\tReject the null hypothesis (series is non-stationary).')


@function_attributes(short_name=None, tags=['stationary', 'stationarity', 'time-series', 'statistics'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-07 13:44', related_items=[])
def perform_timeseries_stationarity_tests(time_series) -> Tuple[ADFResult, KPSSResult]:
    """Tests the time series for stationarity using ADF and KPSS tests.
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import perform_timeseries_stationarity_tests, ADFResult, KPSSResult
        time_series = get_proper_global_spikes_df(curr_active_pipeline).t_rel_seconds.to_numpy()
        # Perform the stationarity tests
        adf_result, kpss_result = perform_timeseries_stationarity_tests(time_series)
    """
    # Augmented Dickey-Fuller (ADF) Test
    adf_result_tuple = adfuller(time_series) # (-3.5758600257897317, 0.0062396609756376, 35, 144596, {'1%': -3.4303952254287307, '5%': -2.86155998899889, '10%': -2.5667806394328094}, -681134.6488980348)
    adf_result = ADFResult.from_tuple(adf_result_tuple)

    # Print ADF results
    adf_result.print_summary()
    print('\n')  # Add space between tests

    # Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
    kpss_result_tuple = kpss(time_series, regression='c')
    kpss_result = KPSSResult.from_tuple(kpss_result_tuple)

    # Print KPSS results
    kpss_result.print_summary()

    return adf_result, kpss_result





# ==================================================================================================================== #
# 2024-11-04 - Moving Misplaced Pickles on GL                                                                          #
# ==================================================================================================================== #
import shutil

def try_perform_move(src_file, target_file, is_dryrun: bool, allow_overwrite_existing: bool):
    """ tries to move the file from src_file to target_file """
    print(f'try_perform_move(src_file: "{src_file}", target_file: "{target_file}")')
    if not src_file.exists():
        print(f'\tsrc_file "{src_file}" does not exist!')
        return False
    else:
        if (target_file.exists() and (not allow_overwrite_existing)):
            print(f'\ttarget_file: "{target_file}" already exists!')
            return False
        else:
            # does not exist, safe to copy
            print(f'\t moving "{src_file}" -> "{target_file}"...')
            if not is_dryrun:
                shutil.move(src_file, target_file)
            else:
                print(f'\t\t(is_dryrun==True, so not actually moving.)')
            print(f'\t\tdone!')
            return True


@function_attributes(short_name=None, tags=['move', 'pickle', 'filesystem', 'GL'], input_requires=[], output_provides=[], uses=['try_perform_move'], used_by=[], creation_date='2024-11-04 19:41', related_items=[])
def try_move_pickle_files_on_GL(good_session_concrete_folders, session_basedirs_dict, computation_script_paths, excluded_session_keys=None, is_dryrun: bool=True, debug_print: bool=False, allow_overwrite_existing: bool=False):
    """
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import try_move_pickle_files_on_GL


    copy_dict, moved_dict, (all_found_pipeline_pkl_files_dict, all_found_global_pkl_files_dict, all_found_pipeline_h5_files_dict) = try_move_pickle_files_on_GL(good_session_concrete_folders, session_basedirs_dict, computation_script_paths,
             is_dryrun: bool=True, debug_print: bool=False)

    """
    ## INPUTS: good_session_concrete_folders, session_basedirs_dict, computation_script_paths
    # session_basedirs_dict: Dict[IdentifyingContext, Path] = {a_session_folder.context:a_session_folder.path for a_session_folder in good_session_concrete_folders}

    # is_dryrun: bool = False
    assert len(good_session_concrete_folders) == len(session_basedirs_dict)
    assert len(good_session_concrete_folders) == len(computation_script_paths)

    script_output_folders = [Path(v).parent for v in computation_script_paths]

    if excluded_session_keys is None:
        excluded_session_keys = []


    # excluded_session_keys = ['kdiba_pin01_one_fet11-01_12-58-54', 'kdiba_gor01_one_2006-6-08_14-26-15', 'kdiba_gor01_two_2006-6-07_16-40-19']
    excluded_session_contexts = [IdentifyingContext(**dict(zip(IdentifyingContext._get_session_context_keys(), v.split('_', maxsplit=3)))) for v in excluded_session_keys]

    # excluded_session_contexts

    # all_found_pkl_files_dict = {}
    all_found_pipeline_pkl_files_dict = {}
    all_found_global_pkl_files_dict = {}
    all_found_pipeline_h5_files_dict = {}

    copy_dict = {}
    moved_dict = {}

    # scripts_output_path
    for a_good_session_concrete_folder, a_session_basedir, a_script_folder in zip(good_session_concrete_folders, session_basedirs_dict, script_output_folders):
        if debug_print:
            print(f'a_good_session_concrete_folder: {a_good_session_concrete_folder}, a_session_basedir: {a_session_basedir}. a_script_folder: {a_script_folder}')
        if a_good_session_concrete_folder.context in excluded_session_contexts:
            if debug_print:
                print(f'skipping excluded session: {a_good_session_concrete_folder.context}')
        else:
            all_found_global_pkl_files_dict[a_session_basedir] = list(a_script_folder.glob('global_computation_results*.pkl'))

            for a_global_file in all_found_global_pkl_files_dict[a_session_basedir]:
                ## iterate through the found global files:
                target_file = a_good_session_concrete_folder.global_computation_result_pickle.with_name(a_global_file.name)
                copy_dict[a_global_file] = target_file
                # if not is_dryrun:
                ## perform the move/copy
                was_success = try_perform_move(src_file=a_global_file, target_file=target_file, is_dryrun=is_dryrun, allow_overwrite_existing=allow_overwrite_existing)
                if was_success:
                    moved_dict[a_global_file] = target_file
            all_found_pipeline_pkl_files_dict[a_session_basedir] = list(a_script_folder.glob('loadedSessPickle*.pkl'))
            for a_file in all_found_pipeline_pkl_files_dict[a_session_basedir]:
                ## iterate through the found global files:
                target_file = a_good_session_concrete_folder.session_pickle.with_name(a_file.name)
                copy_dict[a_file] = target_file
                # if not is_dryrun:
                ## perform the move/copy
                was_success = try_perform_move(src_file=a_file, target_file=target_file, is_dryrun=is_dryrun, allow_overwrite_existing=allow_overwrite_existing)
                if was_success:
                    moved_dict[a_file] = target_file
            all_found_pipeline_h5_files_dict[a_session_basedir] = list(a_script_folder.glob('loadedSessPickle*.h5'))
            for a_file in all_found_pipeline_h5_files_dict[a_session_basedir]:
                ## iterate through the found global files:
                target_file = a_good_session_concrete_folder.pipeline_results_h5.with_name(a_file.name)
                copy_dict[a_file] = target_file
                # if not is_dryrun:
                ## perform the move/copy
                was_success = try_perform_move(src_file=a_file, target_file=target_file, is_dryrun=is_dryrun, allow_overwrite_existing=allow_overwrite_existing)
                if was_success:
                    moved_dict[a_file] = target_file
            # all_found_pkl_files_dict[a_session_basedir] = find_pkl_files(a_script_folder)

    ## discover .pkl files in the root of each folder:
    # all_found_pipeline_pkl_files_dict
    # all_found_global_pkl_files_dict
    ## OUTPUTS: copy_dict
    # copy_dict
    return copy_dict, moved_dict, (all_found_pipeline_pkl_files_dict, all_found_global_pkl_files_dict, all_found_pipeline_h5_files_dict)


# ==================================================================================================================== #
# 2024-11-04 - Custom spike Drawing                                                                                    #
# ==================================================================================================================== #
def test_plotRaw_v_time(active_pf1D, cellind, speed_thresh=False, spikes_color=None, spikes_alpha=None, ax=None, position_plot_kwargs=None, spike_plot_kwargs=None,
    should_include_trajectory=True, should_include_spikes=True, should_include_filter_excluded_spikes=True, should_include_labels=True, use_filtered_positions=False, use_pandas_plotting=False, **kwargs):
    """ Builds one subplot for each dimension of the position data
    Updated to work with both 1D and 2D Placefields

    should_include_trajectory:bool - if False, will not try to plot the animal's trajectory/position
        NOTE: Draws the spike_positions actually instead of the continuously sampled animal position

    should_include_labels:bool - whether the plot should include text labels, like the title, axes labels, etc
    should_include_spikes:bool - if False, will not try to plot points for spikes
    use_pandas_plotting:bool = False
    use_filtered_positions:bool = False # If True, uses only the filtered positions (which are missing the end caps) and the default a.plot(...) results in connected lines which look bad.


    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import test_plotRaw_v_time


        _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')

        active_config = deepcopy(curr_active_pipeline.active_configs[global_epoch_name])
        active_pf1D = deepcopy(global_pf1D)

        fig = plt.figure(figsize=(23, 9.7), clear=True, num='test_plotRaw_v_time')
        # Need axes:
        # Layout Subplots in Figure:
        gs = fig.add_gridspec(1, 8)
        gs.update(wspace=0, hspace=0.05) # set the spacing between axes. # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
        ax_activity_v_time = fig.add_subplot(gs[0, :-1]) # all except the last element are the trajectory over time
        ax_pf_tuning_curve = fig.add_subplot(gs[0, -1], sharey=ax_activity_v_time) # The last element is the tuning curve
        # if should_include_labels:
            # ax_pf_tuning_curve.set_title('Normalized Placefield', fontsize='14')
        ax_pf_tuning_curve.set_xticklabels([])
        ax_pf_tuning_curve.set_yticklabels([])


        cellind: int = 2

        kwargs = {}
        # jitter the curve_value for each spike based on the time it occured along the curve:
        spikes_color_RGB = kwargs.get('spikes_color', (0, 0, 0))
        spikes_alpha = kwargs.get('spikes_alpha', 0.8)
        # print(f'spikes_color: {spikes_color_RGB}')
        should_plot_bins_grid = kwargs.get('should_plot_bins_grid', False)

        should_include_trajectory = kwargs.get('should_include_trajectory', True) # whether the plot should include
        should_include_labels = kwargs.get('should_include_labels', True) # whether the plot should include text labels, like the title, axes labels, etc
        should_include_plotRaw_v_time_spikes = kwargs.get('should_include_spikes', True) # whether the plot should include plotRaw_v_time-spikes, should be set to False to plot completely with the new all spikes mode
        use_filtered_positions: bool = kwargs.pop('use_filtered_positions', False)

        # position_plot_kwargs = {'color': '#393939c8', 'linewidth': 1.0, 'zorder':5} | kwargs.get('position_plot_kwargs', {}) # passed into `active_epoch_placefields1D.plotRaw_v_time`
        position_plot_kwargs = {'color': '#757575c8', 'linewidth': 1.0, 'zorder':5} | kwargs.get('position_plot_kwargs', {}) # passed into `active_epoch_placefields1D.plotRaw_v_time`


        # _out = test_plotRaw_v_time(active_pf1D=active_pf1D, cellind=cellind)
        # spike_plot_kwargs = {'linestyle':'none', 'markersize':5.0, 'marker': '.', 'markerfacecolor':spikes_color_RGB, 'markeredgecolor':spikes_color_RGB, 'zorder':10} ## OLDER
        spike_plot_kwargs = {'zorder':10} ## OLDER



        # active_pf1D.plotRaw_v_time(cellind, ax=ax_activity_v_time, spikes_alpha=spikes_alpha,
        # 	position_plot_kwargs=position_plot_kwargs,
        # 	spike_plot_kwargs=spike_plot_kwargs,
        # 	should_include_labels=should_include_labels, should_include_trajectory=should_include_trajectory, should_include_spikes=should_include_plotRaw_v_time_spikes,
        # 	use_filtered_positions=use_filtered_positions,
        # ) # , spikes_color=spikes_color, spikes_alpha=spikes_alpha

        _out = test_plotRaw_v_time(active_pf1D=active_pf1D, cellind=cellind, ax=ax_activity_v_time, spikes_alpha=spikes_alpha,
            position_plot_kwargs=position_plot_kwargs,
            spike_plot_kwargs=spike_plot_kwargs,
            should_include_labels=should_include_labels, should_include_trajectory=should_include_trajectory, should_include_spikes=should_include_plotRaw_v_time_spikes,
            use_filtered_positions=use_filtered_positions,
        )

        _out = _subfn_plot_pf1D_placefield(active_epoch_placefields1D=active_pf1D, placefield_cell_index=cellind,
                                        ax_activity_v_time=ax_activity_v_time, ax_pf_tuning_curve=ax_pf_tuning_curve, pf_tuning_curve_ax_position='right')
        _out


    # active_pf1D: ['spk_pos', 'spk_t', 'ndim', 'cell_ids', 'speed_thresh', 'position', '', '']

    """
    from scipy.signal import savgol_filter
    from neuropy.plotting.figure import pretty_plot
    from neuropy.utils.misc import is_iterable

    if ax is None:
        fig, ax = plt.subplots(active_pf1D.ndim, 1, sharex=True)
        fig.set_size_inches([23, 9.7])

    if not is_iterable(ax):
        ax = [ax]

    # plot trajectories
    pos_df = active_pf1D.position.to_dataframe()

    # self.x, self.y contain filtered positions, pos_df's columns contain all positions.
    if not use_pandas_plotting: # don't need to worry about 't' for pandas plotting, we'll just use the one in the dataframe.
        if use_filtered_positions:
            t = active_pf1D.t
        else:
            t = pos_df.t.to_numpy()

    if active_pf1D.ndim < 2:
        if not use_pandas_plotting:
            if use_filtered_positions:
                variable_array = [active_pf1D.x]
            else:
                variable_array = [pos_df.x.to_numpy()]
        else:
            variable_array = ['x']
        label_array = ["X position (cm)"]
    else:
        if not use_pandas_plotting:
            if use_filtered_positions:
                variable_array = [active_pf1D.x, active_pf1D.y]
            else:
                variable_array = [pos_df.x.to_numpy(), pos_df.y.to_numpy()]
        else:
            variable_array = ['x', 'y']
        label_array = ["X position (cm)", "Y position (cm)"]

    for a, pos, ylabel in zip(ax, variable_array, label_array):
        if should_include_trajectory:
            if not use_pandas_plotting:
                a.plot(t, pos, **(position_plot_kwargs or {}))
            else:
                pos_df.plot(x='t', y=pos, ax=a, legend=False, **(position_plot_kwargs or {})) # changed to pandas.plot because the filtered positions were missing the end caps, and the default a.plot(...) resulted in connected lines which looked bad.

        if should_include_labels:
            a.set_xlabel("Time (seconds)")
            a.set_ylabel(ylabel)
        pretty_plot(a)


    # Define the normal line function
    def normal_line(t_val: float, x_val: float, slope_normal: float, delta: float=0.5):
        """
        Computes points on the normal line at t_val.

        Parameters:
        - t_val: The time at which the normal is computed.
        - x_val: The position at t_val.
        - slope_normal: The slope of the normal line at t_val.
        - delta: The range around t_val to plot the normal line.

        Returns:
        - t_normal: Array of t values for the normal line.
        - y_normal: Array of y values for the normal line.
        - is_vertical: Boolean indicating if the normal line is vertical.
        """
        t_min: float = (t_val - delta)
        t_max: float = (t_val + delta)

        # t_normal = np.array([t_val, t_val])
        # t_normal = np.linspace(t_min, t_max, 10)

        if np.isinf(slope_normal):
            # Normal line is vertical
            t_normal = np.array([t_val, t_val])
            y_min = x_val - delta
            y_max = x_val + delta
            y_normal = np.array([y_min, y_max])
            is_vertical = True
        elif np.isclose(slope_normal, 0.0, atol=1e-3): # slope_normal == 0
            # Normal line is horizontal
            t_normal = np.linspace(t_min, t_max, 10)
            y_normal = np.full_like(t_normal, x_val)
            is_vertical = False
        else:
            t_normal = np.linspace(t_min, t_max, 10)
            y_normal = x_val + slope_normal * (t_normal - t_val)
            is_vertical = False
        return t_normal, y_normal, is_vertical


    # Compute the derivative dx/dt
    # Step 2: Smooth the Data Using Savitzky-Golay Filter
    window_length = 11  # Must be odd
    polyorder = 3
    x = deepcopy(pos)
    t_delta: float = (t[1] - t[0])
    override_t_delta: float = kwargs.get('override_t_delta', t_delta)

    x_smooth = savgol_filter(pos, window_length=window_length, polyorder=polyorder)
    dx_dt = savgol_filter(x, window_length=window_length, polyorder=polyorder, deriv=1, delta=override_t_delta)
    # dx_dt = np.gradient(x_smooth, t)  # Approximate derivative

    # tangents = []
    normals = []
    normal_ts = []
    normal_ys = []
    normal_slopes = []
    normal_is_vertical = []

    for i, (t_val, x_val, slope_tangent) in enumerate(zip(t, x_smooth, dx_dt)):
        # Avoid division by zero; handle zero slope separately
        if np.isclose(slope_tangent, 0.0, atol=1e-3):
            slope_normal = 0  # Horizontal normal line
        else:
            slope_normal = -1 / slope_tangent

        normal_slopes.append(slope_normal)
        t_normal, y_normal, is_vertical = normal_line(t_val, x_val, slope_normal, delta=override_t_delta)
        normal_ts.append((t_normal[0], t_normal[-1],)) # first and last value
        normal_ys.append((y_normal[0], y_normal[-1],)) # first and last value
        normal_is_vertical.append(is_vertical)
        normals.append((t_normal, y_normal, is_vertical))

    slope_tangents = deepcopy(dx_dt)
    # slope_normals = 1.0/slope_tangents
    # normal_df: pd.DataFrame = pd.DataFrame(normals, columns=['t', 'y', 'is_vertical'])
    normal_df: pd.DataFrame = pd.DataFrame({'t': t, 'x': x, 'x_smooth': x_smooth, 'is_vertical': normal_is_vertical})
    normal_df[['t_min', 't_max']] = normal_ts
    normal_df[['y_min', 'y_max']] = normal_ys
    normal_df['slope_normal'] = normal_slopes
    normal_df['slope_tangent'] = slope_tangents
    # normal_df['x'] = x
    # normal_df['x_smooth'] = x_smooth

    # plot spikes on trajectory
    if cellind is not None:
        if should_include_spikes:
            # Grab correct spike times/positions
            if speed_thresh and (not should_include_filter_excluded_spikes):
                spk_pos_, spk_t_ = active_pf1D.run_spk_pos, active_pf1D.run_spk_t # TODO: these don't exist
            else:
                spk_pos_, spk_t_ = active_pf1D.spk_pos, active_pf1D.spk_t

            # spk_tangents = np.interp(spk_t_, slope_tangents, slope_tangents)
            spk_t = spk_t_[cellind]
            # spk_pos_ = spk_pos_[cellind]

            spk_normals_tmin = np.interp(spk_t, normal_df['t'].values, normal_df['t_min'].values)
            spk_normals_tmax = np.interp(spk_t, normal_df['t'].values, normal_df['t_max'].values)
            spk_normals_slope = np.interp(spk_t, normal_df['t'].values, normal_df['slope_normal'].values)
            spk_normals_ymin = np.interp(spk_t, normal_df['t'].values, normal_df['y_min'].values)
            spk_normals_ymax = np.interp(spk_t, normal_df['t'].values, normal_df['y_max'].values)

            # spk_tangents = np.interp(spk_t_, slope_tangents, slope_tangents)

            #TODO 2024-11-04 17:39: - [ ] Finish

            if spike_plot_kwargs is None:
                spike_plot_kwargs = {}

            if spikes_alpha is None:
                spikes_alpha = 0.5 # default value of 0.5

            if spikes_color is not None:
                spikes_color_RGBA = [*spikes_color, spikes_alpha]
                # Check for existing values in spike_plot_kwargs which will be overriden
                markerfacecolor = spike_plot_kwargs.get('markerfacecolor', None)
                # markeredgecolor = spike_plot_kwargs.get('markeredgecolor', None)
                if markerfacecolor is not None:
                    if markerfacecolor != spikes_color_RGBA:
                        print(f"WARNING: spike_plot_kwargs's extant 'markerfacecolor' and 'markeredgecolor' values will be overriden by the provided spikes_color argument, meaning its original value will be lost.")
                        spike_plot_kwargs['markerfacecolor'] = spikes_color_RGBA
                        spike_plot_kwargs['markeredgecolor'] = spikes_color_RGBA
            else:
                # assign the default
                spikes_color_RGBA = [*(0, 0, 0.8), spikes_alpha]


            # interpolate the normal lines: spk_t_, spk_pos_
            # Select indices where normals will be plotted
            indices = np.arange(0, len(t), 10)  # Every 10th point

            # Prev-dot-based _____________________________________________________________________________________________________ #
            # for a, pos in zip(ax, spk_pos_[cellind]):
            # 	# WARNING: if spike_plot_kwargs contains the 'markerfacecolor' key, it's value will override plot's color= argument, so the specified spikes_color will be ignored.
            # 	a.plot(spk_t_[cellind], pos, color=spikes_color_RGBA, **(spike_plot_kwargs or {})) # , color=[*spikes_color, spikes_alpha]
            # 	#TODO 2023-09-06 02:23: - [ ] Note that without extra `spike_plot_kwargs` this plots spikes as connected lines without markers which is nearly always wrong.

            # 2024-11-04 - Lines normal to the position plot _____________________________________________________________________ #
            # Determine the vertical span (delta) for the spike lines
            # Here, delta_y is set to a small fraction of the y-axis range
            # Alternatively, you can set a fixed value
            spike_plot_kwargs.setdefault('color', spikes_color_RGBA)
            spike_plot_kwargs.setdefault('linewidth', spike_plot_kwargs.get('linewidth', 1))  # Default line width

            delta_y = []
            for a_ax, pos_label in zip(ax, variable_array):
                y_min, y_max = a_ax.get_ylim()
                span = y_max - y_min
                a_delta_y = span * 0.01  # 1% of y-axis range
                print(f'a_delta_y: {a_delta_y}')
                # a_delta_y = 0.5  # 1% of y-axis range
                delta_y.append(a_delta_y)

            print(f'delta_y: {delta_y}')
            # Plot spikes for each dimension
            for dim, (a_ax, a_delta_y) in enumerate(zip(ax, delta_y)):
                spk_t = spk_t_[cellind]
                spk_pos = spk_pos_[cellind][:, dim] if active_pf1D.ndim > 1 else spk_pos_[cellind]

                # Plot normal lines
                # Calculate ymin and ymax for each spike
                # ymin = spk_pos - delta
                # ymax = spk_pos + delta
                # Use ax.vlines to plot all spikes at once
                # a_ax.vlines(spk_t, ymin, ymax, **spike_plot_kwargs)
                # a_ax.vlines(spk_t, spk_normals_ymin, spk_normals_ymax, **spike_plot_kwargs)

                # Plot normal lines
                for i, (tspike, tmin, tmax, slope, ymin, ymax) in enumerate(zip(spk_t, spk_normals_tmin, spk_normals_tmax, spk_normals_slope, spk_normals_ymin, spk_normals_ymax)):
                    # if is_vertical:
                    # 	plt.vlines(t_normal[0], y_normal[0], y_normal[1], colors='red', linestyles='--', linewidth=1)
                    # plt.plot(t_normal, y_normal, color='red', linestyle='--', linewidth=1)

                    a_ax.plot([(tspike-a_delta_y), (tspike+a_delta_y)], [ymin, ymax], color='#ff00009b', linestyle='solid', linewidth=2, label='Normal Line' if i == 0 else "")  # Label only the first line to avoid duplicate legends
                    # a_ax.plot([tmin, tmax], [ymin, ymax], color='red', linestyle='--', linewidth=1, label='Normal Line' if i == 0 else "")  # Label only the first line to avoid duplicate legends
                    # a_ax.vlines(spk_t, ymin, ymax, **spike_plot_kwargs)



        # Put info on title
        if should_include_labels:
            ax[0].set_title(
                "Cell "
                + str(active_pf1D.cell_ids[cellind])
                + ":, speed_thresh="
                + str(active_pf1D.speed_thresh)
            )
    return ax


# ==================================================================================================================== #
# 2024-11-01 - Cell First Firing - Cell's first firing -- during PBE, theta, or resting?                                                                                      #
# ==================================================================================================================== #

from functools import reduce

from neuropy.core.neuron_identities import NeuronIdentityDataframeAccessor
from neuropy.core.flattened_spiketrains import SpikesAccessor
from neuropy.core.epoch import ensure_dataframe
from pyphocorehelpers.indexing_helpers import reorder_columns_relative

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df
from neuropy.utils.mixins.AttrsClassHelpers import SimpleFieldSizesReprMixin
from pyphocorehelpers.indexing_helpers import partition_df
from neuropy.core.user_annotations import UserAnnotationsManager
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import FileOutputManager, FigureOutputLocation, ContextToPathMode # for CellsFirstSpikeTimes

@define(slots=False, eq=False, repr=False)
class CellsFirstSpikeTimes(SimpleFieldSizesReprMixin):
    """ First spike times


    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import CellsFirstSpikeTimes

    all_cells_first_spike_time_df, global_spikes_df, (global_spikes_dict, first_spikes_dict), hdf5_out_path = CellsFirstSpikeTimes.compute_cell_first_firings(curr_active_pipeline, hdf_save_parent_path=collected_outputs_path)
    all_cells_first_spike_time_df

    """
    global_spikes_df: pd.DataFrame = field()
    all_cells_first_spike_time_df: pd.DataFrame = field()

    global_spikes_dict: Dict[str, pd.DataFrame] = field()
    first_spikes_dict: Dict[str, pd.DataFrame] = field()

    global_position_df: pd.DataFrame = field()
    hdf5_out_path: Optional[Path] = field()

    @property
    def neuron_uids(self):
        """The neuron_ids property."""
        return self.all_cells_first_spike_time_df['neuron_uid'].unique()

    @property
    def neuron_ids(self):
        """The neuron_ids property."""
        return self.all_cells_first_spike_time_df['aclu'].unique()


    def __attrs_post_init__(self):
        """ after initializing, run post_init_cleanup() to order the columns """
        self.post_init_cleanup()


    @classmethod
    def init_from_pipeline(cls, curr_active_pipeline, hdf_save_parent_path: Path=None, should_include_only_spikes_after_initial_laps=False) -> "CellsFirstSpikeTimes":
        """
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import CellsFirstSpikeTimes

        _obj: CellsFirstSpikeTimes = CellsFirstSpikeTimes.init_from_pipeline(curr_active_pipeline=curr_active_pipeline)
        """
        all_cells_first_spike_time_df, global_spikes_df, (global_spikes_dict, first_spikes_dict), global_position_df, hdf5_out_path = CellsFirstSpikeTimes.compute_cell_first_firings(curr_active_pipeline, hdf_save_parent_path=hdf_save_parent_path, should_include_only_spikes_after_initial_laps=should_include_only_spikes_after_initial_laps)
        # global_position_df = deepcopy(curr_active_pipeline.sess.position.df)
        # session_uid: str = curr_active_pipeline.get_session_context().get_description(separator="|", include_property_names=False)
        # global_position_df['session_uid'] = session_uid  # Provide an appropriate session identifier here
        _obj: CellsFirstSpikeTimes = CellsFirstSpikeTimes(global_spikes_df=global_spikes_df, all_cells_first_spike_time_df=all_cells_first_spike_time_df,
                             global_spikes_dict=global_spikes_dict, first_spikes_dict=first_spikes_dict, global_position_df=deepcopy(global_position_df), # sess.position.to_dataframe()
                             hdf5_out_path=hdf5_out_path)
        return _obj


    @classmethod
    def init_from_batch_hdf5_exports(cls, first_spike_activity_data_h5_files: List[Union[str, Path]]) -> "CellsFirstSpikeTimes":
        """

        """
        all_sessions_global_spikes_df, all_sessions_first_spike_combined_df, exact_category_counts, (all_sessions_global_spikes_dict, all_sessions_first_spikes_dict, all_sessions_extra_dfs_dict_dict) = cls.load_batch_hdf5_exports(first_spike_activity_data_h5_files=first_spike_activity_data_h5_files)
        global_position_df = all_sessions_extra_dfs_dict_dict.get('global_position_df', None)

        _obj: CellsFirstSpikeTimes = CellsFirstSpikeTimes(global_spikes_df=deepcopy(all_sessions_global_spikes_df), all_cells_first_spike_time_df=deepcopy(all_sessions_first_spike_combined_df),
                                                            global_spikes_dict=deepcopy(all_sessions_global_spikes_dict), first_spikes_dict=deepcopy(all_sessions_first_spikes_dict), hdf5_out_path=None, global_position_df=global_position_df)
        return _obj



    def add_session_info(self, t_delta_dict):
        """ post-hoc after loading
        """
        for k, v in self.first_spikes_dict.items():
            if 'session_name' in v.columns:
                v['session_t_delta'] = v.session_name.map(lambda x: t_delta_dict.get(IdentifyingContext.try_init_from_session_key(session_str=x, separator='-').get_description(separator='_'), {}).get('t_delta', None))
            else:
                print(f'k: {k}')

        for k, v in self.global_spikes_dict.items():
            if 'session_name' in v.columns:
                v['session_t_delta'] = v.session_name.map(lambda x: t_delta_dict.get(IdentifyingContext.try_init_from_session_key(session_str=x, separator='-').get_description(separator='_'), {}).get('t_delta', None))
            else:
                print(f'k: {k}')

        self.all_cells_first_spike_time_df['session_t_delta'] = self.all_cells_first_spike_time_df.session_name.map(lambda x: t_delta_dict.get(IdentifyingContext.try_init_from_session_key(session_str=x, separator='-').get_description(separator='_'), {}).get('t_delta', None))
        self.global_spikes_df['session_t_delta'] = self.global_spikes_df.session_name.map(lambda x: t_delta_dict.get(IdentifyingContext.try_init_from_session_key(session_str=x, separator='-').get_description(separator='_'), {}).get('t_delta', None))




    def post_init_cleanup(self):
        """ orders the columns """
        ordered_column_names = ['neuron_uid', 'format_name', 'animal', 'exper_name', 'session_name', 'aclu', 'session_uid']

        for k, v in self.first_spikes_dict.items():
            self.first_spikes_dict[k] = reorder_columns_relative(v, column_names=ordered_column_names, # , 'session_datetime'
                                            relative_mode='start')

        for k, v in self.global_spikes_dict.items():
            self.global_spikes_dict[k] = reorder_columns_relative(v, column_names=ordered_column_names, # , 'session_datetime'
                                            relative_mode='start')

        self.global_spikes_df = reorder_columns_relative(self.global_spikes_df, column_names=ordered_column_names, # , 'session_datetime'
                                            relative_mode='start')

        self.all_cells_first_spike_time_df = reorder_columns_relative(self.all_cells_first_spike_time_df, column_names=ordered_column_names, # , 'session_datetime'
                                                    relative_mode='start')

        ## add 'session_t_delta'?
        ## add 'session_datetime'?



    # @function_attributes(short_name=None, tags=['first-spike', 'cell-analysis'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-01 13:59', related_items=[])
    @classmethod
    def _subfn_get_first_spikes(cls, spikes_df: pd.DataFrame):
            if 'neuron_uid' in spikes_df:
                column_name: str = 'neuron_uid'
            else:
                column_name: str = 'aclu'

            earliest_spike_df = spikes_df.groupby([column_name]).agg(t_rel_seconds_idxmin=('t_rel_seconds', 'idxmin'), t_rel_seconds_min=('t_rel_seconds', 'min')).reset_index() # 't_rel_seconds_idxmin', 't_rel_seconds_min'
            # first_aclu_spike_records_df: pd.DataFrame = spikes_df[np.isin(spikes_df['t_rel_seconds'], earliest_spike_df['t_rel_seconds_min'].values)]
            # Select rows using the indices of the minimal t_rel_seconds
            first_aclu_spike_records_df: pd.DataFrame = spikes_df.loc[earliest_spike_df['t_rel_seconds_idxmin']] ## ChatGPT claimed correct
            # 2024-11-08 17:10 I don't get why these differ. It makes zero sense to me.


            return first_aclu_spike_records_df

    @classmethod
    def _subfn_build_first_spike_dataframe(cls, first_spikes_dict):
        """
        Builds a dataframe containing each 'aclu' value along with its first spike time for each category,
        and determines the earliest spike category (excluding 'any').

        Parameters:
        - first_spikes_dict (dict): A dictionary where keys are category names and values are dataframes
                                    containing spike data, including 'aclu' and 't_rel_seconds' columns.

        Returns:
        - pd.DataFrame: A dataframe with 'aclu', first spike times per category, and the earliest spike category.
        """
        from neuropy.utils.indexing_helpers import union_of_arrays

        # Step 1: Prepare list of dataframes with first spike times per category
        category_column_inclusion_dict = dict(zip(list(first_spikes_dict.keys()), [['aclu', 't_rel_seconds']]*len(first_spikes_dict))) ## as a minimum each category includes ['t_rel_seconds']
        ## extra columns used to prevent duplication
        category_column_extra_columns_dict = {'any': ['shank', 'cluster', 'qclu'],
                                              'lap': ['lap', 'maze_relative_lap', 'maze_id'],
                                            #   'lap': ['x', 'y', 'lin_pos', 'speed', 'traj', 'lap', 'theta_phase_radians', 'maze_relative_lap', 'maze_id'],
                                              }
        for category, extra_columns in category_column_extra_columns_dict.items():
            category_column_inclusion_dict[category] = category_column_inclusion_dict[category] + extra_columns


        any_df_aclus = union_of_arrays([df['aclu'].unique() for category, df in first_spikes_dict.items()])
        n_unique_aclus: int = len(any_df_aclus)

        dfs = []
        for category, df in first_spikes_dict.items():
            ## each incoming df is a first_spikes_df, so it only has one spike from eahc aclu
            df_grouped = deepcopy(df)[category_column_inclusion_dict[category]].reset_index(drop=True)
            # Group by 'aclu' and get the minimum 't_rel_seconds' (first spike time)
            # df_grouped = df.groupby('aclu')['t_rel_seconds'].min().reset_index()
            # Rename the 't_rel_seconds' column to include the category
            if category != 'any':
                extra_category_columns = category_column_extra_columns_dict.get(category, [])
                extra_columns_rename_dict = dict(zip(extra_category_columns, [f'{category}_spike_{v}' for v in extra_category_columns]))
            else:
                extra_columns_rename_dict = {} # empty, don't rename
            df_grouped.rename(columns={'t_rel_seconds': f'first_spike_{category}', **extra_columns_rename_dict}, inplace=True) ## rename each 't_rel_seconds' to a unique column name

            assert set(df_grouped['aclu'].unique()) == set(any_df_aclus), f"set(any_df_aclus): {set(any_df_aclus)}, set(df_grouped['aclu'].unique()): {set(df_grouped['aclu'].unique())}"

            dfs.append(df_grouped)

        assert np.all([np.shape(a_df)[0] == n_unique_aclus for a_df in dfs]), f"every df must have the same alus (all of them)!  {[np.shape(a_df) == n_unique_aclus for a_df in dfs]}"
        # Step 2: Merge all dataframes on 'aclu'
        df_final = reduce(lambda left, right: pd.merge(left, right, on='aclu', how='outer'), dfs)
        assert len(df_final['aclu'].unique()) == n_unique_aclus, f"final must have the same alus as before! len(df_final['aclu'].unique()): {len(df_final['aclu'].unique())}, n_unique_aclus: {n_unique_aclus}"
        # Step 3: Determine earliest spike category (excluding 'any')
        # Get the list of columns containing first spike times, excluding 'any'
        spike_time_columns = [col for col in df_final.columns if col.startswith('first_spike_') and col != 'first_spike_any']

        # Function to get the earliest spike category for each row
        def get_earliest_category(row):
            # Extract spike times, excluding 'any'
            spike_times = row[spike_time_columns].dropna()
            if spike_times.empty:
                return None  # No spike times available in categories excluding 'any'
            # Find the minimum spike time
            min_spike_time = spike_times.min()
            # Get categories with the minimum spike time
            min_spike_columns = spike_times[spike_times == min_spike_time].index.tolist()
            # Extract category names
            earliest_categories = [col.replace('first_spike_', '') for col in min_spike_columns]
            # Join categories if there's a tie
            return ', '.join(earliest_categories)

        # Apply the function to determine the earliest spike category
        df_final['earliest_spike_category'] = df_final.apply(get_earliest_category, axis=1)

        # Optionally, add the earliest spike time (excluding 'any')
        df_final['earliest_spike_time'] = df_final[spike_time_columns].min(axis=1)

        return df_final

    @classmethod
    def _parse_split_session_key_with_prefix(cls, a_session_key: str):
        # a_session_key: str = '2024-11-05-kdiba-gor01-one-2006-6-08_14-26-15'
        a_key_split = a_session_key.split(sep='-')
        session_descriptor_start_idx: int = a_key_split.index('kdiba')
        assert session_descriptor_start_idx != -1
        pre_session_info: str = '-'.join(a_key_split[:session_descriptor_start_idx]) # '2024-11-05'
        # pre_session_info

        true_session_key: str = '-'.join(a_key_split[session_descriptor_start_idx:]) # 'kdiba-gor01-one-2006-6-08_14-26-15'
        # true_session_key
        return true_session_key, pre_session_info

    @classmethod
    def _slice_by_valid_time_subsets(cls, a_global_spikes_df, session_uid, first_valid_pos_time, last_valid_pos_time):
        trimmed_global_spikes_df = deepcopy(a_global_spikes_df).spikes.time_sliced(first_valid_pos_time, last_valid_pos_time)
        trimmed_result_tuple = cls.perform_compute_cell_first_firings(global_spikes_df=trimmed_global_spikes_df)
        ## OUTPUTS: trimmed_global_spikes_df, trimmed_all_cells_first_spike_time_df
        # trimmed_all_cells_first_spike_time_df, trimmed_global_spikes_df, (trimmed_global_spikes_dict, trimmed_first_spikes_dict) = trimmed_result_tuple
        return trimmed_result_tuple

    # ==================================================================================================================== #
    # After the first laps                                                                                                 #
    # ==================================================================================================================== #
    @classmethod
    def _include_only_spikes_after_initial_laps(cls, a_global_spikes_df, initial_laps_end_time=np.inf, last_valid_pos_time=np.inf):
        initial_laps_end_time: float = a_global_spikes_df[a_global_spikes_df['lap'] == 2]['t_rel_seconds'].max() # last spike in lap id=1 - 41.661858989158645
        post_initial_lap_global_spikes_df = deepcopy(a_global_spikes_df).spikes.time_sliced(initial_laps_end_time, last_valid_pos_time) # trim to be after the first lap
        post_initial_lap_tuple = cls.perform_compute_cell_first_firings(global_spikes_df=post_initial_lap_global_spikes_df)
        # post_initial_lap_all_cells_first_spike_time_df, post_initial_lap_global_spikes_df, (post_first_lap_global_spikes_dict, post_first_lap_first_spikes_dict) = post_initial_lap_tuple
        return post_initial_lap_tuple, initial_laps_end_time

    @classmethod
    def perform_compute_cell_first_firings(cls, global_spikes_df: pd.DataFrame):
        """
        requires a spikes_df with session columns

        Usage:
            global_spikes_df: pd.DataFrame = deepcopy(get_proper_global_spikes_df(curr_active_pipeline)).drop(columns=['neuron_type'], inplace=False) ## already has columns ['lap', 'maze_id', 'PBE_id'
            global_spikes_df = global_spikes_df.neuron_identity.make_neuron_indexed_df_global(curr_active_pipeline.get_session_context(), add_expanded_session_context_keys=True, add_extended_aclu_identity_columns=True)
            # Perform the computations ___________________________________________________________________________________________ #
            all_cells_first_spike_time_df, global_spikes_df, (global_spikes_dict, first_spikes_dict) = cls.perform_compute_cell_first_firings(global_spikes_df=global_spikes_df)
            ## add the sess properties to the output df:
            all_cells_first_spike_time_df = all_cells_first_spike_time_df.neuron_identity.make_neuron_indexed_df_global(curr_active_pipeline.get_session_context(), add_expanded_session_context_keys=True, add_extended_aclu_identity_columns=True)

        """
        # ==================================================================================================================== #
        # Separate Theta/Ripple/etc dfs                                                                                        #
        # ==================================================================================================================== #
        # global_spikes_df_theta_df, global_spikes_df_non_theta_df = partition_df_dict(global_spikes_df, partitionColumn='is_theta')
        # global_spikes_df_theta_df

        global_spikes_theta_df = deepcopy(global_spikes_df[global_spikes_df['is_theta'] == True])
        global_spikes_ripple_df = deepcopy(global_spikes_df[global_spikes_df['is_ripple'] == True])
        global_spikes_neither_df = deepcopy(global_spikes_df[np.logical_and((global_spikes_df['is_ripple'] != True), (global_spikes_df['is_theta'] != True))])

        # find first spikes of the PBE and lap periods:
        global_spikes_PBE_df = deepcopy(global_spikes_df)[global_spikes_df['PBE_id'] > -1]
        global_spikes_laps_df = deepcopy(global_spikes_df)[global_spikes_df['lap'] > -1]

        ## main output products:
        global_spikes_dict = {'any': global_spikes_df, 'theta': global_spikes_theta_df, 'ripple': global_spikes_ripple_df, 'neither': global_spikes_neither_df,
                              'PBE': global_spikes_PBE_df, 'lap': global_spikes_laps_df,
                              }

        first_spikes_dict = {k:cls._subfn_get_first_spikes(v) for k, v in global_spikes_dict.items()}
        # partition_df(global_spikes_df, 'is_theta')

        # first_aclu_spike_records_df: pd.DataFrame = first_spikes_dict['any']

        # neuron_ids = {k:v.aclu.unique() for k, v in global_spikes_dict.items()}
        # at_least_one_decoder_neuron_ids = union_of_arrays(*list(neuron_ids.values()))

        ## Check whether the first
        # first_aclu_spike_records_df['is_theta']

        # first_aclu_spike_records_df['is_ripple']
        all_cells_first_spike_time_df: pd.DataFrame = cls._subfn_build_first_spike_dataframe(first_spikes_dict)
        # all_cells_first_spike_time_df = all_cells_first_spike_time_df.neuron_identity.make_neuron_indexed_df_global(curr_active_pipeline.get_session_context(), add_expanded_session_context_keys=True, add_extended_aclu_identity_columns=True) ## why isn't it already neuron-indexed?

        ## extra computations:
        all_cells_first_spike_time_df['theta_to_ripple_lead_lag_diff'] = (all_cells_first_spike_time_df['first_spike_ripple'] - all_cells_first_spike_time_df['first_spike_theta']) ## if theta came first, diff should be positive

        assert len(all_cells_first_spike_time_df) == len(all_cells_first_spike_time_df['aclu'].unique()), f"end result must have one entry for every unique aclu"

        return all_cells_first_spike_time_df, global_spikes_df, (global_spikes_dict, first_spikes_dict)


    @classmethod
    def compute_cell_first_firings(cls, curr_active_pipeline, hdf_save_parent_path: Path=None, should_include_only_spikes_after_initial_laps:bool=False): # , save_hdf: bool=True
        """
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_cell_first_firings

        all_cells_first_spike_time_df, global_spikes_df, (global_spikes_dict, first_spikes_dict) = compute_cell_first_firings(curr_active_pipeline)
        all_cells_first_spike_time_df

        global_spikes_df: pd.DataFrame = deepcopy(get_proper_global_spikes_df(curr_active_pipeline)).drop(columns=['neuron_type'], inplace=False).neuron_identity.make_neuron_indexed_df_global(curr_active_pipeline.get_session_context(), add_expanded_session_context_keys=True, add_extended_aclu_identity_columns=True)

        Actual INPUTS: global_spikes_df: pd.DataFrame,


        ## only for saving to .h5


        from pipeline uses: curr_active_pipeline.get_custom_pipeline_filenames_from_parameters(),
        curr_active_pipeline.get_session_context()


        """
        # BEGIN FUNCTION BODY ________________________________________________________________________________________________ #
        # _, _, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        # global_session = curr_active_pipeline.filtered_sessions[global_epoch_name]
        # Get existing laps from session:
        # global_epoch = curr_active_pipeline.filtered_epochs[global_epoch_name]
        # t_start, t_end = global_epoch.start_end_times

        # running_epochs = ensure_dataframe(deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps.as_epoch_obj()))
        # pbe_epochs = ensure_dataframe(deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].pbe)) ## less selective than replay, which has cell participation and other requirements
        # all_epoch = ensure_dataframe(deepcopy(global_session.epochs))


        a_session_context = curr_active_pipeline.get_session_context() # IdentifyingContext.try_init_from_session_key(session_str=a_session_uid, separator='|')
        session_uid: str = a_session_context.get_description(separator="|", include_property_names=False)
        last_valid_pos_time = UserAnnotationsManager.get_hardcoded_specific_session_override_dict().get(a_session_context, {}).get('track_end_t', np.nan)
        first_valid_pos_time = UserAnnotationsManager.get_hardcoded_specific_session_override_dict().get(a_session_context, {}).get('track_start_t', np.nan)

        ## global_position_df
        global_position_df = deepcopy(curr_active_pipeline.sess.position.df)
        global_position_df = global_position_df.position.time_sliced(first_valid_pos_time, last_valid_pos_time)
        global_position_df['session_uid'] = session_uid  # Provide an appropriate session identifier here

        # global_spikes_df: pd.DataFrame = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].spikes_df).drop(columns=['neuron_type'], inplace=False) ## already has columns ['lap', 'maze_id', 'PBE_id'
        global_spikes_df: pd.DataFrame = deepcopy(get_proper_global_spikes_df(curr_active_pipeline)).drop(columns=['neuron_type'], inplace=False) ## already has columns ['lap', 'maze_id', 'PBE_id'
        global_spikes_df = deepcopy(global_spikes_df).spikes.time_sliced(first_valid_pos_time, last_valid_pos_time)

        if should_include_only_spikes_after_initial_laps:
            initial_laps_end_time: float = global_spikes_df[global_spikes_df['lap'] == 2]['t_rel_seconds'].max() # last spike in lap id=1 - 41.661858989158645
            global_spikes_df = deepcopy(global_spikes_df).spikes.time_sliced(initial_laps_end_time, last_valid_pos_time) # trim to be after the first lap post_initial_lap_global_spikes_df
            global_position_df = global_position_df.position.time_sliced(initial_laps_end_time, last_valid_pos_time)


        global_spikes_df = global_spikes_df.neuron_identity.make_neuron_indexed_df_global(a_session_context, add_expanded_session_context_keys=True, add_extended_aclu_identity_columns=True)
        global_position_df['session_uid'] = session_uid  # Provide an appropriate session identifier here

        # Perform the computations ___________________________________________________________________________________________ #
        all_cells_first_spike_time_df, global_spikes_df, (global_spikes_dict, first_spikes_dict) = cls.perform_compute_cell_first_firings(global_spikes_df=global_spikes_df)
        ## add the sess properties to the output df:
        all_cells_first_spike_time_df = all_cells_first_spike_time_df.neuron_identity.make_neuron_indexed_df_global(a_session_context, add_expanded_session_context_keys=True, add_extended_aclu_identity_columns=True) ## why isn't it already neuron-indexed?

        # Save to .h5 or CSV _________________________________________________________________________________________________ #
        if (hdf_save_parent_path is not None):
            custom_save_filepaths, custom_save_filenames, custom_suffix = curr_active_pipeline.get_custom_pipeline_filenames_from_parameters() # 'normal_computed-frateThresh_5.0-qclu_[1, 2]'
            complete_output_prefix: str = '_'.join([a_session_context.get_description(separator='-'), custom_suffix]) # 'kdiba-gor01-one-2006-6-08_14-26-15__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]'
            Assert.path_exists(hdf_save_parent_path)
            hdf5_out_path = hdf_save_parent_path.joinpath(f"{complete_output_prefix}_first_spike_activity_data.h5").resolve()
            print(f'hdf5_out_path: {hdf5_out_path}')
            # Save the data to an HDF5 file
            cls.save_data_to_hdf5(all_cells_first_spike_time_df, global_spikes_df, global_spikes_dict, first_spikes_dict, filename=hdf5_out_path, global_position_df=global_position_df) # Path(r'K:\scratch\collected_outputs\kdiba-gor01-one-2006-6-08_14-26-15__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]_first_spike_activity_data.h5')
        else:
            hdf5_out_path = None

        return all_cells_first_spike_time_df, global_spikes_df, (global_spikes_dict, first_spikes_dict), global_position_df, hdf5_out_path




    def sliced_by_neuron_id(self, included_neuron_ids, key_name='aclu') -> pd.DataFrame:
        """ gets the slice of spikes with the specified `included_neuron_ids` """
        assert included_neuron_ids is not None
        test_obj = deepcopy(self)

        for k, v in test_obj.first_spikes_dict.items():
            test_obj.first_spikes_dict[k] = v[v[key_name].isin(included_neuron_ids)].reset_index(drop=True)
        for k, v in test_obj.global_spikes_dict.items():
            # test_obj.global_spikes_dict[k] = v.spikes.sliced_by_neuron_id(included_neuron_ids=included_neuron_ids)
            test_obj.global_spikes_dict[k] = v[v[key_name].isin(included_neuron_ids)].reset_index(drop=True)

        # test_obj.global_spikes_df = test_obj.global_spikes_df.spikes.sliced_by_neuron_id(included_neuron_ids=included_neuron_ids)
        test_obj.global_spikes_df = test_obj.global_spikes_df[test_obj.global_spikes_df[key_name].isin(included_neuron_ids)].reset_index(drop=True)

        # test_obj.all_cells_first_spike_time_df = test_obj.all_cells_first_spike_time_df.spikes.sliced_by_neuron_id(included_neuron_ids=included_neuron_ids)
        test_obj.all_cells_first_spike_time_df = test_obj.all_cells_first_spike_time_df[test_obj.all_cells_first_spike_time_df[key_name].isin(included_neuron_ids)].reset_index(drop=True)

        return test_obj # self._obj[self._obj['aclu'].isin(included_neuron_ids)] ## restrict to only the shared aclus for both short and long



    # ==================================================================================================================== #
    # HDF5 Serialization                                                                                                   #
    # ==================================================================================================================== #

    def save_to_hdf5(self, hdf_save_path: Path):
        """ Save to .h5 or CSV
        """
        print(f'hdf_save_path: {hdf_save_path}')
        # Save the data to an HDF5 file
        did_save_successfully: bool = False
        try:
            self.save_data_to_hdf5(self.all_cells_first_spike_time_df, self.global_spikes_df, self.global_spikes_dict, self.first_spikes_dict, filename=hdf_save_path, global_position_df=self.global_position_df) # Path(r'K:\scratch\collected_outputs\kdiba-gor01-one-2006-6-08_14-26-15__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]_first_spike_activity_data.h5')
            did_save_successfully = True
            self.hdf5_out_path = hdf_save_path
        except Exception as e:
            raise

        if not did_save_successfully:
            self.hdf5_out_path = None
        return did_save_successfully


    @classmethod
    def save_data_to_hdf5(cls, all_cells_first_spike_time_df, global_spikes_df, global_spikes_dict, first_spikes_dict, filename='output_file.h5', **kwargs_extra_dfs):
        """
        Saves the given DataFrames and dictionaries of DataFrames to an HDF5 file.

        Parameters:
        - all_cells_first_spike_time_df (pd.DataFrame): DataFrame containing first spike times and categories.
        - global_spikes_df (pd.DataFrame): DataFrame containing all spikes.
        - global_spikes_dict (dict): Dictionary of DataFrames for each spike category.
        - first_spikes_dict (dict): Dictionary of DataFrames containing first spikes per category.
        - filename (str): Name of the HDF5 file to save the data to.
        """
        with pd.HDFStore(filename, mode='w') as store:
            # Save the main DataFrames
            store.put('all_cells_first_spike_time_df', all_cells_first_spike_time_df)
            store.put('global_spikes_df', global_spikes_df)

            # Save the global_spikes_dict
            for key, df in global_spikes_dict.items():
                store.put(f'global_spikes_dict/{key}', df)

            # Save the first_spikes_dict
            for key, df in first_spikes_dict.items():
                store.put(f'first_spikes_dict/{key}', df)

            for key, df in kwargs_extra_dfs.items():
                store.put(f'extra_dfs/{key}', df)

        print(f"Data successfully saved to {filename}")

    @classmethod
    def load_data_from_hdf5(cls, filename='output_file.h5'):
        """
        Loads the DataFrames and dictionaries of DataFrames from an HDF5 file.

        Parameters:
        - filename (str): Name of the HDF5 file to load the data from.

        Returns:
        - all_cells_first_spike_time_df (pd.DataFrame)
        - global_spikes_df (pd.DataFrame)
        - global_spikes_dict (dict)
        - first_spikes_dict (dict)

        Usage:

            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import CellsFirstSpikeTimes
            hdf_load_path = Path('K:/scratch/collected_outputs/kdiba-gor01-one-2006-6-08_14-26-15__withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]_first_spike_activity_data.h5').resolve()
            Assert.path_exists(hdf_load_path)
            # Load the data back from the HDF5 file
            all_cells_first_spike_time_df_loaded, global_spikes_df_loaded, global_spikes_dict_loaded, first_spikes_dict_loaded = CellsFirstSpikeTimes.load_data_from_hdf5(filename=hdf_load_path)
            all_cells_first_spike_time_df_loaded

        """
        with pd.HDFStore(filename, mode='r') as store:
            # Load the main DataFrames
            all_cells_first_spike_time_df = store['all_cells_first_spike_time_df']
            global_spikes_df = store['global_spikes_df']

            # Initialize dictionaries
            global_spikes_dict = {}
            first_spikes_dict = {}
            extra_dfs_dict = {}

            # Load keys for global_spikes_dict
            global_spikes_keys = [key.split('/')[-1] for key in store.keys() if key.startswith('/global_spikes_dict/')]
            for key in global_spikes_keys:
                df = store[f'global_spikes_dict/{key}']
                global_spikes_dict[key] = df

            # Load keys for first_spikes_dict
            first_spikes_keys = [key.split('/')[-1] for key in store.keys() if key.startswith('/first_spikes_dict/')]
            for key in first_spikes_keys:
                df = store[f'first_spikes_dict/{key}']
                first_spikes_dict[key] = df

            # Load keys for extra_dfs
            extra_dfs_keys = [key.split('/')[-1] for key in store.keys() if key.startswith('/extra_dfs/')]
            for key in extra_dfs_keys:
                df = store[f'extra_dfs/{key}']
                extra_dfs_dict[key] = df


        print(f"Data successfully loaded from {filename}")
        return all_cells_first_spike_time_df, global_spikes_df, global_spikes_dict, first_spikes_dict, extra_dfs_dict


    @classmethod
    def load_batch_hdf5_exports(cls, first_spike_activity_data_h5_files):
        """

        all_sessions_global_spikes_df, all_sessions_first_spike_combined_df, exact_category_counts = CellsFirstSpikeTimes.load_batch_hdf5_exports(first_spike_activity_data_h5_files=first_spike_activity_data_h5_files)

        """
        first_spike_activity_data_h5_files = [Path(v).resolve() for v in first_spike_activity_data_h5_files] ## should parse who name and stuff... but we don't.
        all_sessions_first_spike_activity_tuples: List[Tuple] = [CellsFirstSpikeTimes.load_data_from_hdf5(filename=hdf_load_path) for hdf_load_path in first_spike_activity_data_h5_files] ## need to export those globally unique identifiers for each aclu within a session

        # all_sessions_all_cells_first_spike_time_df_loaded

        # for i, an_all_cells_first_spike_time_df in enumerate(all_sessions_all_cells_first_spike_time_df_loaded):
        total_counts = []
        all_sessions_global_spikes_df = []

        all_sessions_global_spikes_dict = {}
        all_sessions_first_spikes_dict = {}
        all_sessions_extra_dfs_dict_dict = {}

        for i, (a_path, a_first_spike_time_tuple) in enumerate(zip(first_spike_activity_data_h5_files, all_sessions_first_spike_activity_tuples)):
            all_cells_first_spike_time_df_loaded, global_spikes_df_loaded, global_spikes_dict_loaded, first_spikes_dict_loaded, extra_dfs_dict_loaded = a_first_spike_time_tuple ## unpack

            # # Parse out the session context from the filename ____________________________________________________________________ #
            # session_key, params_key = a_path.stem.split('__')
            # # session_key # 'kdiba-gor01-one-2006-6-08_14-26-15'
            # # params_key # 'withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]_first_spike_activity_data'
            # session_parts = session_key.split('-', maxsplit=3)
            # assert len(session_parts) == 4, f"session_parts: {session_parts}"
            # format_name, animal, exper_name, session_name = session_parts
            # reconstructed_session_context = IdentifyingContext(format_name=format_name, animal=animal, exper_name=exper_name, session_name=session_name)
            # # print(f'reconstructed_session_context: {reconstructed_session_context}')
            # ## seems wrong: reconstructed_session_context

            # all_cells_first_spike_time_df_loaded = all_cells_first_spike_time_df_loaded.neuron_identity.make_neuron_indexed_df_global(reconstructed_session_context, add_expanded_session_context_keys=True, add_extended_aclu_identity_columns=True)
            # global_spikes_df_loaded = global_spikes_df_loaded.neuron_identity.make_neuron_indexed_df_global(reconstructed_session_context, add_expanded_session_context_keys=True, add_extended_aclu_identity_columns=True)

            # all_cells_first_spike_time_df_loaded['path'] = a_path.as_posix()
            # all_cells_first_spike_time_df_loaded['session_key'] = session_key
            # all_cells_first_spike_time_df_loaded['params_key'] = params_key
            total_counts.append(all_cells_first_spike_time_df_loaded)
            all_sessions_global_spikes_df.append(global_spikes_df_loaded)

            for k, v in global_spikes_dict_loaded.items():
                if k not in all_sessions_global_spikes_dict:
                    all_sessions_global_spikes_dict[k] = []
                all_sessions_global_spikes_dict[k].append(v)

            for k, v in first_spikes_dict_loaded.items():
                if k not in all_sessions_first_spikes_dict:
                    all_sessions_first_spikes_dict[k] = []
                all_sessions_first_spikes_dict[k].append(v)


            for k, v in extra_dfs_dict_loaded.items():
                if k not in all_sessions_extra_dfs_dict_dict:
                    all_sessions_extra_dfs_dict_dict[k] = [] # add this dataframe name
                ## add the session column to `v` if it's missing
                # if 'session_key' not in v.columns:
                #     v['session_key'] = session_key
                # if 'params_key' not in v.columns:
                #     v['params_key'] = params_key
                # if 'session_uid' not in v.columns:
                #     v['session_uid'] = reconstructed_session_context.get_description(separator="|")
                all_sessions_extra_dfs_dict_dict[k].append(v) # append to this df name


            # first_spikes_dict_loaded
            # all_cells_first_spike_time_df_loaded
            # 1. Counting Exact Category Combinations
            # exact_category_counts = all_cells_first_spike_time_df_loaded['earliest_spike_category'].value_counts(dropna=False)
            # print("Exact Category Counts:")
            # print(exact_category_counts)

            # an_all_cells_first_spike_time_df
        # end for

        all_sessions_global_spikes_dict = {k:pd.concat(v, axis='index') for k, v in all_sessions_global_spikes_dict.items()}
        all_sessions_first_spikes_dict = {k:pd.concat(v, axis='index') for k, v in all_sessions_first_spikes_dict.items()}
        # for extra_dataframe_name, extra_dataframe_df_list in all_sessions_extra_dfs_dict_dict.items():

        # all_sessions_extra_dfs_dict_dict = {extra_dataframe_name:{k:pd.concat(v, axis='index') for k, v in extra_dataframe_df_list.items()} for extra_dataframe_name, extra_dataframe_df_list in all_sessions_extra_dfs_dict_dict.items()}
        all_sessions_extra_dfs_dict_dict = {extra_dataframe_name:pd.concat(extra_dataframe_df_list, axis='index') for extra_dataframe_name, extra_dataframe_df_list in all_sessions_extra_dfs_dict_dict.items()}

        all_sessions_first_spike_combined_df: pd.DataFrame = pd.concat(total_counts, axis='index')
        # all_sessions_first_spike_combined_df
        exact_category_counts = all_sessions_first_spike_combined_df['earliest_spike_category'].value_counts(dropna=False)
        # print("Exact Category Counts:")
        # print(exact_category_counts)
        all_sessions_global_spikes_df: pd.DataFrame = pd.concat(all_sessions_global_spikes_df, axis='index')
        return all_sessions_global_spikes_df, all_sessions_first_spike_combined_df, exact_category_counts, (all_sessions_global_spikes_dict, all_sessions_first_spikes_dict, all_sessions_extra_dfs_dict_dict)


    # ==================================================================================================================== #
    # CSV Outputs                                                                                                          #
    # ==================================================================================================================== #
    @classmethod
    def save_data_to_csvs(cls, all_cells_first_spike_time_df: pd.DataFrame, global_spikes_df: pd.DataFrame, global_spikes_dict: Dict[str, pd.DataFrame], first_spikes_dict: Dict[str, pd.DataFrame], output_dir: Union[str, Path] = 'output_csvs') -> None:
        """
        Saves the given DataFrames and dictionaries of DataFrames to several CSV files organized in a directory structure.

        Parameters:
        - all_cells_first_spike_time_df (pd.DataFrame): DataFrame containing first spike times and categories.
        - global_spikes_df (pd.DataFrame): DataFrame containing all spikes.
        - global_spikes_dict (dict): Dictionary of DataFrames for each spike category.
        - first_spikes_dict (dict): Dictionary of DataFrames containing first spikes per category.
        - output_dir (str or Path): Directory where the CSV files will be saved.

        Directory Structure:
        output_dir/
            all_cells_first_spike_time_df.csv
            global_spikes_df.csv
            global_spikes_dict/
                any.csv
                theta.csv
                ripple.csv
                neither.csv
                PBE.csv
                lap.csv
            first_spikes_dict/
                any.csv
                theta.csv
                ripple.csv
                neither.csv
                PBE.csv
                lap.csv

        Usage:

            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import CellsFirstSpikeTimes
            CellsFirstSpikeTimes.save_data_to_csvs(
                all_cells_first_spike_time_df,
                global_spikes_df,
                global_spikes_dict,
                first_spikes_dict,
                output_dir=Path('path/to/output_directory')
            )
        """
        output_dir = Path(output_dir)
        # Create the main output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving CSV files to directory: {output_dir.resolve()}")

        # Save the main DataFrames
        main_df_filenames = {
            'all_cells_first_spike_time_df.csv': all_cells_first_spike_time_df,
            'global_spikes_df.csv': global_spikes_df
        }
        for filename, df in main_df_filenames.items():
            file_path = output_dir / filename
            df.to_csv(file_path, index=False)
            print(f"Saved {filename}")

        # Define subdirectories for dictionaries
        dict_subdirs = {
            'global_spikes_dict': global_spikes_dict,
            'first_spikes_dict': first_spikes_dict
        }

        for subdir_name, data_dict in dict_subdirs.items():
            subdir_path = output_dir / subdir_name
            subdir_path.mkdir(exist_ok=True)
            print(f"Saving dictionary '{subdir_name}' to subdirectory: {subdir_path.resolve()}")

            for key, df in data_dict.items():
                # Sanitize the key to create a valid filename
                sanitized_key = "".join([c if c.isalnum() or c in (' ', '_') else '_' for c in key])
                filename = f"{sanitized_key}.csv"
                file_path = subdir_path / filename
                df.to_csv(file_path, index=False)
                print(f"Saved {subdir_name}/{filename}")

        print("All CSV files have been successfully saved.")


    def save_to_csvs(self, output_dir: Union[str, Path] = 'output_csvs') -> bool:
        """
        Saves the instance's DataFrames and dictionaries of DataFrames to CSV files.

        Parameters:
        - output_dir (str or Path): Directory where the CSV files will be saved.

        Returns:
        - bool: True if all files were saved successfully, False otherwise.

        Usage:

            _obj: CellsFirstSpikeTimes = CellsFirstSpikeTimes.init_from_pipeline(curr_active_pipeline=curr_active_pipeline)
            _obj.save_to_csvs(output_dir=Path('path/to/output_directory'))
        """
        try:
            self.save_data_to_csvs(
                all_cells_first_spike_time_df=self.all_cells_first_spike_time_df,
                global_spikes_df=self.global_spikes_df,
                global_spikes_dict=self.global_spikes_dict,
                first_spikes_dict=self.first_spikes_dict,
                output_dir=output_dir
            )
            # Optionally, you can store the output directory path if needed
            # self.csv_out_path = Path(output_dir).resolve()
            return True
        except Exception as e:
            print(f"An error occurred while saving CSV files: {e}")
            return False


    # ==================================================================================================================== #
    # Plotting and Visualization                                                                                           #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['matplotlib', 'scatter', 'spikes', 'position', 'time'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-05 13:23', related_items=[])
    def plot_first_spike_scatter_figure(self, aclu_to_color_map=None):
        """ plots a scatterplot showing the first spike for each cell during PBEs vs. Laps


        # global_session.config.plotting_config
        active_config = deepcopy(curr_active_pipeline.active_configs[global_epoch_name])
        active_pf1D = deepcopy(global_pf1D)
        aclu_to_color_map = {v.cell_uid:v.color.tolist() for v in active_config.plotting_config.pf_neuron_identities}
        fig, ax = cells_first_spike_times.plot_first_spike_scatter_figure(aclu_to_color_map=aclu_to_color_map)

        """
        ## INPUTS: active_config
        # type(active_config.plotting_config.pf_colormap)

        self.all_cells_first_spike_time_df['color'] = self.all_cells_first_spike_time_df['aclu'].map(lambda x: aclu_to_color_map.get(x, [1.0, 1.0, 0.0, 1.0]))
        column_names = ['first_spike_any', 'first_spike_theta', 'first_spike_lap', 'first_spike_PBE']
        interpolated_position_column_names = []
        for a_col in column_names:
            ## interpolate positions for each of these spike times
            self.all_cells_first_spike_time_df[f'interp_pos_{a_col}'] = np.interp(self.all_cells_first_spike_time_df[a_col], self.global_position_df.t, self.global_position_df.x)
            interpolated_position_column_names.append(f'interp_pos_{a_col}')

        column_to_interpolated_position_column_name_dict = dict(zip(column_names, interpolated_position_column_names))
        self.all_cells_first_spike_time_df

        ## plot the spike timecourse:
        fig = plt.figure(num='test_new_spikes', clear=True)

        ax = self.global_position_df.plot(x='t', y='x', ax=fig.gca(), c=(0.3, 0.3, 0.3, 0.2))

        spike_scatter_kwargs = dict(s=25)

        ## find extrema
        # active_col_names = column_names
        active_col_names = ['first_spike_any', 'first_spike_lap']
        earliest_first_spike_t: float = self.all_cells_first_spike_time_df[active_col_names].min(axis=0).min()
        latest_first_spike_t: float = self.all_cells_first_spike_time_df[active_col_names].max(axis=0).max()
        ax.set_xlim(earliest_first_spike_t, latest_first_spike_t)

        # column_to_interpolated_position_column_name_dict['first_spike_any']
        self.all_cells_first_spike_time_df.plot.scatter(x='first_spike_any', y=column_to_interpolated_position_column_name_dict['first_spike_any'], c='color', ax=ax, marker='d', **spike_scatter_kwargs)
        self.all_cells_first_spike_time_df.plot.scatter(x='first_spike_lap', y=column_to_interpolated_position_column_name_dict['first_spike_lap'], c='color', ax=ax, marker='*', **spike_scatter_kwargs)

        # cells_first_spike_times.all_cells_first_spike_time_df.plot.scatter(x='first_spike_any', y='interpolated_y', c='color', ax=ax) # , c='color'
        return fig, ax


    @function_attributes(short_name=None, tags=['pyqtgraph', 'raster', 'spikes'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-11-05 13:23', related_items=[])
    def plot_first_lap_spike_relative_first_PBE_spike_scatter_figure(self, defer_show = False):
        """ plots a raster plot showing the first spike for each PBE for each cell (rows) relative to the first lap spike (t=0)

        test_obj: CellsFirstSpikeTimes = CellsFirstSpikeTimes.init_from_batch_hdf5_exports(first_spike_activity_data_h5_files=first_spike_activity_data_h5_files)
        app, win, plots, plots_data = test_obj.plot_first_lap_spike_relative_first_PBE_spike_scatter_figure()

        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import new_plot_raster_plot, NewSimpleRaster
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomInfiniteLine import CustomInfiniteLine
        import pyphoplacecellanalysis.External.pyqtgraph as pg

        ## INPUTS: active_config
        # type(active_config.plotting_config.pf_colormap)
        ## align to first lap spike (first_spike_lap)
        self.all_cells_first_spike_time_df['lap_spike_relative_first_spike'] = self.all_cells_first_spike_time_df['first_spike_PBE'] - self.all_cells_first_spike_time_df['first_spike_lap']
        # self.all_cells_first_spike_time_df['color'] = self.all_cells_first_spike_time_df['aclu'].map(lambda x: aclu_to_color_map.get(x, [1.0, 1.0, 0.0, 1.0]))
        # column_names = ['first_spike_any', 'first_spike_theta', 'first_spike_lap', 'first_spike_PBE']

        ## plot the spike timecourse:
        # spike_scatter_kwargs = dict(s=25)

        ## find extrema
        # active_col_names = column_names
        active_col_names = ['lap_spike_relative_first_spike', ]
        earliest_first_spike_t: float = self.all_cells_first_spike_time_df[active_col_names].min(axis=0).min()
        latest_first_spike_t: float = self.all_cells_first_spike_time_df[active_col_names].max(axis=0).max()
        # ax.set_xlim(earliest_first_spike_t, latest_first_spike_t)


        # _temp_active_spikes_df = deepcopy(test_obj.all_cells_first_spike_time_df)[['aclu', 'neuron_uid', 'lap_spike_relative_first_spike']].rename(columns={'lap_spike_relative_first_spike':'t_rel_seconds'})
        _temp_active_spikes_df = deepcopy(self.all_cells_first_spike_time_df)[['neuron_uid', 'lap_spike_relative_first_spike']].rename(columns={'lap_spike_relative_first_spike':'t_rel_seconds'})
        # Use pd.factorize to create new integer codes for 'neuron_uid'
        _temp_active_spikes_df['aclu'], uniques = pd.factorize(_temp_active_spikes_df['neuron_uid'])
        # Optionally, add 1 to start 'aclu' from 1 instead of 0
        _temp_active_spikes_df['aclu'] = _temp_active_spikes_df['aclu'] + 1
        # Now, 'aclu' contains unique integer IDs corresponding to 'neuron_uid'
        print(_temp_active_spikes_df[['neuron_uid', 'aclu']].drop_duplicates())

        _temp_active_spikes_df
        # shared_aclus = deepcopy(_temp_active_spikes_df['neuron_uid'].unique())
        shared_aclus = deepcopy(_temp_active_spikes_df['aclu'].unique())
        shared_aclus
        # Assuming _temp_active_spikes_df is your DataFrame


        app, win, plots, plots_data = new_plot_raster_plot(_temp_active_spikes_df, shared_aclus, scatter_plot_kwargs=None,
                                                            scatter_app_name=f'lap_spike_relative_first_spike_raster', defer_show=defer_show, active_context=None)

        root_plot = plots['root_plot']
        # Create a vertical line at x=3
        v_line = CustomInfiniteLine(pos=0.0, angle=90, pen=pg.mkPen('r', width=2), label='first lap spike')
        root_plot.addItem(v_line)
        plots['v_line'] = v_line

        ## Set Labels
        # plots['root_plot'].set_xlabel('First PBE spike relative to first lap spike (t=0)')
        # plots['root_plot'].set_ylabel('Cell')
        plots['root_plot'].setTitle("First PBE spike relative to first lap spike (t=0)", color='white', size='24pt')
        # plots['root_plot'].setLabel('top', 'First PBE spike relative to first lap spike (t=0)', size='22pt') # , color='blue'
        plots['root_plot'].setLabel('left', 'Cell ID', color='white', size='12pt') # , units='V', color='red'
        plots['root_plot'].setLabel('bottom', 'Time (relative to first lap spike for each cell)', color='white', units='s', size='12pt') # , color='blue'


        return app, win, plots, plots_data

    def plot_PhoJonathan_plots_with_time_indicator_lines(self, curr_active_pipeline, included_neuron_ids=None, write_vector_format=False, write_png=True, override_fig_man: Optional[FileOutputManager]=None, time_point_formatting_kwargs_dict=None, n_max_page_rows=1, defer_draw: bool=False):
        """

        """
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import add_time_indicator_lines
        from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
        from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import BatchPhoJonathanFiguresHelper

        if included_neuron_ids is None:
            included_neuron_ids = self.all_cells_first_spike_time_df.aclu.unique()

        if time_point_formatting_kwargs_dict is None:
            time_point_formatting_kwargs_dict = {'lap': dict(color='orange', alpha=0.8), 'PBE': dict(color='purple', alpha=0.8)}

        filtered_cells_first_spike_times: CellsFirstSpikeTimes = self.sliced_by_neuron_id(included_neuron_ids)
        later_lap_appearing_aclus_df = filtered_cells_first_spike_times.all_cells_first_spike_time_df ## find ones that appear only on later laps
        included_neuron_ids = later_lap_appearing_aclus_df['aclu'].to_numpy() ## get the aclus that only appear on later laps

        ## plot each aclu in a separate figures
        active_out_figure_container_dict: Dict[IdentifyingContext, MatplotlibRenderPlots] = BatchPhoJonathanFiguresHelper.perform_run(curr_active_pipeline, shared_aclus=included_neuron_ids, n_max_page_rows=n_max_page_rows, disable_top_row=True,
                                                                                                                                       progress_print=False, write_png=False, write_vector_format=False, # explicitly don't save here, because we need to add the indicator lines
                                                                                                                                    )
        ## Inputs: later_lap_appearing_aclus_df
        later_lap_appearing_aclus_times_dict: Dict[types.aclu_index, Dict[str, float]] = {aclu_tuple.aclu:{'lap': aclu_tuple.first_spike_lap, 'PBE': aclu_tuple.first_spike_PBE} for aclu_tuple in later_lap_appearing_aclus_df.itertuples(index=False)}

        # ## add the lines:
        modified_figure_container_dict: Dict[IdentifyingContext, MatplotlibRenderPlots] = add_time_indicator_lines(active_figures_dict=active_out_figure_container_dict, later_lap_appearing_aclus_times_dict=later_lap_appearing_aclus_times_dict, time_point_formatting_kwargs_dict=time_point_formatting_kwargs_dict, defer_draw=False)

        ## perform saving if needed:
        if (write_png or write_vector_format):
            print(f'perfomring save...')
            saved_file_paths = BatchPhoJonathanFiguresHelper._perform_save_batch_plotted_figures(curr_active_pipeline, active_out_figure_container_dict=modified_figure_container_dict, write_vector_format=write_vector_format, write_png=write_png, override_fig_man=override_fig_man, progress_print=True, debug_print=False)
            print(f'\tsaved_file_paths: {saved_file_paths}')

        return modified_figure_container_dict


# ==================================================================================================================== #
# 2024-10-09 - Building Custom Individual time_bin decoded posteriors                                                  #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['individual_time_bin', 'posterior'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-09 09:27', related_items=[])
def _perform_build_individual_time_bin_decoded_posteriors_df(curr_active_pipeline, track_templates, all_directional_laps_filter_epochs_decoder_result, transfer_column_names_list: Optional[List[str]]=None):
    """
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _perform_build_individual_time_bin_decoded_posteriors_df
    filtered_laps_time_bin_marginals_df = _perform_build_individual_time_bin_decoded_posteriors_df(curr_active_pipeline, track_templates=track_templates, all_directional_laps_filter_epochs_decoder_result=all_directional_laps_filter_epochs_decoder_result)

    """
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import co_filter_epochs_and_spikes

    ## INPUTS: all_directional_laps_filter_epochs_decoder_result
    if transfer_column_names_list is None:
        transfer_column_names_list = []
    # transfer_column_names_list: List[str] = ['maze_id', 'lap_dir', 'lap_id']
    TIME_OVERLAP_PREVENTION_EPSILON: float = 1e-12
    (laps_directional_marginals_tuple, laps_track_identity_marginals_tuple, laps_non_marginalized_decoder_marginals_tuple), laps_marginals_df = all_directional_laps_filter_epochs_decoder_result.compute_marginals(epoch_idx_col_name='lap_idx', epoch_start_t_col_name='lap_start_t',
                                                                                                                                                        additional_transfer_column_names=['start','stop','label','duration','lap_id','lap_dir','maze_id','is_LR_dir'])
    laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir  = laps_directional_marginals_tuple
    laps_track_identity_marginals, laps_track_identity_all_epoch_bins_marginal, laps_most_likely_track_identity_from_decoder, laps_is_most_likely_track_identity_Long = laps_track_identity_marginals_tuple
    non_marginalized_decoder_marginals, non_marginalized_decoder_all_epoch_bins_marginal, most_likely_decoder_idxs, non_marginalized_decoder_all_epoch_bins_decoder_probs_df = laps_non_marginalized_decoder_marginals_tuple
    laps_time_bin_marginals_df: pd.DataFrame = all_directional_laps_filter_epochs_decoder_result.build_per_time_bin_marginals_df(active_marginals_tuple=(laps_directional_marginals, laps_track_identity_marginals, non_marginalized_decoder_marginals),
                                                                                                                                columns_tuple=(['P_LR', 'P_RL'], ['P_Long', 'P_Short'], ['long_LR', 'long_RL', 'short_LR', 'short_RL']), transfer_column_names_list=transfer_column_names_list)
    laps_time_bin_marginals_df['start'] = laps_time_bin_marginals_df['start'] + TIME_OVERLAP_PREVENTION_EPSILON ## ENSURE NON-OVERLAPPING

    ## INPUTS: laps_time_bin_marginals_df
    # active_min_num_unique_aclu_inclusions_requirement: int = track_templates.min_num_unique_aclu_inclusions_requirement(curr_active_pipeline, required_min_percentage_of_active_cells=0.33333333333333)
    active_min_num_unique_aclu_inclusions_requirement = None # must be none for individual `time_bin` periods
    filtered_laps_time_bin_marginals_df, active_spikes_df = co_filter_epochs_and_spikes(active_spikes_df=get_proper_global_spikes_df(curr_active_pipeline, minimum_inclusion_fr_Hz=curr_active_pipeline.global_computation_config.rank_order_shuffle_analysis.minimum_inclusion_fr_Hz),
                                                                    active_epochs_df=laps_time_bin_marginals_df, included_aclus=track_templates.any_decoder_neuron_IDs, min_num_unique_aclu_inclusions=active_min_num_unique_aclu_inclusions_requirement,
                                                                    epoch_id_key_name='lap_individual_time_bin_id', no_interval_fill_value=-1, add_unique_aclus_list_column=True, drop_non_epoch_spikes=True)
    return filtered_laps_time_bin_marginals_df


# ==================================================================================================================== #
# 2024-10-08 - Reliability and Active Cell Testing                                                                     #
# ==================================================================================================================== #

# appearing_or_disappearing_aclus, appearing_stability_df, appearing_aclus, disappearing_stability_df, disappearing_aclus
@function_attributes(short_name=None, tags=['performance'], input_requires=[], output_provides=[], uses=['_do_train_test_split_decode_and_evaluate'], used_by=[], creation_date='2024-10-08 00:00', related_items=[])
def _perform_run_rigorous_decoder_performance_assessment(curr_active_pipeline, included_neuron_IDs, active_laps_decoding_time_bin_size: float = 0.25, force_recompute_directional_train_test_split_result: bool = False, debug_print=False):
    """ runs for a specific subset of cells
    """
    # Inputs: all_directional_pf1D_Decoder, alt_directional_merged_decoders_result
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrainTestSplitResult, TrainTestLapsSplitting, CustomDecodeEpochsResult, decoder_name, epoch_split_key, get_proper_global_spikes_df, DirectionalPseudo2DDecodersResult
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _do_train_test_split_decode_and_evaluate
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import PfND
    from neuropy.core.session.dataSession import Laps
    # from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _check_result_laps_epochs_df_performance

    # t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    # global_session = curr_active_pipeline.filtered_sessions[global_epoch_name]

    directional_train_test_split_result: TrainTestSplitResult = curr_active_pipeline.global_computation_results.computed_data.get('TrainTestSplit', None)

    if (directional_train_test_split_result is None) or force_recompute_directional_train_test_split_result:
        ## recompute
        if debug_print:
            print(f"'TrainTestSplit' not computed, recomputing...")
        curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_train_test_split'], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
        directional_train_test_split_result: TrainTestSplitResult = curr_active_pipeline.global_computation_results.computed_data['TrainTestSplit']
        assert directional_train_test_split_result is not None, f"faiiled even after recomputation"
        if debug_print:
            print('\tdone.')

    training_data_portion: float = directional_train_test_split_result.training_data_portion
    test_data_portion: float = directional_train_test_split_result.test_data_portion
    if debug_print:
        print(f'training_data_portion: {training_data_portion}, test_data_portion: {test_data_portion}')

    # test_epochs_dict: Dict[types.DecoderName, pd.DataFrame] = directional_train_test_split_result.test_epochs_dict
    # train_epochs_dict: Dict[types.DecoderName, pd.DataFrame] = directional_train_test_split_result.train_epochs_dict
    # train_lap_specific_pf1D_Decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = directional_train_test_split_result.train_lap_specific_pf1D_Decoder_dict
    # OUTPUTS: train_test_split_laps_df_dict

    # MAIN _______________________________________________________________________________________________________________ #

    complete_decoded_context_correctness_tuple, laps_marginals_df, all_directional_pf1D_Decoder, all_test_epochs_df, test_all_directional_decoder_result, all_directional_laps_filter_epochs_decoder_result, _out_separate_decoder_results = _do_train_test_split_decode_and_evaluate(curr_active_pipeline=curr_active_pipeline, active_laps_decoding_time_bin_size=active_laps_decoding_time_bin_size,
                                                                                                                                                                                                                                                  included_neuron_IDs=included_neuron_IDs,
                                                                                                                                                                                                                                                  force_recompute_directional_train_test_split_result=force_recompute_directional_train_test_split_result, compute_separate_decoder_results=True, debug_print=debug_print)
    (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = complete_decoded_context_correctness_tuple
    if debug_print:
        print(f"percent_laps_track_identity_estimated_correctly: {round(percent_laps_track_identity_estimated_correctly*100.0, ndigits=3)}%")

    if _out_separate_decoder_results is not None:
        assert len(_out_separate_decoder_results) == 3, f"_out_separate_decoder_results: {_out_separate_decoder_results}"
        test_decoder_results_dict, train_decoded_results_dict, train_decoded_measured_diff_df_dict = _out_separate_decoder_results
        ## OUTPUTS: test_decoder_results_dict, train_decoded_results_dict
    # _remerged_laps_dfs_dict = {}
    # for a_decoder_name, a_test_epochs_df in test_epochs_dict.items():
    #     a_train_epochs_df = train_epochs_dict[a_decoder_name]
    #     a_train_epochs_df['test_train_epoch_type'] = 'train'
    #     a_test_epochs_df['test_train_epoch_type'] = 'test'
    #     _remerged_laps_dfs_dict[a_decoder_name] = pd.concat([a_train_epochs_df, a_test_epochs_df], axis='index')
    #     _remerged_laps_dfs_dict[a_decoder_name] = _add_extra_epochs_df_columns(epochs_df=_remerged_laps_dfs_dict[a_decoder_name])

    ## INPUTS: test_all_directional_decoder_result, all_directional_pf1D_Decoder
    # epochs_bin_by_bin_performance_analysis_df = test_all_directional_decoder_result.get_lap_bin_by_bin_performance_analysis_df(active_pf_2D=deepcopy(all_directional_pf1D_Decoder), debug_print=debug_print) # active_pf_2D: used for binning position columns # active_pf_2D: used for binning position columns
    # epochs_bin_by_bin_performance_analysis_df: pd.DataFrame = test_all_directional_decoder_result.epochs_bin_by_bin_performance_analysis_df
    # _out_subset_decode_dict[active_laps_decoding_time_bin_size] = epochs_track_identity_marginal_df
    # epochs_bin_by_bin_performance_analysis_df['shuffle_idx'] = int(i)

    return (complete_decoded_context_correctness_tuple, laps_marginals_df, all_directional_pf1D_Decoder, all_test_epochs_df, test_all_directional_decoder_result, all_directional_laps_filter_epochs_decoder_result, _out_separate_decoder_results)


@function_attributes(short_name=None, tags=['long_short', 'firing_rate'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-09-17 05:22', related_items=['determine_neuron_exclusivity_from_firing_rate'])
def compute_all_cells_long_short_firing_rate_df(global_spikes_df: pd.DataFrame):
    """ computes the firing rates for all cells (not just placecells or excitatory cells) for the long and short track periods, and then their differences
    These firing rates are not spatially binned because they aren't just place cells.

    columns: ['LS_diff_firing_rate_Hz']: will be positive for Short-preferring cells and negative for Long-preferring ones.

    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_all_cells_long_short_firing_rate_df

        df_combined = compute_all_cells_long_short_firing_rate_df(global_spikes_df=global_spikes_df)
        df_combined

        print(list(df_combined.columns)) # ['long_num_spikes_count', 'short_num_spikes_count', 'global_num_spikes_count', 'long_firing_rate_Hz', 'short_firing_rate_Hz', 'global_firing_rate_Hz', 'LS_diff_firing_rate_Hz', 'firing_rate_percent_diff']

    """
    ## Needs to consider not only place cells but interneurons as well
    # global_all_spikes_counts # 73 rows
    # global_spikes_df.aclu.unique() # 108

    ## Split into the pre- and post- delta epochs
    # global_spikes_df['t_rel_seconds']
    # global_spikes_df

    # is_theta, is_ripple, maze_id, maze_relative_lap

    ## Split on 'maze_id'
    # partition

    from pyphocorehelpers.indexing_helpers import partition_df, reorder_columns_relative
    # from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import add_spikes_df_placefield_inclusion_columns


    ## INPUTS: global_spikes_df

    ## Compute effective epoch duration by finding the earliest and latest spike in epoch.
    def _add_firing_rates_from_computed_durations(a_df: pd.DataFrame):
        spike_times = a_df['t_rel_seconds'].values
        end_t = np.nanmax(spike_times)
        start_t = np.nanmin(spike_times)
        duration_t: float = end_t - start_t
        return duration_t, (start_t, end_t)


    partitioned_dfs = dict(zip(*partition_df(global_spikes_df, partitionColumn='maze_id'))) # non-maze is also an option, right?
    long_all_spikes_df: pd.DataFrame = partitioned_dfs[1]
    short_all_spikes_df: pd.DataFrame = partitioned_dfs[2]

    ## sum total number of spikes over the entire duration
    # Performed 1 aggregation grouped on column: 'aclu'
    long_all_spikes_count_df = long_all_spikes_df.groupby(['aclu']).agg(num_spikes_count=('t_rel_seconds', 'count')).reset_index()[['aclu', 'num_spikes_count']].set_index('aclu')
    # Performed 1 aggregation grouped on column: 'aclu'
    short_all_spikes_count_df = short_all_spikes_df.groupby(['aclu']).agg(num_spikes_count=('t_rel_seconds', 'count')).reset_index()[['aclu', 'num_spikes_count']].set_index('aclu')

    ## TODO: exclude replay periods

    ## OUTPUTS: long_all_spikes_count_df, short_all_spikes_count_df

    long_duration_t, _long_start_end_tuple = _add_firing_rates_from_computed_durations(long_all_spikes_df)
    long_all_spikes_count_df['firing_rate_Hz'] = long_all_spikes_count_df['num_spikes_count'] / long_duration_t

    short_duration_t, _short_start_end_tuple = _add_firing_rates_from_computed_durations(short_all_spikes_df)
    short_all_spikes_count_df['firing_rate_Hz'] = short_all_spikes_count_df['num_spikes_count'] / short_duration_t

    global_duration_t: float = long_duration_t + short_duration_t

    ## OUTPUTS: long_all_spikes_count_df, short_all_spikes_count_df

    # long_all_spikes_count_df
    # short_all_spikes_count_df

    # Performed 2 aggregations grouped on column: 't_rel_seconds'
    # long_all_spikes_df[['t_rel_seconds']].agg(t_rel_seconds_min=('t_rel_seconds', 'min'), t_rel_seconds_max=('t_rel_seconds', 'max')).reset_index()

    # short_all_spikes_df[['t_rel_seconds']].agg(t_rel_seconds_min=('t_rel_seconds', 'min'), t_rel_seconds_max=('t_rel_seconds', 'max')).reset_index()
    # long_all_spikes_df = long_all_spikes_df.groupby(['t_rel_seconds']).agg(t_rel_seconds_min=('t_rel_seconds', 'min'), t_rel_seconds_max=('t_rel_seconds', 'max')).reset_index()

    # Add prefixes to column names
    df1_prefixed = long_all_spikes_count_df.add_prefix("long_")
    df2_prefixed = short_all_spikes_count_df.add_prefix("short_")

    # Combine along the index
    df_combined = pd.concat([df1_prefixed, df2_prefixed], axis=1)

    ## Move the "height" columns to the end
    # df_combined = reorder_columns_relative(df_combined, column_names=list(filter(lambda column: column.endswith('_firing_rate_Hz'), existing_columns)), relative_mode='end')
    # df_combined = reorder_columns_relative(df_combined, column_names=['long_firing_rate_Hz', 'short_firing_rate_Hz'], relative_mode='end')

    df_combined = reorder_columns_relative(df_combined, column_names=['long_num_spikes_count', 'short_num_spikes_count', 'long_firing_rate_Hz', 'short_firing_rate_Hz'], relative_mode='end')

    # ['long_firing_rate_Hz', 'short_firing_rate_Hz', 'long_num_spikes_count', 'short_num_spikes_count', 'LS_diff_firing_rate_Hz', 'firing_rate_percent_diff']

    ## Compare the differnece between the two periods
    df_combined['LS_diff_firing_rate_Hz'] = df_combined['long_firing_rate_Hz'] - df_combined['short_firing_rate_Hz']

    # Calculate the percent difference in firing rate
    df_combined["firing_rate_percent_diff"] = (df_combined['LS_diff_firing_rate_Hz'] / df_combined["long_firing_rate_Hz"]) * 100

    df_combined['global_num_spikes_count'] = df_combined['long_num_spikes_count'] + df_combined['short_num_spikes_count']
    df_combined['global_firing_rate_Hz'] = df_combined['global_num_spikes_count'] / global_duration_t

    df_combined = reorder_columns_relative(df_combined, column_names=['long_num_spikes_count', 'short_num_spikes_count', 'global_num_spikes_count', 'long_firing_rate_Hz', 'short_firing_rate_Hz', 'global_firing_rate_Hz'], relative_mode='start')\


    # df_combined["long_num_spikes_percent"] = (df_combined['long_num_spikes_count'] / df_combined["global_num_spikes_count"]) * 100
    # df_combined["short_num_spikes_percent"] = (df_combined['short_num_spikes_count'] / df_combined["global_num_spikes_count"]) * 100

    # df_combined["firing_rate_percent_diff"] = (df_combined['LS_diff_firing_rate_Hz'] / df_combined["long_firing_rate_Hz"]) * 100


    return df_combined

@function_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-10-08 10:37', related_items=['compute_all_cells_long_short_firing_rate_df'])
def determine_neuron_exclusivity_from_firing_rate(df_combined: pd.DataFrame, firing_rate_required_diff_Hz: float = 1.0, maximum_opposite_period_firing_rate_Hz: float = 1.0):
    """
    firing_rate_required_diff_Hz: float = 1.0 # minimum difference required for a cell to be considered Long- or Short-"preferring"
    maximum_opposite_period_firing_rate_Hz: float = 1.0 # maximum allowed firing rate in the opposite period to be considered exclusive

    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_all_cells_long_short_firing_rate_df, determine_neuron_exclusivity_from_firing_rate

        df_combined = compute_all_cells_long_short_firing_rate_df(global_spikes_df=global_spikes_df)
        (LpC_df, SpC_df, LxC_df, SxC_df), (LpC_aclus, SpC_aclus, LxC_aclus, SxC_aclus) = determine_neuron_exclusivity_from_firing_rate(df_combined=df_combined, firing_rate_required_diff_Hz=firing_rate_required_diff_Hz,
                                                                                                                               maximum_opposite_period_firing_rate_Hz=maximum_opposite_period_firing_rate_Hz)

        ## Extract the aclus
        print(f'LpC_aclus: {LpC_aclus}')
        print(f'SpC_aclus: {SpC_aclus}')

        print(f'LxC_aclus: {LxC_aclus}')
        print(f'SxC_aclus: {SxC_aclus}')

    """
    # Sort by column: 'LS_diff_firing_rate_Hz' (ascending)
    df_combined = df_combined.sort_values(['LS_diff_firing_rate_Hz'])
    # df_combined = df_combined.sort_values(['firing_rate_percent_diff'])
    df_combined

    # df_combined['LS_diff_firing_rate_Hz']

    # df_combined['firing_rate_percent_diff']

    LpC_df = df_combined[df_combined['LS_diff_firing_rate_Hz'] > firing_rate_required_diff_Hz]
    SpC_df = df_combined[df_combined['LS_diff_firing_rate_Hz'] < -firing_rate_required_diff_Hz]

    LxC_df = LpC_df[LpC_df['short_firing_rate_Hz'] <= maximum_opposite_period_firing_rate_Hz]
    SxC_df = SpC_df[SpC_df['long_firing_rate_Hz'] <= maximum_opposite_period_firing_rate_Hz]


    ## Let's consider +/- 50% diff XxC cells
    # LpC_df = df_combined[df_combined['firing_rate_percent_diff'] > 50.0]
    # SpC_df = df_combined[df_combined['firing_rate_percent_diff'] < -50.0]

    ## Extract the aclus"
    LpC_aclus = LpC_df.index.values
    SpC_aclus = SpC_df.index.values

    print(f'LpC_aclus: {LpC_aclus}')
    print(f'SpC_aclus: {SpC_aclus}')

    LxC_aclus = LxC_df.index.values
    SxC_aclus = SxC_df.index.values

    print(f'LxC_aclus: {LxC_aclus}')
    print(f'SxC_aclus: {SxC_aclus}')

    ## OUTPUTS: LpC_df, SpC_df, LxC_df, SxC_df

    ## OUTPUTS: LpC_aclus, SpC_aclus, LxC_aclus, SxC_aclus
    return (LpC_df, SpC_df, LxC_df, SxC_df), (LpC_aclus, SpC_aclus, LxC_aclus, SxC_aclus)


# ==================================================================================================================== #
# 2024-10-04 - Parsing `ProgrammaticDisplayFunctionTesting` output folder                                              #
# ==================================================================================================================== #
from pyphocorehelpers.assertion_helpers import Assert
from pyphocorehelpers.indexing_helpers import partition_df_dict, partition
from typing import Dict, List, Tuple, Optional, Callable, Union, Any, NewType
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types

ContextDescStr = NewType('ContextDescStr', str) # like '2023-07-11_kdiba_gor01_one'
ImageNameStr = NewType('ImageNameStr', str) # like '2006-6-07_11-26-53/kdiba_gor01_one_2006-6-07_11-26-53_maze1__display_1d_placefield_validations.pdf'

class ProgrammaticDisplayFunctionTestingFolderImageLoading:
    """ Loads image from the folder
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import ProgrammaticDisplayFunctionTestingFolderImageLoading

    """

    @function_attributes(short_name=None, tags=['ProgrammaticDisplayFunctionTesting', 'parse', 'filesystem'], input_requires=[], output_provides=[], uses=[], used_by=['parse_ProgrammaticDisplayFunctionTesting_image_folder'], creation_date='2024-10-04 12:21', related_items=[])
    @classmethod
    def parse_image_path(cls, programmatic_display_function_testing_path: Path, file_path: Path, debug_print=False) -> Tuple[IdentifyingContext, str, datetime]:
        """ Parses `"C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting"`
        "C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting/2023-04-11/kdiba/gor01/one/2006-6-09_1-22-43/kdiba_gor01_one_2006-6-09_1-22-43_batch_plot_test_long_only_[45].png"
        Write a function that parses the following path structure: "./2023-04-11/kdiba/gor01/one/2006-6-09_1-22-43/kdiba_gor01_one_2006-6-09_1-22-43_batch_plot_test_long_only_[45].png"
        Into the following variable names: `/image_export_day_date/format_name/animal/exper_name/session_name/image_name`
        """
        test_relative_image_path = file_path.relative_to(programmatic_display_function_testing_path) # .resolve() ## RESOLVE MESSES UP SYMLINKS!
        if debug_print:
            print(f'{test_relative_image_path = }')

        # Split the path into components
        # parts = file_path.strip("./").split(os.sep)
        # Convert to a list of path components
        parts = test_relative_image_path.parts
        if debug_print:
            print(f'parts: {parts}')

        if len(parts) < 6:
            raise ValueError(f'parsed path should have at least 6 parts, but this one only has: {len(parts)}.\nparts: {parts}')

        if len(parts) > 6:
            joined_final_part: str = '/'.join(parts[5:]) # return anything after that back into a str
            parts = parts[:5] + (joined_final_part, )

        Assert.len_equals(parts, 6)

        # Assign the variables from the path components
        image_export_day_date = parts[0]      # "./2023-04-11"
        format_name = parts[1]                # "kdiba"
        animal = parts[2]                     # "gor01"
        exper_name = parts[3]                 # "one"
        session_name = parts[4]               # "2006-6-09_1-22-43"
        image_name = parts[5]                 # "kdiba_gor01_one_2006-6-09_1-22-43_batch_plot_test_long_only_[45].png"

        session_context = IdentifyingContext(format_name=format_name, animal=animal, exper_name=exper_name, session_name=session_name)
        # Parse image_export_day_date as a date (YYYY-mm-dd)
        image_export_day_date: datetime = datetime.strptime(image_export_day_date, "%Y-%m-%d")

        # return image_export_day_date, format_name, animal, exper_name, session_name, image_name
        return session_context, image_name, image_export_day_date, file_path


    @function_attributes(short_name=None, tags=['ProgrammaticDisplayFunctionTesting', 'filesystem', 'images', 'load'], input_requires=[], output_provides=[], uses=['parse_image_path'], used_by=[], creation_date='2024-10-04 12:21', related_items=[])
    @classmethod
    def parse_ProgrammaticDisplayFunctionTesting_image_folder(cls, programmatic_display_function_testing_path: Path, save_csv: bool = True):
        """
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import ProgrammaticDisplayFunctionTestingFolderImageLoading

            programmatic_display_function_testing_path: Path = Path('/home/halechr/repos/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting').resolve()
            programmatic_display_function_outputs_df, programmatic_display_function_outputs_tuples, csv_out_path = ProgrammaticDisplayFunctionTestingFolderImageLoading.parse_ProgrammaticDisplayFunctionTesting_image_folder(programmatic_display_function_testing_path=programmatic_display_function_testing_path)
            programmatic_display_function_outputs_df

        """
        # programmatic_display_function_outputs_dict: Dict[IdentifyingContext, List] = {}
        programmatic_display_function_outputs_tuples: List[Tuple[IdentifyingContext, str, datetime]] = []

        Assert.path_exists(programmatic_display_function_testing_path)

        # Recursively enumerate all files in the directory
        def enumerate_files(directory: Path):
            return [file for file in directory.rglob('*') if file.is_file()]

        # Example usage
        all_files = enumerate_files(programmatic_display_function_testing_path)

        for test_image_path in all_files:
            try:
                # image_export_day_date, format_name, animal, exper_name, session_name, image_name = parse_image_path(programmatic_display_function_testing_path, test_image_path)
                # session_context, image_name, image_export_day_date = parse_image_path(programmatic_display_function_testing_path, test_image_path)
                # print(image_export_day_date, format_name, animal, exper_name, session_name, image_name)
                programmatic_display_function_outputs_tuples.append(cls.parse_image_path(programmatic_display_function_testing_path, test_image_path))
            except ValueError as e:
                # couldn't parse, skipping
                pass
            except Exception as e:
                raise e

        programmatic_display_function_outputs_df: pd.DataFrame = pd.DataFrame.from_records(programmatic_display_function_outputs_tuples, columns=['context', 'image_name', 'export_date', 'file_path'])
        # Sort by columns: 'context' (ascending), 'image_name' (ascending), 'export_date' (descending)
        programmatic_display_function_outputs_df = programmatic_display_function_outputs_df.sort_values(['context', 'image_name', 'export_date'], ascending=[True, True, False], key=lambda s: s.apply(str) if s.name in ['context'] else s).reset_index(drop=True)
        if save_csv:
            csv_out_path = programmatic_display_function_testing_path.joinpath('../../PhoDibaPaper2024Book/data').resolve().joinpath('programmatic_display_function_image_paths.csv')
            programmatic_display_function_outputs_df.to_csv(csv_out_path)

        return programmatic_display_function_outputs_df, programmatic_display_function_outputs_tuples, csv_out_path


    # @classmethod
    # def load_saved_ProgrammaticDisplayFunctionTesting_csv_and_build_widget(cls, programmatic_display_function_outputs_df: pd.DataFrame):
    @classmethod
    def build_ProgrammaticDisplayFunctionTesting_browsing_widget(cls, programmatic_display_function_outputs_df: pd.DataFrame):
        """
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import parse_ProgrammaticDisplayFunctionTesting_image_folder

            programmatic_display_function_testing_path: Path = Path('/home/halechr/repos/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting').resolve()
            programmatic_display_function_outputs_df, programmatic_display_function_outputs_tuples, csv_out_path = ProgrammaticDisplayFunctionTestingFolderImageLoading.parse_ProgrammaticDisplayFunctionTesting_image_folder(programmatic_display_function_testing_path=programmatic_display_function_testing_path)
            programmatic_display_function_outputs_df

        """
        from pyphocorehelpers.gui.Jupyter.JupyterImageNavigatorWidget import ImageNavigator, ContextSidebar, build_context_images_navigator_widget

        # Assert.path_exists(in_path)

        # _final_out_dict_dict: Dict[ContextDescStr, Dict[ImageNameStr, List[Tuple[str, str]]]] = {}
        _final_out_dict_dict: Dict[ContextDescStr, Dict[ImageNameStr, Dict[datetime, Path]]] = {}
        for ctx, a_ctx_df in partition_df_dict(programmatic_display_function_outputs_df, partitionColumn='context').items():
            _final_out_dict_dict[ctx] = {}
            for an_img_name, an_img_df in partition_df_dict(a_ctx_df, partitionColumn='image_name').items():
                # _final_out_dict_dict[ctx][an_img_name] = list(zip(an_img_df['export_date'].values, an_img_df['file_path'].values)) #partition_df_dict(an_img_df, partitionColumn='image_name')
                _final_out_dict_dict[ctx][an_img_name] = {datetime.strptime(k, "%Y-%m-%d"):Path(v).resolve() for k, v in dict(zip(an_img_df['export_date'].values, an_img_df['file_path'].values)).items() if v.endswith('.png')}

        """
        {'2023-07-11_kdiba_gor01_one': {'2006-6-07_11-26-53/kdiba_gor01_one_2006-6-07_11-26-53_maze1__display_1d_placefield_validations.pdf': [('2023-07-11',
            'C:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\EXTERNAL\\Screenshots\\ProgrammaticDisplayFunctionTesting\\2023-07-11\\2023-07-11\\kdiba\\gor01\\one\\2006-6-07_11-26-53\\kdiba_gor01_one_2006-6-07_11-26-53_maze1__display_1d_placefield_validations.pdf')],
        '2006-6-07_11-26-53/kdiba_gor01_one_2006-6-07_11-26-53_maze2__display_1d_placefield_validations.pdf': [('2023-07-11',
            'C:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\EXTERNAL\\Screenshots\\ProgrammaticDisplayFunctionTesting\\2023-07-11\\2023-07-11\\kdiba\\gor01\\one\\2006-6-07_11-26-53\\kdiba_gor01_one_2006-6-07_11-26-53_maze2__display_1d_placefield_validations.pdf')],
            ...
        """
        ## INPUTS: _final_out_dict_dict: Dict[ContextDescStr, Dict[ImageNameStr, Dict[datetime, Path]]]
        context_tabs_dict = {curr_context_desc_str:build_context_images_navigator_widget(curr_context_images_dict, curr_context_desc_str=curr_context_desc_str, max_num_widget_debug=2) for curr_context_desc_str, curr_context_images_dict in list(_final_out_dict_dict.items())}
        sidebar = ContextSidebar(context_tabs_dict)


        return sidebar, context_tabs_dict, _final_out_dict_dict





# ==================================================================================================================== #
# 2024-08-21 Plotting Generated Transition Matrix Sequences                                                            #
# ==================================================================================================================== #
from matplotlib.colors import Normalize

@function_attributes(short_name=None, tags=['colormap', 'grayscale', 'image'], input_requires=[], output_provides=[], uses=[], used_by=['blend_images'], creation_date='2024-08-21 00:00', related_items=[])
def apply_colormap(image: np.ndarray, color: tuple) -> np.ndarray:
    colored_image = np.zeros((*image.shape, 3), dtype=np.float32)
    for i in range(3):
        colored_image[..., i] = image * color[i]
    return colored_image

@function_attributes(short_name=None, tags=['image'], input_requires=[], output_provides=[], uses=['apply_colormap'], used_by=[], creation_date='2024-08-21 00:00', related_items=[])
def blend_images(images: list, cmap=None) -> np.ndarray:
    """ Tries to pre-combine images to produce an output image of the same size

    # 'coolwarm'
    images = [a_seq_mat.todense().T for i, a_seq_mat in enumerate(sequence_frames_sparse)]
    blended_image = blend_images(images)
    # blended_image = blend_images(images, cmap='coolwarm')
    blended_image

    # blended_image = Image.fromarray(blended_image, mode="RGB")
    # # blended_image = get_array_as_image(blended_image, desired_height=100, desired_width=None, skip_img_normalization=True)
    # blended_image


    """
    if cmap is None:
        # Non-colormap mode:
        # Ensure images are in the same shape
        combined_image = np.zeros_like(images[0], dtype=np.float32)

        for img in images:
            combined_image += img.astype(np.float32)

    else:
        # colormap mode
        # Define a colormap (blue to red)
        cmap = plt.get_cmap(cmap)
        norm = Normalize(vmin=0, vmax=(len(images) - 1))

        combined_image = np.zeros((*images[0].shape, 3), dtype=np.float32)

        for i, img in enumerate(images):
            color = cmap(norm(i))[:3]  # Get RGB color from colormap
            colored_image = apply_colormap(img, color)
            combined_image += colored_image

    combined_image = np.clip(combined_image, 0, 255)  # Ensure pixel values are within valid range
    return combined_image.astype(np.uint8)


def visualize_multiple_image_items(images: list, threshold=1e-3) -> None:
    """ Sample multiple pg.ImageItems overlayed on one another

    # Example usage:
    image1 = np.random.rand(100, 100) * 100  # Example image 1
    image2 = np.random.rand(100, 100) * 100  # Example image 2
    image3 = np.random.rand(100, 100) * 100  # Example image 3

    image1
    # Define the threshold

    _out = visualize_multiple_image_items([image1, image2, image3], threshold=50)

    """
    app = pg.mkQApp('visualize_multiple_image_items')  # Initialize the Qt application
    win = pg.GraphicsLayoutWidget(show=True)
    view = win.addViewBox()
    view.setAspectLocked(True)

    for img in images:
        if threshold is not None:
            # Create a masked array, masking values below the threshold
            img = np.ma.masked_less(img, threshold)

        image_item = pg.ImageItem(img)
        view.addItem(image_item)

    # QtGui.QApplication.instance().exec_()
    return app, win, view


# ==================================================================================================================== #
# 2024-08-16 - Image Processing Techniques                                                                             #
# ==================================================================================================================== #
def plot_grad_quiver(sobel_x, sobel_y, downsample_step=1):
    """

    # Compute the magnitude of the gradient
    gradient_magnitude = np.hypot(sobel_x, sobel_y)
    gradient_direction = np.arctan2(sobel_y, sobel_x)

    """
    # Compute the magnitude of the gradient
    gradient_magnitude = np.hypot(sobel_x, sobel_y)
    gradient_direction = np.arctan2(sobel_y, sobel_x)

    # Create a grid of coordinates for plotting arrows
    Y, X = np.meshgrid(np.arange(gradient_magnitude.shape[0]), np.arange(gradient_magnitude.shape[1]), indexing='ij')

    # Downsample the arrow plot for better visualization (optional)

    X_downsampled = X[::downsample_step, ::downsample_step]
    Y_downsampled = Y[::downsample_step, ::downsample_step]
    sobel_x_downsampled = sobel_x[::downsample_step, ::downsample_step]
    sobel_y_downsampled = sobel_y[::downsample_step, ::downsample_step]

    # Plotting the gradient magnitude and arrows representing the direction
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(gradient_magnitude, cmap='gray', origin='lower')
    plt.quiver(X_downsampled, Y_downsampled, sobel_x_downsampled, sobel_y_downsampled,
            color='red', angles='xy', scale_units='xy') # , scale=5, width=0.01
    plt.title('Gradient Magnitude with Direction Arrows')
    plt.axis('off')
    plt.show()

    return fig



# ==================================================================================================================== #
# 2024-07-15 - Factored out of Across Session Notebook                                                                 #
# ==================================================================================================================== #





# ==================================================================================================================== #
# 2024-06-26 - Shuffled WCorr Output with working histogram                                                            #
# ==================================================================================================================== #
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
from typing import NewType

import neuropy.utils.type_aliases as types
from neuropy.utils.misc import build_shuffled_ids, shuffle_ids # used in _SHELL_analyze_leave_one_out_decoding_results
from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.SequenceBasedComputations import WCorrShuffle, SequenceBasedComputationsContainer

from neuropy.utils.mixins.indexing_helpers import get_dict_subset

from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme

DecodedEpochsResultsDict = NewType('DecodedEpochsResultsDict', Dict[types.DecoderName, DecodedFilterEpochsResult]) # A Dict containing the decoded filter epochs result for each of the four 1D decoder names
ShuffleIdx = NewType('ShuffleIdx', int)

# ---------------------------------------------------------------------------- #
#                      2024-06-15 - Significant Remapping                      #
# ---------------------------------------------------------------------------- #
@function_attributes(short_name=None, tags=['remapping'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-06-15 00:00', related_items=[])
def _add_cell_remapping_category(neuron_replay_stats_df, loaded_track_limits: Dict, x_midpoint: float=72.0):
    """ yanked from `_perform_long_short_endcap_analysis to compute within the batch processing notebook

    'has_long_pf', 'has_short_pf'

    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _add_cell_remapping_category

        neuron_replay_stats_df = deepcopy(neuron_replay_stats_table)
        neuron_replay_stats_df, (non_disappearing_endcap_cells_df, disappearing_endcap_cells_df, minorly_changed_endcap_cells_df, significant_distant_remapping_endcap_aclus) = _add_cell_remapping_category(neuron_replay_stats_df=neuron_replay_stats_df,
                                                            loaded_track_limits = {'long_xlim': np.array([59.0774, 228.69]), 'short_xlim': np.array([94.0156, 193.757]), 'long_ylim': np.array([138.164, 146.12]), 'short_ylim': np.array([138.021, 146.263])},
        )
        neuron_replay_stats_df

    """
    # `loaded_track_limits` = deepcopy(owning_pipeline_reference.sess.config.loaded_track_limits) # {'long_xlim': array([59.0774, 228.69]), 'short_xlim': array([94.0156, 193.757]), 'long_ylim': array([138.164, 146.12]), 'short_ylim': array([138.021, 146.263])}
    # x_midpoint: float = owning_pipeline_reference.sess.config.x_midpoint
    # pix2cm: float = owning_pipeline_reference.sess.config.pix2cm

    ## INPUTS: loaded_track_limits
    print(f'loaded_track_limits: {loaded_track_limits}')

    if 'has_long_pf' not in neuron_replay_stats_df.columns:
        neuron_replay_stats_df['has_long_pf'] = np.logical_not(np.isnan(neuron_replay_stats_df['long_pf_peak_x']))
    if 'has_short_pf' not in neuron_replay_stats_df.columns:
        neuron_replay_stats_df['has_short_pf'] = np.logical_not(np.isnan(neuron_replay_stats_df['short_pf_peak_x']))

    long_xlim = loaded_track_limits['long_xlim']
    # long_ylim = loaded_track_limits['long_ylim']
    short_xlim = loaded_track_limits['short_xlim']
    # short_ylim = loaded_track_limits['short_ylim']

    occupancy_midpoint: float = x_midpoint # 142.7512402496278 # 150.0
    left_cap_x_bound: float = (long_xlim[0] - x_midpoint) #-shift by midpoint - 72.0 # on long track
    right_cap_x_bound: float = (long_xlim[1] - x_midpoint) # 72.0 # on long track
    min_significant_remapping_x_distance: float = 50.0 # from long->short track
    # min_significant_remapping_x_distance: float = 100.0

    ## STATIC:
    # occupancy_midpoint: float = 142.7512402496278 # 150.0
    # left_cap_x_bound: float = -72.0 # on long track
    # right_cap_x_bound: float = 72.0 # on long track
    # min_significant_remapping_x_distance: float = 40.0 # from long->short track

    # Extract the peaks of the long placefields to find ones that have peaks outside the boundaries
    long_pf_peaks = neuron_replay_stats_df[neuron_replay_stats_df['has_long_pf']]['long_pf_peak_x'] - occupancy_midpoint # this shift of `occupancy_midpoint` is to center the midpoint of the track at 0.
    is_left_cap = (long_pf_peaks < left_cap_x_bound)
    is_right_cap = (long_pf_peaks > right_cap_x_bound)
    # is_either_cap =  np.logical_or(is_left_cap, is_right_cap)

    # Adds ['is_long_peak_left_cap', 'is_long_peak_right_cap', 'is_long_peak_either_cap'] columns:
    neuron_replay_stats_df['is_long_peak_left_cap'] = False
    neuron_replay_stats_df['is_long_peak_right_cap'] = False
    neuron_replay_stats_df.loc[is_left_cap.index, 'is_long_peak_left_cap'] = is_left_cap # True
    neuron_replay_stats_df.loc[is_right_cap.index, 'is_long_peak_right_cap'] = is_right_cap # True

    neuron_replay_stats_df['is_long_peak_either_cap'] = np.logical_or(neuron_replay_stats_df['is_long_peak_left_cap'], neuron_replay_stats_df['is_long_peak_right_cap'])

    # adds ['LS_pf_peak_x_diff'] column
    neuron_replay_stats_df['LS_pf_peak_x_diff'] = neuron_replay_stats_df['long_pf_peak_x'] - neuron_replay_stats_df['short_pf_peak_x']

    cap_cells_df: pd.DataFrame = neuron_replay_stats_df[np.logical_and(neuron_replay_stats_df['has_long_pf'], neuron_replay_stats_df['is_long_peak_either_cap'])]
    num_total_endcap_cells: int = len(cap_cells_df)

    # "Disppearing" cells fall below the 1Hz firing criteria on the short track:
    disappearing_endcap_cells_df: pd.DataFrame = cap_cells_df[np.logical_not(cap_cells_df['has_short_pf'])]
    num_disappearing_endcap_cells: int = len(disappearing_endcap_cells_df)
    print(f'num_disappearing_endcap_cells/num_total_endcap_cells: {num_disappearing_endcap_cells}/{num_total_endcap_cells}')

    non_disappearing_endcap_cells_df: pd.DataFrame = cap_cells_df[cap_cells_df['has_short_pf']] # "non_disappearing" cells are those with a placefield on the short track as well
    num_non_disappearing_endcap_cells: int = len(non_disappearing_endcap_cells_df)
    print(f'num_non_disappearing_endcap_cells/num_total_endcap_cells: {num_non_disappearing_endcap_cells}/{num_total_endcap_cells}')

    # Classify the non_disappearing cells into two groups:
    # 1. Those that exhibit significant remapping onto somewhere else on the track
    non_disappearing_endcap_cells_df['has_significant_distance_remapping'] = (np.abs(non_disappearing_endcap_cells_df['LS_pf_peak_x_diff']) >= min_significant_remapping_x_distance) # The most a placefield could translate intwards would be (35 + (pf_width/2.0)) I think.
    num_significant_position_remappping_endcap_cells: int = len(non_disappearing_endcap_cells_df[non_disappearing_endcap_cells_df['has_significant_distance_remapping'] == True])
    print(f'num_significant_position_remappping_endcap_cells/num_non_disappearing_endcap_cells: {num_significant_position_remappping_endcap_cells}/{num_non_disappearing_endcap_cells}')

    # 2. Those that seem to remain where they were on the long track, perhaps being "sampling-clipped" or translated adjacent to the platform. These two subcases can be distinguished by a change in the placefield's length (truncated cells would be a fraction of the length, although might need to account for scaling with new track length)
    significant_distant_remapping_endcap_cells_df: pd.DataFrame = non_disappearing_endcap_cells_df[non_disappearing_endcap_cells_df['has_significant_distance_remapping'] == True] ## why only endcap cells?
    minorly_changed_endcap_cells_df: pd.DataFrame = non_disappearing_endcap_cells_df[non_disappearing_endcap_cells_df['has_significant_distance_remapping'] == False]
    # significant_distant_remapping_endcap_aclus = non_disappearing_endcap_cells_df[non_disappearing_endcap_cells_df['has_significant_distance_remapping']].index # Int64Index([3, 5, 7, 11, 14, 38, 41, 53, 57, 61, 62, 75, 78, 79, 82, 83, 85, 95, 98, 100, 102], dtype='int64')

    return neuron_replay_stats_df, (non_disappearing_endcap_cells_df, disappearing_endcap_cells_df, minorly_changed_endcap_cells_df, significant_distant_remapping_endcap_cells_df,)


# ==================================================================================================================== #
# Older                                                                                                                #
# ==================================================================================================================== #

# ==================================================================================================================== #
# 2024-04-05 - Back to the laps                                                                                        #
# ==================================================================================================================== #

# ==================================================================================================================== #
# 2024-01-17 - Lap performance validation                                                                              #
# ==================================================================================================================== #
from neuropy.analyses.placefields import PfND
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol
from neuropy.utils.mixins.binning_helpers import find_minimum_time_bin_duration
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult, _check_result_laps_epochs_df_performance
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import CompleteDecodedContextCorrectness

@function_attributes(short_name=None, tags=['ground-truth', 'laps', 'performance'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-15 14:00', related_items=[])
def add_groundtruth_information(curr_active_pipeline, a_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult):
    """    takes 'laps_df' and 'result_laps_epochs_df' to add the ground_truth and the decoded posteriors:

        a_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = alt_directional_merged_decoders_result


        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import add_groundtruth_information

        result_laps_epochs_df = add_groundtruth_information(curr_active_pipeline, a_directional_merged_decoders_result=a_directional_merged_decoders_result, result_laps_epochs_df=result_laps_epochs_df)


    """
    from neuropy.core import Laps

    ## Inputs: a_directional_merged_decoders_result, laps_df

    ## Get the most likely direction/track from the decoded posteriors:
    laps_directional_marginals, laps_directional_all_epoch_bins_marginal, laps_most_likely_direction_from_decoder, laps_is_most_likely_direction_LR_dir = a_directional_merged_decoders_result.laps_directional_marginals_tuple
    laps_track_identity_marginals, laps_track_identity_all_epoch_bins_marginal, laps_most_likely_track_identity_from_decoder, laps_is_most_likely_track_identity_Long = a_directional_merged_decoders_result.laps_track_identity_marginals_tuple

    result_laps_epochs_df: pd.DataFrame = a_directional_merged_decoders_result.laps_epochs_df

    # Ensure it has the 'lap_track' column
    ## Compute the ground-truth information using the position information:
    # adds columns: ['maze_id', 'is_LR_dir']
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    laps_obj: Laps = curr_active_pipeline.sess.laps
    laps_df = laps_obj.to_dataframe()
    laps_df: pd.DataFrame = Laps._update_dataframe_computed_vars(laps_df=laps_df, t_start=t_start, t_delta=t_delta, t_end=t_end, global_session=curr_active_pipeline.sess) # NOTE: .sess is used because global_session is missing the last two laps

    ## 2024-01-17 - Updates the `a_directional_merged_decoders_result.laps_epochs_df` with both the ground-truth values and the decoded predictions
    result_laps_epochs_df['maze_id'] = laps_df['maze_id'].to_numpy()[np.isin(laps_df['lap_id'], result_laps_epochs_df['lap_id'])] # this works despite the different size because of the index matching
    ## add the 'is_LR_dir' groud-truth column in:
    result_laps_epochs_df['is_LR_dir'] = laps_df['is_LR_dir'].to_numpy()[np.isin(laps_df['lap_id'], result_laps_epochs_df['lap_id'])] # this works despite the different size because of the index matching

    ## Add the decoded results to the laps df:
    result_laps_epochs_df['is_most_likely_track_identity_Long'] = laps_is_most_likely_track_identity_Long
    result_laps_epochs_df['is_most_likely_direction_LR'] = laps_is_most_likely_direction_LR_dir


    ## Update the source:
    a_directional_merged_decoders_result.laps_epochs_df = result_laps_epochs_df

    return result_laps_epochs_df


@function_attributes(short_name=None, tags=['laps', 'groundtruth', 'context-decoding', 'context-discrimination'], input_requires=[], output_provides=[], uses=[], used_by=['perform_sweep_lap_groud_truth_performance_testing'], creation_date='2024-04-05 18:40', related_items=['DirectionalPseudo2DDecodersResult'])
def _perform_variable_time_bin_lap_groud_truth_performance_testing(owning_pipeline_reference,
                                                                    desired_laps_decoding_time_bin_size: float = 0.5, desired_ripple_decoding_time_bin_size: Optional[float] = None, use_single_time_bin_per_epoch: bool=False,
                                                                    included_neuron_ids: Optional[NDArray]=None) -> Tuple[DirectionalPseudo2DDecodersResult, pd.DataFrame, CompleteDecodedContextCorrectness]:
    """ 2024-01-17 - Pending refactor from ReviewOfWork_2024-01-17.ipynb

    Makes a copy of the 'DirectionalMergedDecoders' result and does the complete process of re-calculation for the provided time bin sizes. Finally computes the statistics about correctly computed contexts from the laps.

    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _perform_variable_time_bin_lap_groud_truth_performance_testing

        a_directional_merged_decoders_result, result_laps_epochs_df, complete_decoded_context_correctness_tuple = _perform_variable_time_bin_lap_groud_truth_performance_testing(curr_active_pipeline, desired_laps_decoding_time_bin_size=1.5, desired_ripple_decoding_time_bin_size=None)
        (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = complete_decoded_context_correctness_tuple

        ## Filtering with included_neuron_ids:
        a_directional_merged_decoders_result, result_laps_epochs_df, complete_decoded_context_correctness_tuple = _perform_variable_time_bin_lap_groud_truth_performance_testing(curr_active_pipeline, desired_laps_decoding_time_bin_size=1.5, included_neuron_ids=included_neuron_ids)


    """
    from neuropy.utils.indexing_helpers import paired_incremental_sorting, union_of_arrays, intersection_of_arrays, find_desired_sort_indicies
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import PfND

    ## Copy the default result:
    directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = owning_pipeline_reference.global_computation_results.computed_data['DirectionalMergedDecoders']
    alt_directional_merged_decoders_result: DirectionalPseudo2DDecodersResult = deepcopy(directional_merged_decoders_result)

    if included_neuron_ids is not None:
        # prune the decoder by the provided `included_neuron_ids`

        ## Due to a limitation of the merged pseudo2D decoder (`alt_directional_merged_decoders_result.all_directional_pf1D_Decoder`) that prevents `.get_by_id(...)` from working, we have to rebuild the pseudo2D decoder from the four pf1D decoders:
        restricted_all_directional_decoder_pf1D_dict: Dict[str, BasePositionDecoder] = deepcopy(alt_directional_merged_decoders_result.all_directional_decoder_dict) # copy the dictionary
        ## Filter the dictionary using .get_by_id(...)
        restricted_all_directional_decoder_pf1D_dict = {k:v.get_by_id(included_neuron_ids) for k,v in restricted_all_directional_decoder_pf1D_dict.items()}

        restricted_all_directional_pf1D = PfND.build_merged_directional_placefields(restricted_all_directional_decoder_pf1D_dict, debug_print=False)
        restricted_all_directional_pf1D_Decoder = BasePositionDecoder(restricted_all_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)

        # included_neuron_ids = intersection_of_arrays(alt_directional_merged_decoders_result.all_directional_pf1D_Decoder.neuron_IDs, included_neuron_ids)
        # is_aclu_included_list = np.isin(alt_directional_merged_decoders_result.all_directional_pf1D_Decoder.pf.ratemap.neuron_ids, included_neuron_ids)
        # included_aclus = np.array(alt_directional_merged_decoders_result.all_directional_pf1D_Decoder.pf.ratemap.neuron_ids)[is_aclu_included_list
        # modified_decoder = alt_directional_merged_decoders_result.all_directional_pf1D_Decoder.get_by_id(included_aclus)

        ## Set the result:
        alt_directional_merged_decoders_result.all_directional_pf1D_Decoder = restricted_all_directional_pf1D_Decoder

        # ratemap method:
        # modified_decoder_pf_ratemap = alt_directional_merged_decoders_result.all_directional_pf1D_Decoder.pf.ratemap.get_by_id(included_aclus)
        # alt_directional_merged_decoders_result.all_directional_pf1D_Decoder.pf.ratemap = modified_decoder_pf_ratemap

        # alt_directional_merged_decoders_result.all_directional_pf1D_Decoder = alt_directional_merged_decoders_result.all_directional_pf1D_Decoder.get_by_id(included_neuron_ids, defer_compute_all=True)

    all_directional_pf1D_Decoder = alt_directional_merged_decoders_result.all_directional_pf1D_Decoder

    # Modifies alt_directional_merged_decoders_result, a copy of the original result, with new timebins
    long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
    # t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()

    if use_single_time_bin_per_epoch:
        print(f'WARNING: use_single_time_bin_per_epoch=True so time bin sizes will be ignored.')

    ## Decode Laps:
    global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs) # global_epoch_name='maze_any' (? same as global_epoch_name?)
    min_possible_laps_time_bin_size: float = find_minimum_time_bin_duration(global_any_laps_epochs_obj.to_dataframe()['duration'].to_numpy())
    laps_decoding_time_bin_size: float = min(desired_laps_decoding_time_bin_size, min_possible_laps_time_bin_size) # 10ms # 0.002
    if use_single_time_bin_per_epoch:
        laps_decoding_time_bin_size = None

    alt_directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result = all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(owning_pipeline_reference.sess.spikes_df), filter_epochs=global_any_laps_epochs_obj, decoding_time_bin_size=laps_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch, debug_print=False)

    ## Decode Ripples:
    if desired_ripple_decoding_time_bin_size is not None:
        global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name].replay))
        min_possible_time_bin_size: float = find_minimum_time_bin_duration(global_replays['duration'].to_numpy())
        ripple_decoding_time_bin_size: float = min(desired_ripple_decoding_time_bin_size, min_possible_time_bin_size) # 10ms # 0.002
        if use_single_time_bin_per_epoch:
            ripple_decoding_time_bin_size = None
        alt_directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result = all_directional_pf1D_Decoder.decode_specific_epochs(deepcopy(owning_pipeline_reference.sess.spikes_df), global_replays, decoding_time_bin_size=ripple_decoding_time_bin_size, use_single_time_bin_per_epoch=use_single_time_bin_per_epoch)

    ## Post Compute Validations:
    alt_directional_merged_decoders_result.perform_compute_marginals()

    result_laps_epochs_df: pd.DataFrame = add_groundtruth_information(owning_pipeline_reference, a_directional_merged_decoders_result=alt_directional_merged_decoders_result)

    laps_decoding_time_bin_size = alt_directional_merged_decoders_result.laps_decoding_time_bin_size
    print(f'laps_decoding_time_bin_size: {laps_decoding_time_bin_size}')

    ## Uses only 'result_laps_epochs_df'
    complete_decoded_context_correctness_tuple = _check_result_laps_epochs_df_performance(result_laps_epochs_df)

    # Unpack like:
    # (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = complete_decoded_context_correctness_tuple

    return alt_directional_merged_decoders_result, result_laps_epochs_df, complete_decoded_context_correctness_tuple


@function_attributes(short_name=None, tags=['laps', 'groundtruith', 'sweep', 'context-decoding', 'context-discrimination'], input_requires=[], output_provides=[], uses=['_perform_variable_time_bin_lap_groud_truth_performance_testing'], used_by=[], creation_date='2024-04-05 22:47', related_items=[])
def perform_sweep_lap_groud_truth_performance_testing(curr_active_pipeline, included_neuron_ids_list: List[NDArray], desired_laps_decoding_time_bin_size:float=1.5):
    """ Sweeps through each `included_neuron_ids` in the provided `included_neuron_ids_list` and calls `_perform_variable_time_bin_lap_groud_truth_performance_testing(...)` to get its laps ground-truth performance.
    Can be used to assess the contributes of each set of cells (exclusive, rate-remapping, etc) to the discrimination decoding performance.

    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import perform_sweep_lap_groud_truth_performance_testing

        desired_laps_decoding_time_bin_size: float = 1.0
        included_neuron_ids_list = [short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset]

        _output_tuples_list = perform_sweep_lap_groud_truth_performance_testing(curr_active_pipeline,
                                                                                included_neuron_ids_list=included_neuron_ids_list,
                                                                                desired_laps_decoding_time_bin_size=desired_laps_decoding_time_bin_size)

        percent_laps_correctness_df: pd.DataFrame = pd.DataFrame.from_records([complete_decoded_context_correctness_tuple.percent_correct_tuple for (a_directional_merged_decoders_result, result_laps_epochs_df, complete_decoded_context_correctness_tuple) in _output_tuples_list],
                                columns=("track_ID_correct", "dir_correct", "complete_correct"))
        percent_laps_correctness_df



    Does not modifiy the curr_active_pipeline (pure)


    """
    _output_tuples_list = []
    for included_neuron_ids in included_neuron_ids_list:
        a_lap_ground_truth_performance_testing_tuple = _perform_variable_time_bin_lap_groud_truth_performance_testing(curr_active_pipeline, desired_laps_decoding_time_bin_size=desired_laps_decoding_time_bin_size, included_neuron_ids=included_neuron_ids)
        _output_tuples_list.append(a_lap_ground_truth_performance_testing_tuple)

        # ## Unpacking `a_lap_ground_truth_performance_testing_tuple`:
        # a_directional_merged_decoders_result, result_laps_epochs_df, complete_decoded_context_correctness_tuple = a_lap_ground_truth_performance_testing_tuple
        # (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = complete_decoded_context_correctness_tuple

        # a_directional_merged_decoders_result, result_laps_epochs_df, (is_decoded_track_correct, is_decoded_dir_correct, are_both_decoded_properties_correct), (percent_laps_track_identity_estimated_correctly, percent_laps_direction_estimated_correctly, percent_laps_estimated_correctly) = a_lap_ground_truth_performance_testing_tuple
    return _output_tuples_list


# ---------------------------------------------------------------------------- #
#             2024-03-29 - Rigorous Decoder Performance assessment             #
# ---------------------------------------------------------------------------- #
# Quantify cell contributions to decoders
# Inputs: all_directional_pf1D_Decoder, alt_directional_merged_decoders_result

import portion as P # Required for interval search: portion~=2.3.0
from neuropy.core.epoch import Epoch, ensure_dataframe
from neuropy.utils.efficient_interval_search import convert_PortionInterval_to_epochs_df, _convert_start_end_tuples_list_to_PortionInterval
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsResult, TrackTemplates
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df
from sklearn.metrics import mean_squared_error

## Get custom decoder that is only trained on a portion of the laps
## Build the `BasePositionDecoder` for each of the four templates analagous to what is done in `_long_short_decoding_analysis_from_decoders`:
# long_LR_laps_one_step_decoder_1D, long_RL_laps_one_step_decoder_1D, short_LR_laps_one_step_decoder_1D, short_RL_laps_one_step_decoder_1D  = [BasePositionDecoder.init_from_stateful_decoder(deepcopy(results_data.get('pf1D_Decoder', None))) for results_data in (long_LR_results, long_RL_results, short_LR_results, short_RL_results)]

## Restrict the data post-hoc?

## Time-dependent decoder?

## Split the lap epochs into training and test periods.
##### Ideally we could test the lap decoding error by sampling randomly from the time bins and omitting 1/6 of time bins from the placefield building (effectively the training data). These missing bins will be used as the "test data" and the decoding error will be computed by decoding them and subtracting the actual measured position during these bins.



# ==================================================================================================================== #
# 2024-03-09 - Filtering                                                                                               #
# ==================================================================================================================== #

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult, filter_and_update_epochs_and_spikes
# from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import HeuristicReplayScoring
from neuropy.core.epoch import ensure_dataframe, find_data_indicies_from_epoch_times

@function_attributes(short_name=None, tags=['filtering'], input_requires=[], output_provides=[], uses=[], used_by=['_perform_filter_replay_epochs'], creation_date='2024-04-25 06:38', related_items=[])
def _apply_filtering_to_marginals_result_df(active_result_df: pd.DataFrame, filtered_epochs_df: pd.DataFrame, filtered_decoder_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult]):
    """ after filtering the epochs (for user selections, validity, etc) apply the same filtering to a results df.

    Applied to `filtered_decoder_filter_epochs_decoder_result_dict` to build a dataframe

    """
    ## INPUTS: active_result_df, filtered_epochs_df

    # found_data_indicies = find_data_indicies_from_epoch_times(active_result_df, epoch_times=np.squeeze(any_good_selected_epoch_times[:,0]), t_column_names=['ripple_start_t'], atol=1e-2)
    # found_data_indicies = find_data_indicies_from_epoch_times(active_result_df, epoch_times=np.squeeze(any_good_selected_epoch_times[:,0]), t_column_names=['ripple_start_t'], atol=1e-2)
    found_data_indicies = find_data_indicies_from_epoch_times(active_result_df, epoch_times=np.squeeze(filtered_epochs_df['start'].to_numpy()), t_column_names=['ripple_start_t'], atol=1e-3)

    # ripple_all_epoch_bins_marginals_df.epochs.matching_epoch_times_slice(any_good_selected_epoch_times)

    active_result_df = active_result_df.loc[found_data_indicies].copy().reset_index(drop=True)
    direction_max_indices = active_result_df[['P_LR', 'P_RL']].values.argmax(axis=1)
    track_identity_max_indices = active_result_df[['P_Long', 'P_Short']].values.argmax(axis=1)

    ## INPUTS: filtered_decoder_filter_epochs_decoder_result_dict

    df_column_names = [list(a_df.filter_epochs.columns) for a_df in filtered_decoder_filter_epochs_decoder_result_dict.values()]
    print(f"df_column_names: {df_column_names}")
    selected_df_column_names = ['wcorr', 'pearsonr']

    # merged_dfs_dict = {a_name:a_df.filter_epochs[selected_df_column_names].add_suffix(f"_{a_name}") for a_name, a_df in filtered_decoder_filter_epochs_decoder_result_dict.items()}
    # merged_dfs_dict = pd.concat([a_df.filter_epochs[selected_df_column_names].add_suffix(f"_{a_name}") for a_name, a_df in filtered_decoder_filter_epochs_decoder_result_dict.items()], axis='columns')
    # merged_dfs_dict

    # filtered_decoder_filter_epochs_decoder_result_dict['short_LR'][a_column_name], filtered_decoder_filter_epochs_decoder_result_dict['short_RL'][a_column_name]

    ## BEST/COMPARE OUT DF:
    # active_result_df = deepcopy(active_result_df)

    # Get only the best direction long/short values for each metric:
    for a_column_name in selected_df_column_names: # = 'wcorr'
        assert len(direction_max_indices) == len(filtered_decoder_filter_epochs_decoder_result_dict['long_LR'].filter_epochs)
        active_result_df[f'long_best_{a_column_name}'] = np.where(direction_max_indices, filtered_decoder_filter_epochs_decoder_result_dict['long_LR'].filter_epochs[a_column_name].to_numpy(), filtered_decoder_filter_epochs_decoder_result_dict['long_RL'].filter_epochs[a_column_name].to_numpy())
        active_result_df[f'short_best_{a_column_name}'] = np.where(direction_max_indices, filtered_decoder_filter_epochs_decoder_result_dict['short_LR'].filter_epochs[a_column_name].to_numpy(), filtered_decoder_filter_epochs_decoder_result_dict['short_RL'].filter_epochs[a_column_name].to_numpy())
        active_result_df[f'{a_column_name}_abs_diff'] = active_result_df[f'long_best_{a_column_name}'].abs() - active_result_df[f'short_best_{a_column_name}'].abs()


    ## ['wcorr_abs_diff', 'pearsonr_abs_diff']
    return active_result_df

## INPUTS: decoder_ripple_filter_epochs_decoder_result_dict
@function_attributes(short_name=None, tags=['filter', 'replay', 'IMPORTANT', 'PERFORMANCE', 'SLOW'], input_requires=[], output_provides=[], uses=['_apply_filtering_to_marginals_result_df'], used_by=[], creation_date='2024-04-24 18:03', related_items=[])
def _perform_filter_replay_epochs(curr_active_pipeline, global_epoch_name, track_templates, decoder_ripple_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult], ripple_all_epoch_bins_marginals_df: pd.DataFrame, ripple_decoding_time_bin_size: float,
            should_only_include_user_selected_epochs:bool=True, **additional_selections_context):
    """ the main replay epochs filtering function.

    if should_only_include_user_selected_epochs is True, it only includes user selected (annotated) ripples


    Usage:
        from neuropy.core.epoch import find_data_indicies_from_epoch_times
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _perform_filter_replay_epochs

        filtered_epochs_df, filtered_decoder_filter_epochs_decoder_result_dict, filtered_ripple_all_epoch_bins_marginals_df = _perform_filter_replay_epochs(curr_active_pipeline, global_epoch_name, track_templates, decoder_ripple_filter_epochs_decoder_result_dict, ripple_all_epoch_bins_marginals_df, ripple_decoding_time_bin_size=ripple_decoding_time_bin_size)
        filtered_epochs_df

    """
    from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicReplayScoring, HeuristicScoresTuple

    # 2024-03-04 - Filter out the epochs based on the criteria:
    filtered_epochs_df, active_spikes_df = filter_and_update_epochs_and_spikes(curr_active_pipeline, global_epoch_name, track_templates, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1)
    ## 2024-03-08 - Also constrain the user-selected ones (just to try it):
    decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times = DecoderDecodedEpochsResult.load_user_selected_epoch_times(curr_active_pipeline, track_templates=track_templates, **additional_selections_context)

    ## filter the epochs by something and only show those:
    # INPUTS: filtered_epochs_df
    # filtered_ripple_simple_pf_pearson_merged_df = filtered_ripple_simple_pf_pearson_merged_df.epochs.matching_epoch_times_slice(active_epochs_df[['start', 'stop']].to_numpy())
    ## Update the `decoder_ripple_filter_epochs_decoder_result_dict` with the included epochs:
    filtered_decoder_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = {a_name:a_result.filtered_by_epoch_times(filtered_epochs_df[['start', 'stop']].to_numpy()) for a_name, a_result in decoder_ripple_filter_epochs_decoder_result_dict.items()} # working filtered
    # print(f"any_good_selected_epoch_times.shape: {any_good_selected_epoch_times.shape}") # (142, 2)
    ## Constrain again now by the user selections
    if should_only_include_user_selected_epochs:
        filtered_decoder_filter_epochs_decoder_result_dict: Dict[str, DecodedFilterEpochsResult] = {a_name:a_result.filtered_by_epoch_times(any_good_selected_epoch_times) for a_name, a_result in filtered_decoder_filter_epochs_decoder_result_dict.items()}
    # filtered_decoder_filter_epochs_decoder_result_dict

    # 🟪 2024-02-29 - `compute_pho_heuristic_replay_scores`
    filtered_decoder_filter_epochs_decoder_result_dict, _out_new_scores, partition_result_dict = HeuristicReplayScoring.compute_all_heuristic_scores(track_templates=track_templates, a_decoded_filter_epochs_decoder_result_dict=filtered_decoder_filter_epochs_decoder_result_dict)

    if should_only_include_user_selected_epochs:
        filtered_epochs_df = filtered_epochs_df.epochs.matching_epoch_times_slice(any_good_selected_epoch_times)

    ## OUT: filtered_decoder_filter_epochs_decoder_result_dict, filtered_epochs_df

    # `ripple_all_epoch_bins_marginals_df`
    filtered_ripple_all_epoch_bins_marginals_df = deepcopy(ripple_all_epoch_bins_marginals_df)
    filtered_ripple_all_epoch_bins_marginals_df = _apply_filtering_to_marginals_result_df(filtered_ripple_all_epoch_bins_marginals_df, filtered_epochs_df=filtered_epochs_df, filtered_decoder_filter_epochs_decoder_result_dict=filtered_decoder_filter_epochs_decoder_result_dict)
    assert len(filtered_epochs_df) == len(filtered_ripple_all_epoch_bins_marginals_df), f"len(filtered_epochs_df): {len(filtered_epochs_df)} != len(active_result_df): {len(filtered_ripple_all_epoch_bins_marginals_df)}"

    df = filtered_ripple_all_epoch_bins_marginals_df

    # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
    session_name: str = curr_active_pipeline.session_name
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    df = DecoderDecodedEpochsResult.add_session_df_columns(df, session_name=session_name, time_bin_size=None, t_start=t_start, curr_session_t_delta=t_delta, t_end=t_end, time_col='ripple_start_t')
    df["time_bin_size"] = ripple_decoding_time_bin_size
    df['is_user_annotated_epoch'] = True # if it's filtered here, it's true
    return filtered_epochs_df, filtered_decoder_filter_epochs_decoder_result_dict, filtered_ripple_all_epoch_bins_marginals_df


@function_attributes(short_name=None, tags=['filter', 'epoch_selection', 'export', 'h5'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-03-08 13:28', related_items=[])
def export_numpy_testing_filtered_epochs(curr_active_pipeline, global_epoch_name, track_templates, required_min_percentage_of_active_cells: float = 0.333333, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1, **additional_selections_context):
    """ Save testing variables to file 'NeuroPy/tests/neuropy_pf_testing.h5'
    exports: original_epochs_df, filtered_epochs_df, active_spikes_df

    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import export_numpy_testing_filtered_epochs

    finalized_output_cache_file = export_numpy_testing_filtered_epochs(curr_active_pipeline, global_epoch_name, track_templates, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1)
    finalized_output_cache_file

    """
    from neuropy.core import Epoch
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import filter_and_update_epochs_and_spikes

    global_replays = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].replay)
    if isinstance(global_replays, pd.DataFrame):
        global_replays = Epoch(global_replays.epochs.get_valid_df())
    original_spikes_df = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].spikes_df)
    original_spikes_df = original_spikes_df.spikes.sliced_by_neuron_id(track_templates.any_decoder_neuron_IDs)
    # Start Filtering
    original_epochs_df = deepcopy(global_replays.to_dataframe())

    # 2024-03-04 - Filter out the epochs based on the criteria:
    filtered_epochs_df, filtered_spikes_df = filter_and_update_epochs_and_spikes(curr_active_pipeline, global_epoch_name, track_templates, epoch_id_key_name='ripple_epoch_id', no_interval_fill_value=-1)


    decoder_user_selected_epoch_times_dict, any_good_selected_epoch_times = DecoderDecodedEpochsResult.load_user_selected_epoch_times(curr_active_pipeline, track_templates=track_templates, **additional_selections_context)
    print(f"any_good_selected_epoch_times.shape: {any_good_selected_epoch_times.shape}") # (142, 2)

    ## Save for NeuroPy testing:
    finalized_output_cache_file='../NeuroPy/tests/neuropy_epochs_testing.h5'
    sess_identifier_key='sess'
    original_epochs_df.to_hdf(finalized_output_cache_file, key=f'{sess_identifier_key}/original_epochs_df', format='table')
    filtered_epochs_df.to_hdf(finalized_output_cache_file, key=f'{sess_identifier_key}/filtered_epochs_df', format='table')
    # selected_epochs_df.to_hdf(finalized_output_cache_file, key=f'{sess_identifier_key}/selected_epochs_df', format='table')
    any_good_selected_epoch_times_df = pd.DataFrame(any_good_selected_epoch_times, columns=['start', 'stop'])
    any_good_selected_epoch_times_df.to_hdf(finalized_output_cache_file, key=f'{sess_identifier_key}/any_good_selected_epoch_times_df', format='table')


    # basic_epoch_column_names = ['start', 'stop', 'label', 'duration', 'ripple_idx', 'P_Long']
    # test_df: pd.DataFrame = deepcopy(ripple_simple_pf_pearson_merged_df[basic_epoch_column_names])
    # test_df.to_hdf(finalized_output_cache_file, key=f'{sess_identifier_key}/test_df', format='table')
    return finalized_output_cache_file


# ==================================================================================================================== #
# Old type display helpers                                                                                             #
# ==================================================================================================================== #

def register_type_display(func_to_register, type_to_register):
    """ adds the display function (`func_to_register`) it decorates to the class (`type_to_register) as a method


    """
    @wraps(func_to_register)
    def wrapper(*args, **kwargs):
        return func_to_register(*args, **kwargs)

    function_name: str = func_to_register.__name__ # get the name of the function to be added as the property
    setattr(type_to_register, function_name, wrapper) # set the function as a method with the same name as the decorated function on objects of the class.
    return wrapper



# ==================================================================================================================== #
# 2024-02-15 - Radon Transform / Weighted Correlation, etc helpers                                                     #
# ==================================================================================================================== #


# ==================================================================================================================== #
# 2024-02-08 - Plot Single ACLU Heatmaps for Each Decoder                                                              #
# ==================================================================================================================== #
from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle

@function_attributes(short_name=None, tags=['plot', 'heatmap', 'peak'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-02-08 00:00', related_items=[])
def plot_peak_heatmap_test(curr_aclu_z_scored_tuning_map_matrix_dict, xbin, point_dict=None, ax_dict=None, extra_decoder_values_dict=None, tuning_curves_dict=None, include_tuning_curves=False):
    """ 2024-02-06 - Plots the four position-binned-activity maps (for each directional decoding epoch) as a 4x4 subplot grid using matplotlib.

    """
    from pyphoplacecellanalysis.Pho2D.matplotlib.visualize_heatmap import visualize_heatmap
    if tuning_curves_dict is None:
        assert include_tuning_curves == False

    # figure_kwargs = dict(layout="tight")
    figure_kwargs = dict(layout="none")

    if ax_dict is None:
        if not include_tuning_curves:
            # fig = plt.figure(layout="constrained", figsize=[9, 7], dpi=220, clear=True) # figsize=[Width, height] in inches.
            fig = plt.figure(figsize=[8, 7], dpi=220, clear=True, **figure_kwargs)
            long_width_ratio = 1
            ax_dict = fig.subplot_mosaic(
                [
                    ["ax_long_LR", "ax_long_RL"],
                    ["ax_short_LR", "ax_short_RL"],
                ],
                # set the height ratios between the rows
                # set the width ratios between the columns
                width_ratios=[long_width_ratio, long_width_ratio],
                sharex=True, sharey=False,
                gridspec_kw=dict(wspace=0.027, hspace=0.112) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
            )
        else:
            # tuning curves mode:
            fig = plt.figure(figsize=[9, 7], dpi=220, clear=True, **figure_kwargs)
            long_width_ratio = 1
            ax_dict = fig.subplot_mosaic(
                [
                    ["ax_long_LR_curve", "ax_long_RL_curve"],
                    ["ax_long_LR", "ax_long_RL"],
                    ["ax_short_LR", "ax_short_RL"],
                    ["ax_short_LR_curve", "ax_short_RL_curve"],
                ],
                # set the height ratios between the rows
                # set the width ratios between the columns
                width_ratios=[long_width_ratio, long_width_ratio],
                height_ratios=[1, 7, 7, 1], # tuning curves are smaller than laps
                sharex=True, sharey=False,
                gridspec_kw=dict(wspace=0.027, hspace=0.112) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
            )
            curve_ax_names = ["ax_long_LR_curve", "ax_long_RL_curve", "ax_short_LR_curve", "ax_short_RL_curve"]

    else:
        if not include_tuning_curves:
            # figure already exists, reuse the axes
            assert len(ax_dict) == 4
            assert list(ax_dict.keys()) == ["ax_long_LR", "ax_long_RL", "ax_short_LR", "ax_short_RL"]
        else:
            # tuning curves mode:
            assert len(ax_dict) == 8
            assert list(ax_dict.keys()) == ["ax_long_LR_curve", "ax_long_RL_curve", "ax_long_LR", "ax_long_RL", "ax_short_LR", "ax_short_RL", "ax_short_LR_curve", "ax_short_RL_curve"]



    # Get the colormap to use and set the bad color
    cmap = mpl.colormaps.get_cmap('viridis')  # viridis is the default colormap for imshow
    cmap.set_bad(color='black')

    # Compute extents for imshow:
    imshow_kwargs = {
        'origin': 'lower',
        # 'vmin': 0,
        # 'vmax': 1,
        'cmap': cmap,
        'interpolation':'nearest',
        'aspect':'auto',
        'animated':True,
        'show_xticks':False,
    }

    _old_data_to_ax_mapping = dict(zip(['maze1_odd', 'maze1_even', 'maze2_odd', 'maze2_even'], ["ax_long_LR", "ax_long_RL", "ax_short_LR", "ax_short_RL"]))
    data_to_ax_mapping = dict(zip(['long_LR', 'long_RL', 'short_LR', 'short_RL'], ["ax_long_LR", "ax_long_RL", "ax_short_LR", "ax_short_RL"]))

    # ['long_LR', 'long_RL', 'short_LR', 'short_RL']


    for k, v in curr_aclu_z_scored_tuning_map_matrix_dict.items():
        # is_first_item = (k == list(curr_aclu_z_scored_tuning_map_matrix_dict.keys())[0])
        is_last_item = (k == list(curr_aclu_z_scored_tuning_map_matrix_dict.keys())[-1])

        curr_ax = ax_dict[data_to_ax_mapping[k]]
        curr_ax.clear()

        # hist_data = np.random.randn(1_500)
        # xbin_centers = np.arange(len(hist_data))+0.5
        # ax_dict["ax_LONG_pf_tuning_curve"] = plot_placefield_tuning_curve(xbin_centers, (-1.0 * curr_cell_normalized_tuning_curve), ax_dict["ax_LONG_pf_tuning_curve"], is_horizontal=True)

        n_epochs:int = np.shape(v)[1]
        epoch_indicies = np.arange(n_epochs)

        # Posterior distribution heatmaps at each point.
        xmin, xmax, ymin, ymax = (xbin[0], xbin[-1], epoch_indicies[0], epoch_indicies[-1])
        imshow_kwargs['extent'] = (xmin, xmax, ymin, ymax)

        # plot heatmap:
        curr_ax.set_xticklabels([])
        curr_ax.set_yticklabels([])
        fig, ax, im = visualize_heatmap(v.copy(), ax=curr_ax, title=f'{k}', layout='none', defer_show=True, **imshow_kwargs) # defer_show so it doesn't produce a separate figure for each!
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))


        if include_tuning_curves:
            tuning_curve = tuning_curves_dict[k]
            curr_curve_ax = ax_dict[f"{data_to_ax_mapping[k]}_curve"]
            curr_curve_ax.clear()

            if tuning_curve is not None:
                # plot curve heatmap:
                if not is_last_item:
                    curr_curve_ax.set_xticklabels([])
                    # Leave the position x-ticks on for the last item

                curr_curve_ax.set_yticklabels([])
                ymin, ymax = 0, 1
                imshow_kwargs['extent'] = (xmin, xmax, 0, 1)
                fig, curr_curve_ax, im = visualize_heatmap(tuning_curve.copy(), ax=curr_curve_ax, title=f'{k}', defer_show=True, **imshow_kwargs) # defer_show so it doesn't produce a separate figure for each!
                curr_curve_ax.set_xlim((xmin, xmax))
                curr_curve_ax.set_ylim((0, 1))

            point_ax = curr_curve_ax # draw the lines on the tuning curve axis

        else:
            point_ax = ax

        if point_dict is not None:
            if k in point_dict:
                # have points to plot
                point_ax.vlines(point_dict[k], ymin=ymin, ymax=ymax, colors='r', label=f'{k}_peak')


    # fig.tight_layout()
    # NOTE: these layout changes don't seem to take effect until the window containing the figure is resized.
    # fig.set_layout_engine('compressed') # TAKEWAY: Use 'compressed' instead of 'constrained'
    fig.set_layout_engine('none') # disabling layout engine. Strangely still allows window to resize and the plots scale, so I'm not sure what the layout engine is doing.


    # ax_dict["ax_SHORT_activity_v_time"].plot([1, 2, 3, 3, 3, 2, 1, 0, 0, 0, 1, 2, 3, 3, 1, 2, 0, 0])
    # ax_dict["ax_SHORT_pf_tuning_curve"] = plot_placefield_tuning_curve(xbin_centers, curr_cell_normalized_tuning_curve, ax_dict["ax_SHORT_pf_tuning_curve"], is_horizontal=True)
    # ax_dict["ax_SHORT_pf_tuning_curve"].set_xticklabels([])
    # ax_dict["ax_SHORT_pf_tuning_curve"].set_yticklabels([])
    # ax_dict["ax_SHORT_pf_tuning_curve"].set_box

    return fig, ax_dict

# INPUTS: directional_active_lap_pf_results_dicts, test_aclu: int = 26, xbin_centers, decoder_aclu_peak_location_df_merged

def plot_single_heatmap_set_with_points(directional_active_lap_pf_results_dicts, xbin_centers, xbin, decoder_aclu_peak_location_df_merged: pd.DataFrame, aclu: int = 26, **kwargs):
    """ 2024-02-06 - Plot all four decoders for a single aclu, with overlayed red lines for the detected peaks.

    plot_single_heatmap_set_with_points

    plot_cell_position_binned_activity_over_time

    Usage:

        decoders_tuning_curves_dict = track_templates.decoder_normalized_tuning_curves_dict_dict.copy()

        extra_decoder_values_dict = {'tuning_curves': decoders_tuning_curves_dict, 'points': decoder_aclu_peak_location_df_merged}

        # decoders_tuning_curves_dict
        xbin_centers = deepcopy(active_pf_dt.xbin_centers)
        xbin = deepcopy(active_pf_dt.xbin)
        fig, ax_dict = plot_single_heatmap_set_with_points(directional_active_lap_pf_results_dicts, xbin_centers, xbin, extra_decoder_values_dict=extra_decoder_values_dict, aclu=4,
                                                        decoders_tuning_curves_dict=decoders_tuning_curves_dict, decoder_aclu_peak_location_df_merged=decoder_aclu_peak_location_df_merged,
                                                            active_context=curr_active_pipeline.build_display_context_for_session('single_heatmap_set_with_points'))

    """
    from neuropy.utils.result_context import IdentifyingContext

    ## TEst: Look at a single aclu value
    # test_aclu: int = 26
    # test_aclu: int = 28

    active_context: IdentifyingContext = kwargs.get('active_context', IdentifyingContext())
    active_context = active_context.overwriting_context(aclu=aclu)

    decoders_tuning_curves_dict = kwargs.get('decoders_tuning_curves_dict', None)

    matching_aclu_df = decoder_aclu_peak_location_df_merged[decoder_aclu_peak_location_df_merged.aclu == aclu].copy()
    assert len(matching_aclu_df) > 0, f"matching_aclu_df: {matching_aclu_df} for aclu == {aclu}"
    new_peaks_dict: Dict = list(matching_aclu_df.itertuples(index=False))[0]._asdict() # {'aclu': 28, 'long_LR_peak': 185.29063638457257, 'long_RL_peak': nan, 'short_LR_peak': 176.75276643746625, 'short_RL_peak': nan, 'LR_peak_diff': 8.537869947106316, 'RL_peak_diff': nan}

    # long_LR_name, long_RL_name, short_LR_name, short_RL_name
    curr_aclu_z_scored_tuning_map_matrix_dict = {}
    curr_aclu_mean_epoch_peak_location_dict = {}
    curr_aclu_median_peak_location_dict = {}
    curr_aclu_extracted_decoder_peak_locations_dict = {}

    ## Find the peak location for each epoch:
    for a_name, a_decoder_directional_active_lap_pf_result in directional_active_lap_pf_results_dicts.items():
        # print(f'a_name: {a_name}')
        matrix_idx = a_decoder_directional_active_lap_pf_result.aclu_to_matrix_IDX_map[aclu]
        curr_aclu_z_scored_tuning_map_matrix = a_decoder_directional_active_lap_pf_result.z_scored_tuning_map_matrix[:,matrix_idx,:] # .shape (22, 80, 56)
        curr_aclu_z_scored_tuning_map_matrix_dict[a_name] = curr_aclu_z_scored_tuning_map_matrix

        # curr_aclu_mean_epoch_peak_location_dict[a_name] = np.nanmax(curr_aclu_z_scored_tuning_map_matrix, axis=-1)
        assert np.shape(curr_aclu_z_scored_tuning_map_matrix)[-1] == len(xbin_centers), f"np.shape(curr_aclu_z_scored_tuning_map_matrix)[-1]: {np.shape(curr_aclu_z_scored_tuning_map_matrix)} != len(xbin_centers): {len(xbin_centers)}"
        curr_peak_value = new_peaks_dict[f'{a_name}_peak']
        # print(f'curr_peak_value: {curr_peak_value}')
        curr_aclu_extracted_decoder_peak_locations_dict[a_name] = curr_peak_value

        curr_aclu_mean_epoch_peak_location_dict[a_name] = np.nanargmax(curr_aclu_z_scored_tuning_map_matrix, axis=-1)
        curr_aclu_mean_epoch_peak_location_dict[a_name] = xbin_centers[curr_aclu_mean_epoch_peak_location_dict[a_name]] # convert to actual positions instead of indicies
        curr_aclu_median_peak_location_dict[a_name] = np.nanmedian(curr_aclu_mean_epoch_peak_location_dict[a_name])

    # curr_aclu_mean_epoch_peak_location_dict # {'maze1_odd': array([ 0, 55, 54, 55, 55, 53, 50, 55, 52, 52, 55, 53, 53, 52, 51, 52, 55, 55, 53, 55, 55, 54], dtype=int64), 'maze2_odd': array([46, 45, 43, 46, 45, 46, 46, 46, 45, 45, 44, 46, 44, 45, 46, 45, 44, 44, 45, 45], dtype=int64)}


    if decoders_tuning_curves_dict is not None:
        curr_aclu_tuning_curves_dict = {name:v.get(aclu, None) for name, v in decoders_tuning_curves_dict.items()}
    else:
        curr_aclu_tuning_curves_dict = None

    # point_value = curr_aclu_median_peak_location_dict
    point_value = curr_aclu_extracted_decoder_peak_locations_dict
    fig, ax_dict = plot_peak_heatmap_test(curr_aclu_z_scored_tuning_map_matrix_dict, xbin=xbin, point_dict=point_value, tuning_curves_dict=curr_aclu_tuning_curves_dict, include_tuning_curves=True)
    # Set window title and plot title
    perform_update_title_subtitle(fig=fig, ax=None, title_string=f"Position-Binned Activity per Lap - aclu {aclu}", subtitle_string=None, active_context=active_context, use_flexitext_titles=True)

    # fig, ax_dict = plot_peak_heatmap_test(curr_aclu_z_scored_tuning_map_matrix_dict, xbin=xbin, point_dict=curr_aclu_extracted_decoder_peak_locations_dict) # , defer_show=True

    # fig.show()
    return fig, ax_dict


# ==================================================================================================================== #
# Usability/Conveninece Helpers                                                                                        #
# ==================================================================================================================== #

