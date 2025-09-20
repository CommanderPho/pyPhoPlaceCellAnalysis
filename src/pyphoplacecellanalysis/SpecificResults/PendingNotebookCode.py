from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    ## typehinting only imports here
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster

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
from neuropy.plotting.placemaps import perform_plot_occupancy
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.indexing_helpers import union_of_arrays
from neuropy.utils.result_context import IdentifyingContext
import nptyping as ND
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
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult, SingleEpochDecodedResult

from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing import NewType
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
import neuropy.utils.type_aliases as types
decoder_name: TypeAlias = str # a string that describes a decoder, such as 'LongLR' or 'ShortRL'
epoch_split_key: TypeAlias = str # a string that describes a split epoch, such as 'train' or 'test'
DecoderName = NewType('DecoderName', str)

import matplotlib.pyplot as plt
import pyphoplacecellanalysis.External.pyqtgraph as pg

from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity

from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr, shape_only_repr
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

from benedict import benedict
from neuropy.utils.mixins.indexing_helpers import get_dict_subset
from neuropy.utils.indexing_helpers import flatten_dict
from attrs import define, field, Factory, asdict # used for `ComputedResult`

from neuropy.utils.indexing_helpers import get_values_from_keypaths, set_value_by_keypath, update_nested_dict

from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWidget
from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import SynchronizedPlotMode
from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget import PyqtgraphTimeSynchronizedWidget


import numpy as np
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtWidgets, QtCore

from copy import deepcopy
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path

from pyphoplacecellanalysis.External.pyqtgraph import QtCore, QtGui, QtWidgets
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import mkQApp

from neuropy.utils.result_context import IdentifyingContext
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.Filesystem.path_helpers import sanitize_filename_for_Windows
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import LongShortDisplayConfigManager


from neuropy.utils.mixins.binning_helpers import BinningContainer, BinningInfo
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

import numpy as np
from pyphoplacecellanalysis.Analysis.Decoder.transition_matrix import TransitionMatrixComputations
from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult
from neuropy.utils.mixins.binning_helpers import BinningContainer, BinningInfo
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import LongShortTrackDataframeAccessor
from typing import Dict, List, Tuple, Optional, Callable, Union, Any, TypeVar
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray

# import neuropy.utils.type_aliases as types
import pyphoplacecellanalysis.General.type_aliases as types
from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol
from pyphocorehelpers.assertion_helpers import Assert

from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DecoderIdentityColors, long_short_display_config_manager, apply_LR_to_RL_adjustment
from pyphocorehelpers.gui.Qt.color_helpers import ColormapHelpers, ColorFormatConverter, debug_print_color, build_adjusted_color


# ==================================================================================================================================================================================================================================================================================== #
# 2025-09-05 - Bapun Co/plotting                                                                                                                                                                                                                                                       #
# ==================================================================================================================================================================================================================================================================================== #
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool import GenericPyQtGraphContainer
from neuropy.analyses.placefields import PfND
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, BayesianPlacemapPositionDecoder, DecodedFilterEpochsResult, Zhang_Two_Step
from neuropy.core.epoch import Epoch, ensure_dataframe, ensure_Epoch, EpochsAccessor
from neuropy.analyses.placefields import Position
from neuropy.core.session.SessionSelectionAndFiltering import build_custom_epochs_filters
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder, DataSessionFormatBaseRegisteredClass
from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass
from neuropy.core.epoch import Epoch, EpochsAccessor, ensure_dataframe, ensure_Epoch


@function_attributes(short_name=None, tags=['bapun'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-09-19 17:50', related_items=[])
def final_process_bapun_all_comps(curr_active_pipeline, posthoc_save: bool=True, override_parameters_flat_keypaths_dict=None):
    """ 
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import final_process_bapun_all_comps
    curr_active_pipeline = final_process_bapun_all_comps(curr_active_pipeline=curr_active_pipeline, posthoc_save=True)
    
    """
    
    from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline
    from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_extended_computations


    active_data_mode_name = 'bapun'
    # active_data_mode_name = 'rachel'

    known_data_session_type_properties_dict = DataSessionFormatRegistryHolder.get_registry_known_data_session_type_dict(override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict)
    active_data_session_types_registered_classes_dict = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()


    print(f'active_data_session_types_registered_classes_dict: {active_data_session_types_registered_classes_dict}')
    active_data_mode_registered_class = active_data_session_types_registered_classes_dict[active_data_mode_name]
    active_data_mode_type_properties = known_data_session_type_properties_dict[active_data_mode_name]

    # basedir = Path('/media/halechr/MAX/Data/Rachel/Cho_241117_Session2').resolve()
    ## INPUTS: basedir 


    session_epochs: Epoch = BapunDataSessionFormatRegisteredClass.session_fixup_epochs(sess=curr_active_pipeline.sess)
    session_epochs

    curr_epoch_names: List[str] = curr_active_pipeline.sess.epochs.to_dataframe()['label'].to_list()
    print(f'curr_epoch_names: {curr_epoch_names}')

    # epoch_name_includelist = ['pre', 'maze1', 'post1', 'maze2', 'post2']
    # epoch_name_includelist = ['pre', 'roam', 'sprinkle', 'post']
    # epoch_name_includelist = ['roam', 'sprinkle']

    # active_session_filter_configurations = build_custom_epochs_filters(curr_active_pipeline.sess, epoch_name_includelist=['pre', 'maze1', 'post1', 'maze2', 'post2']) ## ALL possible epochs

    # active_session_filter_configurations = build_custom_epochs_filters(curr_active_pipeline.sess, epoch_name_includelist=['maze1', 'maze2', 'maze_GLOBAL']) ## ALL possible epochs
    active_session_filter_configurations = build_custom_epochs_filters(curr_active_pipeline.sess, epoch_name_includelist=['maze1', 'maze2', 'maze_GLOBAL']) ## ALL possible epochs

    # active_session_filter_configurations = active_data_mode_registered_class.build_default_filter_functions(sess=curr_active_pipeline.sess)
    # active_session_filter_configurations = build_custom_epochs_filters(curr_active_pipeline.sess, epoch_name_includelist=['pre', 'roam', 'maze', 'sprinkle', 'post']) ## ALL possible epochs
    # active_session_filter_configurations = build_custom_epochs_filters(curr_active_pipeline.sess, epoch_name_includelist=['pre', 'roam', 'sprinkle', 'post']) ## ALL possible epochs


    # active_session_filter_configurations = active_data_mode_registered_class.build_default_filter_functions(sess=curr_active_pipeline.sess, epoch_name_includelist=epoch_name_includelist) # build_filters_pyramidal_epochs(sess=curr_kdiba_pipeline.sess)
    # active_session_filter_configurations = build_custom_epochs_filters(curr_active_pipeline.sess, epoch_name_includelist=['maze','sprinkle'])
    # active_session_filter_configurations = build_custom_epochs_filters(curr_active_pipeline.sess, epoch_name_includelist=['maze', 'sprinkle'])
    # active_session_filter_configurations = build_custom_epochs_filters(curr_active_pipeline.sess, epoch_name_includelist=['roam', 'sprinkle']) # , 'maze'

    # active_session_filter_configurations = active_data_mode_registered_class.build_filters_pyramidal_epochs(curr_active_pipeline.sess, epoch_name_includelist=['maze','sprinkle'])
    # active_session_filter_configurations

    curr_active_pipeline.filter_sessions(active_session_filter_configurations)

    # ==================================================================================================================================================================================================================================================================================== #
    # COMPUTATION CONFIGS                                                                                                                                                                                                                                                                  #
    # ==================================================================================================================================================================================================================================================================================== #
    active_session_computation_configs = active_data_mode_registered_class.build_default_computation_configs(sess=curr_active_pipeline.sess, time_bin_size=0.5)

    # grid_bin_bounds=(((-83.33747881216672, 110.15967332926644), (-94.89955475226206, 97.07387994733473)))


    bapun_open_field_grid_bin_bounds = (((-120.0, 120.0), (-120.0, 120.0)))
    curr_active_pipeline.get_all_parameters()
    # curr_active_pipeline.update_parameters(grid_bin_bounds = (((-120.0, 120.0), (-120.0, 120.0))))
    curr_active_pipeline.sess.config.grid_bin_bounds = (((-120.0, 120.0), (-120.0, 120.0)))

    # override_parameters_flat_keypaths_dict = {'grid_bin_bounds': (((-120.0, 120.0), (-120.0, 120.0))), # 'rank_order_shuffle_analysis.minimum_inclusion_fr_Hz': minimum_inclusion_fr_Hz,
    # 										#   'sess.config.preprocessing_parameters.laps.use_direction_dependent_laps': False, # lap_estimation_parameters
    #                                         }

    # curr_active_pipeline.update_parameters(override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict) # should already be updated, but try it again anyway.


    # ==================================================================================================================================================================================================================================================================================== #
    # Update computation_epochs to be only the maze ones                                                                                                                                                                                                                                   #
    # ==================================================================================================================================================================================================================================================================================== #
    # activity_only_epoch_names: List[str] = ['maze1', 'maze2', 'maze_GLOBAL']

    ## activity_only_epochs_df:
    epochs_df = ensure_dataframe(deepcopy(curr_active_pipeline.sess.epochs))
    # activity_only_epochs_df: pd.DataFrame = epochs_df[epochs_df['label'].isin(['maze1', 'maze2', 'maze_GLOBAL'])]

    activity_only_epochs_df: pd.DataFrame = epochs_df[epochs_df['label'].isin(['maze1', 'maze2'])].epochs.get_non_overlapping_df()
    activity_only_epochs: Epoch = ensure_Epoch(activity_only_epochs_df, metadata=curr_active_pipeline.sess.epochs.metadata)

    ## GLobal only ('maze_GLOBAL')
    epochs_df = ensure_dataframe(deepcopy(curr_active_pipeline.sess.epochs))
    global_activity_only_epochs_df: pd.DataFrame = epochs_df[epochs_df['label'].isin(['maze_GLOBAL'])].epochs.get_non_overlapping_df()
    global_activity_only_epoch: Epoch = ensure_Epoch(global_activity_only_epochs_df, metadata=curr_active_pipeline.sess.epochs.metadata)

    ## OUTPUTS: activity_only_epochs, global_activity_only_epoch

    ## OUTPUTS: activity_only_epoch

    ## Need 2 diff active_session_computation_configs:
    active_session_computation_configs[0].pf_params.computation_epochs = deepcopy(activity_only_epochs)

    global_only_sess_comp_config = deepcopy(active_session_computation_configs[0])
    global_only_sess_comp_config.pf_params.computation_epochs = deepcopy(global_activity_only_epoch)
    if len(active_session_computation_configs) < 2:
        active_session_computation_configs.append(global_only_sess_comp_config)
    else:
        active_session_computation_configs[1] = global_only_sess_comp_config

    
    ## UPDATES: active_session_computation_configs

    ## Set linearization mode to umap so it doesn't consume all the memory when trying to linearize position:
    active_session_computation_configs[0].pf_params.linearization_method = "umap"

    for an_epoch_name, a_sess in curr_active_pipeline.filtered_sessions.items():
        ## forcibly compute the linearized position so it doesn't fallback to "isomap" method which eats all the memory
        a_pos_df: pd.DataFrame = a_sess.position.compute_linearized_position(method='umap')
        
    # ==================================================================================================================================================================================================================================================================================== #
    # Ready to compute                                                                                                                                                                                                                                                                     #
    # ==================================================================================================================================================================================================================================================================================== #

    curr_active_pipeline.reload_default_computation_functions()
        
    active_computation_functions_name_includelist = ['pf_computation',
                                                    'pfdt_computation',
                                                    'position_decoding',
                                                    #  'position_decoding_two_step',
                                                    #  'extended_pf_peak_information',
                                                    ] # 'ratemap_peaks_prominence2d'



    for i, a_config in enumerate(active_session_computation_configs):
        active_epoch_names: List[str] = a_config.pf_params.computation_epochs.labels.tolist() ## should be same as config
        print(f'i: {i}, active_epoch_names: {active_epoch_names}') # (activity_only_epoch_names)

        # curr_active_pipeline.perform_computations(active_session_computation_configs[0], computation_functions_name_excludelist=['_perform_spike_burst_detection_computation', '_perform_velocity_vs_pf_density_computation', '_perform_velocity_vs_pf_simplified_count_density_computation']) # SpikeAnalysisComputations._perform_spike_burst_detection_computation
        # curr_active_pipeline.perform_computations(active_session_computation_configs[0], computation_functions_name_includelist=active_computation_functions_name_includelist, enabled_filter_names=activity_only_epoch_names, overwrite_extant_results=True, fail_on_exception=False, debug_print=True) # SpikeAnalysisComputations._perform_spike_burst_detection_computation
        curr_active_pipeline.perform_computations(a_config, computation_functions_name_includelist=active_computation_functions_name_includelist, enabled_filter_names=active_epoch_names, overwrite_extant_results=False, fail_on_exception=False, debug_print=True) # SpikeAnalysisComputations._perform_spike_burst_detection_computation


    # ==================================================================================================================================================================================================================================================================================== #
    # COMPUTE DONE                                                                                                                                                                                                                                                                         #
    # ==================================================================================================================================================================================================================================================================================== #
    print(f'\tcompute done!')
    
    # curr_active_pipeline.computation_results['maze'].accumulated_errors
    curr_active_pipeline.clear_all_failed_computations()

    curr_active_pipeline.prepare_for_display(root_output_dir=r'Output', should_smooth_maze=True) # TODO: pass a display config
    # curr_active_pipeline.prepare_for_display(root_output_dir=r'W:\Data\Output', should_smooth_maze=True) # TODO: pass a display config
    

    if posthoc_save:
        print(f'attempting to save the pipeline...')
        _out = curr_active_pipeline.save_pipeline(saving_mode=PipelineSavingScheme.TEMP_THEN_OVERWRITE)
        _out = curr_active_pipeline.save_global_computation_results()#save_pipeline(saving_mode=PipelineSavingScheme.TEMP_THEN_OVERWRITE, active_pickle_filename='loadedSessPickle_2025-02-27.pkl')
        print(f'done.')
        
    return curr_active_pipeline



@function_attributes(short_name=None, tags=['rachel', 'bapun'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-09-10 07:01', related_items=[])
def post_process_non_kdiba(curr_active_pipeline):
    """ processes either Bapun or Rachel sessions

    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import post_process_non_kdiba

        post_process_non_kdiba(curr_active_pipeline)
        
    """    

    ## build a true global session encompassing all epochs
    # curr_active_pipeline.sess.epochs

    def _subfn_add_approx_head_dir_columns(a_session):
        # INPUTS: a_session 
        # global_pos_obj: Position = deepcopy(a_session.position)
        global_pos_obj: Position = a_session.position # do NOT do a deepcopy, edit in place
        # global_pos_df: pd.DataFrame = global_pos_obj.compute_higher_order_derivatives().position.compute_smoothed_position_info(N=15)
        global_pos_df: pd.DataFrame = global_pos_obj.adding_approx_head_dir_columns(N=15, n_dir_angular_bins=8) # ().position.compute_smoothed_position_info(N=15)
        return global_pos_df



    # included_epochs = ['roam', 'sprinkle']

    global_pos_df = _subfn_add_approx_head_dir_columns(a_session=curr_active_pipeline.sess)

    included_epochs = curr_active_pipeline.active_completed_computation_result_names
    print(f'included_epochs: {included_epochs}')
    for an_epoch_name in included_epochs:
        a_session = deepcopy(curr_active_pipeline.filtered_sessions[an_epoch_name])
        # INPUTS: a_session 
        global_pos_df = _subfn_add_approx_head_dir_columns(a_session=a_session)



@function_attributes(short_name=None, tags=['IMPORTANT', 'pseduo3D', 'pseudoND', 'context-decoding', 'bapun', 'WORKING'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-09-09 10:50', related_items=[])
def build_contextual_pf2D_decoder(curr_active_pipeline, epochs_to_create_global_from_names = ['roam', 'sprinkle']):
    """ The generalized context decoder for Bapun session, which is created out of the specified `epochs_to_create_global_from_names` and then used to decode the 'maze_any' epoch at the specified time bin size.
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import build_contextual_pf2D_decoder, decode_using_contextual_pf2D_decoder
        
        contextual_pf2D_dict, contextual_pf2D, contextual_pf2D_Decoder = build_contextual_pf2D_decoder(curr_active_pipeline, epochs_to_create_global_from_names = ['roam', 'sprinkle'])
    """
    pf2D_Decoder_dict = {k:deepcopy(curr_active_pipeline.computation_results[k].computed_data.pf2D_Decoder) for k in epochs_to_create_global_from_names}
    
    # epochs_to_decode_names = ['maze_any']
    # epochs_df = ensure_dataframe(curr_active_pipeline.sess.epochs)
    # epochs_df = epochs_df.epochs.adding_concatenated_epoch(epochs_to_create_global_from_names=['pre', 'roam', 'sprinkle', 'post'], created_epoch_name='maze_any')
    # global_only_epoch: Epoch = ensure_Epoch(epochs_df[(epochs_df['label'] == 'maze_any')])
    # global_only_epoch

    ## Combine the non-directional PDFs and renormalize to get the directional PDF:
    # Inputs: long_LR_pf1D, long_RL_pf1D
    # long_directional_decoder_dict = dict(zip(long_directional_decoder_names, [deepcopy(long_LR_pf1D), deepcopy(long_RL_pf1D)]))
    ## INPUTS: pf2D_Decoder_dict
    contextual_pf2D_dict = {k:deepcopy(v.pf) for k, v in pf2D_Decoder_dict.items()}
    a_pf = None
    for k, v in contextual_pf2D_dict.items():
        if a_pf is None:
            a_pf = v
        else:
            v, did_update_bins = v.conform_to_position_bins(a_pf)
            print(f'k: {k}: did_update_bins: {did_update_bins}')

    contextual_pf2D: PfND = PfND.build_merged_directional_placefields(contextual_pf2D_dict, debug_print=False)
    contextual_pf2D_Decoder: BasePositionDecoder = BasePositionDecoder(contextual_pf2D, setup_on_init=True, post_load_on_init=True, debug_print=False)
    # return (contextual_pf2D_dict, contextual_pf2D, contextual_pf2D_Decoder)
    ## OUTPUTS: contextual_pf2D_dict, contextual_pf2D, contextual_pf2D_Decoder

    # 2m 35.5s
    return contextual_pf2D_dict, contextual_pf2D, contextual_pf2D_Decoder




@function_attributes(short_name=None, tags=['IMPORTANT', 'pseduo3D', 'pseudoND', 'context-decoding', 'bapun', 'WORKING'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-09-09 10:50', related_items=[])
def decode_using_contextual_pf2D_decoder(curr_active_pipeline, contextual_pf2D_Decoder: BasePositionDecoder, active_laps_decoding_time_bin_size: float = 0.75):
    """ The generalized context decoder for Bapun session, which is created out of the specified `epochs_to_create_global_from_names` and then used to decode the 'maze_any' epoch at the specified time bin size.
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import build_contextual_pf2D_decoder, decode_using_contextual_pf2D_decoder
        ## Build the merged decoder `contextual_pf2D`
        contextual_pf2D_dict, contextual_pf2D, contextual_pf2D_Decoder = build_contextual_pf2D_decoder(curr_active_pipeline, epochs_to_create_global_from_names = ['roam', 'sprinkle'])
        ## Use `contextual_pf2D` to decode specific epochs:
        all_context_filter_epochs_decoder_result, global_only_epoch = decode_using_contextual_pf2D_decoder(curr_active_pipeline, contextual_pf2D_Decoder=contextual_pf2D_Decoder, active_laps_decoding_time_bin_size=0.75)

    """
    desired_global_created_epoch_name: str = 'maze_any'
    # epochs_to_decode_names = ['maze_any']
    epochs_df = ensure_dataframe(curr_active_pipeline.sess.epochs)
    epochs_to_merge_as_global_epoch_names: List[str] = [v for v in epochs_df['label'].to_list() if (v != desired_global_created_epoch_name)]
    print(f'epochs_to_merge_as_global_epoch_names: {epochs_to_merge_as_global_epoch_names}')
    epochs_df = epochs_df.epochs.adding_concatenated_epoch(epochs_to_create_global_from_names=epochs_to_merge_as_global_epoch_names, created_epoch_name=desired_global_created_epoch_name)
    global_only_epoch: Epoch = ensure_Epoch(epochs_df[(epochs_df['label'] == desired_global_created_epoch_name)])
    # global_only_epoch

    epochs_to_decode_dict = {desired_global_created_epoch_name: deepcopy(global_only_epoch)}

    # global_spikes_df: pd.DataFrame = get_proper_global_spikes_df(curr_active_pipeline)
    global_spikes_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.spikes_df)
    # get_proper_global_spikes_df(owning_pipeline_reference, minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values)
    # global_measured_position_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.position.to_dataframe()).dropna(subset=['x', 'y']) # computation_result.sess.position.to_dataframe()
    all_context_filter_epochs_decoder_result: DecodedFilterEpochsResult = contextual_pf2D_Decoder.decode_specific_epochs(spikes_df=deepcopy(global_spikes_df), filter_epochs=ensure_dataframe(epochs_to_decode_dict[desired_global_created_epoch_name]), decoding_time_bin_size=active_laps_decoding_time_bin_size, debug_print=False)
    all_context_filter_epochs_decoder_result: SingleEpochDecodedResult = all_context_filter_epochs_decoder_result.get_result_for_epoch(0)
    all_context_filter_epochs_decoder_result
    ## OUTPUTS: contextual_pf2D_dict, contextual_pf2D, contextual_pf2D_Decoder, all_context_filter_epochs_decoder_result
    # 2m 35.5s
    return all_context_filter_epochs_decoder_result, global_only_epoch


def build_combined_time_synchronized_Bapun_decoders_window(curr_active_pipeline, included_filter_names: List[str]=None, fixed_window_duration = 15.0, controlling_widget=None, context=None, create_new_controlling_widget=True) -> GenericPyQtGraphContainer:
    """ Builds a single window with time_synchronized (time-dependent placefield) plotters controlled by an internal 2DRasterPlot widget.
    
    Usage:
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool import GenericPyQtGraphContainer
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import build_combined_time_synchronized_Bapun_decoders_window

        _out_container: GenericPyQtGraphContainer = build_combined_time_synchronized_Bapun_decoders_window(curr_active_pipeline, included_filter_names=['maze1', 'maze2', 'maze'], fixed_window_duration = 15.0)
    """
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
    from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPositionDecoderPlotter import TimeSynchronizedPositionDecoderPlotter
    from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedOccupancyPlotter import TimeSynchronizedOccupancyPlotter
    from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPlacefieldsPlotter import TimeSynchronizedPlacefieldsPlotter
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper

    from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig
    from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import _temp_debug_two_step_plots_animated_pyqtgraph
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool import GenericPyQtGraphContainer
    

    if context is not None:
        ## Finally, add the display function to the active context
        active_display_fn_identifying_ctx = context.adding_context('combined_time_synchronized_plotters', display_fn_name='combined_time_synchronized_plotters')
        active_display_fn_identifying_ctx_string = active_display_fn_identifying_ctx.get_description(separator='|') # Get final discription string:
        title = f'All Time Synchronized Plotters <{active_display_fn_identifying_ctx_string}>'
    else:
        title = 'All Time Synchronized Plotters'
    
    if included_filter_names is None:
        included_filter_names = ['sprinkle', 'roam']
        
    
    def _merge_plotters(a_controlling_widget, is_controlling_widget_external=False, debug_print=False, **_out_sync_plotters) -> GenericPyQtGraphContainer:
        """ implicitly captures title from the outer function """
        if len(_out_sync_plotters) > 0:
            # out_Width_Height_Tuple = list(_out_sync_plotters.values())[0].desired_widget_size(desired_page_height = 600.0, debug_print=True)
            out_Width_Height_Tuple = list(_out_sync_plotters.values())[0].size()
            out_Width_Height_Tuple = (out_Width_Height_Tuple.width(), out_Width_Height_Tuple.height())
            if debug_print:
                print(f'out_Width_Height_Tuple: {out_Width_Height_Tuple}')
            
            final_desired_width, final_desired_height = out_Width_Height_Tuple
            if debug_print:
                print(f'final_desired_width: {final_desired_width}, final_desired_height: {final_desired_height}')
        
        # build a win of type PhoDockAreaContainingWindow
        root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title=title, defer_show=True)
        
        _display_configs = {}
        _display_dock_items = {}
        _display_sync_connections = {}
        
        for a_name, a_sync_plotter in _out_sync_plotters.items():
            _display_configs[a_name] = CustomDockDisplayConfig(showCloseButton=False)
            _, _display_dock_items[a_name] = root_dockAreaWindow.add_display_dock(f"{a_name}", dockSize=(final_desired_width, final_desired_height), widget=a_sync_plotter, dockAddLocationOpts=['right'], display_config=_display_configs[a_name])
        # END for a_name, a_sync_plotter in _out_sync_plotter...
        
        if a_controlling_widget is not None:
            if not is_controlling_widget_external:
                controlling_widget_id: str = 'Controller'
                a_controlling_widget, _display_dock_items[controlling_widget_id] = root_dockAreaWindow.add_display_dock(identifier=f'{controlling_widget_id}', widget=a_controlling_widget, dockAddLocationOpts=['bottom'])
                
        root_dockAreaWindow.show()
        
        ## Register the children items as drivables/drivers:
        # root_dockAreaWindow.connection_man.register_drivable(curr_sync_occupancy_plotter)
        # root_dockAreaWindow.connection_man.register_drivable(curr_placefields_plotter)
        # Note needed now that DockAreaWrapper sets up drivables/drivers automatically from widgets
        root_dockAreaWindow.try_register_any_control_widgets()
        
        if a_controlling_widget is not None:
            root_dockAreaWindow.connection_man.register_driver(a_controlling_widget)
            # Wire up signals such that time-synchronized plotters are controlled by the RasterPlot2D:
            for a_name, a_sync_plotter in _out_sync_plotters.items():
                _display_sync_connections[a_name] = root_dockAreaWindow.connection_man.connect_drivable_to_driver(drivable=a_sync_plotter, driver=a_controlling_widget,
                                                                custom_connect_function=(lambda driver, drivable: pg.SignalProxy(driver.window_scrolled, delay=0.2, rateLimit=60, slot=drivable.on_window_changed_rate_limited)))
            # END for a_name, a_sync_plotter in _out_sync_plotter...

        _out_container: GenericPyQtGraphContainer = GenericPyQtGraphContainer(name='build_combined_time_synchronized_plotters_window')       
        _out_container.ui.root_dockAreaWindow = root_dockAreaWindow
        _out_container.ui.app = app
        _out_container.ui.display_sync_connections = _display_sync_connections
        _out_container.ui.display_dock_items = _display_dock_items
        _out_container.ui.sync_plotters = _out_sync_plotters
        _out_container.ui.controlling_widget = controlling_widget

        _out_container.plot_data.display_configs = _display_configs
        if context is not None:
            _out_container.plot_data.display_context = context
        if included_filter_names is not None:
            _out_container.params.included_filter_names = included_filter_names ## captured

        return _out_container
    
    

    # ==================================================================================================================================================================================================================================================================================== #
    # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
    # ==================================================================================================================================================================================================================================================================================== #
    global_timeline_start_time: float = np.min([curr_active_pipeline.computation_results[a_filter_name].computed_data['pf2D_Decoder'].pf.filtered_spikes_df['t'].min() for a_filter_name in included_filter_names])
    
    all_epochs_spikes_df: pd.DataFrame = pd.concat([curr_active_pipeline.computation_results[a_filter_name].computed_data['pf2D_Decoder'].pf.filtered_spikes_df for a_filter_name in included_filter_names], axis='index', verify_integrity=True).drop_duplicates(subset=['t_seconds'], ignore_index=True).sort_values(by='t_seconds', ascending=True).reset_index(drop=True) # deepcopy(active_one_step_decoder.pf.filtered_spikes_df) ## #TODO 2025-09-05 06:00: - [ ] This is not right, it's only the first epoch
    
    vis_cols_to_drop = [col for col in ['visualization_raster_y_location', 'visualization_raster_emphasis_state'] if col in all_epochs_spikes_df.columns]
    if len(vis_cols_to_drop) > 0:
        all_epochs_spikes_df = all_epochs_spikes_df.drop(columns=vis_cols_to_drop, inplace=False)
    
    # pg.setConfigOptions(imageAxisOrder='row-major')  # best performance
    
    # Build the 2D Raster Plotter using a fixed window duration    
    if (controlling_widget is None):
        if create_new_controlling_widget:
            spike_raster_plt_2d = Spike2DRaster.init_from_independent_data(all_epochs_spikes_df, window_duration=fixed_window_duration, window_start_time=global_timeline_start_time,
                                                                        neuron_colors=None, neuron_sort_order=None, application_name='TimeSynchronizedPlotterControlSpikeRaster2D',
                                                                        enable_independent_playback_controller=False, should_show=False, parent=None) # setting , parent=spike_raster_plt_3d makes a single window
            spike_raster_plt_2d.setWindowTitle('2D Raster Control Window')
            # Update the 2D Scroll Region to the initial value:
            spike_raster_plt_2d.update_scroll_window_region(global_timeline_start_time, (global_timeline_start_time + fixed_window_duration), block_signals=False)
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
        

    ## Build the specific filter results:
    _out_sync_plotters = {}
    

    for a_filter_name in included_filter_names:
        active_session_configuration_context = curr_active_pipeline.filtered_contexts[a_filter_name]
        computation_result = curr_active_pipeline.computation_results[a_filter_name]

        # Get the decoders from the computation result:
        active_one_step_decoder = computation_result.computed_data['pf2D_Decoder']
        active_two_step_decoder = computation_result.computed_data.get('pf2D_TwoStepDecoder', None)

        time_binned_position_df = computation_result.computed_data.get('extended_stats', {}).get('time_binned_position_df', None)

        active_measured_positions = computation_result.sess.position.to_dataframe()

        ## Build the connected position plotter:
        curr_position_decoder_plotter = TimeSynchronizedPositionDecoderPlotter(active_one_step_decoder=active_one_step_decoder, active_two_step_decoder=active_two_step_decoder)
        if active_measured_positions is not None:
            curr_position_decoder_plotter.params.AnimalTrajectoryPlottingMixin_all_time_pos_df = deepcopy(active_measured_positions)
            curr_position_decoder_plotter.params.AnimalTrajectoryPlottingMixin_filtered_pos_df = deepcopy(active_measured_positions)
            curr_position_decoder_plotter.AnimalTrajectoryPlottingMixin_on_setup()
            curr_position_decoder_plotter.AnimalTrajectoryPlottingMixin_on_buildUI()
            curr_position_decoder_plotter.AnimalTrajectoryPlottingMixin_update_plots()
            
        
        # curr_position_decoder_plotter.show()
        # _conn = pg.SignalProxy(spike_raster_plt_2d.window_scrolled, delay=0.2, rateLimit=60, slot=curr_position_decoder_plotter.on_window_changed_rate_limited) ## connect to plotter
        _out_sync_plotters[a_filter_name] = curr_position_decoder_plotter
    # END for a_filter_name in included_filter_names...

    _out_container = _merge_plotters(controlling_widget, is_controlling_widget_external=is_controlling_widget_external, **_out_sync_plotters)
    
    return _out_container # (controlling_widget, curr_sync_occupancy_plotter, curr_placefields_plotter), root_dockAreaWindow, app

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, SingleEpochDecodedResult

@function_attributes(short_name=None, tags=['track', 'decoded-continuous'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-09-10 11:56', related_items=[])
def _add_context_marginal_to_timeline(active_2d_plot, a_filter_epochs_decoded_result: SingleEpochDecodedResult, name='marginal_ctxt'):
    """
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, SingleEpochDecodedResult
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _add_context_marginal_to_timeline, _add_context_decoded_epoch_marginals_to_timeline
        
        ## Decode PBEs please
        pbes = deepcopy(curr_active_pipeline.sess.pbe)
        ripple_decoding_time_bin_size: float = 0.025 # 25ms
        global_spikes_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.spikes_df)
        pbe_decoder_result: DecodedFilterEpochsResult = contextual_pf2D_Decoder.decode_specific_epochs(spikes_df=deepcopy(global_spikes_df), filter_epochs=ensure_dataframe(pbes), decoding_time_bin_size=ripple_decoding_time_bin_size, debug_print=False)

        _out = _add_context_marginal_to_timeline(active_2d_plot, a_filter_epochs_decoded_result=all_context_filter_epochs_decoder_result, name='global context')

    """
    p_x_given_n = deepcopy(a_filter_epochs_decoded_result.p_x_given_n)

    marginal_z = np.nansum(p_x_given_n, axis=(0, 1)) 
    marginal_z = marginal_z / np.sum(marginal_z, axis=0, keepdims=True) # sum over all directions for each time_bin (so there's a normalized distribution at each timestep)
    # print(f'marginal_z.shape: {np.shape(marginal_z)}')
    _out = active_2d_plot.add_docked_marginal_track(name=name, time_window_centers=deepcopy(a_filter_epochs_decoded_result.time_bin_container.centers), a_1D_posterior=marginal_z, a_variable_name='p_x_given_n')
    return _out

@function_attributes(short_name=None, tags=['track', 'multi-track', 'decoded-epochs'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-09-10 11:55', related_items=[])
def _add_context_decoded_epoch_marginals_to_timeline(active_2d_plot, decoded_epochs_result: DecodedFilterEpochsResult, epochs_name: str = 'pbe'):
    """ 
    Usage:
    
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, SingleEpochDecodedResult
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _add_context_marginal_to_timeline, _add_context_decoded_epoch_marginals_to_timeline
        
        ## Decode PBEs please
        pbes = deepcopy(curr_active_pipeline.sess.pbe)
        ripple_decoding_time_bin_size: float = 0.025 # 25ms
        global_spikes_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.spikes_df)
        pbe_decoder_result: DecodedFilterEpochsResult = contextual_pf2D_Decoder.decode_specific_epochs(spikes_df=deepcopy(global_spikes_df), filter_epochs=ensure_dataframe(pbes), decoding_time_bin_size=ripple_decoding_time_bin_size, debug_print=False)

        _out_pbe_tracks = _add_context_decoded_epoch_marginals_to_timeline(active_2d_plot=active_2d_plot, decoded_epochs_result=pbe_decoder_result)

    """
    
    decoded_epochs_track_name: str = f'{epochs_name}[{ripple_decoding_time_bin_size}]'

    slices_posteriors = [np.nansum(a_p_x_given_x, axis=(0, 1)) for a_p_x_given_x in decoded_epochs_result.p_x_given_n_list]
    slices_posteriors = [(marginal_z / np.sum(marginal_z, axis=0, keepdims=True)) for marginal_z in slices_posteriors]

    _out_epochs_tracks = active_2d_plot.add_docked_decoded_posterior_slices_track(name=decoded_epochs_track_name, slices_time_window_centers=decoded_epochs_result.time_window_centers, slices_posteriors=slices_posteriors, measured_position_df=None)
    return _out_epochs_tracks



# ==================================================================================================================================================================================================================================================================================== #
# 2025-08-26 - Final Correct Context Decoding Stabilities:                                                                                                                                                                                                                             #
# ==================================================================================================================================================================================================================================================================================== #
@function_attributes(short_name=None, tags=['decoding', 'performance'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-08-26 17:59', related_items=[])
def determine_percent_correctly_decoded_contexts(curr_active_pipeline, time_bin_size: float=0.060) -> pd.DataFrame:
    """ 
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import determine_percent_correctly_decoded_contexts
    ## find the number of correctly decoded components:
    records_df: pd.DataFrame = determine_percent_correctly_decoded_contexts(curr_active_pipeline, time_bin_size=time_bin_size)
    records_df
    
    """
    from pyphocorehelpers.assertion_helpers import Assert
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionIdentityDataframeAccessor

    def _subfn_determine_num_correctly_decoded_time_bins(a_decoded_marginal_posterior_df):
        """find the number of correctly decoded components:
        
            worse_percent_correct, (percent_correct_pre, n_correct_pre, n_total_pre), (percent_correct_post, n_correct_post, n_total_post) = _subfn_determine_num_correctly_decoded_time_bins(a_decoded_marginal_posterior_df=a_decoded_marginal_posterior_df)
        """
        Assert.require_columns(a_decoded_marginal_posterior_df, required_columns=['P_Long', 'pre_post_delta_category'])
        a_decoded_marginal_posterior_df['is_most_likely_decoder_Long'] = (a_decoded_marginal_posterior_df['P_Long'] > 0.5)

        _split_df = a_decoded_marginal_posterior_df.pho.partition_df_dict('pre_post_delta_category')

        is_correct_pre_delta = _split_df['pre-delta']['is_most_likely_decoder_Long']
        is_correct_post_delta = np.logical_not(_split_df['post-delta']['is_most_likely_decoder_Long'])


        n_correct_pre: int = np.sum(is_correct_pre_delta)
        n_total_pre: int = len(_split_df['pre-delta'])
        percent_correct_pre: float = float(n_correct_pre)/float(n_total_pre)
        
        n_correct_post: int = np.sum(is_correct_post_delta)
        n_total_post: int = len(_split_df['post-delta'])
        percent_correct_post: float = float(n_correct_post)/float(n_total_post)
        
        worse_percent_correct: float = min(percent_correct_pre, percent_correct_post)
        
        return worse_percent_correct, (percent_correct_pre, n_correct_pre, n_total_pre), (percent_correct_post, n_correct_post, n_total_post)
    
    # ==================================================================================================================================================================================================================================================================================== #
    # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
    # ==================================================================================================================================================================================================================================================================================== #
    valid_EpochComputations_result: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['EpochComputations']
    a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result

    # a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, time_bin_size=0.050, known_named_decoding_epochs_type='pbe', masked_time_bin_fill_type='ignore') # , decoder_identifier='long_LR'
    # a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, time_bin_size=0.025, known_named_decoding_epochs_type='pbe', masked_time_bin_fill_type='ignore', data_grain='per_epoch') # , time_bin_size=0.050, known_named_decoding_epochs_type='pbe', masked_time_bin_fill_type='ignore', decoder_identifier='long_LR'

    # a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, time_bin_size=0.025, known_named_decoding_epochs_type='laps', masked_time_bin_fill_type='ignore', data_grain='per_epoch') ## Laps
    # any_matching_contexts_list, result_context_dict, decoder_context_dict, decoded_marginal_posterior_df_context_dict = a_new_fully_generic_result.get_results_matching_contexts(context_query=a_target_context)

    # common_constraint_dict = dict(trained_compute_epochs='laps', pfND_ndim=1, time_bin_size=0.025, masked_time_bin_fill_type='ignore')
    # common_constraint_dict = dict(trained_compute_epochs='laps', time_bin_size=0.060, masked_time_bin_fill_type='nan_filled') # , pfND_ndim=1
    common_constraint_dict = dict(trained_compute_epochs='laps', time_bin_size=time_bin_size, masked_time_bin_fill_type='dropped')

    _output_dict = {}
    ## Laps context:
    a_Laps_target_context: IdentifyingContext = IdentifyingContext(known_named_decoding_epochs_type='laps', data_grain='per_time_bin', **common_constraint_dict)
    ## Global context:
    a_global_target_context: IdentifyingContext = IdentifyingContext(known_named_decoding_epochs_type='global', data_grain='per_time_bin', **common_constraint_dict)
    ## PBEs context:
    a_PBEs_target_context: IdentifyingContext = IdentifyingContext(known_named_decoding_epochs_type='pbe', **common_constraint_dict, data_grain='per_time_bin') 
    _active_target_context_list = [a_Laps_target_context, a_global_target_context, a_PBEs_target_context]
    records_df = []
    for a_target_context in _active_target_context_list:
        try:
            best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context, debug_print=False)
            a_num_counts_tuple  = _subfn_determine_num_correctly_decoded_time_bins(a_decoded_marginal_posterior_df=a_decoded_marginal_posterior_df)
            _output_dict[best_matching_context] = a_num_counts_tuple
            worse_percent_correct, (percent_correct_pre, n_correct_pre, n_total_pre), (percent_correct_post, n_correct_post, n_total_post) = a_num_counts_tuple
            a_record = dict(**best_matching_context.to_dict(), worse_percent_correct=worse_percent_correct, percent_correct_pre=percent_correct_pre, n_correct_pre=n_correct_pre, n_total_pre=n_total_pre,  percent_correct_post=percent_correct_post, n_correct_post=n_correct_post, n_total_post=n_total_post)
            records_df.append(a_record)            

        except TypeError as e:
            print(f'WARN: err: {e} for ctxt: {a_target_context}. Skipping.')
            pass
        except Exception as e:
            raise
    ## END for a_target_context in ...
    
    ## build output df:
    records_df: pd.DataFrame = pd.DataFrame.from_records(records_df)
    records_df = records_df.across_session_identity.add_session_df_columns_from_pipeline(curr_active_pipeline=curr_active_pipeline, time_bin_size=time_bin_size, time_col=None)
    return records_df



# ==================================================================================================================================================================================================================================================================================== #
# 2025-08-19 - Each Cell's Time of reaching pf inclusion criteria                                                                                                                                                                                                                      #
# ==================================================================================================================================================================================================================================================================================== #

from neuropy.core.epoch import subdivide_epochs, ensure_dataframe
from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent
import matplotlib.pyplot as plt
import numpy as np

@metadata_attributes(short_name=None, tags=['pf1D_dt', 'aclu', 'stability', 'pf'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-08-18 14:24', related_items=[])
class AcluFirstPlacefieldStabilityThresholdFigure:
    """ plot the time that each cell crossed the stability threshold (on the occupancy normalized pf)
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import AcluFirstPlacefieldStabilityThresholdFigure

    df_merged, decoder_outputs, pf1D_dt_outputs, pf1D_dt_snapshot_outputs = AcluFirstPlacefieldStabilityThresholdFigure._compute_for_all_decoders(curr_active_pipeline, track_templates, fr_threshold_Hz=2.0)
    df_merged
    fig, ax = AcluFirstPlacefieldStabilityThresholdFigure.plot_aclus_first_significance_figure(curr_active_pipeline=curr_active_pipeline, track_templates=track_templates, df_merged=df_merged, is_delta_relative=False)
    
    """
    @classmethod
    def _compute_for_all_decoders(cls, curr_active_pipeline, track_templates, subdivide_bin_size: float = 0.050, fr_threshold_Hz: float = 1.0) -> pd.DataFrame:
        """ Computes the first-significance time for each aclu within each decoder

        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import AcluFirstPlacefieldStabilityThresholdFigure
            
            df_merged, decoder_outputs, pf1D_dt_outputs, pf1D_dt_snapshot_outputs = AcluFirstPlacefieldStabilityThresholdFigure._compute_for_all_decoders(curr_active_pipeline, track_templates, fr_threshold_Hz=2.0)
            # 10m+ not sure why I started taking so long, I think I just modified the return values (returning more of them)

        """
        # an_epoch_results = long_LR_results

        # track_templates: TrackTemplates = directional_laps_results.get_templates(minimum_inclusion_fr_Hz=minimum_inclusion_fr_Hz, included_qclu_values=included_qclu_values) # non-shared-only
        # long_LR_decoder, long_RL_decoder, short_LR_decoder, short_RL_decoder = track_templates.get_decoders()

        # # Unpack all directional variables:
        ## INPUTS: long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj
        # decoder_template_names = (long_LR_name, long_RL_name, short_LR_name, short_RL_name)
        decoder_template_names = track_templates.get_decoder_names()
        ## These are the "*_odd"/"*_even" names:
        long_LR_name, short_LR_name, long_RL_name, short_RL_name = ['maze1_odd', 'maze2_odd', 'maze1_even', 'maze2_even'] ## OLD style
        # Unpacking for `(long_LR_name, long_RL_name, short_LR_name, short_RL_name)`
        long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj = [curr_active_pipeline.computation_results[an_epoch_name].computation_config.pf_params.computation_epochs for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)] # note has global also
        (long_LR_results, long_RL_results, short_LR_results, short_RL_results) = [curr_active_pipeline.computation_results[an_epoch_name].computed_data for an_epoch_name in (long_LR_name, long_RL_name, short_LR_name, short_RL_name)]

        epoch_results_objs_dict = dict(zip(decoder_template_names, (long_LR_results, long_RL_results, short_LR_results, short_RL_results)))
        epochs_objs_dict = dict(zip(decoder_template_names, (long_LR_epochs_obj, long_RL_epochs_obj, short_LR_epochs_obj, short_RL_epochs_obj)))

        pf1D_dt_outputs = {}
        pf1D_dt_snapshot_outputs = {}
        # subdiv_df_outputs = {}
        decoder_outputs = {}

        ## Loop through each epoch:
        for a_decoder_name, an_epoch_results in epoch_results_objs_dict.items():    
            a_pf1D_dt: PfND_TimeDependent = deepcopy(an_epoch_results.pf1D_dt)
            df: pd.DataFrame = ensure_dataframe(deepcopy(epochs_objs_dict[a_decoder_name])) ## Should be the laps
            df['epoch_type'] = 'lap'
            df['interval_type_id'] = 666 ## sentinal value, unsure what the 'interval_type_id' was used for but it's better to have it unique. 
            subdiv_df: pd.DataFrame = subdivide_epochs(df, subdivide_bin_size)
            # subdiv_df_outputs[a_decoder_name] = deepcopy(subdiv_df)
            _a_pf1D_dt_snapshots = a_pf1D_dt.batch_snapshotting(subdiv_df, is_start_relative_t=False, reset_at_start=True)
            included_neuron_IDs = deepcopy(a_pf1D_dt.included_neuron_IDs)
            # pf1D_dt_snapshot_outputs[a_decoder_name] = deepcopy(_a_pf1D_dt_snapshots)
            pf1D_dt_snapshot_outputs[a_decoder_name] = _a_pf1D_dt_snapshots
            pf1D_dt_outputs[a_decoder_name] = a_pf1D_dt
            _outs = PfND_TimeDependent.find_aclu_stabilizing_times(_a_pf1D_dt_snapshots=_a_pf1D_dt_snapshots, included_neuron_IDs=included_neuron_IDs, fr_threshold_Hz=fr_threshold_Hz)
            # (aclu_first_firing_snapshot_duration_fraction, aclu_first_firing_snapshot_timestep, aclu_first_firing_snapshot_idx), (snapshot_timestamps, _a_pf1D_dt_snapshots) = _outs ## Unpack
            _aclu_first_firing_tuple, (snapshot_timestamps, _a_pf1D_dt_snapshots) = _outs
            # decoder_outputs[a_decoder_name] = _outs
            decoder_outputs[a_decoder_name] = _aclu_first_firing_tuple ## Just the firing rate tuple
            
        ## 3m 47s
        ## Build combined data columns:
        data_col_names = ['duration_fraction', 'snap_t', 'snap_idx']

        outs_df = []
        for a_decoder_name, _outs in decoder_outputs.items():
            # _aclu_first_firing_tuple, (snapshot_timestamps, _a_pf1D_dt_snapshots) = _outs
            _aclu_first_firing_tuple = _outs
            _an_outs_df = pd.DataFrame.from_records(_aclu_first_firing_tuple).T
            _an_outs_df.columns = [f'{c}_{a_decoder_name}' for c in data_col_names]
            _an_outs_df = _an_outs_df.reset_index(names=['aclu'])
            outs_df.append(_an_outs_df)

        # outs_df = [df_long_LR, df_short_LR, df_long_RL, df_short_RL]
        df_merged: pd.DataFrame = outs_df[0]
        for df in outs_df[1:]:
            df_merged = pd.merge(df_merged, df, on='aclu', how='outer')

        df_merged = df_merged.sort_values(by='aclu', ascending=True, inplace=False).reset_index(drop=True)
        # df_merged.drop_duplicates(subset=['aclu'], inplace=False)

        ## Convert to delta relative times:
        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()

        ## INPUTS: decoder_template_names

        snap_t_col_names = [f'snap_t_{a_decoder_name}' for a_decoder_name in decoder_template_names] # ['snap_t_long_LR', 'snap_t_long_RL', 'snap_t_short_LR', 'snap_t_short_RL']
        delta_rel_snap_t_col_names = [f'delta_rel_snap_t_{a_decoder_name}' for a_decoder_name in decoder_template_names]
        df_merged[delta_rel_snap_t_col_names] = deepcopy(df_merged[snap_t_col_names]) - t_delta

        # subdiv_df_outputs is not used
        return df_merged, decoder_outputs, pf1D_dt_outputs, pf1D_dt_snapshot_outputs


    @function_attributes(short_name=None, tags=['figure', 'plot', 'matplotlib', 'aclu'], input_requires=[], output_provides=[], uses=['plot_laps_2d'], used_by=[], creation_date='2025-08-19 13:01', related_items=[])
    @classmethod
    def plot_aclus_first_significance_figure(cls, curr_active_pipeline, track_templates, df_merged: pd.DataFrame, is_delta_relative: bool=False, extant_ax=None):
        """ plot the time that each cell crossed the stability threshold (on the occupancy normalized pf)
        
        INPUTS: long_LR_name, long_RL_name, short_LR_name, short_RL_name
        
        Usage:
            fig, ax = AcluFirstPlacefieldStabilityThresholdFigure.plot_aclus_first_significance_figure(curr_active_pipeline=curr_active_pipeline, track_templates=track_templates, df_merged=df_merged, is_delta_relative=False)
        
        """
        ## Add in the position/laps
        from pyphoplacecellanalysis.PhoPositionalData.plotting.laps import plot_laps_2d

        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_session = curr_active_pipeline.filtered_sessions[global_epoch_name]

        axes_list = None
        if extant_ax is not None:
            raise NotImplementedError(f'OH NO')
            # axes_list = [extant_ax]
            
        fig = plt.figure(layout="constrained", figsize=(24, 10), clear=True)
        ax_dict = fig.subplot_mosaic(
            [
                ["ax_cum_hist"],
                ["ax_main"],
            ],
            # set the height ratios between the rows
            height_ratios=[2, 8],
            # height_ratios=[1, 1],
            sharex=True,
            gridspec_kw=dict(wspace=0, hspace=0) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
        )
        axes_list = [ax_dict['ax_main']]
        plot_laps_kwargs = dict(include_velocity=False, include_accel=False, figsize=(24, 10), axes_list=axes_list, span_where_kwargs=dict(alpha=0.1))
        fig, out_axes_list = plot_laps_2d(global_session, legacy_plotting_mode=True, **plot_laps_kwargs)
        ax = out_axes_list[0]
        ymin, ymax = ax.get_ylim()
        ymid: float = (float(ymax) - float(ymin))/2.0
        
        ## Plot the aclu lines:
        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()

        # decoder_template_names = (long_LR_name, long_RL_name, short_LR_name, short_RL_name)
        decoder_template_names = track_templates.get_decoder_names()
        long_decoder_names = track_templates.get_decoder_names()[:2]
        short_decoder_names = track_templates.get_decoder_names()[2:]

        ## INPUTS: decoder_template_names
        snap_t_col_names = [f'snap_t_{a_decoder_name}' for a_decoder_name in decoder_template_names] # ['snap_t_long_LR', 'snap_t_long_RL', 'snap_t_short_LR', 'snap_t_short_RL']
        
        # fig, ax = plt.subplots()
        # ax.set_xlim(t_start, t_end)
        # ax.set_xlim(t_start-t_delta, t_end-t_delta)

        # delta_line = ax.vlines(0.0, ymin=-1, ymax=1)
        if is_delta_relative:
            delta_line = plt.axvline(t_delta, color='k', linestyle='--')
        else:
            delta_line = plt.axvline(0.0, color='k', linestyle='--')
            
        base_line = plt.axhline(ymid, color='k', linestyle='--')

        # snap_t_col_names = [f'snap_t_{a_decoder_name}' for a_decoder_name in decoder_template_names] # ['snap_t_long_LR', 'snap_t_long_RL', 'snap_t_short_LR', 'snap_t_short_RL']
        # duration_cols = ['duration_fraction_long_LR', 'duration_fraction_short_LR', 'duration_fraction_long_RL', 'duration_fraction_short_RL']
        if not is_delta_relative:
            active_cols = deepcopy(snap_t_col_names)
        else:
            delta_rel_snap_t_col_names = [f'delta_rel_snap_t_{a_decoder_name}' for a_decoder_name in decoder_template_names]
            df_merged[delta_rel_snap_t_col_names] = deepcopy(df_merged[snap_t_col_names]) - t_delta
            active_cols = deepcopy(delta_rel_snap_t_col_names) # ['delta_rel_snap_t_long_LR', 'delta_rel_snap_t_long_RL', 'delta_rel_snap_t_short_LR', 'delta_rel_snap_t_short_RL']
            
        _all_common_kwargs = dict(lw=0.5, alpha=0.7)
        _left_common_kwargs = dict(ymin=ymid, ymax=ymax, **_all_common_kwargs)
        _right_common_kwargs = dict(ymin=ymin, ymax=ymid, **_all_common_kwargs)

        active_col_kwargs_list = [dict(**_left_common_kwargs, label='Long_LR', colors='red'),
                                dict(**_right_common_kwargs, label='Long_RL', colors='orange'),
                                dict(**_left_common_kwargs, label='Short_LR', colors='blue'),
                                dict(**_right_common_kwargs, label='Short_RL', colors='cyan')]

        offsets = np.linspace(-0.4, 0.4, len(df_merged)) ## stagger the aclus

        for i, row in df_merged.iterrows():
            for j, col in enumerate(active_cols):
                if not pd.isna(row[col]):
                    an_aclu: int = int(row['aclu'])
                    an_x = row[col]
                    _lines_artist = ax.vlines(an_x, **active_col_kwargs_list[j]) #+ offsets[j] , -1.0, min(1.0, row[col]), colors=f"C{j}"
                    # Adding text inside the plot
                    _aclu_label_artist = ax.text(an_x, offsets[i], f'{an_aclu}', fontsize=9)

                    # _lines_artist = ax.vlines(row['aclu'] + offsets[j], -1.0, min(1.0, row[col]), colors=f"C{j}")

        # plt.legend()
        
        fig.canvas.manager.set_window_title("plot_aclus_first_significance_figure")
        ax.set_xlabel("t (seconds)")
        # ax.set_ylabel("Y Label")
        ax.set_title("First Neuron Pf Significant Time")

        ax.set_ylim(ymin, ymax)

        ## Histogram of cells over time plotted on `ax_dict["ax_cum_hist"]`:        
        long_col_names = active_cols[:2]
        short_col_names = active_cols[2:]

        long_aclu_times = np.concatenate([df_merged[a_col].to_numpy()[np.logical_not(np.isnan(df_merged[a_col].to_numpy()))] for a_col in long_col_names]) ## all long_times flattened
        short_aclu_times = np.concatenate([df_merged[a_col].to_numpy()[np.logical_not(np.isnan(df_merged[a_col].to_numpy()))] for a_col in short_col_names]) ## all long_times flattened

        long_hist_artist = ax_dict['ax_cum_hist'].hist(x=long_aclu_times, cumulative=True, label='long')
        short_hist_artist = ax_dict['ax_cum_hist'].hist(x=short_aclu_times, cumulative=True, label='short')
        
        # plt.show()

        return fig, ax_dict


# ==================================================================================================================================================================================================================================================================================== #
# Pre 2025-08-17                                                                                                                                                                                                                                                                       #
# ==================================================================================================================================================================================================================================================================================== #

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import SingleBarResult, InstantaneousSpikeRateGroupsComputation, SpikeRateTrends
import nptyping as ND
from nptyping import NDArray
from neuropy.core.user_annotations import SessionCellExclusivityRecord

class SpareRunningSequenceScore:
    """ Computes 'Spare' (as in the sport American Bowling) scoring of decoded posteriors
    
    STATUS #TODO 2025-08-05 09:16: - [ ] SpareRunningSequenceScore refinements, finished implementation, now need to check
    
    #TODO 2025-08-11 10:09: - [ ] Potential issue: doesn't the jump integration method reward larger jumps by making the function grow faster for long jumps than short ones? I mean I suppose it ends earlier too, but kinda opposite of what I'd like. 

    #TODO 2025-08-11 12:11: - [ ] It really doesn't work very well, much to my surprise. It skips even laps, etc.  

    
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import SpareRunningSequenceScore
        from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _helper_add_interpolated_position_columns_to_decoded_result_df

        ## Run heuristic continuously to determine when to split sequences and thus where replays are

        valid_EpochComputations_result: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['EpochComputations']
        a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result

        ## INPUTS: a_new_fully_generic_result
        # a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='nan_filled', data_grain='per_time_bin')
        a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='ignore', data_grain='per_time_bin')
        best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context, debug_print=False)
        ## OUTPUTS: a_result, a_decoder, a_decoded_marginal_posterior_df
        ## INPUTS: curr_active_pipeline, a_result, a_decoder, a_decoded_marginal_posterior_df
        global_measured_position_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.position.to_dataframe())
        a_decoded_marginal_posterior_df: pd.DataFrame = _helper_add_interpolated_position_columns_to_decoded_result_df(a_result=a_result, a_decoder=a_decoder, a_decoded_marginal_posterior_df=a_decoded_marginal_posterior_df, global_measured_position_df=global_measured_position_df)

        global_decoded_result: SingleEpochDecodedResult = a_result.get_result_for_epoch(0)
        p_x_given_n: NDArray[ND.Shape["N_POS_BINS, 4, N_TIME_BINS"], np.floating] = deepcopy(global_decoded_result.p_x_given_n) # .shape # (59, 4, 69488)
        time_bin_centers: NDArray[ND.Shape["N_TIME_BINS"], np.floating] = deepcopy(global_decoded_result.time_bin_container.centers)
        xbin: NDArray[ND.Shape["N_POS_BINS"], np.floating] = deepcopy(a_decoder.xbin)

        ## INPUTS: p_x_given_n
        out_decoder_spare_scores, out_decoder_spare_scores_extras = SpareRunningSequenceScore.bowling_spare_integration(p_x_given_n=p_x_given_n)
        out_decoder_spare_scores
    """
    @function_attributes(short_name=None, tags=['score', 'bowling', 'spare', 'compute'], input_requires=[], output_provides=[], uses=[], used_by=['bowling_spare_integration'], creation_date='2025-08-04 10:11', related_items=[])
    @classmethod
    def compute_spare_operation(cls, a_p_x_given_n: NDArray, final_score_only: bool=True) -> List[NDArray]:
        """ Breaks the `a_p_x_given_n` for a single decoder into separate subsequences based on the change direction (currently ignoring no bins) and then computes the "bowling_spare" score for each time bin
        
        if final_score_only: return only the maximal score (the last sequence bin) instead of the score for each bin within the sequence.
        
        Usage:
        
            out_spare_score, (p_x_given_n_segments, most_likely_pos_idxs_segments, segement_lengths, sign_change_indicies) = compute_spare_operation(a_p_x_given_n=a_p_x_given_n, final_score_only=False)
        """
        ## start at the end of the posterior
        n_pos, n_time_bins = np.shape(a_p_x_given_n) # np.shape(p_x_given_n) - (59, 69488) -(n_pos, n_time_bins)
        max_pos_index: int = n_pos - 1
        a_most_likely_pos_idxs: NDArray = np.argmax(a_p_x_given_n, axis=0) ## find the max position bins (69488, ) - (n_time_bins, )
        # out_spare_score = np.full_like(a_p_x_given_n, fill_value=np.nan)
        out_spare_score = [] # np.full_like(a_p_x_given_n, fill_value=np.nan)
        
        ## find the "miss" bins
        # most_likely_pos_idx_change = [0, np.diff(a_most_likely_pos_idxs)]
        # sign_change_locations = np.diff(np.sign(most_likely_pos_idx_change)) # -1 if x < 0, 0 if x==0, 1 if x > 0
        diff = np.diff(a_most_likely_pos_idxs)
        signs = np.sign(diff)
        sign_change_indicies = np.where(np.diff(signs) != 0)[0] + 1 ## I believe the +1 is to handle the loss of an index when we performed np.diff(...)
        p_x_given_n_segments = np.split(a_p_x_given_n, sign_change_indicies, axis=1)
        most_likely_pos_idxs_segments = np.split(a_most_likely_pos_idxs, sign_change_indicies)
        n_segments: int = len(p_x_given_n_segments)
        segement_lengths = np.array([np.shape(v)[-1] for v in p_x_given_n_segments]) ## each segment is [n_pos_bins, n_seg_time_bins]

        for seg_idx, a_seg in enumerate(p_x_given_n_segments):
            a_seg_len: int = segement_lengths[seg_idx]
            a_most_likely_pos_seg = most_likely_pos_idxs_segments[seg_idx]
            a_seq_spare_score = []
            for t_idx in reversed(np.arange(a_seg_len)):
                ## start in the last frame and work forward until the first
                # sign_change_locations[t_idx]
                if t_idx > 0:
                    ## for any but the first index in the series            
                    start_bound = a_most_likely_pos_seg[t_idx]
                    # need to know the max index                
                    end_bound = a_most_likely_pos_seg[t_idx-1]
                    
                else:
                    ## if it is the first bound in the series, we need to decide which side to integrate from (it should be the closest to curr peak:
                    curr_pos_idx: int = a_most_likely_pos_seg[t_idx]
                    _curr_pre_pos_bins: int = (max_pos_index - curr_pos_idx)
                    _curr_post_pos_bins: int = (curr_pos_idx - 0)
                    _curr_should_start_at_pre: bool = _curr_pre_pos_bins <= _curr_post_pos_bins
                    if _curr_should_start_at_pre:
                        start_bound = 0
                        end_bound = a_most_likely_pos_seg[t_idx]
                    else:
                        start_bound = a_most_likely_pos_seg[t_idx]
                        end_bound = max_pos_index

                # out_spare_score[t_idx] =  
                a_seq_spare_score.append(np.nansum(a_seg[start_bound:end_bound, t_idx])) ## sum over all values of the segment
            # for t_idx in reversed(np.arange(a_seg_len))
            a_seq_spare_score = np.nan_to_num(np.array(a_seq_spare_score), nan=0.0)
            a_seq_spare_score = np.cumsum(a_seq_spare_score)
            if final_score_only:
                a_seq_spare_score = a_seq_spare_score[-1] ## only the last bin, which is by defn maximal
            out_spare_score.append(a_seq_spare_score)
        # END for seg_idx, a_seg in enumerate(p_x_given_n_segments)
                  
        return out_spare_score, (p_x_given_n_segments, most_likely_pos_idxs_segments, segement_lengths, sign_change_indicies)


    @function_attributes(short_name=None, tags=['MAIN', 'score', 'spare'], input_requires=[], output_provides=[], uses=['compute_spare_operation'], used_by=[], creation_date='2025-08-04 10:11', related_items=[])
    @classmethod
    def bowling_spare_integration(cls, p_x_given_n: NDArray, decoder_names = ['Long_LR', 'Long_RL', 'Short_LR', 'Short_RL']) -> NDArray:
        """ Start at the end of the posterior 


        Usage:
        
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import SpareRunningSequenceScore

            out_decoder_spare_scores, out_decoder_spare_scores_extras = SpareRunningSequenceScore.bowling_spare_integration(p_x_given_n=p_x_given_n)
            out_decoder_spare_scores

        """
        n_pos, n_decoders, n_time_bins = np.shape(p_x_given_n) # np.shape(p_x_given_n) - (59, 4, 69488) -(n_pos, 4, n_time_bins)
        assert n_decoders == len(decoder_names), f"{decoder_names} length not equal to n_decoders: {n_decoders}"
        _decoder_prob_sum_over_all_positions = np.nansum(p_x_given_n, axis=0) ## sum over all positions (4, 69488)
        most_likely_decoder_idxs: NDArray = np.argmax(_decoder_prob_sum_over_all_positions, axis=0) ## find the max decoder idx for each time bins (4, 69488) - (n_time_bins, )
        # most_likely_decoder_idxs # .shape
        # p_x_given_n.shape # p_x_given_n.shape (59, 4, 69488)
        # p_x_given_n[most_likely_decoder_idxs, :]
        most_likely_pos_idxs: NDArray = np.argmax(p_x_given_n, axis=0) ## find the max position bins (4, 69488) - (n_decoders, n_time_bins)
        a_most_likely_pos_idxs = most_likely_pos_idxs[0,:] ## single decoder result .shape (n_time_bins)
            
        out_decoder_spare_scores = {} ## one for each decoder
        out_decoder_spare_scores_extras = {}
        ## compute each decoder indepednently
        # for decoder_idx in np.arange(n_decoders):
        for decoder_idx, a_decoder_name in enumerate(decoder_names):
            a_p_x_given_n = deepcopy(p_x_given_n[:, decoder_idx, :])
            ## Normalize to this decoder by summing over each time bin and dividing by the output
            a_p_x_given_n = a_p_x_given_n / np.nansum(a_p_x_given_n, axis=0)
            
            most_likely_decoder_idxs: NDArray = np.argmax(_decoder_prob_sum_over_all_positions, axis=0) ## find the max decoder idx for each time bins (4, 69488) - (n_time_bins, )
            a_most_likely_pos_idxs: NDArray = np.argmax(a_p_x_given_n, axis=0) ## find the max position bins (4, 69488) - (n_decoders, n_time_bins)
            a_most_likely_pos_idxs = most_likely_pos_idxs[0,:] ## single decoder result .shape (n_time_bins)

            out_spare_score, (p_x_given_n_segments, most_likely_pos_idxs_segments, segement_lengths, sign_change_indicies) = cls.compute_spare_operation(a_p_x_given_n=a_p_x_given_n, final_score_only=False)
            # out_decoder_spare_scores.append(out_spare_score)
            out_decoder_spare_scores[a_decoder_name] = out_spare_score
            out_decoder_spare_scores_extras[a_decoder_name] = dict(p_x_given_n_segments=p_x_given_n_segments, most_likely_pos_idxs_segments=most_likely_pos_idxs_segments,
                                                                    segement_lengths=segement_lengths, sign_change_indicies=sign_change_indicies)
        # Extract the maximum locations for each time bin
        P_max_ind = np.argmax(p_x_given_n, axis=1)
        return out_decoder_spare_scores, out_decoder_spare_scores_extras


    @classmethod
    def add_spikeRaster2D_interval_rects(cls, active_2d_plot: Spike2DRaster, seg_df: pd.DataFrame, **kwargs):
        """ 
        
        spare_seq_dfs_datasources_dict, spare_seq_dfs_dict = SpareRunningSequenceScore.add_spikeRaster2D_interval_rects(active_2d_plot, seg_df=seg_df)
        """
        
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs, inline_mkColor
        
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.EpochRenderingMixin import EpochRenderingMixin, RenderedEpochsItemsContainer
        from pyphoplacecellanalysis.General.Model.Datasources.IntervalDatasource import IntervalsDatasource
        from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol

        if 't_duration' not in seg_df.columns:
            seg_df['t_duration'] = seg_df['t_end'] - seg_df['t_start']

        ## Use the three dataframes as separate Epoch series:
        spare_seq_dfs_dict = {
            'SpareSeqScore': seg_df,
        }

        spare_seq_epochs_formatting_dict = {
            'SpareSeqScore':dict(y_location=-5.0, height=0.9, pen_color=inline_mkColor('purple', 0.8), brush_color=inline_mkColor('purple', 0.5)),
        }

        # required_vertical_offsets, required_interval_heights = EpochRenderingMixin.build_stacked_epoch_layout([0.2], epoch_render_stack_height=0.9, interval_stack_location='below') # ratio of heights to each interval
        # stacked_epoch_layout_dict = {interval_key:dict(y_location=y_location, height=height) for interval_key, y_location, height in zip(list(spare_seq_epochs_formatting_dict.keys()), required_vertical_offsets, required_interval_heights)} # Build a stacked_epoch_layout_dict to update the display
        # stacked_epoch_layout_dict # {'LapsAll': {'y_location': -3.6363636363636367, 'height': 3.6363636363636367}, 'LapsTrain': {'y_location': -21.818181818181817, 'height': 18.18181818181818}, 'LapsTest': {'y_location': -40.0, 'height': 18.18181818181818}}
        # stacked_epoch_layout_dict = {}
        
        # replaces 'y_location', 'position' for each dict:
        # spare_seq_epochs_formatting_dict = {k:(v|stacked_epoch_layout_dict[k]) for k, v in spare_seq_epochs_formatting_dict.items()}
        spare_seq_epochs_formatting_dict = {k:v for k, v in spare_seq_epochs_formatting_dict.items()}
        
        
        ## INPUTS: train_test_split_laps_dfs_dict
        spare_seq_dfs_dict = {k:TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(df=v, required_columns_synonym_dict=IntervalsDatasource._time_column_name_synonyms) for k, v in spare_seq_dfs_dict.items()}

        ## Build interval datasources for them:
        spare_seq_dfs_datasources_dict = {k:General2DRenderTimeEpochs.build_render_time_epochs_datasource(v) for k, v in spare_seq_dfs_dict.items()}
        ## INPUTS: active_2d_plot, train_test_split_laps_epochs_formatting_dict, train_test_split_laps_dfs_datasources_dict
        assert len(spare_seq_epochs_formatting_dict) == len(spare_seq_dfs_datasources_dict)
        for k, an_interval_ds in spare_seq_dfs_datasources_dict.items():
            an_interval_ds.update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, **(spare_seq_epochs_formatting_dict[k] | kwargs)))

        ## Full output: train_test_split_laps_dfs_datasources_dict

        # actually add the epochs:
        for k, an_interval_ds in spare_seq_dfs_datasources_dict.items():
            active_2d_plot.add_rendered_intervals(an_interval_ds, name=f'{k}', debug_print=False) # adds the interval

        return spare_seq_dfs_datasources_dict, spare_seq_dfs_dict




@function_attributes(short_name=None, tags=['UNFINISHED', 'median', 'dominant'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-07-29 17:41', related_items=[])
def recompute_dominant_cells_from_median(across_session_inst_fr_computation_dict: Dict[IdentifyingContext, InstantaneousSpikeRateGroupsComputation]) -> Dict[IdentifyingContext, InstantaneousSpikeRateGroupsComputation]:
    """ Usage:
    
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import recompute_dominant_cells_from_median
    _OUT_UPDATED_across_session_inst_fr_computation_dict, df_combined, session_cell_exclusivity_annotations = recompute_dominant_cells_from_median(across_session_inst_fr_computation_dict=across_session_inst_fr_computation_dict)
    
    ## Print for UserAnnotations
    print(',\n'.join([': '.join([k.get_initialization_code_string(), str(v)]) for k, v in session_cell_exclusivity_annotations.items()]))


    """
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import SingleBarResult, InstantaneousSpikeRateGroupsComputation, SpikeRateTrends
    import nptyping as ND
    from nptyping import NDArray
    from neuropy.core.user_annotations import SessionCellExclusivityRecord

    # an_out: InstantaneousSpikeRateGroupsComputation = across_session_inst_fr_computation_dict[IdentifyingContext(format_name= 'kdiba', animal= 'gor01', exper_name= 'one', session_name= '2006-6-08_14-26-15')]

    _OUT_UPDATED_across_session_inst_fr_computation_dict = deepcopy(across_session_inst_fr_computation_dict)
    session_cell_exclusivity_annotations: Dict[IdentifyingContext, SessionCellExclusivityRecord] = {}
    
    _updated_dfs = []
    # for a_ctxt, an_out in across_session_inst_fr_computation_dict.items():
    for a_ctxt, an_out in _OUT_UPDATED_across_session_inst_fr_computation_dict.items():
        theta_trends_list: List[SpikeRateTrends] = [an_out.AnyC_ThetaDeltaMinus, an_out.AnyC_ThetaDeltaPlus]

        _theta_trends_cell_medians = []
        for a_named_epochs_period_result in theta_trends_list: 
            # SpikeRateTrends = an_out.AnyC_ThetaDeltaMinus
            epoch_agg_inst_fr_list: NDArray[ND.Shape["N_EPOCHS, N_CELLS"], Any] = deepcopy(a_named_epochs_period_result.epoch_agg_inst_fr_list) #.shape (40, 66) - (n_epochs, n_neurons)
            cell_agg_inst_fr_list: NDArray[ND.Shape["N_CELLS"], Any] = np.median(epoch_agg_inst_fr_list, axis=0) ## Aggregation function applied here -- MEDIAN
            # a_named_epochs_period_result.epoch_agg_inst_fr_list = deepcopy(epoch_agg_inst_fr_list)
            a_named_epochs_period_result.cell_agg_inst_fr_list = deepcopy(cell_agg_inst_fr_list) ## OVERRIDE WITH NEW ONE -- UPDATES
            
            _theta_trends_cell_medians.append(cell_agg_inst_fr_list)

        _theta_trends_cell_medians = np.vstack(_theta_trends_cell_medians).T # (66, 2) - (n_neurons, 2)
        _theta_trends_cell_medians_df: pd.DataFrame = pd.DataFrame(_theta_trends_cell_medians, columns=['ThetaDeltaMinus', 'ThetaDeltaPlus'])
        _theta_trends_cell_medians_df['median_diff'] = _theta_trends_cell_medians_df['ThetaDeltaPlus'] - _theta_trends_cell_medians_df['ThetaDeltaMinus'] 
        _theta_trends_cell_medians_df['median_diff_idx'] = _theta_trends_cell_medians_df['median_diff'] / (_theta_trends_cell_medians_df['ThetaDeltaMinus'] + _theta_trends_cell_medians_df['ThetaDeltaPlus'])

        ## Add the cell identity columns:
        _theta_trends_cell_medians_df['aclu'] = deepcopy(an_out.AnyC_aclus)


        ## Find LxC-like properties

        dominant_cell_threshold: float = 0.8

        _theta_trends_cell_medians_df['is_LdC'] = _theta_trends_cell_medians_df['median_diff_idx'] <= (-dominant_cell_threshold)
        _theta_trends_cell_medians_df['is_SdC'] = _theta_trends_cell_medians_df['median_diff_idx'] >= dominant_cell_threshold
        
        # ## Find long and short dominant cells:
        _theta_trends_cell_medians_df[_theta_trends_cell_medians_df['is_LdC']]
        _theta_trends_cell_medians_df[_theta_trends_cell_medians_df['is_SdC']]

        ## UPDATE:
        an_out.LxC_aclus = np.array(_theta_trends_cell_medians_df[_theta_trends_cell_medians_df['is_LdC']]['aclu'].tolist())
        an_out.SxC_aclus = np.array(_theta_trends_cell_medians_df[_theta_trends_cell_medians_df['is_SdC']]['aclu'].tolist())

        session_cell_exclusivity_annotations[a_ctxt] = SessionCellExclusivityRecord(
            LxC=deepcopy(an_out.LxC_aclus).tolist(),
            SxC=deepcopy(an_out.SxC_aclus).tolist(),
            # Others=deepcopy(an_out.AnyC_aclus).tolist(),
        )
        
        # # LxC: `long_exclusive.track_exclusive_aclus`
        # # ThetaDeltaMinus: `long_laps`
        # an_out.LxC_ThetaDeltaMinus = an_out.AnyC_ThetaDeltaMinus.get_by_id(an_out.LxC_aclus) # = SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=long_laps, included_neuron_ids=self.LxC_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds, epoch_handling_mode=epoch_handling_mode, **kwargs)
        # # ThetaDeltaPlus: `short_laps`
        # an_out.LxC_ThetaDeltaPlus = an_out.AnyC_ThetaDeltaPlus.get_by_id(an_out.LxC_aclus) # SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=short_laps, included_neuron_ids=self.LxC_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds, epoch_handling_mode=epoch_handling_mode, **kwargs)

        # # SxC: `short_exclusive.track_exclusive_aclus`
        # # ThetaDeltaMinus: `long_laps`
        # an_out.SxC_ThetaDeltaMinus = an_out.AnyC_ThetaDeltaMinus.get_by_id(an_out.SxC_aclus) # SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=long_laps, included_neuron_ids=self.SxC_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds, epoch_handling_mode=epoch_handling_mode, **kwargs)
        # # ThetaDeltaPlus: `short_laps`
        # an_out.SxC_ThetaDeltaPlus = an_out.AnyC_ThetaDeltaPlus.get_by_id(an_out.SxC_aclus) # SpikeRateTrends.init_from_spikes_and_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=short_laps, included_neuron_ids=self.SxC_aclus, instantaneous_time_bin_size_seconds=self.instantaneous_time_bin_size_seconds, epoch_handling_mode=epoch_handling_mode, **kwargs)

        if an_out.active_identifying_session_ctx is not None:
            ## Add the extended neuron identifiers (like the global neuron_uid, session_uid) columns
            _theta_trends_cell_medians_df = _theta_trends_cell_medians_df.neuron_identity.make_neuron_indexed_df_global(an_out.active_identifying_session_ctx, add_expanded_session_context_keys=True, add_extended_aclu_identity_columns=True)
            
        ## OUTPUTS: _theta_trends_cell_medians_df
        _updated_dfs.append(_theta_trends_cell_medians_df)

    _updated_dfs

    ## Merge the resultant dfs
    # Concatenate the two dataframes
    df_combined = pd.concat(_updated_dfs, ignore_index=True)
    ## Drop duplicates, keeping the first (corresponding to the SxC/LxC over the AnyC, although all the values are the same so only the 'active_set_membership' column would need to be changed): 
    df_combined = df_combined.drop_duplicates(subset=['neuron_uid'], keep='first', inplace=False)
    ## Add extra columns:
    # df_combined['inst_time_bin_seconds'] = float(self.instantaneous_time_bin_size_seconds)
    # df_combined

    # ['neuron_uid']

    ## Find long and short dominant cells:
    df_combined[df_combined['is_LdC']]
    df_combined[df_combined['is_SdC']]
    
    return (_OUT_UPDATED_across_session_inst_fr_computation_dict, df_combined, session_cell_exclusivity_annotations)



@metadata_attributes(short_name=None, tags=['heuristic', 'continuous-heuristic', 'not-yet-working'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-07-29 00:40', related_items=[])
class ContinuousHeuristicScoring:
    """ 
        Most recent functions to attempt to do continuous/non-PBE period sequence detection via a version of my heuristic decoder

    Usage:


        import numpy as np
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import ContinuousHeuristicScoring

        # Example usage

        # a_result.most_likely_position_indicies_list[0].shape # (2, 69488)
        # np.sum(p_x_given_n, axis=1) ## sum over 4 decoders to find most likely

        _decoder_prob_sum_over_all_positions = np.nansum(p_x_given_n, axis=0) ## sum over all positions (4, 69488)
        most_likely_decoder_idxs: NDArray = np.argmax(_decoder_prob_sum_over_all_positions, axis=0) ## find the max decoder idx for each time bins (4, 69488) - (n_time_bins, )
        # most_likely_decoder_idxs # .shape
        # p_x_given_n.shape # p_x_given_n.shape (59, 4, 69488)
        # p_x_given_n[most_likely_decoder_idxs, :]
        most_likely_pos_idxs: NDArray = np.argmax(p_x_given_n, axis=0) ## find the max position bins (4, 69488) - (n_decoders, n_time_bins)
        a_most_likely_pos_idxs = most_likely_pos_idxs[0,:] ## single decoder result .shape (n_time_bins)
        # a_most_likely_pos_idxs

        # pos_bin_edges: NDArray = deepcopy(xbin)
        # track_templates: TrackTemplates
        use_bin_units_instead_of_realworld:bool=False
        # max_ignore_bins:float = 2 ## for PBEs
        max_ignore_bins:float = 9 ## for Laps
        # 5 skip bins
        # same_thresh_cm: float = 6.0
        same_thresh_cm: float = 60.0
        # max_jump_distance_cm: float = 60.0
        max_jump_distance_cm: float = 200.0
        debug_print=False

        # pos_bounds = [np.min([track_templates.long_LR_decoder.xbin, track_templates.short_LR_decoder.xbin]), np.max([track_templates.long_LR_decoder.xbin, track_templates.short_LR_decoder.xbin])] # [37.0773897438341, 253.98616538463315]
        num_pos_bins: int = track_templates.long_LR_decoder.n_xbin_centers
        xbin_edges: NDArray = deepcopy(track_templates.long_LR_decoder.xbin)
        decoder_track_length_dict = track_templates.get_track_length_dict()  # {'long_LR': 214.0, 'long_RL': 214.0, 'short_LR': 144.0, 'short_RL': 144.0}

        time_window_centers: NDArray = deepcopy(a_result.time_window_centers[0]) # (n_time_bins)

        if not use_bin_units_instead_of_realworld:
            a_most_likely_pos_cm = [xbin_edges[v] for v in a_most_likely_pos_idxs]
            active_most_likely_pos = deepcopy(a_most_likely_pos_cm)
        else:
            active_most_likely_pos = deepcopy(a_most_likely_pos_idxs)
            
        if isinstance(active_most_likely_pos, list):
            active_most_likely_pos = np.array(active_most_likely_pos)

        ## INPUTS: active_most_likely_pos -- original data series

        df_runs, (valid, counts), (P_diff, is_excessive_change_index, series_idx) = ContinuousHeuristicScoring.find_longest_run_of_non_excessive_diffs(active_most_likely_pos=active_most_likely_pos, time_window_centers=time_window_centers, max_jump_distance_cm=max_jump_distance_cm, min_suffix_merge_seq_n_bins=20, max_ignore_bins=0, min_prefix_merge_seq_n_bins=20)
        df_runs

        ## Add to SpikeRaster2D as intervals:
        cont_heuristics_dfs_datasources_dict, cont_heuristics_dfs_dict = ContinuousHeuristicScoring.add_spikeRaster2D_interval_rects(active_2d_plot, df_runs=df_runs)

    """
    @classmethod
    def add_spikeRaster2D_interval_rects(cls, active_2d_plot: Spike2DRaster, df_runs: pd.DataFrame, **kwargs):
        """ 
        
        cont_heuristics_dfs_datasources_dict, cont_heuristics_dfs_dict = ContinuousHeuristicScoring.add_spikeRaster2D_interval_rects(active_2d_plot, df_runs=df_runs)
        """
        
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs, inline_mkColor
        
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.EpochRenderingMixin import EpochRenderingMixin, RenderedEpochsItemsContainer
        from pyphoplacecellanalysis.General.Model.Datasources.IntervalDatasource import IntervalsDatasource
        from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol

        if 't_duration' not in df_runs.columns:
            df_runs['t_duration'] = df_runs['t_end'] - df_runs['t_start']

        ## Use the three dataframes as separate Epoch series:
        cont_heuristics_dfs_dict = {
            'ContinuousHeuristic': df_runs,
        }

        cont_heuristics_epochs_formatting_dict = {
            'ContinuousHeuristic':dict(y_location=-10.0, height=7.5, pen_color=inline_mkColor('purple', 0.8), brush_color=inline_mkColor('purple', 0.5)),
        }

        required_vertical_offsets, required_interval_heights = EpochRenderingMixin.build_stacked_epoch_layout([0.2], epoch_render_stack_height=10.0, interval_stack_location='below') # ratio of heights to each interval
        stacked_epoch_layout_dict = {interval_key:dict(y_location=y_location, height=height) for interval_key, y_location, height in zip(list(cont_heuristics_epochs_formatting_dict.keys()), required_vertical_offsets, required_interval_heights)} # Build a stacked_epoch_layout_dict to update the display
        # stacked_epoch_layout_dict # {'LapsAll': {'y_location': -3.6363636363636367, 'height': 3.6363636363636367}, 'LapsTrain': {'y_location': -21.818181818181817, 'height': 18.18181818181818}, 'LapsTest': {'y_location': -40.0, 'height': 18.18181818181818}}

        # replaces 'y_location', 'position' for each dict:
        cont_heuristics_epochs_formatting_dict = {k:(v|stacked_epoch_layout_dict[k]) for k, v in cont_heuristics_epochs_formatting_dict.items()}
        
        ## INPUTS: train_test_split_laps_dfs_dict
        cont_heuristics_dfs_dict = {k:TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(df=v, required_columns_synonym_dict=IntervalsDatasource._time_column_name_synonyms) for k, v in cont_heuristics_dfs_dict.items()}

        ## Build interval datasources for them:
        cont_heuristics_dfs_datasources_dict = {k:General2DRenderTimeEpochs.build_render_time_epochs_datasource(v) for k, v in cont_heuristics_dfs_dict.items()}
        ## INPUTS: active_2d_plot, train_test_split_laps_epochs_formatting_dict, train_test_split_laps_dfs_datasources_dict
        assert len(cont_heuristics_epochs_formatting_dict) == len(cont_heuristics_dfs_datasources_dict)
        for k, an_interval_ds in cont_heuristics_dfs_datasources_dict.items():
            an_interval_ds.update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, **(cont_heuristics_epochs_formatting_dict[k] | kwargs)))

        ## Full output: train_test_split_laps_dfs_datasources_dict

        # actually add the epochs:
        for k, an_interval_ds in cont_heuristics_dfs_datasources_dict.items():
            active_2d_plot.add_rendered_intervals(an_interval_ds, name=f'{k}', debug_print=False) # adds the interval

        return cont_heuristics_dfs_datasources_dict, cont_heuristics_dfs_dict


    @function_attributes(short_name=None, tags=['continuous-heuristic', 'replay-detection', 'PBEs', 'WORKING', 'AI'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-07-29 08:20', related_items=[])
    @classmethod
    def label_false_series_with_bridging(cls, is_excessive: np.ndarray, min_prefix_merge_seq_n_bins: int = 4, max_ignore_bins:int=2, min_suffix_merge_seq_n_bins: int = 2):
        """ 
        1. Compute the 1st-order-diff of the decoded most-likely-positions 
        2. determine violations where the change exceeds the `is_excessive`
        min_prefix_merge_seq_n_bins: minimum number of bins to consider for bridging
        
        
        # Output might look like:
        # [ 0 0 0 0  -1  0 0 0  -1 -1  1 1 ]
        # where the single True intrusion at position 4 was bridged into series 0,
        # but the double True run at 8–9 was not.

        """
        # 1) RLE helper
        def rle(arr):
            # returns (values, lengths, start_positions)
            n = arr.size
            if n == 0:
                return np.array([], bool), np.array([], int), np.array([], int)
            # find boundaries where value changes
            change_idx = np.nonzero(np.concatenate(([True], arr[1:] != arr[:-1], [True])))[0]
            lengths = np.diff(change_idx)
            starts = change_idx[:-1]
            vals = arr[starts]
            return vals, lengths, starts

        # make a copy we can mutate
        arr = is_excessive.copy()
        
        # Run initial RLE
        vals, lengths, starts = rle(arr)
        
        # 2) Find intrusion runs:
        #    A run of True (vals==True, lengths<2) whose previous False-run is >=4
        #    and whose following False-run is >=2.
        intrusion_mask = np.zeros_like(arr, bool)
        for i, (v, L, st) in enumerate(zip(vals, lengths, starts)):
            if not v and L >= min_prefix_merge_seq_n_bins:              # false-run of length >=4
                # check if next is a short intrusion
                if i+1 < len(vals) and vals[i+1] and lengths[i+1] < max_ignore_bins:
                    # check the run after intrusion
                    if i+2 < len(vals) and not vals[i+2] and lengths[i+2] >= min_suffix_merge_seq_n_bins:
                        # mark that short True-run for flipping
                        intrusion_start = starts[i+1]
                        intrusion_len   = lengths[i+1]
                        intrusion_mask[intrusion_start:intrusion_start+intrusion_len] = True

        # 3) Flip only the marked “short intrusions” from True to False
        arr[intrusion_mask] = False

        # 4) Rerun RLE on the corrected Boolean array and label each False-block
        vals2, lengths2, starts2 = rle(arr)
        series_index = np.full(arr.shape, -1, dtype=int)
        series_id = 0

        for v, L, st in zip(vals2, lengths2, starts2):
            if not v:
                series_index[st:st+L] = series_id
                series_id += 1
            # if v is True, we leave series_index at -1

        return series_index


    @classmethod
    def find_longest_run_of_non_excessive_diffs(cls, active_most_likely_pos: NDArray, time_window_centers: NDArray, max_jump_distance_cm: float, min_prefix_merge_seq_n_bins: int = 4, max_ignore_bins:int=2, min_suffix_merge_seq_n_bins: int = 2):
        """ 
        
        1. Compute the 1st-order-diff (`P_diff`) of the decoded most-likely-positions (`active_most_likely_pos`)
        2. determine violations where the change exceeds the `is_excessive`
        min_prefix_merge_seq_n_bins: minimum number of bins to consider for bridging
        
        
        active_most_likely_pos


        Usage:

        ## INPUTS: active_most_likely_pos -- original data series

        df_runs, (valid, counts), (P_diff, is_excessive_change_index, series_idx) = ContinuousHeuristicScoring.find_longest_run_of_non_excessive_diffs(active_most_likely_pos=active_most_likely_pos, time_window_centers=time_window_centers, max_jump_distance_cm=max_jump_distance_cm, max_ignore_bins=0,
                                                                                                                                                        # min_prefix_merge_seq_n_bins=20, min_suffix_merge_seq_n_bins=20,
                                                                                                                                                        min_prefix_merge_seq_n_bins=200, min_suffix_merge_seq_n_bins=200,
                                                                                                                                                        )
        cont_heuristics_dfs_datasources_dict, cont_heuristics_dfs_dict = ContinuousHeuristicScoring.add_spikeRaster2D_interval_rects(active_2d_plot, df_runs=df_runs)

        """
        P_diff = np.diff(active_most_likely_pos) # (min: -258, mean: , max: 258, np.nanmean(P_diff): -0.002035577375650436, np.nanmedian(P_diff): 0.0, np.nanstd(P_diff): 122.22033112262098
        is_excessive_change_index = (np.abs(P_diff) > max_jump_distance_cm)
        # is_excessive_change_index # array([False, False, False, ...,  True, False, False])

        # 1) label runs with your function
        series_idx = ContinuousHeuristicScoring.label_false_series_with_bridging(is_excessive_change_index, min_prefix_merge_seq_n_bins=min_prefix_merge_seq_n_bins, max_ignore_bins=max_ignore_bins, min_suffix_merge_seq_n_bins=min_suffix_merge_seq_n_bins)

        # 2) count sizes
        valid = (series_idx >= 0)
        counts = np.bincount(series_idx[valid], minlength=series_idx.max()+1)
        ## OUTPUTS: valid, counts

        # Sorted run IDs from largest to smallest
        sorted_ids = np.argsort(counts)[::-1]

        records = []

        for rid in sorted_ids:
            diff_bins = np.where(series_idx == rid)[0]
            if diff_bins.size == 0:
                # no diffs in this run (shouldn’t usually happen)  
                continue

            # span of diffs is diff_bins.min() … diff_bins.max()
            start_diff = diff_bins.min()
            end_diff   = diff_bins.max()

            # that covers positions[start_diff] through positions[end_diff+1]
            pos_start = start_diff
            pos_end = end_diff + 1

            # slice out the actual positions
            seg = active_most_likely_pos[pos_start : pos_end+1]

            records.append({
                "run_id":       int(rid),
                "run_length":   int(counts[rid]),    # number of diffs
                "pos_start":    int(pos_start),
                "pos_end":      int(pos_end),
                't_start': float(time_window_centers[pos_start]),
                't_end': float(time_window_centers[pos_end]),
                "positions":    list(seg)            # store as list
            })

        df_runs = pd.DataFrame.from_records(records)
        return df_runs, (valid, counts), (P_diff, is_excessive_change_index, series_idx)



# ==================================================================================================================================================================================================================================================================================== #
# 2025-07-08 - Remapping Models - Not yet working                                                                                                                                                                                                                                      #
# ==================================================================================================================================================================================================================================================================================== #

@metadata_attributes(short_name=None, tags=['remapping', 'model', 'not-yet-working'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-07-08 17:40', related_items=[])
class CellFieldRemappingModels:
    """ Various methods to explain how cells remap. """
    inward_endcap_offset_magnitude: float = 15.0
    short_track_distance_bound: float = 57.0 # the amplitude of the max endcap endpoint for the short track (furthest valid point away from midpoint)
    
    scale_factor_track_body_only: float = 0.7 # (35.0/50.0)
    scale_factor_entire_track_length: float = (57.0/72.0)

    # linear_translation_wiggle_room_cm: float = 40.0 ## allow point to be anywhere within 40.0cm of expected point by translation
    linear_translation_wiggle_room_cm: float = 50.0 ## allow point to be anywhere within 40.0cm of expected point by translation

    @classmethod
    def nearest_endcap_anchored(cls, L: float, m: float) -> float:
        """ transformation between long (L) and short (S) for endcap-anchored fields """
        # Define the piecewise function using np.piecewise
        return np.piecewise(L, [(L < m), (L == m), (L > m)],           # Conditions
                        [lambda L, m: L + cls.inward_endcap_offset_magnitude, # anchored to -endcap -> (-endcap_L + 15)
                         lambda L, m: m,
                         lambda L, m: L - cls.inward_endcap_offset_magnitude, # anchored to +endcap -> (+endcap_L - 15)
                         ], m)
    
    @classmethod
    def left_endcap_anchored(cls, L: float, m: float) -> float:
        """ transformation between long (L) and short (S) for left endcap-anchored fields """
        return (L + cls.inward_endcap_offset_magnitude) # anchored to -endcap -> (-endcap_L + 15)

    @classmethod
    def right_endcap_anchored(cls, L: float, m: float) -> float:
        """ transformation between long (L) and short (S) for right endcap-anchored fields """
        return (L - cls.inward_endcap_offset_magnitude) # anchored to +endcap -> (+endcap_L - 15) 


    # @classmethod
    # def midpoint_anchored(cls, L: float, m: float) -> float:
    #     """ transformation between long (L) and short (S) for endcap-anchored fields """
    #     # Define the piecewise function using np.piecewise
    #     return L ## L is equal S
    
    @classmethod
    def room_anchored(cls, L: float, m: float) -> float:
        """ transformation between long (L) and short (S) for endcap-anchored fields """
        return np.piecewise(L, [(L >= cls.short_track_distance_bound), ((L >= m) and (L < cls.short_track_distance_bound)), ((L < m) and (L > -cls.short_track_distance_bound)), (L <= -cls.short_track_distance_bound)],           # Conditions
                        [lambda L, m: cls.short_track_distance_bound, ## has to clip to endcap
                         lambda L, m: L,
                         lambda L, m: L,
                         lambda L, m: -cls.short_track_distance_bound, ## clips to -endcap
                         ], m)

    @classmethod
    def linear_scaling_by_track_body(cls, L: float, m: float) -> float:
        """ transformation between long (L) and short (S) for endcap-anchored fields """
        return L * cls.scale_factor_track_body_only # 0.7 
    

    @classmethod
    def linear_scaling_by_entire_track(cls, L: float, m: float) -> float:
        """ transformation between long (L) and short (S) for endcap-anchored fields """
        return L * cls.scale_factor_entire_track_length
    

    

    @classmethod
    def get_model_test_fns_dict(cls) -> Dict:
        """ returns a dict containing each of the remapping models
        """
        raise NotImplementedError(f' #TODO 2025-07-23 15:47: - [ ] THIS DOES NOT FULLY WORK and is UNTESTED')
        return {'nearest_endcap_anchored':cls.nearest_endcap_anchored, 'left_endcap_anchored':cls.left_endcap_anchored, 'right_endcap_anchored':cls.right_endcap_anchored, 
                # 'midpoint_anchored':cls.midpoint_anchored,
                'room_anchored': cls.room_anchored, 
                'linear_scaling_by_track_body': cls.linear_scaling_by_track_body, 'linear_scaling_by_entire_track': cls.linear_scaling_by_entire_track}

    # ==================================================================================================================================================================================================================================================================================== #
    # END MODELS                                                                                                                                                                                                                                                                           #
    # ==================================================================================================================================================================================================================================================================================== #

    @classmethod
    def is_non_linear_remapping(cls, L: float, S: float, m: float) -> bool:
        """ Not a model, takes an L and S field peak and returns True if this remapping requires non-linear shifts. """
        LS_diff: float = S - L # S - updated, L - original
        is_non_linear: bool = False

        expected_S_by_lin_trans: float = cls.nearest_endcap_anchored(L=L, m=m)
        expected_to_observed_diff_cm: float = expected_S_by_lin_trans - S # difference between expected S and the observed one
        
        does_exceed_allowed_deviation: bool = (np.abs(expected_to_observed_diff_cm) > cls.linear_translation_wiggle_room_cm)
        if does_exceed_allowed_deviation:
            ## implies (S  exceeds the allowed deviation from the S expected by translation
            is_non_linear = True
            return is_non_linear
        
        ## First question is: does it go the opposite way of the remapping?
        if L >= m:
             ## is right of midpoint
             if (LS_diff > 0.0):
                  ## implies (S > L) -> moved ++ (to right)
                  is_non_linear = True
                  return is_non_linear ## Moved OPPOSITE DIRECTION to expected translation
        else:
             ## is left of midpoint
             if (LS_diff < 0.0):
                  ## implies (L > S) -> moved -- (to left)
                  is_non_linear = True
                  return is_non_linear ## Moved OPPOSITE DIRECTION to expected translation
             
        # ## Second question: does it greatly exceed the contraction distance? (it shouldn't):
        # contraction_amount_cm: float = 15.0
        # wiggle_room_factor: float = 0.2
        # if np.abs(LS_diff) > 40.0: #(contraction_amount_cm + (contraction_amount_cm * wiggle_room_factor)):
        #     is_non_linear = True
        #     return is_non_linear ## point moved much further than the translation would expect
        
        return is_non_linear ## likely False


    @function_attributes(short_name=None, tags=['main', 'FIXUP', 'has_considerable_remapping'], input_requires=[], output_provides=[], uses=['cls.is_non_linear_remapping'], used_by=[], creation_date='2025-07-08 17:47', related_items=[])
    @classmethod
    def fix_has_considerable_remapping_column(cls, all_neuron_stats_table: pd.DataFrame, x_mid: float = 143.8837168675656) -> pd.DataFrame:
        """ The 'has_considerable_remapping' as currently computed is wrong. This implements an improved computation for this column.

        NOTE: Independent/unrelated to `cls.main_evaluate_remapping_models`
        , L_peak_col_name: str='long_LR_pf1D_peak', L_peak_col_name: str='short_LR_pf1D_peak'
        
        active_scatter_all_neuron_stats_table = CellFieldRemappingModels.fix_has_considerable_remapping_column(all_neuron_stats_table=active_scatter_all_neuron_stats_table)
        
        """
        active_scatter_all_neuron_stats_table: pd.DataFrame = deepcopy(all_neuron_stats_table).fillna(value=np.nan, inplace=False) ## fill all Pandas.NA values with np.nan so they can be correctly cast to floats
        # active_scatter_all_neuron_stats_table

        if ('_BAK_has_considerable_remapping' not in active_scatter_all_neuron_stats_table):
            active_scatter_all_neuron_stats_table['_BAK_has_considerable_remapping'] = deepcopy(active_scatter_all_neuron_stats_table['has_considerable_remapping'])
        else:
            print(f'WARN: active_scatter_all_neuron_stats_table already contains the "_BAK_has_considerable_remapping", a backup has already been made and will not be overwritten!')

        active_scatter_all_neuron_stats_table['has_considerable_remapping'] = False ## start with False

        L_peaks = active_scatter_all_neuron_stats_table['long_LR_pf1D_peak'].to_numpy()
        S_peaks = active_scatter_all_neuron_stats_table['short_LR_pf1D_peak'].to_numpy()
        active_scatter_all_neuron_stats_table['has_considerable_remapping'] = np.logical_or(active_scatter_all_neuron_stats_table['has_considerable_remapping'], [CellFieldRemappingModels.is_non_linear_remapping(L, S, x_mid) for L, S in zip(L_peaks, S_peaks)])

        L_peaks = active_scatter_all_neuron_stats_table['long_RL_pf1D_peak'].to_numpy()
        S_peaks = active_scatter_all_neuron_stats_table['short_RL_pf1D_peak'].to_numpy()
        active_scatter_all_neuron_stats_table['has_considerable_remapping'] = np.logical_or(active_scatter_all_neuron_stats_table['has_considerable_remapping'], [CellFieldRemappingModels.is_non_linear_remapping(L, S, x_mid) for L, S in zip(L_peaks, S_peaks)])
        return active_scatter_all_neuron_stats_table




    @classmethod
    def main_evaluate_remapping_models(cls, active_scatter_all_neuron_stats_table: pd.DataFrame, x_mid: float = 143.8837168675656):
        """ Evaluate each of the models for the remapping cells in `active_scatter_all_neuron_stats_table`

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import CellFieldRemappingModels

        model_errors, best_model_name = CellFieldRemappingModels.main_test_models(active_scatter_all_neuron_stats_table=active_scatter_all_neuron_stats_table)
        
        """
        from sklearn.metrics import mean_squared_error

        _model_test_fns_dict = cls.get_model_test_fns_dict() # {'endcap_anchored':CellFieldRemappingModels.endcap_anchored, 'midpoint_anchored':CellFieldRemappingModels.midpoint_anchored, 'room_anchored': CellFieldRemappingModels.room_anchored, 'linear_scaling': CellFieldRemappingModels.linear_scaling}

        L_peaks = active_scatter_all_neuron_stats_table['long_pf_peak_x'].to_numpy()
        S_peaks_dict = {}

        for k, a_fn in _model_test_fns_dict.items():
            S_peaks = [a_fn(L, x_mid) for L in L_peaks]
            S_peaks_dict[k] = np.array(S_peaks)

        S_peaks_dict
        # --- New code for model evaluation ---

        # 1. Get both the input (L) and the measured output (S) values
        # Ensure there are no NaN values in the columns you're using
        valid_data = active_scatter_all_neuron_stats_table[['long_pf_peak_x', 'short_pf_peak_x']].dropna()
        L_peaks = valid_data['long_pf_peak_x'].to_numpy()
        S_peaks_measured = valid_data['short_pf_peak_x'].to_numpy()


        # 2. Calculate predictions for each model (as in your original code)
        S_peaks_predicted_dict = {}
        for k, a_fn in _model_test_fns_dict.items():
            S_peaks_predicted = np.array([a_fn(L, x_mid) for L in L_peaks])
            S_peaks_predicted_dict[k] = S_peaks_predicted


        # 3. Calculate RMSE for each model
        model_errors = {}
        for model_name, S_predicted in S_peaks_predicted_dict.items():
            # Calculate Mean Squared Error
            mse = mean_squared_error(S_peaks_measured, S_predicted)
            # Calculate Root Mean Squared Error
            rmse = np.sqrt(mse)
            model_errors[model_name] = rmse
            print(f"Model: {model_name:<20} | RMSE: {rmse:.4f}")


        # 4. Find and announce the best model
        best_model_name = min(model_errors, key=model_errors.get)
        print(f"\n🏆 The best model is '{best_model_name}' with an RMSE of {model_errors[best_model_name]:.4f}")
        return model_errors, best_model_name



# ==================================================================================================================================================================================================================================================================================== #
# Long/Short 3D Placefields                                                                                                                                                                                                                                                            #
# ==================================================================================================================================================================================================================================================================================== #
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.Interactive3dDisplayFunctions import Interactive3dDisplayFunctions
from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.InteractivePlaceCellDataExplorer import InteractivePlaceCellDataExplorer
from pyphoplacecellanalysis.General.Pipeline.Stages.Display import get_neuron_identities
from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import DataSeriesColorHelpers
from pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig import SingleNeuronPlottingExtended
from pyphocorehelpers.gui.Qt.color_helpers import ColorFormatConverter
from pyphoplacecellanalysis.PhoPositionalData.plotting.placefield import plot_placefields2D, update_plotColorsPlacefield2D
from pyphoplacecellanalysis.General.Model.Configs.NeuronPlottingParamConfig import NeuronConfigOwningMixin
from pyphocorehelpers.gui.PyVista.CascadingDynamicPlotsList import CascadingDynamicPlotsList
from neuropy.utils.mixins.binning_helpers import get_bin_centers

# curr_active_pipeline.reload_default_display_functions()
# _out = dict()

@metadata_attributes(short_name=None, tags=['3D', 'tracks', 'LS', 'NOT-FINISHED', 'pyvista'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-06-27 00:00', related_items=[])
class LongShort3DPlacefieldsHelpers:
    """ Helps plot the placefields from both the long and short track on the same 3D PyVista plotter.
    Not finished.

    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import LongShort3DPlacefieldsHelpers

    """
    @classmethod
    def _build_merged_long_short_pf2D_neuron_identities(cls, long_pf2D, short_pf2D, should_preview_colors: bool=False):
        """ builds the merged colors and neuron identities for plotting both the long and short track placefields
        Usage:
            ## INPUTS: long_pf2D, short_pf2D
            (long_or_short_colors_dict, long_pf_colors, short_pf_colors), neuron_plotting_configs_dict = _build_merged_long_short_pf2D_neuron_identities(long_pf2D=long_pf2D, short_pf2D=short_pf2D)
            list(neuron_plotting_configs_dict.values())[0].qcolor
            list(neuron_plotting_configs_dict.values())[0].color
            neuron_plotting_configs_dict
        """
        # INPUTS: global_pf2D, long_pf2D, short_pf2D
        long_pf_neuron_identities, pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = get_neuron_identities(long_pf2D)
        long_pf_aclus = set([int(v.extended_identity_tuple.aclu) for v in long_pf_neuron_identities])

        short_pf_neuron_identities, pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = get_neuron_identities(short_pf2D)
        short_pf_aclus = set([int(v.extended_identity_tuple.aclu) for v in short_pf_neuron_identities])
        long_or_short_neuron_identities = long_pf_aclus.union(short_pf_aclus)
        # long_or_short_aclus = np.unique(np.array([int(v.aclu) for v in long_or_short_neuron_identities]))
        long_or_short_aclus = np.unique(np.array([int(v) for v in long_or_short_neuron_identities]))

        n_neurons: int = len(long_or_short_aclus)
        neuron_qcolors_list, neuron_colors_ndarray = DataSeriesColorHelpers.build_cell_colors(n_neurons, colormap_name='PAL-relaxed_bright', colormap_source=None, return_255_array=False)

        ## Preview the new colors
        if should_preview_colors:
            from pyphocorehelpers.gui.Jupyter.simple_widgets import render_colors
            render_colors([ColorFormatConverter.qColor_to_hexstring(a_qcolor, include_alpha=False) for a_qcolor in neuron_qcolors_list])

        # neuron_colors_ndarray.shape
        long_or_short_colors_dict = dict(zip(long_or_short_aclus, neuron_colors_ndarray.T))

        # ipcDataExplorer.params.pf_colors.shape # (4, 52)

        long_pf_colors = np.vstack([long_or_short_colors_dict[aclu] for aclu in long_pf_aclus]).T
        short_pf_colors = np.vstack([long_or_short_colors_dict[aclu] for aclu in short_pf_aclus]).T
        # long_or_short_colors_array = np.vstack(long_or_short_colors_dict.values()).T
        # long_pf_colors.shape # (50, 4)
        neuron_plotting_configs_dict: Dict[int, SingleNeuronPlottingExtended] = DataSeriesColorHelpers.build_cell_display_configs(long_or_short_aclus, neuron_qcolors_list=neuron_qcolors_list)
        
        ## OUTPUTS: long_pf_colors, short_pf_colors
        return (long_or_short_colors_dict, long_pf_colors, short_pf_colors), neuron_plotting_configs_dict



    @function_attributes(short_name=None, tags=['plot', 'long-short'], input_requires=[], output_provides=[], uses=['_build_merged_long_short_pf2D_neuron_identities'], used_by=[], creation_date='2025-06-27 00:50', related_items=[])
    @classmethod
    def _plot_long_short_placefields(cls, ipcDataExplorer, long_pf2D, short_pf2D, maze_y_offset: float = 20.0, enable_update_spikes: bool=False):
        """ 
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import LongShort3DPlacefieldsHelpers
            from pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.InteractivePlaceCellDataExplorer import InteractivePlaceCellDataExplorer

            curr_active_pipeline.reload_default_display_functions()
            _out = {}
            global_any_context = curr_active_pipeline.filtered_contexts[global_any_name]
            _out['_display_3d_interactive_tuning_curves_plotter'] = curr_active_pipeline.display(display_function='_display_3d_interactive_tuning_curves_plotter', active_session_configuration_context=global_any_context,
                                                                                                separate_window = False,
                                                                                                params_kwargs={'show_legend': False, 'should_display_placefield_points': False, 'should_nan_non_visited_elements': False, 'zScalingFactor': 500.0},
                                                                                                #  panel_controls_mode = 'Panel',
                                                                                                # panel_controls_mode = 'Qt',
                                                                                                panel_controls_mode = None,
                                                                                                ) # _display_grid_bin_bounds_validation

            ## Move the long-maze to -`maze_y_offset` units and the short-maze to +`maze_y_offset` units along the y-axis 
            ipcDataExplorer: InteractivePlaceCellDataExplorer = _out['_display_3d_interactive_tuning_curves_plotter']['ipcDataExplorer']
            pActiveTuningCurvesPlotter = _out['_display_3d_interactive_tuning_curves_plotter']['plotter']
            pane = _out['_display_3d_interactive_tuning_curves_plotter']['pane']
            pane = LongShort3DPlacefieldsHelpers._plot_long_short_placefields(ipcDataExplorer=ipcDataExplorer, long_pf2D=long_pf2D, short_pf2D=short_pf2D)

        """
        def _subfn_scale_ybin_centers_to_track_width(ybin: NDArray, track_y_center: float = 0.0, track_y_width: float = 22.0) -> NDArray:
            data_y_range: float = np.ptp(ybin)
            data_y_center_offset: float = ybin[0] + (data_y_range / 2.0)

            _adjusted_ybin_centers = deepcopy(ybin)

            if track_y_width is not None:
                # First: center the data around 0
                _adjusted_ybin_centers = _adjusted_ybin_centers - data_y_center_offset
                # Then: scale to desired width
                _adjusted_ybin_centers = (_adjusted_ybin_centers / data_y_range) * float(track_y_width)
                # Finally: offset to desired center
                _adjusted_ybin_centers = _adjusted_ybin_centers + track_y_center
            else:
                # Just center without scaling
                _adjusted_ybin_centers = _adjusted_ybin_centers - data_y_center_offset + track_y_center

            return _adjusted_ybin_centers


        # ==================================================================================================================================================================================================================================================================================== #
        # Begin Function Body                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #


        ## Remove old placefields:
        ipcDataExplorer.remove_all_rendered_placefields()
        
        ## Adjust mazes
        long_maze_bg = ipcDataExplorer.long_maze_bg
        short_maze_bg = ipcDataExplorer.short_maze_bg

        ipcDataExplorer.params.maze_y_offset = maze_y_offset

        ## normal (-, +) offsets        
        long_y_offset: float = -maze_y_offset
        short_y_offset: float = maze_y_offset
        
        # ## positive-only offsets:
        # long_y_offset: float = 0.0
        # short_y_offset: float = maze_y_offset        

        ipcDataExplorer.params.long_y_offset = long_y_offset
        ipcDataExplorer.params.short_y_offset = short_y_offset
        
        long_maze_bg.SetPosition(0.0, ipcDataExplorer.params.long_y_offset, 0.0)
        short_maze_bg.SetPosition(0.0, ipcDataExplorer.params.short_y_offset, 0.0)
        
        long_pf2D = deepcopy(long_pf2D)
        short_pf2D = deepcopy(short_pf2D)

        # long_adjusted_ybin_centers = _subfn_scale_ybin_centers_to_track_width(ybin_centers=long_pf2D.ratemap.ybin_centers, track_y_center=-maze_y_offset)
        # short_adjusted_ybin_centers = _subfn_scale_ybin_centers_to_track_width(ybin_centers=short_pf2D.ratemap.ybin_centers, track_y_center=maze_y_offset)
        print(f'long_pf2D.ratemap.ybin: {long_pf2D.ratemap.ybin}')
        print(f'short_pf2D.ratemap.ybin: {short_pf2D.ratemap.ybin}')
        # long_adjusted_ybin = _subfn_scale_ybin_centers_to_track_width(ybin=long_pf2D.ratemap.ybin, track_y_center=-ipcDataExplorer.params.long_y_offset)
        # short_adjusted_ybin = _subfn_scale_ybin_centers_to_track_width(ybin=short_pf2D.ratemap.ybin, track_y_center=-ipcDataExplorer.params.short_y_offset)
        long_adjusted_ybin = _subfn_scale_ybin_centers_to_track_width(ybin=long_pf2D.ratemap.ybin, track_y_center=0.0)
        short_adjusted_ybin = _subfn_scale_ybin_centers_to_track_width(ybin=short_pf2D.ratemap.ybin, track_y_center=0.0)
        # long_adjusted_ybin = _subfn_scale_ybin_centers_to_track_width(ybin=long_pf2D.ratemap.ybin, track_y_center=-11.0)
        # short_adjusted_ybin = _subfn_scale_ybin_centers_to_track_width(ybin=short_pf2D.ratemap.ybin, track_y_center=-11.0)

        # long_adjusted_ybin = _subfn_scale_ybin_centers_to_track_width(ybin=long_pf2D.ratemap.ybin, track_y_center=long_pf2D.ratemap.ybin[0])
        # short_adjusted_ybin = _subfn_scale_ybin_centers_to_track_width(ybin=short_pf2D.ratemap.ybin, track_y_center=short_pf2D.ratemap.ybin[0])


        print(f'long_adjusted_ybin: {long_adjusted_ybin}')
        print(f'short_adjusted_ybin: {short_adjusted_ybin}')
        
        long_pf2D.ratemap.ybin = long_adjusted_ybin
        short_pf2D.ratemap.ybin = short_adjusted_ybin
        
        # long_pf2D.ratemap.ybin_centers = get_bin_centers(long_adjusted_ybin)
        # short_pf2D.ratemap.ybin_centers = get_bin_centers(short_adjusted_ybin)
        
        ## INPUTS: long_pf2D, short_pf2D
        (long_or_short_colors_dict, long_pf_colors, short_pf_colors), neuron_plotting_configs_dict = cls._build_merged_long_short_pf2D_neuron_identities(long_pf2D=long_pf2D, short_pf2D=short_pf2D)

        ## OUTPUTS: long_pf_colors, short_pf_colors, neuron_plotting_configs_dict
        # ipcDataExplorer.params.zScalingFactor = 500.0
        ipcDataExplorer.params.zScalingFactor = 2000.0
        ipcDataExplorer.params.show_legend = False
        ipcDataExplorer.params.should_display_placefield_points = False
        # ipcDataExplorer.params.should_display_placefield_points = True

        ipcDataExplorer.params.should_nan_non_visited_elements = False
        # ipcDataExplorer.params.should_nan_non_visited_elements = True
        
        
        # ipcDataExplorer.params.

        # {'show_legend': False, 'should_display_placefield_points': False, 'should_nan_non_visited_elements': False, 'zScalingFactor': 500.0}
        # ==================================================================================================================================================================================================================================================================================== #
        # Begin Plotting Part                                                                                                                                                                                                                                                                  #
        # ==================================================================================================================================================================================================================================================================================== #

        ## INPUTS: ipcDataExplorer, pActiveTuningCurvesPlotter
        # INPUTS: global_pf2D, long_pf2D, short_pf2D
        _temp_input_params = get_dict_subset(ipcDataExplorer.params, ['should_use_normalized_tuning_curves','should_pdf_normalize_manually','should_nan_non_visited_elements','should_force_placefield_custom_color','should_display_placefield_points', 'should_override_disable_smooth_shading', 'nan_opacity',
                                                'zScalingFactor', 'show_legend'])

        # _temp_input_params['clip_below_plane'] = 1 # 0.5 * ipcDataExplorer.params.zScalingFactor

        _temp_input_params['clip_below_plane'] = 0.6 # 0.2 * ipcDataExplorer.params.zScalingFactor
        _temp_input_params['clip_below_plane']

        # ipcDataExplorer.p, ipcDataExplorer.plots['tuningCurvePlotActors'], ipcDataExplorer.plots_data['tuningCurvePlotData'], ipcDataExplorer.plots['tuningCurvePlotLegendActor'], temp_plots_data = plot_placefields2D(ipcDataExplorer.p, deepcopy(long_pf2D), pf_colors=ipcDataExplorer.params.pf_colors, zScalingFactor=ipcDataExplorer.params.zScalingFactor, show_legend=ipcDataExplorer.params.show_legend, series_prefix='long' **_temp_input_params) # note that the get_dict_subset(...) thing is just a safe way to get only the relevant members.
        ## INPUTS: long_pf_colors, short_pf_colors
        # ipcDataExplorer.p, ipcDataExplorer.plots['tuningCurvePlotActors'], ipcDataExplorer.plots_data['tuningCurvePlotData'], ipcDataExplorer.plots['tuningCurvePlotLegendActor'], temp_plots_data = plot_placefields2D(ipcDataExplorer.p, active_placefields=deepcopy(long_pf2D), pf_colors=long_pf_colors, zScalingFactor=ipcDataExplorer.params.zScalingFactor, show_legend=ipcDataExplorer.params.show_legend, series_prefix='long' **_temp_input_params) # note that the get_dict_subset(...) thing is just a safe way to get only the relevant members.

        ## Long Track pfs:
        # clip_bounds = long_maze_bg.GetBounds()
        clip_bounds = None
        _long_outs = plot_placefields2D(ipcDataExplorer.p, active_placefields=deepcopy(long_pf2D), pf_colors=long_pf_colors, series_prefix='long', clip_bounds=clip_bounds, **_temp_input_params)
        p, long_tuningCurvePlotActors, long_tuningCurvePlotData, long_tuningCurvePlotLegendActor, long_temp_plots_data = _long_outs
        for k, a_nested_actors_dict in long_tuningCurvePlotActors.items():
            # print(f'k: {k}, v: {v}')
            for a_subactor_key, a_subactor in a_nested_actors_dict.items():
                if a_subactor is not None:
                    a_subactor.SetPosition(0.0, ipcDataExplorer.params.long_y_offset, 0.0) ## long offset
                else:
                    # print(f'[{k}][{a_subactor_key}] is None!')
                    pass

        ## Short Track pfs:
        # clip_bounds = short_maze_bg.GetBounds()
        clip_bounds = None
        _short_outs = plot_placefields2D(ipcDataExplorer.p, active_placefields=deepcopy(short_pf2D), pf_colors=short_pf_colors, series_prefix='short', clip_bounds=clip_bounds, **_temp_input_params) 
        p, short_tuningCurvePlotActors, short_tuningCurvePlotData, short_tuningCurvePlotLegendActor, short_temp_plots_data = _short_outs
        for k, a_nested_actors_dict in short_tuningCurvePlotActors.items():
            # print(f'k: {k}, v: {v}')
            for a_subactor_key, a_subactor in a_nested_actors_dict.items():
                if a_subactor is not None:
                    a_subactor.SetPosition(0.0, ipcDataExplorer.params.short_y_offset, 0.0) ## short offset
                else:
                    # print(f'[{k}][{a_subactor_key}] is None!')
                    pass
                


        #TODO 2025-06-27 08:13: - [ ] hardcoded track offsets to re-align appropriately
        assert (maze_y_offset==20.0), f"2025-06-27 08:13: hardcoded track offsets only work to re-allign with `maze_y_offset=20.0`"
        long_maze_bg.SetPosition(0.0, -163.0, 0.0)
        short_maze_bg.SetPosition(0.0, -123.0, 0.0)
        
        # ==================================================================================================================================================================================================================================================================================== #
        # Update Combined Variables                                                                                                                                                                                                                                                            #
        # ==================================================================================================================================================================================================================================================================================== #
        ## Combine long/short entries:
        combined_paired_dict = {}
        combined_tuningCurvePlotActors = {}
        combined_tuningCurvePlotData = {}
        combined_temp_plots_data = {}

        is_flat_key_mode: bool = False
        ## INPUTS: long_tuningCurvePlotActors, long_tuningCurvePlotData, long_temp_plots_data
        ## INPUTS: short_tuningCurvePlotActors, short_tuningCurvePlotData, short_temp_plots_data

        ## INPUTS: long_temp_plots_data, long_temp_plots_data

        ## Plots Actors:
        for k, a_nested_actors_dict in long_tuningCurvePlotActors.items():
            # print(f'k: {k}, v: {a_nested_actors_dict}')
            combined_key: str = f'long_{k}'
            combined_paired_dict[k] = {'long': combined_key, 'short': None}
                
            if is_flat_key_mode:
                combined_tuningCurvePlotActors[combined_key] = a_nested_actors_dict
            else:
                combined_tuningCurvePlotActors[k] = {'long': a_nested_actors_dict}
                a_short_nested_actors_dict = short_tuningCurvePlotActors.get(k, None)
                if a_short_nested_actors_dict is not None:
                    combined_tuningCurvePlotActors[k]['short'] = a_short_nested_actors_dict

        for k, a_nested_actors_dict in short_tuningCurvePlotActors.items():
            # print(f'k: {k}, v: {a_nested_actors_dict}')
            combined_key: str = f'short_{k}'
            if k not in combined_paired_dict:
                combined_paired_dict[k] = {'long': None, 'short': combined_key} ## create a new entry
            else:
                ## already in there    
                # combined_paired_dict.setdefault(k, {'long': None, 'short': a_nested_actors_dict})
                combined_paired_dict[k]['short'] = combined_key
                
            if is_flat_key_mode:
                combined_tuningCurvePlotActors[combined_key] = a_nested_actors_dict
            else:
                ## handle keys only in short
                if (k not in combined_tuningCurvePlotActors) and (a_nested_actors_dict is not None):
                    combined_tuningCurvePlotActors[k] = {'short': a_nested_actors_dict}


        combined_tuningCurvePlotActors = {k:CascadingDynamicPlotsList(**v) for k, v in combined_tuningCurvePlotActors.items()} ## convert each child to a CascadingDynamicPlotsList

        ## Plots Data:
        ## INPUTS: long_tuningCurvePlotData, short_tuningCurvePlotData

        combined_tuningCurvePlotData = {}
        for k, a_nested_plots_data in long_tuningCurvePlotData.items():
            # print(f'k: {k}, v: {a_nested_actors_dict}')
            combined_key: str = f'long_{k}'
            if is_flat_key_mode:
                combined_tuningCurvePlotData[combined_key] = a_nested_plots_data
            else:
                combined_tuningCurvePlotData[k] = {'long': a_nested_plots_data}
                a_short_data = short_tuningCurvePlotData.get(k, None)
                if a_short_data is not None:
                    combined_tuningCurvePlotData[k]['short'] = a_short_data

        for k, a_nested_plots_data in short_tuningCurvePlotData.items():
            # print(f'k: {k}, v: {a_nested_actors_dict}')
            combined_key: str = f'short_{k}'
            if is_flat_key_mode:
                combined_tuningCurvePlotData[combined_key] = a_nested_plots_data
            else:
                ## handle keys only in short
                if (k not in combined_tuningCurvePlotData) and (a_nested_plots_data is not None):
                    combined_tuningCurvePlotData[k] = {'short': a_nested_plots_data}


        ## Other Data:
        ## INPUTS: long_temp_plots_data, short_temp_plots_data
        combined_temp_plots_data = {}
        for k, v in long_temp_plots_data.items():
            combined_temp_plots_data[k] = list(v)

        for k, v in short_temp_plots_data.items():
            if k in combined_temp_plots_data:
                combined_temp_plots_data[k].extend(v)
            else:
                print(F'WARN: k: {k} missing from combined_temp_plots_data. combined_temp_plots_data.keys(): {list(combined_temp_plots_data.keys())}')
                combined_temp_plots_data[k] = v
            

        ## Update spikes:
        if enable_update_spikes:
            combined_spikes_df: pd.DataFrame = pd.concat([long_pf2D.spikes_df, short_pf2D.spikes_df], axis='index', ignore_index=True).drop_duplicates(subset=['t_rel_seconds'])
            ipcDataExplorer._spikes_df = combined_spikes_df
        
        ## OUTPUTS: combined_paired_dict, combined_tuningCurvePlotActors, combined_tuningCurvePlotData, combined_temp_plots_data, combined_spikes_df

        ipcDataExplorer.params.combined_paired_dict = combined_paired_dict
        ipcDataExplorer.plots['tuningCurvePlotActors'] = combined_tuningCurvePlotActors
        ipcDataExplorer.plots_data['tuningCurvePlotData'] = combined_tuningCurvePlotData
        # ipcDataExplorer.plots['tuningCurvePlotLegendActor'] = combined_tuningCurvePlotActors

        # Build the widget labels:
        ipcDataExplorer.params.unit_labels = combined_temp_plots_data['unit_labels'] # fetch the unit labels from the extra data dict.
        ipcDataExplorer.params.pf_fragile_linear_neuron_IDXs = combined_temp_plots_data['good_placefield_neuronIDs'] # fetch the unit labels from the extra data dict.
        ## Legend data:
        ipcDataExplorer.plots_data['tuningCurvePlotLegendData'] = combined_temp_plots_data['legend_entries']

        ## build combined legend
        if ipcDataExplorer.params.show_legend:
            ipcDataExplorer.plots['tuningCurvePlotLegendActor'] = ipcDataExplorer.p.add_legend(ipcDataExplorer.plots_data['tuningCurvePlotLegendData'], name='tuningCurvesLegend', 
                                bcolor=(0.05, 0.05, 0.05), border=True,
                                loc='center right', size=[0.05, 0.85]) # vtk.vtkLegendBoxActor


        ## Update: `self.params.pf_active_configs`
        ## INPUTS: neuron_plotting_configs_dict
        # ipcDataExplorer.params.pf_active_configs ## replace
        ipcDataExplorer.params.pf_active_configs = list(neuron_plotting_configs_dict.values())
        # ipcDataExplorer.active_neuron_render_configs = 
        ipcDataExplorer.active_neuron_render_configs_map = NeuronConfigOwningMixin._build_id_index_configs_dict(ipcDataExplorer.active_neuron_render_configs)
        # n_cells: int = len(ipcDataExplorer.active_neuron_render_configs_map)
        n_cells: int = len(ipcDataExplorer.params.pf_active_configs)
        
        ## UPDATES: ipcDataExplorer.params.neuron_colors, ipcDataExplorer.params.neuron_colors_hex
        print(f'n_cells: {n_cells}')
        
        # allocate new neuron_colors array:
        # ipcDataExplorer.params.neuron_colors = np.zeros((4, n_cells))
        # for i, curr_config in enumerate(ipcDataExplorer.params.pf_active_configs):
        #     curr_qcolor = curr_config.qcolor
        #     curr_color = curr_qcolor.getRgbF() # (1.0, 0.0, 0.0, 0.5019607843137255)
        #     ipcDataExplorer.params.neuron_colors[:, i] = curr_color[:]
        #     # self.params.neuron_colors[:, i] = curr_color[:]
        
        long_or_short_colors_array = np.vstack(long_or_short_colors_dict.values()).T
        print(f'np.shape(long_or_short_colors_array): {np.shape(long_or_short_colors_array)}')
        
        ipcDataExplorer.params.neuron_colors = long_or_short_colors_array
        ipcDataExplorer.params.neuron_colors_hex = [v.color for v in list(neuron_plotting_configs_dict.values())]

        ## Called to setup spikes
        if enable_update_spikes:
            ipcDataExplorer.setup_spike_rendering_mixin()
            # ipcDataExplorer._build_flat_color_data()
            # ipcDataExplorer.plot_spikes()
            
        
        # ==================================================================================================================================================================================================================================================================================== #
        # Post-update rebuilding widgets and such:                                                                                                                                                                                                                                             #
        # ==================================================================================================================================================================================================================================================================================== #
        # Update the ipcDataExplorer's colors for spikes and placefields from its configs on init:
        defer_update: bool = False
        # defer_update: bool = True # defer_update=True to prevent self.update_spikes(...) from being erroniously called
        # ipcDataExplorer.on_config_update({neuron_id:a_config.color for neuron_id, a_config in ipcDataExplorer.active_neuron_render_configs_map.items()}, defer_update=False)
        ipcDataExplorer.on_config_update({neuron_id:a_config.color for neuron_id, a_config in ipcDataExplorer.active_neuron_render_configs_map.items()}, defer_update=defer_update) 
    

        ipcDataExplorer.params.panel_controls_mode = None ## override
        # ipcDataExplorer.params.panel_controls_mode = 'Qt' ## override
        ipcDataExplorer.params.should_use_separate_window = False ## override
        
        # build the output panels if desired:
        if ipcDataExplorer.params.panel_controls_mode == 'Qt':
            # Qt-based Placefield controls:
            from pyphoplacecellanalysis.GUI.Qt.PlacefieldVisualSelectionControls.qt_placefield import build_all_placefield_output_panels
            from pyphocorehelpers.gui.Qt.widget_positioning_helpers import WidgetPositioningHelpers, DesiredWidgetLocation, WidgetGeometryInfo
            
            ## try to remove extant controls
            placefieldControlsContainerWidget = ipcDataExplorer.ui.pop('placefieldControlsContainerWidget', None)
            if placefieldControlsContainerWidget is not None:
                print(f'removing extant Qt controls...')
                placefieldControlsContainerWidget.close()
                print(f'done.')        

            # pane: (placefieldControlsContainerWidget, pf_widgets)
            placefieldControlsContainerWidget, pf_widgets = build_all_placefield_output_panels(ipcDataExplorer)
            placefieldControlsContainerWidget.show()
            
            # Adds the placefield controls container widget and each individual pf widget to the ipcDataExplorer.ui in case it needs to reference them later:
            ipcDataExplorer.ui['placefieldControlsContainerWidget'] = placefieldControlsContainerWidget
            
            # Visually align the widgets:
            WidgetPositioningHelpers.align_window_edges(ipcDataExplorer.p, placefieldControlsContainerWidget, relative_position = 'above', resize_to_main=(1.0, None))
            
            # Wrap:
            if not ipcDataExplorer.params.should_use_separate_window:
                from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper
                
                active_root_main_widget = ipcDataExplorer.p.window()
                root_dockAreaWindow, app = DockAreaWrapper.wrap_with_dockAreaWindow(active_root_main_widget, placefieldControlsContainerWidget, title=ipcDataExplorer.data_explorer_name)
            else:
                print(f'Skipping separate window because should_use_separate_window == True')
                root_dockAreaWindow = None
            pane = (root_dockAreaWindow, placefieldControlsContainerWidget, pf_widgets)
            
        elif ipcDataExplorer.params.panel_controls_mode == 'Panel':        
            ### Build Dynamic Panel Interactive Controls for configuring Placefields:
            # Panel library based Placefield controls
            from pyphoplacecellanalysis.GUI.Panel.panel_placefield import build_panel_interactive_placefield_visibility_controls
            pane = build_panel_interactive_placefield_visibility_controls(ipcDataExplorer)
        else:
            # no controls
            pane = None
            # pass
        
        ipcDataExplorer.p.update()
        ipcDataExplorer.p.render() 

        return pane


    @function_attributes(short_name=None, tags=['long-short', 'display', '3D', 'pf', 'peaks', 'promienence', 'ratemap'], input_requires=[], output_provides=[], uses=['_render_peak_prominence_2d_results_on_pyvista_plotter'], used_by=[], creation_date='2025-06-27 04:10', related_items=['pyphoplacecellanalysis.Pho3D.PyVista.peak_prominences.render_all_neuron_peak_prominence_2d_results_on_pyvista_plotter'])
    @classmethod
    def render_long_short_all_neuron_peak_prominence_2d_results_on_pyvista_plotter(cls, ipcDataExplorer, long_peak_prominence_2d_results, short_peak_prominence_2d_results, debug_print=False, **kwargs):
        """
        Computes the appropriate contour/peaks/rectangle/etc components for each neuron_id using the active_peak_prominence_2d_results and uses them to create new:
        Inputs:
            `ipcDataExplorer`: a valid and activate 3D Interactive Tuning Curves Plotter instance, as would be produced by calling `curr_active_pipeline.display('_display_3d_interactive_tuning_curves_plotter', ...)`
            `active_peak_prominence_2d_results`: the computed results from the 'PeakProminence2D' computation stage.
            
        Provides: 
            Modifies ipcDataExplorer's `.plots['tuningCurvePlotActors']` and `.plots_data['tuningCurvePlotActors']` properties just like endogenous ipcDataExplorer functions do.
            FOR EACH neuron_id -> active_neuron_id:
                ipcDataExplorer.plots['tuningCurvePlotActors'][active_neuron_id].peaks: a hierarchy of nested CascadingDynamicPlotsList objects
                ipcDataExplorer.plots_data['tuningCurvePlotData'][active_neuron_id]['peaks']: a series of nested-dicts with the same key hierarchy as the above peaks
            
        Usage:
        
            from pyphoplacecellanalysis.Pho3D.PyVista.peak_prominences.render_all_neuron_peak_prominence_2d_results_on_pyvista_plotter
            
            display_output = {}
            active_config_name = long_LR_name
            print(f'active_config_name: {active_config_name}')
            active_peak_prominence_2d_results = curr_active_pipeline.computation_results[active_config_name].computed_data.get('RatemapPeaksAnalysis', {}).get('PeakProminence2D', None)
            pActiveTuningCurvesPlotter = None
            display_output = display_output | curr_active_pipeline.display('_display_3d_interactive_tuning_curves_plotter', active_config_name, extant_plotter=display_output.get('pActiveTuningCurvesPlotter', None), panel_controls_mode='Qt', should_nan_non_visited_elements=False, zScalingFactor=2000.0) # Works now!
            ipcDataExplorer = display_output['ipcDataExplorer']
            display_output['pActiveTuningCurvesPlotter'] = display_output.pop('plotter') # rename the key from the generic "plotter" to "pActiveSpikesBehaviorPlotter" to avoid collisions with others
            pActiveTuningCurvesPlotter = display_output['pActiveTuningCurvesPlotter']
            root_dockAreaWindow, placefieldControlsContainerWidget, pf_widgets = display_output['pane'] # for Qt mode

            active_peak_prominence_2d_results = curr_active_pipeline.computation_results[active_config_name].computed_data.get('RatemapPeaksAnalysis', {}).get('PeakProminence2D', None)
            LongShort3DPlacefieldsHelpers.render_long_short_all_neuron_peak_prominence_2d_results_on_pyvista_plotter(ipcDataExplorer, active_peak_prominence_2d_results)
            
        """
        from pyphoplacecellanalysis.Pho3D.PyVista.peak_prominences import render_all_neuron_peak_prominence_2d_results_on_pyvista_plotter, _render_peak_prominence_2d_results_on_pyvista_plotter
        
        long_peak_prominence_2d_results_aclus = np.array(list(long_peak_prominence_2d_results.results.keys()))
        short_peak_prominence_2d_results_aclus = np.array(list(short_peak_prominence_2d_results.results.keys()))
        # active_peak_prominence_2d_results_aclus = np.array(list(long_peak_prominence_2d_results.results.keys()))
        either_peak_prominence_2d_results_aclus = np.unique([*long_peak_prominence_2d_results_aclus.tolist(), *short_peak_prominence_2d_results_aclus.tolist()])
        print(f'long_peak_prominence_2d_results_aclus: {long_peak_prominence_2d_results_aclus}')
        print(f'short_peak_prominence_2d_results_aclus: {short_peak_prominence_2d_results_aclus}')
        print(f'either_peak_prominence_2d_results_aclus: {either_peak_prominence_2d_results_aclus}')

        for active_neuron_id in ipcDataExplorer.neuron_ids:
            if debug_print:
                print(f'processing active_neuron_id: {active_neuron_id}...')
                
            _temp_neuron_id_actors = ipcDataExplorer.plots['tuningCurvePlotActors'].get(active_neuron_id, None)
            if (active_neuron_id not in ipcDataExplorer.plots['tuningCurvePlotActors']) or (_temp_neuron_id_actors is None):
                ipcDataExplorer.plots['tuningCurvePlotActors'][active_neuron_id] = CascadingDynamicPlotsList(peaks={})
            _temp_neuron_id_plot_data = ipcDataExplorer.plots_data['tuningCurvePlotData'].get(active_neuron_id, None)
            if (active_neuron_id not in ipcDataExplorer.plots_data['tuningCurvePlotData']) or (_temp_neuron_id_plot_data is None):
                ipcDataExplorer.plots_data['tuningCurvePlotData'][active_neuron_id] = {}

            # Determine if this aclu is present in the `active_peak_prominence_2d_results`
            if active_neuron_id in either_peak_prominence_2d_results_aclus:
                
                try:
                    tuning_curve_is_visible = ipcDataExplorer.plots['tuningCurvePlotActors'][active_neuron_id].main.GetVisibility() # either 0 or 1 depending on the visibility of this cell
                except (KeyError, AttributeError, ValueError) as e:
                    # AttributeError: 'NoneType' object has no attribute 'main'
                    ## get from the configs:
                    tuning_curve_is_visible: int = int(ipcDataExplorer.active_neuron_render_configs_map[active_neuron_id].isVisible)            
                except Exception as e:
                    raise e

                ## Initialize:
                ipcDataExplorer.plots['tuningCurvePlotActors'][active_neuron_id]['peaks'] = {} # sets the .peaks property of the CascadingDynamicPlotsList
                ipcDataExplorer.plots_data['tuningCurvePlotData'][active_neuron_id]['peaks'] = {}
                
                if active_neuron_id in long_peak_prominence_2d_results_aclus:
                    long_all_peaks_data, long_all_peaks_actors = _render_peak_prominence_2d_results_on_pyvista_plotter(ipcDataExplorer, active_peak_prominence_2d_results=long_peak_prominence_2d_results, valid_neuron_id=active_neuron_id, render=False, debug_print=debug_print, **kwargs)
                    ipcDataExplorer.plots['tuningCurvePlotActors'][active_neuron_id]['peaks']['long'] = long_all_peaks_actors
                    ipcDataExplorer.plots_data['tuningCurvePlotData'][active_neuron_id]['peaks']['long'] = long_all_peaks_data
                    long_all_peaks_actors.SetVisibility(tuning_curve_is_visible) # Change the visibility to match the current tuning_curve_visibility_state
                    for k, a_nested_actors_dict in long_all_peaks_actors.items():
                        # print(f'k: {k}, v: {v}')
                        for a_subactor_key, a_subactor in a_nested_actors_dict.items():
                            if a_subactor is not None:
                                a_subactor.SetPosition(0.0, ipcDataExplorer.params.long_y_offset, 0.0) ## long offset
                            else:
                                # print(f'[{k}][{a_subactor_key}] is None!')
                                pass


                if active_neuron_id in short_peak_prominence_2d_results_aclus:
                    short_all_peaks_data, short_all_peaks_actors = _render_peak_prominence_2d_results_on_pyvista_plotter(ipcDataExplorer, active_peak_prominence_2d_results=short_peak_prominence_2d_results, valid_neuron_id=active_neuron_id, render=False, debug_print=debug_print, **kwargs)
                    ipcDataExplorer.plots['tuningCurvePlotActors'][active_neuron_id]['peaks']['short'] = short_all_peaks_actors
                    ipcDataExplorer.plots_data['tuningCurvePlotData'][active_neuron_id]['peaks']['short'] = short_all_peaks_data
                    short_all_peaks_actors.SetVisibility(tuning_curve_is_visible) # Change the visibility to match the current tuning_curve_visibility_state
                    for k, a_nested_actors_dict in short_all_peaks_actors.items():
                        # print(f'k: {k}, v: {v}')
                        for a_subactor_key, a_subactor in a_nested_actors_dict.items():
                            if a_subactor is not None:
                                a_subactor.SetPosition(0.0, ipcDataExplorer.params.short_y_offset, 0.0) ## short offset
                            else:
                                # print(f'[{k}][{a_subactor_key}] is None!')
                                pass
                            

                ipcDataExplorer.plots['tuningCurvePlotActors'][active_neuron_id]['peaks'] = CascadingDynamicPlotsList(**ipcDataExplorer.plots['tuningCurvePlotActors'][active_neuron_id]['peaks'])
                ## visibility and such:                
                # ipcDataExplorer.plots['tuningCurvePlotActors'][active_neuron_id].peaks = all_peaks_actors # sets the .peaks property of the CascadingDynamicPlotsList
                # ipcDataExplorer.plots_data['tuningCurvePlotData'][active_neuron_id]['peaks'] = all_peaks_data
            else:
                # neuron_id is missing from results:
                print(f'WARN: neuron_id: {active_neuron_id} is present in ipcDataExplorer but missing from `active_peak_prominence_2d_results`!')
                # ipcDataExplorer.plots['tuningCurvePlotActors'][active_neuron_id]['peaks'] = None
                # ipcDataExplorer.plots_data['tuningCurvePlotData'][active_neuron_id]['peaks'] = None
                pass
        # END for active_neuron_id in ipcDataExplorer.neuron_i...

        # Once done, render
        ipcDataExplorer.p.render()
        
        if debug_print:
            print('done.')
            
        return ipcDataExplorer

# ==================================================================================================================================================================================================================================================================================== #
# Time Bin Categorization                                                                                                                                                                                                                                                              #
# ==================================================================================================================================================================================================================================================================================== #

import numpy as np
import itertools
from typing import List, Tuple, Dict
# from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import analyze_epoch_dynamics
# from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MeasuredVsDecodedOccupancy
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import EpochComputationFunctions, EpochComputationsComputationsContainer
from neuropy.utils.indexing_helpers import PandasHelpers
        

from enum import Enum, auto

@metadata_attributes(short_name=None, tags=['run-lengths', 'sequence-analysis', 'temporal'], input_requires=[], output_provides=[], uses=[], used_by=['WithinEpochTimeBinDynamics'], creation_date='2025-05-16 04:30', related_items=[])
class TimeBinCategorization(Enum):
    """classifies a single time bin based on its prob 
    
    
        if (self.value == self.pLONG.value):
            pass
        elif (self.value == self.pSHORT.value):
            pass
        elif (self.value == self.MIXED.value):
            pass
        else:
            raise NotImplementedError(f'{self} is not VALID')

            
    """
    pLONG = 'pure.Long'
    pSHORT = 'pure.Short'
    MIXED = 'mixed'

    def __str__(self):
        return self.name
    
    @classmethod
    def list_members(cls) -> List["TimeBinCategorization"]:
        return [cls.pLONG, cls.pSHORT, cls.MIXED]

    @classmethod
    def list_values(cls):
        """Returns a list of all enum values"""
        return list(cls)

    @classmethod
    def list_names(cls):
        """Returns a list of all enum names"""
        return [e.name for e in cls]

    def lower_name(self) -> str:
        """Returns a list of all enum names"""
        return self.name[1:].lower() # 'long', 'short', 'mixed'
        if (self.value == self.pLONG.value):
            pass
        elif (self.value == self.pSHORT.value):
            pass
        elif (self.value == self.MIXED.value):
            pass
        else:
            raise NotImplementedError(f'{self} is not VALID')

    def to_numeric(self) -> int:
        """Returns an integer representation (-1, 0, +1) of the span type"""
        if (self.value == self.pLONG.value):
            return 1
        elif (self.value == self.pSHORT.value):
            return -1
        elif (self.value == self.MIXED.value):
            return 0
        else:
            raise NotImplementedError(f'{self} is not VALID')


    def to_two_column_vector(self) -> NDArray:
        """Returns an integer representation (-1, 0, +1) of the span type"""
        if (self.value == self.pLONG.value):
            return [1, 0]
        elif (self.value == self.pSHORT.value):
            return [0, 1]
        elif (self.value == self.MIXED.value):
            return [0.5, 0.5]
        else:
            raise NotImplementedError(f'{self} is not VALID')
        

    
@metadata_attributes(short_name=None, tags=['run-lengths', 'sequence-analysis', 'temporal'], input_requires=[], output_provides=[], uses=['TimeBinCategorization'], used_by=[], creation_date='2025-05-16 04:28', related_items=[])
class WithinEpochTimeBinDynamics:
    """ This class aims to quantify the bin-to-bin changes in decoded context within a given Epoch event. Ideally it would say something about whether it was static, random, flickering (transitions with change inertia), transitioning, etc.


    Usage:
        ## ⚓🎯 2025-05-15 - Within-epoch transition and run-length sequence analyis
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import WithinEpochTimeBinDynamics, TimeBinCategorization

        sequence_dwell_epochs_df = WithinEpochTimeBinDynamics.analyze_subsequence_temporal_dynamics(curr_active_pipeline, time_bin_size=0.050)
        # int_column_names = [k for k in sequence_dwell_epochs_df.columns if k.startswith('n_')]

        # sequence_dwell_epochs_df.infer_objects()
        sequence_dwell_epochs_df
    
    """
    @classmethod
    def classify(cls, p: float) -> TimeBinCategorization:
        if p > 0.6:
            return TimeBinCategorization.pLONG # 'pure.Long'
        elif p < 0.4:
            return TimeBinCategorization.pSHORT # 'pure.Short'
        else:
            return TimeBinCategorization.MIXED # 'mixed'

    @classmethod
    def classify_binary(cls, p: float) -> TimeBinCategorization:
        if p >= 0.5:
            return TimeBinCategorization.pLONG # 'pure.Long'
        else:
            return TimeBinCategorization.pSHORT # 'pure.Short'


    @function_attributes(short_name=None, tags=['private', 'run-length'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-16 01:29', related_items=[])
    @classmethod
    def run_length_encoding(cls, seq: List[str]) -> Tuple[NDArray, NDArray, List[str]]:
        """ For a given sequence of categorized events (one categorization for each time-bin), returns stats about the sequence
        """
        n_t_bins: int = len(seq)
        
        if n_t_bins == 0:
            return np.array([], dtype=int), np.array([], dtype=int), [], {}
        changes = [0] + [i for i in range(1, len(seq)) if seq[i] != seq[i-1]] + [len(seq)]
        subseq_lengths = np.diff(changes)
        subseq_start_idxs = changes[:-1]
        run_subseq_type_id = [seq[i] for i in subseq_start_idxs]

        type_string_seq = [f'{val}[{length}]' for val, length in zip(run_subseq_type_id, subseq_lengths)]

        type_subsequence_lengths_dict = {k:list() for k in TimeBinCategorization.list_members()}
        
        for type_name, length in zip(run_subseq_type_id, subseq_lengths):
            # type_subsequences_dict[type_name].append()
            type_subsequence_lengths_dict[type_name].append(length)

            

        # type_avg_subseq_lengths_dict = {f"{a_type_name}":np.mean(v) for a_type_name, v in type_subsequence_lengths_dict.items()}
        type_avg_subseq_lengths_dict = {f"mean_len.{a_type_name}":np.mean(v) for a_type_name, v in type_subsequence_lengths_dict.items()}
        type_subseq_lengths_variance_dict = {f"var_len.{a_type_name}":np.nanstd(v) for a_type_name, v in type_subsequence_lengths_dict.items()}
        type_n_bin_counts_dict = {f"n_bins.{a_type_name}":np.nansum(v) for a_type_name, v in type_subsequence_lengths_dict.items()}    
        type_n_bins_ratios_dict = {f"bins_ratio.{a_type_name}":np.nansum(v)/float(n_t_bins) for a_type_name, v in type_subsequence_lengths_dict.items()}

        _out_dict = {
            'type_subseq_lengths_dict': type_subsequence_lengths_dict,
            **type_avg_subseq_lengths_dict,
            **type_subseq_lengths_variance_dict,
            #  'type_subseq_lengths_variance_dict': {a_type_name:np.var(v) for a_type_name, v in type_subsequence_lengths_dict.items()},
            'type_string_seq': type_string_seq,
            **type_n_bin_counts_dict,
            **type_n_bins_ratios_dict, # 'mixed_bins_ratio': 
        }
        # _out_dict.update(**type_subseq_lengths_variance_dict)
        return np.array(subseq_lengths), np.array(subseq_start_idxs), run_subseq_type_id, _out_dict


    @function_attributes(short_name=None, tags=['MAIN', 'transition-analysis'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-16 04:27', related_items=[])
    @classmethod
    def analyze_subsequence_temporal_dynamics(cls, curr_active_pipeline, time_bin_size=0.025, should_show_complex_intermediate_columns: bool = False):
        """ 
        
        complex_column_names: List[str] = ['type_subseq_lengths_dict', 'state_seq', 'type_string_seq']

        
        """        
        from pyphocorehelpers.assertion_helpers import Assert

        # active_state_col_name: str = 'state_seq'
        active_state_col_name: str = 'state_seq_binary'
        
        complex_column_names: List[str] = ['type_subseq_lengths_dict', 'state_seq', 'type_string_seq']
        transferred_column_names: List[str] = ['pre_post_delta_category', 'pre_post_delta_id', 'delta_aligned_start_t']

        valid_EpochComputations_result: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['EpochComputations']
        a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result

        Assert.all_are_not_None(a_new_fully_generic_result)

        # common_constraint_dict = dict(trained_compute_epochs='laps', pfND_ndim=1, time_bin_size=0.025, masked_time_bin_fill_type='ignore')
        common_constraint_dict = dict(trained_compute_epochs='laps', time_bin_size=time_bin_size, masked_time_bin_fill_type='nan_filled') # , pfND_ndim=1


        ## PBEs context:
        a_target_context: IdentifyingContext = IdentifyingContext(known_named_decoding_epochs_type='pbe', data_grain='per_time_bin', **common_constraint_dict) ## Laps , data_grain='per_epoch'
        best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context)
        Assert.all_are_not_None(a_decoded_marginal_posterior_df)

        ## INPUTS: a_decoded_marginal_posterior_df
        epoch_df = deepcopy(a_decoded_marginal_posterior_df)
        epoch_df['state_seq'] = epoch_df['P_Long'].map(cls.classify)
        epoch_df['state_seq_binary'] = epoch_df['P_Long'].map(cls.classify_binary) # .map(lambda x: cls.classify_binary(x).to_numeric()).astype(int)
        
        #  ['P_LR', 'P_RL', 'P_Long', 'P_Short', 'long_LR', 'long_RL', 'short_LR', 'short_RL', 'result_t_bin_idx', 'epoch_df_idx', 'parent_epoch_label', 'parent_epoch_duration', 'label', 'start', 't_bin_center', 'stop', 'delta_aligned_start_t',
        #  'session_name', 'time_bin_size', 'pre_post_delta_category', 'trained_compute_epochs', 'pfND_ndim', 'decoder_identifier', 'known_named_decoding_epochs_type', 'masked_time_bin_fill_type', 'data_grain', 'format_name', 'animal', 'exper_name', 'epochs_source', 'included_qclu_values', 'minimum_inclusion_fr_Hz', 'is_t_bin_center_fake', 'pre_post_delta_id', 'state_seq']
        print(f'epoch_df.columns: {list(epoch_df.columns)}') 

        epoch_split_df_dict = epoch_df.pho.partition_df_dict('parent_epoch_label')
        ## INPUTS: epoch_split_df_dict

        results = []

        ## iterate through split epochs to compute stats
        for i, (epoch_label, an_epoch_df) in enumerate(epoch_split_df_dict.items()):
            # print(f'i: {i}, epoch_label: {epoch_label}')
            epoch_start_t = an_epoch_df['start'].to_numpy()[0]
            epoch_end_t =  an_epoch_df['stop'].to_numpy()[-1]

            a_result_dict = {
                'epoch_start_t': epoch_start_t,
                'epoch_end_t': epoch_end_t,
                'epoch_label': epoch_label,
            }

            for a_col_name in transferred_column_names:
                if a_col_name in an_epoch_df:
                    curr_values = an_epoch_df[a_col_name].to_numpy()
                    if len(curr_values) > 2:
                        Assert.all_equal(curr_values)
                    a_value = curr_values[0] ## get the first, they better all match
                    a_result_dict[a_col_name] = a_value
                else:
                    # print(f'WARN: column "{a_col_name}" is missing from epoch_df!')
                    pass
                    
            # epoch_states = state_seq[epoch_start_t:epoch_end_t]
            epoch_states = an_epoch_df[active_state_col_name].to_numpy()
            n_t_bins: int = len(epoch_states)
            n_transitions: int = np.sum(epoch_states[1:] != epoch_states[:-1])
            subseq_lengths, subseq_start_idxs, run_subseq_type_id, an_epoch_out_dict = cls.run_length_encoding(epoch_states)

            ideal_epoch_states_col_vectors: NDArray = np.vstack([c.to_two_column_vector() for c in epoch_states]).T
            

            np.ones((1, subseq_lengths[0]))

            np.tile(np.array([1, 0]).T, (1, subseq_lengths[0]))
            
            # dist_from_ideal: float = np.nansum((ideal_epoch_states_col_vectors - np.where((an_epoch_df['state_seq_binary'] == TimeBinCategorization.pLONG), an_epoch_df['P_Long'], an_epoch_df['P_Short'])), axis=1)
            
            dist_from_ideal: float = np.nansum((ideal_epoch_states_col_vectors - an_epoch_df[['P_Long', 'P_Short']].to_numpy().T), axis=1)


            # dist_from_ideal: float = np.nansum(1.0 - np.where((an_epoch_df['state_seq_binary'] == TimeBinCategorization.pLONG), an_epoch_df['P_Long'], an_epoch_df['P_Short']))

            a_result_dict.update({
                'n_t_bins': n_t_bins,
                'n_transitions': int(n_transitions),
                # 'type_subseq_lengths_dict': subsequences,
                'lengths': subseq_lengths.tolist(),
                'state_seq': run_subseq_type_id,
                **an_epoch_out_dict,
            })
            
            results.append(a_result_dict)

        ## END for i, (epoch_label, an_epoch_df) in enumera..



        sequence_dwell_epochs_df = pd.DataFrame(results)
        sequence_dwell_epochs_df = sequence_dwell_epochs_df.sort_values('epoch_start_t', ascending=True, inplace=False).reset_index(drop=True)
        
        if not should_show_complex_intermediate_columns:
            sequence_dwell_epochs_df.drop(columns=complex_column_names, inplace=True)

        sequence_dwell_epochs_df = sequence_dwell_epochs_df.convert_dtypes() ## correctly converts columns to integers, etc, but replaces np.nan with <NA>
        ## Move the "height" columns to the end
        sequence_dwell_epochs_df = PandasHelpers.reordering_columns_relative(sequence_dwell_epochs_df, column_names=['lengths'], relative_mode='end') # list(filter(lambda column: column.endswith('_peak_heights'), existing_columns))
        return sequence_dwell_epochs_df



# ==================================================================================================================================================================================================================================================================================== #
# 2025-05-15 - Meas vs. Decoded Occupancy                                                                                                                                                                                                                                              #
# ==================================================================================================================================================================================================================================================================================== #

@metadata_attributes(short_name=None, tags=['VALIDATION', 'occupancy', 'working', 'figure5'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-15 14:22', related_items=[])
class MeasuredVsDecodedOccupancy:
    """ 2025-05-15 - A validation that Kamran had me to do that showed the expected result
    
    
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MeasuredVsDecodedOccupancy
    """
    @function_attributes(short_name=None, tags=['MAIN'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-15 19:57', related_items=[])
    @classmethod
    def analyze_and_plot_meas_vs_decoded_occupancy(cls, best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df, track_templates, figure_title='Laps', plot_in_same_figure:bool=True, should_max_normalize: bool=False, skip_plotting_measured: bool=False, debug_print=False, **kwargs):
        """ analyze and plot

        
        Usage:
            from neuropy.plotting.placemaps import plot_placefield_occupancy, perform_plot_occupancy
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MeasuredVsDecodedOccupancy

            
            valid_EpochComputations_result: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['EpochComputations']
            a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result


            # common_constraint_dict = dict(trained_compute_epochs='laps', pfND_ndim=1, time_bin_size=0.025, masked_time_bin_fill_type='ignore')
            common_constraint_dict = dict(trained_compute_epochs='laps', time_bin_size=0.025, masked_time_bin_fill_type='nan_filled') # , pfND_ndim=1

            ## Laps context:
            a_target_context: IdentifyingContext = IdentifyingContext(known_named_decoding_epochs_type='laps', data_grain='per_time_bin', **common_constraint_dict) ## Laps , data_grain='per_epoch'
            best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context)
            MeasuredVsDecodedOccupancy.analyze_and_plot_meas_vs_decoded_occupancy(best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df, track_templates, figure_title='Laps')

            ## Global context:
            a_target_context: IdentifyingContext = IdentifyingContext(known_named_decoding_epochs_type='global', data_grain='per_time_bin', **common_constraint_dict) ## Laps , data_grain='per_epoch'
            best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context)
            MeasuredVsDecodedOccupancy.analyze_and_plot_meas_vs_decoded_occupancy(best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df, track_templates, figure_title='Global (all-time)')

            ## PBEs context:
            a_target_context: IdentifyingContext = IdentifyingContext(known_named_decoding_epochs_type='pbe', data_grain='per_time_bin', **common_constraint_dict) ## Laps , data_grain='per_epoch'
            best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context)
            MeasuredVsDecodedOccupancy.analyze_and_plot_meas_vs_decoded_occupancy(best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df, track_templates, figure_title='PBEs')


        """
        ## INPUTS: best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df

        
        is_post_delta = (a_decoded_marginal_posterior_df['delta_aligned_start_t'] > 0.0)
        # a_decoded_marginal_posterior_df['is_post_delta'] = is_post_delta

        a_decoded_marginal_posterior_df['pre_post_delta_id'] = 'pre-delta'
        a_decoded_marginal_posterior_df.loc[is_post_delta, 'pre_post_delta_id'] = 'post-delta'
        # a_decoded_marginal_posterior_df['is_post_delta'] = a_decoded_marginal_posterior_df['is_post_delta'].astype(int)
        a_decoded_marginal_posterior_df

        # pre_post_delta_result_splits_dict = a_decoded_marginal_posterior_df.pho.partition_df_dict('pre_post_delta_id')

        n_timebins, flat_time_bin_containers, timebins_p_x_given_n = a_result.flatten() # (59, 4, 69488)
        # timebins_p_x_given_n.shape

        pre_post_delta_timebins_p_x_given_n_dict: Dict[str, NDArray] = {'pre-delta': timebins_p_x_given_n[:, :, np.logical_not(is_post_delta)],
                                                                        'post-delta': timebins_p_x_given_n[:, :, is_post_delta],
        }
        # pre_post_delta_timebins_p_x_given_n_dict

        # ==================================================================================================================================================================================================================================================================================== #
        # Plotting                                                                                                                                                                                                                                                                             #
        # ==================================================================================================================================================================================================================================================================================== #

        # for k, a_result in pre_post_delta_result_dict.items():
        if plot_in_same_figure:
            # Create a single figure with subplots
            fig = plt.figure(layout="constrained", figsize=[18, 8], dpi=220, clear=True, num=f'{figure_title} - plot_meas_vs_decoded_occupancy', **kwargs)
            
            ax_dict = fig.subplot_mosaic(
                [
                    # ["pre-delta", "post-delta"],
                    ["pre-delta_long_LR", "post-delta_long_LR"],
                    ["pre-delta_long_RL", "post-delta_long_RL"],
                    ["pre-delta_short_LR", "post-delta_short_LR"],
                    ["pre-delta_short_RL", "post-delta_short_RL"],
                    # ["long_LR"], ["long_RL"], ["short_LR"], ["short_RL"],
                ],
                # height_ratios=[1],
                sharex=True, sharey=True,
                gridspec_kw=dict(wspace=0.1, hspace=0.1)
            )

            for a_pre_post_delta_name, a_timebins_p_x_given_n in pre_post_delta_timebins_p_x_given_n_dict.items():
                if debug_print:
                    print(f'a_pre_post_delta_name: {a_pre_post_delta_name}')
                # final_ax_key: str = f"{a_pre_post_delta_name}_{}"
                # np.shape(a_timebins_p_x_given_n)
                # ax = ax_dict[a_pre_post_delta_name]  # Get the appropriate subplot axis
                active_ax_dict = {ax_name:v for ax_name, v in ax_dict.items() if (ax_name.split('_', maxsplit=1)[0] == a_pre_post_delta_name)}
                cls.plot_meas_vs_decoded_occupancy(timebins_p_x_given_n=a_timebins_p_x_given_n, track_templates=track_templates, fig=fig, ax_dict=active_ax_dict, a_pre_post_delta_name=a_pre_post_delta_name, should_max_normalize=should_max_normalize, debug_print=debug_print, skip_plotting_measured=skip_plotting_measured, **kwargs)
                # ax.set_title(f'{figure_title} - {a_pre_post_delta_name}')  # Set subplot title

            plt.suptitle(f'{figure_title}')  # Set overall figure title
            return fig, ax_dict
        else:
            # Create separate figures for each condition
            all_figs = []
            for a_pre_post_delta_name, a_timebins_p_x_given_n in pre_post_delta_timebins_p_x_given_n_dict.items():
                if debug_print:
                    print(f'k: {a_pre_post_delta_name}')
                # np.shape(a_timebins_p_x_given_n)
                fig, ax_dict = cls.plot_meas_vs_decoded_occupancy(timebins_p_x_given_n=a_timebins_p_x_given_n, track_templates=track_templates, num=f'{figure_title} - {a_pre_post_delta_name} - plot_meas_vs_decoded_occupancy', a_pre_post_delta_name=a_pre_post_delta_name, should_max_normalize=should_max_normalize, debug_print=debug_print, skip_plotting_measured=skip_plotting_measured, **kwargs)
                plt.suptitle(f'{figure_title} - {a_pre_post_delta_name}')
                all_figs.append((fig, ax_dict))
            return all_figs



    @classmethod
    def plot_meas_vs_decoded_occupancy(cls, timebins_p_x_given_n: NDArray, track_templates, num='plot_meas_vs_decoded_occupancy', fig=None, ax_dict=None, should_max_normalize: bool=False, a_pre_post_delta_name=None, debug_print=False, skip_plotting_measured: bool=False, **kwargs):
        """ from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_meas_vs_decoded_occupancy
        a_result: DecodedFilterEpochsResult
        
        """
        from neuropy.plotting.placemaps import plot_placefield_occupancy, perform_plot_occupancy
        
        ## Measured
        decoders_dict: Dict[types.DecoderName, BasePositionDecoder] = track_templates.get_decoders_dict()
        # ## Decoded:
        # a_result: DecodedFilterEpochsResult = deepcopy(a_result)

        # n_timebins, flat_time_bin_containers, timebins_p_x_given_n = a_result.flatten()
        # timebins_p_x_given_n.shape

        timebins_p_x_given_n = np.nan_to_num(timebins_p_x_given_n)
        timebins_p_x_given_n_occupancy = np.nansum(timebins_p_x_given_n, axis=2) # (n_pos, n_decoders)
        timebins_p_x_given_n_occupancy.shape

        ## sum over all positions to get the scalar per decoder
        scalar_likelihood_per_decoder = np.nansum(timebins_p_x_given_n_occupancy, axis=0) # (n_decoders,)
        normalized_scalar_likelihood_per_decoder = scalar_likelihood_per_decoder / np.nansum(scalar_likelihood_per_decoder)

        a_matching_parts_dict = {'long':'pre-delta', 'short':'post-delta'}
        dir_part_to_arrow_map = {'LR':'<', 'RL':'>'} ## not needed rn
        # n_pos_bins, n_decoders = np.shape(timebins_p_x_given_n_occupancy)

        if (fig is None) or (ax_dict is None):
            # Create a new figure and axes if they are not provided
            fig = plt.figure(layout="constrained", figsize=kwargs.pop('figsize', [18, 8]), dpi=220, clear=True, num=num, **kwargs) # figsize=[Width, height] in inches.
            ax_dict = fig.subplot_mosaic(
                [
                    ["long_LR"], ["long_RL"], ["short_LR"], ["short_RL"],                    
                ],            
                height_ratios=[1, 1, 1, 1],
                sharex=True, sharey=True,
                gridspec_kw=dict(wspace=0, hspace=0)
            )
        else:
            # Use the provided figure and axes
            if (a_pre_post_delta_name is not None) and debug_print:
                print(f'ax_dict: {list(ax_dict.keys())}')
                print(f'\ta_pre_post_delta_name: {a_pre_post_delta_name}')

        for i, (ax_name, ax) in enumerate(ax_dict.items()):
        # for i in np.arange(n_decoders):
            is_measured_result_curr_period: bool = False
            occupancy = timebins_p_x_given_n_occupancy[:,i]
            if a_pre_post_delta_name is not None:
                assert a_pre_post_delta_name in ['pre-delta', 'post-delta'], f'Invalid a_pre_post_delta_name: {a_pre_post_delta_name}.'
                a_pre_post_delta_name_part, a_decoder_name = ax_name.split('_', maxsplit=1) # "post-delta_long_LR" -> ["post-delta", "long_LR"]
                a_long_short_name_part, a_dir_name_part = a_decoder_name.split('_', maxsplit=1) # 'long_LR' -> ['long', 'LR']
                a_decoder: BasePositionDecoder = decoders_dict[a_decoder_name]
                a_formatted_decoder_name: str = a_decoder_name.replace('_LR', ' <', 1).replace('_RL', ' >', 1)
                ax_title: str = f"{a_pre_post_delta_name_part} | "
                is_measured_result_curr_period = a_matching_parts_dict[a_long_short_name_part] == a_pre_post_delta_name # (a_pre_post_delta_name_part == a_pre_post_delta_name)
                if debug_print:
                    print(f'is_measured_result_curr_period: {is_measured_result_curr_period}')

            else:
                a_decoder: BasePositionDecoder = decoders_dict[ax_name]
                a_formatted_decoder_name: str = ax_name.replace('_LR', ' <', 1).replace('_RL', ' >', 1)
                ax_title: str = '' # empty 
                # ax_title: str = f"Decoded Occupancy[{ax_name}]"

            ax_title = f"{ax_title}Decoded Occupancy[{a_formatted_decoder_name}]"
            ax_title = f"{ax_title} (total_decoded={normalized_scalar_likelihood_per_decoder[i]:0.2f})"

            # a_pre_post_delta_name
            measured_occupancy = deepcopy(a_decoder.pf.occupancy)
            occupancy_fig, occupancy_ax = perform_plot_occupancy(occupancy, xbin_centers=None, ybin_centers=None, fig=fig, ax=ax, plot_pos_bin_axes=False, label='decoded', should_max_normalize=should_max_normalize)
            measured_kwargs = dict(alpha=0.9)
            # if not is_measured_result_curr_period:
            #     measured_kwargs = dict(alpha=0.1)
            # else:
            #     measured_kwargs = dict(alpha=1.0)
            if is_measured_result_curr_period:
                ## only plot measured for the correct measured period:
                occupancy_fig, occupancy_ax = perform_plot_occupancy(measured_occupancy, xbin_centers=None, ybin_centers=None, fig=fig, ax=ax, plot_pos_bin_axes=False, label='measured', should_max_normalize=should_max_normalize, **measured_kwargs)
            ax.set_title(ax_title)

        plt.legend(['decoded', 'measured'])
        # occupancy_fig.show()
        return fig, ax_dict




# ==================================================================================================================================================================================================================================================================================== #
# 2025-05-06 - Hairy Marginal on timeline                                                                                                                                                                                                                                              #
# ==================================================================================================================================================================================================================================================================================== #
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import SynchronizedPlotMode
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _helper_add_interpolated_position_columns_to_decoded_result_df
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import _perform_plot_hairy_overlayed_position
from neuropy.utils.matplotlib_helpers import draw_epoch_regions



@function_attributes(short_name=None, tags=['track', 'hairy', 'plot', 'figure', 'matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-06 16:08', related_items=[])
def add_hairy_plot(active_2d_plot, curr_active_pipeline, a_decoded_marginal_posterior_df):
    """ adds a hiary plot the SpikeRaster2D's timeline as a track
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import add_hairy_plot
        
        fig, ax, out_plot_data, dDisplayItem = add_hairy_plot(active_2d_plot, curr_active_pipeline=curr_active_pipeline, a_decoded_marginal_posterior_df=a_decoded_marginal_posterior_df)

    
    """
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import _perform_plot_hairy_overlayed_position


    def _subfn_hide_all_plot_lines(out_plot_data, should_fully_remove_items:bool=False):
        ## get the lines2D object to turn off the default position lines:
        removed_item_names = []
        for a_lines_name, a_lines_collection in out_plot_data.items():
            ## hide all inactive lines:
            print(f'hiding: "{a_lines_name}"')        
            try:
                ## try iteratring the object
                for a_line in a_lines_collection:
                    a_line.set_visible(False)
                removed_item_names.append(a_lines_name)
            except TypeError:
                a_lines_collection.set_visible(False)
                removed_item_names.append(a_lines_name)
            # except AttributeError:
                # when we try to set_visible on non-type
            except Exception as e:
                raise e
        ## end for a_lines_name, a_lin....
        
        ## remove theitems
        if should_fully_remove_items:
            for a_rm_item_name in removed_item_names:
                out_plot_data.pop(a_rm_item_name, None) ## remove the the array

        return out_plot_data           



    ## Build the new dock track:
    dock_identifier: str = 'HairPlot'
    ts_widget, fig, ax_list, dDisplayItem = active_2d_plot.add_new_matplotlib_render_plot_widget(name=dock_identifier)
    ax = ax_list[0]
    ax.clear()
    ax.set_facecolor('white')

    ## OUT: all_directional_continuously_decoded_dict
    ## Draw the position meas/decoded on the plot widget
    ## INPUT: fig, ax_list, all_directional_continuously_decoded_dict, track_templates


    ## INPUTS: a_decoded_marginal_posterior_df

    should_plot_grid_bin_bounds_lines = False

    # plot the basic lap-positions (measured) over time figure:
    _out = dict()
    _out['_display_grid_bin_bounds_validation'] = curr_active_pipeline.display(display_function='_display_grid_bin_bounds_validation', active_session_configuration_context=None, include_includelist=[], save_figure=False, ax=ax) # _display_grid_bin_bounds_validation
    fig = _out['_display_grid_bin_bounds_validation'].figures[0]
    out_axes_list =_out['_display_grid_bin_bounds_validation'].axes
    out_plot_data =_out['_display_grid_bin_bounds_validation'].plot_data

    ax = out_axes_list[0]

    ## get the lines2D object to turn off the default position lines:
    position_lines_2D = out_plot_data['position_lines_2D']
    ## hide all inactive lines:
    for a_line in position_lines_2D:
        a_line.set_visible(False)


    interesting_hair_parameter_kwarg_dict = {
        'defaults': dict(extreme_threshold=0.8, opacity_max=0.7, thickness_ramping_multiplier=35),
        '50_sec_window_scale': dict(extreme_threshold=0.5, thickness_ramping_multiplier=50),
        'full_1700_sec_session_scale': dict(extreme_threshold=0.5, thickness_ramping_multiplier=25), ## really interesting, can see the low-magnitude endcap short-like firing
        'experimental': dict(extreme_threshold=0.8, thickness_ramping_multiplier=55),
        'pbe': dict(extreme_threshold=0.0, opacity_max=0.9, thickness_ramping_multiplier=55),
    }


    out_plot_data = _subfn_hide_all_plot_lines(out_plot_data)
    # an_pos_line_artist, df_viz = _perform_plot_hairy_overlayed_position(df=deepcopy(a_decoded_marginal_posterior_df), ax=ax, extreme_threshold=0.5, thickness_ramping_multiplier=50) # , thickness_ramping_multiplier=5



    ## Named parameter set:
    # an_pos_line_artist, df_viz = _perform_plot_hairy_overlayed_position(df=deepcopy(a_decoded_marginal_posterior_df), ax=ax, **interesting_hair_parameter_kwarg_dict['50_sec_window_scale'])
    # an_pos_line_artist, df_viz = _perform_plot_hairy_overlayed_position(df=deepcopy(a_decoded_marginal_posterior_df), ax=ax, **interesting_hair_parameter_kwarg_dict['full_1700_sec_session_scale'])
    an_pos_line_artist, df_viz = _perform_plot_hairy_overlayed_position(df=deepcopy(a_decoded_marginal_posterior_df), ax=ax, **interesting_hair_parameter_kwarg_dict['pbe'])



    ## sync up the widgets
    active_2d_plot.sync_matplotlib_render_plot_widget(dock_identifier, sync_mode=SynchronizedPlotMode.TO_WINDOW)
    fig.canvas.draw()
    return fig, ax, out_plot_data, dDisplayItem



# def blend_over_white(rgba):
#     rgb = rgba[:, :3]
#     alpha = rgba[:, 3:4]
#     return rgb * alpha + (1 - alpha) * 1  # white background (RGB = 1)

def numpy_rgba_composite(rgba_layers: NDArray[ND.Shape["N_DECODERS, N_POS_BINS, 4"], np.floating], debug_print=False) -> NDArray:
    """
    rgba_layers: (n_layers, H, W, 4) — ordered bottom to top
    Returns: (H, W, 4) — final composited RGBA image
    
    #TODO 2025-05-05 02:23: - [ ] Note when it works `np.shape(rgba_layers) == (4, 59, 1, 4)`
    """
    did_add_singular_W_column: bool = False
    if np.ndim(rgba_layers) < 4:
        rgba_layers = rgba_layers[:, :, None, :]  # (n_decoders, H=n_pos_bins, W=1, 4)
        assert np.ndim(rgba_layers) == 4, f"rgba_layers is the wrong shape. after `rgba_layers = rgba_layers[:, :, None, :]`, np.ndim(rgba_layers): {np.ndim(rgba_layers)} and is still not equal 4!"
        did_add_singular_W_column = True
        
    if debug_print:
        n_layers, height, width, _RGBA_shape = np.shape(rgba_layers)
        print(f'n_layers: {n_layers}, H: {height}, W: {width}, _RGBA_shape: {_RGBA_shape}')
        assert _RGBA_shape == 4, f"_RGBA_shape should be 4 (for RGBA) but is _RGBA_shape: {_RGBA_shape}"

    out_rgb: NDArray[ND.Shape["N_POS_BINS, 3"], np.floating] = np.zeros_like(rgba_layers[0, ..., :3])
    out_alpha: NDArray[ND.Shape["N_POS_BINS"], np.floating] = np.zeros_like(rgba_layers[0, ..., 3])

    for rgba in rgba_layers:
        ## when working for each iteration rgba.shape: (59, 1, 4)
        src_rgb = rgba[..., :3] ## only the RGB components
        src_alpha = rgba[..., 3] ## only the last component (3rd idx)

        out_rgb = src_rgb * src_alpha[..., None] + out_rgb * (1 - src_alpha)[..., None]
        out_alpha = src_alpha + out_alpha * (1 - src_alpha)

    if did_add_singular_W_column:
        return np.concatenate([out_rgb, out_alpha[..., None]], axis=-1)[:, 0, :]
    else:
        return np.concatenate([out_rgb, out_alpha[..., None]], axis=-1)



from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool import GenericMatplotlibContainer
import attrs
from attrs import define, field, Factory, astuple, asdict, fields
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_attribute_field, serialized_field, non_serialized_field, keys_only_repr, shape_only_repr
from neuropy.utils.mixins.HDF5_representable import HDF_SerializationMixin


@metadata_attributes(short_name=None, tags=['figure', 'posteriors'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-04 18:05', related_items=[])
@define(slots=False, eq=False)
class MultiDecoderColorOverlayedPosteriors(ComputedResult):
    """ This class relates to visualizing posterior decoded positions for all four context on the same axes, indicating which posterior is which by assigning each decoder a chracteristic color and weighting their opacity according to their likelyhood for each time bin
        
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MultiDecoderColorOverlayedPosteriors

        import nptyping as ND
        from nptyping import NDArray
        from neuropy.utils.result_context import IdentifyingContext
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _helper_add_interpolated_position_columns_to_decoded_result_df
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MultiDecoderColorOverlayedPosteriors

        ## INPUTS: a_new_fully_generic_result
        # a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='nan_filled', data_grain='per_time_bin')
        a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='ignore', data_grain='per_time_bin')
        best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context, debug_print=False)
        ## OUTPUTS: a_result, a_decoder, a_decoded_marginal_posterior_df
        ## INPUTS: curr_active_pipeline, a_result, a_decoder, a_decoded_marginal_posterior_df
        global_measured_position_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.position.to_dataframe())
        a_decoded_marginal_posterior_df: pd.DataFrame = _helper_add_interpolated_position_columns_to_decoded_result_df(a_result=a_result, a_decoder=a_decoder, a_decoded_marginal_posterior_df=a_decoded_marginal_posterior_df, global_measured_position_df=global_measured_position_df)

        global_decoded_result: SingleEpochDecodedResult = a_result.get_result_for_epoch(0)
        p_x_given_n: NDArray[ND.Shape["N_POS_BINS, 4, N_TIME_BINS"], np.floating] = deepcopy(global_decoded_result.p_x_given_n) # .shape # (59, 4, 69488)

        ## INPUTS: p_x_given_n
        multi_decoder_color_overlay: MultiDecoderColorOverlayedPosteriors = MultiDecoderColorOverlayedPosteriors(p_x_given_n=p_x_given_n, time_bin_centers=time_bin_centers, xbin=xbin, lower_bound_alpha=0.1, drop_below_threshold=1e-3, t_bin_size=0.025)
        multi_decoder_color_overlay.compute_all()
        _out_display_dict = multi_decoder_color_overlay.add_tracks_to_spike_raster_window(active_2d_plot=active_2d_plot, dock_identifier_prefix='MergedColorPlot')



    """
    _VersionedResultMixin_version: str = "2025.05.12_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    spikes_df: pd.DataFrame = serialized_field(repr=shape_only_repr, is_computable=False)
    p_x_given_n: NDArray[ND.Shape["N_POS_BINS, 4, N_TIME_BINS"], np.floating] = serialized_field(repr=shape_only_repr, is_computable=False, metadata={'shape':('N_POS_BINS','4','N_TIME_BINS')}) # .shape # (59, 4, 69488)
    time_bin_centers: NDArray[ND.Shape["N_TIME_BINS"], np.floating] = serialized_field(repr=shape_only_repr, is_computable=False, metadata={'shape':('N_TIME_BINS',)})
    xbin: NDArray[ND.Shape["N_POS_BINS"], np.floating] = serialized_field(repr=shape_only_repr, is_computable=False, metadata={'shape':('N_POS_BINS',)})

    p_x_given_n_track_identity_marginal: NDArray[ND.Shape["N_POS_BINS, 2, N_TIME_BINS"], np.floating] = serialized_field(init=False, default=None, repr=shape_only_repr, is_computable=True, metadata={'shape':('N_POS_BINS','2','N_TIME_BINS')})

    t_bin_size: float = serialized_attribute_field(default=0.025, is_computable=False, repr=True)
    lower_bound_alpha: float = serialized_attribute_field(default=0.1, is_computable=False, repr=True)
    drop_below_threshold: float = serialized_attribute_field(default=1e-3, is_computable=False, repr=True) ## 

    ## Computed Results:
    extra_all_t_bins_outputs_dict_dict: Dict[str, Dict] = field(default=Factory(dict), repr=keys_only_repr)
    active_colors_dict_dict: Dict[str, Dict] = field(default=Factory(dict), repr=keys_only_repr)
    

    def __attrs_post_init__(self):
        # Add post-init logic here        
        self.p_x_given_n_track_identity_marginal = self.compute_track_ID_marginal(p_x_given_n=self.p_x_given_n)



    @function_attributes(short_name=None, tags=['MAIN', 'compute'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-12 23:50', related_items=[])
    def compute_all(self, compute_four_decoder_version: bool=False, progress_print: bool=True):
        """ computes all 
        
        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MultiDecoderColorOverlayedPosteriors

            multi_decoder_color_overlay: MultiDecoderColorOverlayedPosteriors = MultiDecoderColorOverlayedPosteriors(p_x_given_n=p_x_given_n, time_bin_centers=time_bin_centers, xbin=xbin, lower_bound_alpha=0.1, drop_below_threshold=1e-3, t_bin_size=0.025)
            multi_decoder_color_overlay.compute_all()

        
        """
        self.p_x_given_n_track_identity_marginal = self.compute_track_ID_marginal(p_x_given_n=self.p_x_given_n)
        if compute_four_decoder_version:
            self.extra_all_t_bins_outputs_dict_dict['four_decoders'], self.active_colors_dict_dict['four_decoders'] = MultiDecoderColorOverlayedPosteriors.build_four_decoder_version(p_x_given_n=self.p_x_given_n, lower_bound_alpha=self.lower_bound_alpha, drop_below_threshold=self.drop_below_threshold, progress_print=progress_print)
        else:
            self.extra_all_t_bins_outputs_dict_dict.pop('four_decoders', None)
            self.active_colors_dict_dict.pop('four_decoders', None)
            
        self.extra_all_t_bins_outputs_dict_dict['two_decoders'], self.active_colors_dict_dict['two_decoders'] = MultiDecoderColorOverlayedPosteriors.build_two_decoder_version(p_x_given_n=self.p_x_given_n, lower_bound_alpha=self.lower_bound_alpha, drop_below_threshold=self.drop_below_threshold, progress_print=progress_print)
        if progress_print:
            print(f'\tdone.')



    @function_attributes(short_name=None, tags=['decoder_result'], input_requires=[], output_provides=[], uses=['.compute_all_time_bin_RGBA'], used_by=['.compute_all'], creation_date='2025-05-13 00:03', related_items=[])
    @classmethod
    def build_four_decoder_version(cls, p_x_given_n: NDArray[ND.Shape["N_POS_BINS, 4, N_TIME_BINS"], np.floating], lower_bound_alpha:float=0.1, drop_below_threshold:float=1e-3, **kwargs):
        """
        # , time_bin_centers: NDArray[ND.Shape["N_TIME_BINS"], np.floating], xbin: NDArray[ND.Shape["N_POS_BINS"], np.floating]
        
        
        Usage:        
            extra_all_t_bins_outputs_dict, active_colors_dict = MultiDecoderColorOverlayedPosteriors.build_four_decoder_version(p_x_given_n=p_x_given_n, lower_bound_alpha=0.1, drop_below_threshold=1e-3)
            all_t_bins_final_overlayed_out_RGBA = extra_all_t_bins_outputs_dict['all_t_bins_final_overlayed_out_RGBA']
            all_t_bins_per_decoder_out_RGBA = extra_all_t_bins_outputs_dict['all_t_bins_per_decoder_out_RGBA']

        """
        from pyphocorehelpers.gui.Qt.color_helpers import ColormapHelpers
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MultiDecoderColorOverlayedPosteriors
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import SynchronizedPlotMode
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool import GenericMatplotlibContainer, GenericPyQtGraphContainer, PhoBaseContainerTool

        ## Common:
        all_decoder_colors_dict = {'long': '#4169E1', 'short': '#DC143C', 'long_LR': '#4169E1', 'long_RL': '#607B00', 'short_LR': '#DC143C', 'short_RL': '#990099'} ## Just hardcoded version of `additional_cmap_names`
        
        # ==================================================================================================================================================================================================================================================================================== #
        # N_DECODERS == 4 ['long_LR', 'long_RL', 'short_LR', 'short_RL']                                                                                                                                                                                                                       #
        # ==================================================================================================================================================================================================================================================================================== #
        active_cmap_names = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
        active_p_x_given_n = deepcopy(p_x_given_n)

        # OUTPUTS: all_decoder_colors_dict, p_x_given_n
        ## OUTPUTS: active_cmap_names, active_p_x_given_n, time_bin_centers, xbin
        
        ## INPUTS: all_decoder_colors_dict, active_cmap_names
        active_colors_dict = {k:v for k, v in all_decoder_colors_dict.items() if k in active_cmap_names}
        active_decoder_cmap_dict = {k:ColormapHelpers.create_transparent_colormap(color_literal_name=v, lower_bound_alpha=lower_bound_alpha, should_return_LinearSegmentedColormap=True) for k, v in all_decoder_colors_dict.items() if k in active_cmap_names}
        # additional_legend_entries = list(zip(directional_active_lap_pf_results_dicts.keys(), additional_cmap_names.values() )) # ['red', 'purple', 'green', 'orange']

        ## OUTPUTS: active_cmap_decoder_dict
        extra_all_t_bins_outputs_dict = MultiDecoderColorOverlayedPosteriors.compute_all_time_bin_RGBA(p_x_given_n=active_p_x_given_n, produce_debug_outputs=False, drop_below_threshold=drop_below_threshold, active_decoder_cmap_dict=active_decoder_cmap_dict, should_constrain_to_four_decoder=True, **kwargs)

        return extra_all_t_bins_outputs_dict, active_colors_dict


    @function_attributes(short_name=None, tags=['private', 'helper', 'data'], input_requires=[], output_provides=[], uses=[], used_by=['.build_two_decoder_version'], creation_date='2025-05-13 00:05', related_items=[])
    @classmethod
    def compute_track_ID_marginal(cls, p_x_given_n: NDArray[ND.Shape["N_POS_BINS, 4, N_TIME_BINS"], np.floating]):
        """ Computes the two-decoder marginal of trackID

        Usage:        
            p_x_given_n_track_identity_marginal: NDArray[ND.Shape["N_POS_BINS, 2, N_TIME_BINS"], np.floating] = multi_decoder_color_overlay.compute_track_ID_marginal(p_x_given_n=p_x_given_n) # .shape (2, n_time_bins)
            np.shape(p_x_given_n_track_identity_marginal)
            p_x_given_n_track_identity_marginal

        """
        # def _subfn_perform_normalization_check(a_marginal_p_x_given_n):
        #     ## check normalization
        #     col_contains_nan = np.any(np.isnan(a_marginal_p_x_given_n), axis=1)
        #     # np.shape(col_contains_nan)
        #     _post_norm_check_sum = np.nansum(a_marginal_p_x_given_n, axis=1)
        #     assert np.alltrue(_post_norm_check_sum[np.logical_not(col_contains_nan)]), f"the non-nan containing columns should sum to one after renormalization"
            
        def _subfn_perform_area_under_curve_normalization_check(a_marginal_p_x_given_n, axis=(0,1)):
            ## check normalization
            col_contains_nan = np.any(np.isnan(a_marginal_p_x_given_n), axis=axis)
            # np.shape(col_contains_nan)
            _post_norm_check_sum = np.nansum(a_marginal_p_x_given_n, axis=axis)
            assert np.alltrue(_post_norm_check_sum[np.logical_not(col_contains_nan)]), f"the non-nan containing columns should sum to one after renormalization"
            


        n_pos_bins, n_decoders, n_time_bins = np.shape(p_x_given_n)
        assert n_decoders == 4, f"n_decoders: {n_decoders}"
        
        marginal_trackID_p_x_given_n: NDArray[ND.Shape["N_POS_BINS, 2, N_TIME_BINS"], np.floating] = np.zeros(shape=(n_pos_bins, 2, n_time_bins))
            
        ## Long:
        long_only_p_x_given_n: NDArray[ND.Shape["N_POS_BINS, 2, N_TIME_BINS"], np.floating] = deepcopy(p_x_given_n[:, (0,1), :])
        sum_over_all_long_p_x_given_n: NDArray[ND.Shape["N_POS_BINS, N_TIME_BINS"], np.floating] = np.nansum(long_only_p_x_given_n, axis=1) ## sum over the two long columns
        
        ## Short:
        short_only_p_x_given_n: NDArray[ND.Shape["N_POS_BINS, 2, N_TIME_BINS"], np.floating] = deepcopy(p_x_given_n[:, (2,3), :])
        sum_over_all_short_p_x_given_n: NDArray[ND.Shape["N_POS_BINS, N_TIME_BINS"], np.floating] = np.nansum(short_only_p_x_given_n, axis=1) ## sum over the two long columns
        
        ## All
        sum_over_all_decoders_p_x_given_n: NDArray[ND.Shape["N_POS_BINS, N_TIME_BINS"], np.floating] = np.nansum(p_x_given_n, axis=1) ## sum over all decoders to re-normalize
        sum_over_all_decoders_and_all_positions_p_x_given_n: NDArray[ND.Shape["N_TIME_BINS"], np.floating] = np.nansum(p_x_given_n, axis=(0,1)) ## sum over all decoders to re-normalize

        ## build result
        marginal_trackID_p_x_given_n[:, 0, :] = sum_over_all_long_p_x_given_n
        marginal_trackID_p_x_given_n[:, 1, :] = sum_over_all_short_p_x_given_n

        ## Normalize result:
        with np.errstate(divide='ignore', invalid='ignore'): # 
            # marginal_trackID_p_x_given_n = marginal_trackID_p_x_given_n / sum_over_all_decoders_p_x_given_n[:, None, :] ## renormalize by dividing by sum over all decoders
            # _subfn_perform_normalization_check(a_marginal_p_x_given_n=marginal_trackID_p_x_given_n)
            
            marginal_trackID_p_x_given_n = marginal_trackID_p_x_given_n / sum_over_all_decoders_and_all_positions_p_x_given_n ## renormalize by dividing by sum over all decoders
            _subfn_perform_area_under_curve_normalization_check(a_marginal_p_x_given_n=marginal_trackID_p_x_given_n)
            # _post_norm_check_sum: NDArray[ND.Shape["N_TIME_BINS"], np.floating] = np.nansum(marginal_trackID_p_x_given_n, axis=(0,1))
            
            # long_only_p_x_given_n = long_only_p_x_given_n / sum_over_all_long_p_x_given_n[:, None, :] ## renormalize by dividing by sum over all decoders
            # _subfn_perform_normalization_check(a_marginal_p_x_given_n=long_only_p_x_given_n)
        

        # np.shape(_post_norm_check_sum)
        # np.shape(long_only_p_x_given_n)

        # sum_over_all_long_p_x_given_n.shape
        # sum_over_all_decoders_p_x_given_n.shape
        # return long_only_p_x_given_n
        return marginal_trackID_p_x_given_n

        
    @function_attributes(short_name=None, tags=['decoder_result'], input_requires=[], output_provides=[], uses=['.compute_all_time_bin_RGBA', '.compute_track_ID_marginal'], used_by=['.compute_all'], creation_date='2025-05-13 00:03', related_items=[])
    @classmethod
    def build_two_decoder_version(cls, p_x_given_n: NDArray[ND.Shape["N_POS_BINS, 4, N_TIME_BINS"], np.floating], lower_bound_alpha:float=0.1, drop_below_threshold:float=1e-3, **kwargs):
        """
        # , time_bin_centers: NDArray[ND.Shape["N_TIME_BINS"], np.floating], xbin: NDArray[ND.Shape["N_POS_BINS"], np.floating]
        
        Usage:        
            extra_all_t_bins_outputs_dict_dict['two_decoders'], active_colors_dict_dict['two_decoders'] = MultiDecoderColorOverlayedPosteriors.build_two_decoder_version(p_x_given_n=p_x_given_n, lower_bound_alpha=0.1, drop_below_threshold=1e-3)
            all_t_bins_final_overlayed_out_RGBA = extra_all_t_bins_outputs_dict['all_t_bins_final_overlayed_out_RGBA']
            all_t_bins_per_decoder_out_RGBA = extra_all_t_bins_outputs_dict['all_t_bins_per_decoder_out_RGBA']

            
        Usage 2:
            ## INPUTS: p_x_given_n_track_identity_marginal
            extra_all_t_bins_outputs_dict_dict['two_decoders'], active_colors_dict_dict['two_decoders'] = MultiDecoderColorOverlayedPosteriors.build_two_decoder_version(p_x_given_n_track_identity_marginal=p_x_given_n_track_identity_marginal, lower_bound_alpha=0.1, drop_below_threshold=1e-3)

        """
        from pyphocorehelpers.gui.Qt.color_helpers import ColormapHelpers
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MultiDecoderColorOverlayedPosteriors
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import SynchronizedPlotMode
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool import GenericMatplotlibContainer, GenericPyQtGraphContainer, PhoBaseContainerTool

        ## Common:
        all_decoder_colors_dict = {'long': '#4169E1', 'short': '#DC143C'} ## Just hardcoded version of `additional_cmap_names`
        

        # ==================================================================================================================================================================================================================================================================================== #
        # N_DECODERS == 2 ['long', 'short']                                                                                                                                                                                                                                                    #
        # ==================================================================================================================================================================================================================================================================================== #
        active_cmap_names = ['long', 'short']

        # p_x_given_n_track_identity_marginal: NDArray[ND.Shape["N_POS_BINS, 2, N_TIME_BINS"], np.floating]


        ## INPUTS: p_x_given_n

        n_pos_bins, n_decoders, n_time_bins = np.shape(p_x_given_n)
        # assert n_decoders == 4, f"n_decoders: {n_decoders}"
        if (n_decoders == 4):
            # # p_x_given_n_track_identity_marginal: NDArray[ND.Shape["N_POS_BINS, 2, N_TIME_BINS"], np.floating] = long_only_p_x_given_n # .shape (2, n_time_bins)
            p_x_given_n_track_identity_marginal: NDArray[ND.Shape["N_POS_BINS, 2, N_TIME_BINS"], np.floating] = cls.compute_track_ID_marginal(p_x_given_n=p_x_given_n) # .shape (2, n_time_bins)

        elif (n_decoders == 2):
            p_x_given_n_track_identity_marginal: NDArray[ND.Shape["N_POS_BINS, 2, N_TIME_BINS"], np.floating] = deepcopy(p_x_given_n) # .shape (2, n_time_bins)
        else:
            raise ValueError(f'n_decoders: {n_decoders} should be 4 or 2')


        active_p_x_given_n = deepcopy(p_x_given_n_track_identity_marginal)
        # np.shape(active_p_x_given_n)

        # OUTPUTS: all_decoder_colors_dict, p_x_given_n
        ## OUTPUTS: active_cmap_names, active_p_x_given_n, time_bin_centers, xbin
        
        ## INPUTS: all_decoder_colors_dict, active_cmap_names
        active_colors_dict = {k:v for k, v in all_decoder_colors_dict.items() if k in active_cmap_names}
        
        active_decoder_cmap_dict = {k:ColormapHelpers.create_transparent_colormap(color_literal_name=v, lower_bound_alpha=lower_bound_alpha, should_return_LinearSegmentedColormap=True) for k, v in all_decoder_colors_dict.items() if k in active_cmap_names}
        # additional_legend_entries = list(zip(directional_active_lap_pf_results_dicts.keys(), additional_cmap_names.values() )) # ['red', 'purple', 'green', 'orange']

        ## OUTPUTS: active_cmap_decoder_dict
        extra_all_t_bins_outputs_dict = cls.compute_all_time_bin_RGBA(p_x_given_n=active_p_x_given_n, produce_debug_outputs=False, drop_below_threshold=drop_below_threshold, active_decoder_cmap_dict=active_decoder_cmap_dict, should_constrain_to_four_decoder=False, **kwargs)
        
        return extra_all_t_bins_outputs_dict, active_colors_dict
    


    @function_attributes(short_name=None, tags=['MAIN', 'all_t'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-04 18:00', related_items=[])
    @classmethod
    def compute_all_time_bin_RGBA(cls, p_x_given_n: NDArray, active_decoder_cmap_dict: Optional[Dict]=None, produce_debug_outputs: bool = False, drop_below_threshold: float = 1e-2, progress_print: bool = True, color_blend_fn=None, should_constrain_to_four_decoder:bool=True, debug_print=False) -> Tuple[NDArray, NDArray]:
        """ Computes the final RGBA colors for each position x time bin in p_x_given_n by overlaying each of the decoders values
        
        
        NOTE: COMPLETELY INDEPENDENT/DECOUPLED from `cls._test_single_t_bin` (copy/paste synchronized)
        
        #TODO 2025-05-05 04:07: - [ ] Can improve by not showing all four bins, but instead marginalizing over Long/Short and just plotting those. The off-color is ugly, and with only 2 options the colors can actually be orthogonal and easy to read
        
        
        import nptyping as ND
        from nptyping import NDArray
        from neuropy.utils.result_context import IdentifyingContext
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _helper_add_interpolated_position_columns_to_decoded_result_df
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MultiDecoderColorOverlayedPosteriors

        ## INPUTS: a_new_fully_generic_result
        # a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='nan_filled', data_grain='per_time_bin')
        a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='ignore', data_grain='per_time_bin')
        best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context, debug_print=False)
        ## OUTPUTS: a_result, a_decoder, a_decoded_marginal_posterior_df
        ## INPUTS: curr_active_pipeline, a_result, a_decoder, a_decoded_marginal_posterior_df
        global_measured_position_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.position.to_dataframe())
        a_decoded_marginal_posterior_df: pd.DataFrame = _helper_add_interpolated_position_columns_to_decoded_result_df(a_result=a_result, a_decoder=a_decoder, a_decoded_marginal_posterior_df=a_decoded_marginal_posterior_df, global_measured_position_df=global_measured_position_df)

        global_decoded_result: SingleEpochDecodedResult = a_result.get_result_for_epoch(0)
        p_x_given_n: NDArray[ND.Shape["N_POS_BINS, 4, N_TIME_BINS"], np.floating] = deepcopy(global_decoded_result.p_x_given_n) # .shape # (59, 4, 69488)
        # p_x_given_n

        ## INPUTS: p_x_given_n
        extra_all_t_bins_outputs_dict = MultiDecoderColorOverlayedPosteriors.compute_all_time_bin_RGBA(p_x_given_n=p_x_given_n, produce_debug_outputs=False, drop_below_threshold=1e-1)
        all_t_bins_final_overlayed_out_RGBA = extra_all_t_bins_outputs_dict['all_t_bins_final_overlayed_out_RGBA']
        all_t_bins_per_decoder_out_RGBA = extra_all_t_bins_outputs_dict['all_t_bins_per_decoder_out_RGBA']

        ## OUTPUTS: extra_all_t_bins_outputs_dict, all_t_bins_per_decoder_out_RGBA, all_t_bins_final_overlayed_out_RGBA
        
        """
        
        # p_x_given_n: NDArray[ND.Shape["N_TIME_BINS, N_POS_BINS, N_DECODERS"], np.floating]
        ## INPUTS: p_x_given_n, drop_below_threshold, active_decoder_cmap_dict, produce_debug_outputs
        p_x_given_n = deepcopy(p_x_given_n) ## copy p_x_given_n so it isn't modified

        if active_decoder_cmap_dict is None:
            # decoder_names_to_idx_map: Dict[int, types.DecoderName] = {0: 'long_LR', 1: 'long_RL', 2: 'short_LR', 3: 'short_RL'}
            color_dict: Dict[types.DecoderName, pg.QtGui.QColor] = DecoderIdentityColors.build_decoder_color_dict(wants_hex_str=False)
            additional_cmap_names: Dict[types.DecoderName, str] = {k: ColorFormatConverter.qColor_to_hexstring(v) for k, v in color_dict.items()}
            # additional_cmap_names = {'long_LR': '#4169E1', 'long_RL': '#607B00', 'short_LR': '#DC143C', 'short_RL': '#990099'} ## Just hardcoded version of `additional_cmap_names`
            active_decoder_cmap_dict = {k:ColormapHelpers.create_transparent_colormap(color_literal_name=v, lower_bound_alpha=0.1, should_return_LinearSegmentedColormap=True) for k, v in additional_cmap_names.items()}

        if color_blend_fn is None:
            # color_blend_fn = cls.composite_multiplied_alpha
            color_blend_fn = cls.composite_over

        n_pos_bins, n_decoders, n_time_bins = np.shape(p_x_given_n)
        if should_constrain_to_four_decoder:
            assert n_decoders == 4, f"n_decoders: {n_decoders}"
        else:
            if debug_print:
                print(f'n_decoders: {n_decoders}')


        assert (len(active_decoder_cmap_dict) == n_decoders), f"len(active_decoder_cmap_dict): {len(active_decoder_cmap_dict)} != n_decoders: {n_decoders} but it must!"
            
        ## INPUTS: probability_values (n_pos_bins, 4)
        # all_t_bins_per_decoder_out_RGBA: NDArray[ND.Shape["N_TIME_BINS", "N_POS_BINS, N_DECODERS, 4"], np.floating] = np.zeros((n_time_bins, n_pos_bins, n_decoders, 4))
        # all_t_bins_final_overlayed_out_RGBA: NDArray[ND.Shape["N_TIME_BINS", "N_POS_BINS, 4"], np.floating] = np.zeros((n_time_bins, n_pos_bins, 4)) # Pre-compute the overlay so that there's only one color that represents up to all four active decoders
        extra_all_t_bins_outputs_dict: Dict = {
            'all_t_bins_per_decoder_alphas': np.zeros((n_time_bins, n_decoders)),
            'all_t_bins_per_decoder_alpha_weighted_RGBA': np.zeros((n_time_bins, n_pos_bins, n_decoders, 4)),
            'all_t_bins_final_RGBA': np.zeros((n_time_bins, n_pos_bins, 4)),
            'all_t_bins_per_decoder_out_RGBA': np.zeros((n_time_bins, n_pos_bins, n_decoders, 4)),
            'all_t_bins_final_overlayed_out_RGBA': np.zeros((n_time_bins, n_pos_bins, 4)),
        }

        with np.errstate(divide='ignore', invalid='ignore'):
            ## print log spamming division errors
            for a_t_bin_idx in np.arange(n_time_bins):
                ## for each time bin, there are several ways to normalize
                    # 1. normalize over all in each time bin 
                    # 2. normalize to max/min in each time (colormapping), means colors aren't consistent between timebins
                                
                if progress_print:
                    is_every_hundreth_t_bin = (a_t_bin_idx % 1000 == 0)
                    if is_every_hundreth_t_bin:        
                        print(f'a_t_bin_idx: [{a_t_bin_idx}/{n_time_bins}]')

                # single_t_bin_P_values: NDArray[ND.Shape["N_POS_BINS, N_DECODERS"], np.floating] = deepcopy(p_x_given_n[:, :, a_t_bin_idx])
                single_t_bin_P_values: NDArray[ND.Shape["N_POS_BINS, N_DECODERS"], np.floating] = p_x_given_n[:, :, a_t_bin_idx]

                if produce_debug_outputs:
                    _pre_norm_prob_vals = deepcopy(single_t_bin_P_values)

                # DEBUGONLY __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
                if produce_debug_outputs:
                    ## normalize over (decoder, position)
                    sum_over_all_decoder_pos_values: float = np.nansum(_pre_norm_prob_vals, axis=(0, 1)) # sum over pos
                    print(f'sum_over_all_decoder_pos_values: {sum_over_all_decoder_pos_values}')
                    # _all_normed_prob_vals = _pre_norm_prob_vals / sum_over_all_decoder_pos_values ## normalize

                ## normalize over decoder
                sum_over_all_pos_values: NDArray[ND.Shape["N_DECODERS"], np.floating] = np.nansum(single_t_bin_P_values, axis=0) # sum over pos
                if produce_debug_outputs:
                    print(f'sum_over_all_pos_values: {sum_over_all_pos_values}')
                    
                decoder_alphas: NDArray[ND.Shape["N_DECODERS"], np.floating] = sum_over_all_pos_values.copy()
                single_t_bin_P_values = single_t_bin_P_values / sum_over_all_pos_values ## normalize over (decoder x position)

                ## OUT: probability_values
                ## INPUTS: probability_values (n_pos_bins, 4)
                # single_t_bin_out_RGBA = np.zeros((n_pos_bins, 4, 4))

                ## pre-plotting only: mask the tiny values:
                single_t_bin_P_values = cls._prepare_arr_for_conversion_to_RGBA(single_t_bin_P_values, drop_below_threshold=drop_below_threshold)

                ## for each decoder:
                for i, (a_decoder_name, a_cmap) in enumerate(active_decoder_cmap_dict.items()):
                    if produce_debug_outputs:
                        # print(f'i: {i}, a_decoder_name: {a_decoder_name}')
                        pass

                    a_t_bin_a_decoder_P: NDArray[ND.Shape["N_POS_BINS"], np.floating] = single_t_bin_P_values[:, i]
                    ## INPUTS: a_t_bin_a_decoder_P
                    # ignore NaNs when finding data range
                    is_all_nan_slice: bool = np.all(np.isnan(a_t_bin_a_decoder_P)) # to avoid: RuntimeWarning: All-NaN slice encountered
                    if not is_all_nan_slice:
                        a_norm = mpl.colors.Normalize(vmin=np.nanmin(a_t_bin_a_decoder_P), vmax=np.nanmax(a_t_bin_a_decoder_P))
                        
                        # mask the NaNs so the cmap knows to use the “bad” color
                        a_masked = np.ma.masked_invalid(a_t_bin_a_decoder_P)
                        an_rgba: NDArray[ND.Shape["N_POS_BINS, 4"], np.floating] = a_cmap(a_norm(a_masked)) # rgba.shape (n_pos_bins, 4) # the '4' here is for RGBA, not the decoders, RGB per bin
                        # rgb = an_rgba[..., :3] 
                        # single_t_bin_out_RGBA[:, i, :] = an_rgba
                    else:
                        ## #TODO 2025-05-30 06:51: - [ ] how to avoid the RuntimeWarning: All-NaN slice encountered
                        # Handle the all-NaN case by filling with a specific color
                        # # Option 1: Fill with a neutral gray color (visually indicates "no data")
                        # neutral_color = np.array([0.5, 0.5, 0.5, 1.0])  # RGBA for gray
                        # single_t_bin_out_RGBA[:, i, :] = np.tile(neutral_color, (a_t_bin_a_decoder_P.shape[0], 1))
                        
                        # Option 2: Fill with transparent color (if you want these bins to be invisible)
                        # transparent_color = np.array([0.0, 0.0, 0.0, 0.0])  # Fully transparent RGBA
                        # single_t_bin_out_RGBA[:, i, :] = np.tile(transparent_color, (a_t_bin_a_decoder_P.shape[0], 1))
                        
                        # # Option 3: Use the colormap's "bad" color (consistent with how NaNs are handled elsewhere)
                        # bad_color = np.array(a_cmap.get_bad())
                        # single_t_bin_out_RGBA[:, i, :] = np.tile(bad_color, (a_t_bin_a_decoder_P.shape[0], 1))

                        # mask the NaNs so the cmap knows to use the “bad” color
                        a_masked = np.ma.masked_invalid(a_t_bin_a_decoder_P)
                        an_rgba: NDArray[ND.Shape["N_POS_BINS, 4"], np.floating] = a_cmap(a_masked) # rgba.shape (n_pos_bins, 4) # the '4' here is for RGBA, not the decoders, RGB per bin
                        # rgb = an_rgba[..., :3] 
                        # single_t_bin_out_RGBA[:, i, :] = an_rgba
                        
                    ## OUTPUT: an_rgba
                    extra_all_t_bins_outputs_dict['all_t_bins_per_decoder_out_RGBA'][a_t_bin_idx, :, i, :] = an_rgba
                    
                ## END for i, (a_decoder_name, a_cmap) in enum....
                
                # Add dict entries ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
                # single_t_bin_out_RGBA: (n_pos_bins, 4, 4)
                single_t_bin_out_RGBA: NDArray[ND.Shape["N_POS_BINS, N_DECODERS, 4"], np.floating] = extra_all_t_bins_outputs_dict['all_t_bins_per_decoder_out_RGBA'][a_t_bin_idx, :, :, :]
                # single_t_bin_out_alpha_weighted_RGBA: NDArray[ND.Shape["N_POS_BINS, N_DECODERS, 4"], np.floating] = (deepcopy(single_t_bin_out_RGBA) * decoder_alphas[None, :, None]) # (n_pos_bins, n_decoders, 4)
                extra_all_t_bins_outputs_dict['all_t_bins_per_decoder_alphas'][a_t_bin_idx, :] = decoder_alphas
                # extra_all_t_bins_outputs_dict['all_t_bins_per_decoder_alpha_weighted_RGBA'][a_t_bin_idx, :, :, :] = (deepcopy(single_t_bin_out_RGBA) * decoder_alphas[None, :, None]) # (n_pos_bins, n_decoders, 4)
                extra_all_t_bins_outputs_dict['all_t_bins_per_decoder_alpha_weighted_RGBA'][a_t_bin_idx, :, :, :] = (single_t_bin_out_RGBA * decoder_alphas[None, :, None]) # (n_pos_bins, n_decoders, 4)

                extra_all_t_bins_outputs_dict['all_t_bins_final_overlayed_out_RGBA'][a_t_bin_idx, :, :] = color_blend_fn(single_t_bin_out_RGBA, decoder_alphas=decoder_alphas) # (n_pos_bins, 4, 4)?
                
                ## `numpy_rgba_composite` -- expects input = rgba_layers: (n_layers, H, W, 4) - here (H:
                # extra_all_t_bins_outputs_dict['all_t_bins_final_RGBA'][a_t_bin_idx, :, :] = numpy_rgba_composite(np.transpose(single_t_bin_out_RGBA, (1, 0, 2)))  # (n_decoders, H=n_pos_bins, 4)
                # extra_all_t_bins_outputs_dict['all_t_bins_final_RGBA'][a_t_bin_idx, :, :] = numpy_rgba_composite(np.transpose(extra_all_t_bins_outputs_dict['all_t_bins_per_decoder_alpha_weighted_RGBA'][a_t_bin_idx, :, :, :], (1, 0, 2))) # transpose -> (n_pos_bins, n_decoders, 4) => (n_decoders, n_pos_bins, 4)
                extra_all_t_bins_outputs_dict['all_t_bins_final_RGBA'][a_t_bin_idx, :, :] = numpy_rgba_composite(np.transpose(extra_all_t_bins_outputs_dict['all_t_bins_per_decoder_alpha_weighted_RGBA'][a_t_bin_idx, :, :, :], (1, 0, 2))) # transpose -> (n_pos_bins, n_decoders, 4) => (n_decoders, n_pos_bins, 4)

                # numpy_rgba_composite: Returns: (H, W, 4) — final composited RGBA image
            ## END FOR for a_t_bin_idx in np.a...
            # extra_all_t_bins_outputs_dict['all_t_bins_final_overlayed_out_RGBA'] = all_t_bins_final_overlayed_out_RGBA
            # extra_all_t_bins_outputs_dict['all_t_bins_per_decoder_out_RGBA'] = all_t_bins_per_decoder_out_RGBA

        if progress_print:
            print(f'a_t_bin_idx: [{a_t_bin_idx}/{n_time_bins}]')
            print(f'\tdone.')
            

        return extra_all_t_bins_outputs_dict


    @classmethod
    def _prepare_arr_for_conversion_to_RGBA(cls, arr: NDArray, alt_arr_to_use_for_drops: Optional[NDArray]=None, drop_below_threshold: float=0.0000001, skip_scaling:bool=True):
        """ Pure (does not alter `arr`, returns a copy).
        If `drop_below_threshold`
        
        IFF (skip_scaling == False): Regardless of `alt_arr_to_use_for_drops` and `drop_below_threshold`, the image is rescaled to fill its dynamic range by its maximum (meaning the output will be normalized between zero and one).
        `alt_arr_to_use_for_drops` is not used unless `drop_below_threshold` is non-None
        
        Input:
            drop_below_threshold: if None, no indicies are dropped. Otherwise, values of occupancy less than the threshold specified are used to build a mask, which is subtracted from the returned image (the image is NaN'ed out in these places).

            
        """
        # Pre-filter the data:
        with np.errstate(divide='ignore', invalid='ignore'):
            arr = np.array(arr.copy())
            if not skip_scaling:
                arr = arr / np.nanmax(arr) # note scaling by maximum here!
            if (drop_below_threshold is not None) and (drop_below_threshold > 0):          
                if (alt_arr_to_use_for_drops is not None):
                    Assert.same_shape(arr, alt_arr_to_use_for_drops)
                else:
                    alt_arr_to_use_for_drops = arr

                arr[np.where(alt_arr_to_use_for_drops < drop_below_threshold)] = np.nan # null out the values below the threshold for visualization
                
            # arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0) # from prepare_data_for_plotting
            return arr # return the modified and masked image
        

    @classmethod
    def composite_multiplied_alpha(cls, single_t_bin_out_RGBA, decoder_alphas):
        """ Attempts to compute the final overlayed/merged color across decoders (so there's only a single value for all decoders)
        Usage:
            final_overlayed_single_t_bin_out_RGBA = composite_multiplied_alpha(single_t_bin_out_RGBA=single_t_bin_out_RGBA)
        """
        # out_rgb = np.zeros(3)
        return np.nansum((single_t_bin_out_RGBA * decoder_alphas[None, :, None]), axis=1) # (n_pos_bins, 4)


    @classmethod
    def composite_over(cls, single_t_bin_out_RGBA, decoder_alphas):
        """ Attempts to compute the final overlayed/merged color across decoders (so there's only a single value for all decoders)
        
        single_t_bin_out_RGBA: (n_pos_bins, n_decoders, 4)
        decoder_alphas: (n_decoders,)
        Returns: (n_pos_bins, 4)
        """
        n_pos_bins, n_decoders, _ = single_t_bin_out_RGBA.shape
        out_rgba = np.zeros((n_pos_bins, 4))

        for i in range(n_decoders):
            src = single_t_bin_out_RGBA[:, i, :]  # (n_pos_bins, 4)
            valid = ~np.isnan(src).any(axis=1)    # mask for valid RGBA values

            src_rgb = src[valid, :3]
            src_alpha = decoder_alphas[i]
            dst_rgb = out_rgba[valid, :3]
            dst_alpha = out_rgba[valid, 3]

            out_rgba[valid, :3] = src_rgb * src_alpha + dst_rgb * (1 - src_alpha)
            out_rgba[valid, 3] = src_alpha + dst_alpha * (1 - src_alpha)

        return out_rgba


    @function_attributes(short_name=None, tags=['UNUSED', 'HACK', 'matplotlib', 'overlay', 'figure'], input_requires=[], output_provides=[], uses=['numpy_rgba_composite'], used_by=[], creation_date='2025-05-04 18:19', related_items=[])
    @classmethod
    def extract_center_rgba_from_figure(cls, fig, axd, n_pos_bins, subplot_name='matplotlib_combined_rgba', debug_print=False):
        """ Used to reverse-engeinerr the overlayed colors from the matplotlib plot figure


        Extract RGBA values from the center of each position bin in a matplotlib figure.
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            The figure containing the subplot
        axd : dict
            Dictionary of axes from subplot_mosaic
        n_pos_bins : int
            Number of position bins
        subplot_name : str, optional
            Name of the subplot to extract from, default 'matplotlib_combined_rgba'
        debug_print : bool, optional
            Whether to print debug information, default False
            
        Returns:
        --------
        center_rgba : ndarray
            RGBA values at the center of each position bin, shape (n_pos_bins, 4)


        Usage:

            center_rgba = MultiDecoderColorOverlayedPosteriors.extract_center_rgba_from_figure(fig, axd, n_pos_bins, debug_print=False)
            # Now you can use center_rgba for further processing
            center_rgba

            plt.figure()
            plt.imshow(center_rgba[:, None, :])


        """
        # Render the figure to get pixel data
        fig.canvas.draw()

        # Get the dimensions of the figure
        img_w, img_h = fig.canvas.get_width_height()
        if debug_print:
            print(f"Figure dimensions: {img_w} x {img_h}")

        # Get the raw pixel data
        img_data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        if debug_print:
            print(f"Raw data shape: {img_data.shape}")
            unique_values = np.unique(img_data)
            print(f"Number of unique values: {len(unique_values)}")
            print(f"Min/max values: {unique_values.min()}, {unique_values.max()}")

        # Reshape the data - ARGB format needs special handling
        img = img_data.reshape(img_h, img_w, 4)

        # Convert from ARGB to RGBA (move alpha from first to last position)
        img = np.roll(img, 3, axis=2)

        if debug_print:
            print(f"Reshaped image dimensions: {img.shape}")

        # Verify we have the subplot we're looking for
        if subplot_name not in axd:
            error_msg = f"ERROR: '{subplot_name}' subplot not found! Available subplots: {list(axd.keys())}"
            if debug_print:
                print(error_msg)
            raise KeyError(error_msg)
        
        # Get the position of the specific subplot
        bbox = axd[subplot_name].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if debug_print:
            print(f"Subplot bbox: {bbox}")
        
        # Calculate pixel coordinates for the subplot
        x_start = int(bbox.x0 * fig.dpi)
        y_start = int(bbox.y0 * fig.dpi)
        x_end = int(bbox.x1 * fig.dpi)
        y_end = int(bbox.y1 * fig.dpi)
        
        if debug_print:
            print(f"Subplot pixel coordinates: ({x_start}, {y_start}) to ({x_end}, {y_end})")
        
        # Extract just the subplot region
        subplot_img = img[y_start:y_end, x_start:x_end, :]
        if debug_print:
            print(f"Subplot image shape: {subplot_img.shape}")
        
        # Calculate center indices
        if debug_print:
            print(f"n_pos_bins: {n_pos_bins}")
        
        # Calculate center indices - assuming vertical arrangement
        subplot_height = subplot_img.shape[0]
        center_indices = np.ceil(subplot_height * (np.arange(n_pos_bins) + 0.5)/float(n_pos_bins)).astype(int)
        if debug_print:
            print(f"Center indices: {center_indices}")
        
        # Check if indices are within bounds
        if max(center_indices) >= subplot_height:
            warning_msg = "WARNING: Some center indices are out of bounds!"
            if debug_print:
                print(warning_msg)
            center_indices = np.clip(center_indices, 0, subplot_height-1)
        
        # Extract center RGBA values - assuming we want the middle of each row
        center_x = subplot_img.shape[1] // 2
        center_rgba = subplot_img[center_indices, center_x, :]
        
        if debug_print:
            print(f"Extracted center RGBA values shape: {center_rgba.shape}")
            print("Sample RGBA values:")
            for i in range(min(5, len(center_indices))):
                print(f"Position {i}: {center_rgba[i]}")
        
        return center_rgba

    @function_attributes(short_name=None, tags=['DEPRICATED', 'UNUSED', 'DEBUGGING'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-06 19:34', related_items=[])
    @classmethod
    def _plot_single_t_bin_images(cls, fig_num=None, debug_print=True, **kwargs):
        """ plots debugging plots for this data
        """
        # from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import numpy_rgba_composite

        plot_ax_values_dict = kwargs

        subplot_ax_names_list = list(plot_ax_values_dict.keys())


        special_single_t_bin_per_decoder_RGBA_var_name: str = 'single_t_bin_per_decoder_alpha_weighted_RGBA'
        single_t_bin_per_decoder_RGBA = plot_ax_values_dict.get(special_single_t_bin_per_decoder_RGBA_var_name, None)
        if single_t_bin_per_decoder_RGBA is not None:
            subplot_ax_names_list.append(f'programmatic_test_{special_single_t_bin_per_decoder_RGBA_var_name}')
            subplot_ax_names_list.append(f'matplotlib_{special_single_t_bin_per_decoder_RGBA_var_name}')
            

        fig = plt.figure(num=fig_num, layout="constrained", clear=True)
        axd = fig.subplot_mosaic(
            [
                # ["pre_norm", "all_normed", "normed", "rgba", "_realphaed_single_t_bin_out_RGBA", "final_combined_rgba"],
                # ["pre_norm", "normed", "rgba", "_realphaed_single_t_bin_out_RGBA", "final_combined_rgba", "matplotlib_combined_rgba"],
                subplot_ax_names_list,
                # ["main", "BLANK"],
            ],
            sharey=True
        )

        ## Plot the single time bin figure
        # _pre_norm_prob_vals = MultiDecoderColorOverlayedPosteriors._prepare_arr_for_conversion_to_RGBA(_pre_norm_prob_vals, drop_below_threshold=drop_below_threshold)
        # _all_normed_prob_vals = _prepare_arr_for_conversion_to_RGBA(_all_normed_prob_vals, drop_below_threshold=drop_below_threshold)

        # final_overlayed_single_t_bin_out_RGBA = final_overlayed_single_t_bin_out_RGBA[:, None, :]
        # img = deepcopy(final_overlayed_single_t_bin_out_RGBA)[:, None, :]
        # final_rgb_on_white = blend_over_white(final_overlayed_single_t_bin_out_RGBA)  # (n_pos_bins, 3)
        # img = final_rgb_on_white[:, None, :]               # (n_pos_bins, 1, 3)


        # axd['pre_norm'].imshow(_pre_norm_prob_vals)
        # axd['all_normed'].imshow(_all_normed_prob_vals)
        # axd['normed'].imshow(probability_values)
        # axd['rgba'].imshow(single_t_bin_out_RGBA)
        # axd['_realphaed_single_t_bin_out_RGBA'].imshow(_realphaed_single_t_bin_out_RGBA)


        for ax_name, vals in plot_ax_values_dict.items():
            vals_shape = np.shape(vals)
            if (len(vals_shape) == 2) and (vals_shape[-1] == 4):
                vals = vals[:, None, :] ## matplotlib assume's its not an RGBA format dict and plots it wrongly
                
            axd[ax_name].imshow(vals)
            axd[ax_name].set_title(ax_name)    


        ## plot the computed one (which doesn't work):
        # axd['final_combined_rgba'].imshow(img)


        if single_t_bin_per_decoder_RGBA is not None:
            
            ax_name = f'programmatic_test_{special_single_t_bin_per_decoder_RGBA_var_name}'
            rgba_layers = np.transpose(single_t_bin_per_decoder_RGBA, (1, 0, 2))[:, :, None, :]  # (n_decoders, H=n_pos_bins, W=1, 4)
            print(f'numpy_rgba_composite(...):')
            print(f'\trgba_layers.shape: {np.shape(rgba_layers)}')
            final_rgba = numpy_rgba_composite(rgba_layers, debug_print=debug_print)  # (H, W, 4) (59, 1, 4)
            if debug_print:
                print(f'\tfinal_rgba.shape: {np.shape(final_rgba)} (post call to `numpy_rgba_composite(...)`)')
            axd[ax_name].imshow(final_rgba, aspect='auto')
            axd[ax_name].set_title(ax_name)
            print(f'\tfinal_rgba.shape: {np.shape(final_rgba)}')

            ## use matplotlib's rendering to get the final output image:
            ax_name = f'matplotlib_{special_single_t_bin_per_decoder_RGBA_var_name}'
            for i in np.arange(4):
                an_img = single_t_bin_per_decoder_RGBA[:, i, :]
                # an_img = _realphaed_single_t_bin_out_RGBA[:, i, :] # .shape
                an_img = an_img[:, None, :]
                axd[ax_name].imshow(an_img, zorder=i)
                print(f'\t\ti:{i}, an_img.shape: {np.shape(an_img)}')
            axd[ax_name].set_title(ax_name)
            
        return fig, axd


    ## INPUTS: extra_all_t_bins_outputs_dict

    @function_attributes(short_name=None, tags=['plot', 'alt', 'WORKING', 'per_decoder'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-06 19:02', related_items=[])
    @classmethod
    def plot_mutli_t_bin_p_x_given_n_per_decoder(cls, all_t_bins_per_decoder_out_RGBA: NDArray, time_bin_centers=None, xbin=None, t_bin_size = 0.05, ax=None, use_original_bounds=False) -> GenericMatplotlibContainer:
        """ plots the *per_decoder* result onto a single matplotlib axes by iterating over the decoders and using matplotlib's built-in overlay/composition (which works better than the manual attempt used in `plot_mutli_t_bin_RGBA_image`)
        

        Working! Uses a single matplotlib ax to draw `n_decoders` images on top of each other by calling `imshow` sequentially.
        
        Usage:
        
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool import GenericMatplotlibContainer
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MultiDecoderColorOverlayedPosteriors
            
            
            _out: GenericMatplotlibContainer = MultiDecoderColorOverlayedPosteriors.plot_mutli_t_bin_p_x_given_n(extra_all_t_bins_outputs_dict['all_t_bins_per_decoder_alpha_weighted_RGBA'], xbin=xbin, time_bin_centers=time_bin_centers, t_bin_size=0.025, ax=ax)
            fig = _out.fig
            ax = _out.ax
            im_posterior_x_dict = _out.plots.im_posterior_x_dict
        """
        # all_t_bins_per_decoder_out_RGBA: NDArray[ND.Shape["N_TIME_BINS", "N_POS_BINS", "N_DECODERS", "4"], np.floating]
        
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import _subfn_try_get_approximate_recovered_t_pos
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool import GenericMatplotlibContainer

        n_time_bins, n_pos_bins, n_decoders, _n_RGBA_channels = np.shape(all_t_bins_per_decoder_out_RGBA)
        assert _n_RGBA_channels == 4

        if xbin is None:
            xbin = np.arange(n_pos_bins) + 0.5
            use_original_bounds = True
            
        if time_bin_centers is None:
            time_bin_centers = (np.arange(n_time_bins) * t_bin_size) + (0.5 * t_bin_size)
            use_original_bounds = True
            
        # rgba.shape (n_pos_bins, 4) # the 4 here is for RGBA, not the decoders
        # plt.imshow(rgba)

        # img = rgba[:,None,:]            # (59,1,4)
        if ax is None:
            plt.style.use('dark_background')
            # fig = plt.figure(num='NEW_FINAL all_t_bins_final_RGBA', layout="constrained", clear=True)
            fig, ax = plt.subplots(num='NEW_FINAL plot_mutli_t_bin_p_x_given_n', layout="constrained", clear=True)
        else:
            fig = ax.get_figure()           

        # img = np.transpose(active_subset_p_x_given_n, (1, 0, 2))
        
        # Compute extents for imshow:
        main_plot_kwargs = {
            'origin': 'lower',
            'vmin': 0,
            'vmax': 1,
            # 'cmap': cmap,
            'interpolation':'nearest',
            'aspect':'auto',
        } # merges `posterior_heatmap_imshow_kwargs` into main_plot_kwargs, replacing the existing values if present in both
        # Posterior distribution heatmaps at each point.
        enable_set_axis_limits:bool=True
        
        _out: GenericMatplotlibContainer = GenericMatplotlibContainer(name='plot_mutli_t_bin_p_x_given_n')
        _out.fig = fig
        _out.ax = ax 



        ## Determine the actual start/end times:
        x_first_extent = _subfn_try_get_approximate_recovered_t_pos(time_bin_centers, xbin, use_original_bounds=use_original_bounds) # x_first_extent = (xmin, xmax, ymin, ymax)
        xmin, xmax, ymin, ymax = x_first_extent

        _out.plots.im_posterior_x_dict = {}
        
        ## use matplotlib's rendering to get the final output image:
        for i in np.arange(n_decoders):
            an_img = all_t_bins_per_decoder_out_RGBA[:, :, i, :] # .shape (n_time_bins, n_pos_bins, _n_RGBA_channels)
            # an_img = _realphaed_single_t_bin_out_RGBA[:, i, :] # .shape (n_time_bins, n_pos_bins, _n_RGBA_channels)
            # an_img = an_img[:, None, :]
            # ax.imshow(an_img, zorder=i)
            # print(f'\t\ti:{i}, an_img.shape: {np.shape(an_img)}')

            an_img = np.transpose(an_img, (1, 0, 2))
            # print(f'\t\ti:{i}, an_img.shape: {np.shape(an_img)}')

            try:
                im_posterior_x = ax.imshow(an_img, zorder=i, extent=x_first_extent, **(dict(animated=False) | main_plot_kwargs)) 

            except ValueError as err:
                # ValueError: Axis limits cannot be NaN or Inf
                print(f'WARN: active_extent (xmin, xmax, ymin, ymax): {x_first_extent} contains NaN or Inf.\n\terr: {err}')
                # ax.clear() # clear the existing and now invalid image
                im_posterior_x = None
            _out.plots.im_posterior_x_dict[i] = im_posterior_x
        # end for i in np.arange(n_decoders)
        if enable_set_axis_limits:
            ax.set_xlim((xmin, xmax)) # UserWarning: Attempting to set identical low and high xlims makes transformation singular; automatically expanding.
            ax.set_ylim((ymin, ymax))        

        # return fig, ax, _out.plots.im_posterior_x_dict
        return _out



    @function_attributes(short_name=None, tags=['plot', 'matplotlib', 'RGBA_image'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-04 00:00', related_items=[])
    @classmethod
    def plot_mutli_t_bin_RGBA_image(cls, all_t_bins_final_RGBA, time_bin_centers=None, xbin=None, start_t_bin_idx: int = 0, desired_n_seconds: Optional[float] = None, t_bin_size = 0.05, ax=None, use_original_bounds=False):
        """ plots a portion of the color-merged *RGBA_image* onto a matplotlib axes 

        Usage:        
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MultiDecoderColorOverlayedPosteriors

            # all_t_bins_final_RGBA = extra_all_t_bins_outputs_dict['all_t_bins_final_RGBA']
            # all_t_bins_final_RGBA.shape # (69488, 59, 4)

            start_t_bin_idx: int = 1338
            desired_n_seconds: float = 60.0
            ax = None
            fig, ax, im_posterior_x = MultiDecoderColorOverlayedPosteriors.plot_mutli_t_bin_RGBA_image(all_t_bins_final_RGBA=all_t_bins_per_decoder_out_RGBA, start_t_bin_idx=start_t_bin_idx, desired_n_seconds=desired_n_seconds, t_bin_size=0.05, ax=ax)
            plt.show()

        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import _subfn_try_get_approximate_recovered_t_pos
        
        if desired_n_seconds is not None:
            n_samples: int = int(np.ceil(desired_n_seconds / t_bin_size))
        else:
            n_samples: int = np.shape(all_t_bins_final_RGBA)[0]

        print(f'n_samples: {n_samples}')

        # n_samples = 2
        subset_t_bin_indicies = np.arange(n_samples)
        if start_t_bin_idx is not None:
            subset_t_bin_indicies += start_t_bin_idx

        active_subset_all_t_bins_final_RGBA = deepcopy(all_t_bins_final_RGBA[subset_t_bin_indicies, :, :])

        # active_subset_all_t_bins_final_RGBA.shape
        n_dims: int = np.ndim(active_subset_all_t_bins_final_RGBA)
        if n_dims == 4:
            ## separate overlayed matplotlib version
            n_time_bins, n_pos_bins, n_decoders, _n_rgba_channels = np.shape(active_subset_all_t_bins_final_RGBA)
        elif n_dims == 3:
            ## how can this case happen?
            n_pos_bins, n_decoders, _n_rgba_channels = np.shape(active_subset_all_t_bins_final_RGBA)
        else:
            raise NotImplementedError(f'Nope: np.shape(active_subset_all_t_bins_final_RGBA): {np.shape(active_subset_all_t_bins_final_RGBA)}')

        # n_time_bins, n_pos_bins, n_decoders, _n_rgba_channels = np.shape(active_subset_all_t_bins_final_RGBA)

        
        if xbin is None:
            xbin = np.arange(n_pos_bins) + 0.5
            use_original_bounds = True
            
        if time_bin_centers is None:
            time_bin_centers = (subset_t_bin_indicies * t_bin_size) + (0.5 * t_bin_size)
            use_original_bounds = True
            
        # rgba.shape (n_pos_bins, 4) # the 4 here is for RGBA, not the decoders
        # plt.imshow(rgba)

        # img = rgba[:,None,:]            # (59,1,4)
        if ax is None:
            plt.style.use('dark_background')
            # fig = plt.figure(num='NEW_FINAL all_t_bins_final_RGBA', layout="constrained", clear=True)
            fig, ax = plt.subplots(num='NEW_FINAL all_t_bins_final_RGBA', layout="constrained", clear=True)
        else:
            fig = ax.get_figure()           

        img = np.transpose(active_subset_all_t_bins_final_RGBA, (1, 0, 2))
        
        # Compute extents for imshow:
        main_plot_kwargs = {
            'origin': 'lower',
            'vmin': 0,
            'vmax': 1,
            # 'cmap': cmap,
            'interpolation':'nearest',
            'aspect':'auto',
        } # merges `posterior_heatmap_imshow_kwargs` into main_plot_kwargs, replacing the existing values if present in both
        # Posterior distribution heatmaps at each point.
        enable_set_axis_limits:bool=True
        
        ## Determine the actual start/end times:
        x_first_extent = _subfn_try_get_approximate_recovered_t_pos(time_bin_centers, xbin, use_original_bounds=use_original_bounds) # x_first_extent = (xmin, xmax, ymin, ymax)
        xmin, xmax, ymin, ymax = x_first_extent

        try:
            im_posterior_x = ax.imshow(img, extent=x_first_extent, **(dict(animated=False) | main_plot_kwargs)) 
            # assert xmin < xmax
            if enable_set_axis_limits:
                ax.set_xlim((xmin, xmax)) # UserWarning: Attempting to set identical low and high xlims makes transformation singular; automatically expanding.
                ax.set_ylim((ymin, ymax))
        except ValueError as err:
            # ValueError: Axis limits cannot be NaN or Inf
            print(f'WARN: active_extent (xmin, xmax, ymin, ymax): {x_first_extent} contains NaN or Inf.\n\terr: {err}')
            # ax.clear() # clear the existing and now invalid image
            im_posterior_x = None
            

        return fig, ax, im_posterior_x





    @function_attributes(short_name=None, tags=['UNFINISHED', 'TODO', 'TODO_2025-05-04'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-04 20:01', related_items=[])
    @classmethod
    def export_portion_as_images(cls, all_t_bins_out_RGBA, file_save_path='output/all_time_bins.pdf'):
        """ 
        
        MultiDecoderColorOverlayedPosteriors.export_portion_as_images(all_t_bins_out_RGBA=all_t_bins_out_RGBA, file_save_path='output/all_time_bins.pdf')
        
        
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        # Save each time bin as a separate image
        # for t_idx in range(all_t_bins_out_RGBA.shape[0]):
        #     # Create a figure with subplots for each decoder
        #     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
        #     for decoder_idx in range(4):
        #         # Extract RGBA data for this time bin and decoder
        #         rgba_data = all_t_bins_out_RGBA[t_idx, :, decoder_idx, :]
                
        #         # Display as an image
        #         axes[decoder_idx].imshow(rgba_data.reshape(1, -1, 4))
        #         axes[decoder_idx].set_title(f'Decoder {decoder_idx}')
            
        #     plt.tight_layout()
        #     plt.savefig(f'time_bin_{t_idx}.png')
        #     plt.close()

        # Or save all time bins in a single PDF
        with PdfPages(file_save_path, keep_empty=False) as pdf:
            for t_idx in range(all_t_bins_out_RGBA.shape[0]):
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                for decoder_idx in range(4):
                    rgba_data = all_t_bins_out_RGBA[t_idx, :, decoder_idx, :]
                    axes[decoder_idx].imshow(rgba_data.reshape(1, -1, 4))
                    axes[decoder_idx].set_title(f'Decoder {decoder_idx}')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()


    # ==================================================================================================================================================================================================================================================================================== #
    # SpikeRaster2D Decoder Tracks                                                                                                                                                                                                                                                         #
    # ==================================================================================================================================================================================================================================================================================== #

    
    @function_attributes(short_name=None, tags=['track', 'SpikeRaster2D', 'private'], input_requires=[], output_provides=[], uses=['.plot_mutli_t_bin_p_x_given_n'], used_by=[], creation_date='2025-05-06 16:08', related_items=[])
    @classmethod
    def _perform_add_as_track_to_spike_raster_window(cls, active_2d_plot, all_t_bins_final_RGBA, time_bin_centers=None, xbin=None, t_bin_size = 0.025, dock_identifier: str = 'MergedColorPlot'):
        """ adds to SpikeRaster2D as a multi-color context-weighted decoding as a track
        
        
        Usage:
        
            all_t_bins_final_RGBA = extra_all_t_bins_outputs_dict['all_t_bins_final_RGBA']

            ts_widget, fig, ax_list, dDisplayItem = MultiDecoderColorOverlayedPosteriors.add_as_track_to_spike_raster_window(active_2d_plot, all_t_bins_final_RGBA=extra_all_t_bins_outputs_dict['all_t_bins_per_decoder_alpha_weighted_RGBA'], t_bin_size=0.05)

        
        """
        # from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MultiDecoderColorOverlayedPosteriors
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import SynchronizedPlotMode


        ## Build the new dock track:
        
        ts_widget, fig, ax_list, dDisplayItem = active_2d_plot.add_new_matplotlib_render_plot_widget(name=dock_identifier)
        ax = ax_list[0]
        ax.clear()

        ## INPUTS: time_bin_centers, all_t_bins_final_RGBA, xbin
        # all_t_bins_final_RGBA.shape # (69488, 59, 4)
        # fig, ax, im_posterior_x = cls.plot_mutli_t_bin_RGBA_image(all_t_bins_final_RGBA=all_t_bins_final_RGBA, xbin=xbin, time_bin_centers=time_bin_centers, t_bin_size=t_bin_size, ax=ax)
        _out: GenericMatplotlibContainer = cls.plot_mutli_t_bin_p_x_given_n_per_decoder(all_t_bins_final_RGBA, xbin=xbin, time_bin_centers=time_bin_centers, t_bin_size=t_bin_size, ax=ax)
        fig = _out.fig
        ax = _out.ax
        im_posterior_x_dict = _out.plots.im_posterior_x_dict

        ## sync up the widgets
        active_2d_plot.sync_matplotlib_render_plot_widget(dock_identifier, sync_mode=SynchronizedPlotMode.TO_WINDOW)
        # active_2d_plot.sync_matplotlib_render_plot_widget(dock_identifier, sync_mode=SynchronizedPlotMode.TO_GLOBAL_DATA)
        fig.canvas.draw()

        ## OUTPUTS: fig, ax, out_plot_data  
        return ts_widget, fig, ax_list, dDisplayItem
        


    @function_attributes(short_name=None, tags=['tracks', 'plotting'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-13 00:04', related_items=[])
    def add_tracks_to_spike_raster_window(self, active_2d_plot, dock_identifier_prefix: str = 'MergedColorPlot'):
        """ adds two separate tracks to SpikeRaster2D, one for each decoding style (four vs. two) 
                Each added track is a multi-color context-weighted decoding plot
                
        Usage:
                    
            multi_decoder_color_overlay: MultiDecoderColorOverlayedPosteriors = MultiDecoderColorOverlayedPosteriors(p_x_given_n=p_x_given_n, time_bin_centers=time_bin_centers, xbin=xbin, lower_bound_alpha=0.1, drop_below_threshold=1e-3, t_bin_size=0.025)
            multi_decoder_color_overlay.compute_all()
            _out_display_dict = multi_decoder_color_overlay.add_tracks_to_spike_raster_window(active_2d_plot=active_2d_plot, dock_identifier_prefix='MergedColorPlot')

        
        """
        from neuropy.utils.mixins.binning_helpers import get_bin_edges
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _plot_low_firing_time_bins_overlay_image
        

        time_bin_edges: NDArray = get_bin_edges(deepcopy(self.time_bin_centers))
        spikes_df: pd.DataFrame = deepcopy(self.spikes_df)
        # unique_units = np.unique(spikes_df['aclu']) # sorted
        unit_specific_time_binned_spike_counts, unique_units, (is_time_bin_active, inactive_mask, low_firing_bins_mask_rgba) = spikes_df.spikes.compute_unit_time_binned_spike_counts_and_mask(time_bin_edges=time_bin_edges)


        _out_display_dict = {}
        
        if 'four_decoders' in self.extra_all_t_bins_outputs_dict_dict:
            a_result_name: str = 'four_decoders'
            dock_identifier: str = f'{dock_identifier_prefix}-AlphaWeighted-{a_result_name}'
            active_colors_dict = self.active_colors_dict_dict[a_result_name]
            
            _out_tuple = self._perform_add_as_track_to_spike_raster_window(active_2d_plot=active_2d_plot, all_t_bins_final_RGBA=self.extra_all_t_bins_outputs_dict_dict[a_result_name]['all_t_bins_per_decoder_alpha_weighted_RGBA'], time_bin_centers=self.time_bin_centers, xbin=self.xbin, t_bin_size=self.t_bin_size, dock_identifier=dock_identifier)
            ts_widget, fig, ax_list, dDisplayItem = _out_tuple
            ## Add overlay plot to hide bins that don't meet the firing criteria:
            low_firing_bins_image = _plot_low_firing_time_bins_overlay_image(widget=ts_widget, time_bin_edges=time_bin_edges, mask_rgba=low_firing_bins_mask_rgba, zorder=11)
            try:
                ax_inset, rounded_rect_inset =  self.add_decoder_legend_venn(all_decoder_colors_dict=active_colors_dict, ax=ax_list[0], zorder=13)
                ts_widget.plots['decoder_legend_venn'] = dict(ax_inset=ax_inset, background_rounded_rect_inset=rounded_rect_inset)
            except ModuleNotFoundError as e:
                print(f'WARN: {e}. decoder_legend venn will be missing!')
                pass
            except Exception as e:
                raise e

            # _out_display_dict[dock_identifier] = _out_tuple
            _out_display_dict[dock_identifier] = (ts_widget, fig, ax_list, dDisplayItem)


        if 'two_decoders' in self.extra_all_t_bins_outputs_dict_dict:
            a_result_name: str = 'two_decoders'
            dock_identifier: str = f'{dock_identifier_prefix}-AlphaWeighted-{a_result_name}'
            active_colors_dict = self.active_colors_dict_dict[a_result_name]
            
            _out_tuple = self._perform_add_as_track_to_spike_raster_window(active_2d_plot=active_2d_plot, all_t_bins_final_RGBA=self.extra_all_t_bins_outputs_dict_dict[a_result_name]['all_t_bins_per_decoder_alpha_weighted_RGBA'], time_bin_centers=self.time_bin_centers, xbin=self.xbin, t_bin_size=self.t_bin_size, dock_identifier=dock_identifier)
            ts_widget, fig, ax_list, dDisplayItem = _out_tuple
            ## Add overlay plot to hide bins that don't meet the firing criteria:
            low_firing_bins_image = _plot_low_firing_time_bins_overlay_image(widget=ts_widget, time_bin_edges=time_bin_edges, mask_rgba=low_firing_bins_mask_rgba, zorder=11)
            try:
                ax_inset, rounded_rect_inset =  self.add_decoder_legend_venn(all_decoder_colors_dict=active_colors_dict, ax=ax_list[0], zorder=13)
                ts_widget.plots['decoder_legend_venn'] = dict(ax_inset=ax_inset, background_rounded_rect_inset=rounded_rect_inset)
            except ModuleNotFoundError as e:
                print(f'WARN: {e}. decoder_legend venn will be missing!')
                pass
            except Exception as e:
                raise e 

            # _out_display_dict[dock_identifier] = _out_tuple
            _out_display_dict[dock_identifier] = (ts_widget, fig, ax_list, dDisplayItem)

        return _out_display_dict


    # ==================================================================================================================================================================================================================================================================================== #
    # Decoder Legend Venn Diagram:                                                                                                                                                                                                                                                         #
    # ==================================================================================================================================================================================================================================================================================== #

    @classmethod
    def add_decoder_legend_venn(cls, all_decoder_colors_dict: Dict[str, str], ax, defer_render:bool=False, zorder:float=137.0):
        """ Creates an inset axes to serve as the legned, and inside this plots a venn-diagram showing the colors for each decoder, allowing the user to see what they look like overlapping

        Usage:

            all_decoder_colors_dict = {'long_LR': '#4169E1', 'long_RL': '#607B00', 'short_LR': '#DC143C', 'short_RL': '#990099'} ## Just hardcoded version of `additional_cmap_names`
            ax_inset =  MultiDecoderColorOverlayedPosteriors.add_decoder_legend_venn(all_decoder_colors_dict=all_decoder_colors_dict, ax=ax)

        """
        import matplotlib.patches as patches

        assert ax is not None
        fig = ax.get_figure()

        # n_decoders: int = len(all_decoder_colors_dict)
        # cmap = list(all_decoder_colors_dict.values())

        # Create the inset axes
        # Define the size and position of the inset axes relative to the parent axes    
        width = 0.1
        height = 0.65

        # Define the position of the lower-left corner for top-right alignment
        x0 = (1 - width)
        y0 = (1 - height)

        # Create the inset axes, using bbox_to_anchor and bbox_transform for independent positioning
        ax_inset = ax.inset_axes([x0, y0, width, height])
        ax_inset =  cls._build_decoder_legend_venn(all_decoder_colors_dict=all_decoder_colors_dict, ax=ax_inset)
        ax_inset.set_zorder(zorder)  # Set zorder after creation
        
        # Now, add the rounded rectangle patch to the inset axes (ax_inset)
        # We define the rectangle in the coordinate system of ax_inset
        # (0, 0) is the bottom-left corner of ax_inset
        # 1 is the width covering 100% of ax_inset's width
        # 1 is the height covering 100% of ax_inset's height
        rounded_rect_inset = patches.FancyBboxPatch((0, 0), 1, 1,
                                                    boxstyle="round,pad=0.5",
                                                    facecolor='#f5e9e9fa',  # Face color set to white
                                                    edgecolor='#585252fa',
                                                    alpha=0.7, # Alpha set to 0.7
                                                    # mutation_scale=30, # Adjust for desired rounding
                                                    transform=ax_inset.transAxes, # Crucially, use the inset axes' transform
                                                    zorder=(zorder-5)) # Set a low zorder to be in the background

        # Add the patch to the inset axes
        ax_inset.add_patch(rounded_rect_inset)

        # # Optional: Set limits for the inset axes if needed
        ax_inset.set_xlim(0, 1)
        ax_inset.set_ylim(0, 1)
        # # fig.canvas.draw()

        if not defer_render:
            fig.canvas.draw()

        return ax_inset, rounded_rect_inset

    @classmethod
    def _build_decoder_legend_venn(cls, all_decoder_colors_dict: Dict[str, str], ax=None):
        """ builds a simple venn-diagram showing the colors for each decoder, allowing the user to see what they look like overlapping

        Usage:

            all_decoder_colors_dict = {'long_LR': '#4169E1', 'long_RL': '#607B00', 'short_LR': '#DC143C', 'short_RL': '#990099'} ## Just hardcoded version of `additional_cmap_names`
            ax =  MultiDecoderColorOverlayedPosteriors._build_decoder_legend_venn(all_decoder_colors_dict=all_decoder_colors_dict, ax=None)

        """
        from matplotlib.pyplot import subplots
        from venn import draw_venn, generate_colors

        n_sets: int = len(all_decoder_colors_dict)
        cmap = list(all_decoder_colors_dict.values())
        if ax is None:
            ## make new figure if needed
            fig, ax = subplots(figsize=(18, 8))
            
        dataset_dict = {k:{100} for k, v in deepcopy(all_decoder_colors_dict).items()}

        # _out = venn(dataset_dict, fmt="{percentage:.1f}%", cmap=cmap, fontsize=8, legend_loc="upper left", ax=ax)
        petal_labels = {} ## empty labels
        ax = draw_venn(
            petal_labels=petal_labels, dataset_labels=dataset_dict.keys(),
            hint_hidden=False,
            colors=generate_colors(cmap=cmap, n_colors=n_sets),
            # colors=cmap,
            figsize=(8, 8), fontsize=8, legend_loc="upper left", ax=ax
        )

        # # Now, add the rounded rectangle patch to the inset axes (ax_inset)
        # # We define the rectangle in the coordinate system of ax_inset
        # # (0, 0) is the bottom-left corner of ax_inset
        # # 1 is the width covering 100% of ax_inset's width
        # # 1 is the height covering 100% of ax_inset's height
        # rounded_rect_inset = patches.FancyBboxPatch((0, 0), 1, 1,
        #                                             boxstyle="round,pad=0.5",
        #                                             facecolor='#f5e9e9fa',  # Face color set to white
        #                                             edgecolor='#585252fa',
        #                                             alpha=0.7, # Alpha set to 0.7
        #                                             # mutation_scale=30, # Adjust for desired rounding
        #                                             transform=ax.transAxes, # Crucially, use the inset axes' transform
        #                                             zorder=-5) # Set a low zorder to be in the background

        # # Add the patch to the inset axes
        # ax.add_patch(rounded_rect_inset)

        # # # # Optional: Set limits for the inset axes if needed
        # # ax.set_xlim(0, 1)
        # # ax.set_ylim(0, 1)
        # # # fig.canvas.draw()
            
        return ax

    # ==================================================================================================================== #
    # HDF5                                                                                                                 #
    # ==================================================================================================================== #
    @classmethod
    def save_hdf(cls, hdf5_save_file_path: Path, extra_all_t_bins_outputs_dict, **kwargs):
        """ 

        Usage:
            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import MultiDecoderColorOverlayedPosteriors

            ## INPUTS: all_t_bins_final_overlayed_out_RGBA, all_t_bins_per_decoder_out_RGBA
            hdf5_save_file_path = Path(f'output/2025-05-11_all_t_bins_out_RGBA_thresh.h5').resolve()
            
            MultiDecoderColorOverlayedPosteriors.save_hdf(hdf5_save_file_path=hdf5_save_file_path, extra_all_t_bins_outputs_dict=extra_all_t_bins_outputs_dict, drop_below_threshold=drop_below_threshold,
                                                        all_t_bins_final_overlayed_out_RGBA=all_t_bins_final_overlayed_out_RGBA, all_t_bins_per_decoder_out_RGBA=all_t_bins_per_decoder_out_RGBA,
                                                        
            )


        
        Load with reciprocal:
        
            hdf5_load_file_path = Path('output/2025-05-11_all_t_bins_out_RGBA_thresh.h5').resolve()
            all_t_bins_final_RGBA = MultiDecoderColorOverlayedPosteriors.load_hdf(hdf5_load_file_path=hdf5_load_file_path) # Initialize variable

            ## OUTPUTS: all_t_bins_final_RGBA
            all_t_bins_final_RGBA

        """
        import h5py

        drop_below_threshold = kwargs.pop('drop_below_threshold', None)
        all_t_bins_final_overlayed_out_RGBA = kwargs.pop('all_t_bins_final_overlayed_out_RGBA', None)
        all_t_bins_per_decoder_out_RGBA = kwargs.pop('all_t_bins_per_decoder_out_RGBA', None)

        ## INPUTS: all_t_bins_final_overlayed_out_RGBA, all_t_bins_per_decoder_out_RGBA
        # hdf5_save_file_path: Path = Path(f'output/2025-05-09_all_t_bins_out_RGBA_thresh.h5').resolve()
        with h5py.File(hdf5_save_file_path, 'w') as f:
            if drop_below_threshold:
                f.create_dataset('drop_below_threshold', data=drop_below_threshold)
            if all_t_bins_final_overlayed_out_RGBA is None:
                f.create_dataset('final_overlayed_rgba_data', data=all_t_bins_final_overlayed_out_RGBA, compression='gzip', compression_opts=9)
            if all_t_bins_per_decoder_out_RGBA is not None:
                f.create_dataset('rgba_data', data=all_t_bins_per_decoder_out_RGBA, compression='gzip', compression_opts=9)
                
            for k, v in extra_all_t_bins_outputs_dict.items():
                print(f'writing key "{k}"...')
                f.create_dataset(k, data=v, compression='gzip', compression_opts=9)

            print(f'hdf5_save_file_path: "{hdf5_save_file_path}"')

    @classmethod
    def load_hdf(cls, hdf5_load_file_path: Path):
        """ 
        """
        import h5py # For loading data from HDF5 file

        # --- Data Preparation ---
        # Attempt to load data from HDF5 file first.
        # If loading fails, fall back to placeholder data.

        # hdf5_load_file_path: Path = Path('output/2025-05-06_all_t_bins_out_RGBA_thresh.h5').resolve()
        data_loaded_from_hdf5 = False
        all_t_bins_final_RGBA = None # Initialize variable

        print(f"Attempting to load data from: {hdf5_load_file_path}")

        if hdf5_load_file_path.exists():
            try:
                with h5py.File(hdf5_load_file_path, 'r') as f:
                    # The key 'all_t_bins_final_RGBA' is based on the original script's usage:
                    # all_t_bins_final_RGBA = extra_all_t_bins_outputs_dict['all_t_bins_final_RGBA']
                    # and the saving loop: f.create_dataset(k, data=v, ...)
                    dataset_key = 'all_t_bins_final_RGBA'
                    
                    if dataset_key in f:
                        all_t_bins_final_RGBA = f[dataset_key][:] # Load the entire dataset into memory
                        print(f"Successfully loaded '{dataset_key}' from {hdf5_load_file_path}")
                        print(f"Loaded data shape: {all_t_bins_final_RGBA.shape}")
                        
                        # Basic validation of the loaded data structure
                        if not (isinstance(all_t_bins_final_RGBA, np.ndarray) and
                                all_t_bins_final_RGBA.ndim == 3 and
                                all_t_bins_final_RGBA.shape[2] == 4):
                            print(f"Warning: Loaded data for '{dataset_key}' does not have the expected "
                                f"3D shape (time_bins, spatial_bins, 4 channels). "
                                f"Found shape: {all_t_bins_final_RGBA.shape}. Problems might occur.")
                            # Decide if this should be a fatal error or allow continuation
                        
                        data_loaded_from_hdf5 = True
                    else:
                        print(f"Error: Dataset '{dataset_key}' not found in HDF5 file {hdf5_load_file_path}.")
            except Exception as e:
                print(f"Error loading data from HDF5 file {hdf5_load_file_path}: {e}")
        else:
            print(f"Error: HDF5 file not found at {hdf5_load_file_path}.")


        # data_loaded_from_hdf5
        ## OUTPUTS: all_t_bins_final_RGBA
        if data_loaded_from_hdf5:
            return all_t_bins_final_RGBA
        else:
            return None







@function_attributes(short_name=None, tags=['private'], input_requires=[], output_provides=[], uses=[], used_by=['build_decoder_prob_as_a_function_of_position'], creation_date='2025-05-02 12:54', related_items=[])
def _subfn_helper_process_epochs_result_dict(epochs_result_dict: Dict[types.KnownNamedDecodingEpochsType, DecodedFilterEpochsResult], t_delta: float):
    epoch_name_by_pre_post_delta_category_output_dict_dict: Dict[types.KnownNamedDecodingEpochsType, Dict[types.PrePostDeltaCategory, DecodedFilterEpochsResult]] = {}

    for an_epoch_name, an_out_result in epochs_result_dict.items():
        print(f'an_epoch_name: "{an_epoch_name}"')
        an_out_result.filter_epochs = an_out_result.filter_epochs.pho_LS_epoch.adding_pre_post_delta_category_if_needed(t_delta=t_delta)
        pre_post_delta_epoch_decoded_marginal_posterior_df_dict = an_out_result.filter_epochs.pho.partition_df_dict('pre_post_delta_category') # pre_post_delta_category
        ## note that the parititoning produces the hyphen separated string values and we want the unscore-separated ones (as specified in `types.PrePostDeltaCategory`) instead, so we need to manually provide the indexes
        time_col: str = TimeColumnAliasesProtocol.find_first_extant_suitable_columns_name(an_out_result.filter_epochs, col_connonical_name='start', required_columns_synonym_dict={"start":{'begin','start_t','ripple_start_t','lap_start_t'}}, should_raise_exception_on_fail=False)
        print(f'\ttime_col: "{time_col}"')
        an_out_result_dict: Dict[types.PrePostDeltaCategory, DecodedFilterEpochsResult] = {'pre_delta': an_out_result.filtered_by_epoch_times(pre_post_delta_epoch_decoded_marginal_posterior_df_dict['pre-delta'][time_col]),
                            'post_delta': an_out_result.filtered_by_epoch_times(pre_post_delta_epoch_decoded_marginal_posterior_df_dict['post-delta'][time_col]),
        }
        # an_out_result_dict['pre-delta']
        epoch_name_by_pre_post_delta_category_output_dict_dict[an_epoch_name] = deepcopy(an_out_result_dict)


    ## OUTPUTS: epoch_name_by_pre_post_delta_category_output_dict_dict
    return epoch_name_by_pre_post_delta_category_output_dict_dict


@function_attributes(short_name='decoded_prob_as_fn_of_pos', tags=['position', 'debug', 'plot'], input_requires=[], output_provides=[], uses=['_subfn_helper_process_epochs_result_dict'], used_by=[], creation_date='2025-05-02 12:54', related_items=[])
def build_decoder_prob_as_a_function_of_position(epochs_result_dict: Dict[types.KnownNamedDecodingEpochsType, DecodedFilterEpochsResult], xbin_centers, t_delta: float, grid_bin_bounds=None, is_split_by_all_decoders:bool = True, debug_print=False,
                                                 enable_per_decoder_renormalization: bool=True, enable_per_position_bin_renormalization: bool=False, enable_LS_renormalization: bool=False, num='build_decoder_prob_as_a_function_of_position', **kwargs):
    """ Plots the Decoder Context Probability as a function of position (split by pre/post-delta, decoded epochs) to check for bias of certain positions to decode to certain decoder-context (like the right-long endcap region might have a strong bias towards 'long_LR' pre-delta.)

    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import build_decoder_prob_as_a_function_of_position
    
        
        ## INPUTS: a_laps_decoder, a_laps_result, a_pbes_result
        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()

        epochs_result_dict: Dict[types.KnownNamedDecodingEpochsType, DecodedFilterEpochsResult] = {'laps': deepcopy(a_laps_result),
                            'pbe': deepcopy(a_pbes_result),
        }
        
        ## all four decoders as separate rows:
        fig, ax_dict, epoch_name_by_pre_post_delta_category_output_dict_dict, epoch_name_by_probability_values_output_dict_dict = build_decoder_prob_as_a_function_of_position(epochs_result_dict=epochs_result_dict, xbin_centers=deepcopy(a_laps_decoder.xbin_centers), t_delta=t_delta, grid_bin_bounds=deepcopy(a_laps_decoder.pf.config.grid_bin_bounds), is_split_by_all_decoders=True,
                                                                                                                                                                            display_context=curr_active_pipeline.build_display_context_for_session('decoded_occupancy_v_measured'))

        ## just long/short
        fig, ax_dict, epoch_name_by_pre_post_delta_category_output_dict_dict, epoch_name_by_probability_values_output_dict_dict = build_decoder_prob_as_a_function_of_position(epochs_result_dict=epochs_result_dict, xbin_centers=deepcopy(a_laps_decoder.xbin_centers), t_delta=t_delta, grid_bin_bounds=deepcopy(a_laps_decoder.pf.config.grid_bin_bounds), is_split_by_all_decoders=False,
                                                                                                                                                                            display_context=curr_active_pipeline.build_display_context_for_session('decoded_prob_as_fn_of_pos'))
    """
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DecoderIdentityColors   
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_linearized_position_probability, plot_linearized_position_prob_p
    from pyphoplacecellanalysis.Pho2D.track_shape_drawing import add_track_shapes, add_vertical_track_bounds_lines
    from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionsVisualizations
    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
    from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PhoPublicationFigureHelper



    decoder_color_dict: Dict[types.DecoderName, str] = DecoderIdentityColors.build_decoder_color_dict()

    ## INPUTS: epochs_result_dict, a_laps_decoder, 

    epoch_name_by_pre_post_delta_category_output_dict_dict: Dict[types.KnownNamedDecodingEpochsType, Dict[types.PrePostDeltaCategory, DecodedFilterEpochsResult]] = _subfn_helper_process_epochs_result_dict(epochs_result_dict=epochs_result_dict, t_delta=t_delta)
    
    epoch_name_by_probability_values_output_dict_dict: Dict[types.KnownNamedDecodingEpochsType, Dict[str, NDArray]] = {}


    if is_split_by_all_decoders:
        num = f"{num} - 4 decoders"
    else:
        num = f"{num} - 2 decoders"
        

    display_context = kwargs.pop('display_context', None)
    # display_context = display_context.adding

    # ==================================================================================================================================================================================================================================================================================== #
    # Plotting                                                                                                                                                                                                                                                                             #
    # ==================================================================================================================================================================================================================================================================================== #
    def perform_write_to_file_callback(ctxt, fig):
        """ captures: Nothing
        """
        return (ctxt, AcrossSessionsVisualizations.output_figure(final_context=display_context, fig=fig, write_vector_format=True, write_png=False))


    # with mpl.rc_context(PhoPublicationFigureHelper.rc_context_kwargs(prepare_for_publication=True, **{'figure.figsize': (6.5, 2), 'figure.dpi': '220',})):
    with mpl.rc_context(PhoPublicationFigureHelper.rc_context_kwargs(prepare_for_publication=False, **{})):

        # Create a FigureCollector instance
        with FigureCollector(name='decoded_prob_as_fn_of_pos', base_context=display_context) as collector:

            # ## Define common operations to do after making the figure:
            # def setup_common_after_creation(a_collector, fig, axes, sub_context, title=f'<size:22>Track <weight:bold>Remapping</></>'):
            #     """ Captures:

            #     t_split
            #     """
            #     a_collector.contexts.append(sub_context)
                
            #     fig.suptitle('Place Cell Remapping (Example Session)')
                
            #     if ((perform_write_to_file_callback is not None) and (sub_context is not None)):
            #         perform_write_to_file_callback(sub_context, fig)

            # BEGIN FUNCTION BODY
            
            fig = collector.build_or_reuse_figure(fignum=num, fig=kwargs.pop('fig', None), fig_idx=kwargs.pop('fig_idx', 0), figsize=kwargs.pop('figsize', (6.5, 2)), dpi=kwargs.pop('dpi', None), constrained_layout=True, clear=True, **kwargs) 
            _fig_container: GenericMatplotlibContainer = GenericMatplotlibContainer(name='decoded_prob_as_fn_of_pos')
            _fig_container.fig = fig
            
            # fig = plt.figure(layout="constrained", num=num, **kwargs)
            if is_split_by_all_decoders:
                # ax_dict = fig.subplot_mosaic(
                fig, ax_dict = collector.subplot_mosaic(
                    [
                        ["laps|pre_delta"],
                        ["laps|post_delta"],
                        ["pbe|pre_delta"],
                        ["pbe|post_delta"],        
                    ],
                    # set the height ratios between the rows
                    sharex=True, sharey=True,
                    gridspec_kw=dict(wspace=0, hspace=0.15), # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
                    extant_fig=fig,
                ) 
            else:
                # ax_dict = fig.subplot_mosaic(
                fig, ax_dict = collector.subplot_mosaic(
                    [
                        ["laps"],
                        ["pbe"],
                    ],
                    # set the height ratios between the rows
                    sharex=True, sharey=True,
                    gridspec_kw=dict(wspace=0, hspace=0.15), # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
                    extant_fig=fig,
                )

            decoder_idx_to_name_map: Dict[int, str] = {0: 'long_LR', 1: 'long_RL', 2:'short_LR', 3:'short_RL'}
            # pre_post_delta_category_name_to_scatter_kwargs_map = {'long': dict(marker='.'), 'short': dict(marker='<')}
            # pre_post_delta_category_name_to_scatter_kwargs_map = {'pre_delta': dict(marker='<'), 'post_delta': dict(marker='>')}
            pre_post_delta_category_name_to_scatter_kwargs_map = {'pre_delta': dict(marker='<'), 'post_delta': dict(marker='.')}

            for an_epoch_name, an_out_result_dict in epoch_name_by_pre_post_delta_category_output_dict_dict.items():
                for a_pre_post_delta_category_name, an_out_result in an_out_result_dict.items():
                    a_combined_str_name: str = f'{an_epoch_name}|{a_pre_post_delta_category_name}'
                    print(f'a_combined_str_name: "{a_combined_str_name}"')
                    # Compute the sum of probabilities from the p_x_given_n_list
                    summary_values_dict = {}
                    summary_values_dict['all'] = np.squeeze(np.dstack([np.squeeze(v[:,:,:]) for v in an_out_result.p_x_given_n_list])) 
                    for a_decoder_idx, a_decoder_name in decoder_idx_to_name_map.items():
                        summary_values_dict[a_decoder_name] = np.squeeze(np.hstack([np.squeeze(v[:,a_decoder_idx,:]) for v in an_out_result.p_x_given_n_list])) # (n_pos_bins, n_time_bins)
                        probability_values = np.nansum(summary_values_dict[a_decoder_name], axis=-1) # sum over time
                        
                        ## normalize over decoder
                        if enable_per_decoder_renormalization:
                            sum_over_all_decoder_p_values = np.nansum(probability_values) # sum over time
                            probability_values = probability_values / sum_over_all_decoder_p_values ## normalize over decoder

                        summary_values_dict[a_decoder_name] = probability_values # (n_pos_bins)
                        


                    ## normalize over each position decoder
                    if enable_per_position_bin_renormalization:
                        sum_over_all_decoders_per_position = summary_values_dict['long_LR'] + summary_values_dict['long_RL'] + summary_values_dict['short_LR'] + summary_values_dict['short_RL'] # sum for each position, .shape (n_pos_bins,)
                        if debug_print:
                            print(f'sum_over_all_decoders_per_position.shape: {sum_over_all_decoders_per_position.shape}')
                        # probability_values = probability_values / sum_over_all_decoders_per_position ## normalize over decoder
                        for a_decoder_idx, a_decoder_name in decoder_idx_to_name_map.items():
                            ## normalize each one at a time
                            summary_values_dict[a_decoder_name] = summary_values_dict[a_decoder_name] / sum_over_all_decoders_per_position


                    summary_values_dict['long'] = summary_values_dict['long_LR'] + summary_values_dict['long_RL'] #np.squeeze(np.hstack([np.squeeze(v[:,0,:]) for v in an_out_result.p_x_given_n_list])) +  np.squeeze(np.hstack([np.squeeze(v[:,1,:]) for v in an_out_result.p_x_given_n_list]))
                    summary_values_dict['short'] = summary_values_dict['short_LR'] + summary_values_dict['short_RL'] # np.squeeze(np.hstack([np.squeeze(v[:,2,:]) for v in an_out_result.p_x_given_n_list])) + np.squeeze(np.hstack([np.squeeze(v[:,3,:]) for v in an_out_result.p_x_given_n_list]))        


                    ## potentially redundant long/short renormalization:
                    if enable_LS_renormalization:
                        any_long_or_short = summary_values_dict['long'] + summary_values_dict['short']
                        summary_values_dict['long'] = summary_values_dict['long'] / any_long_or_short
                        summary_values_dict['short'] = summary_values_dict['short'] / any_long_or_short

                    # scatter_kwargs = {}
                    scatter_kwargs = pre_post_delta_category_name_to_scatter_kwargs_map[a_pre_post_delta_category_name] # dict(marker='.')
                    
                    if is_split_by_all_decoders:
                        active_decoder_names_list = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
                    else:
                        active_decoder_names_list = ['long', 'short']
                        # active_decoder_names_list = ['long_LR', 'long_RL', 'short_LR', 'short_RL']
                        
                    # for a_decoder_idx, a_decoder_name in decoder_idx_to_name_map.items():
                    for a_decoder_name in active_decoder_names_list:
                        a_full_combined_str_name: str = f'{a_combined_str_name}|{a_decoder_name}'
                        print(f'a_full_combined_str_name: "{a_full_combined_str_name}"')
                        if is_split_by_all_decoders:
                            ax = ax_dict[a_combined_str_name]
                            a_decoder_color = decoder_color_dict[a_decoder_name]
                            scatter_kwargs['color'] = a_decoder_color
                            a_p_x_given_n = summary_values_dict[a_decoder_name]
                            active_combined_str_name: str = a_full_combined_str_name
                        else:
                            ax = ax_dict[an_epoch_name]
                            a_decoder_color = decoder_color_dict[f"{a_decoder_name}_LR"]
                            # a_decoder_color = decoder_color_dict[a_decoder_name]
                            scatter_kwargs['color'] = a_decoder_color
                            a_p_x_given_n = summary_values_dict[a_decoder_name]
                            # active_combined_str_name: str = a_combined_str_name
                            active_combined_str_name: str = a_full_combined_str_name

                        fig1, ax, prob_values1 = plot_linearized_position_prob_p(a_p_x_given_n=a_p_x_given_n, label=active_combined_str_name, figure_title=f'Joint Position Probability', ax=ax, xbin_centers=xbin_centers, y_label=f'P({a_decoder_name}|x)', **scatter_kwargs)
                    ## END for a_decoder_name in active_de...
                    

                    epoch_name_by_probability_values_output_dict_dict[an_epoch_name] = deepcopy(summary_values_dict)
                ## END for a_pre_post_delta_c...
            ## END for an_epoch_name, an_....
            
            if grid_bin_bounds is not None:
                for k, ax in ax_dict.items():
                    long_track_line_collection, short_track_line_collection = add_vertical_track_bounds_lines(grid_bin_bounds=grid_bin_bounds, ax=ax)
            else:
                print(f'WARN: grid_bin_bounds is None so we cannot show the track endcap positions!')
        

    return fig, ax_dict, epoch_name_by_pre_post_delta_category_output_dict_dict, epoch_name_by_probability_values_output_dict_dict






@function_attributes(short_name=None, tags=['plotting', 'prob_v_position'], input_requires=[], output_provides=[], uses=[], used_by=['plot_linearized_position_probability'], creation_date='2025-05-01 17:00', related_items=[])
def plot_linearized_position_prob_p(a_p_x_given_n: NDArray, figure_title='Linearized Position Probability', save_path=None, show_figure=True, ax=None, figsize=(10, 6), label=None, y_label: str = 'P(Long|x)', xbin_centers=None,
                                    add_connecting_line=True, line_alpha=0.2, line_width=0.5, **kwargs):
    """
    Produces a figure showing the probability distribution across linearized positions.
    
    Parameters:
    -----------
    an_out_result : object
        Result object containing p_x_given_n_list attribute
    figure_title : str, optional
        Title for the figure window
    save_path : str, optional
        If provided, saves the figure to this path
    show_figure : bool, optional
        Whether to display the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    probability_values : numpy.ndarray
        The normalized probability values plotted


    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_linearized_position_prob_p
        
        fig1, ax1, prob_values1 = plot_linearized_position_probability(an_out_result=an_out_result_dict['pre-delta'], figure_title='pre-delta Linearized Position Probability')
        fig2, ax2, prob_values2 = plot_linearized_position_probability(an_out_result=an_out_result_dict['post-delta'], figure_title='post-delta Linearized Position Probability', ax=ax1)

    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(num=figure_title, figsize=figsize, clear=True)
    else:
        fig = ax.figure
    
    # Plot the data
    if xbin_centers is None:
        xbin_centers = np.arange(len(a_p_x_given_n))

    scatter = ax.scatter(xbin_centers, a_p_x_given_n, label=label, **kwargs)

    # Add connecting line if requested
    if add_connecting_line:
        # Get the color from the scatter plot if not specified in kwargs
        if 'color' in kwargs:
            line_color = kwargs['color']
        elif hasattr(scatter, 'get_facecolor'):
            # For single color scatter
            line_color = scatter.get_facecolor()[0]
        else:
            # Default color
            line_color = 'blue'
            
        # Plot the connecting line with transparency
        ax.plot(xbin_centers, a_p_x_given_n, color=line_color, alpha=line_alpha, linewidth=line_width)


    ax.set_title(figure_title)
    ax.set_ylabel(y_label)
    ax.set_xlabel('Position x <linearized>')
    
    ax.legend()

    # Optional: save the figure
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or close the figure based on parameter
    if not show_figure:
        plt.close(fig)
        
    return fig, ax, a_p_x_given_n


@function_attributes(short_name=None, tags=['plotting', 'prob_v_position'], input_requires=[], output_provides=[], uses=['plot_linearized_position_prob_p'], used_by=[], creation_date='2025-05-01 17:00', related_items=[])
def plot_linearized_position_probability(an_out_result: DecodedFilterEpochsResult, is_P_long:bool=False, **kwargs):
    """
    Produces a figure showing the probability distribution across linearized positions.
    
    Parameters:
    -----------
    an_out_result : object
        Result object containing p_x_given_n_list attribute
    figure_title : str, optional
        Title for the figure window
    save_path : str, optional
        If provided, saves the figure to this path
    show_figure : bool, optional
        Whether to display the figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    probability_values : numpy.ndarray
        The normalized probability values plotted


    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_linearized_position_probability
        
        fig1, ax1, prob_values1 = plot_linearized_position_probability(an_out_result=an_out_result_dict['pre-delta'], figure_title='pre-delta Linearized Position Probability')
        fig2, ax2, prob_values2 = plot_linearized_position_probability(an_out_result=an_out_result_dict['post-delta'], figure_title='post-delta Linearized Position Probability', ax=ax1)

    """
    # Compute the sum of probabilities from the p_x_given_n_list
    if is_P_long:
        y_label: str = 'P(Long|x)'
        summary_values = np.squeeze(np.hstack([np.squeeze(v[:,0,:]) for v in an_out_result.p_x_given_n_list])) + \
                        np.squeeze(np.hstack([np.squeeze(v[:,1,:]) for v in an_out_result.p_x_given_n_list]))
    else:
        ## P_Short
        y_label: str = 'P(Short|x)'
        summary_values = np.squeeze(np.hstack([np.squeeze(v[:,2,:]) for v in an_out_result.p_x_given_n_list])) + \
                        np.squeeze(np.hstack([np.squeeze(v[:,3,:]) for v in an_out_result.p_x_given_n_list]))


    # Sum across one dimension and normalize
    probability_values = np.nansum(summary_values, axis=-1)
    normalization_factor = np.nansum(probability_values)
    # probability_values = probability_values / normalization_factor
    
    return plot_linearized_position_prob_p(a_p_x_given_n=probability_values, y_label=y_label, **kwargs)







@function_attributes(short_name=None, tags=['plot', 'figure', 'matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-04-14 17:55', related_items=[])
def _plot_all_time_decoded_marginal_figures_non_interactive(curr_active_pipeline, best_matching_context: IdentifyingContext, a_decoded_marginal_posterior_df: pd.DataFrame, epochs_decoding_time_bin_size: float = 0.025, constrained_layout=False, **kwargs):
    """ exports the components needed to show the decoded P_Short/P_Long likelihoods over time

    Usage:
        from neuropy.utils.result_context import IdentifyingContext
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _plot_all_time_decoded_marginal_figures_non_interactive
        from pyphocorehelpers.plotting.media_output_helpers import save_array_as_image, get_array_as_image

        curr_active_pipeline.reload_default_display_functions()

        ## INPUTS: a_new_fully_generic_result
        # a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='nan_filled', data_grain='per_time_bin')
        a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='ignore', data_grain='per_time_bin')
        best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context, debug_print=False)
        epochs_decoding_time_bin_size = best_matching_context.get('time_bin_size', None)
        assert epochs_decoding_time_bin_size is not None

        graphics_output_dict = _plot_all_time_decoded_marginal_figures_non_interactive(curr_active_pipeline=curr_active_pipeline, best_matching_context=best_matching_context, a_decoded_marginal_posterior_df=a_decoded_marginal_posterior_df, epochs_decoding_time_bin_size=epochs_decoding_time_bin_size)



    """
    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector    
    from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import MatplotlibRenderPlots
    from pyphocorehelpers.plotting.media_output_helpers import save_array_as_image
    from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol
    # from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices
    from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
    from neuropy.utils.matplotlib_helpers import get_heatmap_cmap
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from flexitext import flexitext ## flexitext for formatted matplotlib text
    from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
    from neuropy.utils.matplotlib_helpers import FormattedFigureText

    complete_session_context, (session_context, additional_session_context) = curr_active_pipeline.get_complete_session_context()

    desired_width = 4096

    active_context = kwargs.pop('active_context', None)
    if active_context is not None:
        # Update the existing context:
        display_context = active_context.adding_context('display_fn', display_fn_name='generalized_decoded_yellow_blue_marginal_epochs')
    else:
        active_context = curr_active_pipeline.sess.get_context()
        # Build the active context directly:
        display_context = curr_active_pipeline.build_display_context_for_session('generalized_decoded_yellow_blue_marginal_epochs')

    fignum = kwargs.pop('fignum', None)
    if fignum is not None:
        print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')

    defer_render = kwargs.pop('defer_render', False)
    debug_print: bool = kwargs.pop('debug_print', False)
    active_config_name: bool = kwargs.pop('active_config_name', None)

    perform_write_to_file_callback = kwargs.pop('perform_write_to_file_callback', (lambda final_context, fig: curr_active_pipeline.output_figure(final_context, fig)))


    # active_config_name = kwargs.pop('active_config_name', None)

    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    # long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
    long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()


    # ==================================================================================================================================================================================================================================================================================== #
    # Start Building Figure                                                                                                                                                                                                                                                                #
    # ==================================================================================================================================================================================================================================================================================== #

    graphics_output_dict = None

    # for best_matching_context, a_decoded_marginal_posterior_df in flat_decoded_marginal_posterior_df_context_dict.items():
    time_bin_size = epochs_decoding_time_bin_size
    info_string: str = f" - t_bin_size: {time_bin_size}"
    plot_row_identifier: str = best_matching_context.get_description(subset_includelist=['known_named_decoding_epochs_type', 'masked_time_bin_fill_type'], include_property_names=True, key_value_separator=':', separator='|', replace_separator_in_property_names='-')
    a_time_window_centers = a_decoded_marginal_posterior_df['t_bin_center'].to_numpy() 
    a_1D_posterior = a_decoded_marginal_posterior_df[['P_Long', 'P_Short']].to_numpy().T
    # a_1D_posterior = a_decoded_marginal_posterior_df[['P_Long', 'P_Short']].to_numpy()

    # image, out_path = save_array_as_image(a_1D_posterior, desired_height=None, desired_width=desired_width, skip_img_normalization=True)
    
    # # Save image to file
    # active_out_figure_paths, final_context = curr_active_pipeline.output_figure(final_context=complete_session_context.overwriting_context(display='decoded_P_Short_Posterior_global_epoch'), fig=image, write_png=True, write_vector_format=False)
    # _all_tracks_active_out_figure_paths[final_context] = deepcopy(active_out_figure_paths[0])
    
    # ==================================================================================================================================================================================================================================================================================== #
    # Begin Subfunctions                                                                                                                                                                                                                                                                   #
    # ==================================================================================================================================================================================================================================================================================== #

    def _subfn_clean_axes_decorations(an_ax):
        """ removes ticks, titles, and other intrusive elements from each axes
        _subfn_clean_axes_decorations(an_ax=ax_dict["ax_top"])
        """
        an_ax.set_xticklabels([])
        an_ax.set_yticklabels([])    
        an_ax.set_title('') ## remove title
        # fig.canvas.manager.set_window_title()
        


    def _subfn_display_grid_bin_bounds_validation(owning_pipeline_reference, is_x_axis: bool = True, pos_var_names_override=None, ax=None): # , global_computation_results, computation_results, active_configs, include_includelist=None, defer_render=False, save_figure=True
        """ Renders a single figure that shows the 1D linearized position from several different sources to ensure sufficient overlap. Useful for validating that the grid_bin_bounds are chosen reasonably.

        """
        from pyphoplacecellanalysis.Pho2D.track_shape_drawing import NotableTrackPositions, perform_add_1D_track_bounds_lines

        assert owning_pipeline_reference is not None
        long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
        long_grid_bin_bounds, short_grid_bin_bounds, global_grid_bin_bounds = [owning_pipeline_reference.computation_results[a_name].computation_config['pf_params'].grid_bin_bounds for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        # print(long_grid_bin_bounds, short_grid_bin_bounds, global_grid_bin_bounds)
        
        long_epoch_context, short_epoch_context, global_epoch_context = [owning_pipeline_reference.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        long_session, short_session, global_session = [owning_pipeline_reference.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        
        long_pos_df, short_pos_df, global_pos_df, all_pos_df = [a_sess.position.to_dataframe() for a_sess in (long_session, short_session, global_session, owning_pipeline_reference.sess)]
        combined_pos_df = deepcopy(all_pos_df)
        combined_pos_df.loc[long_pos_df.index, 'long_lin_pos'] = long_pos_df.lin_pos
        combined_pos_df.loc[short_pos_df.index, 'short_pos_df'] = short_pos_df.lin_pos
        combined_pos_df.loc[global_pos_df.index, 'global_lin_pos'] = global_pos_df.lin_pos

        title = f'grid_bin_bounds validation across epochs'
        if is_x_axis:
            title = f'{title} - X-axis'
        else:
            title = f'{title} - Y-axis'
            
        const_line_text_label_y_offset: float = 0.05
        const_line_text_label_x_offset: float = 0.1
        
        if is_x_axis:
            ## plot x-positions
            if (pos_var_names_override is not None) and (len(pos_var_names_override) > 0):
                pos_var_names = pos_var_names_override
            else:
                pos_var_names = ['x', 'lin_pos', 'long_lin_pos', 'short_pos_df', 'global_lin_pos'] # use the appropriate defaults
                
            combined_pos_df_plot_kwargs = dict(x='t', y=pos_var_names, title='grid_bin_bounds validation across epochs - positions along x-axis', ax=ax)
        else:
            ## plot y-positions
            if (pos_var_names_override is not None) and (len(pos_var_names_override) > 0):
                pos_var_names = pos_var_names_override
            else:
                pos_var_names = ['y']
            # combined_pos_df_plot_kwargs = dict(x='y', y='t', title='grid_bin_bounds validation across epochs - positions along y-axis')
            combined_pos_df_plot_kwargs = dict(x='t', y=pos_var_names, title='grid_bin_bounds validation across epochs - positions along y-axis', ax=ax)
            
        # Plot all 1D position variables:
        combined_pos_df.plot(**combined_pos_df_plot_kwargs)
        fig = plt.gcf() ## always get the figure and set the title
        if ax is None:
            ax = plt.gca()
        ax.set_title(title)
        fig.canvas.manager.set_window_title(title)
        
        ax.legend(loc='upper left') # Move legend inside the plot, in the top-left corner
        # ax.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Move legend outside the plot
        
        final_context = owning_pipeline_reference.sess.get_context().adding_context('display_fn', display_fn_name='_display_grid_bin_bounds_validation')
        
        ## Add grid_bin_bounds, track limits, and midpoint lines:
        curr_config = owning_pipeline_reference.active_configs['maze_any']

        grid_bin_bounds = curr_config.computation_config.pf_params.grid_bin_bounds # ((37.0773897438341, 250.69004399129707), (137.925447118083, 145.16448776601297))
        # curr_config.computation_config.pf_params.grid_bin # (3.793023081021702, 1.607897707662558)
        # loaded_track_limits = curr_config.active_session_config.loaded_track_limits


        # curr_config.active_session_config.y_midpoint
        
        (long_notable_x_platform_positions, short_notable_x_platform_positions), (long_notable_y_platform_positions, short_notable_y_platform_positions) = NotableTrackPositions.init_notable_track_points_from_session_config(owning_pipeline_reference.sess.config)
        
        if is_x_axis:
            ## plot x-positions
            perform_add_1D_track_bounds_lines_kwargs = dict(long_notable_x_platform_positions=tuple(long_notable_x_platform_positions), short_notable_x_platform_positions=tuple(short_notable_x_platform_positions), is_vertical=False)
        else:
            ## plot y-positions
            perform_add_1D_track_bounds_lines_kwargs = dict(long_notable_x_platform_positions=tuple(long_notable_y_platform_positions), short_notable_x_platform_positions=tuple(short_notable_y_platform_positions), is_vertical=False)
            
        long_track_line_collection, short_track_line_collection = perform_add_1D_track_bounds_lines(**perform_add_1D_track_bounds_lines_kwargs, ax=ax)


        # Plot REAL `grid_bin_bounds` ________________________________________________________________________________________ #
        ((grid_bin_bounds_x0, grid_bin_bounds_x1), (grid_bin_bounds_y0, grid_bin_bounds_y1)) = grid_bin_bounds
        if is_x_axis:
            ## horizontal lines:
            common_ax_bound_kwargs = dict(xmin=ax.get_xbound()[0], xmax=ax.get_xbound()[1])
        else:
            # common_ax_bound_kwargs = dict(ymin=ax.get_ybound()[0], ymax=ax.get_ybound()[1]) 
            common_ax_bound_kwargs = dict(xmin=ax.get_xbound()[0], xmax=ax.get_xbound()[1])  ## y-axis along x (like a 1D plot) Mode

        if is_x_axis:
            ## plot x-positions
            ## horizontal lines:
            ## midpoint line: dotted blue line centered in the bounds (along y)
            x_midpoint = curr_config.active_session_config.x_midpoint # 143.88489208633095
            midpoint_line_collection = ax.hlines(x_midpoint, label='x_midpoint', xmin=ax.get_xbound()[0], xmax=ax.get_xbound()[1], colors='#0000FFAA', linewidths=1.0, linestyles='dashed', zorder=-98) # matplotlib.collections.LineCollection midpoint_line_collection
            ax.text(ax.get_xbound()[1], (x_midpoint + const_line_text_label_y_offset), 'x_mid', ha='right', va='bottom', fontsize=8, color='#0000FFAA', zorder=-98) # Add right-aligned text label slightly above the hline
        
            ## 2 lines corresponding to the x0 and x1 of the grid_bin_bounds:
            grid_bin_bounds_line_collection = ax.hlines([grid_bin_bounds_x0, grid_bin_bounds_x1], label='grid_bin_bounds - after - dark blue', **common_ax_bound_kwargs, colors='#2e2e20', linewidths=2.0, linestyles='solid', zorder=-98) # grid_bin_bounds_line_collection
            # _grid_bin_bound_labels_x_pos = (ax.get_xbound()[1] - const_line_text_label_x_offset)
            _grid_bin_bound_labels_x_pos: float = ax.get_xbound()[0] + ((ax.get_xbound()[1] - ax.get_xbound()[0])/2.0) # center point
            print(f'_grid_bin_bound_labels_x_pos: {_grid_bin_bound_labels_x_pos}')
            ax.text(_grid_bin_bound_labels_x_pos, (grid_bin_bounds_x0 - const_line_text_label_y_offset), 'grid_bin_bounds[x0]', ha='center', va='bottom', fontsize=9, color='#2e2d20', zorder=-97) # Add right-aligned text label slightly above the hline
            ax.text(_grid_bin_bound_labels_x_pos, (grid_bin_bounds_x1 + const_line_text_label_y_offset), 'grid_bin_bounds[x1]', ha='center', va='top', fontsize=9, color='#2e2d20', zorder=-97) # this will be the top (highest y-pos) line.
            
        else:
            ## plot y-positions
            midpoint_line_collection = None
            # grid_bin_bounds_line_collection = None
            grid_bin_bounds_positions_list = [grid_bin_bounds_y0, grid_bin_bounds_y1]
            grid_bin_bounds_label_names_list = ['grid_bin_bounds[y0]', 'grid_bin_bounds[y1]']
            ## 2 lines corresponding to the x0 and x1 of the grid_bin_bounds:
            grid_bin_bounds_line_collection = ax.hlines(grid_bin_bounds_positions_list, label='grid_bin_bounds - after - dark blue', **common_ax_bound_kwargs, colors='#2e2e20', linewidths=2.0, linestyles='solid', zorder=-98) # grid_bin_bounds_line_collection
            # _grid_bin_bound_labels_x_pos = (ax.get_xbound()[1] - const_line_text_label_x_offset)
            _grid_bin_bound_labels_x_pos: float = ax.get_xbound()[0] + ((ax.get_xbound()[1] - ax.get_xbound()[0])/2.0) # center point
            # print(f'_grid_bin_bound_labels_y_pos: {_grid_bin_bound_labels_y_pos}')
            # ax.text(_grid_bin_bound_labels_y_pos, (grid_bin_bounds_y0 - const_line_text_label_y_offset), 'grid_bin_bounds[x0]', ha='center', va='bottom', fontsize=9, color='#2e2d20', zorder=-97) # Add right-aligned text label slightly above the hline
            # ax.text(_grid_bin_bound_labels_y_pos, (grid_bin_bounds_x1 + const_line_text_label_y_offset), 'grid_bin_bounds[x1]', ha='center', va='top', fontsize=9, color='#2e2d20', zorder=-97) # this will be the top (highest y-pos) line.	
            # Iterate through the hlines in the LineCollection and add labels
            assert len(grid_bin_bounds_label_names_list) == len(grid_bin_bounds_positions_list)
            for pos, a_txt_label in zip(grid_bin_bounds_positions_list, grid_bin_bounds_label_names_list):
                # ax.text(ax.get_xbound()[1], (pos + const_line_text_label_y_offset), a_txt_label, color='#2e2d20', fontsize=9, ha='center', va='center', zorder=-97)
                ax.text(_grid_bin_bound_labels_x_pos, (pos + const_line_text_label_y_offset), a_txt_label, color='#2e2d20', fontsize=9, ha='center', va='center', zorder=-97)
        
        # Show legend
        # ax.legend()

        # if save_figure:
        #     saved_figure_paths = owning_pipeline_reference.output_figure(final_context, fig)
        # else:
        #     saved_figure_paths = []

        graphics_output_dict = MatplotlibRenderPlots(name='_display_grid_bin_bounds_validation', figures=(fig,), axes=(ax,), plot_data={'midpoint_line_collection': midpoint_line_collection, 'grid_bin_bounds_line_collection': grid_bin_bounds_line_collection, 'long_track_line_collection': long_track_line_collection, 'short_track_line_collection': short_track_line_collection}, context=final_context, saved_figures=[])
        return graphics_output_dict
    ## END _subfn_display_grid_bin_bounds_validation...


    # ==================================================================================================================================================================================================================================================================================== #
    # Begin Function Body                                                                                                                                                                                                                                                                  #
    # ==================================================================================================================================================================================================================================================================================== #

    graphics_output_dict = {}

    # Shared active_decoder, global_session:
    global_session = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name]) # used for validate_lap_dir_estimations(...) 

    # 'figure.constrained_layout.use': False, 'figure.autolayout': False, 'figure.subplot.bottom': 0.11, 'figure.figsize': (6.4, 4.8)
    # 'figure.constrained_layout.use': constrained_layout, 'figure.autolayout': False, 'figure.subplot.bottom': 0.11, 'figure.figsize': (6.4, 4.8)
    # with mpl.rc_context({'figure.dpi': '220', 'savefig.transparent': True, 'ps.fonttype': 42, 'pdf.fonttype': 42, 'figure.constrained_layout.use': (constrained_layout or False), 'figure.frameon': False, 'figure.figsize': (35, 3), }):
    with mpl.rc_context({'figure.dpi': '100', 'savefig.transparent': True, 'ps.fonttype': 42, 'pdf.fonttype': 42, 'figure.constrained_layout.use': (constrained_layout or False), 'figure.frameon': False, 'figure.figsize': (35, 3), }): # 'figure.figsize': (12.4, 4.8), 
        # Create a FigureCollector instance
        with FigureCollector(name='plot_all_time_decoded_marginal_figures', base_context=display_context) as collector:
            fig, ax_dict = collector.subplot_mosaic(
                # fig = plt.figure(layout="constrained")
                # ax_dict = fig.subplot_mosaic(
                [
                    ["ax_top"],
                    ["ax_decodedMarginal_P_Short_v_time"],
                    # ["ax_position_and_laps_v_time"],
                ],
                # set the width ratios between the columns
                # height_ratios=[2, 1, 4],
                height_ratios=[3, 1],
                sharex=True,
                gridspec_kw=dict(wspace=0, hspace=0) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
            )


            # fig = None
            an_ax = ax_dict["ax_decodedMarginal_P_Short_v_time"] ## no figure (should I be using collector??)

            # decoded posterior overlay __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            variable_name: str = ''
            y_bin_labels = ['P_Long', 'P_Short']
            xbin = None
            active_most_likely_positions = None
            active_posterior = deepcopy(a_1D_posterior)
            posterior_heatmap_imshow_kwargs = dict() # zorder=-1, alpha=0.1
            
            ### construct fake position axis (xbin):
            n_xbins, n_t_bins = np.shape(a_1D_posterior)
            if xbin is None:
                xbin = np.arange(n_xbins)

            ## Actual plotting portion:
            fig, an_ax, _return_out_artists_dict = plot_1D_most_likely_position_comparsions(measured_position_df=None, time_window_centers=a_time_window_centers, xbin=deepcopy(xbin),
                                                                    posterior=active_posterior,
                                                                    active_most_likely_positions_1D=active_most_likely_positions,
                                                                    ax=an_ax, variable_name=variable_name, debug_print=True, enable_flat_line_drawing=False,
                                                                    posterior_heatmap_imshow_kwargs=posterior_heatmap_imshow_kwargs, return_created_artists=True)
            
            label_artists_dict = PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels(an_ax, y_bin_labels=y_bin_labels, enable_draw_decoder_colored_lines=False)
            _subfn_clean_axes_decorations(an_ax=ax_dict["ax_decodedMarginal_P_Short_v_time"])
        

            # # Position/bounds lines ______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            # an_ax = ax_dict["ax_position_and_laps_v_time"]
            # graphics_output_dict: MatplotlibRenderPlots = _subfn_display_grid_bin_bounds_validation(owning_pipeline_reference=curr_active_pipeline, pos_var_names_override=['lin_pos'], ax=an_ax) # (or ['x']) build basic position/bounds figure as a starting point
            # an_ax = graphics_output_dict.axes[0]
            # fig = graphics_output_dict.figures[0]
            # _subfn_clean_axes_decorations(an_ax=ax_dict["ax_position_and_laps_v_time"])


            # Add Epochs (Laps/PBEs/Delta_t/etc) _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            ## from `_display_long_short_laps`
            from pyphoplacecellanalysis.PhoPositionalData.plotting.laps import plot_laps_2d
            
            an_ax = ax_dict["ax_top"]
            fig, out_axes_list = plot_laps_2d(global_session, legacy_plotting_mode=False, include_velocity=False, include_accel=False, axes_list=[an_ax], **kwargs)
            _subfn_clean_axes_decorations(an_ax=ax_dict["ax_top"])



            # final_context = curr_active_pipeline.sess.get_context().adding_context('display_fn', display_fn_name='display_long_short_laps')

            # def _perform_write_to_file_callback():
            #     return curr_active_pipeline.output_figure(final_context, fig)

            # if save_figure:
            #     active_out_figure_paths = _perform_write_to_file_callback()
            # else:
            #     active_out_figure_paths = []
                            
            # graphics_output_dict = MatplotlibRenderPlots(name='_display_long_short_laps', figures=(fig,), axes=out_axes_list, plot_data={}, context=final_context, saved_figures=active_out_figure_paths)

    return graphics_output_dict








from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster, SynchronizedPlotMode
from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget


@function_attributes(short_name=None, tags=['interactive', 'widget', 'plot', 'figure', 'export'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-04-14 10:50', related_items=[])
def _plot_all_time_decoded_marginal_figures(curr_active_pipeline, best_matching_context: IdentifyingContext, a_decoded_marginal_posterior_df: pd.DataFrame, spike_raster_window: Spike3DRasterWindowWidget, active_2d_plot: Spike2DRaster, epochs_decoding_time_bin_size: float = 0.025, write_vector_format:bool=False, write_png:bool=True):
    """ exports the components needed to show the decoded P_Short/P_Long likelihoods over time


    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _plot_all_time_decoded_marginal_figures

    ## INPUTS: a_new_fully_generic_result
    a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='nan_filled', data_grain='per_time_bin')
    best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context, debug_print=False)
    epochs_decoding_time_bin_size = best_matching_context.get('time_bin_size', None)
    assert epochs_decoding_time_bin_size is not None
    _all_tracks_active_out_figure_paths, _all_tracks_out_artists, _all_tracks_out_axes = _plot_all_time_decoded_marginal_figures(curr_active_pipeline=curr_active_pipeline, best_matching_context=best_matching_context, a_decoded_marginal_posterior_df=a_decoded_marginal_posterior_df, spike_raster_window=spike_raster_window, active_2d_plot=active_2d_plot, epochs_decoding_time_bin_size=epochs_decoding_time_bin_size)
    _all_tracks_active_out_figure_paths

    """
    from neuropy.utils.result_context import IdentifyingContext
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
    from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget
    from pyphoplacecellanalysis.GUI.Qt.PlaybackControls.Spike3DRasterBottomPlaybackControlBarWidget import Spike3DRasterBottomPlaybackControlBar
    from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget import PyqtgraphTimeSynchronizedWidget
    # from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.spike_raster_widgets import _setup_spike_raster_window_for_debugging
    width_pixels = 4096
    height_pixels = 135
    # dpi = 100
    dpi = 96
    figsize_inches = (width_pixels/dpi, height_pixels/dpi) # (12.4, 4.8)
    print(f'figsize_inches: {figsize_inches}')

    # width_pixels = 1389 ## Computed
    kwargs = {}
    output_figure_kwargs = ({'width': 3072} | kwargs) # add 'width' to kwargs if not specified
    
    all_global_menus_actionsDict, global_flat_action_dict = spike_raster_window.build_all_menus_actions_dict()
    complete_session_context, (session_context, additional_session_context) = curr_active_pipeline.get_complete_session_context()

    sync_mode: SynchronizedPlotMode = SynchronizedPlotMode.TO_GLOBAL_DATA

    # Gets the existing SpikeRasterWindow or creates a new one if one doesn't already exist:
    # spike_raster_window, (active_2d_plot, active_3d_plot, main_graphics_layout_widget, main_plot_widget, background_static_scroll_plot_widget) = Spike3DRasterWindowWidget.find_or_create_if_needed(curr_active_pipeline, force_create_new=True)
    # spike_raster_window = active_2d_plot.window()

    _all_tracks_active_out_figure_paths = {}
    _all_tracks_out_artists = {}
    _all_tracks_out_axes = {}
    # for best_matching_context, a_decoded_marginal_posterior_df in flat_decoded_marginal_posterior_df_context_dict.items():
    time_bin_size = epochs_decoding_time_bin_size
    info_string: str = f" - t_bin_size: {time_bin_size}"
    plot_row_identifier: str = best_matching_context.get_description(subset_includelist=['known_named_decoding_epochs_type', 'masked_time_bin_fill_type'], include_property_names=True, key_value_separator=':', separator='|', replace_separator_in_property_names='-')
    a_time_window_centers = a_decoded_marginal_posterior_df['t_bin_center'].to_numpy() 
    a_1D_posterior = a_decoded_marginal_posterior_df[['P_Long', 'P_Short']].to_numpy().T

    constrained_layout = False


    # Update the window to the full data extent: _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    start_t = active_2d_plot.total_data_start_time # spike_raster_window.total_data_start_time
    end_t = active_2d_plot.total_data_end_time # spike_raster_window.total_data_end_time
    print(f'start_t: {start_t}, end_t: {end_t}')
    active_2d_plot.Render2DScrollWindowPlot_on_window_update(start_t, end_t)



    with mpl.rc_context({'figure.dpi': str(dpi), 'savefig.transparent': True, 'ps.fonttype': 42, 'pdf.fonttype': 42, 'figure.constrained_layout.use': (constrained_layout or False), 'figure.frameon': False, 'figure.figsize': figsize_inches, }): # 'figure.figsize': (12.4, 4.8), 
        # Create a FigureCollector instance
        # with FigureCollector(name='plot_directional_merged_pf_decoded_epochs', base_context=display_context) as collector:

        curr_identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dock_item = active_2d_plot.add_docked_marginal_track(name=plot_row_identifier, time_window_centers=a_time_window_centers, a_1D_posterior=a_1D_posterior, extended_dock_title_info=info_string, sync_mode=sync_mode)
        
        # from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
        # from neuropy.utils.matplotlib_helpers import get_heatmap_cmap
        # from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
        
        # if a_variable_name is None:
        #     a_variable_name = name

        # if a_dock_config is None:
        #     override_dock_group_name: str = None ## this feature doesn't work
        #     a_dock_config = CustomCyclicColorsDockDisplayConfig(showCloseButton=True, named_color_scheme=NamedColorScheme.grey)
        #     a_dock_config.dock_group_names = [override_dock_group_name] # , 'non-PBE Continuous Decoding'


        # n_xbins, n_t_bins = np.shape(a_1D_posterior)

        # if xbin is None:
        #     xbin = np.arange(n_xbins)

        # ## ✅ Add a new row for each of the four 1D directional decoders:
        # identifier_name: str = name
        # if extended_dock_title_info is not None:
        #     identifier_name += extended_dock_title_info ## add extra info like the time_bin_size in ms
        # print(f'identifier_name: {identifier_name}')
        # widget, matplotlib_fig, matplotlib_fig_axes, dock_item = self.add_new_matplotlib_render_plot_widget(name=identifier_name, dockSize=(25, 200), display_config=a_dock_config, **kwargs)
        # an_ax = matplotlib_fig_axes[0]

        # variable_name: str = a_variable_name
        
        # # active_most_likely_positions = active_marginals.most_likely_positions_1D # Raw decoded positions
        # active_most_likely_positions = None
        # active_posterior = deepcopy(a_1D_posterior)
        # a_time_window_centers = a_df['t_bin_center'].to_numpy() 
        # a_1D_posterior = a_df[['P_Long', 'P_Short']].to_numpy().T

        # posterior_heatmap_imshow_kwargs = dict()
        
        # # most_likely_positions_mode: 'standard'|'corrected'
        # ## Actual plotting portion:
        # fig, an_ax = plot_1D_most_likely_position_comparsions(measured_position_df=None, time_window_centers=time_window_centers, xbin=deepcopy(xbin),
        #                                                         posterior=active_posterior,
        #                                                         active_most_likely_positions_1D=active_most_likely_positions,
        #                                                         ax=an_ax, variable_name=variable_name, debug_print=True, enable_flat_line_drawing=False,
        #                                                         posterior_heatmap_imshow_kwargs=posterior_heatmap_imshow_kwargs)

        # return identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dock_item
        ## END EXPANSION


        _all_tracks_out_artists[curr_identifier_name] = widget
        _all_tracks_out_axes[curr_identifier_name] = matplotlib_fig_axes[0]
        matplotlib_fig_axes[0].set_xlim(active_2d_plot.total_data_start_time, active_2d_plot.total_data_end_time)
        active_2d_plot.sync_matplotlib_render_plot_widget(identifier=curr_identifier_name, sync_mode=sync_mode) ## set sync mode
        ## set figure size:
        matplotlib_fig.set_size_inches(*figsize_inches) #width_pixels/dpi, height_pixels/dpi)
        matplotlib_fig.set_dpi(dpi)

        widget.draw()

        dock_item.hideTitleBar()

        active_out_figure_paths, final_context = curr_active_pipeline.output_figure(final_context=complete_session_context.overwriting_context(display='decoded_P_Short_Posterior'), fig=matplotlib_fig, write_vector_format=write_vector_format, write_png=write_png) # , **output_figure_kwargs
        _all_tracks_active_out_figure_paths[final_context] = deepcopy(active_out_figure_paths[0])


        # Position over time _________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        curr_identifier_name = 'new_curves_separate_plot'

        menu_commands = [
            'AddTimeCurves.Position', ## 2025-03-11 02:32 Running this too soon after launching the window causes weird black bars on the top and bottom of the window
        ]
        # menu_commands = ['actionPseudo2DDecodedEpochsDockedMatplotlibView', 'actionContinuousPseudo2DDecodedMarginalsDockedMatplotlibView'] # , 'AddTimeIntervals.SessionEpochs'

        a_dock = active_2d_plot.find_display_dock(identifier=curr_identifier_name)
        if a_dock is None:
            global_flat_action_dict['AddTimeCurves.Position'].trigger() ## build the item
            a_dock = active_2d_plot.find_display_dock(identifier=curr_identifier_name) ## get the item

        widget: PyqtgraphTimeSynchronizedWidget = a_dock.widgets[0]
        # active_2d_plot.sync_matplotlib_render_plot_widget(identifier=curr_identifier_name, sync_mode=sync_mode) ## set sync mode
        widget.getRootPlotItem().setXRange(active_2d_plot.total_data_start_time, active_2d_plot.total_data_end_time, padding=0) ## global frame
        # widget.getRootPlotItem().set_xlim(active_2d_plot.total_data_start_time, active_2d_plot.total_data_end_time)
        # widget.update(None)
        # widget.draw()
        widget.getRootPlotItem().update()

        _all_tracks_out_artists[curr_identifier_name] = widget
        _all_tracks_out_axes[curr_identifier_name] = widget.getRootPlotItem()

        ## Export Position over time Figure:
        active_out_figure_paths, final_context = curr_active_pipeline.output_figure(final_context=complete_session_context.overwriting_context(display='pos_over_t'), fig=widget.getRootPlotItem(), write_vector_format=write_vector_format, write_png=write_png, **output_figure_kwargs)
        _all_tracks_active_out_figure_paths[final_context] = deepcopy(active_out_figure_paths[0])


        # Epoch Intervals figure: ____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        curr_identifier_name = 'interval_overview'
        _interval_tracks_out_dict = active_2d_plot.prepare_pyqtgraph_intervalPlot_tracks(enable_interval_overview_track=True, should_remove_all_and_re_add=True, should_link_to_main_plot_widget=False)
        interval_window_dock_config, intervals_dock, intervals_time_sync_pyqtgraph_widget, intervals_root_graphics_layout_widget, intervals_plot_item = _interval_tracks_out_dict['intervals']
        if curr_identifier_name in _interval_tracks_out_dict:
            interval_overview_window_dock_config, intervals_overview_dock, intervals_overview_time_sync_pyqtgraph_widget, intervals_overview_root_graphics_layout_widget, intervals_overview_plot_item = _interval_tracks_out_dict[curr_identifier_name]
            intervals_overview_plot_item.setXRange(active_2d_plot.total_data_start_time, active_2d_plot.total_data_end_time, padding=0) ## global frame
            # intervals_overview_time_sync_pyqtgraph_widget.setMaximumHeight(39)

        a_dock, widget = active_2d_plot.find_dock_item_tuple(identifier=curr_identifier_name)
        # active_2d_plot.sync_matplotlib_render_plot_widget(identifier=curr_identifier_name, sync_mode=sync_mode) ## set sync mode        
        widget.getRootPlotItem().setXRange(active_2d_plot.total_data_start_time, active_2d_plot.total_data_end_time, padding=0) ## global frame        
        # widget.update(None)
        # widget.draw()
        _all_tracks_out_artists[curr_identifier_name] = widget
        _all_tracks_out_axes[curr_identifier_name] = widget.getRootPlotItem()

        active_out_figure_paths, final_context = curr_active_pipeline.output_figure(final_context=complete_session_context.overwriting_context(display='interval_epochs_overview'), fig=widget.getRootPlotItem(), write_vector_format=write_vector_format, write_png=write_png, **output_figure_kwargs)
        _all_tracks_active_out_figure_paths[final_context] = deepcopy(active_out_figure_paths[0])


    # ==================================================================================================================================================================================================================================================================================== #
    # Export the combined item:                                                                                                                                                                                                                                                            #
    # ==================================================================================================================================================================================================================================================================================== #
    from PIL import Image 
    from pyphocorehelpers.plotting.media_output_helpers import vertical_image_stack

    # # Let's assume you have a list of images
    out_figs_paths = list(_all_tracks_active_out_figure_paths.values()) # ['image1.png', 'image2.png', 'image3.png']  # replace this with actual paths to your images
    imgs = [Image.open(i) for i in out_figs_paths]
    output_img = vertical_image_stack(imgs, v_overlap=0, padding=0)
    # Save image to file
    active_out_figure_paths, final_context = curr_active_pipeline.output_figure(final_context=complete_session_context.overwriting_context(display='decoded_P_Short_Posterior_global_epoch'), fig=output_img, write_png=True, write_vector_format=False)
    _all_tracks_active_out_figure_paths[final_context] = deepcopy(active_out_figure_paths[0])


    return _all_tracks_active_out_figure_paths, _all_tracks_out_artists, _all_tracks_out_axes








from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_helpers import build_single_plotly_marginal_scatter_and_hist_over_time

@function_attributes(short_name=None, tags=['plotly', 'plotting'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-24 14:55', related_items=[])
def _plot_plotly_stack_marginal_scatter_and_hist_over_time(flat_decoded_marginal_posterior_df_context_dict, session_name: str, t_delta: float, epochs_decoding_time_bin_size: float ) -> Dict:
    """ 
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _plot_plotly_stack_marginal_scatter_and_hist_over_time
    
    session_name: str = curr_active_pipeline.session_name
    t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
    epochs_decoding_time_bin_size: float = 0.025

    a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', time_bin_size= 0.025, known_named_decoding_epochs_type='laps', data_grain='per_time_bin') # , known_named_decoding_epochs_type='laps'
    flat_context_list, flat_result_context_dict, flat_decoder_context_dict, flat_decoded_marginal_posterior_df_context_dict = a_new_fully_generic_result.get_results_matching_contexts(context_query=a_target_context, return_multiple_matches=True, debug_print=True)
    _flat_out_figs_dict = _plot_plotly_stack_marginal_scatter_and_hist_over_time(flat_decoded_marginal_posterior_df_context_dict=flat_decoded_marginal_posterior_df_context_dict, session_name=session_name, t_delta=t_delta, epochs_decoding_time_bin_size=epochs_decoding_time_bin_size)


    
    """
    #INPUTS: a_target_context: IdentifyingContext, a_result: DecodedFilterEpochsResult, a_decoded_marginal_posterior_df: pd.DataFrame, a_decoder: BasePositionDecoder
    _flat_out_figs_dict = {}

    for a_ctxt, a_decoded_marginal_posterior_df in flat_decoded_marginal_posterior_df_context_dict.items():
        print(a_ctxt)
        
        # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
        # a_decoded_marginal_posterior_df['delta_aligned_start_t'] = a_decoded_marginal_posterior_df['start'] - t_delta ## subtract off t_delta
        a_decoded_marginal_posterior_df = a_decoded_marginal_posterior_df.across_session_identity.add_session_df_columns(session_name=session_name, time_bin_size=epochs_decoding_time_bin_size, curr_session_t_delta=t_delta) # , time_col='t'
            
        a_fig, a_figure_context = build_single_plotly_marginal_scatter_and_hist_over_time(a_decoded_posterior_df=a_decoded_marginal_posterior_df, a_target_context=a_ctxt)
        a_fig = a_fig.update_layout(height=300, # Set your desired height
                                    margin=dict(t=20, b=0),  # Set top and bottom margins to 0
                                    )
        _flat_out_figs_dict[a_figure_context] = a_fig
        a_fig.show()
    ## END FOR
    return _flat_out_figs_dict









@function_attributes(short_name=None, tags=['pyqtgraph', 'scatter', 'histogram', 'AI', 'testing', 'unused'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-15 14:30', related_items=['plotly_pre_post_delta_scatter'])
def pyqtgraph_pre_post_delta_scatter(data_results_df: pd.DataFrame, data_context: Optional[IdentifyingContext]=None, 
                                     histogram_bins:int=25, common_plot_kwargs=None, scatter_kwargs=None,
                                     histogram_variable_name='P_Long', hist_kwargs=None,
                                     forced_range_y=[0.0, 1.0], time_delta_tuple=None, is_dark_mode: bool = True,
                                     figure_sup_huge_title_text: str=None, is_top_supertitle: bool = False, figure_footer_text: Optional[str]=None,
                                     existing_widget=None, **kwargs):
    """ Plots a scatter plot of a variable pre/post delta, with a histogram on each end corresponding to the pre/post delta distribution using PyQtGraph
    
    Args:
        data_results_df (pd.DataFrame): DataFrame containing the data to plot
        data_context (Optional[IdentifyingContext], optional): Context info for the plot. Defaults to None.
        histogram_bins (int, optional): Number of bins for histograms. Defaults to 25.
        common_plot_kwargs (dict, optional): Common kwargs for all plots. Defaults to None.
        scatter_kwargs (dict, optional): Kwargs for scatter plot. Defaults to None.
        histogram_variable_name (str, optional): Variable to plot. Defaults to 'P_Long'.
        hist_kwargs (dict, optional): Kwargs for histograms. Defaults to None.
        forced_range_y (list, optional): Y-axis range. Defaults to [0.0, 1.0].
        time_delta_tuple (tuple, optional): Tuple of (start, delta, end) times. Defaults to None.
        is_dark_mode (bool, optional): Use dark mode theme. Defaults to True.
        figure_sup_huge_title_text (str, optional): Super title text. Defaults to None.
        is_top_supertitle (bool, optional): Place supertitle at top. Defaults to False.
        figure_footer_text (Optional[str], optional): Footer text. Defaults to None.
        existing_widget (QWidget, optional): Existing widget to use. Defaults to None.

    Returns:
        tuple: (main_widget, figure_context) - the widget containing the plot and context info
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import pyqtgraph_pre_post_delta_scatter

        ## test AI-generated pyqtgraph version
        plot_row_identifier: str = f'{a_known_decoded_epochs_type.capitalize()} - {a_prefix.capitalize()} - {a_suffix} decoder' # should be like 'Laps (Masked) from Non-PBE decoder'

        fig_widget, figure_context = pyqtgraph_pre_post_delta_scatter(data_results_df=deepcopy(a_decoded_posterior_df), out_scatter_fig=None, 
                                        histogram_variable_name='P_Short', hist_kwargs=dict(), histogram_bins=histogram_bins,
                                        common_plot_kwargs=dict(),
                                        px_scatter_kwargs = dict(x='delta_aligned_start_t', y='P_Short', title=plot_row_identifier))
        fig_widget.show()
    """
    # Create app if needed
    app = mkQApp("Pre-Post Delta Scatter")
    
    # Copy dataframe to avoid modifying the original
    data_results_df = data_results_df.copy()
    debug_print = kwargs.get('debug_print', False)
    
    # Set up display labels
    pre_delta_label = 'Pre-delta'
    post_delta_label = 'Post-delta'

    # Initialize context and figure context dictionary
    if data_context is None:
        data_context = IdentifyingContext()  # empty context
    
    data_context = data_context.adding_context_if_missing(variable_name=histogram_variable_name)
    figure_context_dict = {'histogram_variable_name': histogram_variable_name}
    
    # Get session info
    if 'session_name' in data_results_df.columns:
        num_unique_sessions = data_results_df['session_name'].nunique(dropna=True)
    else:
        num_unique_sessions = 1
    
    data_context = data_context.adding_context_if_missing(num_sessions=num_unique_sessions)
    figure_context_dict['num_unique_sessions'] = num_unique_sessions
    
    # Get time bin size info
    num_unique_time_bin_sizes = data_results_df.time_bin_size.nunique(dropna=True)
    unique_time_bin_sizes = np.unique(data_results_df.time_bin_size.to_numpy())
    
    if debug_print:
        print(f'num_unique_sessions: {num_unique_sessions}, num_unique_time_bins: {num_unique_time_bin_sizes}')
    
    if num_unique_time_bin_sizes == 1:
        assert len(unique_time_bin_sizes) == 1
        figure_context_dict['t_bin_size'] = unique_time_bin_sizes[0]
    else:
        figure_context_dict['n_unique_t_bin_sizes'] = num_unique_time_bin_sizes
    
    # Initialize plotting kwargs
    if hist_kwargs is None:
        hist_kwargs = {}
    
    if scatter_kwargs is None:
        scatter_kwargs = {}
    
    if common_plot_kwargs is None:
        common_plot_kwargs = {}
    
    # Build title
    if num_unique_sessions == 1:
        main_title = f"Session {scatter_kwargs.get('title', 'UNKNOWN')}"
    else:
        main_title = f"Across Sessions {scatter_kwargs.get('title', 'UNKNOWN')} ({num_unique_sessions} Sessions)"
    
    if num_unique_time_bin_sizes > 1:
        main_title = main_title + f" - {num_unique_time_bin_sizes} Time Bin Sizes"
        figure_context_dict['n_tbin'] = num_unique_time_bin_sizes
    elif num_unique_time_bin_sizes == 1:
        time_bin_size = unique_time_bin_sizes[0]
        main_title = main_title + f" - time bin size: {time_bin_size} sec"
    else:
        main_title = main_title + f" - ERR: <No Entries in DataFrame>"
    
    figure_context_dict['title'] = main_title
    
    # Filter data for pre-delta and post-delta
    pre_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] <= 0]
    post_delta_df = data_results_df[data_results_df['delta_aligned_start_t'] > 0]
    
    # ==================================================================================================================== #
    # Build PyQtGraph Widget                                                                                               #
    # ==================================================================================================================== #
    
    # Create main widget if not provided
    if existing_widget is None:
        main_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        main_widget.setLayout(layout)
    else:
        main_widget = existing_widget
        layout = main_widget.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout()
            main_widget.setLayout(layout)
        
        # Clear existing layout
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    # Create title label if needed
    if figure_sup_huge_title_text:
        title_label = QtWidgets.QLabel(figure_sup_huge_title_text)
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        font = title_label.font()
        font.setBold(True)
        font.setPointSize(14)
        title_label.setFont(font)
        layout.addWidget(title_label)
    
    # Create GraphicsLayoutWidget for plots
    plot_widget = pg.GraphicsLayoutWidget()
    layout.addWidget(plot_widget)
    
    # Configure plot widget
    if is_dark_mode:
        plot_widget.setBackground('k')
    else:
        plot_widget.setBackground('w')
    
    # Create three plots side by side
    pre_hist_plot = plot_widget.addPlot(row=0, col=0)
    scatter_plot = plot_widget.addPlot(row=0, col=1)
    post_hist_plot = plot_widget.addPlot(row=0, col=2)
    
    # Link y axes
    pre_hist_plot.setYLink(scatter_plot)
    post_hist_plot.setYLink(scatter_plot)
    
    # Set titles
    pre_hist_plot.setTitle(pre_delta_label)
    scatter_plot.setTitle(main_title)
    post_hist_plot.setTitle(post_delta_label)
    
    # Set axis labels
    pre_hist_plot.setLabel('bottom', "# Events")
    scatter_plot.setLabel('bottom', "Delta-aligned Event Time (seconds)")
    scatter_plot.setLabel('left', "Probability of Short Track")  # This should be based on histogram_variable_name
    post_hist_plot.setLabel('bottom', "# Events")
    
    # Set y range
    if forced_range_y:
        pre_hist_plot.setYRange(forced_range_y[0], forced_range_y[1], padding=0)
        scatter_plot.setYRange(forced_range_y[0], forced_range_y[1], padding=0)
        post_hist_plot.setYRange(forced_range_y[0], forced_range_y[1], padding=0)
    
    # Create legends
    legend = scatter_plot.addLegend()
    
    # ==================================================================================================================== #
    # Add data to plots                                                                                                    #
    # ==================================================================================================================== #
    
    # Plot functions
    def plot_scatter_data(df, plot):
        """Plot scatter data, colored by time_bin_size or other variable"""
        if 'color' in common_plot_kwargs:
            color_by = common_plot_kwargs['color']
            unique_values = df[color_by].unique()
            
            for i, val in enumerate(unique_values):
                subset = df[df[color_by] == val]
                color = pg.intColor(i, len(unique_values))
                scatter = pg.ScatterPlotItem(
                    x=subset['delta_aligned_start_t'].values,
                    y=subset[histogram_variable_name].values,
                    pen=None, brush=color, size=8, alpha=0.5,
                    name=f"{val}"
                )
                plot.addItem(scatter)
        else:
            # Default scatter with single color
            scatter = pg.ScatterPlotItem(
                x=df['delta_aligned_start_t'].values,
                y=df[histogram_variable_name].values,
                pen=None, brush=(0, 135, 255, 150), size=8,
                name="Data"
            )
            plot.addItem(scatter)
    
    def plot_histogram_data(df, plot, is_vertical=True, align="left"):
        """Plot histogram of data
        
        Args:
            df: DataFrame with data
            plot: PyQtGraph plot item
            is_vertical: Whether the histogram is vertical (True) or horizontal (False)
            align: For horizontal histograms, whether to align "left", "right" or "center"
        """
        if len(df) == 0:
            return
            
        y = df[histogram_variable_name].values
        
        # Create bins
        bin_min, bin_max = 0, 1
        if forced_range_y:
            bin_min, bin_max = forced_range_y[0], forced_range_y[1]
        else:
            bin_min, bin_max = np.min(y), np.max(y)
        
        bins = np.linspace(bin_min, bin_max, histogram_bins)
        
        # Get histogram counts
        hist, bin_edges = np.histogram(y, bins=bins)
        bin_width = bin_edges[1] - bin_edges[0]
        
        # For handling different colors
        if 'color' in common_plot_kwargs:
            color_by = common_plot_kwargs['color']
            unique_values = df[color_by].unique()
            
            for i, val in enumerate(unique_values):
                subset = df[df[color_by] == val]
                if len(subset) == 0:
                    continue
                    
                y_subset = subset[histogram_variable_name].values
                hist, bin_edges = np.histogram(y_subset, bins=bins)
                
                color = pg.intColor(i, len(unique_values))
                
                if is_vertical:
                    # Vertical bars - height is the histogram count
                    bars = pg.BarGraphItem(
                        x=bin_edges[:-1], height=hist, 
                        width=bin_width * 0.8,
                        brush=color, pen=None
                    )
                    plot.addItem(bars)
                else:
                    # Horizontal bars with alignment options
                    if align == "left":
                        # Left-aligned: starts at x=0, extends right
                        bars = pg.BarGraphItem(
                            x=0, y=bin_edges[:-1], width=hist,
                            height=bin_width * 0.8,
                            brush=color, pen=None
                        )
                    elif align == "right":
                        # Right-aligned: ends at right edge, extends left
                        # Find the maximum histogram value for scaling
                        max_hist = hist.max() if len(hist) > 0 else 1
                        # Create bars with negative width to extend left
                        bars = pg.BarGraphItem(
                            x=max_hist, y=bin_edges[:-1], width=-hist,
                            height=bin_width * 0.8,
                            brush=color, pen=None
                        )
                    else:  # center
                        # Center-aligned
                        bars = pg.BarGraphItem(
                            x=-hist/2, y=bin_edges[:-1], width=hist,
                            height=bin_width * 0.8,
                            brush=color, pen=None
                        )
                    plot.addItem(bars)
        else:
            # Default histogram with single color
            if is_vertical:
                # Vertical histogram
                bars = pg.BarGraphItem(
                    x=bin_edges[:-1], height=hist, 
                    width=bin_width * 0.8,
                    brush=(100, 100, 255, 150), pen=None
                )
                plot.addItem(bars)
            else:
                # Horizontal histogram with alignment
                if align == "left":
                    bars = pg.BarGraphItem(
                        x=0, y=bin_edges[:-1], width=hist,
                        height=bin_width * 0.8,
                        brush=(100, 100, 255, 150), pen=None
                    )
                elif align == "right":
                    # Find max for scaling
                    max_hist = hist.max() if len(hist) > 0 else 1
                    bars = pg.BarGraphItem(
                        x=max_hist, y=bin_edges[:-1], width=-hist,
                        height=bin_width * 0.8,
                        brush=(100, 100, 255, 150), pen=None
                    )
                else:  # center
                    bars = pg.BarGraphItem(
                        x=-hist/2, y=bin_edges[:-1], width=hist,
                        height=bin_width * 0.8,
                        brush=(100, 100, 255, 150), pen=None
                    )
                plot.addItem(bars)


    # Plot pre-delta histogram (horizontal, left-aligned)
    plot_histogram_data(pre_delta_df, pre_hist_plot, is_vertical=False, align="left")

    # Plot scatter plot
    plot_scatter_data(data_results_df, scatter_plot)

    # Plot post-delta histogram (horizontal, right-aligned)
    plot_histogram_data(post_delta_df, post_hist_plot, is_vertical=False, align="right")

    
    # Add epoch shapes if provided
    if time_delta_tuple is not None:
        assert len(time_delta_tuple) == 3
        earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end = time_delta_tuple
        delta_relative_t_start, delta_relative_t_delta, delta_relative_t_end = (
            np.array([earliest_delta_aligned_t_start, t_delta, latest_delta_aligned_t_end]) - t_delta
        )
        
        # Get track config colors
        long_short_display_config_manager = LongShortDisplayConfigManager()
        
        if is_dark_mode:
            long_epoch_color = pg.mkColor(long_short_display_config_manager.long_epoch_config.mpl_color)
            short_epoch_color = pg.mkColor(long_short_display_config_manager.short_epoch_config.mpl_color)
            y_zero_line_color = pg.mkColor('rgba(50,50,50,100)')  # dark grey
            vertical_epoch_divider_line_color = pg.mkColor('rgba(0,0,0,100)')  # black
        else:
            long_epoch_color = pg.mkColor(long_short_display_config_manager.long_epoch_config_light_mode.mpl_color)
            short_epoch_color = pg.mkColor(long_short_display_config_manager.short_epoch_config_light_mode.mpl_color)
            y_zero_line_color = pg.mkColor('rgba(200,200,200,100)')  # light grey
            vertical_epoch_divider_line_color = pg.mkColor('rgba(255,255,255,100)')  # white
        
        # Add horizontal zero line
        zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(y_zero_line_color, width=9))
        scatter_plot.addItem(zero_line)
        
        # Add vertical divider line at x=0
        divider_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(vertical_epoch_divider_line_color, width=3))
        scatter_plot.addItem(divider_line)
        
        # Add region items for Long and Short epochs
        long_region = pg.LinearRegionItem(
            values=[delta_relative_t_start, delta_relative_t_delta],
            brush=long_epoch_color,
            alpha=0.3,
            movable=False
        )
        scatter_plot.addItem(long_region)
        
        short_region = pg.LinearRegionItem(
            values=[delta_relative_t_delta, delta_relative_t_end],
            brush=short_epoch_color,
            alpha=0.3,
            movable=False
        )
        scatter_plot.addItem(short_region)
        
        # Add text labels for regions
        long_text = pg.TextItem("Long", anchor=(0.5, 0))
        long_text.setPos((delta_relative_t_start + delta_relative_t_delta) / 2, 0.95)
        scatter_plot.addItem(long_text)
        
        short_text = pg.TextItem("Short", anchor=(0.5, 0))
        short_text.setPos((delta_relative_t_delta + delta_relative_t_end) / 2, 0.95)
        scatter_plot.addItem(short_text)
    
    # Add footer text if provided
    if figure_footer_text:
        footer_label = QtWidgets.QLabel(figure_footer_text)
        footer_label.setAlignment(QtCore.Qt.AlignCenter)
        font = footer_label.font()
        font.setPointSize(10)
        footer_label.setFont(font)
        footer_label.setStyleSheet("color: gray;")
        layout.addWidget(footer_label)
    
    # Set column stretch factors to match the original Plotly layout
    plot_widget.ci.layout.setColumnStretchFactor(0, 1)  # Pre-delta histogram (10%)
    plot_widget.ci.layout.setColumnStretchFactor(1, 8)  # Scatter plot (80%)
    plot_widget.ci.layout.setColumnStretchFactor(2, 1)  # Post-delta histogram (10%)
    
    # Create context info
    figure_context = IdentifyingContext(**figure_context_dict)
    figure_context = figure_context.adding_context_if_missing(
        **data_context.get_subset(subset_includelist=['epochs_name', 'data_grain']).to_dict(),
        plot_type='scatter+hist', 
        comparison='pre-post-delta', 
        variable_name=histogram_variable_name
    )
    
    # Create a preferred filename for exporting
    preferred_filename = sanitize_filename_for_Windows(figure_context.get_subset(subset_excludelist=[]).get_description())
    
    # Store metadata with the widget for later access
    main_widget.setProperty('figure_context', figure_context.to_dict())
    main_widget.setProperty('preferred_filename', preferred_filename)
    
    # Add export functionality
    def export_to_png(path=None):
        if path is None:
            path = f"{preferred_filename}.png"
        
        exporter = pg.exporters.ImageExporter(plot_widget.scene())
        exporter.export(path)
        return path
    
    def export_to_svg(path=None):
        if path is None:
            path = f"{preferred_filename}.svg"
        
        exporter = pg.exporters.SVGExporter(plot_widget.scene())
        exporter.export(path)
        return path
    
    # Attach export methods to the widget
    main_widget.export_to_png = export_to_png
    main_widget.export_to_svg = export_to_svg
    
    return main_widget, figure_context


# ==================================================================================================================== #
# 2025-03-03 - Unit Time Binned Spike Count Masking of Decoding                                                        #
# ==================================================================================================================== #


@function_attributes(short_name=None, tags=['plot-helper', 'matplotlib', 'unit-activity', 'black-inactive-time-bins', 'time-bin'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-04 10:11', related_items=[])
def _plot_low_firing_time_bins_overlay_image(widget, time_bin_edges, mask_rgba, zorder=11.0):
    """ plots the black masks for low-firing time bins on the specified widget track
    
    ## Visually dimming low-firing bins

        ## find the time bins with insufficient spikes in them.

        ## Darken them by overlaying something


    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _plot_low_firing_time_bins_overlay_image

        
        # time_window_centers = deepcopy(results1D.continuous_results['global'].time_bin_containers[0].centers)
        ## INPUTS: unique_units, time_bin_edges, unit_specific_time_binned_spike_counts
        time_bin_edges: NDArray = deepcopy(results1D.continuous_results['global'].time_bin_edges[0])
        spikes_df: pd.DataFrame = deepcopy(get_proper_global_spikes_df(curr_active_pipeline))
        # unique_units = np.unique(spikes_df['aclu']) # sorted
        unit_specific_time_binned_spike_counts, unique_units, (is_time_bin_active, inactive_mask, mask_rgba) = spikes_df.spikes.compute_unit_time_binned_spike_counts_and_mask(time_bin_edges=time_bin_edges)
        _plot_low_firing_time_bins_overlay_image(widget=widget, time_bin_edges=time_bin_edges, mask_rgba=mask_rgba)
    """
    ## INPUTS: is_time_bin_active
    # Create mask of inactive time bins
    # inactive_mask = ~is_time_bin_active
    # mask_rgba = np.zeros((1, len(is_time_bin_active), 4), dtype=np.uint8)
    # mask_rgba[0, inactive_mask, :] = [0, 0, 0, 200]  # Black with 80% opacity for inactive bins
    an_ax = widget.axes[0]

    ## OUTPUTS: mask_rgba
    xmin = time_bin_edges[0]
    xmax = time_bin_edges[-1]
    ymin, ymax = an_ax.get_ylim()
    x_first_extent = (xmin, xmax, ymin, ymax)

    # Setup the heatmap colormap
    low_spiking_heatmap_imshow_kwargs = dict(
        origin='lower',
        aspect='auto',
        interpolation='nearest',
        extent=x_first_extent,
        animated=False,
    )

    # Plot the spike counts as a heatmap
    low_firing_bins_image = an_ax.imshow(mask_rgba, **low_spiking_heatmap_imshow_kwargs, zorder=zorder)
    widget.plots.low_firing_bins_image = low_firing_bins_image
    return low_firing_bins_image


@function_attributes(short_name=None, tags=['timeline-track', 'firing-rate', 'unit-spiking'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-03 15:28', related_items=[])
def add_unit_spike_count_visualization(active_2d_plot, neuron_ids: NDArray, time_bin_edges: NDArray, unit_specific_time_binned_spike_counts: NDArray, a_dock_config=None, extended_dock_title_info=None, neuron_colors_map=None, use_neuron_colors=True):
    """Adds a new row visualization for unit-specific time binned spike counts
    
    Args:
        active_2d_plot: The plot container to add this visualization to
        a_decoder_name (str): Name of the decoder for display purposes
        a_position_decoder: The decoder object containing neuron information
        time_window_centers (np.ndarray): Centers of time bins (n_time_bins,)
        unit_specific_time_binned_spike_counts (np.ndarray): Spike counts for each unit over time (n_aclus, n_time_bins)
        a_dock_config: Configuration for the dock
        extended_dock_title_info: Additional title info to append
    
    Returns:
        tuple: The created widget and figure components
        
        
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import add_unit_spike_count_visualization
        
        time_bin_edges: NDArray = deepcopy(results1D.continuous_results['global'].time_bin_edges[0])
        spikes_df: pd.DataFrame = deepcopy(get_proper_global_spikes_df(curr_active_pipeline))

        unique_units = np.unique(spikes_df['aclu']) # sorted
        unit_specific_time_binned_spike_counts: NDArray = np.array([
            np.histogram(spikes_df.loc[spikes_df['aclu'] == unit, 't_rel_seconds'], bins=time_bin_edges)[0]
            for unit in unique_units
        ])

        # unique_units.shape
        unit_specific_time_binned_spike_counts # .shape (n_aclus, n_time_bins)

        ## INPUTS: unique_units, time_bin_edges, unit_specific_time_binned_spike_counts
        widget, matplotlib_fig, matplotlib_fig_axes, dDisplayItem = add_unit_spike_count_visualization(active_2d_plot, neuron_ids=unique_units, time_bin_edges=time_bin_edges, unit_specific_time_binned_spike_counts=unit_specific_time_binned_spike_counts, a_dock_config=None, extended_dock_title_info=None)

        
    Usage 2:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _plot_low_firing_time_bins_overlay_image
    
        target_item_identifiers_list = ['ContinuousDecode_long_LR - t_bin_size: 0.025', 'ContinuousDecode_long_RL - t_bin_size: 0.025', 'ContinuousDecode_short_LR - t_bin_size: 0.025', 'ContinuousDecode_short_RL - t_bin_size: 0.025', 'ContinuousDecode_longnon-PBE-pseudo2D marginals', 'ContinuousDecode_shortnon-PBE-pseudo2D marginals', 'non-PBE_marginal_over_track_ID_ContinuousDecode - t_bin_size: 0.05', 'Masked Non-PBE Pseudo2D']
    for an_identifier in target_item_identifiers_list:
        widget, matplotlib_fig, matplotlib_fig_axes = active_2d_plot.find_matplotlib_render_plot_widget(an_identifier)
        if widget is not None:
            _plot_low_firing_time_bins_overlay_image(widget=widget, time_bin_edges=time_bin_edges, mask_rgba=mask_rgba)
            
    """
    

    
    ## Add a new row displaying unit spike counts over time
    identifier_name: str = f'SpikeCountsOverTime'
    if extended_dock_title_info is not None:
        identifier_name += extended_dock_title_info
    print(f'identifier_name: {identifier_name}')

    widget, matplotlib_fig, matplotlib_fig_axes, dDisplayItem = active_2d_plot.add_new_matplotlib_render_plot_widget(name=identifier_name, dockSize=(80, 200), display_config=a_dock_config)
    an_ax = matplotlib_fig_axes[0]

    variable_name: str = f'Spike Counts'
    
    # Get neuron IDs for y-axis labels
    # neuron_ids = a_position_decoder.pf.ratemap.neuron_ids
    n_neurons, n_time_bins = np.shape(unit_specific_time_binned_spike_counts)
    assert len(neuron_ids) == n_neurons
    n_neurons = len(neuron_ids)
    
    xmin = time_bin_edges[0]
    xmax = time_bin_edges[-1]
    ymin = 0
    ymax = n_neurons
    x_first_extent = (xmin, xmax, ymin, ymax)
    
    # Generate neuron colors if not provided
    if neuron_colors_map is None and use_neuron_colors:
        # from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import paired_separately_sort_neurons, paired_incremental_sort_neurons # _display_directional_template_debugger
        # from neuropy.utils.indexing_helpers import paired_incremental_sorting, union_of_arrays, intersection_of_arrays
        # from pyphoplacecellanalysis.General.Mixins.DataSeriesColorHelpers import UnitColoringMode, DataSeriesColorHelpers
        # from pyphocorehelpers.gui.Qt.color_helpers import QColor, build_adjusted_color
        from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import build_neurons_color_map
        neurons_colors_array = build_neurons_color_map(n_neurons)
        neuron_colors_map = {neuron_id: neurons_colors_array[:, i] for i, neuron_id in enumerate(neuron_ids)}
    
    if use_neuron_colors:
        # Create a colored image matrix where each row uses the neuron's color with intensity proportional to spike count
        image_matrix = np.zeros((n_neurons, n_time_bins, 4))  # RGBA format
        
        # Find max count for normalization
        max_count = np.max(unit_specific_time_binned_spike_counts)
        
        for i, neuron_id in enumerate(neuron_ids):
            base_color = neuron_colors_map[neuron_id]
            for j in range(n_time_bins):
                # Scale color intensity by spike count
                normalized_count = unit_specific_time_binned_spike_counts[i, j] / max_count if max_count > 0 else 0
                # Create a color with intensity proportional to spike count
                image_matrix[i, j] = [
                    base_color[0] * normalized_count,
                    base_color[1] * normalized_count,
                    base_color[2] * normalized_count,
                    1.0  # Full alpha
                ]
        
        # Plot the custom colored spike counts
        image = an_ax.imshow(image_matrix, origin='lower', aspect='auto', extent=x_first_extent, interpolation='nearest')
    else:
        # Use standard heatmap as before
        from neuropy.utils.matplotlib_helpers import get_heatmap_cmap
        
        spike_count_heatmap_imshow_kwargs = dict(
            origin='lower',
            cmap=get_heatmap_cmap(cmap='viridis', bad_color='black', under_color='white', over_color='red'),
            aspect='auto',
            interpolation='nearest',
            extent=x_first_extent,
            animated=False,
        )
        image = an_ax.imshow(unit_specific_time_binned_spike_counts, **spike_count_heatmap_imshow_kwargs)
    

    widget.plots.image = image
    
    # Configure axes
    an_ax.set_xlabel('Time Window')
    an_ax.set_ylabel('Neuron ID')
    

    # Add title
    an_ax.set_title(f'{variable_name} over Time')

    # Update the params
    widget.params.variable_name = variable_name
    if extended_dock_title_info is not None:
        widget.params.extended_dock_title_info = deepcopy(extended_dock_title_info)
    
    ## Update the plots_data - used for crosshairs tracing and other things
    if time_bin_edges is not None:
        widget.plots_data.time_bin_edges = deepcopy(time_bin_edges)
    widget.plots_data.unit_specific_time_binned_spike_counts = deepcopy(unit_specific_time_binned_spike_counts)
    widget.plots_data.neuron_ids = deepcopy(neuron_ids)
    
    active_2d_plot.sync_matplotlib_render_plot_widget(identifier=identifier_name)
    widget.draw()
    return widget, matplotlib_fig, matplotlib_fig_axes, dDisplayItem












# ==================================================================================================================== #
# 2025-02-27 - Filtering Pipeline                                                                                      #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['pipeline', 'filter', 'qclu'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-27 14:31', related_items=[])
def filtered_by_frate_and_qclu(curr_active_pipeline, desired_qclu_subset=[1, 2], desired_minimum_inclusion_fr_Hz: float = 4.0):
    """ Filter and return a copy of pipeline components by qclus and min_fr_Hz
    
    Parameters
    ----------
    curr_active_pipeline : NeuropyPipeline
        The pipeline containing computation results to filter
    desired_qclu_subset : list, optional
        List of quality cluster values to include, by default [1, 2]
    desired_minimum_inclusion_fr_Hz : float, optional
        Minimum firing rate threshold in Hz, by default 4.0
        
    Returns
    -------
    tuple
        (filtered_directional_laps_results, filtered_track_templates, filtered_directional_merged_decoders, filtered_rank_order_results)
        All components filtered to include only specified neurons


    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import filtered_by_frate_and_qclu

        # Get filtered pipeline components
        filtered_directional_laps_results, filtered_track_templates, filtered_merged_decoders, filtered_rank_order = filtered_by_frate_and_qclu(
            curr_active_pipeline, 
            desired_qclu_subset=[1, 2], 
            desired_minimum_inclusion_fr_Hz=4.0
        )

        # Display summary of filtered results
        print(f"Filtered neurons: {len(filtered_track_templates.any_decoder_neuron_IDs)}")
        if filtered_merged_decoders is not None:
            print(f"Filtered decoders available: Yes")
        if filtered_rank_order is not None:
            print(f"Filtered rank order available: Yes")

    """
    filtered_pipeline = deepcopy(curr_active_pipeline)

    # Get the original components from the pipeline
    directional_laps_results = filtered_pipeline.global_computation_results.computed_data['DirectionalLaps']
    
    # Get templates with filtering criteria
    track_templates = directional_laps_results.get_templates(
        minimum_inclusion_fr_Hz=desired_minimum_inclusion_fr_Hz, 
        included_qclu_values=desired_qclu_subset
    )
    
    # Apply filtering to get neurons that meet criteria
    filtered_track_templates = track_templates.filtered_by_frate_and_qclu(
        minimum_inclusion_fr_Hz=desired_minimum_inclusion_fr_Hz, 
        included_qclu_values=desired_qclu_subset
    )
    
    # Get the neuron IDs that passed filtering
    filtered_any_decoder_neuron_IDs = deepcopy(filtered_track_templates.any_decoder_neuron_IDs)
    
    # Filter the directional laps results by these neuron IDs
    filtered_directional_laps_results = directional_laps_results.filtered_by_included_aclus(
        filtered_any_decoder_neuron_IDs
    )
    
    # Filter additional components if they exist in the pipeline
    filtered_directional_merged_decoders = None
    if 'DirectionalMergedDecoders' in filtered_pipeline.global_computation_results.computed_data:
        directional_merged_decoders_result = filtered_pipeline.global_computation_results.computed_data['DirectionalMergedDecoders']
        filtered_directional_merged_decoders = directional_merged_decoders_result.filtered_by_included_aclus(
            filtered_any_decoder_neuron_IDs
        )
    
    filtered_rank_order_results = None
    if 'RankOrder' in filtered_pipeline.global_computation_results.computed_data:
        rank_order_results = filtered_pipeline.global_computation_results.computed_data['RankOrder']
        if hasattr(rank_order_results, 'filtered_by_included_aclus'):
            filtered_rank_order_results = rank_order_results.filtered_by_included_aclus(
                filtered_any_decoder_neuron_IDs
            )
    
    # Return all filtered components
    return filtered_pipeline


# ==================================================================================================================== #
# Pre 2025-02-27                                                                                                       #
# ==================================================================================================================== #

@function_attributes(short_name=None, tags=['USEFUL', 'unused', 'debug', 'visualizztion', 'SpikeRasterWindow'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-14 14:01', related_items=[])
def plot_attached_BinByBinDecodingDebugger(spike_raster_window, curr_active_pipeline, a_decoder: BasePositionDecoder, a_decoded_result: Union[DecodedFilterEpochsResult, SingleEpochDecodedResult], n_max_debugged_time_bins:int=25, name_suffix: str = 'unknoown', **kwargs):
    """ 
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_attached_BinByBinDecodingDebugger

    ## INPUTS: a_decoder, a_decoded_result
        
    a_decoder_name: str = 'long_LR'
    a_decoder = all_directional_pf1D_Decoder_dict[a_decoder_name]
    a_decoded_result = a_continuously_decoded_dict[a_decoder_name]

    ## INPUTS: a_decoder, a_decoded_result
    bin_by_bin_debugger, win, out_pf1D_decoder_template_objects, (plots_container, plots_data), _on_update_fcn = plot_attached_BinByBinDecodingDebugger(spike_raster_window, curr_active_pipeline, a_decoder=a_decoder, a_decoded_result=a_decoded_result)

    
    """
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.BinByBinDecodingDebugger import BinByBinDebuggingData, BinByBinDecodingDebugger

    return BinByBinDecodingDebugger.plot_attached_BinByBinDecodingDebugger(spike_raster_window=spike_raster_window, curr_active_pipeline=curr_active_pipeline, a_decoder=a_decoder, a_decoded_result=a_decoded_result, n_max_debugged_time_bins=n_max_debugged_time_bins, name_suffix=name_suffix, **kwargs)




@function_attributes(short_name=None, tags=['mixin', 'sync', 'QT'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-24 17:07', related_items=[])
class Decoded2DPosteriorTimeSyncMixin:
    """ Implementors recieve simple time updates from another plot """
    
    def update(self, t, defer_render=False):
        raise NotImplementedError

    def _update_plots(self):
        """ Implementor must override! """
        raise NotImplementedError
    
        
    # ==================================================================================================================== #
    # QT Slots                                                                                                             #
    # ==================================================================================================================== #
    
    @QtCore.pyqtSlot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        # print(f'Decoded2DPosteriorTimeSyncMixin.on_window_changed(start_t: {start_t}, end_t: {end_t})')
        self.update(end_t, defer_render=False)
        pass

    @QtCore.pyqtSlot(float, float, float)
    def on_window_duration_changed(self, start_t, end_t, duration):
        """ changes self.half_render_window_duration """
        # print(f'LiveWindowedData.on_window_duration_changed(start_t: {start_t}, end_t: {end_t}, duration: {duration})')
        pass        


    ############### Rate-Limited SLots ###############:
    ##################################################
    ## For use with pg.SignalProxy
    # using signal proxy turns original arguments into a tuple
    @QtCore.pyqtSlot(object)
    def on_window_changed_rate_limited(self, evt):
        self.on_window_changed(*evt)
        

    # @property
    # def n_epochs(self) -> int:
    #     return np.shape(self.active_epochs_df)[0]

    # def lookup_label_from_index(self, an_idx: int) -> int:
    #     """ Looks of the proper epoch "label", as in the value in the 'label' column of active_epochs_df, from a linear index such as that provided by the slider control.

    #     curr_epoch_label = lookup_label_from_index(a_plotter, an_idx)
    #     print(f'curr_epoch_label: {curr_epoch_label} :::') ## end line

    #     """
    #     curr_epoch_label = self.active_epochs_df['label'].iloc[an_idx] # gets the correct epoch label for the linear IDX
    #     curr_redundant_label_lookup_label = self.active_epochs_df.label.to_numpy()[an_idx]
    #     # print(f'curr_redundant_label_lookup_label: {curr_redundant_label_lookup_label} :::') ## end line
    #     assert str(curr_redundant_label_lookup_label) == str(curr_epoch_label), f"curr_epoch_label: {str(curr_epoch_label)} != str(curr_redundant_label_lookup_label): {str(curr_redundant_label_lookup_label)}"
    #     return curr_epoch_label


    # def find_nearest_time_index(self, target_time: float) -> Optional[int]:
    #     """ finds the index of the nearest time from the active epochs
    #     """
    #     from neuropy.utils.indexing_helpers import find_nearest_time
    #     df = self.active_epochs_df
    #     df, closest_index, closest_time, matched_time_difference = find_nearest_time(df=df, target_time=target_time, time_column_name='start', max_allowed_deviation=0.01, debug_print=False)
    #     # df.iloc[closest_index]
    #     return closest_index
    

@function_attributes(short_name=None, tags=['mixin'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-24 17:08', related_items=[])
class DataSlicingVisualizer(Decoded2DPosteriorTimeSyncMixin):
    """Visualizes 3D data slicing using ImageView widget and time slider
    
    Args:
        data (np.ndarray): 3D numpy array to visualize (P[x][y][t]). If None, generates demo data
        title (str, optional): Window title. Defaults to 'Data Slicing Visualizer'
        
        
    Usage:
    
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import DataSlicingVisualizer

        ## INPUTS: time_bin_edges, p_x_given_n

        # p_x_given_n.shape # (59, 8, 103512)
        visualizer = DataSlicingVisualizer(time_bin_edges=time_bin_edges, data=p_x_given_n)
        visualizer.set_data(time_bin_edges=time_bin_edges, new_data=p_x_given_n)
        visualizer.show()

        _out_conn = spike_raster_window.spike_raster_plt_2d.window_scrolled.connect(lambda start_t, end_t: visualizer.on_window_changed(start_t=start_t, end_t=end_t))

    """
    
    def __init__(self, time_bin_edges=None, data=None, measured_df=None, x_bin_edges=None, y_bin_edges=None, title='Data Slicing Visualizer'):
        self.app = pg.mkQApp("Data Slicing Example")
        self.title = title
        self.setup_ui()

        assert time_bin_edges is not None
        assert data is not None
        self.time_bin_edges = time_bin_edges
        self.data = data
        self.measured_df = measured_df
        self.x_bin_edges = x_bin_edges
        self.y_bin_edges = y_bin_edges
        self.position_marker = None

    def setup_ui(self):
        """Initialize the UI components"""
        self.win = QtWidgets.QMainWindow()
        self.win.resize(800, 800)
        self.win.setWindowTitle(self.title)
        
        # Setup central widget and layout
        self.cw = QtWidgets.QWidget()
        self.win.setCentralWidget(self.cw)
        self.layout = QtWidgets.QGridLayout()
        self.cw.setLayout(self.layout)
        
        # Create image view
        self.imv = pg.ImageView()
        self.layout.addWidget(self.imv, 0, 0)
        
        # Add time slider
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.layout.addWidget(self.time_slider, 1, 0)
        self.time_slider.valueChanged.connect(self.update_time_slice)

    def set_data(self, time_bin_edges, new_data, measured_df=None, x_bin_edges=None, y_bin_edges=None):
        self.time_bin_edges = time_bin_edges
        self.data = new_data
        if measured_df is not None:
            self.measured_df = measured_df
        if x_bin_edges is not None:
            self.x_bin_edges = x_bin_edges
        if y_bin_edges is not None:
            self.y_bin_edges = y_bin_edges

        self.time_bin_edges = time_bin_edges
        self.data = new_data
        if measured_df is not None:
            self.measured_df = measured_df
        
        print(f"Data shape: {self.data.shape}, expecting (x, y, t)")
        assert np.shape(self.data)[-1] == (len(self.time_bin_edges)-1), f"np.shape(self.data)[-1]: {np.shape(self.data)[-1]}, (len(self.time_bin_edges)-1): {(len(self.time_bin_edges)-1)}"
        self.time_slider.setMaximum(self.data.shape[-1] - 1) ## last dimension
        self.update_time_slice()



    def update_time_slice(self):
        """Update the image view based on current time slider value"""
        t = self.time_slider.value()
        
        # Get current slice and create normalized version for display
        current_slice = self.data[:,:,t]
        
        # Debug information
        print(f"Slice {t}: Shape={current_slice.shape}, min={np.min(current_slice):.8f}, max={np.max(current_slice):.8f}")
        
        # Set the image with transposition
        self.imv.setImage(current_slice.T)
        
        # Force update the histogram levels based on current data
        min_val = np.min(current_slice)
        max_val = np.max(current_slice)
        if min_val != max_val:  # Avoid division by zero
            self.imv.setLevels(min_val, max_val)
        
        # Add the animal's position marker if we have position data
        self.add_position_marker(t)


    def add_position_marker(self, t_index):
        """Add a marker showing the animal's current position"""
        if self.measured_df is None:
            return
        
        # Get the time for the current slice
        current_time = self.time_bin_edges[t_index]
        
        # Find the closest time point in the measured_df
        if 't' in self.measured_df.columns:
            # Find the closest timestamp
            closest_idx = (self.measured_df['t'] - current_time).abs().idxmin()
            real_x = self.measured_df.loc[closest_idx, 'x']
            real_y = self.measured_df.loc[closest_idx, 'y']
            
            # Convert real coordinates to bin indices
            if hasattr(self, 'x_bin_edges') and hasattr(self, 'y_bin_edges'):
                x_idx = np.digitize(real_x, self.x_bin_edges) - 1
                y_idx = np.digitize(real_y, self.y_bin_edges) - 1
                
                # Clip to valid range
                x_idx = np.clip(x_idx, 0, len(self.x_bin_edges)-2)
                y_idx = np.clip(y_idx, 0, len(self.y_bin_edges)-2)
            else:
                # Estimate using data shape
                x_bins, y_bins = self.data.shape[0:2]
                x_min, x_max = self.measured_df['x'].min(), self.measured_df['x'].max()
                y_min, y_max = self.measured_df['y'].min(), self.measured_df['y'].max()
                
                x_idx = int((real_x - x_min) / (x_max - x_min) * (x_bins-1))
                y_idx = int((real_y - y_min) / (y_max - y_min) * (y_bins-1))
                
                x_idx = np.clip(x_idx, 0, x_bins-1)
                y_idx = np.clip(y_idx, 0, y_bins-1)
            
            # Remove previous marker if it exists
            if self.position_marker is not None:
                self.imv.getView().removeItem(self.position_marker)
            
            # CORRECTED: When we transpose the image, x becomes the second dimension 
            # and y becomes the first dimension in the displayed image
            self.position_marker = pg.ScatterPlotItem()
            self.position_marker.setData(
                [x_idx],  # This is now correctly the horizontal axis in the displayed image
                [y_idx],  # This is now correctly the vertical axis in the displayed image
                size=15, 
                pen=pg.mkPen('w', width=2), 
                brush=pg.mkBrush(255, 0, 0, 200)
            )
            
            # Add marker to the view
            self.imv.getView().addItem(self.position_marker)
            print(f"Added position marker at real position (x={real_x:.2f}, y={real_y:.2f}), bin indices (x={x_idx}, y={y_idx})")



    def show(self):
        """Display the visualization window"""
        self.time_slider.setMaximum(self.data.shape[2] - 1)
        self.update_time_slice()
        
        # Use a better colormap for small probability values
        # self.imv.setColorMap(pg.colormap.get('viridis'))
         # Use a better colormap for small probability values
        self.imv.setColorMap(pg.colormap.get('inferno'))  # 'inferno' or 'hot' work well for log-transformed data
    

        # Don't manually set levels - let the data range drive it
        # self.imv.setHistogramRange(0.0, 1.0)
        # self.imv.setLevels(0.0, 1.0)
        
        self.win.show()


    def test_image_display(self):
        """Generate and display a simple test image to verify ImageView is working"""
        # Create a test pattern - a simple gradient
        test_width, test_height = 100, 100
        test_image = np.zeros((test_width, test_height))
        
        # Create a gradient pattern
        for i in range(test_width):
            for j in range(test_height):
                test_image[i, j] = (i + j) / (test_width + test_height)
        
        # Alternatively, create a checkerboard pattern
        # checkerboard = np.zeros((test_width, test_height))
        # for i in range(test_width):
        #     for j in range(test_height):
        #         checkerboard[i, j] = (i % 20 < 10) ^ (j % 20 < 10)
        
        # Display the test image
        self.imv.setImage(test_image)
        self.imv.setLevels(0, 1)
        print(f"Displaying test image with shape {test_image.shape}")
        print(f"Test image min: {np.min(test_image)}, max: {np.max(test_image)}")
        


    # ==================================================================================================================== #
    # Decoded2DPosteriorTimeSyncMixin Conformances                                                                         #
    # ==================================================================================================================== #
    def find_nearest_time_index(self, target_time: float) -> Optional[int]:
        """ finds the index of the nearest time from the active epochs
        """
        from neuropy.utils.indexing_helpers import find_nearest_time
        time_bin_edges, closest_index, closest_time, matched_time_difference = find_nearest_time(self.time_bin_edges, target_time=target_time, max_allowed_deviation=0.1, debug_print=False)
        return closest_index


    def update(self, t, defer_render=False):
        """ updates the slider 
        """
        closest_index = self.find_nearest_time_index(target_time=t)
        print(f'closest_index: {closest_index}')
        if closest_index is not None:
            # closest_index
            self.time_slider.setValue(closest_index)


    # def _update_plots(self):
    #     """ Implementor must override! """
    #     raise NotImplementedError
    
        
    # ==================================================================================================================== #
    # QT Slots                                                                                                             #
    # ==================================================================================================================== #
    
    @QtCore.pyqtSlot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        # print(f'Decoded2DPosteriorTimeSyncMixin.on_window_changed(start_t: {start_t}, end_t: {end_t})')
        self.update(start_t, defer_render=False)
        

    @QtCore.pyqtSlot(float, float, float)
    def on_window_duration_changed(self, start_t, end_t, duration):
        """ changes self.half_render_window_duration """
        # print(f'LiveWindowedData.on_window_duration_changed(start_t: {start_t}, end_t: {end_t}, duration: {duration})')
        pass        


    ############### Rate-Limited SLots ###############:
    ##################################################
    ## For use with pg.SignalProxy
    # using signal proxy turns original arguments into a tuple
    @QtCore.pyqtSlot(object)
    def on_window_changed_rate_limited(self, evt):
        self.on_window_changed(*evt)
        




# class TwoDimensionalPosteriorDisplayingTSWidget(PyqtgraphTimeSynchronizedWidget):
#     """ Plots the decoded position posterior (2D) at a given moment in time. 

#     Usage:
            # from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import TwoDimensionalPosteriorDisplayingTSWidget
                        
            # new_widget: TwoDimensionalPosteriorDisplayingTSWidget = TwoDimensionalPosteriorDisplayingTSWidget(plot_function_name="pho_test_2025-02-21", data=results2D.continuous_results['long'])
            # new_widget.show()

#     """
#     # Application/Window Configuration Options:
#     applicationName = 'TwoDimensionalPosteriorDisplayingTSApp'
#     windowName = 'TwoDimensionalPosteriorDisplayingTSWidgetWindow'
    
#     enable_debug_print = True
    
#     # sigCrosshairsUpdated = QtCore.Signal(object, str, str) # (self, name, trace_value) - CrosshairsTracingMixin Conformance

#     # @property
#     # def time_window_centers(self):
#     #     """The time_window_centers property."""
#     #     return self.active_one_step_decoder.time_window_centers # get time window centers (n_time_window_centers,)
    

#     # @property
#     # def posterior_variable_to_render(self):
#     #     """The occupancy_mode_to_render property."""
#     #     return self.params.posterior_variable_to_render
#     # @posterior_variable_to_render.setter
#     # def posterior_variable_to_render(self, value):
#     #     self.params.posterior_variable_to_render = value
#     #     # on update, be sure to call self._update_plots()
#     #     self._update_plots()
    
#     @property
#     def last_t(self):
#         raise NotImplementedError(f'Parent property that should not be accessed!')

#     @property
#     def active_plot_target(self):
#         """The active_plot_target property."""
#         return self.getRootPlotItem()
    


#     def __init__(self, name='TwoDimensionalPosteriorDisplayingTSWidget', plot_function_name=None, scrollable_figure=True, application_name=None, window_name=None, parent=None, data=None, **kwargs):
#         """_summary_
#         , disable_toolbar=True, size=(5.0, 4.0), dpi=72
#         ## allows toggling between the various computed occupancies: such as raw counts,  normalized location, and seconds_occupancy
#             occupancy_mode_to_render: ['seconds_occupancy', 'num_pos_samples_occupancy', 'num_pos_samples_smoothed_occupancy', 'normalized_occupancy']
        
#         Calls self.setup(), self.buildUI(), self._update_plots()
#         """
#         self.data = deepcopy(data)
#         super().__init__(application_name=application_name, window_name=(window_name or PyqtgraphTimeSynchronizedWidget.windowName), debug_print=False, **kwargs, parent=parent) # Call the inherited classes __init__ method    
        

#     def setup(self):
#         assert hasattr(self.ui, 'connections')
        
#         # self.setup_spike_rendering_mixin() # NeuronIdentityAccessingMixin
#         # self.app = pg.mkQApp(self.applicationName)
#         # self.params = VisualizationParameters(self.applicationName)
        

#         # # Add a trace region (initially hidden)
#         # self.trace_region = pg.LinearRegionItem(movable=True, brush=(0, 0, 255, 50))
#         # self.trace_region.setZValue(10)  # Ensure it appears above the plot
#         # self.trace_region.hide()  # Initially hide the trace region
#         # self.plot_widget.addItem(self.trace_region)

#         # # Override the PlotWidget's mouse events
#         # self.plot_widget.scene().sigMouseClicked.connect(self.mouse_clicked)
#         # self.plot_widget.scene().sigMouseMoved.connect(self.mouse_moved)
#         # self.plot_widget.scene().sigMouseReleased.connect(self.mouse_released)
#         # self.dragging = False
#         # self.start_pos = None

#         # self.params.shared_axis_order = 'row-major'
#         # self.params.shared_axis_order = 'column-major'
#         # self.params.shared_axis_order = None
        
#         ## Build the colormap to be used:
#         # self.params.cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
#         # self.params.cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map
#         # self.params.image_margins = 0.0
#         # self.params.image_bounds_extent, self.params.x_range, self.params.y_range = pyqtplot_build_image_bounds_extent(self.active_one_step_decoder.xbin, self.active_one_step_decoder.ybin, margin=self.params.image_margins, debug_print=self.enable_debug_print)
#         pass


#     def _buildGraphics(self):
#         """ called by self.buildUI() which usually is not overriden. """
#         from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.CustomGraphicsLayoutWidget import CustomViewBox, CustomGraphicsLayoutWidget

#         ## More Involved Mode:
#         # self.ui.root_graphics_layout_widget = pg.GraphicsLayoutWidget()
#         self.ui.root_graphics_layout_widget = CustomGraphicsLayoutWidget()

#         # self.ui.root_view = self.ui.root_graphics_layout_widget.addViewBox()
#         ## lock the aspect ratio so pixels are always square
#         # self.ui.root_view.setAspectLocked(True)

#         ## Create image item
        
#         # self.ui.imv = pg.ImageItem(border='w')
#         # self.ui.root_view.addItem(self.ui.imv)
#         # self.ui.root_view.setRange(QtCore.QRectF(*self.params.image_bounds_extent))

#         self.ui.root_plot_viewBox = None
#         self.ui.root_plot_viewBox = CustomViewBox()
#         self.ui.root_plot_viewBox.setObjectName('RootPlotCustomViewBox')
        
#         # self.ui.root_plot = self.ui.root_graphics_layout_widget.addPlot(row=0, col=0, title=None) # , name=f'PositionDecoder'
#         self.ui.root_plot = self.ui.root_graphics_layout_widget.addPlot(row=0, col=0, title=None, viewBox=self.ui.root_plot_viewBox)
#         self.ui.root_plot.setObjectName('RootPlot')
#         # self.ui.root_plot.addItem(self.ui.imv, defaultPadding=0.0)  # add ImageItem to PlotItem
#         ## TODO: add item here
#         # self.ui.root_plot.showAxes(True)
#         self.ui.root_plot.hideButtons() # Hides the auto-scale button
        
#         self.ui.root_plot.showAxes(False)     
#         # self.ui.root_plot.setRange(xRange=self.params.x_range, yRange=self.params.y_range, padding=0.0)
#         # Sets only the panning limits:
#         # self.ui.root_plot.setLimits(xMin=self.params.x_range[0], xMax=self.params.x_range[-1], yMin=self.params.y_range[0], yMax=self.params.y_range[-1])

#         ## Sets all limits:
#         # _x, _y, _width, _height = self.params.image_bounds_extent # [23.923329354140844, 123.85967782096927, 241.7178791533281, 30.256480996256016]
#         # self.ui.root_plot.setLimits(minXRange=_width, maxXRange=_width, minYRange=_height, maxYRange=_height)
#         # self.ui.root_plot.setLimits(xMin=self.params.x_range[0], xMax=self.params.x_range[-1], yMin=self.params.y_range[0], yMax=self.params.y_range[-1],
#         #                             minXRange=_width, maxXRange=_width, minYRange=_height, maxYRange=_height)
        
#         self.ui.root_plot.setMouseEnabled(x=False, y=False)
#         self.ui.root_plot.setMenuEnabled(enableMenu=False)
        
#         # ## Optional Interactive Color Bar:
#         # bar = pg.ColorBarItem(values= (0, 1), colorMap=self.params.cmap, width=5, interactive=False) # prepare interactive color bar
#         # # Have ColorBarItem control colors of img and appear in 'plot':
#         # bar.setImageItem(self.ui.imv, insert_in=self.ui.root_plot)
        
#         self.ui.layout.addWidget(self.ui.root_graphics_layout_widget, 0, 0) # add the GLViewWidget to the layout at 0, 0
        
#         # Set the color map:
#         # self.ui.imv.setColorMap(self.params.cmap)
#         ## Set initial view bounds
#         # self.ui.root_view.setRange(QtCore.QRectF(0, 0, 600, 600))

    
#     def update(self, t, defer_render=False):
#         if self.enable_debug_print:
#             print(f'PyqtgraphTimeSynchronizedWidget.update(t: {t})')
    
#         # # Finds the nearest previous decoded position for the time t:
#         # self.last_window_index = np.searchsorted(self.time_window_centers, t, side='left') # side='left' ensures that no future values (later than 't') are ever returned
#         # self.last_window_time = self.time_window_centers[self.last_window_index] # If there is no suitable index, return either 0 or N (where N is the length of `a`).
#         # Update the plots:
#         if not defer_render:
#             self._update_plots()


#     def _update_plots(self):
#         if self.enable_debug_print:
#             print(f'PyqtgraphTimeSynchronizedWidget._update_plots()')

#         # Update the existing one:
#         # self.ui.root_plot.setRange(xRange=self.params.x_range, yRange=self.params.y_range, padding=0.0)
#         # Sets only the panning limits:
#         # self.ui.root_plot.setLimits(xMin=self.params.x_range[0], xMax=self.params.x_range[-1], yMin=self.params.y_range[0], yMax=self.params.y_range[-1])

#         ## Sets all limits:
#         # _x, _y, _width, _height = self.params.image_bounds_extent # [23.923329354140844, 123.85967782096927, 241.7178791533281, 30.256480996256016]
#         # self.ui.root_plot.setLimits(minXRange=_width, maxXRange=_width, minYRange=_height, maxYRange=_height)
#         # self.ui.root_plot.setLimits(xMin=self.params.x_range[0], xMax=self.params.x_range[-1], yMin=self.params.y_range[0], yMax=self.params.y_range[-1],
#         #                             minXRange=_width, maxXRange=_width, minYRange=_height, maxYRange=_height)
        
#         # Update the plots:
#         # curr_time_window_index = self.last_window_index
#         # curr_t = self.last_window_time

#         # if curr_time_window_index is None or curr_t is None:
#         #     return # return without updating

#         # self.setWindowTitle(f'{self.windowName} - {image_title} t = {curr_t}')
#         # self.setWindowTitle(f'PyqtgraphTimeSynchronizedWidget - {image_title} t = {curr_t}')
#         pass

#     # ==================================================================================================================== #
#     # QT Slots                                                                                                             #
#     # ==================================================================================================================== #
    
#     @pg.QtCore.Slot(float, float)
#     def on_window_changed(self, start_t, end_t):
#         # called when the window is updated
#         if self.enable_debug_print:
#             print(f'PyqtgraphTimeSynchronizedWidget.on_window_changed(start_t: {start_t}, end_t: {end_t})')
#         # if self.enable_debug_print:
#         #     profiler = pg.debug.Profiler(disabled=True, delayed=True)

#         self.update(end_t, defer_render=False)
#         # if self.enable_debug_print:
#         #     profiler('Finished calling _update_plots()')
        


# ==================================================================================================================== #
# 2025-02-21 - Angular Binning, Transition Matricies, and More                                                         #
# ==================================================================================================================== #
# # ### Plots: Explore Binning Position and Angle
# # Convert angles to radians
# angles = np.deg2rad(global_pos_df['approx_head_dir_degrees'])

# # Create circular histogram
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.hist(angles, bins=36, density=True, alpha=0.70)

# # Set labels
# ax.set_title("Circular Histogram of Head Direction")

# # Show plot
# plt.show()
# df = deepcopy(global_pos_df)

# # Normalize time to use as radius
# radii = (df['t'] - df['t'].min()) / (df['t'].max() - df['t'].min())

# # Create circular scatter plot
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.scatter(angles, radii, alpha=0.20, s=1)
# # ax.plot(angles, radii, alpha=0.20, s=1)
# # Set labels
# ax.set_title("Circular Scatter Plot of Head Direction Over Time")

# # Show plot
# plt.show()

# # Create circular scatter plot with line connecting points
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# # Plot points
# ax.scatter(angles, radii, alpha=0.75, s=5)  # Smaller point size

# # Connect points with a line
# ax.plot(angles, radii, alpha=0.5, linewidth=1)

# # Set labels
# ax.set_title("Circular Scatter Plot of Head Direction Over Time")

# # Show plot
# plt.show()




# Now I have the columns `global_pos_df[['binned_x', 'binned_y', 'head_dir_angle_binned']]` and I'd like to visualize a heatmap showing:
# 1. and 

import numpy as np
from typing import Dict, List, Tuple, Optional
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui
from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow

class CircularBinnedImageRenderingWindow(BasicBinnedImageRenderingWindow):
    """Renders circular/angular heatmaps within each spatial bin
    
    Usage:

        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import CircularBinnedImageRenderingWindow
        
        # ## INPUTS: occupancy_map, n_xbins, n_ybins n_x_bins=n_xbins, n_y_bins=n_ybins, n_dir_bins=n_dir_bins
        # Create sample angular distribution data
        # n_x_bins, n_y_bins = 10, 10
        # n_angle_bins = 36
        # angular_matrix = np.random.rand(n_x_bins, n_y_bins, n_angle_bins)
        angular_matrix = deepcopy(occupancy_map)

        # Create window
        window = CircularBinnedImageRenderingWindow(
            angular_matrix=angular_matrix,
            xbins=np.arange(n_xbins),
            ybins=np.arange(n_ybins),
            n_angle_bins=n_dir_bins,
            name='angular_distribution',
            title='Angular Distribution per Position Bin'
        )

        window.render_all_circular_heatmaps()

    """
    
    def __init__(self, angular_matrix, xbins=None, ybins=None, n_angle_bins: int=None, **kwargs):
        """
        angular_matrix: shape (n_x_bins, n_y_bins, n_angle_bins) containing angular distribution data
        """
        assert n_angle_bins is not None
        pos_only_mat = np.sum(deepcopy(angular_matrix), axis=-1)
        super().__init__(matrix=pos_only_mat, xbins=xbins, ybins=ybins, **kwargs)
        self.n_angle_bins = n_angle_bins
        self.angular_data = deepcopy(angular_matrix)
        
    def add_circular_heatmap(self, bin_x: int, bin_y: int, angular_data: np.ndarray) -> None:
        """Adds a circular heatmap to a specific spatial bin"""
        # Create circular representation
        theta = np.linspace(0, 2*np.pi, self.n_angle_bins+1)[:-1]  # Angular positions
        r = angular_data  # Radial values from angular distribution
        
        # Convert to cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Create polygon for the circular heatmap
        polygon = QtGui.QPolygonF()
        for px, py in zip(x, y):
            polygon.append(QtCore.QPointF(px + bin_x + 0.5, py + bin_y + 0.5))
            
        # Create path for smooth rendering
        path = QtGui.QPainterPath()
        path.addPolygon(polygon)
        path.closeSubpath()
        
        # Create graphics item
        item = pg.QtGui.QGraphicsPathItem(path)
        
        # Set color based on distribution intensity
        color = pg.mkColor('w')  # Base color
        color.setAlphaF(0.7)     # Semi-transparent
        item.setBrush(pg.mkBrush(color))
        item.setPen(pg.mkPen(None))  # No border
        
        # Add to plot
        assert (window.plot_names is not None) and (len(window.plot_names) > 0) # 'angular_distribution'
        plot_name: str = window.plot_names[0]
        assert plot_name in self.plots, f"plot_name: {plot_name} not in self.plots"
        self.plots[plot_name].mainPlotItem.addItem(item)
        

    def render_all_circular_heatmaps(self):
        """Renders circular heatmaps for all spatial bins"""
        n_x, n_y, _ = self.angular_data.shape
        
        for x in range(n_x):
            for y in range(n_y):
                angular_dist = self.angular_data[x, y]
                # Normalize the distribution
                if np.sum(angular_dist) > 0:
                    angular_dist = angular_dist / np.max(angular_dist)
                    self.add_circular_heatmap(x, y, angular_dist)

    def init_UI(self):
        """Initialize the UI and render circular heatmaps"""
        super().init_UI()
        # self.render_all_circular_heatmaps()


def plot_spatial_angular_distributions(occupancy_map, subsample_factor=5):
    """Plot radar charts of angular distributions across spatial positions
    
    Args:
        occupancy_map (np.ndarray): 3D array (x_bins, y_bins, direction_bins)
        subsample_factor (int): Plot every Nth spatial bin to avoid overcrowding


    Usage:    
        fig, ax = plot_spatial_angular_distributions(occupancy_map, subsample_factor=4)
        plt.show()

    """
    n_x, n_y, n_angles = occupancy_map.shape
    
    # Create main figure
    fig, ax = plt.subplots(figsize=(25, 15), clear=True, num='test')
    
    # Calculate angles for radar plot (in radians)
    # theta = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    # Calculate bin edges for rose plot
    bins = np.linspace(0, 2*np.pi, n_angles+1)    

    
    # Plot radar at each subsampled position
    for i in range(0, n_x, subsample_factor):
        for j in range(0, n_y, subsample_factor):
            # Get angular distribution at this position
            values = occupancy_map[i,j,:]
            
            # Create small axes for this position
            # radar_ax = fig.add_axes([i/n_x, j/n_y, 1/n_x, 1/n_y], projection='polar')
            # radar_ax.plot(theta, values)
            # radar_ax.fill(theta, values, alpha=0.25)
            
            # Create small axes for this position
            _new_radial_ax = fig.add_axes([i/n_x, j/n_y, 1/n_x, 1/n_y], projection='polar')
            
            # Create rose plot using hist
            _new_radial_ax.hist(bins[:-1], bins=bins, weights=values, density=False, histtype='stepfilled')

            # pc = _new_radial_ax.pcolormesh(A, R, hist.T, cmap="magma_r")
            # fig.colorbar(pc)
            # _new_radial_ax.grid(True)

            _new_radial_ax.set_xticks([])
            _new_radial_ax.set_yticks([])
    
    return fig, ax


import numpy as np
import matplotlib.pyplot as plt

def radial_histogram(data, bins=12, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
    counts, edges = np.histogram(data, bins=bins, range=(0, 2*np.pi))
    widths = np.diff(edges)
    ax.bar(edges[:-1], counts, width=widths, bottom=0, align='edge', color='blue', alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def plot_spatial_angular_distributions(occupancy_map, subsample_factor=5):
    """ Usage
    
        fig, ax = plot_spatial_angular_distributions(occupancy_map, subsample_factor=2)
        plt.show()
    """
    n_x, n_y, n_angles = occupancy_map.shape
    fig, ax = plt.subplots(figsize=(25, 15), clear=True, num='test')

    # Draw grid boxes for each x/y bin
    for i in range(n_x):
        for j in range(n_y):
            x0 = i / n_x
            y0 = j / n_y
            w_ = 1 / n_x
            h_ = 1 / n_y
            rect = plt.Rectangle((x0, y0), w_, h_, fill=False, color='black', lw=1, transform=fig.transFigure)
            fig.add_artist(rect)

    # Size of each small polar subplot
    w = 0.6 * (subsample_factor / n_x)
    h = 0.6 * (subsample_factor / n_y)

    for i in range(0, n_x, subsample_factor):
        for j in range(0, n_y, subsample_factor):
            counts = occupancy_map[i, j, :]
            angles = np.hstack([np.full(int(counts[k]), (2*np.pi*(k + 0.5)) / n_angles) for k in range(n_angles)])
            pos_x = i / n_x
            pos_y = j / n_y
            ax_sub = fig.add_axes([pos_x, pos_y, w, h], projection='polar')
            radial_histogram(angles, bins=n_angles, ax=ax_sub)

    return fig, ax


def plot_directional_occupancy(occupancy_map, direction_bin):
    """Plot 2D heatmap for a specific head direction bin

    # 2. Visualize a single direction slice
    direction_bin = 1  # Example: looking at 180 degrees if using 36 bins
    plot_directional_occupancy(occupancy_map, direction_bin)

    # 3. Get total occupancy across all directions
    total_spatial_occupancy = np.sum(occupancy_map, axis=2)
    plt.figure(figsize=(10,8))
    plt.imshow(total_spatial_occupancy, origin='lower')
    plt.colorbar(label='Total Count')
    plt.title('Total Spatial Occupancy')


    """
    plt.figure(figsize=(10,8))
    plt.imshow(occupancy_map[:,:,direction_bin], origin='lower')
    plt.colorbar(label='Count')
    plt.title(f'Occupancy for Direction Bin {direction_bin}')
    plt.xlabel('X bin')
    plt.ylabel('Y bin')


@function_attributes(short_name=None, tags=['HELPER', 'matplotlib'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-09-10 07:05', related_items=[])
def draw_radial_lines(rect_width, rect_height, n_bins):
    """Draws lines from rectangle center to perimeter, creating equal angular divisions
    
    
    Args:
        rect_width (float): Width of rectangle
        rect_height (float): Height of rectangle
        n_bins (int): Number of angular divisions desired
    
    Returns:
        list of tuples: [(x1,y1,x2,y2)] coordinates for each line
        
        
    Usage:
    
    
        plt.figure(num='box_line_test',clear=True)
        # Draw 8 radial divisions in a 100x80 rectangle
        lines = draw_radial_lines(100, 80, 8)

        # Plot the lines
        for x1,y1,x2,y2 in lines:
            plt.plot([x1,x2], [y1,y2], 'k-')
        plt.axis('equal')
        plt.show()


    """
    # Calculate center point
    center_x = rect_width / 2
    center_y = rect_height / 2
    
    # Calculate angles for each division
    angles = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
    
    lines = []
    for angle in angles:
        # Calculate direction vector
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Find intersection with rectangle boundary
        # Scale factor t = min positive value that hits boundary
        t_values = []
        
        # Check horizontal boundaries
        if dx != 0:
            t_values.extend([
                (0 - center_x) / dx,  # Left boundary
                (rect_width - center_x) / dx  # Right boundary
            ])
            
        # Check vertical boundaries
        if dy != 0:
            t_values.extend([
                (0 - center_y) / dy,  # Bottom boundary
                (rect_height - center_y) / dy  # Top boundary
            ])
            
        # Get smallest positive t value
        t = min(t for t in t_values if t > 0)
        
        # Calculate endpoint
        end_x = center_x + t * dx
        end_y = center_y + t * dy
        
        lines.append((center_x, center_y, end_x, end_y))
    
    return lines



@function_attributes(short_name=None, tags=['working', 'angular', 'head_dir_angle_binned'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-21 00:48', related_items=[])
def compute_3d_occupancy_map(df, n_x_bins=50, n_y_bins=50, n_dir_bins=8):
    """Creates a 3D occupancy map with fixed dimensions regardless of observed data
    
    Args:
        df (pd.DataFrame): DataFrame with binned columns
        n_x_bins (int): Number of x position bins
        n_y_bins (int): Number of y position bins
        n_dir_bins (int): Number of head direction bins
        
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import compute_3d_occupancy_map
        # 1. Compute the 3D occupancy map
        

        ## INPUTS: global_pf2D

        xbin_edges = global_pf2D.xbin
        ybin_edges = global_pf2D.ybin
        # Create evenly spaced bin edges from 0 to 360
        n_dir_bins: int = 8
        angle_dir_bin_edges = np.linspace(0, 360, n_dir_bins + 1)

        n_xbins: int = len(xbin_edges) - 1
        n_ybins: int = len(ybin_edges) - 1
        n_dir_bins: int = len(angle_dir_bin_edges) - 1

        print(f'n_xbins: {n_xbins}, n_ybins: {n_ybins}, n_dir_bins: {n_dir_bins}')

        # Use pd.cut with the explicit bin edges
        global_pos_df['head_dir_angle_binned'] = pd.cut(global_pos_df['approx_head_dir_degrees'], bins=angle_dir_bin_edges, labels=False, include_lowest=True)
        global_pos_df = global_pos_df.position.adding_binned_position_columns(xbin_edges=xbin_edges, ybin_edges=ybin_edges)
        global_pos_df = global_pos_df.dropna(axis='index', subset=['binned_x', 'binned_y', 'head_dir_angle_binned'])
        global_pos_df

        
        # occupancy_map, bin_counts = compute_3d_occupancy_map(global_pos_df)

        occupancy_map, bin_counts = compute_3d_occupancy_map(global_pos_df, n_x_bins=n_xbins, n_y_bins=n_ybins, n_dir_bins=n_dir_bins)

        # Print the shape and counts
        print(f"Occupancy map shape: {occupancy_map.shape}")
        print(f"Unique bins per dimension: {bin_counts}")

    """
    # Create all possible combinations
    x_bins = range(n_x_bins)
    y_bins = range(n_y_bins)
    dir_bins = range(n_dir_bins)
    
    # Use crosstab with specific bins to force output size
    occupancy_map = pd.crosstab(
        index=[df['binned_x'], df['binned_y']], 
        columns=df['head_dir_angle_binned'],
        dropna=False  # Keep all combinations
    ).reindex(
        index=pd.MultiIndex.from_product([x_bins, y_bins]),
        columns=dir_bins,
        fill_value=0  # Fill missing combinations with 0
    ).values.reshape(n_x_bins, n_y_bins, n_dir_bins)
    
    return occupancy_map, {'x': n_x_bins, 'y': n_y_bins, 'dir': n_dir_bins}



import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from neuropy.utils.mixins.binning_helpers import get_bin_centers, get_bin_edges

@function_attributes(short_name=None, tags=['sparse', 'transition-matrix', 'N-dimensional', 'binning', 'sparse'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-21 12:35', related_items=[])
def create_sparse_matrix_from_bins(df: pd.DataFrame, bin_edges_dict: Dict[str, NDArray]) -> Optional[csr_matrix]:
    """
    Creates a sparse matrix from a DataFrame, using a dictionary to specify
    bin edges for each relevant column.

    Args:
        df: Pandas DataFrame with columns to be binned.
        bin_edges_dict: Dictionary where keys are column names in `df`
                           and values are 1D arrays of bin edges for that column.

    Returns:
        A scipy.sparse.csr_matrix.  Returns None if the input is invalid.
    """

    # Input validation: Check if all columns in bin_edges_dict exist in df
    if not all(col in df.columns for col in bin_edges_dict):
        print("Error: Not all columns in bin_edges_dict are present in the DataFrame.")
        return None
    if not bin_edges_dict:
        print("Error: bin_edges_dict is empty")
        return None

    # Calculate bin indices for each column
    indices_dict: Dict[str, NDArray] = {}
    for col, edges in bin_edges_dict.items():
        indices_dict[col] = np.digitize(df[col], edges) - 1

    # Determine the dimensionality of the sparse matrix
    num_dimensions: int = len(bin_edges_dict)

    # Create row and column indices based on dimensionality
    row_indices: NDArray
    col_indices: NDArray
    shape: Tuple[int, int]
    data_values: NDArray

    if num_dimensions == 1:
        # 1D case:  Row indices are just the bin indices of the single column
        col_name: str = list(bin_edges_dict.keys())[0]
        row_indices = np.zeros(len(df), dtype=int) # all rows are in the 0th "row"
        col_indices = indices_dict[col_name]
        shape = (1, len(bin_edges_dict[col_name]) - 1) # 1 row, num_bins columns (edges are one longer than centers)
        data_values = df[col_name].to_numpy() # Or a constant, like np.ones(len(df))

    elif num_dimensions == 2:
        # 2D case: Row and column indices from the two columns
        col_names: List[str] = list(bin_edges_dict.keys())
        row_indices = indices_dict[col_names[0]]
        col_indices = indices_dict[col_names[1]]
        shape = (len(bin_edges_dict[col_names[0]]) - 1, len(bin_edges_dict[col_names[1]]) - 1)
        # Assuming you want to count occurrences, use ones as data values
        data_values = np.ones(len(df))

    elif num_dimensions >= 3:
        # "nD" case (represented as a flattened 2D matrix)
        col_names = list(bin_edges_dict.keys())
        row_indices = indices_dict[col_names[0]]  # First column is the row
        col_indices = np.zeros(len(df), dtype=int)

        # Accumulate column indices, multiplying by the size of each dimension
        multiplier: int = 1
        for i in range(1, num_dimensions):
            col_indices += indices_dict[col_names[i]] * multiplier
            # Subtract 1 from the length because edges are one element longer than centers
            multiplier *= (len(bin_edges_dict[col_names[i]]) - 1)

        shape = (len(bin_edges_dict[col_names[0]]) - 1, multiplier)
        data_values = np.ones(len(df)) # Or a constant, like np.ones(len(df))

    else:
        print("Error: Invalid number of dimensions (must be 1, 2, or 3+).")
        return None

    # Remove out-of-bounds indices
    valid_indices: NDArray[np.bool_] = (row_indices >= 0) & (row_indices < shape[0]) & (col_indices >= 0) & (col_indices < shape[1])
    row_indices = row_indices[valid_indices]
    col_indices = col_indices[valid_indices]
    data_values = data_values[valid_indices]

    # Create the sparse matrix (using coo_matrix for efficient construction)
    sparse_matrix: coo_matrix = coo_matrix((data_values, (row_indices, col_indices)), shape=shape)
    return sparse_matrix.tocsr()  # Convert to CSR for efficient row operations








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

@function_attributes(short_name=None, tags=['UNFINISHED', 'plotting', 'computing'], input_requires=[], output_provides=[], uses=['AddNewDecodedPosteriors_MatplotlibPlotCommand', '_perform_plot_multi_decoder_meas_pred_position_track'], used_by=[], creation_date='2025-02-13 14:58', related_items=['_perform_plot_multi_decoder_meas_pred_position_track'])
def add_continuous_decoded_posterior(spike_raster_window, curr_active_pipeline, desired_time_bin_size: float, debug_print=True):
    """ computes the continuously decoded position posteriors (if needed) using the pipeline, then adds them as a new track to the SpikeRaster2D 
    
    Usage:
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import add_continuous_decoded_posterior

        (nested_dock_items, nested_dynamic_docked_widget_container_widgets), (a_continuously_decoded_dict, pseudo2D_decoder, all_directional_pf1D_Decoder_dict) = add_continuous_decoded_posterior(spike_raster_window=spike_raster_window, curr_active_pipeline=curr_active_pipeline, desired_time_bin_size=0.05, debug_print=True)

    """
    # ==================================================================================================================== #
    # COMPUTING                                                                                                            #
    # ==================================================================================================================== #
    
    # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'], computation_kwargs_list=[{'time_bin_size': 0.058}], #computation_kwargs_list=[{'time_bin_size': 0.025}], 
    #                                                   enabled_filter_names=None, fail_on_exception=True, debug_print=False)
    # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['merged_directional_placefields', 'directional_decoders_decode_continuous'], computation_kwargs_list=[{'laps_decoding_time_bin_size': 0.058}, {'time_bin_size': 0.058, 'should_disable_cache':False}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
    curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'], computation_kwargs_list=[{'time_bin_size': desired_time_bin_size, 'should_disable_cache': False}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
    ## get the result data:
    try:
        ## Uses the `global_computation_results.computed_data['DirectionalDecodersDecoded']`
        directional_decoders_decode_result: DirectionalDecodersContinuouslyDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
        pseudo2D_decoder: BasePositionDecoder = directional_decoders_decode_result.pseudo2D_decoder
        all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
        a_continuously_decoded_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict.get(desired_time_bin_size, None)
        info_string: str = f" - t_bin_size: {desired_time_bin_size}"

    except (KeyError, AttributeError) as e:
        # KeyError: 'DirectionalDecodersDecoded'
        print(f'add_all_computed_time_bin_sizes_pseudo2D_decoder_decoded_epochs(...) failed to add any tracks, perhaps because the pipeline is missing any computed "DirectionalDecodersDecoded" global results. Error: "{e}". Skipping.')
        a_continuously_decoded_dict = None
        pseudo2D_decoder = None        
        pass

    except Exception as e:
        raise


    # # output_dict = _cmd.prepare_and_perform_add_pseudo2D_decoder_decoded_epoch_marginals(curr_active_pipeline=_cmd._active_pipeline, active_2d_plot=active_2d_plot, continuously_decoded_dict=deepcopy(a_continuously_decoded_dict), info_string=info_string, **enable_rows_config_kwargs)
    # output_dict = AddNewDecodedPosteriors_MatplotlibPlotCommand.prepare_and_perform_add_add_pseudo2D_decoder_decoded_epochs(curr_active_pipeline=curr_active_pipeline, active_2d_plot=active_2d_plot, continuously_decoded_dict=deepcopy(a_continuously_decoded_dict), info_string=info_string, a_pseudo2D_decoder=pseudo2D_decoder, debug_print=debug_print, **kwargs)
    # for a_key, an_output_tuple in output_dict.items():
    #     identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dDisplayItem = an_output_tuple                
    #     # if a_key not in all_time_bin_sizes_output_dict:
    #     #     all_time_bin_sizes_output_dict[a_key] = [] ## init empty list
    #     # all_time_bin_sizes_output_dict[a_key].append(an_output_tuple)
        
    #     assert (identifier_name not in flat_all_time_bin_sizes_output_tuples_dict), f"identifier_name: {identifier_name} already in flat_all_time_bin_sizes_output_tuples_dict: {list(flat_all_time_bin_sizes_output_tuples_dict.keys())}"
    #     flat_all_time_bin_sizes_output_tuples_dict[identifier_name] = an_output_tuple



    # ==================================================================================================================== #
    # PLOTTING                                                                                                             #
    # ==================================================================================================================== #
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import AddNewDecodedPosteriors_MatplotlibPlotCommand

    display_output = {}
    AddNewDecodedPosteriors_MatplotlibPlotCommand(spike_raster_window, curr_active_pipeline, active_config_name=None, active_context=None, display_output=display_output, action_identifier='actionPseudo2DDecodedEpochsDockedMatplotlibView')

    all_global_menus_actionsDict, global_flat_action_dict = spike_raster_window.build_all_menus_actions_dict()
    if debug_print:
        print(list(global_flat_action_dict.keys()))


    ## extract the components so the `background_static_scroll_window_plot` scroll bar is the right size:
    active_2d_plot = spike_raster_window.spike_raster_plt_2d
    
    active_2d_plot.params.enable_non_marginalized_raw_result = False
    active_2d_plot.params.enable_marginal_over_direction = False
    active_2d_plot.params.enable_marginal_over_track_ID = True


    menu_commands = [
        # 'DockedWidgets.LongShortDecodedEpochsDockedMatplotlibView',
        # 'DockedWidgets.DirectionalDecodedEpochsDockedMatplotlibView',
        # 'DockedWidgets.TrackTemplatesDecodedEpochsDockedMatplotlibView',
        # 'DockedWidgets.Pseudo2DDecodedEpochsDockedMatplotlibView',
        f'DockedWidgets.Pseudo2DDecodedEpochsDockedMatplotlibView_tbin_{desired_time_bin_size}' # 0.05
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

    return (nested_dock_items, nested_dynamic_docked_widget_container_widgets), (a_continuously_decoded_dict, pseudo2D_decoder, all_directional_pf1D_Decoder_dict)




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
    import nptyping as ND
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
    import nptyping as ND
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
# 2025-01-20 - Easy Decoding                                                                                           #
# ==================================================================================================================== #
@function_attributes(short_name=None, tags=['decoding', 'ACTIVE', 'useful'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-20 15:26', related_items=[])
def easy_independent_decoding(a_decoder: BasePositionDecoder, spikes_df: pd.DataFrame, time_bin_size: float = 0.025, t_start: float = 0.0, t_end: float = 2093.8978568242164):
    """ Uses the provieded decoder, spikes, and time binning parameters to decode the neural activity for the specified epoch.

    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import easy_independent_decoding


    time_bin_size: float = 0.025
    t_start = 0.0
    t_end = 2093.8978568242164
    _decoded_pos_outputs, (unit_specific_time_binned_spike_counts, time_bin_edges, spikes_df) = easy_independent_decoding(long_LR_decoder, spikes_df=spikes_df, time_bin_size=time_bin_size, t_start=t_start, t_end=t_end)
    most_likely_positions, p_x_given_n, most_likely_position_indicies, flat_outputs_container = _decoded_pos_outputs


    """
    neuron_IDs = deepcopy(a_decoder.neuron_IDs) # array([  2,   5,   8,  10,  14,  15,  23,  24,  25,  26,  31,  32,  33,  41,  49,  50,  51,  55,  58,  64,  69,  70,  73,  74,  75,  76,  78,  82,  83,  85,  86,  90,  92,  93,  96, 109])
    spikes_df: pd.DataFrame = deepcopy(spikes_df).spikes.sliced_by_neuron_id(neuron_IDs) ## filter everything down
    time_bin_edges: NDArray = np.arange(t_start, t_end + time_bin_size, time_bin_size)
    unique_units = np.unique(spikes_df['aclu']) # sorted
    unit_specific_time_binned_spike_counts: NDArray = np.array([
        np.histogram(spikes_df.loc[spikes_df['aclu'] == unit, 't_rel_seconds'], bins=time_bin_edges)[0]
        for unit in unique_units
    ])

    ## OUTPUT: time_bin_edges, unit_specific_time_binned_spike_counts
    _decoded_pos_outputs = a_decoder.decode(unit_specific_time_binned_spike_counts=unit_specific_time_binned_spike_counts, time_bin_size=time_bin_size, output_flat_versions=True, debug_print=True)
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
            _, ax, _return_out_artists_dict = plot_1D_most_likely_position_comparsions(pos_df, variable_name='x', time_window_centers=None, xbin=None, posterior=None, active_most_likely_positions_1D=None, ax=ax_list[0],
                                                            enable_flat_line_drawing=enable_flat_line_drawing, debug_print=debug_print, return_created_artists=True, **kwargs)
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
    if 'is_valid_epoch' in ripple_merged_complete_epoch_stats_df:
        ripple_merged_complete_epoch_stats_df = ripple_merged_complete_epoch_stats_df[ripple_merged_complete_epoch_stats_df['is_valid_epoch']] ## 136, 71 included requiring both
    else:
        print(f'WARN: missing column "is_valid_epoch" in `ripple_merged_complete_epoch_stats_df`')
        
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

@function_attributes(short_name=None, tags=['matplotlib', 'helper'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-14 14:09', related_items=[])
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
        # global_epoch_name = curr_active_pipeline.find_Global_epoch_name()
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
import nptyping as ND
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
# 2024-06-26 - Shuffled WCorr Output with working histogram                                                            #
# ==================================================================================================================== #
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
import nptyping as ND
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

