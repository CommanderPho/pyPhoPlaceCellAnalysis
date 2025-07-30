 
from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING

from neuropy.utils.result_context import IdentifyingContext

if TYPE_CHECKING:
    ## typehinting only imports here
    from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult


from copy import deepcopy
import sys
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import h5py
from typing_extensions import TypeAlias
from nptyping import NDArray
import pyphoplacecellanalysis.General.type_aliases as types
import numpy as np
import pandas as pd
from pyphocorehelpers.assertion_helpers import Assert
from pathlib import Path
import shutil ## for deleting directories

# NeuroPy (Diba Lab Python Repo) Loading
from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns
from neuropy.utils.dynamic_container import DynamicContainer # for _perform_two_step_position_decoding_computation
from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs # used in _subfn_compute_decoded_epochs to get only the valid (non-overlapping) epochs
from neuropy.core.epoch import Epoch, TimeColumnAliasesProtocol, subdivide_epochs, ensure_dataframe, ensure_Epoch
from neuropy.analyses.placefields import HDF_SerializationMixin, PfND

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder, computation_precidence_specifying_function, global_function

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, BayesianPlacemapPositionDecoder, DecodedFilterEpochsResult, SingleEpochDecodedResult, Zhang_Two_Step
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsResult, TrackTemplates, TrainTestSplitResult


from pyphoplacecellanalysis.General.Model.ComputationResults import ComputedResult
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field, custom_define
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin, HDF_Converter
from neuropy.utils.mixins.indexing_helpers import UnpackableMixin
from attrs import asdict, astuple, define, field, Factory
from neuropy.utils.indexing_helpers import PandasHelpers
from neuropy.core.epoch import EpochsAccessor, Epoch, ensure_dataframe
from neuropy.core.position import Position
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, SimpleFieldSizesReprMixin
from neuropy.core.epoch import Epoch, TimeColumnAliasesProtocol, subdivide_epochs, ensure_dataframe, ensure_Epoch

# import portion as P # Required for interval search: portion~=2.3.0
from pyphocorehelpers.indexing_helpers import partition_df_dict, partition_df

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df

import pyphoplacecellanalysis.General.type_aliases as types


# ### For _perform_recursive_latent_placefield_decoding
# from neuropy.utils import position_util
# from neuropy.core import Position
# from neuropy.analyses.placefields import perform_compute_placefields

"""-------------- Specific Computation Functions to be registered --------------"""
""" 2025-02-20 08:55 Renamed 'subdivide' -> 'frame_divide', 'subdivision' -> 'frame_division' throughout my recent non-PBE-related functions for clearity. 

# ==================================================================================================================== #
# In EpochComputationFunctions.py:                                                                                     #
# ==================================================================================================================== #
# Direct variable renames:
subdivided_epochs_results -> frame_divided_epochs_results
subdivided_epochs_df -> frame_divided_epochs_df
global_subivided_epochs_obj -> global_frame_divided_epochs_obj
global_subivided_epochs_df -> global_frame_divided_epochs_df
subdivided_epochs_specific_decoded_results_dict -> frame_divided_epochs_specific_decoded_results_dict

# Function/parameter renames:
subdivide_bin_size -> frame_divide_bin_size
min_subdivision_resolution -> min_frame_division_resolution
actual_subdivision_step_size -> actual_frame_division_step_size
num_subdivisions -> num_frame_divisions



# ==================================================================================================================== #
# In decoder_plotting_mixins.py:                                                                                       #
# ==================================================================================================================== #
subdivide_bin_size -> frame_divide_bin_size

"""

# [/c:/Users/pho/repos/Spike3DWorkEnv/NeuroPy/neuropy/core/session/Formats/Specific/KDibaOldDataSessionFormat.py:142](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/NeuroPy/neuropy/core/session/Formats/Specific/KDibaOldDataSessionFormat.py:142)
# ```python
#     @classmethod
#     def POSTLOAD_estimate_laps_and_replays(cls, sess):
#         """ a POSTLOAD function: after loading, estimates the laps and replays objects (replacing those loaded). """
#         print(f'POSTLOAD_estimate_laps_and_replays()...')
        
#         # 2023-05-16 - Laps conformance function (TODO 2023-05-16 - factor out?)
#         # lap_estimation_parameters = DynamicContainer(N=20, should_backup_extant_laps_obj=True) # Passed as arguments to `sess.replace_session_laps_with_estimates(...)`

#         lap_estimation_parameters = sess.config.preprocessing_parameters.epoch_estimation_parameters.laps
#         assert lap_estimation_parameters is not None

#         use_direction_dependent_laps: bool = lap_estimation_parameters.pop('use_direction_dependent_laps', True)
#         sess.replace_session_laps_with_estimates(**lap_estimation_parameters, should_plot_laps_2d=False) # , time_variable_name=None
#         ## add `use_direction_dependent_laps` back in:
#         lap_estimation_parameters.use_direction_dependent_laps = use_direction_dependent_laps

#         ## Apply the laps as the limiting computation epochs:
#         # computation_config.pf_params.computation_epochs = sess.laps.as_epoch_obj().get_non_overlapping().filtered_by_duration(1.0, 30.0)
#         if use_direction_dependent_laps:
#             print(f'.POSTLOAD_estimate_laps_and_replays(...): WARN: {use_direction_dependent_laps}')
#             # TODO: I think this is okay here.


#         # Get the non-lap periods using PortionInterval's complement method:
#         non_running_periods = Epoch.from_PortionInterval(sess.laps.as_epoch_obj().to_PortionInterval().complement()) # TODO 2023-05-24- Truncate to session .t_start, .t_stop as currently includes infinity, but it works fine.
        

#         # ## TODO 2023-05-19 - FIX SLOPPY PBE HANDLING
#         PBE_estimation_parameters = sess.config.preprocessing_parameters.epoch_estimation_parameters.PBEs
#         assert PBE_estimation_parameters is not None
#         PBE_estimation_parameters.require_intersecting_epoch = non_running_periods # 2023-10-06 - Require PBEs to occur during the non-running periods, REQUIRED BY KAMRAN contrary to my idea of what PBE is.
        
#         new_pbe_epochs = sess.compute_pbe_epochs(sess, active_parameters=PBE_estimation_parameters)
#         sess.pbe = new_pbe_epochs
#         updated_spk_df = sess.compute_spikes_PBEs()

#         # 2023-05-16 - Replace loaded replays (which are bad) with estimated ones:
        
        
        
#         # num_pre = session.replay.
#         replay_estimation_parameters = sess.config.preprocessing_parameters.epoch_estimation_parameters.replays
#         assert replay_estimation_parameters is not None
#         ## Update the parameters with the session-specific values that couldn't be determined until after the session was loaded:
#         replay_estimation_parameters.require_intersecting_epoch = non_running_periods
#         replay_estimation_parameters.min_inclusion_fr_active_thresh = 1.0
#         replay_estimation_parameters.min_num_unique_aclu_inclusions = 5
#         sess.replace_session_replays_with_estimates(**replay_estimation_parameters)
        
#         # ### Get both laps and existing replays as PortionIntervals to check for overlaps:
#         # replays = sess.replay.epochs.to_PortionInterval()
#         # laps = sess.laps.as_epoch_obj().to_PortionInterval() #.epochs.to_PortionInterval()
#         # non_lap_replays = Epoch.from_PortionInterval(replays.difference(laps)) ## Exclude anything that occcurs during the laps themselves.
#         # sess.replay = non_lap_replays.to_dataframe() # Update the session's replay epochs from those that don't intersect the laps.

#         # print(f'len(replays): {len(replays)}, len(laps): {len(laps)}, len(non_lap_replays): {non_lap_replays.n_epochs}')
        

#         # TODO 2023-05-22: Write the parameters somewhere:
#         replays = sess.replay.epochs.to_PortionInterval()

#         ## This is the inverse approach of the new method, which loads the parameters from `sess.config.preprocessing_parameters`
#         # sess.config.preprocessing_parameters = DynamicContainer(epoch_estimation_parameters=DynamicContainer.init_from_dict({
#         #     'laps': lap_estimation_parameters,
#         #     'PBEs': PBE_estimation_parameters,
#         #     'replays': replay_estimation_parameters
#         # }))

#         return sess
# ```


@metadata_attributes(short_name=None, tags=['global', 'decode', 'result', 'extracted'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-06-30 07:39', related_items=[])
@custom_define(slots=False, eq=False, repr=False)
class ComputeGlobalEpochBase(ComputedResult):
    """ Relates to using all time on the track except for detected PBEs as the placefield inputs. This includes the laps and the intra-lap times. 
    Importantly `lap_dir` is poorly defined for the periods between the laps, so something like head-direction might have to be used.

    #TODO 2025-02-13 12:40: - [ ] Should compute the correct Epochs, add it to the sessions as a new Epoch (I guess as a new FilteredSession context!! 


    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import ComputeGlobalEpochBase

    History:
    Extracted from `Compute_NonPBE_Epochs` on 2025-06-30 07:40  

    """
    _VersionedResultMixin_version: str = "2025.06.30_0" # to be updated in your IMPLEMENTOR to indicate its version

    single_global_epoch_df: pd.DataFrame = serialized_field()

    @property
    def frame_divide_bin_size(self) -> float:
        """The frame_divide_bin_size property."""
        return self.pos_df.attrs['frame_divide_bin_size']
    @frame_divide_bin_size.setter
    def frame_divide_bin_size(self, value):
        self.pos_df.attrs['frame_divide_bin_size'] = value


    @classmethod
    def init_from_pipeline(cls, curr_active_pipeline):
        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        # Build an Epoch object containing a single epoch, corresponding to the global epoch for the entire session:
        single_global_epoch_df: pd.DataFrame = pd.DataFrame({'start': [t_start], 'stop': [t_end], 'label': [0]})
        # single_global_epoch_df['label'] = single_global_epoch_df.index.to_numpy()
        # single_global_epoch: Epoch = Epoch(self.single_global_epoch_df)
        _obj = cls(single_global_epoch_df=single_global_epoch_df) # , global_epoch_only_non_PBE_epoch_df=global_epoch_only_non_PBE_epoch_df, a_new_training_df_dict=a_new_training_df_dict, a_new_test_df_dict=a_new_test_df_dict, skip_training_test_split=skip_training_test_split
        return _obj

    def __attrs_post_init__(self):
        # Add post-init logic here
        pass


    @function_attributes(short_name=None, tags=['pure'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-06-30 08:12', related_items=[])
    @classmethod
    def recompute(cls, single_global_epoch: Epoch, curr_active_pipeline, pfND_ndim: int = 2, epochs_decoding_time_bin_size: float = 0.025, a_new_training_df_dict: Optional[Dict]=None, a_new_testing_epoch_obj_dict: Optional[Dict[types.DecoderName, Epoch]]=None):
        """ For a specified decoding time_bin_size and ndim (1D or 2D), copies the global pfND, builds new epoch objects, then decodes both train_test and continuous epochs

        test_epoch_specific_decoded_results_dict, continuous_specific_decoded_results_dict, new_decoder_dict, new_pfs_dict = a_new_NonPBE_Epochs_obj.recompute(curr_active_pipeline=curr_active_pipeline, epochs_decoding_time_bin_size = 0.058)

        pfND_ndim: 1 - 1D, 2 - 2D

        single_global_epoch: Epoch = Epoch(self.single_global_epoch_df),
        a_new_training_df_dict = self.a_new_training_df_dict, a_new_testing_epoch_obj_dict = self.a_new_testing_epoch_obj_dict


        """
        
        from neuropy.analyses.placefields import PfND
        # from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult, SingleEpochDecodedResult
        # from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult

        # 25ms
        # epochs_decoding_time_bin_size: float = 0.050 # 50ms
        # epochs_decoding_time_bin_size: float = 0.250 # 250ms

        skip_training_test_split: bool = (a_new_training_df_dict is None) or (a_new_testing_epoch_obj_dict is None)

        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name].computed_data for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        long_computation_config, short_computation_config, global_computation_config = [curr_active_pipeline.computation_results[an_epoch_name].computation_config for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]

        if pfND_ndim == 1:
            ## Uses 1D Placefields
            print(f'Uses 1D Placefields')
            long_pfND, short_pfND, global_pfND = long_results.pf1D, short_results.pf1D, global_results.pf1D
        else:
            ## Uses 2D Placefields
            print(f'Uses 2D Placefields')
            long_pfND, short_pfND, global_pfND = long_results.pf2D, short_results.pf2D, global_results.pf2D
            # long_pfND_decoder, short_pfND_decoder, global_pfND_decoder = long_results.pf2D_Decoder, short_results.pf2D_Decoder, global_results.pf2D_Decoder


        non_directional_names_to_default_epoch_names_map = dict(zip(['long', 'short', 'global'], [long_epoch_name, short_epoch_name, global_epoch_name]))

        original_pfs_dict: Dict[types.DecoderName, PfND] = {'long': deepcopy(long_pfND), 'short': deepcopy(short_pfND), 'global': deepcopy(global_pfND)} ## Uses ND Placefields

        # t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        # Build an Epoch object containing a single epoch, corresponding to the global epoch for the entire session:
        # single_global_epoch_df: pd.DataFrame = pd.DataFrame({'start': [t_start], 'stop': [t_end], 'label': [0]})
        # single_global_epoch_df['label'] = single_global_epoch_df.index.to_numpy()
        # single_global_epoch: Epoch = Epoch(single_global_epoch_df)

        # single_global_epoch: Epoch = Epoch(self.single_global_epoch_df)

        # # Time-dependent
        # long_pf1D_dt: PfND_TimeDependent = long_results.pf1D_dt
        # long_pf2D_dt: PfND_TimeDependent = long_results.pf2D_dt
        # short_pf1D_dt: PfND_TimeDependent = short_results.pf1D_dt
        # short_pf2D_dt: PfND_TimeDependent = short_results.pf2D_dt
        # global_pf1D_dt: PfND_TimeDependent = global_results.pf1D_dt
        # global_pf2D_dt: PfND_TimeDependent = global_results.pf2D_dt

        # Build new Decoders and Placefields _________________________________________________________________________________ #
        if skip_training_test_split:
            # Non-training, use originals
            # new_decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = {a_name:BasePositionDecoder(pf=a_pfs).replacing_computation_epochs(epochs=deepcopy(curr_active_pipeline.filtered_sessions[non_directional_names_to_default_epoch_names_map[a_name]].non_pbe)) for a_name, a_pfs in original_pfs_dict.items()} ## build new simple decoders
            new_decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = {a_name:BasePositionDecoder(pf=a_pfs).replacing_computation_epochs(epochs=deepcopy(curr_active_pipeline.filtered_sessions[non_directional_names_to_default_epoch_names_map[a_name]].laps)) for a_name, a_pfs in original_pfs_dict.items()} ## build new simple decoders

        else:
            ## extract values:

            ## INPUTS: (a_new_training_df_dict, a_new_testing_epoch_obj_dict), (a_new_test_df_dict, a_new_testing_epoch_obj_dict)
            new_decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = {a_name:BasePositionDecoder(pf=a_pfs).replacing_computation_epochs(epochs=deepcopy(a_new_training_df_dict[a_name])) for a_name, a_pfs in original_pfs_dict.items()} ## build new simple decoders


        new_pfs_dict: Dict[types.DecoderName, PfND] =  {k:deepcopy(a_new_decoder.pf) for k, a_new_decoder in new_decoder_dict.items()}  ## Uses 2D Placefields
        ## OUTPUTS: new_decoder_dict, new_pfs_dict

        ## INPUTS: (a_new_training_df_dict, a_new_testing_epoch_obj_dict), (new_decoder_dict, new_pfs_dict)

        ## Do Decoding of only the test epochs to validate performance
        if skip_training_test_split:
            test_epoch_specific_decoded_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = {}
        else:
            test_epoch_specific_decoded_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = {a_name:a_new_decoder.decode_specific_epochs(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), filter_epochs=deepcopy(a_new_testing_epoch_obj_dict[a_name]), decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False) for a_name, a_new_decoder in new_decoder_dict.items()}

        ## Do Continuous Decoding (for all time (`single_global_epoch`), using the decoder from each epoch) -- slowest dict comp
        continuous_specific_decoded_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = {a_name:a_new_decoder.decode_specific_epochs(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), filter_epochs=deepcopy(single_global_epoch), decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False) for a_name, a_new_decoder in new_decoder_dict.items()}

        return test_epoch_specific_decoded_results_dict, continuous_specific_decoded_results_dict, new_decoder_dict, new_pfs_dict


    @function_attributes(short_name=None, tags=['MAIN', 'compute', 'pure'], input_requires=[], output_provides=[], uses=['cls.build_frame_divided_epochs', 'cls.recompute', 'DecodingResultND'], used_by=['perform_compute_non_PBE_epochs'], creation_date='2025-02-18 09:40', related_items=[])
    @classmethod
    def compute_all(cls, curr_active_pipeline, single_global_epoch: Epoch, epochs_decoding_time_bin_size: float = 0.025, frame_divide_bin_size: float = 0.5, compute_1D: bool = True, compute_2D: bool = True, a_new_training_df_dict: Optional[Dict]=None, a_new_testing_epoch_obj_dict: Optional[Dict[types.DecoderName, Epoch]]=None) -> Tuple[Optional[DecodingResultND], Optional[DecodingResultND]]:
        """ computes all pfs, decoders, and then performs decodings on both continuous and subivided epochs.

        ## OUTPUTS: global_continuous_decoded_epochs_result2D, a_continuous_decoded_result2D, p_x_given_n2D
        # (test_epoch_specific_decoded_results1D_dict, continuous_specific_decoded_results1D_dict, new_decoder1D_dict, new_pf1Ds_dict), frame_divided_epochs_specific_decoded_results1D_dict, ## 1D Results
        # (test_epoch_specific_decoded_results2D_dict, continuous_specific_decoded_results2D_dict, new_decoder2D_dict, new_pf2Ds_dict), frame_divided_epochs_specific_decoded_results2D_dict, global_continuous_decoded_epochs_result2D # 2D results

        single_global_epoch: Epoch = Epoch(self.single_global_epoch_df)

        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import ComputeGlobalEpochBase

            a_new_global_epoch_base_obj: ComputeGlobalEpochBase = ComputeGlobalEpochBase.init_from_pipeline(curr_active_pipeline=curr_active_pipeline)
            results1D, results2D = ComputeGlobalEpochBase.compute_all(curr_active_pipeline, single_global_epoch=a_new_global_epoch_base_obj.single_global_epoch_df, epochs_decoding_time_bin_size=0.025, frame_divide_bin_size=0.50, compute_1D=True, compute_2D=True)

        """
        from neuropy.core.epoch import Epoch, ensure_dataframe, ensure_Epoch
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df

        single_global_epoch = ensure_Epoch(single_global_epoch)

        # Build frame_divided epochs first since they're needed for both 1D and 2D
        (global_frame_divided_epochs_obj, global_frame_divided_epochs_df), global_pos_df = cls.build_frame_divided_epochs(curr_active_pipeline, frame_divide_bin_size=frame_divide_bin_size)

        results1D, results2D = None, None

        if compute_1D:
            test_epoch_specific_decoded_results1D_dict, continuous_specific_decoded_results1D_dict, new_decoder1D_dict, new_pf1Ds_dict = cls.recompute(single_global_epoch = deepcopy(single_global_epoch), curr_active_pipeline=curr_active_pipeline, pfND_ndim=1, epochs_decoding_time_bin_size=epochs_decoding_time_bin_size,
                                                                                                                                                a_new_training_df_dict = a_new_training_df_dict, a_new_testing_epoch_obj_dict = a_new_testing_epoch_obj_dict)
            frame_divided_epochs_specific_decoded_results1D_dict = {a_name:a_new_decoder.decode_specific_epochs(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), filter_epochs=deepcopy(global_frame_divided_epochs_obj), decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False) for a_name, a_new_decoder in new_decoder1D_dict.items()}
            results1D = DecodingResultND(ndim=1, 
                test_epoch_results=test_epoch_specific_decoded_results1D_dict, 
                continuous_results=continuous_specific_decoded_results1D_dict,
                decoders=new_decoder1D_dict, pfs=new_pf1Ds_dict,
                frame_divided_epochs_results=frame_divided_epochs_specific_decoded_results1D_dict, 
                frame_divided_epochs_df=deepcopy(global_frame_divided_epochs_df), pos_df=global_pos_df)

        if compute_2D:
            test_epoch_specific_decoded_results2D_dict, continuous_specific_decoded_results2D_dict, new_decoder2D_dict, new_pf2Ds_dict = cls.recompute(single_global_epoch = deepcopy(single_global_epoch), curr_active_pipeline=curr_active_pipeline, pfND_ndim=2, epochs_decoding_time_bin_size=epochs_decoding_time_bin_size,
                                                                                                                                                a_new_training_df_dict = a_new_training_df_dict, a_new_testing_epoch_obj_dict = a_new_testing_epoch_obj_dict)
            frame_divided_epochs_specific_decoded_results2D_dict = {a_name:a_new_decoder.decode_specific_epochs(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), filter_epochs=deepcopy(global_frame_divided_epochs_obj), decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False) for a_name, a_new_decoder in new_decoder2D_dict.items()}
            results2D = DecodingResultND(ndim=2, 
                test_epoch_results=test_epoch_specific_decoded_results2D_dict,
                continuous_results=continuous_specific_decoded_results2D_dict,
                decoders=new_decoder2D_dict, pfs=new_pf2Ds_dict,
                frame_divided_epochs_results=frame_divided_epochs_specific_decoded_results2D_dict, 
                frame_divided_epochs_df=deepcopy(global_frame_divided_epochs_df), pos_df=global_pos_df)

        return results1D, results2D


    @classmethod
    @function_attributes(short_name=None, tags=['frame_division', 'pure'], input_requires=[], output_provides=[], uses=['subdivide_epochs'], used_by=['cls.compute_all(...)'], creation_date='2025-02-11 00:00', related_items=[])
    def build_frame_divided_epochs(cls, curr_active_pipeline, frame_divide_bin_size: Optional[float] = 1.0):
        """ Splits the epochs into fixed-size "frames"
        
        frame_divide_bin_size = 1.0 # Specify the size of each sub-epoch in seconds

        Usage:

            from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import build_frame_divided_epochs

            frame_divide_bin_size: float = 1.0
            (global_frame_divided_epochs_obj, global_frame_divided_epochs_df), global_pos_df = Compute_NonPBE_Epochs.build_frame_divided_epochs(curr_active_pipeline, frame_divide_bin_size=frame_divide_bin_size)
            ## Do Decoding of only the test epochs to validate performance
            frame_divided_epochs_specific_decoded_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = {a_name:a_new_decoder.decode_specific_epochs(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), filter_epochs=deepcopy(global_frame_divided_epochs_obj), decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False) for a_name, a_new_decoder in new_decoder_dict.items()}


        """
        ## OUTPUTS: test_epoch_specific_decoded_results_dict, continuous_specific_decoded_results_dict

        ## INPUTS: new_decoder_dict
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        # Build an Epoch object containing a single epoch, corresponding to the global epoch for the entire session:
        single_global_epoch_df: pd.DataFrame = pd.DataFrame({'start': [t_start], 'stop': [t_end], 'label': [0]})
        # single_global_epoch_df['label'] = single_global_epoch_df.index.to_numpy()
        single_global_epoch: Epoch = Epoch(single_global_epoch_df)

        ## Common:
        global_session = curr_active_pipeline.filtered_sessions[global_epoch_name]
        global_pos_obj: "Position" = deepcopy(global_session.position)
        global_pos_df: pd.DataFrame = global_pos_obj.compute_higher_order_derivatives().position.compute_smoothed_position_info(N=15)
        ## OUTPUTS: global_pos_df, global_pos_df
        

        df: pd.DataFrame = ensure_dataframe(deepcopy(single_global_epoch)) 
        df['maze_name'] = 'global'
        # df['interval_type_id'] = 666
        
        if frame_divide_bin_size is not None:
            frame_divided_df: pd.DataFrame = subdivide_epochs(df, frame_divide_bin_size)
            frame_divided_df['label'] = deepcopy(frame_divided_df.index.to_numpy())
            frame_divided_df['stop'] = frame_divided_df['stop'] - 1e-12
            global_frame_divided_epochs_obj = ensure_Epoch(frame_divided_df)

            # ## Do Decoding of only the test epochs to validate performance
            # frame_divided_epochs_specific_decoded_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = {a_name:a_new_decoder.decode_specific_epochs(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), filter_epochs=deepcopy(global_frame_divided_epochs_obj), decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False) for a_name, a_new_decoder in new_decoder_dict.items()}

            ## OUTPUTS: frame_divided_epochs_specific_decoded_results_dict
            # takes 4min 30 sec to run

            ## Adds the 'global_frame_division_idx' column to 'global_pos_df' so it can get the measured positions by plotting
            # INPUTS: global_frame_divided_epochs_obj, original_pos_dfs_dict
            # global_frame_divided_epochs_obj
            
            global_frame_divided_epochs_df = global_frame_divided_epochs_obj.epochs.to_dataframe() #.rename(columns={'t_rel_seconds':'t'})
            global_frame_divided_epochs_df['label'] = deepcopy(global_frame_divided_epochs_df.index.to_numpy())
            # global_pos_df: pd.DataFrame = deepcopy(global_session.position.to_dataframe()) #.rename(columns={'t':'t_rel_seconds'})

            ## Extract Measured Position:
            ## INPUTS: global_pos_df
            global_pos_df.time_point_event.adding_epochs_identity_column(epochs_df=global_frame_divided_epochs_df, epoch_id_key_name='global_frame_division_idx', epoch_label_column_name='label', drop_non_epoch_events=True, should_replace_existing_column=True) # , override_time_variable_name='t_rel_seconds'

            ## Adds the ['frame_division_epoch_start_t'] columns to `stacked_flat_global_pos_df` so we can figure out the appropriate offsets
            frame_divided_epochs_properties_df: pd.DataFrame = deepcopy(global_frame_divided_epochs_df)
            frame_divided_epochs_properties_df['global_frame_division_idx'] = deepcopy(frame_divided_epochs_properties_df.index) ## add explicit 'global_frame_division_idx' column
            frame_divided_epochs_properties_df = frame_divided_epochs_properties_df.rename(columns={'start': 'frame_division_epoch_start_t'})[['global_frame_division_idx', 'frame_division_epoch_start_t']]
            global_pos_df = PandasHelpers.add_explicit_dataframe_columns_from_lookup_df(global_pos_df, frame_divided_epochs_properties_df, join_column_name='global_frame_division_idx')


        else:
            ## no frame division needed
            raise NotImplementedError(f'#TODO 2025-06-30 14:49: - [ ] Not yet implemented, always subidivdes at least once')
            frame_divided_df: pd.DataFrame = deepcopy(df)
            frame_divided_df['label'] = deepcopy(frame_divided_df.index.to_numpy())
            global_frame_divided_epochs_obj = ensure_Epoch(frame_divided_df)
            global_frame_divided_epochs_df = global_frame_divided_epochs_obj.epochs.to_dataframe() #.rename(columns={'t_rel_seconds':'t'})
            global_frame_divided_epochs_df['label'] = deepcopy(global_frame_divided_epochs_df.index.to_numpy())
            
        ## END if
        

        global_pos_df.sort_values(by=['t'], inplace=True) # Need to re-sort by timestamps once done

        if global_pos_df.attrs is None:
            global_pos_df.attrs = {}

        global_pos_df.attrs.update({'frame_divide_bin_size': frame_divide_bin_size})

        if global_frame_divided_epochs_df.attrs is None:
            global_frame_divided_epochs_df.attrs = {}

        global_frame_divided_epochs_df.attrs.update({'frame_divide_bin_size': frame_divide_bin_size})

        return (global_frame_divided_epochs_obj, global_frame_divided_epochs_df), global_pos_df


    # For serialization/pickling: ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # # Remove the unpicklable entries.
        # _non_pickled_fields = ['curr_active_pipeline', 'track_templates']
        # for a_non_pickleable_field in _non_pickled_fields:
        #     del state[a_non_pickleable_field]
        return state


    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)

        # _non_pickled_field_restore_defaults = dict(zip(['curr_active_pipeline', 'track_templates'], [None, None]))
        # for a_field_name, a_default_restore_value in _non_pickled_field_restore_defaults.items():
        #     if a_field_name not in state:
        #         state[a_field_name] = a_default_restore_value

        self.__dict__.update(state)
        # # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(WCorrShuffle, self).__init__() # from

    def __repr__(self):
        """ 2024-01-11 - Renders only the fields and their sizes
        """
        from pyphocorehelpers.print_helpers import strip_type_str_to_classname
        attr_reprs = []
        for a in self.__attrs_attrs__:
            attr_type = strip_type_str_to_classname(type(getattr(self, a.name)))
            if 'shape' in a.metadata:
                shape = ', '.join(a.metadata['shape'])  # this joins tuple elements with a comma, creating a string without quotes
                attr_reprs.append(f"{a.name}: {attr_type} | shape ({shape})")  # enclose the shape string with parentheses
            else:
                attr_reprs.append(f"{a.name}: {attr_type}")
        content = ",\n\t".join(attr_reprs)
        return f"{type(self).__name__}({content}\n)"



    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)




@metadata_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-06-30 07:41', related_items=[])
@custom_define(slots=False, eq=False)
class DecodingResultND(UnpackableMixin, ComputedResult):
    """Contains all decoding results for either 1D or 2D computations
    
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import DecodingResultND
    
    results2D: DecodingResultND = DecodingResultND(ndim=2, 
    test_epoch_results=test_epoch_specific_decoded_results2D_dict,
    continuous_results=continuous_specific_decoded_results2D_dict,
    decoders=new_decoder2D_dict, pfs=new_pf2Ds_dict,
    frame_divided_epochs_results=frame_divided_epochs_specific_decoded_results2D_dict, 
    frame_divided_epochs_df=deepcopy(global_frame_divided_epochs_df), pos_df=global_pos_df)

    # results2D
    
    # Unpack all fields in order
    ndim, pos_df, pfs, decoders, test_epoch_results, continuous_results, frame_divided_epochs_df, frame_divided_epochs_results = results2D
    # *test_args = results2D
    # print(len(test_args))
    # anUPDATED_TUPLE_2D, UPDATED_frame_divided_epochs_specific_decoded_results2D_dict = results2D
    ndim

    History:
        2025-06-30 08:24 "NonPBEDimensionalDecodingResult" -> "DecodingResultND"


    """
    _VersionedResultMixin_version: str = "2025.06.30_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    ndim: int = serialized_attribute_field()  # 1 or 2
    pos_df: pd.DataFrame = serialized_field()
    
    pfs: Dict[types.DecoderName, PfND] = serialized_field()
    decoders: Dict[types.DecoderName, BasePositionDecoder] = serialized_field()

    test_epoch_results: Dict[types.DecoderName, DecodedFilterEpochsResult] = serialized_field()
    continuous_results: Dict[types.DecoderName, DecodedFilterEpochsResult] = serialized_field()
    
    frame_divided_epochs_df: pd.DataFrame = serialized_field(metadata={'desc': 'used for rendering a series of successive 2D decoded posteriors as "frames" on a 1D timeline.'})
    frame_divided_epochs_results: Dict[types.DecoderName, DecodedFilterEpochsResult] = serialized_field(metadata={'desc': 'used for rendering a series of successive 2D decoded posteriors as "frames" on a 1D timeline.'})

    @property
    def a_result2D(self) -> DecodedFilterEpochsResult:
        return self.frame_divided_epochs_results['global']

    @property
    def a_new_global2D_decoder(self) -> BasePositionDecoder:
        return self.decoders['global']

    def __attrs_post_init__(self):
        assert self.ndim in (1, 2), f"ndim must be 1 or 2, got {self.ndim}"
        

    def add_frame_division_epoch_start_t_to_pos_df(self):
        ## Adds the ['frame_division_epoch_start_t'] columns to `stacked_flat_global_pos_df` so we can figure out the appropriate offsets
        pos_df: pd.DataFrame = deepcopy(self.pos_df)
        frame_divided_epochs_df: pd.DataFrame = deepcopy(self.frame_divided_epochs_df)
        frame_divided_epochs_df['global_frame_division_idx'] = deepcopy(frame_divided_epochs_df.index)
        frame_divided_epochs_df = frame_divided_epochs_df.rename(columns={'start': 'frame_division_epoch_start_t'})[['global_frame_division_idx', 'frame_division_epoch_start_t']]
        pos_df = PandasHelpers.add_explicit_dataframe_columns_from_lookup_df(pos_df, frame_divided_epochs_df, join_column_name='global_frame_division_idx')
        pos_df.sort_values(by=['t'], inplace=True) # Need to re-sort by timestamps once done
        self.pos_df = pos_df
        return self.pos_df


    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # # Remove the unpicklable entries.
        # _non_pickled_fields = ['curr_active_pipeline', 'track_templates']
        # for a_non_pickleable_field in _non_pickled_fields:
        #     del state[a_non_pickleable_field]
        return state


    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)
        
        # _non_pickled_field_restore_defaults = dict(zip(['curr_active_pipeline', 'track_templates'], [None, None]))
        # for a_field_name, a_default_restore_value in _non_pickled_field_restore_defaults.items():
        #     if a_field_name not in state:
        #         state[a_field_name] = a_default_restore_value

        self.__dict__.update(state)
        # # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(WCorrShuffle, self).__init__() # from


    def __repr__(self):
        """ 2024-01-11 - Renders only the fields and their sizes
        """
        from pyphocorehelpers.print_helpers import strip_type_str_to_classname
        attr_reprs = []
        for a in self.__attrs_attrs__:
            attr_type = strip_type_str_to_classname(type(getattr(self, a.name)))
            if 'shape' in a.metadata:
                shape = ', '.join(a.metadata['shape'])  # this joins tuple elements with a comma, creating a string without quotes
                attr_reprs.append(f"{a.name}: {attr_type} | shape ({shape})")  # enclose the shape string with parentheses
            else:
                attr_reprs.append(f"{a.name}: {attr_type}")
        content = ",\n\t".join(attr_reprs)
        return f"{type(self).__name__}({content}\n)"


    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)
        
    # ==================================================================================================================== #
    # UnpackableMixin conformances                                                                                         #
    # ==================================================================================================================== #
    # def UnpackableMixin_unpacking_excludes(self) -> List:
    #     """Excludes ndim from unpacking"""
    #     return ['ndim']
    
@metadata_attributes(short_name=None, tags=['non-pbe', 'compute', 'result'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-01 00:00', related_items=['DecodingResultND'])
@custom_define(slots=False, eq=False, repr=False)
class Compute_NonPBE_Epochs(ComputedResult):
    """ Relates to using all time on the track except for detected PBEs as the placefield inputs. This includes the laps and the intra-lap times. 
    Importantly `lap_dir` is poorly defined for the periods between the laps, so something like head-direction might have to be used.
    
    #TODO 2025-02-13 12:40: - [ ] Should compute the correct Epochs, add it to the sessions as a new Epoch (I guess as a new FilteredSession context!! 
    
    
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import Compute_NonPBE_Epochs
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import Compute_NonPBE_Epochs
    
    """
    _VersionedResultMixin_version: str = "2024.02.18_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    single_global_epoch_df: pd.DataFrame = serialized_field()
    global_epoch_only_non_PBE_epoch_df: pd.DataFrame = serialized_field()
    # a_new_global_training_df: pd.DataFrame = serialized_field()
    # a_new_global_test_df: pd.DataFrame = serialized_field()

    skip_training_test_split: bool = serialized_attribute_field(default=False)

    a_new_training_df_dict: Dict[types.DecoderName, pd.DataFrame] = serialized_field()
    a_new_test_df_dict: Dict[types.DecoderName, pd.DataFrame] = serialized_field()
    
    a_new_training_epoch_obj_dict: Dict[types.DecoderName, Epoch] = serialized_field(init=False)
    a_new_testing_epoch_obj_dict: Dict[types.DecoderName, Epoch] = serialized_field(init=False)
    

    # frame_divide_bin_size: float = 0.5
    @property
    def frame_divide_bin_size(self) -> float:
        """The frame_divide_bin_size property."""
        return self.pos_df.attrs['frame_divide_bin_size']
    @frame_divide_bin_size.setter
    def frame_divide_bin_size(self, value):
        self.pos_df.attrs['frame_divide_bin_size'] = value



    @classmethod
    def _compute_non_PBE_epochs_from_sess(cls, sess, track_identity: Optional[str]=None, interval_datasource_name: Optional[str]=None, **additional_df_metdata) -> pd.DataFrame:
        """ Builds a dictionary of train/test-split epochs for ['long', 'short', 'global'] periods
        
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _adding_global_non_PBE_epochs
        
        a_new_training_df, a_new_test_df, a_new_training_df_dict, a_new_test_df_dict = Compute_NonPBE_Epochs._compute_non_PBE_epochs_from_sess(sess=long_session)
        a_new_training_df
        a_new_test_df

            
        sess.pbe
        sess.epochs
        curr_active_pipeline.find_LongShortDelta_times()
        
        Usage:
        
            t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
            , t_start: float, t_delta: float, t_end: float
        
        """                
        PBE_df: pd.DataFrame = ensure_dataframe(deepcopy(sess.pbe))
        ## Build up a new epoch -- this works successfully for filter epochs as well, although 'maze' label is incorrect
        epochs_df: pd.DataFrame = deepcopy(sess.epochs).epochs.adding_global_epoch_row()
        global_epoch_only_df: pd.DataFrame = epochs_df.epochs.label_slice('maze')

        # t_start, t_stop = epochs_df.epochs.t_start, epochs_df.epochs.t_stop
        global_epoch_only_non_PBE_epoch_df: pd.DataFrame = global_epoch_only_df.epochs.subtracting(PBE_df)
        global_epoch_only_non_PBE_epoch_df = global_epoch_only_non_PBE_epoch_df.epochs.modify_each_epoch_by(additive_factor=-0.008, final_output_minimum_epoch_duration=0.040)

        # 'global'
        # f'global_NonPBE_TRAIN'

        df_metadata = {}
        if track_identity is not None:
            df_metadata['track_identity'] = track_identity
        if interval_datasource_name is not None:
            df_metadata['interval_datasource_name'] = interval_datasource_name
            
        df_metadata.update(**additional_df_metdata)        
        # for a_df_metadata_key, a_v in additional_df_metdata.items():
            

        ## Add the metadata:
        if len(df_metadata) > 0:
            global_epoch_only_non_PBE_epoch_df = global_epoch_only_non_PBE_epoch_df.epochs.adding_or_updating_metadata(**df_metadata)
        
        ## Add the maze_id column to the epochs:
        
        # a_new_global_training_df = a_new_global_training_df.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
        # a_new_global_test_df = a_new_global_test_df.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)

        # maze_id_to_maze_name_map = {-1:'none', 0:'long', 1:'short'}
        # a_new_global_training_df['maze_name'] = a_new_global_training_df['maze_id'].map(maze_id_to_maze_name_map)
        # a_new_global_test_df['maze_name'] = a_new_global_test_df['maze_id'].map(maze_id_to_maze_name_map)
        return global_epoch_only_non_PBE_epoch_df
    



    @function_attributes(short_name=None, tags=['epochs', 'non-PBE', 'session', 'metadata'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-28 04:10', related_items=[]) 
    @classmethod
    def _adding_global_non_PBE_epochs_to_sess(cls, sess, t_start: float, t_delta: float, t_end: float, training_data_portion: float = 5.0/6.0) -> Tuple[Dict[types.DecoderName, pd.DataFrame], Dict[types.DecoderName, pd.DataFrame]]:
        """ Builds a dictionary of train/test-split epochs for ['long', 'short', 'global'] periods
        
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _adding_global_non_PBE_epochs
        
        a_new_training_df, a_new_test_df, a_new_training_df_dict, a_new_test_df_dict = _adding_global_non_PBE_epochs(curr_active_pipeline)
        a_new_training_df
        a_new_test_df

            
        curr_active_pipeline.sess.pbe
        curr_active_pipeline.sess.epochs
        curr_active_pipeline.find_LongShortDelta_times()
        
        Usage:
        
            t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
            
        
        """
        maze_id_to_maze_name_map = {-1:'none', 0:'long', 1:'short'}
        epoch_overlap_prevention_kwargs = dict(additive_factor=-0.008, final_output_minimum_epoch_duration=0.040) # passed to `*df.epochs.modify_each_epoch_by(...)`
        
        PBE_df: pd.DataFrame = ensure_dataframe(deepcopy(sess.pbe))
        laps_df = ensure_dataframe(deepcopy(sess.laps))
        

        ## Build up a new epoch
        epochs_df: pd.DataFrame = deepcopy(sess.epochs).epochs.adding_global_epoch_row()
        global_epoch_only_df: pd.DataFrame = epochs_df.epochs.label_slice('maze')
        
        # t_start, t_stop = epochs_df.epochs.t_start, epochs_df.epochs.t_stop
        global_epoch_only_non_PBE_epoch_df: pd.DataFrame = global_epoch_only_df.epochs.subtracting(PBE_df)
        global_epoch_only_non_PBE_epoch_df= global_epoch_only_non_PBE_epoch_df.epochs.modify_each_epoch_by(**epoch_overlap_prevention_kwargs)
        
        ## endcap-only (non_PBE and non_lap/running) epochs:
        a_new_global_epoch_only_non_pbe_endcaps_df: pd.DataFrame = deepcopy(global_epoch_only_non_PBE_epoch_df).epochs.subtracting(laps_df)
        a_new_global_epoch_only_non_pbe_endcaps_df = a_new_global_epoch_only_non_pbe_endcaps_df.epochs.modify_each_epoch_by(**epoch_overlap_prevention_kwargs) # minimum length to consider is 50ms, contract each epoch inward by -8ms (4ms on each side)
        a_new_global_epoch_only_non_pbe_endcaps_df = a_new_global_epoch_only_non_pbe_endcaps_df.epochs.adding_or_updating_metadata(track_identity='global', interval_datasource_name=f'global_EndcapsNonPBE') # train_test_period='train', training_data_portion=training_data_portion, 
        a_new_global_epoch_only_non_pbe_endcaps_df = a_new_global_epoch_only_non_pbe_endcaps_df.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
        a_new_global_epoch_only_non_pbe_endcaps_df['maze_name'] = a_new_global_epoch_only_non_pbe_endcaps_df['maze_id'].map(maze_id_to_maze_name_map)


        # ==================================================================================================================== #
        # training/test Split                                                                                                  #
        # ==================================================================================================================== #

        ## this training/test isn't required:
        a_new_global_training_df, a_new_global_test_df = global_epoch_only_non_PBE_epoch_df.epochs.split_into_training_and_test(training_data_portion=training_data_portion, group_column_name ='label', additional_epoch_identity_column_names=['label'], skip_get_non_overlapping=False, debug_print=False) # a_laps_training_df, a_laps_test_df both comeback good here.
        ## Drop test epochs that are too short:
        a_new_global_test_df = a_new_global_test_df.epochs.modify_each_epoch_by(final_output_minimum_epoch_duration=0.100) # 100ms minimum test epochs

        ## Add the metadata:
        a_new_global_training_df = a_new_global_training_df.epochs.adding_or_updating_metadata(track_identity='global', train_test_period='train', training_data_portion=training_data_portion, interval_datasource_name=f'global_NonPBE_TRAIN')
        a_new_global_test_df = a_new_global_test_df.epochs.adding_or_updating_metadata(track_identity='global', train_test_period='test', training_data_portion=training_data_portion, interval_datasource_name=f'global_NonPBE_TEST')
        
        ## Add the maze_id column to the epochs:
        
        a_new_global_training_df = a_new_global_training_df.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
        a_new_global_test_df = a_new_global_test_df.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)

        
        a_new_global_training_df['maze_name'] = a_new_global_training_df['maze_id'].map(maze_id_to_maze_name_map)
        a_new_global_test_df['maze_name'] = a_new_global_test_df['maze_id'].map(maze_id_to_maze_name_map)

        # ==================================================================================================================== #
        # Splits the global epochs into the long/short epochs                                                                  #
        # ==================================================================================================================== #
        # partitionColumn: str ='maze_id'
        partitionColumn: str ='maze_name'
        ## INPUTS: a_new_test_df, a_new_training_df, modern_names_list
        a_new_test_df_dict = partition_df_dict(a_new_global_test_df, partitionColumn=partitionColumn)
        # a_new_test_df_dict = dict(zip(modern_names_list, list(a_new_test_df_dict.values())))
        a_new_training_df_dict = partition_df_dict(a_new_global_training_df, partitionColumn=partitionColumn)
        # a_new_training_df_dict = dict(zip(modern_names_list, list(a_new_training_df_dict.values())))

        ## add back in 'global' epoch
        a_new_test_df_dict['global'] = deepcopy(a_new_global_test_df)
        a_new_training_df_dict['global'] = deepcopy(a_new_global_test_df)
        
        # ==================================================================================================================== #
        # Set Metadata                                                                                                         #
        # ==================================================================================================================== #
        a_new_test_df_dict: Dict[types.DecoderName, pd.DataFrame] = {k:v.epochs.adding_or_updating_metadata(track_identity=k, train_test_period='test', training_data_portion=training_data_portion, interval_datasource_name=f'{k}_NonPBE_TEST') for k, v in a_new_test_df_dict.items() if k != 'none'}
        a_new_training_df_dict: Dict[types.DecoderName, pd.DataFrame] = {k:v.epochs.adding_or_updating_metadata(track_identity=k, train_test_period='train', training_data_portion=training_data_portion, interval_datasource_name=f'{k}_NonPBE_TRAIN') for k, v in a_new_training_df_dict.items() if k != 'none'}

        ## OUTPUTS: new_decoder_dict, new_decoder_dict, new_decoder_dict, a_new_training_df_dict, a_new_test_df_dict

        ## OUTPUTS: training_data_portion, a_new_training_df, a_new_test_df, a_new_training_df_dict, a_new_test_df_dict
        return a_new_training_df_dict, a_new_test_df_dict, (global_epoch_only_non_PBE_epoch_df, a_new_global_training_df, a_new_global_test_df)



    @function_attributes(short_name=None, tags=['epochs', 'non-PBE', 'pipeline'], input_requires=[], output_provides=[], uses=['cls._adding_global_non_PBE_epochs_to_sess(...)'], used_by=[], creation_date='2025-01-28 04:10', related_items=[]) 
    @classmethod
    def _adding_global_non_PBE_epochs(cls, curr_active_pipeline, training_data_portion: float = 5.0/6.0) -> Tuple[Dict[types.DecoderName, pd.DataFrame], Dict[types.DecoderName, pd.DataFrame]]:
        """ Builds a dictionary of train/test-split epochs for ['long', 'short', 'global'] periods
        
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _adding_global_non_PBE_epochs
        
        a_new_training_df, a_new_test_df, a_new_training_df_dict, a_new_test_df_dict = _adding_global_non_PBE_epochs(curr_active_pipeline)
        a_new_training_df
        a_new_test_df

            
        curr_active_pipeline.sess.pbe
        curr_active_pipeline.sess.epochs
        curr_active_pipeline.find_LongShortDelta_times()
        
        """
        # from neuropy.core.epoch import EpochsAccessor, Epoch, ensure_dataframe
        # # import portion as P # Required for interval search: portion~=2.3.0
        # from pyphocorehelpers.indexing_helpers import partition_df_dict, partition_df
                
        # PBE_df: pd.DataFrame = ensure_dataframe(deepcopy(curr_active_pipeline.sess.pbe))
        # ## Build up a new epoch
        # epochs_df: pd.DataFrame = deepcopy(curr_active_pipeline.sess.epochs).epochs.adding_global_epoch_row()
        # global_epoch_only_df: pd.DataFrame = epochs_df.epochs.label_slice('maze')
        
        # # t_start, t_stop = epochs_df.epochs.t_start, epochs_df.epochs.t_stop
        # global_epoch_only_non_PBE_epoch_df: pd.DataFrame = global_epoch_only_df.epochs.subtracting(PBE_df)
        # global_epoch_only_non_PBE_epoch_df= global_epoch_only_non_PBE_epoch_df.epochs.modify_each_epoch_by(additive_factor=-0.008, final_output_minimum_epoch_duration=0.040)
        
        # a_new_global_training_df, a_new_global_test_df = global_epoch_only_non_PBE_epoch_df.epochs.split_into_training_and_test(training_data_portion=training_data_portion, group_column_name ='label', additional_epoch_identity_column_names=['label'], skip_get_non_overlapping=False, debug_print=False) # a_laps_training_df, a_laps_test_df both comeback good here.
        # ## Drop test epochs that are too short:
        # a_new_global_test_df = a_new_global_test_df.epochs.modify_each_epoch_by(final_output_minimum_epoch_duration=0.100) # 100ms minimum test epochs

        # ## Add the metadata:
        # a_new_global_training_df = a_new_global_training_df.epochs.adding_or_updating_metadata(track_identity='global', train_test_period='train', training_data_portion=training_data_portion, interval_datasource_name=f'global_NonPBE_TRAIN')
        # a_new_global_test_df = a_new_global_test_df.epochs.adding_or_updating_metadata(track_identity='global', train_test_period='test', training_data_portion=training_data_portion, interval_datasource_name=f'global_NonPBE_TEST')
        
        # ## Add the maze_id column to the epochs:
        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        # a_new_global_training_df = a_new_global_training_df.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)
        # a_new_global_test_df = a_new_global_test_df.epochs.adding_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end)

        # maze_id_to_maze_name_map = {-1:'none', 0:'long', 1:'short'}
        # a_new_global_training_df['maze_name'] = a_new_global_training_df['maze_id'].map(maze_id_to_maze_name_map)
        # a_new_global_test_df['maze_name'] = a_new_global_test_df['maze_id'].map(maze_id_to_maze_name_map)

        # # ==================================================================================================================== #
        # # Splits the global epochs into the long/short epochs                                                                  #
        # # ==================================================================================================================== #
        # # partitionColumn: str ='maze_id'
        # partitionColumn: str ='maze_name'
        # ## INPUTS: a_new_test_df, a_new_training_df, modern_names_list
        # a_new_test_df_dict = partition_df_dict(a_new_global_test_df, partitionColumn=partitionColumn)
        # # a_new_test_df_dict = dict(zip(modern_names_list, list(a_new_test_df_dict.values())))
        # a_new_training_df_dict = partition_df_dict(a_new_global_training_df, partitionColumn=partitionColumn)
        # # a_new_training_df_dict = dict(zip(modern_names_list, list(a_new_training_df_dict.values())))

        # ## add back in 'global' epoch
        # a_new_test_df_dict['global'] = deepcopy(a_new_global_test_df)
        # a_new_training_df_dict['global'] = deepcopy(a_new_global_test_df)
        
        # # ==================================================================================================================== #
        # # Set Metadata                                                                                                         #
        # # ==================================================================================================================== #
        # a_new_test_df_dict: Dict[types.DecoderName, pd.DataFrame] = {k:v.epochs.adding_or_updating_metadata(track_identity=k, train_test_period='test', training_data_portion=training_data_portion, interval_datasource_name=f'{k}_NonPBE_TEST') for k, v in a_new_test_df_dict.items() if k != 'none'}
        # a_new_training_df_dict: Dict[types.DecoderName, pd.DataFrame] = {k:v.epochs.adding_or_updating_metadata(track_identity=k, train_test_period='train', training_data_portion=training_data_portion, interval_datasource_name=f'{k}_NonPBE_TRAIN') for k, v in a_new_training_df_dict.items() if k != 'none'}

        # ## OUTPUTS: new_decoder_dict, new_decoder_dict, new_decoder_dict, a_new_training_df_dict, a_new_test_df_dict

        ## OUTPUTS: training_data_portion, a_new_training_df, a_new_test_df, a_new_training_df_dict, a_new_test_df_dict
        # return a_new_training_df_dict, a_new_test_df_dict, (global_epoch_only_non_PBE_epoch_df, a_new_global_training_df, a_new_global_test_df)

        return cls._adding_global_non_PBE_epochs_to_sess(sess=curr_active_pipeline.sess, t_start=t_start, t_delta=t_delta, t_end=t_end, training_data_portion=training_data_portion)
    

    @function_attributes(short_name=None, tags=['UNFINISHED', 'UNUSED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-20 09:32', related_items=[])
    @classmethod
    def _build_filtered_sessions_for_PBE_epochs(cls, curr_active_pipeline):
        """ builds 3 new filtered sessions ('long_nonPBE', 'short_nonPBE', 'global_nonPBE') and adds them to curr_active_pipeline.filtered_sessions
        Note that running-direction (LR v. RL) doesn't make sense for this.
        """
        pass
        
    

    @classmethod
    def init_from_pipeline(cls, curr_active_pipeline, training_data_portion: float = 5.0/6.0, skip_training_test_split: bool=True):
        a_new_training_df_dict, a_new_test_df_dict, (global_epoch_only_non_PBE_epoch_df, a_new_global_training_df, a_new_global_test_df) = cls._adding_global_non_PBE_epochs(curr_active_pipeline, training_data_portion=training_data_portion)
        
        t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
        # Build an Epoch object containing a single epoch, corresponding to the global epoch for the entire session:
        single_global_epoch_df: pd.DataFrame = pd.DataFrame({'start': [t_start], 'stop': [t_end], 'label': [0]})
        # single_global_epoch_df['label'] = single_global_epoch_df.index.to_numpy()
        # single_global_epoch: Epoch = Epoch(self.single_global_epoch_df)
        
        _obj = cls(single_global_epoch_df=single_global_epoch_df, global_epoch_only_non_PBE_epoch_df=global_epoch_only_non_PBE_epoch_df, a_new_training_df_dict=a_new_training_df_dict, a_new_test_df_dict=a_new_test_df_dict, skip_training_test_split=skip_training_test_split)
        return _obj

    def __attrs_post_init__(self):
        # Add post-init logic here
        self.a_new_training_epoch_obj_dict = {k:Epoch(deepcopy(v)).get_non_overlapping() for k, v in self.a_new_training_df_dict.items()}
        self.a_new_testing_epoch_obj_dict = {k:Epoch(deepcopy(v)).get_non_overlapping() for k, v in self.a_new_test_df_dict.items()}


        # curr_active_pipeline.filtered_sessions[global_any_name].non_PBE_epochs
        pass
    


    @function_attributes(short_name=None, tags=['MAIN', 'compute'], input_requires=[], output_provides=[], uses=['ComputeGlobalEpochBase.compute_all(...)'], used_by=['perform_compute_non_PBE_epochs'], creation_date='2025-02-18 09:40', related_items=[])
    def compute_all(self, curr_active_pipeline, epochs_decoding_time_bin_size: float = 0.025, frame_divide_bin_size: float = 0.5, compute_1D: bool = True, compute_2D: bool = True, skip_training_test_split: Optional[bool] = None) -> Tuple[Optional[DecodingResultND], Optional[DecodingResultND]]:
        """ computes all pfs, decoders, and then performs decodings on both continuous and subivided epochs.
        
        ## OUTPUTS: global_continuous_decoded_epochs_result2D, a_continuous_decoded_result2D, p_x_given_n2D
        # (test_epoch_specific_decoded_results1D_dict, continuous_specific_decoded_results1D_dict, new_decoder1D_dict, new_pf1Ds_dict), frame_divided_epochs_specific_decoded_results1D_dict, ## 1D Results
        # (test_epoch_specific_decoded_results2D_dict, continuous_specific_decoded_results2D_dict, new_decoder2D_dict, new_pf2Ds_dict), frame_divided_epochs_specific_decoded_results2D_dict, global_continuous_decoded_epochs_result2D # 2D results
        
        Updates:
             self.skip_training_test_split


        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import Compute_NonPBE_Epochs

            long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
            a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = Compute_NonPBE_Epochs.init_from_pipeline(curr_active_pipeline=curr_active_pipeline)
            ## apply the new epochs to the session:
            curr_active_pipeline.filtered_sessions[global_epoch_name].non_PBE_epochs = deepcopy(a_new_NonPBE_Epochs_obj.global_epoch_only_non_PBE_epoch_df)

            results1D, results2D = a_new_NonPBE_Epochs_obj.compute_all(curr_active_pipeline, epochs_decoding_time_bin_size=0.025, frame_divide_bin_size=0.50, compute_1D=True, compute_2D=True)
        
        """
        if skip_training_test_split is not None:
            self.skip_training_test_split = skip_training_test_split ## override existing
        else:
            skip_training_test_split = self.skip_training_test_split

        if skip_training_test_split:
            a_new_training_df_dict = None
            a_new_testing_epoch_obj_dict = None
        else:
            a_new_training_df_dict = self.a_new_training_df_dict
            a_new_testing_epoch_obj_dict = self.a_new_testing_epoch_obj_dict


        single_global_epoch: Epoch = Epoch(self.single_global_epoch_df)

        return ComputeGlobalEpochBase.compute_all(curr_active_pipeline=curr_active_pipeline, single_global_epoch = deepcopy(single_global_epoch),
                                                                    epochs_decoding_time_bin_size=epochs_decoding_time_bin_size, frame_divide_bin_size=frame_divide_bin_size, compute_1D=compute_1D, compute_2D=compute_2D,
                                                                    a_new_training_df_dict = a_new_training_df_dict, a_new_testing_epoch_obj_dict = a_new_testing_epoch_obj_dict,
                                                 )


    @classmethod
    @function_attributes(short_name=None, tags=['non_PBE', 'non_PBE_Endcaps', 'epochs', 'update', 'pipeline'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-18 18:49', related_items=[])
    def update_session_non_pbe_epochs(cls, sess, filtered_sessions=None, save_on_compute=True) -> Tuple[bool, Any, Any]:
        """Updates non_PBE epochs for both main session and filtered sessions and tracks changes

        Usage: 
        
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import Compute_NonPBE_Epochs

            did_any_non_pbe_epochs_change, curr_active_pipeline.stage.sess, curr_active_pipeline.stage.filtered_sessions = Compute_NonPBE_Epochs.update_session_non_pbe_epochs(curr_active_pipeline.sess, filtered_sessions=curr_active_pipeline.filtered_sessions)

        """
        from neuropy.core.session.dataSession import DataSession
        from neuropy.core.epoch import Epoch
        
        did_change = False
        
        # Backup original non_pbe epochs
        original_non_pbe = deepcopy(getattr(sess, 'non_pbe', None))
        original_non_pbe_endcaps = deepcopy(getattr(sess, 'non_pbe_endcaps', None))
        
        # Update main session
        sess.non_pbe = DataSession.compute_non_PBE_epochs(sess, save_on_compute=save_on_compute)
        sess.non_pbe_endcaps = DataSession.compute_non_PBE_EndcapsOnly_epochs(sess, save_on_compute=save_on_compute)
        
        # Check if main session changed - compare the dataframes directly
        did_change = did_change or (original_non_pbe is None) or (not original_non_pbe.to_dataframe().equals(sess.non_pbe.to_dataframe()))
        did_change = did_change or (original_non_pbe_endcaps is None) or (not original_non_pbe_endcaps.to_dataframe().equals(sess.non_pbe_endcaps.to_dataframe()))
        
        # Update filtered sessions if provided
        if filtered_sessions is not None:
            for filter_name, filtered_session in filtered_sessions.items():
                original_filtered_non_pbe = deepcopy(getattr(filtered_session, 'non_pbe', None))
                filtered_session.non_pbe = sess.non_pbe.time_slice(t_start=filtered_session.t_start, t_stop=filtered_session.t_stop)
                # 'non_pbe_endcaps'
                original_filtered_non_pbe_endcaps = deepcopy(getattr(filtered_session, 'non_pbe_endcaps', None))
                filtered_session.non_pbe_endcaps = sess.non_pbe_endcaps.time_slice(t_start=filtered_session.t_start, t_stop=filtered_session.t_stop)                

                # Check if filtered session changed
                did_change = did_change or (original_filtered_non_pbe is None) or (not original_filtered_non_pbe.to_dataframe().equals(filtered_session.non_pbe.to_dataframe()))
                # Check if filtered session changed for non_pbe_endcaps
                did_change = did_change or (original_filtered_non_pbe_endcaps is None) or (not original_filtered_non_pbe_endcaps.to_dataframe().equals(filtered_session.non_pbe_endcaps.to_dataframe()))
                    
        return did_change, sess, filtered_sessions


    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # # Remove the unpicklable entries.
        # _non_pickled_fields = ['curr_active_pipeline', 'track_templates']
        # for a_non_pickleable_field in _non_pickled_fields:
        #     del state[a_non_pickleable_field]
        return state


    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)
        
        # _non_pickled_field_restore_defaults = dict(zip(['curr_active_pipeline', 'track_templates'], [None, None]))
        # for a_field_name, a_default_restore_value in _non_pickled_field_restore_defaults.items():
        #     if a_field_name not in state:
        #         state[a_field_name] = a_default_restore_value

        self.__dict__.update(state)
        # # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(WCorrShuffle, self).__init__() # from

    def __repr__(self):
        """ 2024-01-11 - Renders only the fields and their sizes
        """
        from pyphocorehelpers.print_helpers import strip_type_str_to_classname
        attr_reprs = []
        for a in self.__attrs_attrs__:
            attr_type = strip_type_str_to_classname(type(getattr(self, a.name)))
            if 'shape' in a.metadata:
                shape = ', '.join(a.metadata['shape'])  # this joins tuple elements with a comma, creating a string without quotes
                attr_reprs.append(f"{a.name}: {attr_type} | shape ({shape})")  # enclose the shape string with parentheses
            else:
                attr_reprs.append(f"{a.name}: {attr_type}")
        content = ",\n\t".join(attr_reprs)
        return f"{type(self).__name__}({content}\n)"
    


    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)




# ==================================================================================================================== #
# Global Computation Functions                                                                                         #
# ==================================================================================================================== #

# ==================================================================================================================== #
# 2025-03-09 - Build Output Posteriors                                                                                 #
# ==================================================================================================================== #
from typing import Literal
# Define a type that can only be one of these specific strings
KnownNamedDecodingEpochsType = Literal['laps', 'replay', 'ripple', 'pbe', 'non_pbe', 'non_pbe_endcaps', 'global']
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import MaskedTimeBinFillType

GenericResultTupleIndexType: TypeAlias = MaskedTimeBinFillType # an template/stand-in variable that aims to abstract away the unique-hashable index of a single result computed with a given set of parameters. Not yet fully implemented 2025-03-09 17:50 


@function_attributes(short_name=None, tags=['DEPRICATED', 'replaced', 'not-general'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-11 13:50', related_items=['GenericDecoderDictDecodedEpochsDictResult'])
@define(slots=False, repr=False, eq=False)
class GeneralDecoderDictDecodedEpochsDictResult(ComputedResult):
    """ REPLACED BY `GenericDecoderDictDecodedEpochsDictResult` on 2025-03-11 13:51 
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import GeneralDecoderDictDecodedEpochsDictResult, GenericResultTupleIndexType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType
    
    
    """
    _VersionedResultMixin_version: str = "2025.03.09_0" # to be updated in your IMPLEMENTOR to indicate its version

    filter_epochs_to_decode_dict: Dict[KnownNamedDecodingEpochsType, Epoch] = serialized_field(default=Factory(dict), repr=False)
    filter_epochs_pseudo2D_continuous_specific_decoded_result: Dict[KnownNamedDecodingEpochsType, DecodedFilterEpochsResult] = serialized_field(default=Factory(dict), repr=False)
    filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict: Dict[KnownNamedDecodingEpochsType, Dict[GenericResultTupleIndexType, pd.DataFrame]] = serialized_field(default=Factory(dict), repr=False)
    
    # ==================================================================================================================== #
    # Plotting Methods                                                                                                     #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['figure', 'plotly'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-21 13:40', related_items=[])
    def build_plotly_marginal_scatter_and_hist_over_time(self, histogram_bins: int = 25, debug_print = False):
        """ adds new tracks
        
        Adds 3 tracks like: ['ContinuousDecode_longnon-PBE-pseudo2D marginals', 'ContinuousDecode_shortnon-PBE-pseudo2D marginals', 'non-PBE_marginal_over_track_ID_ContinuousDecode - t_bin_size: 0.05']
    
        ## Compute and plot the new tracks:
        non_PBE_all_directional_pf1D_Decoder, pseudo2D_continuous_specific_decoded_result, continuous_decoded_results_dict, non_PBE_marginal_over_track_ID, (time_bin_containers, time_window_centers) = nonPBE_results._build_merged_joint_placefields_and_decode(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)))
        unique_added_track_identifiers = nonPBE_results.add_to_SpikeRaster2D_tracks(active_2d_plot=active_2d_plot, non_PBE_all_directional_pf1D_Decoder=non_PBE_all_directional_pf1D_Decoder, pseudo2D_continuous_specific_decoded_result=pseudo2D_continuous_specific_decoded_result, continuous_decoded_results_dict=continuous_decoded_results_dict, non_PBE_marginal_over_track_ID=non_PBE_marginal_over_track_ID, time_window_centers=time_window_centers)
        
        
        
        """
        import plotly.io as pio
        template: str = 'plotly_dark' # set plotl template
        pio.templates.default = template
        from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_helpers import plotly_pre_post_delta_scatter

        ## INPUTS: a_general_decoder_dict_decoded_epochs_dict_result

        # 'masked_laps': 'Laps (Masked)', 'masked_laps': 'Laps (Nan-masked)')
        # masked_bin_fill_modes: ['ignore', 'last_valid', 'nan_filled', 'dropped']

        _flat_out_figs_dict = {}

        for a_known_decoded_epochs_type, a_decoded_posterior_dfs_dict in self.filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict.items():
            if debug_print:
                print(f'a_known_decoded_epochs_type: "{a_known_decoded_epochs_type}"')
            for masking_bin_fill_mode, a_decoded_posterior_df in a_decoded_posterior_dfs_dict.items():
                if debug_print:
                    print(f'\tmasking_bin_fill_mode: "{masking_bin_fill_mode}"')
                plot_row_identifier: str = f'{a_known_decoded_epochs_type.capitalize()} - {masking_bin_fill_mode.capitalize()} decoder' # should be like 'Laps (Masked) from Non-PBE decoder'
                
                fig, figure_context = plotly_pre_post_delta_scatter(data_results_df=deepcopy(a_decoded_posterior_df), out_scatter_fig=None, 
                                                histogram_variable_name='P_Short', hist_kwargs=dict(), histogram_bins=histogram_bins,
                                                common_plot_kwargs=dict(),
                                                px_scatter_kwargs = dict(x='delta_aligned_start_t', y='P_Short', title=plot_row_identifier))
                _flat_out_figs_dict[figure_context] = fig
                
        # ['laps', 'non_PBE']
        # ['a', 'masked', 'dropping_masked']
        return _flat_out_figs_dict

    # Utility Methods ____________________________________________________________________________________________________ #
    def __setstate__(self, state):
        # Restore instance attributes
        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)
        self.__dict__.update(state)
        
    def __repr__(self):
        """ 2024-01-11 - Renders only the fields and their sizes
        """
        from pyphocorehelpers.print_helpers import strip_type_str_to_classname
        attr_reprs = []
        for a in self.__attrs_attrs__:
            attr_type = strip_type_str_to_classname(type(getattr(self, a.name)))
            if 'shape' in a.metadata:
                shape = ', '.join(a.metadata['shape'])  # this joins tuple elements with a comma, creating a string without quotes
                attr_reprs.append(f"{a.name}: {attr_type} | shape ({shape})")  # enclose the shape string with parentheses
            else:
                attr_reprs.append(f"{a.name}: {attr_type}")
        content = ",\n\t".join(attr_reprs)
        return f"{type(self).__name__}({content}\n)"


    def to_hdf(self, file_path, key: str, debug_print=False, **kwargs):
        """Custom HDF5 serialization function that handles complex nested dictionaries without global expansion.
        
        Explicitly serializes each complex field in a type-specific manner.
        """
        # First call the parent implementation to handle the basic fields
        super().to_hdf(file_path, key, debug_print=debug_print, **kwargs)
        
        # Now handle our complex dictionary fields manually
        file_mode = kwargs.get('file_mode', 'a')  # default to append
        
        # 1. Handle filter_epochs_to_decode_dict
        if self.filter_epochs_to_decode_dict:
            for epoch_type, epoch_obj in self.filter_epochs_to_decode_dict.items():
                # Convert Epoch to DataFrame and save it
                epoch_key = f"{key}/filter_epochs_to_decode_dict/{epoch_type}"
                epoch_df = epoch_obj.to_dataframe()
                epoch_df.to_hdf(file_path, key=epoch_key)
        
        # 2. Handle filter_epochs_pseudo2D_continuous_specific_decoded_result
        if self.filter_epochs_pseudo2D_continuous_specific_decoded_result:
            for epoch_type, decoded_result in self.filter_epochs_pseudo2D_continuous_specific_decoded_result.items():
                result_key = f"{key}/filter_epochs_pseudo2D_continuous_specific_decoded_result/{epoch_type}"
                # Use the DecodedFilterEpochsResult's serialization method if available
                if hasattr(decoded_result, 'to_hdf'):
                    decoded_result.to_hdf(file_path, key=result_key)
                else:
                    # Fallback: save key attributes as separate datasets
                    # For example, save p_x_given_n arrays and other essential components
                    with h5py.File(file_path, file_mode) as f:
                        # Create a group if it doesn't exist
                        if result_key not in f:
                            f.create_group(result_key)
                        
                        # Save essential attributes
                        if hasattr(decoded_result, 'p_x_given_n_list'):
                            for i, p_x in enumerate(decoded_result.p_x_given_n_list):
                                f.create_dataset(f"{result_key}/p_x_given_n_{i}", data=p_x)
        
        # 3. Handle filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict
        if self.filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict:
            for epoch_type, inner_dict in self.filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict.items():
                for fill_mode, df in inner_dict.items():
                    # DataFrame serialization is straightforward
                    df_key = f"{key}/filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict/{epoch_type}/{fill_mode}"
                    df.to_hdf(file_path, key=df_key)


    @classmethod
    def read_hdf(cls, file_path, key: str, **kwargs):
        """Read a previously saved GeneralDecoderDictDecodedEpochsDictResult from HDF5.
        
        Rebuilds complex nested dictionary structures from their serialized components.
        """
        # Create a new instance with basic attributes
        result = cls()
        
        # Populate filter_epochs_to_decode_dict
        filter_epochs_to_decode_dict = {}
        epochs_group_key = f"{key}/filter_epochs_to_decode_dict"
        
        # Check if the group exists
        with h5py.File(file_path, 'r') as f:
            if epochs_group_key in f:
                # List all epoch types saved
                epoch_types = list(f[epochs_group_key].keys())
                
                for epoch_type in epoch_types:
                    epoch_key = f"{epochs_group_key}/{epoch_type}"
                    # Read the epoch dataframe
                    epoch_df = pd.read_hdf(file_path, key=epoch_key)
                    # Convert back to Epoch object
                    filter_epochs_to_decode_dict[epoch_type] = Epoch(epoch_df)
        
        result.filter_epochs_to_decode_dict = filter_epochs_to_decode_dict
        
        # Similar logic for other complex dictionaries...
        
        return result




@define(slots=False, repr=False, eq=False)
class EpochComputationsComputationsContainer(ComputedResult):
    """ Holds the result from `perform_compute_non_PBE_epochs`


    Usage:

        from neuropy.core.epoch import EpochsAccessor, Epoch, ensure_dataframe, ensure_Epoch
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import EpochComputationFunctions, EpochComputationsComputationsContainer, DecodingResultND, Compute_NonPBE_Epochs, KnownFilterEpochs
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import GeneralDecoderDictDecodedEpochsDictResult, GenericResultTupleIndexType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType

        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

        ## Unpack from pipeline:
        nonPBE_results: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['EpochComputations']
        a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = nonPBE_results.a_new_NonPBE_Epochs_obj
        results1D: DecodingResultND = nonPBE_results.results1D
        results2D: DecodingResultND = nonPBE_results.results2D

        epochs_decoding_time_bin_size = nonPBE_results.epochs_decoding_time_bin_size
        frame_divide_bin_size = nonPBE_results.frame_divide_bin_size

        print(f'{epochs_decoding_time_bin_size = }, {frame_divide_bin_size = }')

        assert (results1D is not None)
        assert (results2D is not None)

        ## New computed properties:
        a_general_decoder_dict_decoded_epochs_dict_result: GeneralDecoderDictDecodedEpochsDictResult = nonPBE_results.a_general_decoder_dict_decoded_epochs_dict_result ## get the pre-decoded result
        a_general_decoder_dict_decoded_epochs_dict_result

            
    """
    _VersionedResultMixin_version: str = "2025.03.09_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    training_data_portion: float = serialized_attribute_field(default=(5.0/6.0))
    epochs_decoding_time_bin_size: float = serialized_attribute_field(default=0.020) 
    frame_divide_bin_size: float = serialized_attribute_field(default=10.0)

    a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = serialized_field(default=None, repr=False)
    results1D: Optional[DecodingResultND] = serialized_field(default=None, repr=False)
    results2D: Optional[DecodingResultND] = serialized_field(default=None, repr=False)

    a_general_decoder_dict_decoded_epochs_dict_result: GeneralDecoderDictDecodedEpochsDictResult = serialized_field(default=None, is_computable=True, repr=False, metadata={'field_added': '2025.03.09_0'})
    a_generic_decoder_dict_decoded_epochs_dict_result: GenericDecoderDictDecodedEpochsDictResult = serialized_field(default=None, is_computable=True, repr=False, metadata={'field_added': '2025.04.14_0'})

    # Utility Methods ____________________________________________________________________________________________________ #

    # def to_dict(self) -> Dict:
    #     # return asdict(self, filter=attrs.filters.exclude((self.__attrs_attrs__.is_global))) #  'is_global'
    #     return {k:v for k, v in self.__dict__.items() if k not in ['is_global']}
    
    # def to_hdf(self, file_path, key: str, debug_print=False, enable_hdf_testing_mode:bool=False, **kwargs):
    #     """ Saves the object to key in the hdf5 file specified by file_path
    #     enable_hdf_testing_mode: bool - default False - if True, errors are not thrown for the first field that cannot be serialized, and instead all are attempted to see which ones work.


    #     Usage:
    #         hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
    #         _pfnd_obj: PfND = long_one_step_decoder_1D.pf
    #         _pfnd_obj.to_hdf(hdf5_output_path, key='test_pfnd')
    #     """
    #     super().to_hdf(file_path, key=key, debug_print=debug_print, enable_hdf_testing_mode=enable_hdf_testing_mode, **kwargs)
    #     # handle custom properties here

    def __setstate__(self, state):
        # Restore instance attributes
        # For `VersionedResultMixin`
        self._VersionedResultMixin__setstate__(state)
        self.__dict__.update(state)
        
    def __repr__(self):
        """ 2024-01-11 - Renders only the fields and their sizes
        """
        from pyphocorehelpers.print_helpers import strip_type_str_to_classname
        attr_reprs = []
        for a in self.__attrs_attrs__:
            attr_type = strip_type_str_to_classname(type(getattr(self, a.name)))
            if 'shape' in a.metadata:
                shape = ', '.join(a.metadata['shape'])  # this joins tuple elements with a comma, creating a string without quotes
                attr_reprs.append(f"{a.name}: {attr_type} | shape ({shape})")  # enclose the shape string with parentheses
            else:
                attr_reprs.append(f"{a.name}: {attr_type}")
        content = ",\n\t".join(attr_reprs)
        return f"{type(self).__name__}({content}\n)"
    

    # ==================================================================================================================== #
    # Marginalization Methods                                                                                              #
    # ==================================================================================================================== #
    # From `General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions._build_merged_directional_placefields`
    def _build_merged_joint_placefields_and_decode(self, spikes_df: pd.DataFrame, filter_epochs:Optional[Epoch]=None, debug_print=False):
        """ Merges the computed directional placefields into a Pseudo2D decoder, with the pseudo last axis corresponding to the decoder index.

        NOTE: this builds a **decoder** not just placefields, which is why it depends on the time_bin_sizes (which will later be used for decoding)		

        Requires:
            ['sess']

        Provides:

        Usage:


            long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

            ## Unpack from pipeline:
            nonPBE_results: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['EpochComputations']
            
            non_PBE_all_directional_pf1D_Decoder, pseudo2D_continuous_specific_decoded_result, non_PBE_marginal_over_track_ID, (time_bin_containers, time_window_centers, track_marginal_posterior_df) = nonPBE_results._build_merged_joint_placefields_and_decode(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)))
            
            
        """
        from neuropy.analyses.placefields import PfND
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult

        a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = self.a_new_NonPBE_Epochs_obj
        results1D: DecodingResultND = self.results1D
        # results2D: DecodingResultND = self.results2D

        epochs_decoding_time_bin_size = self.epochs_decoding_time_bin_size
        frame_divide_bin_size = self.frame_divide_bin_size

        print(f'{epochs_decoding_time_bin_size = }, {frame_divide_bin_size = }')

        ## INPUTS: results1D, results1D.continuous_results, a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs

        unique_decoder_names: List[str] = ['long', 'short']
        # results1D.pfs
        # results1D.decoders # BasePositionDecoder

        pfs: Dict[types.DecoderName, PfND] = {k:deepcopy(v) for k, v in results1D.pfs.items() if k in unique_decoder_names}
        # decoders: Dict[types.DecoderName, BasePositionDecoder] = {k:deepcopy(v) for k, v in results1D.decoders.items() if k in unique_decoder_names}
        continuous_decoded_results_dict: Dict[str, DecodedFilterEpochsResult] = {k:deepcopy(v) for k, v in results1D.continuous_results.items() if k in unique_decoder_names}
        # DirectionalPseudo2DDecodersResult(

        ## Combine the non-directional PDFs and renormalize to get the directional PDF:
        non_PBE_all_directional_pf1D: PfND = PfND.build_merged_directional_placefields(pfs, debug_print=False)
        non_PBE_all_directional_pf1D_Decoder: BasePositionDecoder = BasePositionDecoder(non_PBE_all_directional_pf1D, setup_on_init=True, post_load_on_init=True, debug_print=False)
        # non_PBE_all_directional_pf1D_Decoder

        if filter_epochs is None:
            # use global epoch
            # single_global_epoch_df: pd.DataFrame = Epoch(deepcopy(a_new_NonPBE_Epochs_obj.single_global_epoch_df))
            single_global_epoch: Epoch = Epoch(deepcopy(a_new_NonPBE_Epochs_obj.single_global_epoch_df))
            filter_epochs = single_global_epoch


        # takes 6.3 seconds
        ## Do Continuous Decoding (for all time (`single_global_epoch`), using the decoder from each epoch) -- slowest dict comp
        pseudo2D_continuous_specific_decoded_result: DecodedFilterEpochsResult = non_PBE_all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=deepcopy(filter_epochs),
                                                                                                                                                decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False)

        ## OUTPUTS: pseudo2D_continuous_specific_decoded_results, non_PBE_all_directional_pf1D, non_PBE_all_directional_pf1D_Decoder
        # 3.3s

        #@ build_generalized_non_marginalized_raw_posteriors
        # ==================================================================================================================== #
        # Compute Marginals over TrackID                                                                                       #
        # ==================================================================================================================== #
        # pseudo2D_continuous_specific_decoded_result: DecodedFilterEpochsResult = pseudo2D_continuous_specific_decoded_result
        assert len(pseudo2D_continuous_specific_decoded_result.p_x_given_n_list) == 1

        # NOTE: non_marginalized_raw_result is a marginal_over_track_ID since there are only two elements
        non_PBE_marginal_over_track_IDs_list, non_PBE_marginal_over_track_ID_posterior_df = DirectionalPseudo2DDecodersResult.build_generalized_non_marginalized_raw_posteriors(pseudo2D_continuous_specific_decoded_result, unique_decoder_names=unique_decoder_names) ## Must be failing here:
        non_PBE_marginal_over_track_ID = non_PBE_marginal_over_track_IDs_list[0]['p_x_given_n']
        time_bin_containers = pseudo2D_continuous_specific_decoded_result.time_bin_containers[0]
        time_window_centers = time_bin_containers.centers
        # p_x_given_n.shape # (62, 4, 209389)

        ## Build into a marginal df like `all_sessions_laps_df` - uses `time_window_centers`, pseudo2D_continuous_specific_decoded_result, non_PBE_marginal_over_track_ID:
        track_marginal_posterior_df : pd.DataFrame = deepcopy(non_PBE_marginal_over_track_ID_posterior_df) # pd.DataFrame({'t':deepcopy(time_window_centers), 'P_Long': np.squeeze(non_PBE_marginal_over_track_ID[0, :]), 'P_Short': np.squeeze(non_PBE_marginal_over_track_ID[1, :]), 'time_bin_size': pseudo2D_continuous_specific_decoded_result.decoding_time_bin_size})
        
        if 'time_bin_size' not in track_marginal_posterior_df.columns:
            track_marginal_posterior_df['time_bin_size'] = pseudo2D_continuous_specific_decoded_result.decoding_time_bin_size ## add time_bin_size column if needed

        # track_marginal_posterior_df['delta_aligned_start_t'] = track_marginal_posterior_df['t'] - t_delta ## subtract off t_delta

        ## OUTPUTS: non_PBE_marginal_over_track_ID, time_bin_containers, time_window_centers
        return non_PBE_all_directional_pf1D_Decoder, pseudo2D_continuous_specific_decoded_result, continuous_decoded_results_dict, non_PBE_marginal_over_track_ID, (time_bin_containers, time_window_centers, track_marginal_posterior_df)


    @function_attributes(short_name=None, tags=['posteriors'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-09 05:21', related_items=[])
    @classmethod
    def _build_output_decoded_posteriors(cls, non_PBE_all_directional_pf1D_Decoder: BasePositionDecoder, filter_epochs_to_decode_dict: Dict[KnownNamedDecodingEpochsType, Epoch], unique_decoder_names: List[str], spikes_df: pd.DataFrame, epochs_decoding_time_bin_size: float,
                                        session_name: str, t_start: float, t_delta: float, t_end: float) -> GeneralDecoderDictDecodedEpochsDictResult:
        """ Given a also produces  Unit Time Binned Spike Count Masking of Decodings
        
        Breakdown:
        
        1. decoding for particular filter_epochs
        2. build raw posteriors from across the decoded epochs, returning a list of DynamicResults and a pd.DataFrame with the columns specified in unique_decoder_names (e.g. ['long', 'short'] or ['Long_LR', 'Long_RL', ...])
        3. mask the decoded result from step 1 by determining the number of spikes, number of unique active cells, etc in each decoding time bin (not in each epoch, which we do elsewhere). This discards bins with insufficient activity to properly decoding, which usually result in predictably noisy posteriors.
        4. build raw posteriors (most importantly dataframe) from the masked results
        5. test two different types of masking


        Not Needed: , pseudo2D_continuous_specific_decoded_result: DecodedFilterEpochsResult
        Usage:
            
            session_name: str = curr_active_pipeline.session_name
            t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()
            
            a_general_decoder_dict_decoded_epochs_dict_result: GeneralDecoderDictDecodedEpochsDictResult = EpochComputationsComputationsContainer._build_output_decoded_posteriors(non_PBE_all_directional_pf1D_Decoder=non_PBE_all_directional_pf1D_Decoder, pseudo2D_continuous_specific_decoded_result=pseudo2D_continuous_specific_decoded_result,
                filter_epochs_to_decode_dict=filter_epochs_to_decode_dict,
                unique_decoder_names=['long', 'short'], spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), epochs_decoding_time_bin_size=epochs_decoding_time_bin_size,
                session_name=session_name, t_start=t_start, t_delta=t_delta, t_end=t_end,
            )
        
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult, DecodedFilterEpochsResult, DirectionalPseudo2DDecodersResult
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult, MaskedTimeBinFillType


        assert epochs_decoding_time_bin_size is not None, f"epochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}"
        # time_bin_size: float = pseudo2D_continuous_specific_decoded_result.decoding_time_bin_size
        
        filter_epochs_pseudo2D_continuous_specific_decoded_result: Dict[KnownNamedDecodingEpochsType, DecodedFilterEpochsResult] = {}
        filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict: Dict[KnownNamedDecodingEpochsType, Dict[MaskedTimeBinFillType, pd.DataFrame]] = {}
        
        for a_decoded_epoch_type_name, a_filter_epoch_obj in filter_epochs_to_decode_dict.items():
            # a_decoded_epoch_type_name: like 'laps', 'ripple', or 'non_pbe'
            
            a_filtered_epochs_df = ensure_dataframe(deepcopy(a_filter_epoch_obj)).epochs.filtered_by_duration(min_duration=epochs_decoding_time_bin_size*2)
            # active_filter_epochs = a_filter_epoch_obj
            active_filter_epochs = a_filtered_epochs_df
            a_pseudo2D_continuous_specific_decoded_result: DecodedFilterEpochsResult = non_PBE_all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=deepcopy(active_filter_epochs), decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False)
            
            # a_filtered_epochs_df = a_pseudo2D_continuous_specific_decoded_result.filter_epochs.epochs.filtered_by_duration(min_duration=epochs_decoding_time_bin_size*2)
            # a_pseudo2D_continuous_specific_decoded_result = a_pseudo2D_continuous_specific_decoded_result.filtered_by_epoch_times(included_epoch_start_times=a_filtered_epochs_df['start'].to_numpy())
            

            filter_epochs_pseudo2D_continuous_specific_decoded_result[a_decoded_epoch_type_name] = a_pseudo2D_continuous_specific_decoded_result ## add result to outputs dict
            # a_pseudo2D_split_to_1D_continuous_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = a_pseudo2D_continuous_specific_decoded_result.split_pseudo2D_result_to_1D_result(pseudo2D_decoder_names_list=unique_decoder_names)
            a_non_PBE_marginal_over_track_ID, a_non_PBE_marginal_over_track_ID_posterior_df = DirectionalPseudo2DDecodersResult.build_generalized_non_marginalized_raw_posteriors(a_pseudo2D_continuous_specific_decoded_result, unique_decoder_names=unique_decoder_names) #[0]['p_x_given_n']


            
            ## MASKED:
            # masked_pseudo2D_continuous_specific_decoded_result, _mask_index_tuple = a_pseudo2D_continuous_specific_decoded_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(spikes_df), masked_bin_fill_mode='last_valid') ## Masks the low-firing bins so they don't confound the analysis.
            masked_laps_pseudo2D_continuous_specific_decoded_result, _mask_index_tuple = a_pseudo2D_continuous_specific_decoded_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(spikes_df), masked_bin_fill_mode='last_valid') ## Masks the low-firing bins so they don't confound the analysis.
            # masked_laps_pseudo2D_split_to_1D_continuous_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = masked_laps_pseudo2D_continuous_specific_decoded_result.split_pseudo2D_result_to_1D_result(pseudo2D_decoder_names_list=unique_decoder_names)
            masked_laps_non_PBE_marginal_over_track_ID, masked_laps_non_PBE_marginal_over_track_ID_posterior_df = DirectionalPseudo2DDecodersResult.build_generalized_non_marginalized_raw_posteriors(masked_laps_pseudo2D_continuous_specific_decoded_result, unique_decoder_names=unique_decoder_names) #[0]['p_x_given_n']

            ## MASKED with NaNs (no backfill):
            dropping_masked_laps_pseudo2D_continuous_specific_decoded_result, _dropping_mask_index_tuple = a_pseudo2D_continuous_specific_decoded_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(spikes_df), masked_bin_fill_mode='nan_filled') ## Masks the low-firing bins so they don't confound the analysis.
            # dropping_masked_laps_pseudo2D_split_to_1D_continuous_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = dropping_masked_laps_pseudo2D_continuous_specific_decoded_result.split_pseudo2D_result_to_1D_result(pseudo2D_decoder_names_list=unique_decoder_names)
            dropping_masked_laps_non_PBE_marginal_over_track_ID, dropping_masked_laps_non_PBE_marginal_over_track_ID_posterior_df = DirectionalPseudo2DDecodersResult.build_generalized_non_marginalized_raw_posteriors(dropping_masked_laps_pseudo2D_continuous_specific_decoded_result, unique_decoder_names=unique_decoder_names) #[0]['p_x_given_n']
            
            # from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionIdentityDataframeAccessor

            ## Build into a marginal df like `all_sessions_laps_df`:
            
            # track_marginal_posterior_df_list = [track_marginal_posterior_df, masked_track_marginal_posterior_df]

            # masked_bin_fill_modes: ['ignore', 'last_valid', 'nan_filled', 'dropped']

            # _track_marginal_posterior_df_dict = {'track_marginal_posterior_df':track_marginal_posterior_df, 'masked_track_marginal_posterior_df': masked_track_marginal_posterior_df}
            decoded_filter_epoch_track_marginal_posterior_df_dict: Dict[MaskedTimeBinFillType, pd.DataFrame] = {# 'track_marginal_posterior_df':track_marginal_posterior_df,
                                                                                            'ignore': a_non_PBE_marginal_over_track_ID_posterior_df,
                                                                                            'last_valid': masked_laps_non_PBE_marginal_over_track_ID_posterior_df,
                                                                                            'nan_filled': dropping_masked_laps_non_PBE_marginal_over_track_ID_posterior_df,
                                                                                            # 'a_non_PBE_marginal_over_track_ID_posterior_df': a_non_PBE_marginal_over_track_ID_posterior_df,
                                                                                            # 'masked_laps_non_PBE_marginal_over_track_ID_posterior_df': masked_laps_non_PBE_marginal_over_track_ID_posterior_df,
                                                                                            # 'dropping_masked_laps_non_PBE_marginal_over_track_ID_posterior_df': dropping_masked_laps_non_PBE_marginal_over_track_ID_posterior_df,
                                                                                            }

            # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
            for k in list(decoded_filter_epoch_track_marginal_posterior_df_dict.keys()):
                a_df = decoded_filter_epoch_track_marginal_posterior_df_dict[k]
                a_df['delta_aligned_start_t'] = a_df['t'] - t_delta ## subtract off t_delta    
                a_df = a_df.across_session_identity.add_session_df_columns(session_name=session_name, time_bin_size=epochs_decoding_time_bin_size, curr_session_t_delta=t_delta, time_col='t')
                decoded_filter_epoch_track_marginal_posterior_df_dict[k] = a_df
                
            # ['dropping_masked', 'dropping_masked', 'masked', 'dropping_masked']
            filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict[a_decoded_epoch_type_name] = decoded_filter_epoch_track_marginal_posterior_df_dict
            # # track_marginal_posterior_df = _track_marginal_posterior_df_dict['track_marginal_posterior_df']
            # # masked_track_marginal_posterior_df = _track_marginal_posterior_df_dict['masked_track_marginal_posterior_df']

            # if 'track_marginal_posterior_df' in decoded_filter_epoch_track_marginal_posterior_df_dict:
            #     track_marginal_posterior_df = decoded_filter_epoch_track_marginal_posterior_df_dict['track_marginal_posterior_df']
                
            # if 'masked_track_marginal_posterior_df' in decoded_filter_epoch_track_marginal_posterior_df_dict:
            #     masked_track_marginal_posterior_df = decoded_filter_epoch_track_marginal_posterior_df_dict['masked_track_marginal_posterior_df']
                
            # if 'laps_non_PBE_marginal_over_track_ID_posterior_df' in decoded_filter_epoch_track_marginal_posterior_df_dict:
            #     laps_non_PBE_marginal_over_track_ID_posterior_df = decoded_filter_epoch_track_marginal_posterior_df_dict['laps_non_PBE_marginal_over_track_ID_posterior_df']

            # if 'masked_laps_non_PBE_marginal_over_track_ID_posterior_df' in decoded_filter_epoch_track_marginal_posterior_df_dict:
            #     masked_laps_non_PBE_marginal_over_track_ID_posterior_df = decoded_filter_epoch_track_marginal_posterior_df_dict['masked_laps_non_PBE_marginal_over_track_ID_posterior_df']

            # if 'dropping_masked_laps_non_PBE_marginal_over_track_ID_posterior_df' in decoded_filter_epoch_track_marginal_posterior_df_dict:
            #     dropping_masked_laps_non_PBE_marginal_over_track_ID_posterior_df = decoded_filter_epoch_track_marginal_posterior_df_dict['dropping_masked_laps_non_PBE_marginal_over_track_ID_posterior_df']


            # track_marginal_posterior_df
            # masked_track_marginal_posterior_df

            ## UPDATES: track_marginal_posterior_df, masked_track_marginal_posterior_df
            ## UPDATES: laps_non_PBE_marginal_over_track_ID_posterior_df, masked_laps_non_PBE_marginal_over_track_ID_posterior_df
            # masked_laps_non_PBE_marginal_over_track_ID_posterior_df
            # dropping_masked_laps_non_PBE_marginal_over_track_ID_posterior_df
            
            ## OUTPUTS: laps_non_PBE_marginal_over_track_ID_posterior_df, dropping_masked_laps_non_PBE_marginal_over_track_ID_posterior_df, masked_laps_non_PBE_marginal_over_track_ID_posterior_df

        # END for a_...

        # return filter_epochs_pseudo2D_continuous_specific_decoded_result, filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict
        return GeneralDecoderDictDecodedEpochsDictResult(filter_epochs_to_decode_dict=filter_epochs_to_decode_dict, filter_epochs_pseudo2D_continuous_specific_decoded_result=filter_epochs_pseudo2D_continuous_specific_decoded_result,
                                                         filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict=filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict)



    # ==================================================================================================================== #
    # NEW Context-General Method 2025-04-05 10:12                                                                          #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['posteriors'], input_requires=[], output_provides=[], uses=['DirectionalPseudo2DDecodersResult.perform_compute_specific_marginals'], used_by=['_do_perform_decoding'], creation_date='2025-05-02 16:16', related_items=[])
    @classmethod
    def _build_context_general_output_decoded_posteriors(cls, non_PBE_all_directional_pf1D_Decoder: BasePositionDecoder, filter_epochs_to_decode_dict: Dict[types.GenericResultTupleIndexType, Epoch], unique_decoder_names: List[str], spikes_df: pd.DataFrame, epochs_decoding_time_bin_size: float,
                                        session_name: str, t_start: float, t_delta: float, t_end: float, debug_print:bool=True) -> Tuple:
        """ Given a also produces  Unit Time Binned Spike Count Masking of Decodings
        
        History: based off of `_build_output_decoded_posteriors` on 2025-04-05 10:41 with the purpose of generalizing the result to the real flat general IdentifyingContext-keyed class instead of `GeneralDecoderDictDecodedEpochsDictResult`
        
        
        
        Breakdown:
        
        1. decoding for particular filter_epochs
        2. build raw posteriors from across the decoded epochs, returning a list of DynamicResults and a pd.DataFrame with the columns specified in unique_decoder_names (e.g. ['long', 'short'] or ['Long_LR', 'Long_RL', ...])
        3. mask the decoded result from step 1 by determining the number of spikes, number of unique active cells, etc in each decoding time bin (not in each epoch, which we do elsewhere). This discards bins with insufficient activity to properly decoding, which usually result in predictably noisy posteriors.
        4. build raw posteriors (most importantly dataframe) from the masked results
        5. test two different types of masking


        Not Needed: , pseudo2D_continuous_specific_decoded_result: DecodedFilterEpochsResult
        Usage:
            
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import EpochComputationsComputationsContainer

            ## INPUTS: a_decoder
            session_name: str = curr_active_pipeline.session_name
            t_start, t_delta, t_end = curr_active_pipeline.find_LongShortDelta_times()


            epochs_decoding_time_bin_size = 0.025
            
            a_base_context = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', time_bin_size=time_bin_size, data_grain='per_time_bin') # , known_named_decoding_epochs_type='laps', masked_time_bin_fill_type='ignore'
            a_best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_matching_contexts(a_base_context, return_multiple_matches=False)
            
            filter_epochs_to_decode_dict = {IdentifyingContext(known_named_decoding_epochs_type='laps'):deepcopy(laps_df),
                                            # IdentifyingContext(known_named_decoding_epochs_type='pbes'):deepcopy(non_pbe_df),
                                            IdentifyingContext(known_named_decoding_epochs_type='non_pbe_endcaps'):deepcopy(non_pbe_endcaps_df),
            }

            _temp_out_tuple = EpochComputationsComputationsContainer._build_context_general_output_decoded_posteriors(non_PBE_all_directional_pf1D_Decoder=a_decoder, filter_epochs_to_decode_dict=filter_epochs_to_decode_dict,
                # unique_decoder_names=['long', 'short'],
                unique_decoder_names=['long_LR', 'long_RL', 'short_LR', 'short_RL'],
                spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), epochs_decoding_time_bin_size=epochs_decoding_time_bin_size,
                session_name=session_name, t_start=t_start, t_delta=t_delta, t_end=t_end,
            )

            filter_epochs_to_decoded_dict, filter_epochs_pseudo2D_continuous_specific_decoded_result, filter_epochs_decoder_dict, filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict = _temp_out_tuple

                    
        """
        from pyphocorehelpers.assertion_helpers import Assert
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult, DecodedFilterEpochsResult, DirectionalPseudo2DDecodersResult
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult, MaskedTimeBinFillType
        from neuropy.core.epoch import Epoch, TimeColumnAliasesProtocol, subdivide_epochs, ensure_dataframe, ensure_Epoch

        assert epochs_decoding_time_bin_size is not None, f"epochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}"

        filter_epochs_to_decoded_dict: Dict[types.GenericResultTupleIndexType, pd.DataFrame] = {} # NOTE: needs to be different variable than the incomming `filter_epochs_to_decode_dict` because that is being iterated over.
        filter_epochs_pseudo2D_continuous_specific_decoded_result: Dict[types.GenericResultTupleIndexType, DecodedFilterEpochsResult] = {}
        filter_epochs_decoder_dict: Dict[types.GenericResultTupleIndexType, BasePositionDecoder] = {}
        filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict: Dict[types.GenericResultTupleIndexType, pd.DataFrame] = {}
        
        for a_decoded_epoch_context, a_filter_epoch_obj in filter_epochs_to_decode_dict.items():
            # a_decoded_epoch_type_name: like 'laps', 'ripple', or 'non_pbe'
            assert not isinstance(a_decoded_epoch_context, str), f"a_decoded_epoch_context: {a_decoded_epoch_context} should be a real context not a string!"
            
            a_filtered_epochs_df = ensure_dataframe(deepcopy(a_filter_epoch_obj)).epochs.filtered_by_duration(min_duration=epochs_decoding_time_bin_size*2)
            # active_filter_epochs = a_filter_epoch_obj
            active_filter_epochs = a_filtered_epochs_df
            a_pseudo2D_continuous_specific_decoded_result: DecodedFilterEpochsResult = non_PBE_all_directional_pf1D_Decoder.decode_specific_epochs(spikes_df=deepcopy(spikes_df), filter_epochs=deepcopy(active_filter_epochs), decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False)
            
            ## add time bin to the epoch
            a_decoded_epoch_context = a_decoded_epoch_context.overwriting_context(decoding_time_bin_size=epochs_decoding_time_bin_size)

            # from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionIdentityDataframeAccessor

            ## Build into a marginal df like `all_sessions_laps_df`:
            masked_bin_fill_modes: List[MaskedTimeBinFillType] = ['ignore', 'last_valid', 'nan_filled', 'dropped']
            
            for a_masked_bin_fill_mode in masked_bin_fill_modes:
                ## MASKED:
                a_masked_decoded_epoch_context = deepcopy(a_decoded_epoch_context).overwriting_context(masked_time_bin_fill_type=a_masked_bin_fill_mode) # IdentifyingContext
                if debug_print:
                    print(f'\ta_masked_decoded_epoch_context: {a_masked_decoded_epoch_context}')
                

                a_masked_decoded_result, _mask_index_tuple = a_pseudo2D_continuous_specific_decoded_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(spikes_df), masked_bin_fill_mode=a_masked_bin_fill_mode) ## Masks the low-firing bins so they don't confound the analysis.

                # _a_masked_unused_marginal, a_masked_posterior_df = DirectionalPseudo2DDecodersResult.build_generalized_non_marginalized_raw_posteriors(a_masked_decoded_result, unique_decoder_names=unique_decoder_names) #[0]['p_x_given_n']

                a_masked_posterior_df = DirectionalPseudo2DDecodersResult.perform_compute_specific_marginals(a_result=a_masked_decoded_result, marginal_context=a_masked_decoded_epoch_context) # 2025-05-02 - improved method
                

                ## spruce up the `a_masked_posterior_df` with some extra fields
                # t_bin_col_name: str = 't'
                # t_bin_col_name: str = 't_bin_center'
                t_bin_col_name: str = TimeColumnAliasesProtocol.find_first_extant_suitable_columns_name(df=a_masked_posterior_df, col_connonical_name='t', required_columns_synonym_dict={'t':['t','t_bin_center']}, should_raise_exception_on_fail=True)
                a_masked_posterior_df['delta_aligned_start_t'] = a_masked_posterior_df[t_bin_col_name] - t_delta ## subtract off t_delta    
                a_masked_posterior_df = a_masked_posterior_df.across_session_identity.add_session_df_columns(session_name=session_name, time_bin_size=epochs_decoding_time_bin_size, curr_session_t_delta=t_delta, time_col=t_bin_col_name)
                
                ## OUPUTS: a_masked_decoded_result, a_masked_posterior_df
                filter_epochs_to_decoded_dict[a_masked_decoded_epoch_context] = deepcopy(active_filter_epochs)
                filter_epochs_pseudo2D_continuous_specific_decoded_result[a_masked_decoded_epoch_context] = a_masked_decoded_result ## add result to outputs dict
                filter_epochs_decoder_dict[a_masked_decoded_epoch_context] = deepcopy(non_PBE_all_directional_pf1D_Decoder)
                filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict[a_masked_decoded_epoch_context] = a_masked_posterior_df
            ## END for a_masked_bin_fill_m...
            # Assert.same_length(filter_epochs_to_decoded_dict, filter_epochs_pseudo2D_continuous_specific_decoded_result, filter_epochs_decoder_dict, filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict)

            ## UPDATES: filter_epochs_to_decoded_dict, filter_epochs_pseudo2D_continuous_specific_decoded_result, filter_epochs_decoder_dict, filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict

        # END for a_decoded_epoch_context, a_fi...
        
        Assert.same_length(filter_epochs_to_decoded_dict, filter_epochs_pseudo2D_continuous_specific_decoded_result, filter_epochs_decoder_dict, filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict)
        
        return (filter_epochs_to_decoded_dict, filter_epochs_pseudo2D_continuous_specific_decoded_result, filter_epochs_decoder_dict, filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict) ## return a plain tuple of dicts
    


    # ==================================================================================================================== #
    # Plotting Methods                                                                                                     #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['plotting', 'TO_DEPRICATE'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-21 13:40', related_items=[])
    def add_to_SpikeRaster2D_tracks(self, active_2d_plot, non_PBE_all_directional_pf1D_Decoder, pseudo2D_continuous_specific_decoded_result, continuous_decoded_results_dict, non_PBE_marginal_over_track_ID, time_window_centers):
        """ adds new tracks
        
        Adds 3 tracks like: ['ContinuousDecode_longnon-PBE-pseudo2D marginals', 'ContinuousDecode_shortnon-PBE-pseudo2D marginals', 'non-PBE_marginal_over_track_ID_ContinuousDecode - t_bin_size: 0.05']
    
        ## Compute and plot the new tracks:
        non_PBE_all_directional_pf1D_Decoder, pseudo2D_continuous_specific_decoded_result, continuous_decoded_results_dict, non_PBE_marginal_over_track_ID, (time_bin_containers, time_window_centers) = nonPBE_results._build_merged_joint_placefields_and_decode(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)))
        unique_added_track_identifiers = nonPBE_results.add_to_SpikeRaster2D_tracks(active_2d_plot=active_2d_plot, non_PBE_all_directional_pf1D_Decoder=non_PBE_all_directional_pf1D_Decoder, pseudo2D_continuous_specific_decoded_result=pseudo2D_continuous_specific_decoded_result, continuous_decoded_results_dict=continuous_decoded_results_dict, non_PBE_marginal_over_track_ID=non_PBE_marginal_over_track_ID, time_window_centers=time_window_centers)
        
        
        
        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import AddNewDecodedPosteriors_MatplotlibPlotCommand, AddNewDecodedEpochMarginal_MatplotlibPlotCommand
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, CustomCyclicColorsDockDisplayConfig, NamedColorScheme
        from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers


        # a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = self.a_new_NonPBE_Epochs_obj
        results1D: DecodingResultND = self.results1D
        # results2D: DecodingResultND = self.results2D

        epochs_decoding_time_bin_size = self.epochs_decoding_time_bin_size
        frame_divide_bin_size = self.frame_divide_bin_size

        print(f'{epochs_decoding_time_bin_size = }, {frame_divide_bin_size = }')
        

        ## Main INPUT: continuous_specific_decoded_results_dict
        # display_output = {}
        unique_added_track_identifiers: List[str] = []


        AddNewDecodedPosteriors_MatplotlibPlotCommand._build_dock_group_id(extended_dock_title_info='non-PBE-pseudo2D marginals')
        ## INPUTS: pseudo2D_continuous_specific_decoded_result, non_PBE_marginal_over_track_ID

        # override_dock_group_name: str = 'non_pbe_continuous_decoding_plot_group'
        override_dock_group_name: str = None ## this feature doesn't work
        _cont_posteriors_output_dict = AddNewDecodedPosteriors_MatplotlibPlotCommand.prepare_and_perform_custom_decoder_decoded_epochs(curr_active_pipeline=None, active_2d_plot=active_2d_plot,
                                                                                                                        continuously_decoded_dict=continuous_decoded_results_dict, info_string='non-PBE-pseudo2D marginals', # results1D.continuous_results
                                                                                                                        xbin=deepcopy(results1D.decoders['global'].xbin), skip_plotting_measured_positions=False, debug_print=False,
                                                                                                                        dock_group_name=override_dock_group_name)


        # dict long/short
        unique_added_track_identifiers.extend([v[0] for k, v in _cont_posteriors_output_dict.items()]) # v[0] is the identifier
        
        # display_output.update(_cont_posteriors_output_dict)

        # ==================================================================================================================== #
        # Plot the Decodings and their Marginals over TrackID as new Tracks                                                    #
        # ==================================================================================================================== #

        ## INPUTS: non_PBE_marginal_over_track_ID

        ## Manually call `AddNewDecodedEpochMarginal_MatplotlibPlotCommand` to add the custom marginals track to the active SpikeRaster3DWindow
        time_bin_size = epochs_decoding_time_bin_size
        info_string: str = f" - t_bin_size: {time_bin_size}"

        dock_config = CustomCyclicColorsDockDisplayConfig(showCloseButton=True, named_color_scheme=NamedColorScheme.grey)
        dock_config.dock_group_names = [override_dock_group_name] # , 'non-PBE Continuous Decoding'

        _marginalized_post_output_dict = {}
        a_posterior_name: str = 'non-PBE_marginal_over_track_ID'
        assert non_PBE_marginal_over_track_ID.shape[0] == 2, f"expected the 2 marginalized pseudo-y bins for the decoder in non_PBE_marginal_over_track_ID.shape[1]. but found non_PBE_marginal_over_track_ID.shape: {non_PBE_marginal_over_track_ID.shape}"
        _marginalized_post_output_dict[a_posterior_name] = AddNewDecodedEpochMarginal_MatplotlibPlotCommand._perform_add_new_decoded_posterior_marginal_row(curr_active_pipeline=None, active_2d_plot=active_2d_plot, a_dock_config=dock_config,
                                                                                            a_variable_name=a_posterior_name, xbin=np.arange(2), time_window_centers=time_window_centers, a_1D_posterior=non_PBE_marginal_over_track_ID, extended_dock_title_info=info_string)
        # display_output.update({'a_posterior_name': _marginalized_post_output_dict[a_posterior_name]})

        ## Draw the "Long", "Short" labels
        identifier_name, widget, matplotlib_fig, matplotlib_fig_axes = _marginalized_post_output_dict[a_posterior_name]
        label_artists_dict = {}
        for i, ax in enumerate(matplotlib_fig_axes):
            label_artists_dict[ax] = PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels(ax, y_bin_labels=['short','long'], enable_draw_decoder_colored_lines=False)
        _marginalized_post_output_dict[a_posterior_name] = (identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, label_artists_dict)
        unique_added_track_identifiers.append(identifier_name)

        return unique_added_track_identifiers



def validate_has_non_PBE_epoch_results(curr_active_pipeline, computation_filter_name='maze', minimum_inclusion_fr_Hz:Optional[float]=None):
    """ for `EpochComputationFunctions.perform_compute_non_PBE_epochs` 
    Returns True if the pipeline has a valid RankOrder results set of the latest version

    TODO: make sure minimum can be passed. Actually, can get it from the pipeline.

    """
    # Unpacking:
    seq_results: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['EpochComputations']
    if seq_results is None:
        return False
    
    a_new_NonPBE_Epochs_obj = seq_results.a_new_NonPBE_Epochs_obj
    if a_new_NonPBE_Epochs_obj is None:
        return False
    

    a_general_decoder_dict_decoded_epochs_dict_result = seq_results.a_general_decoder_dict_decoded_epochs_dict_result
    if a_general_decoder_dict_decoded_epochs_dict_result is None:
        return False

    # _computationPrecidence = 2 # must be done after PlacefieldComputations, DefaultComputationFunctions
    # _is_global = False


def validate_has_generalized_specific_epochs_decoding(curr_active_pipeline, computation_filter_name='maze', minimum_inclusion_fr_Hz:Optional[float]=None):
    """ for `EpochComputationFunctions.perform_generalized_specific_epochs_decoding` 
    Returns True if the pipeline has a valid RankOrder results set of the latest version

    TODO: make sure minimum can be passed. Actually, can get it from the pipeline.

    """
    # Unpacking:
    seq_results: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['EpochComputations']
    if seq_results is None:
        return False
    
    a_new_NonPBE_Epochs_obj = seq_results.a_new_NonPBE_Epochs_obj
    if a_new_NonPBE_Epochs_obj is None:
        return False
    

    a_generic_decoder_dict_decoded_epochs_dict_result = seq_results.a_generic_decoder_dict_decoded_epochs_dict_result
    if a_generic_decoder_dict_decoded_epochs_dict_result is None:
        return False

    if not a_generic_decoder_dict_decoded_epochs_dict_result.has_all_per_epoch_aggregations():
        return False
        



from pyphocorehelpers.programming_helpers import MemoryManagement # used in `EpochComputationFunctions.perform_compute_non_PBE_epochs`

def estimate_memory_requirements_bytes(epochs_decoding_time_bin_size: float, frame_divide_bin_size: float, n_flattened_position_bins: Optional[int]=None, n_neurons: Optional[int]=None, session_duration: Optional[float]=None, n_maze_contexts: int=9) -> Tuple[int, dict]:
    """Estimates memory requirements for non-PBE epoch computations
    
    Args:
        epochs_decoding_time_bin_size (float): Size of each decoding time bin in seconds
        frame_divide_bin_size (float): Size of frame_division bins in seconds
        n_neurons (int, optional): Number of neurons
        session_duration (float, optional): Duration in seconds
        n_maze_contexts (int): Number of maze contexts to process (default 9)
    
    Returns:
        Tuple[int, dict]: (Total estimated bytes, Detailed memory breakdown dictionary)
        
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import estimate_memory_requirements
        
    History:
        'frame_divide' -> 'frame_divide'
        'n_frame_divided_bins' -> 'n_frame_divided_bins'
    """
    # Default values if not provided
    if n_neurons is None:
        n_neurons = 100  # Conservative default
    if session_duration is None:
        session_duration = 3600  # Default 1 hour

    # Calculate array dimensions

    # Use Python's arbitrary precision integers for calculations
    n_flattened_position_bins = int(n_flattened_position_bins)
    n_neurons = int(n_neurons)
    session_duration = int(np.ceil(session_duration))
    n_maze_contexts = int(n_maze_contexts)
    n_time_bins = int(np.ceil(session_duration / epochs_decoding_time_bin_size))
    n_frame_divided_bins = int(np.ceil(session_duration / frame_divide_bin_size))    
    n_max_decoded_bins = int(max(n_time_bins, n_frame_divided_bins))
    
    bytes_per_float = int(8)
    
    # Calculate memory for each major array type
    itemized_memory_breakdown = {
        'spike_counts': (n_time_bins * n_neurons) * bytes_per_float,
        'firing_rates': (n_max_decoded_bins * n_neurons) * bytes_per_float, 
        'position_decoded': (n_max_decoded_bins * 2) * bytes_per_float,  
        'posterior': (n_max_decoded_bins * n_flattened_position_bins) * bytes_per_float,
        # 'posterior_intermediate_computation': (n_max_decoded_bins * n_flattened_position_bins * n_neurons) * bytes_per_float, 
        'occupancy': n_flattened_position_bins * bytes_per_float, 
        # 'worst_case_scenario': (n_max_decoded_bins * n_flattened_position_bins * n_neurons) * bytes_per_float, 
    }

    # itemized_memory_breakdown_GB = {k:(v/1e9) for k, v in itemized_memory_breakdown.items()}
    # Account for multiple maze contexts
    total_memory = sum(itemized_memory_breakdown.values()) * n_maze_contexts
    
    # Calculate the worst case scenario separately to avoid overflow
    worst_case_memory = int(n_max_decoded_bins) * int(n_flattened_position_bins) * int(n_neurons) * int(bytes_per_float)

    intermediate_itemized_memory_breakdown = {
        'worst_case_scenario': worst_case_memory, 
    }
    total_memory += sum(intermediate_itemized_memory_breakdown.values())
    
    # Add 20% overhead for temporary arrays and computations
    total_memory_with_overhead = int(total_memory * 1.2)
    
    return total_memory_with_overhead, itemized_memory_breakdown


# 'epoch_computations', '
class EpochComputationFunctions(AllFunctionEnumeratingMixin, metaclass=ComputationFunctionRegistryHolder):
    _computationGroupName = 'epoch_computations'
    _computationGlobalResultGroupName = 'EpochComputations'
    _computationPrecidence = 1006
    _is_global = True

    @function_attributes(short_name='non_PBE_epochs_results', tags=['epochs', 'nonPBE'],
        input_requires=[], output_provides=[], uses=['EpochComputationsComputationsContainer', 'Compute_NonPBE_Epochs'], used_by=[], creation_date='2025-02-18 09:45', related_items=[],
        requires_global_keys=[], provides_global_keys=['EpochComputations'],
        validate_computation_test=validate_has_non_PBE_epoch_results, is_global=True)
    def perform_compute_non_PBE_epochs(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, training_data_portion: float=(5.0/6.0), epochs_decoding_time_bin_size: float = 0.050, frame_divide_bin_size:float=10.0,
                                        compute_1D: bool = True, compute_2D: bool = True, drop_previous_result_and_compute_fresh:bool=False, skip_training_test_split: bool = True, debug_print_memory_breakdown: bool=False):
        """ Performs the computation of non-PBE epochs for the session and all filtered epochs. Stacks things up hardcore yeah.

        Requires:
            ['sess']

        Provides:
            global_computation_results.computed_data['EpochComputations']
                ['EpochComputations'].odd_ripple
                ['RankOrder'].even_ripple
                ['RankOrder'].odd_laps
                ['RankOrder'].even_laps

        dict(epochs_decoding_time_bin_size=epochs_decoding_time_bin_size, drop_previous_result_and_compute_fresh=False, frame_divide_bin_size=10.0)



        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import EpochFilteringMode, _compute_proper_filter_epochs
        
        print(f'perform_compute_non_PBE_epochs(..., training_data_portion={training_data_portion}, epochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}, frame_divide_bin_size: {frame_divide_bin_size})')


        long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()        
        try:
            available_MB: int = MemoryManagement.get_available_system_memory_MB() # Get available memory in MegaBytes
            available_GB: int = available_MB / 1024  # Gigabytes
            print(f'available RAM: {available_GB:.2f} GB')

            # Estimate memory requirements
            # active_sess = owning_pipeline_reference.sess
            active_sess = owning_pipeline_reference.filtered_sessions[global_epoch_name]
            
            def _subfn_perform_estimate_required_memory(specific_n_flattened_position_bins: int, n_dim_str: str):
                """ captures: epochs_decoding_time_bin_size, frame_divide_bin_size, active_sess
                """
                required_memory_bytes, itemized_mem_breakdown = estimate_memory_requirements_bytes(epochs_decoding_time_bin_size=epochs_decoding_time_bin_size, frame_divide_bin_size=frame_divide_bin_size, n_flattened_position_bins=specific_n_flattened_position_bins, n_neurons=active_sess.neurons.n_neurons, session_duration=active_sess.duration)
                required_memory_GB: int = int(required_memory_bytes) / int(1e9) # Gigabytes
                itemized_mem_breakdown = {f"{k}_{n_dim_str}":v for k, v in itemized_mem_breakdown.items()}
                return required_memory_GB, itemized_mem_breakdown

            itemized_mem_breakdown = {}
            total_required_memory_GB: int = 0
            if compute_1D:
                n_flattened_position_bins_1D: int = computation_results[global_epoch_name].computed_data.pf1D.n_flattened_position_bins
                required_memory_GB_1D, breakdown1D = _subfn_perform_estimate_required_memory(specific_n_flattened_position_bins=n_flattened_position_bins_1D, n_dim_str="1D")
                itemized_mem_breakdown.update(breakdown1D)
                total_required_memory_GB += int(required_memory_GB_1D)
            if compute_2D:
                n_flattened_position_bins_2D: int = computation_results[global_epoch_name].computed_data.pf2D.n_flattened_position_bins
                required_memory_GB_2D, breakdown2D = _subfn_perform_estimate_required_memory(specific_n_flattened_position_bins=n_flattened_position_bins_2D, n_dim_str="2D")
                itemized_mem_breakdown.update(breakdown2D)
                total_required_memory_GB += int(required_memory_GB_2D)
            
            itemized_mem_breakdown_GB = {k:int(v)/int(1e9) for k, v in itemized_mem_breakdown.items()}
            if debug_print_memory_breakdown:
                    print("Memory breakdown (GB):")
                    for k, v in itemized_mem_breakdown_GB.items():
                        print(f"\t{k}: {v:.3f}")
                
            print(f"Total memory required: {total_required_memory_GB:.2f} GB")
            if total_required_memory_GB > available_GB:
                print("Memory breakdown (GB):")
                for k, v in itemized_mem_breakdown_GB.items():
                    print(f"\t{k}: {v:.3f}")
                        
                raise MemoryError(f"Estimated Insufficient Memory: Operation would require {total_required_memory_GB:.2f} GB (have {available_GB:.2f} GB available.")
                # return global_computation_results
                
            # ==================================================================================================================== #
            # Proceed with computation                                                                                             #
            # ==================================================================================================================== #
            if include_includelist is not None:
                print(f'WARN: perform_compute_non_PBE_epochs(...): include_includelist: {include_includelist} is specified but include_includelist is currently ignored! Continuing with defaults.')

            if drop_previous_result_and_compute_fresh:
                removed_epoch_computations_result = global_computation_results.computed_data.pop('EpochComputations', None)
                if removed_epoch_computations_result is not None:
                    print(f'removed previous "EpochComputations" result and computing fresh since `drop_previous_result_and_compute_fresh == True`')

            if ('EpochComputations' not in global_computation_results.computed_data) or (not hasattr(global_computation_results.computed_data, 'EpochComputations')):
                # initialize
                global_computation_results.computed_data['EpochComputations'] = EpochComputationsComputationsContainer(training_data_portion=training_data_portion, epochs_decoding_time_bin_size=epochs_decoding_time_bin_size, frame_divide_bin_size=frame_divide_bin_size,
                                                                                                                    a_new_NonPBE_Epochs_obj=None, results1D=None, results2D=None, is_global=True)

            else:
                ## get and update existing:
                global_computation_results.computed_data['EpochComputations'].training_data_portion = training_data_portion
                global_computation_results.computed_data['EpochComputations'].epochs_decoding_time_bin_size = epochs_decoding_time_bin_size
                global_computation_results.computed_data['EpochComputations'].frame_divide_bin_size = frame_divide_bin_size
                global_computation_results.computed_data['EpochComputations'].a_new_NonPBE_Epochs_obj = None ## cheap to recompute
                

            # global_computation_results.computed_data['EpochComputations'].included_qclu_values = included_qclu_values
            if (not hasattr(global_computation_results.computed_data['EpochComputations'], 'a_new_NonPBE_Epochs_obj') or (global_computation_results.computed_data['EpochComputations'].a_new_NonPBE_Epochs_obj is None)):
                # initialize a new result
                a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = Compute_NonPBE_Epochs.init_from_pipeline(curr_active_pipeline=owning_pipeline_reference, training_data_portion=training_data_portion, skip_training_test_split=skip_training_test_split)
                global_computation_results.computed_data['EpochComputations'].a_new_NonPBE_Epochs_obj = a_new_NonPBE_Epochs_obj
            else:
                ## get the existing one:
                a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = global_computation_results.computed_data['EpochComputations'].a_new_NonPBE_Epochs_obj
            
            ## apply the new epochs to the session:
            owning_pipeline_reference.filtered_sessions[global_epoch_name].non_PBE = deepcopy(a_new_NonPBE_Epochs_obj.global_epoch_only_non_PBE_epoch_df) ## Only adds to global_epoch? Not even .sess?

            results1D, results2D = a_new_NonPBE_Epochs_obj.compute_all(owning_pipeline_reference, epochs_decoding_time_bin_size=epochs_decoding_time_bin_size, frame_divide_bin_size=frame_divide_bin_size, compute_1D=compute_1D, compute_2D=compute_2D, skip_training_test_split=skip_training_test_split)
            if (results1D is not None) and compute_1D:
                global_computation_results.computed_data['EpochComputations'].results1D = results1D

            if (results2D is not None) and compute_2D:
                global_computation_results.computed_data['EpochComputations'].results2D = results2D

            global_computation_results.computed_data['EpochComputations'].a_new_NonPBE_Epochs_obj = a_new_NonPBE_Epochs_obj
            

            # ==================================================================================================================== #
            # 2025-03-09 - Compute the Decoded Marginals for the known epochs (laps, ripples, etc)                                 #
            # ==================================================================================================================== #
            ## Common/shared for all decoded epochs:
            unique_decoder_names = ['long', 'short']
            non_PBE_all_directional_pf1D_Decoder, pseudo2D_continuous_specific_decoded_result, continuous_decoded_results_dict, non_PBE_marginal_over_track_ID, (time_bin_containers, time_window_centers, track_marginal_posterior_df) = global_computation_results.computed_data['EpochComputations']._build_merged_joint_placefields_and_decode(spikes_df=deepcopy(get_proper_global_spikes_df(owning_pipeline_reference))) # , filter_epochs=deepcopy(global_any_laps_epochs_obj)

            global_session = owning_pipeline_reference.filtered_sessions[global_epoch_name]
            
            ## from dict of filter_epochs to decode:
            global_replays_df: pd.DataFrame = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(global_session.replay))
            global_any_laps_epochs_obj = owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs # global_session.get
            # filter_epochs_to_decode_dict: Dict[KnownNamedDecodingEpochsType, Epoch] = {'laps': ensure_Epoch(deepcopy(global_any_laps_epochs_obj)),
            #                                                                         'pbe': ensure_Epoch(deepcopy(global_session.pbe.get_non_overlapping())),
            #                                 #  'ripple': ensure_Epoch(deepcopy(global_session.ripple)),
            #                                 #   'replay': ensure_Epoch(deepcopy(global_replays_df)),
            #                                 'non_pbe': ensure_Epoch(deepcopy(global_session.non_pbe)),
            #                                 }            
            filter_epochs_to_decode_dict: Dict[KnownNamedDecodingEpochsType, Epoch] = {'laps': ensure_Epoch(deepcopy(global_any_laps_epochs_obj)),
                                                                                    **{k:ensure_Epoch(deepcopy(v.get_non_overlapping())) for k, v in global_session.to_dict().items() if k in ('pbe', 'non_pbe')},
                                            }
            
            ## constrain all epochs to be at least two decoding time bins long, or drop them entirely:
            filter_epochs_to_decode_dict = {k:_compute_proper_filter_epochs(epochs_df=v, desired_decoding_time_bin_size=epochs_decoding_time_bin_size, minimum_event_duration=(2.0 * epochs_decoding_time_bin_size), mode=EpochFilteringMode.DropShorter)[0] for k, v in filter_epochs_to_decode_dict.items()} # `[0]` gets just the dataframe, as in DropShorter mode the time_bin_size is unchanged

            ## Perform the decoding and masking as needed for invalid bins:
            session_name: str = owning_pipeline_reference.session_name
            t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()

            # filter_epochs_pseudo2D_continuous_specific_decoded_result, filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict
            a_general_decoder_dict_decoded_epochs_dict_result: GeneralDecoderDictDecodedEpochsDictResult = EpochComputationsComputationsContainer._build_output_decoded_posteriors(non_PBE_all_directional_pf1D_Decoder=non_PBE_all_directional_pf1D_Decoder, # pseudo2D_continuous_specific_decoded_result=pseudo2D_continuous_specific_decoded_result,
                filter_epochs_to_decode_dict=filter_epochs_to_decode_dict,
                unique_decoder_names=unique_decoder_names, spikes_df=deepcopy(get_proper_global_spikes_df(owning_pipeline_reference)), epochs_decoding_time_bin_size=epochs_decoding_time_bin_size,
                session_name=session_name, t_start=t_start, t_delta=t_delta, t_end=t_end,
            )
            
            ## update the result object, adding the decoded result if needed
            global_computation_results.computed_data['EpochComputations'].a_general_decoder_dict_decoded_epochs_dict_result = a_general_decoder_dict_decoded_epochs_dict_result

            ## OUTPUTS: filter_epochs_pseudo2D_continuous_specific_decoded_result, filter_epochs_decoded_filter_epoch_track_marginal_posterior_df_dict
            # 58sec

        except MemoryError as mem_err:
            print(f"Insufficient memory: {str(mem_err)}")
            raise 
            # raise MemoryError(f"Insufficient memory: {str(mem_err)}")
        
        except Exception as e:
            raise
            # return None, f"Computation failed: {str(e)}"

        
        return global_computation_results
    


    @function_attributes(short_name='generalized_specific_epochs_decoding', tags=['BasePositionDecoder', 'computation', 'decoder', 'epoch'],
                          input_requires=[], output_provides=[],
                          requires_global_keys=['EpochComputations'], provides_global_keys=[], # 'EpochComputations'
                          uses=['GeneralizedDecodedEpochsComputationsContainer', 'GenericDecoderDictDecodedEpochsDictResult', 'GenericDecoderDictDecodedEpochsDictResult.batch_user_compute_fn'], used_by=[], creation_date='2025-04-14 12:40',
        validate_computation_test=validate_has_generalized_specific_epochs_decoding, is_global=True)
    def perform_generalized_specific_epochs_decoding(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, epochs_decoding_time_bin_size: float = 0.050, drop_previous_result_and_compute_fresh:bool=False, force_recompute:bool=False):
        """ Computes the most-general epoch decoding imaginable, creating several dictionaries of IdentifyingContext objects that identify the parameters undewr which decoding was performed.


        Refactored from `generalized_decode_epochs_dict_and_export_results_completion_function`


        Usage:
            from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import EpochComputationFunctions, EpochComputationsComputationsContainer

            valid_EpochComputations_result: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['EpochComputations']
            a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result


        """
        from neuropy.core.epoch import EpochsAccessor, Epoch, ensure_dataframe, ensure_Epoch, TimeColumnAliasesProtocol
        from pyphocorehelpers.print_helpers import get_now_rounded_time_str
        from neuropy.utils.result_context import DisplaySpecifyingIdentifyingContext
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import EpochFilteringMode, _compute_proper_filter_epochs
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, SingleEpochDecodedResult
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsResult, TrackTemplates, DecoderDecodedEpochsResult
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrainTestSplitResult, TrainTestLapsSplitting, CustomDecodeEpochsResult, decoder_name, epoch_split_key, get_proper_global_spikes_df, DirectionalPseudo2DDecodersResult
        from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult #, KnownNamedDecoderTrainedComputeEpochsType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType, DataTimeGrain, GenericResultTupleIndexType
        

        # ==================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                  #
        # ==================================================================================================================== #

        # ==================================================================================================================== #
        # New 2025-03-11 Generic Result:                                                                                       #
        # ==================================================================================================================== #

        ## Unpack from pipeline:
        valid_EpochComputations_result: EpochComputationsComputationsContainer = global_computation_results.computed_data['EpochComputations'] # owning_pipeline_reference.global_computation_results.computed_data['EpochComputations']
        assert valid_EpochComputations_result is not None
        epochs_decoding_time_bin_size: float = valid_EpochComputations_result.epochs_decoding_time_bin_size ## just get the standard size. Currently assuming all things are the same size!
        print(f'\tepochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}')
        assert epochs_decoding_time_bin_size == valid_EpochComputations_result.epochs_decoding_time_bin_size, f"\tERROR: nonPBE_results.epochs_decoding_time_bin_size: {valid_EpochComputations_result.epochs_decoding_time_bin_size} != epochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}"

        # ==================================================================================================================== #
        # Create and add the output                                                                                            #
        # ==================================================================================================================== #        
        if drop_previous_result_and_compute_fresh:            
            removed_epoch_computations_result = getattr(valid_EpochComputations_result, 'a_generic_decoder_dict_decoded_epochs_dict_result', None)
            valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result = None # set to None to drop the result
            if removed_epoch_computations_result is not None:
                print(f'removed previous "EpochComputations.a_generic_decoder_dict_decoded_epochs_dict_result" result and computing fresh since `drop_previous_result_and_compute_fresh == True`')

        if (not hasattr(valid_EpochComputations_result, 'a_generic_decoder_dict_decoded_epochs_dict_result')) or (getattr(valid_EpochComputations_result, 'a_generic_decoder_dict_decoded_epochs_dict_result', None) is None):
            # initialize
            a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = GenericDecoderDictDecodedEpochsDictResult.batch_user_compute_fn(curr_active_pipeline=owning_pipeline_reference, force_recompute=force_recompute, time_bin_size=epochs_decoding_time_bin_size, debug_print=debug_print)
            valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result = a_new_fully_generic_result
            global_computation_results.computed_data['EpochComputations'].a_generic_decoder_dict_decoded_epochs_dict_result = a_new_fully_generic_result

        else:
            ## get and update existing:
            a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result ## get existing
            ## TODO: update as needed here
            print(f'WARN 2025-04-14 - Not yet finished -- perform update here.')
            global_computation_results.computed_data['EpochComputations'].a_generic_decoder_dict_decoded_epochs_dict_result = a_new_fully_generic_result
            

        # ==================================================================================================================== #
        # Compute `TimeBinAggregation` results                                                                                 #
        # ==================================================================================================================== #
        from neuropy.analyses.time_bin_aggregation import TimeBinAggregation

        if not a_new_fully_generic_result.has_all_per_epoch_aggregations():
            print(f'\tMissing TimeBinAggregation computations. Recomputing...')
            # ==================================================================================================================================================================================================================================================================================== #
            # Phase 5 - Get the corrected 'per_epoch' results from the 'per_time_bin' versions                                                                                                                                                                                                     #
            # ==================================================================================================================================================================================================================================================================================== #
            ## get all non-global, `data_grain= 'per_time_bin'`
            flat_context_list, flat_result_context_dict, flat_decoder_context_dict, flat_decoded_marginal_posterior_df_context_dict = a_new_fully_generic_result.get_results_matching_contexts(context_query=IdentifyingContext(trained_compute_epochs='laps', decoder_identifier='pseudo2D',
                                                                                                                                                                                                                                time_bin_size=epochs_decoding_time_bin_size,
                                                                                                                                                                                                                                known_named_decoding_epochs_type=['pbe', 'laps', 'non_pbe'],
                                                                                                                                                                                                                                masked_time_bin_fill_type=('ignore', 'dropped'), data_grain= 'per_time_bin'))        

            _newly_updated_values_tuple = a_new_fully_generic_result.compute_all_per_epoch_aggregations_from_per_time_bin_results(flat_decoded_marginal_posterior_df_context_dict=flat_decoded_marginal_posterior_df_context_dict)
            # per_time_bin_to_per_epoch_context_map_dict, flat_decoded_marginal_posterior_df_per_epoch_marginals_df_context_dict, flat_decoded_marginal_posterior_df_per_time_bin_marginals_df_context_dict = _newly_updated_values_tuple

        return global_computation_results





from pyphocorehelpers.plotting.hairy_lines_plot import _perform_plot_hairy_overlayed_position

# ==================================================================================================================== #
# Display Functions/Plotting                                                                                           #
# ==================================================================================================================== #

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
import pyqtgraph as pg
import pyqtgraph.exporters
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import FileOutputManager, export_pyqtgraph_plot
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer # for context_nested_docks/single_context_nested_docks

from neuropy.utils.indexing_helpers import paired_incremental_sorting, union_of_arrays, intersection_of_arrays
import plotly.express as px

class EpochComputationDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ RankOrderGlobalDisplayFunctions
    These display functions compare results across several contexts.
    Must have a signature of: (owning_pipeline_reference, global_computation_results, computation_results, active_configs, ..., **kwargs) at a minimum
    """

    @function_attributes(short_name='generalized_decoded_yellow_blue_marginal_epochs', tags=['yellow-blue-plots', 'directional_merged_decoder_decoded_epochs', 'directional'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], requires_global_keys=["global_computation_results.computed_data['EpochComputations']"], uses=['plot_1D_most_likely_position_comparsions', 'FigureCollector'], used_by=[], creation_date='2025-04-16 05:49', related_items=[], is_global=True)
    def _display_generalized_decoded_yellow_blue_marginal_epochs(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True,
                                                    single_plot_fixed_height=50.0, size=(35, 3), dpi=100, constrained_layout=True, override_fig_man: Optional[FileOutputManager]=None, **kwargs):
            """ Displays one figure containing the track_ID marginal, decoded continuously over the entire recording session along with the animal's position.
            
            
            Based off of ``
            
            Usage:
                # getting `_display_generalized_decoded_yellow_blue_marginal_epochs` into shape
                curr_active_pipeline.reload_default_display_functions()


                _out = dict()
                _out['_display_generalized_decoded_yellow_blue_marginal_epochs'] = curr_active_pipeline.display(display_function='_display_generalized_decoded_yellow_blue_marginal_epochs', active_session_configuration_context=None) # _display_directional_track_template_pf1Ds


            """
            from neuropy.utils.result_context import IdentifyingContext
            from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol
            # from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices
            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
            from neuropy.utils.matplotlib_helpers import get_heatmap_cmap
            from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
            from pyphoplacecellanalysis.General.Mixins.ExportHelpers import FileOutputManager, FigureOutputLocation, ContextToPathMode	
            

            import matplotlib as mpl
            import matplotlib.pyplot as plt
            from flexitext import flexitext ## flexitext for formatted matplotlib text

            from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
            from neuropy.utils.matplotlib_helpers import FormattedFigureText
            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PhoPublicationFigureHelper

            ## Unpack from pipeline:
            valid_EpochComputations_result: EpochComputationsComputationsContainer = owning_pipeline_reference.global_computation_results.computed_data['EpochComputations'] # owning_pipeline_reference.global_computation_results.computed_data['EpochComputations']
            assert valid_EpochComputations_result is not None
            epochs_decoding_time_bin_size: float = valid_EpochComputations_result.epochs_decoding_time_bin_size ## just get the standard size. Currently assuming all things are the same size!
            print(f'\tepochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}')
            assert epochs_decoding_time_bin_size == valid_EpochComputations_result.epochs_decoding_time_bin_size, f"\tERROR: nonPBE_results.epochs_decoding_time_bin_size: {valid_EpochComputations_result.epochs_decoding_time_bin_size} != epochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}"
            a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result ## get existing

            ## INPUTS: a_new_fully_generic_result
            # a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='nan_filled', data_grain='per_time_bin')
            a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='ignore', data_grain='per_time_bin')
            best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context, debug_print=False)
            epochs_decoding_time_bin_size = best_matching_context.get('time_bin_size', None)
            assert epochs_decoding_time_bin_size is not None

            ## OUTPUTS: a_decoded_marginal_posterior_df

            complete_session_context, (session_context, additional_session_context) = owning_pipeline_reference.get_complete_session_context()

            active_context = kwargs.pop('active_context', None)
            if active_context is not None:
                # Update the existing context:
                display_context = active_context.adding_context('display_fn', display_fn_name='generalized_decoded_yellow_blue_marginal_epochs')
            else:
                active_context = owning_pipeline_reference.sess.get_context()
                # Build the active context directly:
                display_context = owning_pipeline_reference.build_display_context_for_session('generalized_decoded_yellow_blue_marginal_epochs')

            fignum = kwargs.pop('fignum', None)
            if fignum is not None:
                print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')

            # defer_render = kwargs.pop('defer_render', False)
            # debug_print: bool = kwargs.pop('debug_print', False)
            # active_config_name: bool = kwargs.pop('active_config_name', None)

            # perform_write_to_file_callback = kwargs.pop('perform_write_to_file_callback', (lambda final_context, fig: owning_pipeline_reference.output_figure(final_context, fig)))

            long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()
            # long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
            long_session, short_session, global_session = [owning_pipeline_reference.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]

            # ==================================================================================================================================================================================================================================================================================== #
            # Start Building Figure                                                                                                                                                                                                                                                                #
            # ==================================================================================================================================================================================================================================================================================== #


            owning_pipeline_reference.reload_default_display_functions()


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
                an_ax.set_xticks([])  # Remove tick marks
                an_ax.set_yticks([])
                an_ax.set_title('') ## remove title
                

            # ==================================================================================================================================================================================================================================================================================== #
            # Begin Function Body                                                                                                                                                                                                                                                                  #
            # ==================================================================================================================================================================================================================================================================================== #

            graphics_output_dict = {}

            # Shared active_decoder, global_session:
            global_session = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name]) # used for validate_lap_dir_estimations(...) 


            active_context = kwargs.pop('active_context', owning_pipeline_reference.sess.get_context())
            if active_context is not None:
                    display_context = active_context.adding_context('display_fn', display_fn_name='generalized_decoded_yellow_blue_marginal_epochs')

            if override_fig_man is not None:
                print(f'override_fig_man is not None! Custom output path will be used!')
                test_display_output_path = override_fig_man.get_figure_save_file_path(display_context, make_folder_if_needed=False)
                print(f'\ttest_display_output_path: "{test_display_output_path}"')
    

            def _perform_write_to_file_callback(final_context, fig):
                """ captures: override_fig_man """
                if save_figure:
                    return owning_pipeline_reference.output_figure(final_context, fig, override_fig_man=override_fig_man)
                else:
                    pass # do nothing, don't save
                

            with mpl.rc_context(PhoPublicationFigureHelper.rc_context_kwargs({'figure.dpi': str(dpi), 'figure.constrained_layout.use': (constrained_layout or False), 'figure.frameon': False, 'figure.figsize': size, })): # 'figure.figsize': (12.4, 4.8), 
                # Create a FigureCollector instance
                with FigureCollector(name='generalized_decoded_yellow_blue_marginal_epochs', base_context=display_context) as collector:
                    fig, ax_dict = collector.subplot_mosaic(
                        [
                            ["ax_top"],
                            ["ax_decodedMarginal_P_Short_v_time"],
                            # ["ax_position_and_laps_v_time"],
                        ],
                        # set the width ratios between the columns
                        height_ratios=[3, 1],
                        sharex=True,
                        gridspec_kw=dict(wspace=0, hspace=0) # `wspace=0`` is responsible for sticking the pf and the activity axes together with no spacing
                    )


                    # fig = None
                    an_ax = ax_dict["ax_decodedMarginal_P_Short_v_time"] ## no figure (should I be using collector??)

                    # decoded posterior overlay __________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
                    variable_name: str = ''
                    y_bin_labels = ['long', 'short']
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
                    
                    label_artists_dict = PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels(an_ax, y_bin_labels=y_bin_labels, enable_draw_decoder_colored_lines=False, should_use_outer_labels=False,
                                                                                                                # additional_label_kwargs = dict(fontsize=12, fontweight='bold'),
                                                                                                                )
                    _subfn_clean_axes_decorations(an_ax=ax_dict["ax_decodedMarginal_P_Short_v_time"])
                    # an_ax.set_ylabel('marginal long/short')

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
                    # an_ax.set_xlabel('')

                    # ==================================================================================================================================================================================================================================================================================== #
                    # Titles/Formatting/Marginas and Saving                                                                                                                                                                                                                                                #
                    # ==================================================================================================================================================================================================================================================================================== #
                    active_config = deepcopy(a_decoder.pf.config)

                    subtitle_string = active_config.str_for_display(is_2D=False) # , normal_to_extras_line_sep=","
                    # print(f'subtitle_string: {subtitle_string}')

                    ## BUild figure titles:
                    # INPUTS: main_fig
                    fig.suptitle('')
                    # text_formatter = FormattedFigureText() # .init_from_margins(left_margin=0.01)
                    # text_formatter.setup_margins(fig, left_margin=0.01) # , left_margin=0.1
                    text_formatter = FormattedFigureText.init_from_margins(left_margin=0.01, right_margin=0.99) # , top_margin=0.9
                    # text_formatter.setup_margins(fig, left_margin=0.01, top_margin=0.9)
                    text_formatter.setup_margins(fig)
                    title_string: str = f"generalized_decoded_yellow_blue_marginal_epochs"
                    # session_footer_string: str =  active_context.get_description(subset_includelist=['format_name', 'animal', 'exper_name', 'session_name'], separator=' | ') 
                    session_footer_string: str =  active_context.get_description(separator=' | ') 

                    # subtitle_string = '\n'.join([f'{active_config.str_for_display(is_2D)}'])
                    # header_text_obj = flexitext(text_formatter.left_margin, 0.9, f'<size:22><weight:bold>{title_string}</></>\n<size:10>{subtitle_string}</>', va="bottom", xycoords="figure fraction") # , wrap=False
                    header_text_obj = flexitext(0.01, 0.85, f'<size:20><weight:bold>{title_string}</></>\n<size:9>{subtitle_string}</>', va="bottom", xycoords="figure fraction") # , wrap=False
                    footer_text_obj = text_formatter.add_flexitext_context_footer(active_context=active_context) # flexitext((text_formatter.left_margin*0.1), (text_formatter.bottom_margin*0.25), text_formatter._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")
                    
                    window_title_string: str = f"{title_string} - {session_footer_string}"
                    fig.canvas.manager.set_window_title(window_title_string) # sets the window's title
                    if ((_perform_write_to_file_callback is not None) and (display_context is not None)):
                        _perform_write_to_file_callback(display_context, fig)

                    graphics_output_dict['label_objects'] = {'header': header_text_obj, 'footer': footer_text_obj, 'formatter': text_formatter}
            ## END with mpl.rc_context({'figure.dpi': '...


            graphics_output_dict['collector'] = collector

            return graphics_output_dict




    @function_attributes(short_name='trackID_marginal_hairy_position', tags=['context-decoder-comparison', 'hairly-plot', 'decoded_position', 'directional'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], requires_global_keys=["global_computation_results.computed_data['EpochComputations']"], uses=['_perform_plot_hairy_overlayed_position', '_helper_add_interpolated_position_columns_to_decoded_result_df', '_display_grid_bin_bounds_validation', 'FigureCollector'], used_by=[], creation_date='2025-05-03 00:00', related_items=[], is_global=True)
    def _display_decoded_trackID_marginal_hairy_position(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True,
                                                    size=(35, 6), dpi=100, constrained_layout=True, override_fig_man: Optional[FileOutputManager]=None, extreme_threshold: float=0.8, opacity_max:float=0.7, thickness_ramping_multiplier:float=35.0, prob_to_thickness_ramping_function=None, disable_all_grid_bin_bounds_lines: bool = False,
                                                    a_var_name_to_color_map = {'P_Long': 'red', 'P_Short': 'blue'}, ax=None, **kwargs):
            """ Displays one figure containing the track_ID marginal, decoded continuously over the entire recording session along with the animal's position.
            
            
            #TODO 2025-05-05 15:02: - [ ] Increasing `extreme_threshold` should not have an effect on the thicknesses, only the masked/unmasked regions extreme_threshold=0.9, thickness_ramping_multiplier=50

                        
            Based off of ``
            
            Usage:
                # getting `_display_generalized_decoded_yellow_blue_marginal_epochs` into shape
                curr_active_pipeline.reload_default_display_functions()


                _out = dict()
                _out['trackID_marginal_hairy_position'] = curr_active_pipeline.display(display_function='trackID_marginal_hairy_position', active_session_configuration_context=None) # _display_directional_track_template_pf1Ds


            """
            from neuropy.utils.result_context import IdentifyingContext
            from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol
            from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
            from pyphoplacecellanalysis.General.Mixins.ExportHelpers import FileOutputManager, FigureOutputLocation, ContextToPathMode	
            
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            from flexitext import flexitext ## flexitext for formatted matplotlib text

            from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
            from neuropy.utils.matplotlib_helpers import FormattedFigureText

            from neuropy.utils.result_context import IdentifyingContext
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import _helper_add_interpolated_position_columns_to_decoded_result_df
            from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PhoPublicationFigureHelper

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



            export_dpi_multiplier: float = kwargs.pop('export_dpi_multiplier', 2.0)
            export_dpi: int = int(np.ceil(dpi * export_dpi_multiplier))
            
            # ==================================================================================================================================================================================================================================================================================== #
            # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
            # ==================================================================================================================================================================================================================================================================================== #

            ## Unpack from pipeline:
            valid_EpochComputations_result: EpochComputationsComputationsContainer = owning_pipeline_reference.global_computation_results.computed_data['EpochComputations'] # owning_pipeline_reference.global_computation_results.computed_data['EpochComputations']
            assert valid_EpochComputations_result is not None
            epochs_decoding_time_bin_size: float = valid_EpochComputations_result.epochs_decoding_time_bin_size ## just get the standard size. Currently assuming all things are the same size!
            print(f'\tepochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}')
            assert epochs_decoding_time_bin_size == valid_EpochComputations_result.epochs_decoding_time_bin_size, f"\tERROR: nonPBE_results.epochs_decoding_time_bin_size: {valid_EpochComputations_result.epochs_decoding_time_bin_size} != epochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}"
            a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result ## get existing

            ## INPUTS: a_new_fully_generic_result
            # a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='nan_filled', data_grain='per_time_bin')
            a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='ignore', data_grain='per_time_bin')
            best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context, debug_print=False)
            epochs_decoding_time_bin_size = best_matching_context.get('time_bin_size', None)
            assert epochs_decoding_time_bin_size is not None

            ## OUTPUTS: a_decoded_marginal_posterior_df

            complete_session_context, (session_context, additional_session_context) = owning_pipeline_reference.get_complete_session_context()

            active_context = kwargs.pop('active_context', None)
            if active_context is not None:
                # Update the existing context:
                display_context = active_context.adding_context('display_fn', display_fn_name='trackID_marginal_hairy_position')
            else:
                # active_context = owning_pipeline_reference.sess.get_context()
                active_context = deepcopy(complete_session_context) # owning_pipeline_reference.sess.get_context()
                
                # Build the active context directly:
                display_context = owning_pipeline_reference.build_display_context_for_session('trackID_marginal_hairy_position')

            fignum = kwargs.pop('fignum', None)
            if fignum is not None:
                print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')


            ## OUTPUTS: active_context, display_context
            active_display_context = display_context.overwriting_context(extreme_threshold=extreme_threshold, opacity_max=opacity_max, thickness_ramping_multiplier=thickness_ramping_multiplier) ## include any that are just the slightest big different
            # active_display_context = deepcopy(display_context)

            # defer_render = kwargs.pop('defer_render', False)
            # debug_print: bool = kwargs.pop('debug_print', False)
            # active_config_name: bool = kwargs.pop('active_config_name', None)

            # perform_write_to_file_callback = kwargs.pop('perform_write_to_file_callback', (lambda final_context, fig: owning_pipeline_reference.output_figure(final_context, fig)))

            global_measured_position_df: pd.DataFrame = deepcopy(owning_pipeline_reference.sess.position.to_dataframe())
            a_decoded_marginal_posterior_df: pd.DataFrame = _helper_add_interpolated_position_columns_to_decoded_result_df(a_result=a_result, a_decoder=a_decoder, a_decoded_marginal_posterior_df=a_decoded_marginal_posterior_df, global_measured_position_df=global_measured_position_df)


            # ==================================================================================================================================================================================================================================================================================== #
            # Start Building Figure                                                                                                                                                                                                                                                                #
            # ==================================================================================================================================================================================================================================================================================== #

            graphics_output_dict = {}

            if override_fig_man is not None:
                print(f'override_fig_man is not None! Custom output path will be used!')
                test_display_output_path = override_fig_man.get_figure_save_file_path(active_display_context, make_folder_if_needed=False)
                print(f'\ttest_display_output_path: "{test_display_output_path}"')
    

            def _perform_write_to_file_callback(final_context, fig):
                """ captures: override_fig_man, export_dpi """
                if save_figure:
                    return owning_pipeline_reference.output_figure(final_context, fig, override_fig_man=override_fig_man, dpi=export_dpi)
                else:
                    pass # do nothing, don't save
                

            with mpl.rc_context(PhoPublicationFigureHelper.rc_context_kwargs({'figure.dpi': str(dpi), 'figure.constrained_layout.use': (constrained_layout or False), 'figure.frameon': False, 'figure.figsize': size, })): # 'figure.figsize': (12.4, 4.8), 
                # Create a FigureCollector instance
                with FigureCollector(name='trackID_marginal_hairy_position', base_context=active_display_context) as collector:
                    # from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _perform_plot_hairy_overlayed_position

                    ## INPUTS: a_decoded_marginal_posterior_df

                    ## plot the basic lap-positions (measured) over time figure:
                    graphics_output_dict = owning_pipeline_reference.display(display_function='_display_grid_bin_bounds_validation', active_session_configuration_context=None, include_includelist=[], save_figure=False, ax=ax) # _display_grid_bin_bounds_validation
                    fig = graphics_output_dict.figures[0]
                    out_axes_list = graphics_output_dict.axes
                    out_plot_data = graphics_output_dict.plot_data

                    ## get the lines2D object to turn off the default position lines:
                    position_lines_2D = out_plot_data.get('position_lines_2D', None)
                    if position_lines_2D is not None:
                        ## hide all inactive lines:
                        for a_line in position_lines_2D:
                            a_line.set_visible(False)
                
                    if disable_all_grid_bin_bounds_lines:
                        out_plot_data = _subfn_hide_all_plot_lines(out_plot_data=out_plot_data)
                    
                    an_pos_line_artist, df_viz = _perform_plot_hairy_overlayed_position(df=deepcopy(a_decoded_marginal_posterior_df), ax=out_axes_list[0],
                                                                                                     extreme_threshold=extreme_threshold, opacity_max=opacity_max, thickness_ramping_multiplier=thickness_ramping_multiplier, prob_to_thickness_ramping_function=prob_to_thickness_ramping_function, a_var_name_to_color_map=a_var_name_to_color_map) # , thickness_ramping_multiplier=5
                    out_plot_data['an_pos_line_artist'] = an_pos_line_artist
                    out_plot_data['df_viz'] = df_viz
                    
                    collector.post_hoc_append(figs=graphics_output_dict.figures, axes=out_axes_list, contexts=[active_display_context])


                    # ==================================================================================================================================================================================================================================================================================== #
                    # Titles/Formatting/Marginas and Saving                                                                                                                                                                                                                                                #
                    # ==================================================================================================================================================================================================================================================================================== #
                    active_config = deepcopy(a_decoder.pf.config)

                    subtitle_string = active_config.str_for_display(is_2D=False) # , normal_to_extras_line_sep=","
                    subtitle_string = f"{subtitle_string} - only extreme context probs (P(Ctx) > {extreme_threshold}) are shown"

                    # print(f'subtitle_string: {subtitle_string}')

                    ## BUild figure titles:
                    # INPUTS: main_fig
                    fig.suptitle('')
                    out_axes_list[0].set_title('')
                    
                    # text_formatter = FormattedFigureText() # .init_from_margins(left_margin=0.01)
                    # text_formatter.setup_margins(fig, left_margin=0.01) # , left_margin=0.1
                    text_formatter = FormattedFigureText.init_from_margins(left_margin=0.01, right_margin=0.99) # , top_margin=0.9
                    # text_formatter.setup_margins(fig, left_margin=0.01, top_margin=0.9)
                    text_formatter.setup_margins(fig)
                    title_string: str = f"trackID_marginal_hairy_position"
                    # session_footer_string: str =  active_context.get_description(subset_includelist=['format_name', 'animal', 'exper_name', 'session_name'], separator=' | ') 
                    session_footer_string: str =  active_context.get_description(separator=' | ') 

                    # subtitle_string = '\n'.join([f'{active_config.str_for_display(is_2D)}'])
                    # header_text_obj = flexitext(text_formatter.left_margin, 0.9, f'<size:22><weight:bold>{title_string}</></>\n<size:10>{subtitle_string}</>', va="bottom", xycoords="figure fraction") # , wrap=False
                    header_text_obj = flexitext(0.01, 0.85, f'<size:20><weight:bold>{title_string}</></>\n<size:9>{subtitle_string}</>', va="bottom", xycoords="figure fraction") # , wrap=False
                    footer_text_obj = text_formatter.add_flexitext_context_footer(active_context=active_context) # flexitext((text_formatter.left_margin*0.1), (text_formatter.bottom_margin*0.25), text_formatter._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")
                    

                    complete_title_string: str = f"{title_string} - {session_footer_string}"
                    # complete_title_string: str = f"{complete_title_string} - {subtitle_string}"
                    
                    window_title_string: str = complete_title_string
                    
                    fig.canvas.manager.set_window_title(window_title_string) # sets the window's title
                    if ((_perform_write_to_file_callback is not None) and (active_display_context is not None)):
                        _perform_write_to_file_callback(active_display_context, fig)

                    graphics_output_dict['label_objects'] = {'header': header_text_obj, 'footer': footer_text_obj, 'formatter': text_formatter}
            ## END with mpl.rc_context({'figure.dpi': '...


            graphics_output_dict['collector'] = collector

            return graphics_output_dict


    @function_attributes(short_name='trackID_weighted_position_posterior', tags=['context-decoder-comparison', 'decoded_position', 'directional'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], requires_global_keys=["global_computation_results.computed_data['EpochComputations']"], uses=['FigureCollector'], used_by=[], creation_date='2025-05-03 00:00', related_items=[], is_global=True)
    def _display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, override_fig_man: Optional[FileOutputManager]=None, ax=None,
                                                                                    custom_export_formats: Optional[Dict[str, Any]]=None, parent_output_folder: Optional[Path] = None, time_bin_size: float=0.025, delete_previous_outputs_folder:bool=True, desired_height:int=1200, 
                                                                                    masked_time_bin_fill_type='ignore', enable_ripple_merged_export: bool = True, enable_laps_merged_export: bool = True, **kwargs):
            """ Exports individual posteriors to file in an overlayed manner
            
            
            "HeatmapExportConfig"
            
            #TODO 2025-05-05 15:02: - [ ] Increasing `extreme_threshold` should not have an effect on the thicknesses, only the masked/unmasked regions extreme_threshold=0.9, thickness_ramping_multiplier=50

                        
            Based off of ``
            
            Usage:
                from pyphoplacecellanalysis.Pho2D.data_exporting import HeatmapExportConfig, HeatmapExportKind

                            
                # getting `_display_generalized_decoded_yellow_blue_marginal_epochs` into shape
                curr_active_pipeline.reload_default_display_functions()


                _out = dict()
                _out['trackID_weighted_position_posterior'] = curr_active_pipeline.display(display_function='trackID_weighted_position_posterior', active_session_configuration_context=None) # _display_directional_track_template_pf1Ds

                ## Show output paths:
                graphics_output_dict = _out['trackID_weighted_position_posterior']

                out_paths: Dict[types.KnownNamedDecoderTrainedComputeEpochsType, Dict[types.DecoderName, Path]] = graphics_output_dict['out_paths']
                out_custom_formats_dict: Dict[types.KnownNamedDecodingEpochsType, Dict[types.DecoderName, Dict[str, List[HeatmapExportConfig]]]] = graphics_output_dict['out_custom_formats_dict']
                flat_merged_images = graphics_output_dict['flat_merged_images']
                flat_imgs_dict = graphics_output_dict['flat_imgs_dict']
            

                ## Handle just the paths:
                out_paths = deepcopy(_out_curr['out_paths'])
                for k, v_dict in out_paths.items():
                    for a_decoder_name, a_path in v_dict.items():
                        file_uri_from_path(a_path)
                        fullwidth_path_widget(a_path=a_path, file_name_label=f"{k}[{a_decoder_name}]:")

                        

                ## Handle Output configs/images:
                out_custom_formats_dict: Dict = _out['out_custom_formats_dict']
                # flat_imgs = []
                flat_merged_images = {}

                for a_series_name, v_dict in out_custom_formats_dict.items():
                    # a_series_name: ['laps', 'ripple']
                    for a_decoder_name, a_rendered_configs_dict in v_dict.items():
                        
                        for a_config_name, a_rendered_config_list in a_rendered_configs_dict.items():
                            # 'raw_rgba'
                            # print(a_rendered_config_list)
                            # len(a_rendered_config_list)
                            flat_imgs = []
                            
                            for i, a_config in enumerate(a_rendered_config_list):      
                                # posterior_save_path = a_config.posterior_saved_path
                                _posterior_image = a_config.posterior_saved_image
                                flat_imgs.append(_posterior_image)
                                
                                # print(F'a_rendered_config: {type(a_rendered_config)}')
                                # type(a_rendered_config_list[0])
                                # print(F'a_rendered_config: {list(a_rendered_config.keys())}')
                                # file_uri_from_path(a_path)
                                # fullwidth_path_widget(a_path=a_path, file_name_label=f"{a_series_name}[{a_decoder_name}]:")
                                # flat_img_out_paths.append(a_path)
                        
                            ## OUTPUTS: flat_imgs
                            _merged_img = horizontal_image_stack(flat_imgs, padding=10, separator_color='white')
                            flat_merged_images[a_series_name] = _merged_img
                            

                # flat_img_out_paths
                flat_merged_images


                

            """
            from neuropy.utils.result_context import IdentifyingContext
            from pyphoplacecellanalysis.Pho2D.data_exporting import HeatmapExportConfig, PosteriorExporting, HeatmapExportKind
            from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import FixedCustomColormaps
            from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import Assert
            from benedict import benedict
            from datetime import datetime, date, timedelta
            from pyphocorehelpers.print_helpers import get_now_rounded_time_str
            from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path

            from pyphocorehelpers.plotting.media_output_helpers import vertical_image_stack, horizontal_image_stack, image_grid # used in `_subfn_build_combined_output_images`
            from pyphocorehelpers.image_helpers import ImageHelpers
            from pyphocorehelpers.plotting.media_output_helpers import ImagePostRenderFunctionSets, ImageOperationsAndEffects
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecodedFilterEpochsResult, DirectionalPseudo2DDecodersResult


            # global_dropped_keys, local_dropped_keys = curr_active_pipeline.perform_drop_computed_result(computed_data_keys_to_drop = ['DirectionalDecodersDecoded'], debug_print=True)

            ## Does this not perform the required pre-req computations if they're missing? For example this function requires: `requires_global_keys=['DirectionalLaps', 'DirectionalMergedDecoders']`, so does it do those if they're missing, or not because they aren't in the computations list?
            # owning_pipeline_reference.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'],
            #                                       computation_kwargs_list=[{'time_bin_size': time_bin_size, 'should_disable_cache':False}], 
            #                                       enabled_filter_names=None, fail_on_exception=True, debug_print=False)
            

            owning_pipeline_reference.resolve_and_execute_full_required_computation_plan(computation_functions_name_includelist=['directional_decoders_decode_continuous'],
                                                  computation_kwargs_list=[{'time_bin_size': time_bin_size, 'should_disable_cache':False}], 
                                                  enabled_filter_names=None, fail_on_exception=True, debug_print=False)


            DAY_DATE_STR: str = date.today().strftime("%Y-%m-%d")
            DAY_DATE_TO_USE = f'{DAY_DATE_STR}' # used for filenames throught the notebook
            print(f'DAY_DATE_STR: {DAY_DATE_STR}, DAY_DATE_TO_USE: {DAY_DATE_TO_USE}')

            NOW_DATETIME: str = get_now_rounded_time_str()
            NOW_DATETIME_TO_USE = f'{NOW_DATETIME}' # used for filenames throught the notebook
            print(f'NOW_DATETIME: {NOW_DATETIME}, NOW_DATETIME_TO_USE: {NOW_DATETIME_TO_USE}')

            # export_dpi_multiplier: float = kwargs.pop('export_dpi_multiplier', 2.0)
            # dpi = kwargs.pop('dpi', 100)
            # export_dpi: int = int(np.ceil(dpi * export_dpi_multiplier))

            # ==================================================================================================================================================================================================================================================================================== #
            # Build outputs:                                                                                                                                                                                                                                                              #
            # ==================================================================================================================================================================================================================================================================================== #

            graphics_output_dict = {'parent_output_folder': parent_output_folder, 'time_bin_size': time_bin_size}

            Assert.path_exists(parent_output_folder)

            # ==================================================================================================================================================================================================================================================================================== #
            # BEGIN FUNCTION BODY                                                                                                                                                                                                                                                                  #
            # ==================================================================================================================================================================================================================================================================================== #

            
            ## Unpack from pipeline:
            valid_EpochComputations_result: EpochComputationsComputationsContainer = owning_pipeline_reference.global_computation_results.computed_data['EpochComputations'] # owning_pipeline_reference.global_computation_results.computed_data['EpochComputations']
            assert valid_EpochComputations_result is not None
            epochs_decoding_time_bin_size: float = valid_EpochComputations_result.epochs_decoding_time_bin_size ## just get the standard size. Currently assuming all things are the same size!
            print(f'\tepochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}')
            assert epochs_decoding_time_bin_size == valid_EpochComputations_result.epochs_decoding_time_bin_size, f"\tERROR: nonPBE_results.epochs_decoding_time_bin_size: {valid_EpochComputations_result.epochs_decoding_time_bin_size} != epochs_decoding_time_bin_size: {epochs_decoding_time_bin_size}"
            a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result ## get existing
            
            if a_new_fully_generic_result is None:
                ## need to recompute 'generalized_specific_epochs_decoding'
                owning_pipeline_reference.perform_specific_computation(computation_functions_name_includelist=['generalized_specific_epochs_decoding'],
                                        computation_kwargs_list=[{'epochs_decoding_time_bin_size': time_bin_size, 'drop_previous_result_and_compute_fresh':False, 'force_recompute': False}], 
                                        enabled_filter_names=None, fail_on_exception=True, debug_print=False)

                a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result ## get existing

            
            assert a_new_fully_generic_result is not None, f"Missing valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result (valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result == None)"

            ## INPUTS: a_new_fully_generic_result
            # a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='nan_filled', data_grain='per_time_bin')
            a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='ignore', data_grain='per_time_bin')
            best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context, debug_print=False)
            epochs_decoding_time_bin_size = best_matching_context.get('time_bin_size', None)
            assert epochs_decoding_time_bin_size is not None

            ## OUTPUTS: a_decoded_marginal_posterior_df

            complete_session_context, (session_context, additional_session_context) = owning_pipeline_reference.get_complete_session_context()

            active_context = kwargs.pop('active_context', None)
            if active_context is not None:
                # Update the existing context:
                active_context = deepcopy(active_context).overwriting_context(time_bin_size=time_bin_size)
                display_context = active_context.adding_context('display_fn', display_fn_name='trackID_weighted_position_posterior', time_bin_size=time_bin_size)
            else:
                # active_context = owning_pipeline_reference.sess.get_context()
                active_context = deepcopy(complete_session_context).overwriting_context(time_bin_size=time_bin_size) # owning_pipeline_reference.sess.get_context()
                
                # Build the active context directly:
                display_context = deepcopy(active_context).overwriting_context(display_fn_name='trackID_weighted_position_posterior', time_bin_size=time_bin_size)
                # display_context = owning_pipeline_reference.build_display_context_for_session('trackID_weighted_position_posterior', time_bin_size=time_bin_size)

            fignum = kwargs.pop('fignum', None)
            if fignum is not None:
                print(f'WARNING: fignum will be ignored but it was specified as fignum="{fignum}"!')


            ## OUTPUTS: active_context, display_context
            # active_display_context = display_context.overwriting_context(extreme_threshold=extreme_threshold, opacity_max=opacity_max, thickness_ramping_multiplier=thickness_ramping_multiplier) ## include any that are just the slightest big different
            # active_display_context = deepcopy(display_context)
            

            needs_discover_default_collected_outputs_dir: bool = True
            if parent_output_folder is not None:
                if isinstance(parent_output_folder, str):
                    parent_output_folder = Path(parent_output_folder).resolve()
                    if parent_output_folder.exists():
                        needs_discover_default_collected_outputs_dir = False # we're good, the provided dir exists

            if needs_discover_default_collected_outputs_dir:
                    ## if none is provided it tries to find one in collected_outputs
                    known_collected_outputs_paths = [Path(v).resolve() for v in ['/Users/pho/data/collected_outputs',
                                                                                '/Volumes/SwapSSD/Data/collected_outputs', r"K:/scratch/collected_outputs", '/Users/pho/Dropbox (University of Michigan)/MED-DibaLabDropbox/Data/Pho/Outputs/output/collected_outputs', r'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs',
                                                                                '/home/halechr/FastData/collected_outputs/', '/home/halechr/cloud/turbo/Data/Output/collected_outputs']]
                    collected_outputs_directory = find_first_extant_path(known_collected_outputs_paths)
                    assert collected_outputs_directory.exists(), f"collected_outputs_directory: {collected_outputs_directory} does not exist! Is the right computer's config commented out above?"
                    # fullwidth_path_widget(scripts_output_path, file_name_label='Scripts Output Path:')
                    print(f'collected_outputs_directory: "{collected_outputs_directory}"')
                    # Create a 'figures' subfolder if it doesn't exist
                    figures_folder: Path = collected_outputs_directory.joinpath('figures', '_temp_individual_posteriors').resolve()
                    figures_folder.mkdir(parents=False, exist_ok=True)
                    assert figures_folder.exists()
                    print(f'\tfigures_folder: "{figures_folder}"')
                    ## this is good
                    parent_output_folder = figures_folder


            Assert.path_exists(parent_output_folder)
            posterior_out_folder = parent_output_folder.joinpath(DAY_DATE_TO_USE).resolve()
            posterior_out_folder.mkdir(parents=True, exist_ok=True)
            save_path = posterior_out_folder.resolve()

            _parent_save_context: IdentifyingContext = owning_pipeline_reference.build_display_context_for_session('trackID_weighted_position_posterior') ## why is this done?
            _specific_session_output_folder = save_path.joinpath(active_context.get_description(subset_excludelist=['format_name', 'display_fn_name', 'time_bin_size'])).resolve()

            _specific_session_output_folder.mkdir(parents=True, exist_ok=True)
            print(f'\tspecific_session_output_folder: "{_specific_session_output_folder}"')

            ## OUTPUTS: _parent_save_context, _specific_session_output_folder
            graphics_output_dict['parent_output_folder'] = parent_output_folder
            graphics_output_dict['parent_save_context'] = _parent_save_context
            graphics_output_dict['parent_specific_session_output_folder'] = _specific_session_output_folder

            # INPUTS _____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #


            # ==================================================================================================================================================================================================================================================================================== #
            # Separate export for each masked_time_bin_fill_type  - LAPS                                                                                                                                                                                                                           #
            # ==================================================================================================================================================================================================================================================================================== #


            try:
                laps_trained_decoder_search_context = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='laps', masked_time_bin_fill_type=('ignore', 'nan_filled', 'dropped'), data_grain='per_time_bin') # , data_grain= 'per_time_bin -- not really relevant: ['masked_time_bin_fill_type', 'known_named_decoding_epochs_type', 'data_grain']
                # laps_trained_decoder_search_context = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='laps', masked_time_bin_fill_type='dropped', data_grain='per_time_bin')
                flat_context_list, flat_result_context_dict, flat_decoder_context_dict, flat_decoded_marginal_posterior_df_context_dict = a_new_fully_generic_result.get_results_matching_contexts(context_query=laps_trained_decoder_search_context, return_multiple_matches=True, debug_print=True)

                active_ctxts = [
                                IdentifyingContext(trained_compute_epochs= 'laps', pfND_ndim= 1, decoder_identifier= 'pseudo2D', time_bin_size= epochs_decoding_time_bin_size, known_named_decoding_epochs_type= 'laps', masked_time_bin_fill_type= 'ignore'), 
                                # IdentifyingContext(trained_compute_epochs= 'laps', pfND_ndim= 1, decoder_identifier= 'pseudo2D', time_bin_size= epochs_decoding_time_bin_size, known_named_decoding_epochs_type= 'laps', masked_time_bin_fill_type= 'nan_filled', data_grain= 'per_time_bin'),
                                # IdentifyingContext(trained_compute_epochs= 'laps', pfND_ndim= 1, decoder_identifier= 'pseudo2D', time_bin_size= epochs_decoding_time_bin_size, known_named_decoding_epochs_type= 'laps', masked_time_bin_fill_type= 'dropped', data_grain= 'per_time_bin'),
                ]


                flat_result_context_dict = {k:v for k, v in flat_result_context_dict.items() if k in active_ctxts}

                # decoder_laps_filter_epochs_decoder_result_dict = deepcopy(flat_result_context_dict)
                decoder_laps_filter_epochs_decoder_result_dict = {f"psuedo2D_{k.get('masked_time_bin_fill_type')}":deepcopy(v) for k, v in flat_result_context_dict.items()}
                decoder_laps_filter_epochs_decoder_result_dict
                
            except Exception as e:
                raise e


            ## OUTPUTS: decoder_laps_filter_epochs_decoder_result_dict
            

            # ==================================================================================================================================================================================================================================================================================== #
            # Separate export for each masked_time_bin_fill_type  - PBE                                                                                                                                                                                                                            #
            # ==================================================================================================================================================================================================================================================================================== #

            pbe_trained_decoder_search_context = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='pbe', masked_time_bin_fill_type=('ignore', 'nan_filled', 'dropped'), data_grain='per_time_bin') # , data_grain= 'per_time_bin -- not really relevant: ['masked_time_bin_fill_type', 'known_named_decoding_epochs_type', 'data_grain']
            # laps_trained_decoder_search_context = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='laps', masked_time_bin_fill_type='dropped', data_grain='per_time_bin')
            flat_context_list, flat_result_context_dict, flat_decoder_context_dict, flat_decoded_marginal_posterior_df_context_dict = a_new_fully_generic_result.get_results_matching_contexts(context_query=pbe_trained_decoder_search_context, return_multiple_matches=True, debug_print=True)

            active_ctxts = [
                            IdentifyingContext(trained_compute_epochs= 'laps', pfND_ndim= 1, decoder_identifier= 'pseudo2D', time_bin_size=epochs_decoding_time_bin_size, known_named_decoding_epochs_type= 'pbe', masked_time_bin_fill_type= 'ignore'), 
                            # IdentifyingContext(trained_compute_epochs= 'laps', pfND_ndim= 1, decoder_identifier= 'pseudo2D', time_bin_size= epochs_decoding_time_bin_size, known_named_decoding_epochs_type= 'pbe', masked_time_bin_fill_type= 'nan_filled', data_grain= 'per_time_bin'),
                            # IdentifyingContext(trained_compute_epochs= 'laps', pfND_ndim= 1, decoder_identifier= 'pseudo2D', time_bin_size= epochs_decoding_time_bin_size, known_named_decoding_epochs_type= 'pbe', masked_time_bin_fill_type= 'dropped', data_grain= 'per_time_bin'),
            ]

            flat_result_context_dict = {k:v for k, v in flat_result_context_dict.items() if k in active_ctxts}
            decoder_ripple_filter_epochs_decoder_result_dict = {f"psuedo2D_{k.get('masked_time_bin_fill_type')}":deepcopy(v) for k, v in flat_result_context_dict.items()}
            

            filter_epochs_ripple_df: Optional[pd.DataFrame] = kwargs.pop('filter_epochs_ripple_df', None)
            if filter_epochs_ripple_df is not None:
                ## use `filter_epochs_ripple` to filter the above `decoder_ripple_filter_epochs_decoder_result_dict`                
                ## INPUTS: high_heuristic_only_filtered_decoder_filter_epochs_decoder_result_dict, filtered_decoder_filter_epochs_decoder_result_dict
                # INPUTS: included_heuristic_ripple_start_times, high_heuristic_only_filtered_decoder_filter_epochs_decoder_result_dict, excluded_heuristic_ripple_start_times, low_heuristic_only_filtered_decoder_filter_epochs_decoder_result_dict
                included_decoder_ripple_filter_epochs_decoder_result_dict = {} # deepcopy(decoder_ripple_filter_epochs_decoder_result_dict)
                
                for k, all_epoch_result in decoder_ripple_filter_epochs_decoder_result_dict.items():
                    # all_filter_epochs_df: pd.DataFrame = deepcopy(all_epoch_result.filter_epochs)
                    
                    ## Result to filter
                    included_filter_epoch_result: DecodedFilterEpochsResult = deepcopy(all_epoch_result)

                    # included_filter_epoch_times = filter_epochs_ripple_df[['start', 'stop']].to_numpy() # Both 'start', 'stop' column matching
                    included_filter_epoch_times = filter_epochs_ripple_df['start'].to_numpy() # Both 'start', 'stop' column matching
                    included_decoder_ripple_filter_epochs_decoder_result_dict[k] = included_filter_epoch_result.filtered_by_epoch_times(included_epoch_start_times=included_filter_epoch_times) ## returns a modified result


                # decoder_ripple_filter_epochs_decoder_result_dict = {k:v for k, v in decoder_ripple_filter_epochs_decoder_result_dict.items()}
                
                print(f'filtering down to {len(included_filter_epoch_times)} filter epochs.')
                # decoder_ripple_filter_epochs_decoder_result_dict = {k:included_decoder_ripple_filter_epochs_decoder_result_dict[k] for k, v in included_decoder_ripple_filter_epochs_decoder_result_dict.items()} ## replace with the filtered version
                decoder_ripple_filter_epochs_decoder_result_dict = included_decoder_ripple_filter_epochs_decoder_result_dict
                
                print(f'\tdone.')

                ## OUTPUTS: all_filter_epochs_df, all_filter_epochs_df
                ## OUTPUTS: included_filter_epoch_times_to_all_epoch_index_arr
            else:
                print(f'no filter epochs provided.')


            ## OUTPUTS: decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict
            
            ## INPUTS: decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict
            ## UPDATES (in-place): decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict

            # Add the maze_id to the active_filter_epochs so we can see how properties change as a function of which track the replay event occured on:
            session_name: str = owning_pipeline_reference.session_name
            t_start, t_delta, t_end = owning_pipeline_reference.find_LongShortDelta_times()
            _common_add_columns_kwargs = dict(session_name=session_name, t_start=t_start, t_delta=t_delta, t_end=t_end, should_raise_exception_on_fail=True)
        
            decoder_laps_filter_epochs_decoder_result_dict = {k:DecodedFilterEpochsResult.perform_add_additional_epochs_columns(a_result=a_result, **_common_add_columns_kwargs) for k, a_result in decoder_laps_filter_epochs_decoder_result_dict.items()}
            decoder_ripple_filter_epochs_decoder_result_dict = {k:DecodedFilterEpochsResult.perform_add_additional_epochs_columns(a_result=a_result, **_common_add_columns_kwargs) for k, a_result in decoder_ripple_filter_epochs_decoder_result_dict.items()}
        
            ## OUTPUTS: decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict

            # ==================================================================================================================================================================================================================================================================================== #
            # Figure Export Copmonent                                                                                                                                                                                                                                                              #
            # ==================================================================================================================================================================================================================================================================================== #
            # using: perform_export_all_decoded_posteriors_as_images

            ## INPUTS:: filtered_decoder_filter_epochs_decoder_result_dict, long_like_during_post_delta_only_filter_epochs
            # active_epochs_decoder_result_dict = deepcopy(filtered_decoder_filter_epochs_decoder_result_dict)

            ## Build a makeshift dict with just the pseudo2D in it:
            ## INPUTS:
            a_decoder = deepcopy(flat_decoder_context_dict[active_ctxts[0]]) ## same decoder works for any of them:
            # a_decoder = deepcopy(a_laps_trained_decoder)

            ## INPUTS: active_epochs_decoder_result_dict

            print(f'\tspecific_session_output_folder: "{_specific_session_output_folder.as_posix()}"')

            if custom_export_formats is None:

                custom_export_formats: Dict[str, HeatmapExportConfig] = {
                    # 'greyscale': HeatmapExportConfig.init_greyscale(desired_height=desired_height, post_render_image_functions_builder_fn=_build_no_op_image_export_functions_dict),
                    # 'color': HeatmapExportConfig(colormap=colormap, export_kind=HeatmapExportKind.COLORMAPPED, desired_height=desired_height, post_render_image_functions_builder_fn=_build_no_op_image_export_functions_dict, **kwargs),
                    # 'raw_rgba': HeatmapExportConfig.init_for_export_kind(export_kind=HeatmapExportKind.RAW_RGBA, lower_bound_alpha=0.1, drop_below_threshold=1e-2, desired_height=desired_height),
                    'raw_rgba': HeatmapExportConfig.init_for_export_kind(export_kind=HeatmapExportKind.RAW_RGBA, 
                                                                        raw_RGBA_only_parameters = dict(spikes_df=deepcopy(get_proper_global_spikes_df(owning_pipeline_reference)), xbin=deepcopy(a_decoder.xbin), lower_bound_alpha=0.1, drop_below_threshold=1e-3, t_bin_size=time_bin_size, use_four_decoders_version=False), desired_height=desired_height, 
                                                                        post_render_image_functions_builder_fn=ImagePostRenderFunctionSets._build_mergedColorDecoders_image_export_functions_dict),
                                                                                                                                                
                    # 'raw_rgba_four_decoders': HeatmapExportConfig.init_for_export_kind(export_kind=HeatmapExportKind.RAW_RGBA, 
                    #                                                     raw_RGBA_only_parameters = dict(spikes_df=deepcopy(get_proper_global_spikes_df(owning_pipeline_reference)), xbin=deepcopy(a_decoder.xbin), lower_bound_alpha=0.1, drop_below_threshold=1e-2, t_bin_size=time_bin_size,  use_four_decoders_version=True),
                    #                                                     desired_height=desired_height, post_render_image_functions_builder_fn=ImagePostRenderFunctionSets._build_mergedColorDecoders_image_export_functions_dict),

                }
                # custom_export_formats = None


            # Delete the existing directory if it exists:
            if delete_previous_outputs_folder:
                ## delete the outputs created:
                for a_format_name in custom_export_formats.keys():
                    _an_export_format_output_folder = _specific_session_output_folder.joinpath(a_format_name)                    
                    try:
                        if _an_export_format_output_folder.exists():
                            print(f'\tdeleting previous outputs folder at "{_an_export_format_output_folder.as_posix()}"...')
                            shutil.rmtree(_an_export_format_output_folder)
                            print(f'\t\tsuccessfully deleted extant folder.')
                            _an_export_format_output_folder.mkdir(parents=True, exist_ok=True) # re-make the folder
                            
                    except Exception as e:
                        print(f'\tError deleting folder "{_an_export_format_output_folder.as_posix()}": {e}')
                        continue

            # Main export function _______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
            out_paths, out_custom_formats_dict = PosteriorExporting.perform_export_all_decoded_posteriors_as_images(decoder_laps_filter_epochs_decoder_result_dict=decoder_laps_filter_epochs_decoder_result_dict,
                                                                                                                     decoder_ripple_filter_epochs_decoder_result_dict=decoder_ripple_filter_epochs_decoder_result_dict,
                                                                                                                    _save_context=_parent_save_context, parent_output_folder=_specific_session_output_folder,
                                                                                                                    desired_height=desired_height, custom_export_formats=custom_export_formats, combined_img_padding=6, combined_img_separator_color=(0, 0, 0, 255))

            graphics_output_dict['out_paths'] = deepcopy(out_paths) # 'out_paths': out_paths
            graphics_output_dict['out_custom_formats_dict'] = deepcopy(out_custom_formats_dict)
            print(f'\tout_paths: {out_paths}')
            print(f'done.')


            # ==================================================================================================================================================================================================================================================================================== #
            # TODO 2025-05-30 17:54: - [ ] Export 1D results in the "competition normalized" way that Kamran likes                                                                                                                                                                                 #
            # ==================================================================================================================================================================================================================================================================================== #

            print(f'beginning export of 1D results in the normalizations style Kamran likes...')

            _in_pseudo2D_dict = {'laps': None, 'ripple': None}
            ## INPUTS: decoder_laps_filter_epochs_decoder_result_dict, decoder_ripple_filter_epochs_decoder_result_dict
            if enable_laps_merged_export:
                _in_pseudo2D_dict['laps'] = deepcopy(decoder_laps_filter_epochs_decoder_result_dict[f'psuedo2D_{masked_time_bin_fill_type}'])
            if enable_ripple_merged_export:
                _in_pseudo2D_dict['ripple'] = deepcopy(decoder_ripple_filter_epochs_decoder_result_dict[f'psuedo2D_{masked_time_bin_fill_type}'])

            _pseudo2D_split_to_1D_continuous_results_dict_dict = {}

            for a_decoded_epoch_name, a_pseudo2D_decoder_continuously_decoded_result in _in_pseudo2D_dict.items():

                # From `General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions.prepare_and_perform_add_add_pseudo2D_decoder_decoded_epochs`
                # all_directional_continuously_decoded_dict = most_recent_continuously_decoded_dict or {}
                # a_pseudo2D_decoder_continuously_decoded_result: DecodedFilterEpochsResult = deepcopy(decoder_ripple_filter_epochs_decoder_result_dict[f'psuedo2D_{masked_time_bin_fill_type}']) ## only the ignore result


                ## INPUTS: laps_pseudo2D_continuous_specific_decoded_result: DecodedFilterEpochsResult
                # unique_decoder_names = ('long', 'short')
                # a_non_PBE_marginal_over_track_ID, a_non_PBE_marginal_over_track_ID_posterior_df = DirectionalPseudo2DDecodersResult.build_generalized_non_marginalized_raw_posteriors(a_pseudo2D_decoder_continuously_decoded_result, unique_decoder_names=unique_decoder_names) #AssertionError: only works when curr_array_shape[1]: 4 correspond to the unique_decoder_names: ('long', 'short'). (typically all-directional decoder).

                unique_decoder_names = ('long_LR', 'long_RL', 'short_LR', 'short_RL')
                a_pseudo2D_split_to_1D_continuous_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = a_pseudo2D_decoder_continuously_decoded_result.split_pseudo2D_result_to_1D_result(pseudo2D_decoder_names_list=unique_decoder_names)
                # a_masked_pseudo2D_split_to_1D_continuous_results_dict: Dict[types.DecoderName, DecodedFilterEpochsResult] = a_masked_pseudo2D_decoder_continuously_decoded_result.split_pseudo2D_result_to_1D_result(pseudo2D_decoder_names_list=unique_decoder_names)
                a_pseudo2D_split_to_1D_continuous_results_dict = {k:DecodedFilterEpochsResult.perform_add_additional_epochs_columns(a_result=a_result, **_common_add_columns_kwargs) for k, a_result in a_pseudo2D_split_to_1D_continuous_results_dict.items()} ## add the extra columns if needed
                # OUTPUTS: a_pseudo2D_split_to_1D_continuous_results_dict, a_masked_pseudo2D_split_to_1D_continuous_results_dict
                _pseudo2D_split_to_1D_continuous_results_dict_dict[a_decoded_epoch_name] = deepcopy(a_pseudo2D_split_to_1D_continuous_results_dict)


            # Run an export function again _______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ #
        
            pseudo2D_split_to_1D_custom_export_formats: Dict[str, HeatmapExportConfig] = {
                'greyscale_shared_norm': HeatmapExportConfig.init_greyscale(vmin=0.0, vmax=1.0, desired_height=desired_height, post_render_image_functions_builder_fn=ImagePostRenderFunctionSets._build_no_op_image_export_functions_dict),
                # 'cleaned_greyscale_shared_norm': HeatmapExportConfig(colormap=FixedCustomColormaps.get_custom_greyscale_with_low_values_dropped_cmap(low_value_cutoff=0.01, full_opacity_threshold=0.4, grey_value=0.1), export_kind=HeatmapExportKind.COLORMAPPED, vmin=0.0, vmax=1.0, desired_height=desired_height, post_render_image_functions_builder_fn=ImagePostRenderFunctionSets._build_no_op_image_export_functions_dict),
                'viridis_shared_norm': HeatmapExportConfig(colormap='viridis', export_kind=HeatmapExportKind.COLORMAPPED, vmin=0.0, vmax=1.0, desired_height=desired_height, post_render_image_functions_builder_fn=ImagePostRenderFunctionSets._build_no_op_image_export_functions_dict), # 2025-07-24 - The format Kamran likes where they are globally normalized
            }
            pseudo2D_split_to_1D_out_paths, pseudo2D_split_to_1D_out_custom_formats_dict = PosteriorExporting.perform_export_all_decoded_posteriors_as_images(decoder_laps_filter_epochs_decoder_result_dict=_pseudo2D_split_to_1D_continuous_results_dict_dict['laps'],
                                                                                                                        decoder_ripple_filter_epochs_decoder_result_dict=_pseudo2D_split_to_1D_continuous_results_dict_dict['ripple'], ## just the ripples
                                                                                                                    _save_context=_parent_save_context, parent_output_folder=_specific_session_output_folder,
                                                                                                                    desired_height=desired_height, custom_export_formats=pseudo2D_split_to_1D_custom_export_formats, combined_img_padding=6,
                                                                                                                    #  combined_img_separator_color=(255, 255, 255, 255),
                                                                                                                    combined_img_separator_color=(200, 46, 33, 10),
                                                                                                                     )
            if not isinstance(graphics_output_dict['out_paths'], benedict):
                graphics_output_dict['out_paths'] = benedict(graphics_output_dict['out_paths']) # 'out_paths': out_paths
            # graphics_output_dict['out_paths'].merge(pseudo2D_split_to_1D_out_paths)
            graphics_output_dict['out_paths'].merge({k:v for k, v in pseudo2D_split_to_1D_out_paths.items() if (v is not None)})

            if not isinstance(graphics_output_dict['out_custom_formats_dict'], benedict):
                graphics_output_dict['out_custom_formats_dict'] = benedict(graphics_output_dict['out_custom_formats_dict']) # 'out_paths': out_paths
            # graphics_output_dict['out_custom_formats_dict'].merge(pseudo2D_split_to_1D_out_custom_formats_dict)
            graphics_output_dict['out_custom_formats_dict'].merge({k:v for k, v in pseudo2D_split_to_1D_out_custom_formats_dict.items() if (v is not None)})

            # print(f'\tout_paths: {pseudo2D_split_to_1D_out_paths}')
            print(f'done.')


            # ==================================================================================================================================================================================================================================================================================== #
            # END/RESUME: Merged Outputs                                                                                                                                                                                                                                                           #
            # ==================================================================================================================================================================================================================================================================================== #
            # Build Merged across time images and flattened dict of images _______________________________________________________ #

            # from pyphoplacecellanalysis.Pho2D.data_exporting import HeatmapExportConfig, HeatmapExportKind

            # out_paths: Dict[types.KnownNamedDecoderTrainedComputeEpochsType, Dict[types.DecoderName, Path]] = graphics_output_dict['out_paths']
            # out_custom_formats_dict: Dict[types.KnownNamedDecodingEpochsType, Dict[types.DecoderName, Dict[str, List[HeatmapExportConfig]]]] = graphics_output_dict['out_custom_formats_dict']

            # flat_imgs = []
            graphics_output_dict['flat_parent_save_paths'] = []
            flat_imgs_dict: Dict[IdentifyingContext, List] = {}
            flat_merged_images = {}
            flat_merged_image_paths = {}

            for a_known_epoch_type_name, v_dict in out_custom_formats_dict.items():
                # a_known_epoch_type_name: ['laps', 'ripple']
                for a_decoder_name, a_rendered_configs_dict in v_dict.items():
                    
                    for a_config_name, a_rendered_config_list in a_rendered_configs_dict.items():
                        # 'raw_rgba'
                        # print(a_rendered_config_list)
                        # len(a_rendered_config_list)
                        
                        a_ctxt = IdentifyingContext(known_epoch_type_name=a_known_epoch_type_name, decoder=a_decoder_name, config=a_config_name)
                        flat_imgs = []
                        
                        parent_save_path = None
                        for i, a_config in enumerate(a_rendered_config_list):                                  
                            if parent_save_path is None:
                                posterior_save_path = a_config.posterior_saved_path
                                parent_save_path = posterior_save_path.parent.resolve()
                                graphics_output_dict['flat_parent_save_paths'].append(parent_save_path)

                            _posterior_image = a_config.posterior_saved_image
                            flat_imgs.append(_posterior_image)                            
                            # print(F'a_rendered_config: {type(a_rendered_config)}')
                        ## END  for i, a_config in enum...
                        ## OUTPUTS: flat_imgs
                        # _merged_img = horizontal_image_stack(flat_imgs, padding=10, separator_color='white')
                        _merged_img = vertical_image_stack(flat_imgs, padding=10, separator_color='white')
                        flat_merged_images[a_known_epoch_type_name] = _merged_img
                        flat_imgs_dict[a_ctxt] = flat_imgs
                        
                        ## Save the image to disk if we want
                        # _merged_img.save
                        if (_merged_img is not None) and (parent_save_path is not None):
                            ## Save the image:
                            _img_path = parent_save_path.joinpath(f'merged_{a_known_epoch_type_name}[{i}].png').resolve()
                            try:
                                _merged_img.save(_img_path)
                                flat_merged_image_paths[a_ctxt] = _img_path
                            except Exception as e:
                                raise e
                        
                        

            # flat_img_out_paths
            # flat_merged_images

            graphics_output_dict['flat_merged_images'] = flat_merged_images
            graphics_output_dict['flat_merged_image_paths'] = flat_merged_image_paths
            graphics_output_dict['flat_imgs_dict'] = flat_imgs_dict
            

            ## one last cleanup step
            
            
            ImageHelpers.clear_cached_fonts()
            
            return graphics_output_dict





# ==================================================================================================================== #
# Private Methods                                                                                                      #
# ==================================================================================================================== #





from pyphocorehelpers.DataStructure.enum_helpers import ExtendedEnum

class KnownFilterEpochs(ExtendedEnum):
    """Describes the type of file progress actions that can be performed to get the right verbage.
    Used by `_subfn_compute_decoded_epochs(...)`
   
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import KnownFilterEpochs
    """
    LAP = "lap"
    PBE = "pbe"
    RIPPLE = "ripple"
    REPLAY = "replay"
    NON_PBE = "non_pbe"
    GENERIC = "GENERIC"


    # BEGIN PROBLEMATIC ENUM CODE ________________________________________________________________________________________ #
    @property
    def default_figure_name(self):
        return KnownFilterEpochs.default_figure_nameList()[self]

    # Static properties
    @classmethod
    def default_figure_nameList(cls):
        return cls.build_member_value_dict([f'Laps',f'PBEs',f'Ripples',f'Replays',f'Non-PBEs',f'Generic'])


    @classmethod
    def _perform_get_filter_epochs_df(cls, sess, filter_epochs, min_epoch_included_duration=None, debug_print=False):
        """DOES NOT WORK due to messed-up `.epochs.get_non_overlapping_df`

        Args:
            sess (_type_): computation_result.sess
            filter_epochs (_type_): _description_
            min_epoch_included_duration: only applies to Replay for some reason?

        Raises:
            NotImplementedError: _description_
        """
        # post_process_epochs_fn = lambda filter_epochs_df: filter_epochs_df.epochs.get_non_overlapping_df(debug_print=debug_print) # post_process_epochs_fn should accept an epochs dataframe and return a clean copy of the epochs dataframe
        post_process_epochs_fn = lambda filter_epochs_df: filter_epochs_df.epochs.get_valid_df() # post_process_epochs_fn should accept an epochs dataframe and return a clean copy of the epochs dataframe

        if debug_print:
            print(f'')
        if isinstance(filter_epochs, str):
            try:
                filter_epochs = cls.init(value=filter_epochs) # init an enum object from the string
            except Exception as e:
                print(f'filter_epochs "{filter_epochs}" could not be parsed into KnownFilterEpochs but is string.')
                raise e

        if isinstance(filter_epochs, cls):
            if filter_epochs.name == cls.LAP.name:
                ## Lap-Epochs Decoding:
                laps_copy = deepcopy(sess.laps)
                active_filter_epochs = laps_copy.as_epoch_obj() # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                active_filter_epochs = post_process_epochs_fn(active_filter_epochs)

            elif filter_epochs.name == cls.PBE.name:
                ## PBEs-Epochs Decoding:
                active_filter_epochs = deepcopy(sess.pbe) # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                active_filter_epochs = post_process_epochs_fn(active_filter_epochs)
            
            elif filter_epochs.name == cls.RIPPLE.name:
                ## Ripple-Epochs Decoding:
                active_filter_epochs = deepcopy(sess.ripple) # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                # note we need to make sure we have a valid label to start because `.epochs.get_non_overlapping_df()` requires one.
                active_filter_epochs['label'] = active_filter_epochs.index.to_numpy() # integer ripple indexing
                active_filter_epochs = post_process_epochs_fn(active_filter_epochs)
                active_filter_epochs['label'] = active_filter_epochs.index.to_numpy() # integer ripple indexing
                
            elif filter_epochs.name == cls.REPLAY.name:
                active_filter_epochs = deepcopy(sess.replay) # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                active_filter_epochs = post_process_epochs_fn(active_filter_epochs)
                if min_epoch_included_duration is not None:
                    active_filter_epochs = active_filter_epochs[active_filter_epochs.duration >= min_epoch_included_duration] # only include those epochs which are greater than or equal to two decoding time bins
                    
            elif filter_epochs.name == cls.NON_PBE.name:
                ## non-PBEs-Epochs Decoding:
                active_filter_epochs = deepcopy(sess.non_pbe) # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                active_filter_epochs = post_process_epochs_fn(active_filter_epochs)

            else:
                print(f'filter_epochs "{filter_epochs.name}" could not be parsed into KnownFilterEpochs but is string.')
                active_filter_epochs = None
                raise NotImplementedError

        else:
            # Use it filter_epochs raw, hope it's right. It should be some type of Epoch or pd.DataFrame object.
            active_filter_epochs = filter_epochs
            ## TODO: why even allow passing in a raw Epoch object? It's not clear what the use case is.
            raise NotImplementedError


        # Finally, convert back to Epoch object:
        assert isinstance(active_filter_epochs, pd.DataFrame)
        # active_filter_epochs = Epoch(active_filter_epochs)
        if debug_print:
            print(f'active_filter_epochs: {active_filter_epochs}')
        return active_filter_epochs


    @classmethod
    def perform_get_filter_epochs_df(cls, sess, filter_epochs, min_epoch_included_duration=None, **kwargs):
        """Temporary wrapper for `process_functionList` to replace `_perform_get_filter_epochs_df`

        Args:
            sess (_type_): computation_result.sess
            filter_epochs (_type_): _description_
            min_epoch_included_duration: only applies to Replay for some reason?

        """
        # `process_functionList` version:
        return cls.process_functionList(sess=sess, filter_epochs=filter_epochs, min_epoch_included_duration=min_epoch_included_duration, default_figure_name='', **kwargs)[0] # [0] gets the returned active_filter_epochs
        # proper `_perform_get_filter_epochs_df` version:
        # return cls._perform_get_filter_epochs_df(sess=computation_result.sess, filter_epochs=filter_epochs, min_epoch_included_duration=min_epoch_included_duration)

    @classmethod
    def process_functionList(cls, sess, filter_epochs, min_epoch_included_duration, default_figure_name='stacked_epoch_slices_matplotlib_subplots'):
        # min_epoch_included_duration = decoding_time_bin_size * float(2) # 0.06666
        # min_epoch_included_duration = 0.06666

        if isinstance(filter_epochs, str):
            print(f'filter_epochs string: "{filter_epochs}"')
            filter_epochs = cls.init(value=filter_epochs) # init an enum object from the string
            # filter_epochs = cls.init(filter_epochs, fallback_value=cls.GENERIC) # init an enum object from the string
            default_figure_name = filter_epochs.default_figure_name

            if filter_epochs.name == KnownFilterEpochs.LAP.name:
                ## Lap-Epochs Decoding:
                laps_copy = deepcopy(sess.laps)
                active_filter_epochs = laps_copy.as_epoch_obj() # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                # pre_exclude_n_epochs = active_filter_epochs.n_epochs
                active_filter_epochs = active_filter_epochs.epochs.get_non_overlapping_df()
                # post_exclude_n_epochs = active_filter_epochs.n_epochs                    
                # num_excluded_epochs = post_exclude_n_epochs - pre_exclude_n_epochs
                # if num_excluded_epochs > 0:
                #     print(f'num_excluded_epochs: {num_excluded_epochs} due to overlap.')
                # ## Build Epochs:
                epoch_description_list = [f'lap[{epoch_tuple.lap_id}]' for epoch_tuple in active_filter_epochs[['lap_id']].itertuples()] # Short

            elif filter_epochs.name == KnownFilterEpochs.PBE.name:
                ## PBEs-Epochs Decoding:
                active_filter_epochs = deepcopy(sess.pbe) # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                active_filter_epochs = active_filter_epochs.epochs.get_non_overlapping_df()
            
            elif filter_epochs.name == KnownFilterEpochs.RIPPLE.name:
                ## Ripple-Epochs Decoding:
                active_filter_epochs = deepcopy(sess.ripple) # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                active_filter_epochs = active_filter_epochs.epochs.get_non_overlapping_df()
                active_filter_epochs['label'] = active_filter_epochs.index.to_numpy() # integer ripple indexing
                epoch_description_list = [f'ripple[{epoch_tuple.label}]' for epoch_tuple in active_filter_epochs[['label']].itertuples()] # SHORT
                
            elif filter_epochs.name == KnownFilterEpochs.REPLAY.name:
                active_filter_epochs = deepcopy(sess.replay) # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                active_filter_epochs = active_filter_epochs.epochs.get_non_overlapping_df()
                if min_epoch_included_duration is not None:
                    active_filter_epochs = active_filter_epochs[active_filter_epochs.duration >= min_epoch_included_duration] # only include those epochs which are greater than or equal to two decoding time bins
                epoch_description_list = [f'{default_figure_name} {epoch_tuple.label}' for epoch_tuple in active_filter_epochs[['label']].itertuples()]
                

            elif filter_epochs.name == KnownFilterEpochs.NON_PBE.name:
                active_filter_epochs = deepcopy(sess.non_pbe) # epoch object
                if not isinstance(active_filter_epochs, pd.DataFrame):
                    active_filter_epochs = active_filter_epochs.to_dataframe()
                active_filter_epochs = active_filter_epochs.epochs.get_non_overlapping_df()
                if min_epoch_included_duration is not None:
                    active_filter_epochs = active_filter_epochs[active_filter_epochs.duration >= min_epoch_included_duration] # only include those epochs which are greater than or equal to two decoding time bins
                epoch_description_list = [f'{default_figure_name} {epoch_tuple.label}' for epoch_tuple in active_filter_epochs[['label']].itertuples()]
                

            else:
                print(f'filter_epochs "{filter_epochs.name}" could not be parsed into KnownFilterEpochs but is string.')
                raise NotImplementedError

            # Finally, convert back to Epoch object:
            assert isinstance(active_filter_epochs, pd.DataFrame)
            # active_filter_epochs = Epoch(active_filter_epochs)

        else:
            # Use it raw, hope it's right
            active_filter_epochs = filter_epochs
            default_figure_name = f'{default_figure_name}_CUSTOM'
            epoch_description_list = [f'{default_figure_name} {epoch_tuple.label}' for epoch_tuple in active_filter_epochs.to_dataframe()[['label']].itertuples()]

        return active_filter_epochs, default_figure_name, epoch_description_list





        # ## non_PBE Epochs:
        # active_file_suffix = '.non_pbe.npy'
        # found_datafile = Epoch.from_file(fp.with_suffix(active_file_suffix))
        # if (not force_recompute) and (found_datafile is not None):
        # 	print('Loading success: {}.'.format(active_file_suffix))
        # 	session.non_pbe = found_datafile
        # else:
        # 	# Otherwise load failed, perform the fallback computation
        # 	if not force_recompute:
        # 		print('Failure loading {}. Must recompute.\n'.format(active_file_suffix))
        # 	else:
        # 		print(f'force_recompute is True, recomputing...')
        # 	try:
        # 		# active_pbe_parameters = kwargs.pop('pbe_epoch_detection_params', session.config.preprocessing_parameters.epoch_estimation_parameters.PBEs)
        # 		active_non_pbe_parameters = {} # session.config.preprocessing_parameters.epoch_estimation_parameters.PBEs
        # 		session.non_pbe = DataSession.compute_non_PBE_epochs(session, active_parameters=active_non_pbe_parameters, save_on_compute=True)
        # 	except (ValueError, AttributeError) as e:
        # 		print(f'Computation failed with error {e}. Skipping .non_pbe')
        # 		session.non_pbe = None
                