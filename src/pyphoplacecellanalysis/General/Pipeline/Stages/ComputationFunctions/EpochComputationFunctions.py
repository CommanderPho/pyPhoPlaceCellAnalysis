 
from __future__ import annotations # prevents having to specify types for typehinting as strings
from typing import TYPE_CHECKING

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

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, BayesianPlacemapPositionDecoder, DecodedFilterEpochsResult, Zhang_Two_Step
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


@custom_define(slots=False, eq=False)
class NonPBEDimensionalDecodingResult(UnpackableMixin, ComputedResult):
    """Contains all decoding results for either 1D or 2D computations
    
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import NonPBEDimensionalDecodingResult
    
    results2D: NonPBEDimensionalDecodingResult = NonPBEDimensionalDecodingResult(ndim=2, 
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


    """
    _VersionedResultMixin_version: str = "2024.02.18_0" # to be updated in your IMPLEMENTOR to indicate its version
    
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
    

    def recompute(self, curr_active_pipeline, pfND_ndim: int = 2, epochs_decoding_time_bin_size: float = 0.025, skip_training_test_split: bool = False):
        """ For a specified decoding time_bin_size and ndim (1D or 2D), copies the global pfND, builds new epoch objects, then decodes both train_test and continuous epochs

        test_epoch_specific_decoded_results_dict, continuous_specific_decoded_results_dict, new_decoder_dict, new_pfs_dict = a_new_NonPBE_Epochs_obj.recompute(curr_active_pipeline=curr_active_pipeline, epochs_decoding_time_bin_size = 0.058)
        
        pfND_ndim: 1 - 1D, 2 - 2D
        
        """
        from neuropy.core.epoch import Epoch, ensure_dataframe, ensure_Epoch
        from neuropy.analyses.placefields import PfND
        # from neuropy.analyses.time_dependent_placefields import PfND_TimeDependent
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder, DecodedFilterEpochsResult, SingleEpochDecodedResult
        # from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult

         # 25ms
        # epochs_decoding_time_bin_size: float = 0.050 # 50ms
        # epochs_decoding_time_bin_size: float = 0.250 # 250ms

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

        single_global_epoch: Epoch = Epoch(self.single_global_epoch_df)

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
            new_decoder_dict: Dict[types.DecoderName, BasePositionDecoder] = {a_name:BasePositionDecoder(pf=a_pfs).replacing_computation_epochs(epochs=deepcopy(curr_active_pipeline.filtered_sessions[non_directional_names_to_default_epoch_names_map[a_name]].non_pbe)) for a_name, a_pfs in original_pfs_dict.items()} ## build new simple decoders
            
        else:
            ## extract values:
            a_new_training_df_dict = self.a_new_training_df_dict
            a_new_testing_epoch_obj_dict: Dict[types.DecoderName, Epoch] = self.a_new_testing_epoch_obj_dict
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


    @function_attributes(short_name=None, tags=['MAIN', 'compute'], input_requires=[], output_provides=[], uses=['self.__class__.build_frame_divided_epochs(...)'], used_by=[], creation_date='2025-02-18 09:40', related_items=[])
    def compute_all(self, curr_active_pipeline, epochs_decoding_time_bin_size: float = 0.025, frame_divide_bin_size: float = 0.5, compute_1D: bool = True, compute_2D: bool = True, skip_training_test_split: Optional[bool] = None) -> Tuple[Optional[NonPBEDimensionalDecodingResult], Optional[NonPBEDimensionalDecodingResult]]:
        """ computes all pfs, decoders, and then performs decodings on both continuous and subivided epochs.
        
        ## OUTPUTS: global_continuous_decoded_epochs_result2D, a_continuous_decoded_result2D, p_x_given_n2D
        # (test_epoch_specific_decoded_results1D_dict, continuous_specific_decoded_results1D_dict, new_decoder1D_dict, new_pf1Ds_dict), frame_divided_epochs_specific_decoded_results1D_dict, ## 1D Results
        # (test_epoch_specific_decoded_results2D_dict, continuous_specific_decoded_results2D_dict, new_decoder2D_dict, new_pf2Ds_dict), frame_divided_epochs_specific_decoded_results2D_dict, global_continuous_decoded_epochs_result2D # 2D results
        
        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import Compute_NonPBE_Epochs

            long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
            a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = Compute_NonPBE_Epochs.init_from_pipeline(curr_active_pipeline=curr_active_pipeline)
            ## apply the new epochs to the session:
            curr_active_pipeline.filtered_sessions[global_epoch_name].non_PBE_epochs = deepcopy(a_new_NonPBE_Epochs_obj.global_epoch_only_non_PBE_epoch_df)

            results1D, results2D = a_new_NonPBE_Epochs_obj.compute_all(curr_active_pipeline, epochs_decoding_time_bin_size=0.025, frame_divide_bin_size=0.50, compute_1D=True, compute_2D=True)
        
        """
        # from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df

        if skip_training_test_split is not None:
            self.skip_training_test_split = skip_training_test_split ## override existing
        else:
            skip_training_test_split = self.skip_training_test_split

        # Build frame_divided epochs first since they're needed for both 1D and 2D
        (global_frame_divided_epochs_obj, global_frame_divided_epochs_df), global_pos_df = self.__class__.build_frame_divided_epochs(curr_active_pipeline, frame_divide_bin_size=frame_divide_bin_size)
        
        results1D, results2D = None, None
        
        if compute_1D:
            test_epoch_specific_decoded_results1D_dict, continuous_specific_decoded_results1D_dict, new_decoder1D_dict, new_pf1Ds_dict = self.recompute(curr_active_pipeline=curr_active_pipeline, pfND_ndim=1, epochs_decoding_time_bin_size=epochs_decoding_time_bin_size, skip_training_test_split=skip_training_test_split)
            frame_divided_epochs_specific_decoded_results1D_dict = {a_name:a_new_decoder.decode_specific_epochs(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), filter_epochs=deepcopy(global_frame_divided_epochs_obj), decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False) for a_name, a_new_decoder in new_decoder1D_dict.items()}
            results1D = NonPBEDimensionalDecodingResult(ndim=1, 
                test_epoch_results=test_epoch_specific_decoded_results1D_dict, 
                continuous_results=continuous_specific_decoded_results1D_dict,
                decoders=new_decoder1D_dict, pfs=new_pf1Ds_dict,
                frame_divided_epochs_results=frame_divided_epochs_specific_decoded_results1D_dict, 
                frame_divided_epochs_df=deepcopy(global_frame_divided_epochs_df), pos_df=global_pos_df)

        if compute_2D:
            test_epoch_specific_decoded_results2D_dict, continuous_specific_decoded_results2D_dict, new_decoder2D_dict, new_pf2Ds_dict = self.recompute(curr_active_pipeline=curr_active_pipeline, pfND_ndim=2, epochs_decoding_time_bin_size=epochs_decoding_time_bin_size, skip_training_test_split=skip_training_test_split)
            frame_divided_epochs_specific_decoded_results2D_dict = {a_name:a_new_decoder.decode_specific_epochs(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)), filter_epochs=deepcopy(global_frame_divided_epochs_obj), decoding_time_bin_size=epochs_decoding_time_bin_size, debug_print=False) for a_name, a_new_decoder in new_decoder2D_dict.items()}
            results2D = NonPBEDimensionalDecodingResult(ndim=2, 
                test_epoch_results=test_epoch_specific_decoded_results2D_dict,
                continuous_results=continuous_specific_decoded_results2D_dict,
                decoders=new_decoder2D_dict, pfs=new_pf2Ds_dict,
                frame_divided_epochs_results=frame_divided_epochs_specific_decoded_results2D_dict, 
                frame_divided_epochs_df=deepcopy(global_frame_divided_epochs_df), pos_df=global_pos_df)

        return results1D, results2D
        
    @classmethod
    @function_attributes(short_name=None, tags=['frame_division'], input_requires=[], output_provides=[], uses=[], used_by=['cls.compute_all(...)'], creation_date='2025-02-11 00:00', related_items=[])
    def build_frame_divided_epochs(cls, curr_active_pipeline, frame_divide_bin_size: float = 1.0):
        """ 
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

        df: pd.DataFrame = ensure_dataframe(deepcopy(single_global_epoch)) 
        df['maze_name'] = 'global'
        # df['interval_type_id'] = 666

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
        global_session = curr_active_pipeline.filtered_sessions[global_epoch_name]
        global_frame_divided_epochs_df = global_frame_divided_epochs_obj.epochs.to_dataframe() #.rename(columns={'t_rel_seconds':'t'})
        global_frame_divided_epochs_df['label'] = deepcopy(global_frame_divided_epochs_df.index.to_numpy())
        # global_pos_df: pd.DataFrame = deepcopy(global_session.position.to_dataframe()) #.rename(columns={'t':'t_rel_seconds'})
        
        ## Extract Measured Position:
        global_pos_obj: "Position" = deepcopy(global_session.position)
        global_pos_df: pd.DataFrame = global_pos_obj.compute_higher_order_derivatives().position.compute_smoothed_position_info(N=15)
        global_pos_df.time_point_event.adding_epochs_identity_column(epochs_df=global_frame_divided_epochs_df, epoch_id_key_name='global_frame_division_idx', epoch_label_column_name='label', drop_non_epoch_events=True, should_replace_existing_column=True) # , override_time_variable_name='t_rel_seconds'
        
        ## Adds the ['frame_division_epoch_start_t'] columns to `stacked_flat_global_pos_df` so we can figure out the appropriate offsets
        frame_divided_epochs_properties_df: pd.DataFrame = deepcopy(global_frame_divided_epochs_df)
        frame_divided_epochs_properties_df['global_frame_division_idx'] = deepcopy(frame_divided_epochs_properties_df.index) ## add explicit 'global_frame_division_idx' column
        frame_divided_epochs_properties_df = frame_divided_epochs_properties_df.rename(columns={'start': 'frame_division_epoch_start_t'})[['global_frame_division_idx', 'frame_division_epoch_start_t']]
        global_pos_df = PandasHelpers.add_explicit_dataframe_columns_from_lookup_df(global_pos_df, frame_divided_epochs_properties_df, join_column_name='global_frame_division_idx')
        global_pos_df.sort_values(by=['t'], inplace=True) # Need to re-sort by timestamps once done

        if global_pos_df.attrs is None:
            global_pos_df.attrs = {}
            
        global_pos_df.attrs.update({'frame_divide_bin_size': frame_divide_bin_size})
        
        if global_frame_divided_epochs_df.attrs is None:
            global_frame_divided_epochs_df.attrs = {}
            
        global_frame_divided_epochs_df.attrs.update({'frame_divide_bin_size': frame_divide_bin_size})


        
        return (global_frame_divided_epochs_obj, global_frame_divided_epochs_df), global_pos_df


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
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import EpochComputationFunctions, EpochComputationsComputationsContainer, NonPBEDimensionalDecodingResult, Compute_NonPBE_Epochs, KnownFilterEpochs
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import GeneralDecoderDictDecodedEpochsDictResult, GenericResultTupleIndexType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType

        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()

        ## Unpack from pipeline:
        nonPBE_results: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['EpochComputations']
        a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = nonPBE_results.a_new_NonPBE_Epochs_obj
        results1D: NonPBEDimensionalDecodingResult = nonPBE_results.results1D
        results2D: NonPBEDimensionalDecodingResult = nonPBE_results.results2D

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
    results1D: Optional[NonPBEDimensionalDecodingResult] = serialized_field(default=None, repr=False)
    results2D: Optional[NonPBEDimensionalDecodingResult] = serialized_field(default=None, repr=False)

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
        results1D: NonPBEDimensionalDecodingResult = self.results1D
        # results2D: NonPBEDimensionalDecodingResult = self.results2D

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
                _a_masked_unused_marginal, a_masked_posterior_df = DirectionalPseudo2DDecodersResult.build_generalized_non_marginalized_raw_posteriors(a_masked_decoded_result, unique_decoder_names=unique_decoder_names) #[0]['p_x_given_n']
                ## spruce up the `a_masked_posterior_df` with some extra fields
                a_masked_posterior_df['delta_aligned_start_t'] = a_masked_posterior_df['t'] - t_delta ## subtract off t_delta    
                a_masked_posterior_df = a_masked_posterior_df.across_session_identity.add_session_df_columns(session_name=session_name, time_bin_size=epochs_decoding_time_bin_size, curr_session_t_delta=t_delta, time_col='t')
                
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
        results1D: NonPBEDimensionalDecodingResult = self.results1D
        # results2D: NonPBEDimensionalDecodingResult = self.results2D

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
        input_requires=[], output_provides=[], uses=['EpochComputationsComputationsContainer'], used_by=[], creation_date='2025-02-18 09:45', related_items=[],
        requires_global_keys=[], provides_global_keys=['EpochComputations'],
        validate_computation_test=validate_has_non_PBE_epoch_results, is_global=True)
    def perform_compute_non_PBE_epochs(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, training_data_portion: float=(5.0/6.0), epochs_decoding_time_bin_size: float = 0.050, frame_divide_bin_size:float=10.0,
                                        compute_1D: bool = True, compute_2D: bool = False, drop_previous_result_and_compute_fresh:bool=False, skip_training_test_split: bool = True, debug_print_memory_breakdown: bool=False):
        """ Performs the computation of non-PBE epochs for the session and all filtered epochs. Stacks things up hardcore yeah.

        Requires:
            ['sess']

        Provides:
            global_computation_results.computed_data['EpochComputations']
                ['EpochComputations'].odd_ripple
                ['RankOrder'].even_ripple
                ['RankOrder'].odd_laps
                ['RankOrder'].even_laps


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
            filter_epochs_to_decode_dict: Dict[KnownNamedDecodingEpochsType, Epoch] = {'laps': ensure_Epoch(deepcopy(global_any_laps_epochs_obj)),
                                                                                    'pbe': ensure_Epoch(deepcopy(global_session.pbe.get_non_overlapping())),
                                            #  'ripple': ensure_Epoch(deepcopy(global_session.ripple)),
                                            #   'replay': ensure_Epoch(deepcopy(global_replays_df)),
                                            'non_pbe': ensure_Epoch(deepcopy(global_session.non_pbe)),
                                            }
            # filter_epochs_to_decode_dict

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
                          requires_global_keys=['EpochComputations'], provides_global_keys=['EpochComputations'],
                          uses=['GeneralizedDecodedEpochsComputationsContainer', 'GenericDecoderDictDecodedEpochsDictResult'], used_by=[], creation_date='2025-04-14 12:40',
        validate_computation_test=validate_has_generalized_specific_epochs_decoding, is_global=True)
    def perform_generalized_specific_epochs_decoding(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, epochs_decoding_time_bin_size: float = 0.050, drop_previous_result_and_compute_fresh:bool=False, force_recompute:bool=False):
        """ 

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
            
        return global_computation_results





# ==================================================================================================================== #
# Display Functions/Plotting                                                                                           #
# ==================================================================================================================== #

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
import pyqtgraph as pg
import pyqtgraph.exporters
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer # for context_nested_docks/single_context_nested_docks

from neuropy.utils.indexing_helpers import paired_incremental_sorting, union_of_arrays, intersection_of_arrays
import plotly.express as px

class EpochComputationDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ RankOrderGlobalDisplayFunctions
    These display functions compare results across several contexts.
    Must have a signature of: (owning_pipeline_reference, global_computation_results, computation_results, active_configs, ..., **kwargs) at a minimum
    """


    @function_attributes(short_name='generalized_decoded_yellow_blue_marginal_epochs', tags=['yellow-blue-plots', 'directional_merged_decoder_decoded_epochs', 'directional'], conforms_to=['output_registering', 'figure_saving'], input_requires=[], output_provides=[], uses=['plot_decoded_epoch_slices'], used_by=[], creation_date='2024-01-04 02:59', related_items=[], is_global=True)
    def _display_generalized_decoded_yellow_blue_marginal_epochs(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, save_figure=True, included_any_context_neuron_ids=None,
                                                    single_plot_fixed_height=50.0, max_num_lap_epochs: int = 25, max_num_ripple_epochs: int = 45, size=(15,7), dpi=72, constrained_layout=True, scrollable_figure=True,
                                                    skip_plotting_measured_positions=True, skip_plotting_most_likely_positions=True, **kwargs):
            """ Renders two windows, one with the decoded laps and another with the decoded ripple posteriors, computed using the merged pseudo-2D decoder.

            """
            from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol
            # from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_decoded_epoch_slices
            from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
            from neuropy.utils.matplotlib_helpers import get_heatmap_cmap
            from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers

            import matplotlib as mpl
            import matplotlib.pyplot as plt
            from flexitext import flexitext ## flexitext for formatted matplotlib text

            from pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots import FigureCollector
            from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import PlottingHelpers
            from neuropy.utils.matplotlib_helpers import FormattedFigureText
        

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

            defer_render = kwargs.pop('defer_render', False)
            debug_print: bool = kwargs.pop('debug_print', False)
            active_config_name: bool = kwargs.pop('active_config_name', None)

            perform_write_to_file_callback = kwargs.pop('perform_write_to_file_callback', (lambda final_context, fig: owning_pipeline_reference.output_figure(final_context, fig)))


            ## INPUTS: a_new_fully_generic_result

            # a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', masked_time_bin_fill_type='dropped', data_grain='per_time_bin') # , known_named_decoding_epochs_type='laps', time_bin_size=0.025
            a_target_context: IdentifyingContext = IdentifyingContext(trained_compute_epochs='laps', pfND_ndim=1, decoder_identifier='pseudo2D', known_named_decoding_epochs_type='global', masked_time_bin_fill_type='nan_filled', data_grain='per_time_bin') # , known_named_decoding_epochs_type='laps', time_bin_size=0.025
            # flat_context_list, flat_result_context_dict, flat_decoder_context_dict, flat_decoded_marginal_posterior_df_context_dict = a_new_fully_generic_result.get_results_matching_contexts(context_query=a_target_context, return_multiple_matches=True, debug_print=False)
            # flat_context_list
            # flat_decoded_marginal_posterior_df_context_dict
            best_matching_context, a_result, a_decoder, a_decoded_marginal_posterior_df = a_new_fully_generic_result.get_results_best_matching_context(context_query=a_target_context, debug_print=False)

            print(f'best_matching_context: {best_matching_context}')
            ## OUTPUTS: flat_decoder_context_dict


            # Extract kwargs for figure rendering
            render_merged_pseudo2D_decoder_laps = kwargs.pop('render_merged_pseudo2D_decoder_laps', False)
            
            render_directional_marginal_laps = kwargs.pop('render_directional_marginal_laps', True)
            render_directional_marginal_ripples = kwargs.pop('render_directional_marginal_ripples', False)
            render_track_identity_marginal_laps = kwargs.pop('render_track_identity_marginal_laps', False)
            render_track_identity_marginal_ripples = kwargs.pop('render_track_identity_marginal_ripples', False)

            directional_merged_decoders_result = kwargs.pop('directional_merged_decoders_result', None)
            if directional_merged_decoders_result is not None:
                print("WARN: User provided a custom directional_merged_decoders_result as a kwarg. This will be used instead of the computed result global_computation_results.computed_data['DirectionalMergedDecoders'].")
                
            else:
                directional_merged_decoders_result = global_computation_results.computed_data['DirectionalMergedDecoders']


            # get the time bin size from the decoder:
            laps_decoding_time_bin_size: float = directional_merged_decoders_result.laps_decoding_time_bin_size
            ripple_decoding_time_bin_size: float = directional_merged_decoders_result.ripple_decoding_time_bin_size


            # figure_name: str = kwargs.pop('figure_name', 'directional_laps_overview_figure')
            # _out_data = RenderPlotsData(name=figure_name, out_colors_heatmap_image_matrix_dicts={})

            # Recover from the saved global result:
            # directional_laps_results = global_computation_results.computed_data['DirectionalLaps']
            # directional_merged_decoders_result = global_computation_results.computed_data['DirectionalMergedDecoders']

            # requires `laps_is_most_likely_direction_LR_dir` from `laps_marginals`
            long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.find_LongShortGlobal_epoch_names()

            graphics_output_dict = {}

            # Shared active_decoder, global_session:
            active_decoder = directional_merged_decoders_result.all_directional_pf1D_Decoder
            global_session = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name]) # used for validate_lap_dir_estimations(...) 

            # 'figure.constrained_layout.use': False, 'figure.autolayout': False, 'figure.subplot.bottom': 0.11, 'figure.figsize': (6.4, 4.8)
            # 'figure.constrained_layout.use': constrained_layout, 'figure.autolayout': False, 'figure.subplot.bottom': 0.11, 'figure.figsize': (6.4, 4.8)
            with mpl.rc_context({'figure.dpi': '220', 'savefig.transparent': True, 'ps.fonttype': 42, 'figure.constrained_layout.use': (constrained_layout or False), 'figure.frameon': False, }): # 'figure.figsize': (12.4, 4.8), 
                # Create a FigureCollector instance
                with FigureCollector(name='plot_directional_merged_pf_decoded_epochs', base_context=display_context) as collector:

                    ## Define the overriden plot function that internally calls the normal plot function but also permits doing operations before and after, such as building titles or extracting figures to save them:
                    def _mod_plot_decoded_epoch_slices(*args, **subfn_kwargs):
                        """ implicitly captures: owning_pipeline_reference, collector, perform_write_to_file_callback, save_figure, skip_plotting_measured_positions=True, skip_plotting_most_likely_positions=True

                        NOTE: each call requires adding the additional kwarg: `_main_context=_main_context`
                        """
                        assert '_mod_plot_kwargs' in subfn_kwargs
                        _mod_plot_kwargs = subfn_kwargs.pop('_mod_plot_kwargs')
                        assert 'final_context' in _mod_plot_kwargs
                        _main_context = _mod_plot_kwargs['final_context']
                        assert _main_context is not None
                        # Build the rest of the properties:
                        sub_context: IdentifyingContext = owning_pipeline_reference.build_display_context_for_session('directional_merged_pf_decoded_epochs', **_main_context)
                        sub_context_str: str = sub_context.get_description(subset_includelist=['t_bin'], include_property_names=True) # 't-bin_0.5' # str(sub_context.get_description())
                        modified_name: str = subfn_kwargs.pop('name', '')
                        if len(sub_context_str) > 0:
                            modified_name = f"{modified_name}_{sub_context_str}"
                        subfn_kwargs['name'] = modified_name # update the name by appending 't-bin_0.5'
                        
                        # Call the main plot function:
                        out_plot_tuple = plot_decoded_epoch_slices(*args, skip_plotting_measured_positions=skip_plotting_measured_positions, skip_plotting_most_likely_positions=skip_plotting_most_likely_positions, **subfn_kwargs)
                        # Post-plot call:
                        assert len(out_plot_tuple) == 4
                        params, plots_data, plots, ui = out_plot_tuple # [2] corresponds to 'plots' in params, plots_data, plots, ui = laps_plots_tuple
                        # post_hoc_append to collector
                        mw = ui.mw # MatplotlibTimeSynchronizedWidget
                        
                        y_bin_labels = _mod_plot_kwargs.get('y_bin_labels', None)
                        if y_bin_labels is not None:
                            label_artists_dict = {}
                            for i, ax in enumerate(plots.axs):
                                label_artists_dict[ax] = PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels(ax, y_bin_labels=y_bin_labels, enable_draw_decoder_colored_lines=False)
                            plots['label_artists_dict'] = label_artists_dict
                            

                        if mw is not None:
                            fig = mw.getFigure()
                            collector.post_hoc_append(figs=mw.fig, axes=mw.axes, contexts=sub_context)
                            title = mw.params.name
                        else:
                            fig = plots.fig
                            collector.post_hoc_append(figs=fig, axes=plots.axs, contexts=sub_context)
                            title = params.name

                        # Recover the proper title:
                        assert title is not None, f"title: {title}"
                        print(f'title: {title}')
                        
                        if ((perform_write_to_file_callback is not None) and (sub_context is not None)):
                            if save_figure:
                                perform_write_to_file_callback(sub_context, fig)
                            
                        # Close if defer_render
                        if defer_render:
                            if mw is not None:
                                mw.close()

                        return out_plot_tuple
                    ## END def _mod_plot_decoded_epoch_slices...


                    if render_merged_pseudo2D_decoder_laps:
                        # Merged Pseduo2D Decoder Posteriors:
                        _main_context = {'decoded_epochs': 'Laps', 'Pseudo2D': 'Posterior', 't_bin': laps_decoding_time_bin_size}
                        global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs) # global_epoch_name='maze_any'
                        graphics_output_dict['raw_posterior_laps_plot_tuple'] = _mod_plot_decoded_epoch_slices(
                            global_any_laps_epochs_obj, directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result,
                            global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                            name='Directional_Posterior',
                            active_marginal_fn=lambda filter_epochs_decoder_result: DirectionalPseudo2DDecodersResult.build_non_marginalized_raw_posteriors(filter_epochs_decoder_result),
                            single_plot_fixed_height=single_plot_fixed_height, debug_test_max_num_slices=max_num_lap_epochs,
                            size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure,
                            _mod_plot_kwargs=dict(final_context=_main_context, y_bin_labels=['long_LR', 'long_RL', 'short_LR', 'short_RL']),
                            **deepcopy(kwargs)
                        )            
                        # identifier_name, widget, matplotlib_fig, matplotlib_fig_axes = output_dict[a_posterior_name]
                        # label_artists_dict = {}
                        # for i, ax in enumerate(matplotlib_fig_axes):
                        #     label_artists_dict[ax] = PlottingHelpers.helper_matplotlib_add_pseudo2D_marginal_labels(ax, y_bin_labels=['long_LR', 'long_RL', 'short_LR', 'short_RL'], enable_draw_decoder_colored_lines=False)
                        # output_dict[a_posterior_name] = (identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, label_artists_dict)
                        
                        

                    if render_track_identity_marginal_laps:
                        # Laps Track-identity (Long/Short) Marginal:
                        _main_context = {'decoded_epochs': 'Laps', 'Marginal': 'TrackID', 't_bin': laps_decoding_time_bin_size}
                        global_any_laps_epochs_obj = deepcopy(owning_pipeline_reference.computation_results[global_epoch_name].computation_config.pf_params.computation_epochs) # global_epoch_name='maze_any'
                        # global_session = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name]) # used for validate_lap_dir_estimations(...) 
                        graphics_output_dict['track_identity_marginal_laps_plot_tuple'] = _mod_plot_decoded_epoch_slices(
                            global_any_laps_epochs_obj, directional_merged_decoders_result.all_directional_laps_filter_epochs_decoder_result,
                            global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                            name='TrackIdentity_Marginal_LAPS',
                            active_marginal_fn=lambda filter_epochs_decoder_result: DirectionalPseudo2DDecodersResult.build_custom_marginal_over_long_short(filter_epochs_decoder_result),
                            single_plot_fixed_height=single_plot_fixed_height, debug_test_max_num_slices=max_num_lap_epochs,
                            size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure,
                            _mod_plot_kwargs=dict(final_context=_main_context, y_bin_labels=['long', 'short']),
                            **deepcopy(kwargs)
                        )


                    if render_track_identity_marginal_ripples:
                        # Ripple Track-identity (Long/Short) Marginal:
                        _main_context = {'decoded_epochs': 'Ripple', 'Marginal': 'TrackID', 't_bin': ripple_decoding_time_bin_size}
                        global_replays = TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(deepcopy(global_session.replay))
                        # global_session = deepcopy(owning_pipeline_reference.filtered_sessions[global_epoch_name]) # used for validate_lap_dir_estimations(...) 
                        graphics_output_dict['track_identity_marginal_ripples_plot_tuple'] = _mod_plot_decoded_epoch_slices(
                            global_replays, directional_merged_decoders_result.all_directional_ripple_filter_epochs_decoder_result,
                            global_pos_df=global_session.position.to_dataframe(), xbin=active_decoder.xbin,
                            name='TrackIdentity_Marginal_Ripples',
                            active_marginal_fn=lambda filter_epochs_decoder_result: DirectionalPseudo2DDecodersResult.build_custom_marginal_over_long_short(filter_epochs_decoder_result),
                            single_plot_fixed_height=single_plot_fixed_height, debug_test_max_num_slices=max_num_ripple_epochs,
                            size=size, dpi=dpi, constrained_layout=constrained_layout, scrollable_figure=scrollable_figure,
                            _mod_plot_kwargs=dict(final_context=_main_context, y_bin_labels=['long', 'short']),
                            **deepcopy(kwargs)
                        )


            graphics_output_dict['collector'] = collector

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
                