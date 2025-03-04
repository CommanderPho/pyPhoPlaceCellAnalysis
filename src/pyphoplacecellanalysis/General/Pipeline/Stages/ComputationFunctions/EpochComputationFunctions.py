from copy import deepcopy
import sys
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import pyphoplacecellanalysis.General.type_aliases as types
import numpy as np
import pandas as pd

# NeuroPy (Diba Lab Python Repo) Loading
from neuropy.utils.mixins.binning_helpers import build_df_discretized_binned_position_columns
from neuropy.utils.dynamic_container import DynamicContainer # for _perform_two_step_position_decoding_computation
from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs # used in _subfn_compute_decoded_epochs to get only the valid (non-overlapping) epochs
from neuropy.core.epoch import Epoch, subdivide_epochs, ensure_dataframe, ensure_Epoch
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
# import portion as P # Required for interval search: portion~=2.3.0
from pyphocorehelpers.indexing_helpers import partition_df_dict, partition_df

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df

from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, SimpleFieldSizesReprMixin


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
    



    @function_attributes(short_name=None, tags=['epochs', 'non-PBE', 'session'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-28 04:10', related_items=[]) 
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
        PBE_df: pd.DataFrame = ensure_dataframe(deepcopy(sess.pbe))
        ## Build up a new epoch
        epochs_df: pd.DataFrame = deepcopy(sess.epochs).epochs.adding_global_epoch_row()
        global_epoch_only_df: pd.DataFrame = epochs_df.epochs.label_slice('maze')
        
        # t_start, t_stop = epochs_df.epochs.t_start, epochs_df.epochs.t_stop
        global_epoch_only_non_PBE_epoch_df: pd.DataFrame = global_epoch_only_df.epochs.subtracting(PBE_df)
        global_epoch_only_non_PBE_epoch_df= global_epoch_only_non_PBE_epoch_df.epochs.modify_each_epoch_by(additive_factor=-0.008, final_output_minimum_epoch_duration=0.040)
        

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

        maze_id_to_maze_name_map = {-1:'none', 0:'long', 1:'short'}
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
        long_session.non_pbe
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
    @function_attributes(short_name=None, tags=['non_PBE', 'epochs', 'update', 'pipeline'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-18 18:49', related_items=[])
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
        
        # Update main session
        sess.non_pbe = DataSession.compute_non_PBE_epochs(sess, save_on_compute=save_on_compute)
        
        # Check if main session changed - compare the dataframes directly
        did_change = did_change or (original_non_pbe is None) or (not original_non_pbe.to_dataframe().equals(sess.non_pbe.to_dataframe()))
        
        # Update filtered sessions if provided
        if filtered_sessions is not None:
            for filter_name, filtered_session in filtered_sessions.items():
                original_filtered_non_pbe = deepcopy(getattr(filtered_session, 'non_pbe', None))
                filtered_session.non_pbe = sess.non_pbe.time_slice(t_start=filtered_session.t_start, t_stop=filtered_session.t_stop)
                # Check if filtered session changed
                did_change = did_change or (original_filtered_non_pbe is None) or (not original_filtered_non_pbe.to_dataframe().equals(filtered_session.non_pbe.to_dataframe()))
                    
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

@define(slots=False, repr=False, eq=False)
class EpochComputationsComputationsContainer(ComputedResult):
    """ Holds the result from a single rank-ordering (odd/even) comparison between odd/even


    Usage:

        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import EpochComputationsComputationsContainer

        wcorr_shuffle_results: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data.get('EpochComputations', None)
        if wcorr_shuffle_results is not None:    
            wcorr_ripple_shuffle: WCorrShuffle = wcorr_shuffle_results.wcorr_ripple_shuffle
            print(f'wcorr_ripple_shuffle.n_completed_shuffles: {wcorr_ripple_shuffle.n_completed_shuffles}')
        else:
            print(f'EpochComputations is not computed.')
            
    """
    _VersionedResultMixin_version: str = "2025.02.18_0" # to be updated in your IMPLEMENTOR to indicate its version
    
    training_data_portion: float = serialized_attribute_field(default=(5.0/6.0))
    epochs_decoding_time_bin_size: float = serialized_attribute_field(default=0.020) 
    frame_divide_bin_size: float = serialized_attribute_field(default=10.0)

    a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = serialized_field(default=None, repr=False)
    results1D: Optional[NonPBEDimensionalDecodingResult] = serialized_field(default=None, repr=False)
    results2D: Optional[NonPBEDimensionalDecodingResult] = serialized_field(default=None, repr=False)


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
            
            non_PBE_all_directional_pf1D_Decoder, pseudo2D_continuous_specific_decoded_result, non_PBE_marginal_over_track_ID, (time_bin_containers, time_window_centers) = nonPBE_results._build_merged_joint_placefields_and_decode(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)))
            
            
        """
        from neuropy.analyses.placefields import PfND
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult
        from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df

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
        non_PBE_marginal_over_track_ID = DirectionalPseudo2DDecodersResult.build_generalized_non_marginalized_raw_posteriors(pseudo2D_continuous_specific_decoded_result, unique_decoder_names=unique_decoder_names)[0]['p_x_given_n']
        time_bin_containers = pseudo2D_continuous_specific_decoded_result.time_bin_containers[0]
        time_window_centers = time_bin_containers.centers
        # p_x_given_n.shape # (62, 4, 209389)

        ## Build into a marginal df like `all_sessions_laps_df` - uses `time_window_centers`, pseudo2D_continuous_specific_decoded_result, non_PBE_marginal_over_track_ID:
        track_marginal_posterior_df : pd.DataFrame = pd.DataFrame({'t':deepcopy(time_window_centers), 'P_Long': np.squeeze(non_PBE_marginal_over_track_ID[0, :]), 'P_Short': np.squeeze(non_PBE_marginal_over_track_ID[1, :]), 'time_bin_size': pseudo2D_continuous_specific_decoded_result.decoding_time_bin_size})
        # track_marginal_posterior_df['delta_aligned_start_t'] = track_marginal_posterior_df['t'] - t_delta ## subtract off t_delta
        track_marginal_posterior_df


        ## OUTPUTS: non_PBE_marginal_over_track_ID, time_bin_containers, time_window_centers
        return non_PBE_all_directional_pf1D_Decoder, pseudo2D_continuous_specific_decoded_result, continuous_decoded_results_dict, non_PBE_marginal_over_track_ID, (time_bin_containers, time_window_centers, track_marginal_posterior_df)
        

    # ==================================================================================================================== #
    # Plotting Methods                                                                                                     #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['UNTESTED'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-21 13:40', related_items=[])
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
    """ Returns True if the pipeline has a valid RankOrder results set of the latest version

    TODO: make sure minimum can be passed. Actually, can get it from the pipeline.

    """
    # Unpacking:
    seq_results: EpochComputationsComputationsContainer = curr_active_pipeline.global_computation_results.computed_data['EpochComputations']
    if seq_results is None:
        return False
    
    a_new_NonPBE_Epochs_obj = seq_results.a_new_NonPBE_Epochs_obj
    if a_new_NonPBE_Epochs_obj is None:
        return False



    # _computationPrecidence = 2 # must be done after PlacefieldComputations, DefaultComputationFunctions
    # _is_global = False

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
    n_time_bins = int(np.ceil(session_duration / epochs_decoding_time_bin_size))
    n_frame_divided_bins = int(np.ceil(session_duration / frame_divide_bin_size))
    
    n_max_decoded_bins: int = max(n_time_bins, n_frame_divided_bins)
    bytes_per_float = 8  # float64
    
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
    
    intermediate_itemized_memory_breakdown = {
        'worst_case_scenario': (n_max_decoded_bins * n_flattened_position_bins * n_neurons) * bytes_per_float, 
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
        input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-18 09:45', related_items=[],
        requires_global_keys=[], provides_global_keys=['EpochComputations'],
        validate_computation_test=validate_has_non_PBE_epoch_results, is_global=True)
    def perform_compute_non_PBE_epochs(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, training_data_portion: float=(5.0/6.0), epochs_decoding_time_bin_size: float = 0.025, frame_divide_bin_size:float=10.0,
                                        compute_1D: bool = True, compute_2D: bool = True, drop_previous_result_and_compute_fresh:bool=False, skip_training_test_split: bool = True, debug_print_memory_breakdown: bool=False):
        """ Performs the computation of the spearman and pearson correlations for the ripple and lap epochs.

        Requires:
            ['sess']

        Provides:
            global_computation_results.computed_data['EpochComputations']
                ['EpochComputations'].odd_ripple
                ['RankOrder'].even_ripple
                ['RankOrder'].odd_laps
                ['RankOrder'].even_laps


        """
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
                required_memory_GB: int = required_memory_bytes / 1e9 # Gigabytes
                itemized_mem_breakdown = {f"{k}_{n_dim_str}":v for k, v in itemized_mem_breakdown.items()}
                return required_memory_GB, itemized_mem_breakdown

            itemized_mem_breakdown = {}
            total_required_memory_GB: int = 0
            if compute_1D:
                n_flattened_position_bins_1D: int = computation_results[global_epoch_name].computed_data.pf1D.n_flattened_position_bins
                required_memory_GB_1D, breakdown1D = _subfn_perform_estimate_required_memory(specific_n_flattened_position_bins=n_flattened_position_bins_1D, n_dim_str="1D")
                itemized_mem_breakdown.update(breakdown1D)
                total_required_memory_GB += required_memory_GB_1D
            if compute_2D:
                n_flattened_position_bins_2D: int = computation_results[global_epoch_name].computed_data.pf2D.n_flattened_position_bins
                required_memory_GB_2D, breakdown2D = _subfn_perform_estimate_required_memory(specific_n_flattened_position_bins=n_flattened_position_bins_2D, n_dim_str="2D")
                itemized_mem_breakdown.update(breakdown2D)
                total_required_memory_GB += required_memory_GB_2D
            
            itemized_mem_breakdown_GB = {k:v/1e9 for k, v in itemized_mem_breakdown.items()}
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
                # initialize a new wcorr result
                a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = Compute_NonPBE_Epochs.init_from_pipeline(curr_active_pipeline=owning_pipeline_reference, training_data_portion=training_data_portion, skip_training_test_split=skip_training_test_split)
                global_computation_results.computed_data['EpochComputations'].a_new_NonPBE_Epochs_obj = a_new_NonPBE_Epochs_obj
            else:
                ## get the existing one:
                a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = global_computation_results.computed_data['EpochComputations'].a_new_NonPBE_Epochs_obj
            
            ## apply the new epochs to the session:
            owning_pipeline_reference.filtered_sessions[global_epoch_name].non_PBE = deepcopy(a_new_NonPBE_Epochs_obj.global_epoch_only_non_PBE_epoch_df)

            results1D, results2D = a_new_NonPBE_Epochs_obj.compute_all(owning_pipeline_reference, epochs_decoding_time_bin_size=epochs_decoding_time_bin_size, frame_divide_bin_size=frame_divide_bin_size, compute_1D=compute_1D, compute_2D=compute_2D, skip_training_test_split=skip_training_test_split)
            if (results1D is not None) and compute_1D:
                global_computation_results.computed_data['EpochComputations'].results1D = results1D

            if (results2D is not None) and compute_2D:
                global_computation_results.computed_data['EpochComputations'].results2D = results2D
                
            global_computation_results.computed_data['EpochComputations'].a_new_NonPBE_Epochs_obj = a_new_NonPBE_Epochs_obj
            
        except MemoryError as mem_err:
            print(f"Insufficient memory: {str(mem_err)}")
            raise 
            # raise MemoryError(f"Insufficient memory: {str(mem_err)}")
        
        except Exception as e:
            raise
            # return None, f"Computation failed: {str(e)}"

        
        return global_computation_results
    



    # @computation_precidence_specifying_function(overriden_computation_precidence=-0.1)
    # @function_attributes(short_name='local_compute_non_PBE_epochs', tags=['epochs', 'nonPBE'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-02-18 09:45', related_items=[],
    #     validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['firing_rate_trends'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['extended_stats']['time_binned_position_df']), is_global=False)
    # def local_perform_compute_non_PBE_epochs(computation_result: ComputationResult, **kwargs):
    #     """ Adds the 'is_LR_dir' column to the laps dataframe and updates 'lap_dir' if needed.        
    #     """
    #     a_new_NonPBE_Epochs_obj: Compute_NonPBE_Epochs = Compute_NonPBE_Epochs.ini (curr_active_pipeline=owning_pipeline_reference, training_data_portion=training_data_portion)
        
    #     computation_result.sess.laps.update_lap_dir_from_smoothed_velocity(pos_input=computation_result.sess.position) # confirmed in-place
    #     # computation_result.sess.laps.update_lap_dir_from_smoothed_velocity(pos_input=computation_result.sess.position)
    #     # curr_sess.laps.update_maze_id_if_needed(t_start=t_start, t_delta=t_delta, t_end=t_end) # this doesn't make sense for the filtered sessions unfortunately.
    #     return computation_result # no changes except to the internal sessions
    

    # @function_attributes(short_name='_perform_specific_epochs_decoding', tags=['BasePositionDecoder', 'computation', 'decoder', 'epoch'],
    #                       input_requires=[ "computation_result.computed_data['pf1D_Decoder']", "computation_result.computed_data['pf2D_Decoder']"], output_provides=["computation_result.computed_data['specific_epochs_decoding']"],
    #                       uses=[], used_by=[], creation_date='2023-04-07 02:16',
    #     validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['specific_epochs_decoding']), is_global=False)
    # def _perform_specific_epochs_decoding(computation_result: ComputationResult, active_config, decoder_ndim:int=2, filter_epochs='ripple', decoding_time_bin_size=0.02, **kwargs):
    #     """ TODO: meant to be used by `_display_plot_decoded_epoch_slices` but needs a smarter way to cache the computations and etc. 
    #     Eventually to replace `pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError._compute_specific_decoded_epochs`

    #     Usage:
    #         ## Test _perform_specific_epochs_decoding
    #         from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.EpochComputationFunctions import EpochComputationFunctions
    #         computation_result = curr_active_pipeline.computation_results['maze1_PYR']
    #         computation_result = EpochComputationFunctions._perform_specific_epochs_decoding(computation_result, curr_active_pipeline.active_configs['maze1_PYR'], filter_epochs='ripple', decoding_time_bin_size=0.02)
    #         filter_epochs_decoder_result, active_filter_epochs, default_figure_name = computation_result.computed_data['specific_epochs_decoding'][('Ripples', 0.02)]

    #     """
    #     ## BEGIN_FUNCTION_BODY _perform_specific_epochs_decoding:
    #     ## Check for previous computations:
    #     needs_compute = True # default to needing to recompute.
    #     computation_tuple_key = (filter_epochs, decoding_time_bin_size, decoder_ndim) # used to be (default_figure_name, decoding_time_bin_size) only

    #     curr_result = computation_result.computed_data.get('specific_epochs_decoding', {})
    #     found_result = curr_result.get(computation_tuple_key, None)
    #     if found_result is not None:
    #         # Unwrap and reuse the result:
    #         filter_epochs_decoder_result, active_filter_epochs, default_figure_name = found_result # computation_result.computed_data['specific_epochs_decoding'][('Laps', decoding_time_bin_size)]
    #         needs_compute = False # we don't need to recompute

    #     if needs_compute:
    #         ## Do the computation:
    #         filter_epochs_decoder_result, active_filter_epochs, default_figure_name = _subfn_compute_decoded_epochs(computation_result, active_config, filter_epochs=filter_epochs, decoding_time_bin_size=decoding_time_bin_size, decoder_ndim=decoder_ndim)

    #         ## Cache the computation result via the tuple key: (default_figure_name, decoding_time_bin_size) e.g. ('Laps', 0.02) or ('Ripples', 0.02)
    #         curr_result[computation_tuple_key] = (filter_epochs_decoder_result, active_filter_epochs, default_figure_name)

    #     computation_result.computed_data['specific_epochs_decoding'] = curr_result
    #     return computation_result



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
                