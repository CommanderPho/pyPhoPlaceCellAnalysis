import sys
from copy import deepcopy
from datetime import timedelta, datetime
from enum import unique, Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

import tables as tb
from attr import define, field, Factory

from neuropy.core import Epoch
from pyphocorehelpers.Filesystem.metadata_helpers import FilesystemMetadata
from pyphocorehelpers.exception_helpers import ExceptionPrintingContext, CapturedException
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionsResults
from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_evaluate_required_computations, batch_extended_computations
from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import main_complete_figure_generations
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import LongShortPipelineTests, JonathanFiringRateAnalysisResult, InstantaneousSpikeRateGroupsComputation
from neuropy.utils.dynamic_container import DynamicContainer
from neuropy.utils.matplotlib_helpers import matplotlib_file_only
from neuropy.utils.mixins.AttrsClassHelpers import custom_define, AttrsBasedClassHelperMixin, serialized_attribute_field, serialized_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_SerializationMixin

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsHelpers

@unique
class SavingOptions(Enum):
    NEVER = "NEVER"
    IF_CHANGED = "IF_CHANGED"
    ALWAYS = "ALWAYS"


@custom_define(slots=False)
class BatchComputationProcessOptions(HDF_SerializationMixin, AttrsBasedClassHelperMixin):
    """ specifies how computations should be ran. Should they be loaded from previously saved data? Computed? Saved? """
    should_load: bool = serialized_attribute_field() # should try to load from existing results from disk at all
        # never
        # always (fail if loading unsuccessful)
        # always (warning but continue if unsuccessful)
    should_compute: bool = serialized_attribute_field() # should try to run computations (which will just verify that loaded computations are good if that option is true)
        # never
        # if needed (required results are missing)
        # always
    should_save: SavingOptions = serialized_attribute_field(default=SavingOptions.NEVER) # should consider save at all
        # never
        # if changed
        # always
    override_file: Optional[Union[str,Path]] = serialized_attribute_field(default=None) # 'output/loadedSessPickle.pkl'
    override_output_file: Optional[Union[str,Path]] = serialized_attribute_field(default=None)

    # override_output_file
    def __attrs_post_init__(self):
        """ called after initializer built by `attrs` library. """
        if self.override_file is not None:
            if self.override_output_file is None:
                # Want the output to default to the input
                self.override_output_file = self.override_file
             


@custom_define(slots=False)
class PipelineCompletionResult(HDF_SerializationMixin, AttrsBasedClassHelperMixin):
    """ Class representing the specific results extracted from the loaded pipeline and returned as return values from the post-execution callback function. """
    long_epoch_name: str = serialized_attribute_field()
    long_laps: Epoch = serialized_field()
    long_replays: Epoch = serialized_field()

    short_epoch_name: str = serialized_attribute_field()
    short_laps: Epoch = serialized_field()
    short_replays: Epoch = serialized_field()

    delta_since_last_compute: timedelta = non_serialized_field() #serialized_attribute_field(serialization_fn=HDF_Converter._prepare_datetime_timedelta_value_to_for_hdf_fn)
    outputs_local: Dict[str, Optional[Path]] = non_serialized_field() # serialization_fn=(lambda f, k, v: f[f'{key}/{sub_k}'] = str(sub_v) for sub_k, sub_v in value.items()), is_hdf_handled_custom=True
    outputs_global: Dict[str, Optional[Path]] = non_serialized_field()

    across_session_results: Dict[str, Optional[object]] = non_serialized_field()

     # HDFMixin Conformances ______________________________________________________________________________________________ #

    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path"""
        super().to_hdf(file_path, key=key, **kwargs)
        # Finish for the custom properties

        #TODO 2023-08-04 12:09: - [ ] Included outputs_local/global

        # with tb.open_file(file_path, mode='a') as f:

        #     outputs_local_key = f"{key}/outputs_local"
        #     an_outputs_local_group = f.create_group(key, 'outputs_local', title='the sessions output file paths.', createparents=True)

        #     value = self.outputs_local
        #     for sub_k, sub_v in value.items():
        #         an_outputs_local_group[f'{outputs_local_key}/{sub_k}'] = str(sub_v)

        #     an_outputs_global_group = f.create_group(key, 'outputs_global', title='the sessions output file paths.', createparents=True)
        #     value = self.outputs_global
        #     outputs_global_key = f"{key}/outputs_global"
        #     for sub_k, sub_v in value.items():
        #         an_outputs_global_group[f'{outputs_global_key}/{sub_k}'] = str(sub_v)


class PipelineCompletionResultTable(tb.IsDescription):
    """ PyTables class representing epoch data built from a dictionary. """
    long_epoch_name = tb.StringCol(itemsize=100)
    # long_laps = EpochTable()
    # long_replays = EpochTable()
    long_n_laps = tb.UInt16Col()
    long_n_replays = tb.UInt16Col()
    short_epoch_name = tb.StringCol(itemsize=100)
    # short_laps = EpochTable()
    # short_replays = EpochTable()
    short_n_laps = tb.UInt16Col()
    short_n_replays = tb.UInt16Col()

    delta_since_last_compute = tb.Time64Col()  # Use tb.Time64Col for timedelta

    # outputs_local = OutputFilesTable()
    # outputs_global = OutputFilesTable()

    # across_sessions_batch_results_inst_fr_comps = tb.StringCol(itemsize=100)


@define(slots=False, repr=False)
class BatchSessionCompletionHandler:
    """ handles completion of a single session's batch processing.

    Allows accumulating results across sessions and runs.
    
    
    Holds powerful options that are used during its `on_complete_success_execution_session` function, which is always passed as the callback for `run_specific_batch`
    
    Passed to `batch_extended_computations(...)` for global computation function calculations:
        self.extended_computations_include_includelist
        self.force_recompute_override_computations_includelist
        self.force_recompute_override_computation_kwargs_dict # #TODO 2024-10-30 08:35: - [ ] is `force_recompute_override_computation_kwargs_dict` actually only used when forcing a recompute, or does passing it when it's the same as the already computed values force it to recompute?

        

    Usage:
        from pyphoplacecellanalysis.General.Batch.runBatch import BatchSessionCompletionHandler

    """

    # Completion Result object returned from callback ____________________________________________________________________ #

    # General:
    debug_print: bool = field(default=False)
    fail_on_exception: bool = field(default=False) # whether to raise exceptions that occur during the callback completion handler or not.

    force_reload_all: bool = field(default=False)
    saving_mode: PipelineSavingScheme = field(default=PipelineSavingScheme.SKIP_SAVING)

    # Multiprocessing
    use_multiprocessing: bool = field(default=False)
    num_processes: Optional[int] = field(default=None)

    # Computations
    # enable_full_pipeline_in_ram: bool = field(default=False)
    ## Error with enable_full_pipeline_in_ram=True:
     # 	delta_since_last_compute=datetime.timedelta(seconds=43, microseconds=466370), outputs_local={'pkl': PosixPath('/nfs/turbo/umms-kdiba/Data/KDIBA/gor01/one/2006-6-08_14-26-15/loadedSessPickle.pkl')}, outputs_global={'pkl': PosixPath('/nfs/turbo/umms-kdiba/Data/KDIBA/gor01/one/2006-6-08_14-26-15/output/global_computation_results.pkl'), 'hdf5': None}, across_session_results={'inst_fr_comps': None, 'curr_active_pipeline': <pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline.NeuropyPipeline object at 0x148e83474700>}))'. Reason: 'AttributeError("Can't pickle local object 'DataSessionFormatBaseRegisteredClass.build_default_filter_functions.<locals>.<dictcomp>.<lambda>'")'

    # a list of functions to be called upon completion, will be called sequentially. 
    completion_functions: List[Callable] = field(default=Factory(list))
    override_user_completion_function_kwargs_dict: Dict[Union[Callable, str], Dict] = field(default=Factory(dict))

    # override_session_computation_results_pickle_filename: Optional[str] = field(default=None) # 'output/loadedSessPickle.pkl'
    BATCH_DATE_TO_USE: str = field(default='0000-00-00_Fake') # BATCH_DATE_TO_USE = '2024-03-27_Apogee'
    collected_outputs_path: Path = field(default=None) # collected_outputs_path = Path(r'C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\output\collected_outputs').resolve()

    ## Computation Options:
    session_computations_options: BatchComputationProcessOptions = field(default=BatchComputationProcessOptions(should_load=True, should_compute=True, should_save=SavingOptions.IF_CHANGED))

    global_computations_options: BatchComputationProcessOptions = field(default=BatchComputationProcessOptions(should_load=True, should_compute=True, should_save=SavingOptions.IF_CHANGED))
    extended_computations_include_includelist: list = field(default=['lap_direction_determination', 'pf_computation', 'pfdt_computation', 'firing_rate_trends',
                                                                    # 'pf_dt_sequential_surprise',
                                                                    'extended_stats',
                                        'long_short_decoding_analyses', 'jonathan_firing_rate_analysis', 'long_short_fr_indicies_analyses', 'short_long_pf_overlap_analyses', 'long_short_post_decoding', # 'long_short_rate_remapping',
                                        # 'ratemap_peaks_prominence2d',
                                        #  'long_short_inst_spike_rate_groups',
                                        'long_short_endcap_analysis',
                                        # 'spike_burst_detection',
                                        'split_to_directional_laps',
                                        'merged_directional_placefields',
                                        'rank_order_shuffle_analysis',
                                        'directional_train_test_split',
                                        'directional_decoders_evaluate_epochs',
                                        # 'directional_decoders_epoch_heuristic_scoring',
                                    ]) # do only specified

    force_global_recompute: bool = field(default=False)
    
    force_recompute_override_computations_includelist: list = field(default=Factory(list)) # empty list by default. For example self.force_recompute_override_computations_includelist = ['rank_order_shuffle_analysis'] would force recomputation of that global computation function
    force_recompute_override_computation_kwargs_dict: list = field(default=Factory(dict))

    # @property
    # def override_session_computation_results_pickle_filename(self) -> Optional[str]:
    #     return self.session_computations_options.override_file
    # @override_session_computation_results_pickle_filename.setter
    # def override_session_computation_results_pickle_filename(self, value):
    #     self.session_computations_options.override_file = value


    # @property
    # def override_global_computation_results_pickle_path(self) -> Optional[Path]:
    #     return self.global_computations_options.override_file
    # @override_global_computation_results_pickle_path.setter
    # def override_global_computation_results_pickle_path(self, value):
    #     self.global_computations_options.override_file = value



    # Figures:
    should_perform_figure_generation_to_file: bool = field(default=True) # controls whether figures are generated to file
    should_generate_all_plots: bool = field(default=False) # controls whether all plots are generated (when True) or if only non-Neptune paper figure specific plots are generated. Has no effect if self.should_perform_figure_generation_to_file is False.


    # Cross-session Results:
    across_sessions_instantaneous_fr_dict: dict = Factory(dict) # Dict[IdentifyingContext] = InstantaneousSpikeRateGroupsComputation
    enable_hdf5_output: bool = field(default=False)


    # the firing rate LxC/SxC refinment criteria
    frs_index_inclusion_magnitude: float = field(default=0.35)
    override_existing_frs_index_values:bool = field(default=False)

    @classmethod
    def _post_fix_filtered_contexts(cls, curr_active_pipeline, debug_print=False) -> bool:
        """ 2023-10-24 - tries to update misnamed `curr_active_pipeline.filtered_contexts`

            curr_active_pipeline.filtered_contexts with correct filter_names

            Uses: `curr_active_pipeline.filtered_epoch`
            Updates: `curr_active_pipeline.filtered_contexts`
            
        Still needed for 2023-11-29 to add back in the 'lap_dir' key

        """
        was_updated = False
        was_updated = was_updated or DirectionalLapsHelpers.post_fixup_filtered_contexts(curr_active_pipeline)
        return was_updated

    @classmethod
    def _update_pipeline_missing_preprocessing_parameters(cls, curr_active_pipeline, debug_print=False):
        """ 2023-05-24 - Adds the previously missing `sess.config.preprocessing_parameters` to each session (filtered and base) in the pipeline.

        Usage:
            from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import _update_pipeline_missing_preprocessing_parameters
            was_updated = _update_pipeline_missing_preprocessing_parameters(curr_active_pipeline)
            was_updated
        """
        from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder
        from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
        from neuropy.core.session.Formats.SessionSpecifications import SessionConfig

        def _subfn_update_session_missing_preprocessing_parameters(sess):
            """ 2023-05-24 - Adds the previously missing `sess.config.preprocessing_parameters` to a single session. Called only by `_update_pipeline_missing_preprocessing_parameters` """
            preprocessing_parameters = getattr(sess.config, 'preprocessing_parameters', None)
            if preprocessing_parameters is None:
                print(f'No existing preprocessing parameters! Assigning them!')
                default_lap_estimation_parameters = DynamicContainer(N=20, should_backup_extant_laps_obj=True, use_direction_dependent_laps=True) # Passed as arguments to `sess.replace_session_laps_with_estimates(...)`
                default_PBE_estimation_parameters = DynamicContainer(sigma=0.030, thresh=(0, 1.5), min_dur=0.030, merge_dur=0.100, max_dur=0.600) # 2023-10-05 Kamran's imposed Parameters, wants to remove the effect of the max_dur which was previously at 0.300
                default_replay_estimation_parameters = DynamicContainer(require_intersecting_epoch=None, min_epoch_included_duration=0.06, max_epoch_included_duration=0.600, maximum_speed_thresh=None, min_inclusion_fr_active_thresh=0.01, min_num_unique_aclu_inclusions=5)

                sess.config.preprocessing_parameters = DynamicContainer(epoch_estimation_parameters=DynamicContainer.init_from_dict({
                        'laps': default_lap_estimation_parameters,
                        'PBEs': default_PBE_estimation_parameters,
                        'replays': default_replay_estimation_parameters
                    }))
                return True
            else:
                if debug_print:
                    print(f'preprocessing parameters exist.')
                # TODO: update them as needed?
                return False
            

        def _subfn_update_session_missing_loaded_track_limits(curr_active_pipeline, always_reload_from_file:bool):
            """ 2024-04-09 - Adds the previously missing `sess.config.loaded_track_limits` to a single session. Called only by `_update_pipeline_missing_preprocessing_parameters` """
            loaded_track_limits = getattr(curr_active_pipeline.sess.config, 'loaded_track_limits', None)
            if (loaded_track_limits is None) or always_reload_from_file:
                print(f'No existing loaded_track_limits parameters! Assigning them!')
                active_data_mode_name: str = curr_active_pipeline.session_data_type
                active_data_session_types_registered_classes_dict = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()
                active_data_mode_registered_class = active_data_session_types_registered_classes_dict[active_data_mode_name]
                a_session = deepcopy(curr_active_pipeline.sess)
                sess_config: SessionConfig = SessionConfig(**deepcopy(a_session.config.__getstate__()))
                a_session.config = sess_config
                # a_session = active_data_mode_registered_class._default_kdiba_exported_load_position_info_mat(basepath=curr_active_pipeline.sess.basepath, session_name=curr_active_pipeline.session_name, session=deepcopy(curr_active_pipeline.sess))
                a_session = active_data_mode_registered_class._default_kdiba_exported_load_position_info_mat(basepath=curr_active_pipeline.sess.basepath, session_name=curr_active_pipeline.session_name, session=a_session)
                # a_session
                curr_active_pipeline.stage.sess = a_session ## apply the session
                # curr_active_pipeline.sess.config = a_session.config # apply the config only...
                return True
            else:
                if debug_print:
                    print(f'loaded_track_limits parameters exist.')
                # TODO: update them as needed?
                return False
            

        # BEGIN MAIN FUNCTION BODY
        was_updated = False
        was_updated = was_updated | _subfn_update_session_missing_preprocessing_parameters(curr_active_pipeline.sess)
        was_updated = was_updated | _subfn_update_session_missing_loaded_track_limits(curr_active_pipeline, always_reload_from_file=False)

        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]:
            was_updated = was_updated | _subfn_update_session_missing_preprocessing_parameters(curr_active_pipeline.filtered_sessions[an_epoch_name])

        # if was_updated:
        #     print(f'config was updated. Saving pipeline.')
        #     curr_active_pipeline.save_pipeline(saving_mode=PipelineSavingScheme.OVERWRITE_IN_PLACE)

        return was_updated

    @function_attributes(short_name=None, tags=['MAIN'], input_requires=[], output_provides=[], uses=['cls._update_pipeline_missing_preprocessing_parameters', 'cls._post_fix_filtered_contexts', 'ComputationKWargParameters', 'PostHocPipelineFixup'], used_by=['on_load_local', 'on_complete_success_execution_session'], creation_date='2025-02-19 19:09', related_items=[])
    @classmethod
    def post_compute_validate(cls, curr_active_pipeline) -> bool:
        """ 2023-05-16 - Ensures that the laps are used for the placefield computation epochs, the number of bins are the same between the long and short tracks. 
        
        NOTE: returns `was_updated`, not `is_valid` or something similar.
        
        """
        from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import ComputationKWargParameters
        from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import PostHocPipelineFixup
        
        if not LongShortPipelineTests(curr_active_pipeline=curr_active_pipeline).validate():
            print(f'ERROR!! Pipeline is invalid according to LongShortPipelineTests!!')
            return False
        
        # 2023-05-24 - Adds the previously missing `sess.config.preprocessing_parameters` to each session (filtered and base) in the pipeline.
        was_updated = cls._update_pipeline_missing_preprocessing_parameters(curr_active_pipeline)
        print(f'were pipeline preprocessing parameters missing and updated?: {was_updated}')

        ## BUG 2023-05-25 - Found ERROR for a loaded pipeline where for some reason the filtered_contexts[long_epoch_name]'s actual context was the same as the short maze ('...maze2'). Unsure how this happened.
        was_updated = was_updated or cls._post_fix_filtered_contexts(curr_active_pipeline) #TODO 2024-11-01 19:39: - [ ] This is where things go amiss it seems

        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        # assert long_epoch_context.filter_name == long_epoch_name, f"long_epoch_context.filter_name: {long_epoch_context.filter_name} != long_epoch_name: {long_epoch_name}"
        # if long_epoch_context.filter_name != long_epoch_name:
        #     print(f"WARNING: filtered_contexts[long_epoch_name]'s actual context name is incorrect. \n\tlong_epoch_context.filter_name: {long_epoch_context.filter_name} != long_epoch_name: {long_epoch_name}\n\tUpdating it. (THIS IS A HACK)")
        #     # fix it if broken
        #     # long_epoch_context.filter_name = long_epoch_name
        #     # was_updated = True
        #     raise NotImplementedError("2023-11-29 - This shouldn't happen since we previously called `cls._post_fix_filtered_contexts(curr_active_pipeline)`!!")
        

        ## call `PostHocPipelineFixup.FINAL_UPDATE_ALL(...)`'s fixup function:
        was_updated = was_updated or PostHocPipelineFixup.FINAL_UPDATE_ALL(curr_active_pipeline, force_recompute=False, is_dry_run=False)


        ## Add `curr_active_pipeline.global_computation_results.computation_config` as needed:
        if curr_active_pipeline.global_computation_results.computation_config is None:
            print('global_computation_results.computation_config is None! Making new one!')
            curr_active_pipeline.global_computation_results.computation_config = ComputationKWargParameters.init_from_pipeline(curr_active_pipeline=curr_active_pipeline)
            print(f'\tdone. Pipeline needs resave!')
            was_updated = was_updated | True

        return was_updated

    
    # Plotting/Figures Helpers ___________________________________________________________________________________________ #
    def try_complete_figure_generation_to_file(self, curr_active_pipeline, enable_default_neptune_plots=False):
        try:
            ## To file only:
            with matplotlib_file_only():
                # Perform non-interactive Matplotlib operations with 'AGG' backend
                # neptuner = batch_perform_all_plots(curr_active_pipeline, enable_neptune=True, neptuner=None)
                main_complete_figure_generations(curr_active_pipeline, enable_default_neptune_plots=enable_default_neptune_plots, save_figures_only=True, save_figure=True, )

            # IF thst's done, clear all the plots:
            # from matplotlib import pyplot as plt
            # plt.close('all') # this takes care of the matplotlib-backed figures.
            curr_active_pipeline.clear_display_outputs()
            curr_active_pipeline.clear_registered_output_files()
            return True # completed successfully (without raising an error at least).

        except Exception as e:
            exception_info = sys.exc_info()
            e = CapturedException(e, exception_info)
            print(f'main_complete_figure_generations failed with exception: {e}')
            if self.fail_on_exception:
                raise e.exc

            return False


    def try_output_neruon_identity_table_to_File(self, file_path, curr_active_pipeline):
        try:
            session_context = curr_active_pipeline.get_session_context()
            session_group_key: str = "/" + session_context.get_description(separator="/", include_property_names=False) # 'kdiba/gor01/one/2006-6-08_14-26-15'
            session_uid: str = session_context.get_description(separator="|", include_property_names=False)

            AcrossSessionsResults.build_neuron_identity_table_to_hdf(file_path, key=session_group_key, spikes_df=curr_active_pipeline.sess.spikes_df, session_uid=session_uid)
            return True # completed successfully

        except Exception as e:
            exception_info = sys.exc_info()
            e = CapturedException(e, exception_info)
            print(f'try_output_neruon_identity_table_to_File failed with exception: {e}')
            # raise e
            return False


    def _try_save_global_computations_if_needed(self, curr_active_pipeline, curr_session_context, newly_computed_values):
        if (len(newly_computed_values) > 0):
            print(f'newly_computed_values: {newly_computed_values}. Saving global results...')
            if (self.saving_mode.value == 'skip_saving'):
                print(f'WARNING: supposed to skip_saving because of self.saving_mode: {self.saving_mode} but supposedly has new global results! Figure out if these are actually new.')
            if self.global_computations_options.should_save != SavingOptions.NEVER:
                try:
                    # curr_active_pipeline.global_computation_results.persist_time = datetime.now()
                    # Try to write out the global computation function results:
                    # curr_active_pipeline.save_global_computation_results()
                    an_override_save_path = (self.global_computations_options.override_output_file or self.global_computations_options.override_file)
                    if an_override_save_path is not None:
                        curr_active_pipeline.save_global_computation_results(override_global_pickle_path=an_override_save_path)
                    else:
                        curr_active_pipeline.save_global_computation_results()
                except Exception as e:
                    print(f'\n\n!!WARNING!!: saving the global results threw the exception: {e}')
                    print(f'\tthe global results are currently unsaved! proceed with caution and save as soon as you can!\n\n\n')
                    if self.fail_on_exception:
                        raise e.exc
            else:
                print(f'\n\n!!WARNING!!: self.global_computations_options.should_save == SavingOptions.NEVER, so the global results are unsaved!')
        else:
            print(f'no changes in global results.')
            if self.global_computations_options.should_save == SavingOptions.ALWAYS:
                print(f'Saving mode == ALWAYS so trying to save despite no changes.')
                try:
                    # curr_active_pipeline.global_computation_results.persist_time = datetime.now()
                    # Try to write out the global computation function results:
                    an_override_save_path = (self.global_computations_options.override_output_file or self.global_computations_options.override_file)
                    if an_override_save_path is not None:
                        curr_active_pipeline.save_global_computation_results(override_global_pickle_path=an_override_save_path)
                    else:
                        curr_active_pipeline.save_global_computation_results()
                except Exception as e:
                    print(f'\n\n!!WARNING!!: saving the global results threw the exception: {e}')
                    if self.fail_on_exception:
                        raise e.exc


    def try_compute_global_computations_if_needed(self, curr_active_pipeline, curr_session_context):
        """ tries to load/compute the global computations if needed depending on the self.global_computations_options specifications.

        Updates the passed `curr_active_pipeline` pipeline object.

        If computations are loaded, they are loaded via `curr_active_pipeline.load_pickled_global_computation_results(...)`
        If computations are needed, they are performed with the `batch_extended_computations(...)` function.


        If `.global_computations_options.should_compute` then computations will be tried and saved out as needed. If an error occurs, those will not be saved.

        """
        from pyphoplacecellanalysis.General.Model.SpecificComputationParameterTypes import ComputationKWargParameters
        
        # self.session_computations_options.override_output_file
        # self.global_computations_options.override_file

        newly_computed_values = []
        if self.global_computations_options.should_load:
            if not self.force_global_recompute: # not just force_reload, needs to recompute whenever the computation fails.
                try:
                    curr_active_pipeline.load_pickled_global_computation_results(override_global_computation_results_pickle_path=self.global_computations_options.override_file)
                except Exception as e:
                    exception_info = sys.exc_info()
                    e = CapturedException(e, exception_info)
                    print(f'cannot load global results: {e}')
                    if self.fail_on_exception:
                        raise e.exc

        # 2023-10-03 - Temporarily override the existing
        ## Add 2024-10-07 - `curr_active_pipeline.global_computation_results.computation_config` as needed:
        needs_build_global_computation_config: bool = True

        if curr_active_pipeline.global_computation_results.computation_config is None:
            # Create a DynamicContainer-backed computation_config
            # print(f'_perform_long_short_instantaneous_spike_rate_groups_analysis is lacking a required computation config parameter! creating a new curr_active_pipeline.global_computation_results.computation_config')
            # curr_active_pipeline.global_computation_results.computation_config = DynamicContainer(instantaneous_time_bin_size_seconds=0.01)
            needs_build_global_computation_config = True
        else:
            print(f'have an existing `global_computation_results.computation_config`: {curr_active_pipeline.global_computation_results.computation_config}')
            if isinstance(curr_active_pipeline.global_computation_results.computation_config, ComputationKWargParameters):
               needs_build_global_computation_config = False ## it is okay
            else:
                 needs_build_global_computation_config = True

            # if curr_active_pipeline.global_computation_results.computation_config.instantaneous_time_bin_size_seconds is None:
            #     print(f'\t curr_active_pipeline.global_computation_results.computation_config.instantaneous_time_bin_size_seconds is None, overriding with 0.01')
            #     curr_active_pipeline.global_computation_results.computation_config.instantaneous_time_bin_size_seconds = 0.01

        if needs_build_global_computation_config:
            print('global_computation_results.computation_config is None! Making new one!')
            curr_active_pipeline.global_computation_results.computation_config = ComputationKWargParameters.init_from_pipeline(curr_active_pipeline=curr_active_pipeline)
            print(f'\tdone. Pipeline needs resave!')
        
        if self.global_computations_options.should_save == SavingOptions.ALWAYS:
            assert self.global_computations_options.should_compute, f"currently  SavingOptions.ALWAYS requires that self.global_computations_options.should_compute == True also but this is not the case!"


        # Computation ________________________________________________________________________________________________________ #
        if self.global_computations_options.should_compute:
            # build computation functions to compute list:
            active_extended_computations_include_includelist = deepcopy(self.extended_computations_include_includelist)
            force_recompute_override_computations_includelist = self.force_recompute_override_computations_includelist or []
            force_recompute_override_computation_kwargs_dict = self.force_recompute_override_computation_kwargs_dict or {} # #TODO 2024-10-30 08:35: - [ ] is `force_recompute_override_computation_kwargs_dict` actually only used when forcing a recompute, or does passing it when it's the same as the already computed values force it to recompute? It seems to force it to recompute
            # ## #TODO 2024-11-06 14:21: - [ ] I think we should use `batch_evaluate_required_computations` instead of `batch_extended_computations` to avoid forcing recomputations.
            # needs_computation_output_dict, valid_computed_results_output_list, remaining_include_function_names = batch_evaluate_required_computations(curr_active_pipeline, include_includelist=active_extended_computations_include_includelist, include_global_functions=True, fail_on_exception=False, progress_print=True,
            #                                         force_recompute=self.force_global_recompute, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, debug_print=False)
            
            try:
                # # 2023-01-* - Call extended computations to build `_display_short_long_firing_rate_index_comparison` figures:
                with ExceptionPrintingContext(suppress=(not self.fail_on_exception)):
                    curr_active_pipeline.reload_default_computation_functions()
                    #TODO 2024-11-06 13:44: - [ ] `force_recompute_override_computations_includelist` is actually comming in with the specified override (when I was just trying to override the parameters)`
                    newly_computed_values += batch_extended_computations(curr_active_pipeline, include_includelist=active_extended_computations_include_includelist, include_global_functions=True, fail_on_exception=True, progress_print=True, # #TODO 2024-11-01 19:33: - [ ] self.force_recompute is True for some reason!?!
                                                                        force_recompute=self.force_global_recompute, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist,
                                                                        computation_kwargs_dict=force_recompute_override_computation_kwargs_dict, debug_print=False)
                    #TODO 2023-07-11 19:20: - [ ] We want to save the global results if they are computed, but we don't want them to be needlessly written to disk even when they aren't changed.
                    return newly_computed_values # return the list of newly computed values

            except Exception as e:
                ## TODO: catch/log saving error and indicate that it isn't saved.
                exception_info = sys.exc_info()
                e = CapturedException(e, exception_info)
                print(f'ERROR perform `batch_extended_computations` or saving GLOBAL COMPUTATION RESULTS for pipeline of curr_session_context: "{curr_session_context}". error: {e}')
                if self.fail_on_exception:
                    raise e.exc

        return newly_computed_values # return


    def try_export_pipeline_hdf5_if_needed(self, curr_active_pipeline, curr_session_context) -> Tuple[Optional[Path], Optional[CapturedException]]:
        """ Export the pipeline's HDF5 as 'pipeline_results.h5' """
        hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('pipeline_results.h5').resolve()
        print(f'pipeline hdf5_output_path: {hdf5_output_path}')
        e = None
        # Only get files newer than date
        skip_overwriting_files_newer_than_specified:bool = False

        newest_file_to_overwrite_date = datetime.now() - timedelta(days=1) # don't overwrite any files more recent than 1 day ago
        can_skip_if_allowed: bool = (hdf5_output_path.exists() and (FilesystemMetadata.get_last_modified_time(hdf5_output_path)<=newest_file_to_overwrite_date))
        if (not skip_overwriting_files_newer_than_specified) or (not can_skip_if_allowed):
            # if skipping is disabled OR skipping is enabled but it's not valid to skip, overwrite.
            # file is folder than the date to overwrite, so overwrite it
            print(f'OVERWRITING (or writing) the file {hdf5_output_path}!')
            try:
                curr_active_pipeline.export_pipeline_to_h5()
                return (hdf5_output_path, None)
            except Exception as e:
                exception_info = sys.exc_info()
                e = CapturedException(e, exception_info)
                print(f"ERROR: encountered exception {e} while trying to build the session HDF output for {curr_session_context}")
                if self.fail_on_exception:
                    raise e.exc
                hdf5_output_path = None # set to None because it failed.
                return (hdf5_output_path, e)

        else:
            print(f'WARNING: file {hdf5_output_path} is newer than the allowed overwrite date, so it will be skipped.')
            print(f'\t\tnewest_file_to_overwrite_date: {newest_file_to_overwrite_date}\t can_skip_if_allowed: {can_skip_if_allowed}\n')
            return (hdf5_output_path, None)


    def try_require_pipeline_has_refined_pfs(self, curr_active_pipeline):
        """ Refine the LxC/SxC designators using the firing rate index metric:
        """
        try:
            jonathan_firing_rate_analysis_result: JonathanFiringRateAnalysisResult = curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis
            ## Get global `long_short_fr_indicies_analysis`:
            long_short_fr_indicies_analysis_results = curr_active_pipeline.global_computation_results.computed_data['long_short_fr_indicies_analysis']
            long_short_fr_indicies_df = long_short_fr_indicies_analysis_results['long_short_fr_indicies_df']
            did_compute = jonathan_firing_rate_analysis_result.refine_exclusivity_by_inst_frs_index(long_short_fr_indicies_df, frs_index_inclusion_magnitude=self.frs_index_inclusion_magnitude, override_existing_frs_index_values=self.override_existing_frs_index_values)
            if did_compute:
                return ['jonathan_firing_rate_analysis']
            else:
                return [] # no computations needed

        except BaseException as e:
            exception_info = sys.exc_info()
            e = CapturedException(e, exception_info)
            print(f"ERROR: encountered exception {e} while trying run `_require_pipeline_has_refined_pfs(...)")
            if self.fail_on_exception:
                raise e.exc
            return []


    # def completion_decorator(self, func):
    #     """ NOT USED. Don't think it works yet. 
    #     func (self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline) to be called """
    #     self.completion_functions.append(func)
        
    #     def wrapper(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict):
    #         print("Something is happening before the function is called.")
    #         across_session_results_extended_dict = func(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict)
    #         print("Something is happening after the function is called.")
    #         return across_session_results_extended_dict
        
    #     return wrapper


    ## Main function that's called with the complete pipeline:
    @function_attributes(short_name=None, tags=['MAIN', 'IMPORTANT', 'callback', 'replay'], input_requires=['filtered_sessions[*].replay'], output_provides=[], uses=['.completion_functions', '.post_compute_validate', '.try_compute_global_computations_if_needed', '.try_complete_figure_generation_to_file', '.try_export_pipeline_hdf5_if_needed'], used_by=['run_specific_batch'], creation_date='2024-07-02 11:44', related_items=[])  
    def on_complete_success_execution_session(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline) -> PipelineCompletionResult:
        """ called when the execute_session completes like:
            `post_run_callback_fn_output = post_run_callback_fn(curr_session_context, curr_session_basedir, curr_active_pipeline)`

            Meant to be assigned like:
            , post_run_callback_fn=_on_complete_success_execution_session

            Captures nothing.

            from Spike3D.scripts.run_BatchAnalysis import _on_complete_success_execution_session


            LOGIC: really we want to recompute global whenever local is recomputed.


        """
        print(f'on_complete_success_execution_session(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
        # print(f'curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}')
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        # Get existing laps from session:
        long_laps, short_laps, global_laps = [curr_active_pipeline.filtered_sessions[an_epoch_name].laps.as_epoch_obj() for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        long_replays, short_replays, global_replays = [Epoch(curr_active_pipeline.filtered_sessions[an_epoch_name].replay.epochs.get_valid_df()) for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        # short_laps.n_epochs: 40, n_long_laps.n_epochs: 40
        # short_replays.n_epochs: 6, long_replays.n_epochs: 8
        if self.debug_print:
            print(f'short_laps.n_epochs: {short_laps.n_epochs}, n_long_laps.n_epochs: {long_laps.n_epochs}')
            print(f'short_replays.n_epochs: {short_replays.n_epochs}, long_replays.n_epochs: {long_replays.n_epochs}')


        was_updated = False

        # ## Post Compute Validate 2023-05-16:
        try:
            was_updated = was_updated | self.post_compute_validate(curr_active_pipeline)
        except Exception as e:
            exception_info = sys.exc_info()
            an_err = CapturedException(e, exception_info)
            print(f'self.post_compute_validate(...) failed with exception: {an_err}')
            raise

        ## Save the pipeline since that's disabled by default now:
        if was_updated and (self.session_computations_options.should_save != SavingOptions.NEVER):
            # override the saving mode:
            print(f'WARNING: basic pipleine was updated by post_compute_validate and needs to be saved to be correct.Overriding self.save_mode to ensure pipeline is saved!')
            self.saving_mode = PipelineSavingScheme.TEMP_THEN_OVERWRITE

        try:
            # self.session_computations_options.override_file
            curr_active_pipeline.save_pipeline(saving_mode=self.saving_mode, active_pickle_filename=self.session_computations_options.override_output_file) # AttributeError: 'PfND_TimeDependent' object has no attribute '_included_thresh_neurons_indx'
        except Exception as e:
            ## TODO: catch/log saving error and indicate that it isn't saved.
            exception_info = sys.exc_info()
            e = CapturedException(e, exception_info)
            print(f'ERROR SAVING PIPELINE for curr_session_context: {curr_session_context}. error: {e}')
            if self.fail_on_exception:
                raise e.exc

        ## GLOBAL FUNCTION:
        if self.force_reload_all and (not self.force_global_recompute):
            print(f'WARNING: self.force_global_recompute was False but self.force_reload_all was true. The global properties must be recomputed when the local functions change, so self.force_global_recompute will be set to True and computation will continue.')
            self.force_global_recompute = True

        if was_updated and (not self.force_global_recompute):
            print(f'WARNING: self.force_global_recompute was False but pipeline was_updated. The global properties must be recomputed when the local functions change, so self.force_global_recompute will be set to True and computation will continue.')
            self.force_global_recompute = True

        newly_computed_values = self.try_compute_global_computations_if_needed(curr_active_pipeline, curr_session_context=curr_session_context)
        ## Try to ensure the 2023-09-29 LxC and SxCs are "refined" by the rate remapping firing rate:
        # newly_computed_values = newly_computed_values + self.try_require_pipeline_has_refined_pfs(curr_active_pipeline)
        self._try_save_global_computations_if_needed(curr_active_pipeline, curr_session_context, newly_computed_values) # Save if needed


        # ### Programmatic Figure Outputs:
        if self.should_perform_figure_generation_to_file:
            self.try_complete_figure_generation_to_file(curr_active_pipeline, enable_default_neptune_plots=self.should_generate_all_plots)
        else:
            print(f'skipping figure generation because should_perform_figure_generation_to_file == False')


        ### Aggregate Outputs specific computations:

        ## Get some more interesting session properties:
        delta_since_last_compute: timedelta = curr_active_pipeline.get_time_since_last_computation()
        print(f'\t time since last computation: {delta_since_last_compute}')

        # Export the pipeline's HDF5:
        if self.enable_hdf5_output:
            hdf5_output_path, hdf5_output_err = self.try_export_pipeline_hdf5_if_needed(curr_active_pipeline=curr_active_pipeline, curr_session_context=curr_session_context)
        else:
            hdf5_output_path, hdf5_output_err = None, None


        # print(f'\t doing specific instantaneous firing rate computation for context: {curr_session_context}...')
        # ## Specify the output file:
        # common_file_path = Path('output/active_across_session_scatter_plot_results.h5')
        # print(f'common_file_path: {common_file_path}')
        # InstantaneousFiringRatesDataframeAccessor.add_results_to_inst_fr_results_table(curr_active_pipeline, common_file_path, file_mode='a')

        try:
            print(f'\t doing specific instantaneous firing rate computation for context: {curr_session_context}...')
            _out_recomputed_inst_fr_comps = InstantaneousSpikeRateGroupsComputation(instantaneous_time_bin_size_seconds=0.003) # 3ms, 10ms
            _out_recomputed_inst_fr_comps.compute(curr_active_pipeline=curr_active_pipeline, active_context=curr_active_pipeline.sess.get_context())
            _out_inst_fr_comps = curr_active_pipeline.global_computation_results.computed_data['long_short_inst_spike_rate_groups']

            if not self.use_multiprocessing:
                # Only modify self in non-multiprocessing mode (only shows 1 always).
                self.across_sessions_instantaneous_fr_dict[curr_session_context] = _out_inst_fr_comps # instantaneous firing rates for this session, doesn't work in multiprocessing mode.
                print(f'\t\t Now have {len(self.across_sessions_instantaneous_fr_dict)} entries in self.across_sessions_instantaneous_fr_dict!')

            # LxC_ReplayDeltaMinus, LxC_ReplayDeltaPlus, SxC_ReplayDeltaMinus, SxC_ReplayDeltaPlus = _out_inst_fr_comps.LxC_ReplayDeltaMinus, _out_inst_fr_comps.LxC_ReplayDeltaPlus, _out_inst_fr_comps.SxC_ReplayDeltaMinus, _out_inst_fr_comps.SxC_ReplayDeltaPlus
            # LxC_ThetaDeltaMinus, LxC_ThetaDeltaPlus, SxC_ThetaDeltaMinus, SxC_ThetaDeltaPlus = _out_inst_fr_comps.LxC_ThetaDeltaMinus, _out_inst_fr_comps.LxC_ThetaDeltaPlus, _out_inst_fr_comps.SxC_ThetaDeltaMinus, _out_inst_fr_comps.SxC_ThetaDeltaPlus
            print(f'\t\t done (success).')

        except Exception as e:
            exception_info = sys.exc_info()
            e = CapturedException(e, exception_info)
            print(f"WARN: on_complete_success_execution_session: encountered exception {e} while trying to compute the instantaneous firing rates and set self.across_sessions_instantaneous_fr_dict[{curr_session_context}]")
            # if self.fail_on_exception:
            #     raise e.exc
            _out_inst_fr_comps = None
            _out_recomputed_inst_fr_comps = None
            pass

        
            
        # On large ram systems, we can return the whole pipeline? No, because the whole pipeline can't be pickled.
        across_session_results_extended_dict = {}

        ## get override kwargs
        override_user_completion_function_kwargs_dict = deepcopy(self.override_user_completion_function_kwargs_dict) ## previously used a blank override config, making it useless. {}
        
        ## run external completion functions:
        for a_fn in self.completion_functions:
            print(f'\t>> calling external computation function: {a_fn.__name__}')
            with ExceptionPrintingContext():
                a_found_override_kwargs = {} ## start empty
                if a_fn.__name__ in override_user_completion_function_kwargs_dict:
                    ## found kwargs
                    a_found_override_kwargs = override_user_completion_function_kwargs_dict.pop(a_fn.__name__, {})
                elif a_fn in override_user_completion_function_kwargs_dict:
                    a_found_override_kwargs = override_user_completion_function_kwargs_dict.pop(a_fn, {})
                else:
                    a_found_override_kwargs = {} # no override

                across_session_results_extended_dict = a_fn(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict, **a_found_override_kwargs)
            

        return PipelineCompletionResult(long_epoch_name=long_epoch_name, long_laps=long_laps, long_replays=long_replays,
                                           short_epoch_name=short_epoch_name, short_laps=short_laps, short_replays=short_replays,
                                           delta_since_last_compute=delta_since_last_compute,
                                           outputs_local={'pkl': curr_active_pipeline.pickle_path},
                                            outputs_global={'pkl': curr_active_pipeline.global_computation_results_pickle_path, 'hdf5': hdf5_output_path},
                                            across_session_results={'inst_fr_comps': _out_inst_fr_comps, 'recomputed_inst_fr_comps': _out_recomputed_inst_fr_comps, **across_session_results_extended_dict})






