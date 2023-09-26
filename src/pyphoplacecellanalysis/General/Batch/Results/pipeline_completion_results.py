import sys
import os
from enum import Enum, unique, auto # SessionBatchProgress
from attrs import define, field, Factory
from datetime import datetime, timedelta
import pathlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd

import tables as tb # for `PipelineCompletionResultTable`

from neuropy.core.epoch import Epoch

from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin, HDF_Converter

## Extended imports here are mostly for `BatchSessionCompletionHandler`
from neuropy.utils.matplotlib_helpers import matplotlib_file_only
from neuropy.utils.result_context import IdentifyingContext
from neuropy.core.session.Formats.BaseDataSessionFormats import find_local_session_paths
from neuropy.utils.dynamic_container import DynamicContainer, override_dict, overriding_dict_with, get_dict_subset


from pyphocorehelpers.print_helpers import CapturedException
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import LongShortPipelineTests
# from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import set_environment_variables, neptune_output_figures
from pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper import main_complete_figure_generations  # for `BatchSessionCompletionHandler`
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import SingleBarResult, InstantaneousSpikeRateGroupsComputation
from neuropy.core.user_annotations import UserAnnotationsManager
from pyphoplacecellanalysis.General.Batch.AcrossSessionResults import AcrossSessionsResults, AcrossSessionsVisualizations, InstantaneousFiringRatesDataframeAccessor

from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_load_session, batch_extended_computations, batch_programmatic_figures, batch_extended_programmatic_figures
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData, loadData



# ==================================================================================================================== #
# Begin Definitions                                                                                                    #
# ==================================================================================================================== #


@unique
class SavingOptions(Enum):
    NEVER = "NEVER"
    IF_CHANGED = "IF_CHANGED"
    ALWAYS = "ALWAYS"


@custom_define(slots=False)
class BatchComputationProcessOptions(HDF_SerializationMixin):
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



@custom_define(slots=False)
class PipelineCompletionResult(HDF_SerializationMixin, AttrsBasedClassHelperMixin):
    """ Class representing the specific results extratracted from the loaded pipeline and returned as return values from the post-execution callback function. """
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



# PyTables Definitions for Output Tables: ____________________________________________________________________________ #
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



# ==================================================================================================================== #
# BatchSessionCompletionHandlers                                                                                       #
# ==================================================================================================================== #

@define(slots=False, repr=False)
class BatchSessionCompletionHandler:
    """ handles completion of a single session's batch processing. 

    Allows accumulating results across sessions and runs.

    
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
    enable_full_pipeline_in_ram: bool = field(default=False)
    
    override_session_computation_results_pickle_filename: Optional[str] = field(default=None) # 'output/loadedSessPickle.pkl'

    session_computations_options: BatchComputationProcessOptions = field(default=BatchComputationProcessOptions(should_load=True, should_compute=True, should_save=SavingOptions.IF_CHANGED))

    global_computations_options: BatchComputationProcessOptions = field(default=BatchComputationProcessOptions(should_load=True, should_compute=True, should_save=SavingOptions.IF_CHANGED))
    extended_computations_include_includelist: list = field(default=['pf_computation', 'pfdt_computation', 'firing_rate_trends', 'pf_dt_sequential_surprise', 'extended_stats',
                                        'long_short_decoding_analyses', 'jonathan_firing_rate_analysis', 'long_short_fr_indicies_analyses', 'short_long_pf_overlap_analyses', 'long_short_post_decoding', 'long_short_rate_remapping',
                                        #  'long_short_inst_spike_rate_groups',
                                        'long_short_endcap_analysis']) # do only specified
    
    force_global_recompute: bool = field(default=False)
    override_global_computation_results_pickle_path: Optional[Path] = field(default=None)

    # Figures:
    should_perform_figure_generation_to_file: bool = field(default=True) # controls whether figures are generated to file
    should_generate_all_plots: bool = field(default=False) # controls whether all plots are generated (when True) or if only non-Neptune paper figure specific plots are generated. Has no effect if self.should_perform_figure_generation_to_file is False.
    
    
    # Cross-session Results:
    across_sessions_instantaneous_fr_dict: dict = Factory(dict) # Dict[IdentifyingContext] = InstantaneousSpikeRateGroupsComputation


    @classmethod
    def post_compute_validate(cls, curr_active_pipeline) -> bool:
        """ 2023-05-16 - Ensures that the laps are used for the placefield computation epochs, the number of bins are the same between the long and short tracks. """
        def _subfn_update_pipeline_missing_preprocessing_parameters(curr_active_pipeline, debug_print=False):
            """ 2023-05-24 - Adds the previously missing `sess.config.preprocessing_parameters` to each session (filtered and base) in the pipeline.

            Usage:
                from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import _update_pipeline_missing_preprocessing_parameters
                was_updated = _update_pipeline_missing_preprocessing_parameters(curr_active_pipeline)
                was_updated
            """
            def _subfn_update_session_missing_preprocessing_parameters(sess):
                """ 2023-05-24 - Adds the previously missing `sess.config.preprocessing_parameters` to a single session. Called only by `_update_pipeline_missing_preprocessing_parameters` """
                preprocessing_parameters = getattr(sess.config, 'preprocessing_parameters', None)
                if preprocessing_parameters is None:
                    print(f'No existing preprocessing parameters! Assigning them!')
                    default_lap_estimation_parameters = dict(N=20, should_backup_extant_laps_obj=True) # Passed as arguments to `sess.replace_session_laps_with_estimates(...)`
                    default_PBE_estimation_parameters = DynamicContainer(sigma=0.030, thresh=(0, 1.5), min_dur=0.030, merge_dur=0.100, max_dur=0.300) # NewPaper's Parameters
                    default_replay_estimation_parameters = DynamicContainer(require_intersecting_epoch=None, min_epoch_included_duration=0.06, max_epoch_included_duration=None, maximum_speed_thresh=None, min_inclusion_fr_active_thresh=0.01, min_num_unique_aclu_inclusions=3)

                    sess.config.preprocessing_parameters = DynamicContainer(epoch_estimation_parameters=DynamicContainer.init_from_dict({
                            'laps': default_lap_estimation_parameters,
                            'PBEs': default_PBE_estimation_parameters,
                            'replays': default_replay_estimation_parameters
                        }))
                    return True
                else:
                    if debug_print:
                        print(f'preprocessing parameters exist.')
                    return False

            # BEGIN MAIN FUNCTION BODY
            was_updated = False
            was_updated = was_updated | _subfn_update_session_missing_preprocessing_parameters(curr_active_pipeline.sess)

            long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
            for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]:
                was_updated = was_updated | _subfn_update_session_missing_preprocessing_parameters(curr_active_pipeline.filtered_sessions[an_epoch_name])

            if was_updated:
                print(f'config was updated. Saving pipeline.')
                curr_active_pipeline.save_pipeline(saving_mode=PipelineSavingScheme.OVERWRITE_IN_PLACE)
            return was_updated




        LongShortPipelineTests(curr_active_pipeline=curr_active_pipeline).validate()
        # 2023-05-24 - Adds the previously missing `sess.config.preprocessing_parameters` to each session (filtered and base) in the pipeline.
        was_updated = _subfn_update_pipeline_missing_preprocessing_parameters(curr_active_pipeline)
        print(f'were pipeline preprocessing parameters missing and updated?: {was_updated}')

        ## BUG 2023-05-25 - Found ERROR for a loaded pipeline where for some reason the filtered_contexts[long_epoch_name]'s actual context was the same as the short maze ('...maze2'). Unsure how this happened.
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
        # assert long_epoch_context.filter_name == long_epoch_name, f"long_epoch_context.filter_name: {long_epoch_context.filter_name} != long_epoch_name: {long_epoch_name}"
        if long_epoch_context.filter_name != long_epoch_name:
            print(f"WARNING: filtered_contexts[long_epoch_name]'s actual context name is incorrect. \n\tlong_epoch_context.filter_name: {long_epoch_context.filter_name} != long_epoch_name: {long_epoch_name}\n\tUpdating it. (THIS IS A HACK)")
            # fix it if broken
            long_epoch_context.filter_name = long_epoch_name
            was_updated = True

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


    def try_compute_global_computations_if_needed(self, curr_active_pipeline, curr_session_context):
        """ tries to load/compute the global computations if needed depending on the self.global_computations_options specifications.
        
        Updates the passed `curr_active_pipeline` pipeline object.

        If computations are loaded, they are loaded via `curr_active_pipeline.load_pickled_global_computation_results(...)`
        If computations are needed, they are performed with the `batch_extended_computations(...)` function.

        
        """
        if self.global_computations_options.should_load:
            if not self.force_global_recompute: # not just force_reload, needs to recompute whenever the computation fails.
                try:
                    curr_active_pipeline.load_pickled_global_computation_results(override_global_computation_results_pickle_path=self.override_global_computation_results_pickle_path)
                except Exception as e:
                    exception_info = sys.exc_info()
                    e = CapturedException(e, exception_info)
                    print(f'cannot load global results: {e}')
                    if self.fail_on_exception:
                        raise e.exc

        if self.global_computations_options.should_save == SavingOptions.ALWAYS:
            assert self.global_computations_options.should_compute, f"currently  SavingOptions.ALWAYS requires that self.global_computations_options.should_compute == True also but this is not the case!"

        if self.global_computations_options.should_compute:
            try:
                # # 2023-01-* - Call extended computations to build `_display_short_long_firing_rate_index_comparison` figures:
                curr_active_pipeline.reload_default_computation_functions()
                newly_computed_values = batch_extended_computations(curr_active_pipeline, include_includelist=self.extended_computations_include_includelist, include_global_functions=True, fail_on_exception=True, progress_print=True, force_recompute=self.force_global_recompute, debug_print=False)
                #TODO 2023-07-11 19:20: - [ ] We want to save the global results if they are computed, but we don't want them to be needlessly written to disk even when they aren't changed.

                if (len(newly_computed_values) > 0):
                    print(f'newly_computed_values: {newly_computed_values}. Saving global results...')
                    if (self.saving_mode.value == 'skip_saving'):
                        print(f'WARNING: supposed to skip_saving because of self.saving_mode: {self.saving_mode} but supposedly has new global results! Figure out if these are actually new.')
                    if self.global_computations_options.should_save != SavingOptions.NEVER:
                        try:
                            # curr_active_pipeline.global_computation_results.persist_time = datetime.now()
                            # Try to write out the global computation function results:
                            curr_active_pipeline.save_global_computation_results()
                        except Exception as e:
                            print(f'\n\n!!WARNING!!: saving the global results threw the exception: {e}')
                            print(f'\tthe global results are currently unsaved! proceed with caution and save as soon as you can!\n\n\n')
                            if self.fail_on_exception:
                                raise e.exc
                    else:
                        print(f'\n\n!!WARNING!!: self.global_computations_options.should_save == False, so the global results are unsaved!')
                else:
                    print(f'no changes in global results.')
                    if self.global_computations_options.should_save == SavingOptions.ALWAYS:
                        print(f'Saving mode == ALWAYS so trying to save despite no changes.')
                        try:
                            # curr_active_pipeline.global_computation_results.persist_time = datetime.now()
                            # Try to write out the global computation function results:
                            curr_active_pipeline.save_global_computation_results()
                        except Exception as e:
                            print(f'\n\n!!WARNING!!: saving the global results threw the exception: {e}')
                            if self.fail_on_exception:
                                raise e.exc
                            
            except Exception as e:
                ## TODO: catch/log saving error and indicate that it isn't saved.
                exception_info = sys.exc_info()
                e = CapturedException(e, exception_info)
                print(f'ERROR SAVING GLOBAL COMPUTATION RESULTS for pipeline of curr_session_context: {curr_session_context}. error: {e}')
                if self.fail_on_exception:
                    raise e.exc


    def try_export_pipeline_hdf5_if_needed(self, curr_active_pipeline, curr_session_context) -> Tuple[Optional[Path], Optional[CapturedException]]:
        """ Export the pipeline's HDF5 as 'pipeline_results.h5' """
        hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('pipeline_results.h5').resolve()
        print(f'pipeline hdf5_output_path: {hdf5_output_path}')
        e = None
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
    

    ## Main function that's called with the complete pipeline:
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
        # long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        # long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]

        # Get existing laps from session:
        long_laps, short_laps, global_laps = [curr_active_pipeline.filtered_sessions[an_epoch_name].laps.as_epoch_obj() for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        long_replays, short_replays, global_replays = [Epoch(curr_active_pipeline.filtered_sessions[an_epoch_name].replay.epochs.get_valid_df()) for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        # short_laps.n_epochs: 40, n_long_laps.n_epochs: 40
        # short_replays.n_epochs: 6, long_replays.n_epochs: 8
        if self.debug_print:
            print(f'short_laps.n_epochs: {short_laps.n_epochs}, n_long_laps.n_epochs: {long_laps.n_epochs}')
            print(f'short_replays.n_epochs: {short_replays.n_epochs}, long_replays.n_epochs: {long_replays.n_epochs}')

        # ## Post Compute Validate 2023-05-16:
        try:
            was_updated = self.post_compute_validate(curr_active_pipeline)
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
            curr_active_pipeline.save_pipeline(saving_mode=self.saving_mode, active_pickle_filename=self.override_session_computation_results_pickle_filename) # AttributeError: 'PfND_TimeDependent' object has no attribute '_included_thresh_neurons_indx'
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

        self.try_compute_global_computations_if_needed(curr_active_pipeline, curr_session_context=curr_session_context)     

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
        hdf5_output_path, hdf5_output_err = self.try_export_pipeline_hdf5_if_needed(curr_active_pipeline=curr_active_pipeline, curr_session_context=curr_session_context)
        

        # print(f'\t doing specific instantaneous firing rate computation for context: {curr_session_context}...')
        # ## Specify the output file:
        # common_file_path = Path('output/active_across_session_scatter_plot_results.h5')
        # print(f'common_file_path: {common_file_path}')
        # InstantaneousFiringRatesDataframeAccessor.add_results_to_inst_fr_results_table(curr_active_pipeline, common_file_path, file_mode='a')

        try:
            print(f'\t doing specific instantaneous firing rate computation for context: {curr_session_context}...')
            _out_inst_fr_comps = InstantaneousSpikeRateGroupsComputation(instantaneous_time_bin_size_seconds=0.01) # 10ms
            _out_inst_fr_comps.compute(curr_active_pipeline=curr_active_pipeline, active_context=curr_active_pipeline.sess.get_context())
            
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
            print(f"ERROR: encountered exception {e} while trying to compute the instantaneous firing rates and set self.across_sessions_instantaneous_fr_dict[{curr_session_context}]")
            if self.fail_on_exception:
                raise e.exc
            _out_inst_fr_comps = None
            
        # On large ram systems, we can return the whole pipeline?
        
        if self.enable_full_pipeline_in_ram:
            across_session_results_extended_dict = {'curr_active_pipeline': curr_active_pipeline}
        else:
            across_session_results_extended_dict = {}
            
        return PipelineCompletionResult(long_epoch_name=long_epoch_name, long_laps=long_laps, long_replays=long_replays,
                                           short_epoch_name=short_epoch_name, short_laps=short_laps, short_replays=short_replays,
                                           delta_since_last_compute=delta_since_last_compute,
                                           outputs_local={'pkl': curr_active_pipeline.pickle_path},
                                            outputs_global={'pkl': curr_active_pipeline.global_computation_results_pickle_path, 'hdf5': hdf5_output_path},
                                            across_session_results={'inst_fr_comps': _out_inst_fr_comps, **across_session_results_extended_dict})
                                          




@define(slots=False, repr=False)
class HDFSpecificBatchSessionCompletionHandler(BatchSessionCompletionHandler):
    """ 2023-08-25 - an alternative completion handler that just the .h5 stuff. 
    
    """
    
    # Cross-session Results:
    across_sessions_instantaneous_fr_dict: dict = Factory(dict) # Dict[IdentifyingContext] = InstantaneousSpikeRateGroupsComputation

    ## Main function that's called with the complete pipeline:
    def on_complete_success_execution_session(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline) -> PipelineCompletionResult:
        """ called when the execute_session completes like:
            `post_run_callback_fn_output = post_run_callback_fn(curr_session_context, curr_session_basedir, curr_active_pipeline)`
            
            Meant to be assigned like:
            , post_run_callback_fn=_on_complete_success_execution_session
            
            Captures nothing.
            
            from Spike3D.scripts.run_BatchAnalysis import _on_complete_success_execution_session
            
            
            LOGIC: really we want to recompute global whenever local is recomputed.
            
            
        """
        print(f'HDFProcessing.on_complete_success_execution_session(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
        # print(f'curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}')
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        # long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        # long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]

        # Get existing laps from session:
        long_laps, short_laps, global_laps = [curr_active_pipeline.filtered_sessions[an_epoch_name].laps.as_epoch_obj() for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        long_replays, short_replays, global_replays = [Epoch(curr_active_pipeline.filtered_sessions[an_epoch_name].replay.epochs.get_valid_df()) for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        # short_laps.n_epochs: 40, n_long_laps.n_epochs: 40
        # short_replays.n_epochs: 6, long_replays.n_epochs: 8
        if self.debug_print:
            print(f'short_laps.n_epochs: {short_laps.n_epochs}, n_long_laps.n_epochs: {long_laps.n_epochs}')
            print(f'short_replays.n_epochs: {short_replays.n_epochs}, long_replays.n_epochs: {long_replays.n_epochs}')

        # ## Post Compute Validate 2023-05-16:
        try:
            was_updated = self.post_compute_validate(curr_active_pipeline)
        except Exception as e:
            exception_info = sys.exc_info()
            an_err = CapturedException(e, exception_info)
            print(f'self.post_compute_validate(...) failed with exception: {an_err}')
            raise 

        delta_since_last_compute: timedelta = curr_active_pipeline.get_time_since_last_computation()
        print(f'\t time since last computation: {delta_since_last_compute}')
        
        ## Save the pipeline since that's disabled by default now:
        try:
            curr_active_pipeline.save_pipeline(saving_mode=self.saving_mode, active_pickle_filename=self.override_session_computation_results_pickle_filename) # AttributeError: 'PfND_TimeDependent' object has no attribute '_included_thresh_neurons_indx'
        except Exception as e:
            ## TODO: catch/log saving error and indicate that it isn't saved.
            exception_info = sys.exc_info()
            e = CapturedException(e, exception_info)
            print(f'ERROR SAVING PIPELINE for curr_session_context: {curr_session_context}. error: {e}')


        ## GLOBAL FUNCTION:
        if self.force_reload_all and (not self.force_global_recompute):
            print(f'WARNING: self.force_global_recompute was False but self.force_reload_all was true. The global properties must be recomputed when the local functions change, so self.force_global_recompute will be set to True and computation will continue.')
            self.force_global_recompute = True
            
        if was_updated and (not self.force_global_recompute):
            print(f'WARNING: self.force_global_recompute was False but pipeline was_updated. The global properties must be recomputed when the local functions change, so self.force_global_recompute will be set to True and computation will continue.')
            self.force_global_recompute = True


        ## GLOBAL FUNCTION:
        if self.force_reload_all and (not self.force_global_recompute):
            print(f'WARNING: self.force_global_recompute was False but self.force_reload_all was true. The global properties must be recomputed when the local functions change, so self.force_global_recompute will be set to True and computation will continue.')
            self.force_global_recompute = True
            
        if was_updated and (not self.force_global_recompute):
            print(f'WARNING: self.force_global_recompute was False but pipeline was_updated. The global properties must be recomputed when the local functions change, so self.force_global_recompute will be set to True and computation will continue.')
            self.force_global_recompute = True

        self.try_compute_global_computations_if_needed(curr_active_pipeline, curr_session_context=curr_session_context)

        ### Aggregate Outputs specific computations:

        ## Get some more interesting session properties:
        
        delta_since_last_compute: timedelta = curr_active_pipeline.get_time_since_last_computation()
        
        print(f'\t time since last computation: {delta_since_last_compute}')

        # Export the pipeline's HDF5:
        hdf5_output_path, hdf5_output_err = self.try_export_pipeline_hdf5_if_needed(curr_active_pipeline=curr_active_pipeline, curr_session_context=curr_session_context)
        
        # ## Specify the output file:
        # common_file_path = Path('output/active_across_session_scatter_plot_results.h5')
        # print(f'common_file_path: {common_file_path}')
        # InstantaneousFiringRatesDataframeAccessor.add_results_to_inst_fr_results_table(curr_active_pipeline, common_file_path, file_mode='a')

        across_session_results_extended_dict = {}
            
        return PipelineCompletionResult(long_epoch_name=long_epoch_name, long_laps=long_laps, long_replays=long_replays,
                                           short_epoch_name=short_epoch_name, short_laps=short_laps, short_replays=short_replays,
                                           delta_since_last_compute=delta_since_last_compute,
                                           outputs_local={'pkl': curr_active_pipeline.pickle_path},
                                            outputs_global={'pkl': curr_active_pipeline.global_computation_results_pickle_path, 'hdf5': hdf5_output_path},
                                            across_session_results=across_session_results_extended_dict)
                                          
