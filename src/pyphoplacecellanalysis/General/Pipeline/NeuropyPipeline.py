#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
NeuropyPipeline.py
"""
from copy import deepcopy
import importlib
import sys
from pathlib import Path
import shutil # for _backup_extant_file(...)

from typing import Callable, List, Optional, Dict
import inspect # used for filter_sessions(...)'s inspect.getsource to compare filters:

import numpy as np
import pandas as pd
import tables as tb
import h5py

from attrs import define, field, Factory


# NeuroPy (Diba Lab Python Repo) Loading
from neuropy import core
# importlib.reload(core)

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.hashing_helpers import get_hash_tuple, freeze
from pyphocorehelpers.mixins.diffable import DiffableObject
from pyphocorehelpers.print_helpers import print_filesystem_file_size, print_object_memory_usage
from pyphocorehelpers.Filesystem.path_helpers import build_unique_filename, backup_extant_file

from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder # hopefully this works without all the other imports
from neuropy.core.session.KnownDataSessionTypeProperties import KnownDataSessionTypeProperties
from neuropy.utils.result_context import IdentifyingContext

from pyphocorehelpers.print_helpers import CapturedException
from pyphoplacecellanalysis.General.Pipeline.Stages.Computation import PipelineWithComputedPipelineStageMixin, ComputedPipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Display import PipelineWithDisplayPipelineStageMixin, PipelineWithDisplaySavingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.Filtering import FilteredPipelineMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import PipelineWithInputStage, PipelineWithLoadableStage, loadData, saveData
from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import PipelineStage
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import OutputsSpecifier

## For serialization:
from neuropy.utils.mixins.AttrsClassHelpers import custom_define, AttrsBasedClassHelperMixin, serialized_field, serialized_attribute_field, non_serialized_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin



from qtpy import QtCore, QtWidgets, QtGui

# Pipeline Logging:
import logging
from pyphocorehelpers.print_helpers import build_module_logger
pipeline_module_logger = build_module_logger('Spike3D.pipeline')

from pyphocorehelpers.DataStructure.enum_helpers import ExtendedEnum # for PipelineSavingScheme

class PipelineSavingScheme(ExtendedEnum):
    """Describes how the pickled pipeline is saved and how it impacts existing files.
    Used by `save_pipeline(...)`

    from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme
    PipelineSavingScheme.SKIP_SAVING
    """
    SKIP_SAVING = "skip_saving"
    TEMP_THEN_OVERWRITE = "temp_then_overwrite" # saves to a temporary filename if extant exists, then compares and overwrites if needed. Prevents ruining the real pickle if pickling is interrupted/fails.
    OVERWRITE_IN_PLACE = "overwrite_in_place" 
    # SAVING = "Saving"
    # GENERIC = "Generic"

    @property
    def shouldSave(self):
        return PipelineSavingScheme.shouldSaveList()[self.value]

    # Static properties
    @classmethod
    def shouldSaveList(cls):
        values_list = [False, True, True]
        assert len(values_list) == len(cls.all_member_values()), f"values_list must have one value for each member of the enum, but got {len(values_list)} values for {len(cls.all_members())} members."
        return dict(zip(cls.all_member_values(), values_list)) # doing it this way (comparing the string value) requires self.value in the shouldSave property!
        # return cls.build_member_value_dict([False, True, True])





@define(slots=False)
class LoadedObjectPersistanceState:
    """Keeps track of the persistence state for an object that has been loaded from disk to keep track of how the object's state relates to the version on disk (the persisted version)"""
    file_path: Path = field(converter=Path)
    load_compare_state: Dict = field(alias='compare_state_on_load', converter=deepcopy)

    def needs_save(self, curr_object) -> bool:
        """Compares the curr_object's state to its state when loaded from disk to see if anything changed and it needs to be re-persisted (by saving)"""
        # lhs_compare_dict = NeuropyPipeline.build_pipeline_compare_dict(curr_object)
        curr_diff = DiffableObject.compute_diff(curr_object.pipeline_compare_dict, self.load_compare_state)
        return len(curr_diff) > 0


    

class NeuropyPipeline(PipelineWithInputStage, PipelineWithLoadableStage, FilteredPipelineMixin, PipelineWithComputedPipelineStageMixin, PipelineWithDisplayPipelineStageMixin, PipelineWithDisplaySavingMixin, HDF_SerializationMixin, QtCore.QObject):
    """ 
    
    Exposes the active sessions via its .sess member.
    
    Stages:
    1. Input/Loading
        .set_input(...)
    2. Filtering
        .filter_sessions(...)
    3. Computation
        .perform_computations(...)
    4. Display
        .prepare_for_display(...)
        
    Usage:
    > From properties:
        curr_kdiba_pipeline = NeuropyPipeline(name='kdiba_pipeline', session_data_type='kdiba', basedir=known_data_session_type_dict['kdiba'].basedir, load_function=known_data_session_type_dict['kdiba'].load_function)
    
    > From KnownDataSessionTypeProperties object:
        curr_kdiba_pipeline = NeuropyPipeline.init_from_known_data_session_type('kdiba', known_data_session_type_dict['kdiba'])



    """
    
    sigStageChanged = QtCore.Signal(object) # Emitted when the pipeline stage changes
    
    
    def __init__(self, name="pipeline", session_data_type='kdiba', basedir=None, outputs_specifier: Optional[OutputsSpecifier]=None, load_function: Callable = None, post_load_functions: List[Callable] = [], parent=None, **kwargs):
        """ 
        Captures:
            pipeline_module_logger (from module)
        """
        super(NeuropyPipeline, self).__init__(parent, **kwargs)
        self.pipeline_name = name
        self.session_data_type = None
        self._stage = None
        self._logger = pipeline_module_logger
        self._logger.info(f'NeuropyPipeline.__init__(name="{name}", session_data_type="{session_data_type}", basedir="{basedir}")')
        
        self._persistance_state = None # indicate that this pipeline doesn't have a corresponding pickle file that it was loaded from
        
        self._plot_object = None
        self._registered_output_files = None # for RegisteredOutputsMixin
        
        _stage_changed_connection = self.sigStageChanged.connect(self.on_stage_changed)
        self.set_input(name=name, session_data_type=session_data_type, basedir=basedir, load_function=load_function, post_load_functions=post_load_functions)

        if outputs_specifier is None:
            outputs_specifier = OutputsSpecifier(basedir)
        self._outputs_specifier = outputs_specifier



    def on_stage_changed(self, new_stage):
        # print(f'NeuropyPipeline.on_stage_changed(new_stage="{new_stage.identity}")')
        self.logger.info(f'NeuropyPipeline.on_stage_changed(new_stage="{new_stage.identity}")')

    @classmethod
    def init_from_known_data_session_type(cls, type_name: str, known_type_properties: KnownDataSessionTypeProperties, override_basepath=None, outputs_specifier: Optional[OutputsSpecifier]=None, override_post_load_functions=None):
        """ Initializes a new pipeline from a known data session type (e.g. 'bapun' or 'kdiba', which loads some defaults) """
        if override_basepath is not None:
            basepath = override_basepath
        else:
            basepath = known_type_properties.basedir
        if override_post_load_functions is not None:
            post_load_functions = override_post_load_functions
        else:
            post_load_functions = known_type_properties.post_load_functions
            
        return cls(name=f'{type_name}_pipeline', session_data_type=type_name, basedir=basepath, outputs_specifier=outputs_specifier,
            load_function=known_type_properties.load_function, post_load_functions=post_load_functions)

    # Load/Save Persistance and Comparison _______________________________________________________________________________ #
    @classmethod
    def try_init_from_saved_pickle_or_reload_if_needed(cls, type_name: str, known_type_properties: KnownDataSessionTypeProperties, override_basepath=None, outputs_specifier: Optional[OutputsSpecifier]=None, override_post_load_functions=None, force_reload=False, active_pickle_filename='loadedSessPickle.pkl', skip_save_on_initial_load=True, progress_print=True, debug_print=False):
        """ After a session has completed the loading stage prior to filtering (after all objects are built and such), it can be pickled to a file to drastically speed up future loading requests (as would have to be done when the notebook is restarted, etc) 
        Tries to find an extant pickled pipeline, and if it exists it loads and returns that. Otherwise, it loads/rebuilds the pipeline from scratch (from the initial raw data files) and then saves a pickled copy out to disk to speed up future loading attempts.
        
        force_reload: bool - If True, the pipeline isn't attempted to be loaded and instead is created fresh each time
        # skip_save_on_initial_load: Bool - if True, the resultant pipeline is not saved to the pickle when done (allowing more computations before saving)
        override_output_basepath: specifies an alternative output folder that things are saved to.
        
        """
        def _ensure_unpickled_pipeline_up_to_date(curr_active_pipeline, active_data_mode_name, basedir, desired_time_variable_name, debug_print=False):
            """ Ensures that all sessions in the pipeline are valid after unpickling, and updates them if they aren't.
            # TODO: NOTE: this doesn't successfully save the changes to the spikes_df.time_variable_name to the pickle (or doesn't load them). This probably can't be pickled and would need to be set on startup.
            
            Usage:
                
                desired_time_variable_name = active_data_mode_registered_class._time_variable_name # Requires desired_time_variable_name
                pipeline_needs_resave = _ensure_unpickled_session_up_to_date(curr_active_pipeline, active_data_mode_name=active_data_mode_name, basedir=basedir, desired_time_variable_name=desired_time_variable_name, debug_print=False)

                ## Save out the changes to the pipeline after computation to the pickle file for easy loading in the future
                if pipeline_needs_resave:
                    curr_active_pipeline.save_pipeline(active_pickle_filename='loadedSessPickle.pkl')
                else:
                    print(f'property already present in pickled version. No need to save.')
            
            """
            def _ensure_unpickled_session_up_to_date(a_sess, active_data_mode_name, basedir, desired_time_variable_name, debug_print=False):
                """ makes sure that the passed in session which was loaded from a pickled pipeline has the required properties and such set. Used for post-hoc fixes when changes are made after pickling. """
                did_add_property = False
                if not hasattr(a_sess.config, 'format_name'):
                    did_add_property = True
                    a_sess.config.format_name = active_data_mode_name
                if (a_sess.basepath != Path(basedir)):
                    did_add_property = True
                    a_sess.config.basepath = Path(basedir)
                if desired_time_variable_name != a_sess.spikes_df.spikes.time_variable_name:
                    if debug_print:
                        print(f'a_sess.spikes_df.spikes.time_variable_name: {a_sess.spikes_df.spikes.time_variable_name}')
                    # did_add_property = True
                    a_sess.spikes_df.spikes.set_time_variable_name(desired_time_variable_name)
                return did_add_property

            curr_active_pipeline.reload_default_computation_functions() # reloads the registered computation and display functions.
            curr_active_pipeline.reload_default_display_functions()

            did_add_property = False
            did_add_property = did_add_property or _ensure_unpickled_session_up_to_date(curr_active_pipeline.sess, active_data_mode_name=active_data_mode_name, basedir=basedir, desired_time_variable_name=desired_time_variable_name, debug_print=debug_print)
            ## Apply to all of the pipeline's filtered sessions:
            if hasattr(curr_active_pipeline, 'filtered_sessions'):
                for a_sess in curr_active_pipeline.filtered_sessions.values():
                    did_add_property = did_add_property or _ensure_unpickled_session_up_to_date(a_sess, active_data_mode_name=active_data_mode_name, basedir=basedir, desired_time_variable_name=desired_time_variable_name, debug_print=debug_print)
                return did_add_property

        ## BEGIN FUNCTION BODY
        if override_basepath is not None:
            basepath = override_basepath
            known_type_properties.basedir = override_basepath # change the known_type_properties default path to the specified override path
        else:
            basepath = known_type_properties.basedir
        if override_post_load_functions is not None:
            post_load_functions = override_post_load_functions
        else:
            post_load_functions = known_type_properties.post_load_functions
        
        ## Build Pickle Path:
        finalized_loaded_sess_pickle_path = Path(basepath).joinpath(active_pickle_filename).resolve()

        if not force_reload:
            if debug_print:
                print(f'finalized_loaded_sess_pickle_path: {finalized_loaded_sess_pickle_path}')
            try:
                loaded_pipeline = loadData(finalized_loaded_sess_pickle_path, debug_print=debug_print)
                
            except (FileNotFoundError):
                # loading failed
                print(f'Failure loading {finalized_loaded_sess_pickle_path}.')
                loaded_pipeline = None
            except EOFError:
                # file corrupted.
                print(f'Failure loading {finalized_loaded_sess_pickle_path}, the file is corrupted and incomplete (REACHED END OF FILE).')
                print(f'\t deleting it and continuing. ')
                finalized_loaded_sess_pickle_path.unlink() # .unlink() deletes a file
                print(f"\t {finalized_loaded_sess_pickle_path} deleted.")
                loaded_pipeline = None
            except Exception as e:
                raise e

            

        else:
            # Otherwise force recompute by setting 'loaded_pipeline = None':
            if progress_print:
                print(f'Skipping loading from pickled file because force_reload == True.')
            loaded_pipeline = None


        if loaded_pipeline is not None:
            ## Successful (at least partially) load
            if progress_print:
                print(f'Loading pickled pipeline success: {finalized_loaded_sess_pickle_path}.')
            if isinstance(loaded_pipeline, NeuropyPipeline):        
                curr_active_pipeline = loaded_pipeline
            else:
                # Otherwise we assume it's a complete computed pipeline pickeled result, in which case the pipeline is located in the 'curr_active_pipeline' key of the loaded dictionary.
                curr_active_pipeline = loaded_pipeline['curr_active_pipeline']
                
            ## Do patching/repair on the unpickled timeline and its sessions to ensure all correct properties are set:
            active_data_session_types_registered_classes_dict = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()
            active_data_mode_registered_class = active_data_session_types_registered_classes_dict[type_name]
            desired_time_variable_name = active_data_mode_registered_class._time_variable_name # Requires desired_time_variable_name
            pipeline_needs_resave = _ensure_unpickled_pipeline_up_to_date(curr_active_pipeline, active_data_mode_name=type_name, basedir=Path(basepath), desired_time_variable_name=desired_time_variable_name, debug_print=debug_print)
            
            curr_active_pipeline._persistance_state = LoadedObjectPersistanceState(finalized_loaded_sess_pickle_path, compare_state_on_load=curr_active_pipeline.pipeline_compare_dict)
            ## Save out the changes to the pipeline after computation to the pickle file for easy loading in the future
            if pipeline_needs_resave:
                if not skip_save_on_initial_load:
                    curr_active_pipeline.save_pipeline(active_pickle_filename=active_pickle_filename)
                else:
                    if progress_print:
                        print(f'pipeline_needs_resave but skip_save_on_initial_load == True, so saving will be skipped entirely. Be sure to save manually if there are changes.')
            else:
                if progress_print:
                    print(f'properties already present in pickled version. No need to save.')
    
            # If we reach this point, the load was a success
            if progress_print:
                print(f'pipeline load success!')


        else:
            # Otherwise load failed, perform the fallback computation
            if debug_print:
                print(f'Must reload/rebuild.')
            curr_active_pipeline = cls.init_from_known_data_session_type(type_name, known_type_properties, override_basepath=Path(basepath), outputs_specifier=outputs_specifier, override_post_load_functions=post_load_functions)
            # Save reloaded pipeline out to pickle for future loading
            if not skip_save_on_initial_load:
                saveData(finalized_loaded_sess_pickle_path, db=curr_active_pipeline) # 589 MB
            else:
                if progress_print:
                    print('skip_save_on_initial_load is True so resultant pipeline will not be saved to the pickle file.')
        # finalized_loaded_sess_pickle_path
        return curr_active_pipeline
    
    @classmethod
    def build_pipeline_compare_dict(cls, a_pipeline):
        """ Builds a dictionary that can be used to compare the progress of two neuropy_pipeline objects
        {'maze1_PYR': {'computation_config': DynamicContainer({'pf_params': <PlacefieldComputationParameters: {'speed_thresh': 10.0, 'grid_bin': (3.793023081021702, 1.607897707662558), 'smooth': (2.0, 2.0), 'frate_thresh': 0.2, 'time_bin_size': 0.1, 'computation_epochs': None};>, 'spike_analysis': DynamicContainer({'max_num_spikes_per_neuron': 20000, 'kleinberg_parameters': DynamicContainer({'s': 2, 'gamma': 0.2}), 'use_progress_bar': False, 'debug_print': False})}),
        'computed_data': ['pf1D',
        'pf2D',
        'pf1D_dt',
        'pf2D_dt',
        'pf2D_Decoder',
        'pf2D_TwoStepDecoder',
        'extended_stats']},
        'maze2_PYR': {'computation_config': DynamicContainer({'pf_params': <PlacefieldComputationParameters: {'speed_thresh': 10.0, 'grid_bin': (3.793023081021702, 1.607897707662558), 'smooth': (2.0, 2.0), 'frate_thresh': 0.2, 'time_bin_size': 0.1, 'computation_epochs': None};>, 'spike_analysis': DynamicContainer({'max_num_spikes_per_neuron': 20000, 'kleinberg_parameters': DynamicContainer({'s': 2, 'gamma': 0.2}), 'use_progress_bar': False, 'debug_print': False})}),
        'computed_data': ['pf1D',
        'pf2D',
        'pf1D_dt',
        'pf2D_dt',
        'pf2D_Decoder',
        'pf2D_TwoStepDecoder',
        'extended_stats']},
        'maze_PYR': {'computation_config': DynamicContainer({'pf_params': <PlacefieldComputationParameters: {'speed_thresh': 10.0, 'grid_bin': (3.793023081021702, 1.607897707662558), 'smooth': (2.0, 2.0), 'frate_thresh': 0.2, 'time_bin_size': 0.1, 'computation_epochs': None};>, 'spike_analysis': DynamicContainer({'max_num_spikes_per_neuron': 20000, 'kleinberg_parameters': DynamicContainer({'s': 2, 'gamma': 0.2}), 'use_progress_bar': False, 'debug_print': False})}),
        'computed_data': ['pf1D',
        'pf2D',
        'pf1D_dt',
        'pf2D_dt',
        'pf2D_Decoder',
        'pf2D_TwoStepDecoder',
        'extended_stats']}}

        Usage:
            compare_dict = build_pipeline_compare_dict(curr_active_pipeline)   
            compare_dict

        """
        out_results_dict = dict(last_completed_stage = a_pipeline.last_completed_stage,
                        active_config_names = None,
                        filtered_epochs = None,
                        filtered_session_names = None,
                        active_completed_computation_result_names = None,
                        active_incomplete_computation_result_status_dicts= None,
                        computation_result_computed_data_names = None,
            )
        
        # If prior to the filtered stage, not much to compare
        if a_pipeline.is_filtered:
            out_results_dict.update(active_config_names = tuple(a_pipeline.active_config_names),
                filtered_epochs = freeze(a_pipeline.filtered_epochs),
                filtered_session_names = tuple(a_pipeline.filtered_session_names)
            )
            
            
        if a_pipeline.is_computed:
            if hasattr(a_pipeline, 'computation_results'):
                comp_config_results_list = {}
                for a_name, a_result in a_pipeline.computation_results.items():
                    # ['sess', 'computation_config', 'computed_data', 'accumulated_errors']                    
                    try:
                        comp_config_results_list[a_name] = dict(computation_config=a_result.computation_config, computed_data=tuple((a_result.computed_data or {}).keys()))
                    except AttributeError as e:
                        # revert to old way
                        comp_config_results_list[a_name] = dict(computation_config=a_result['computation_config'], computed_data=tuple(a_result['computed_data'].keys())) # old way
                        # New way should raise `TypeError`: TypeError: 'ComputationResult' object is not subscriptable

                    except Exception:
                        # unhandled exception
                        raise
            else:
                comp_config_results_list = None

            # ## Add the global_computation_results to the comp_config_results_list with a common key '__GLOBAL__':
            # if hasattr(a_pipeline, 'global_computation_results'):
            #     if comp_config_results_list is None:
            #         comp_config_results_list = {}
            #     the_global_result = a_pipeline.global_computation_results
            #     comp_config_results_list['__GLOBAL__'] = dict(computation_config=(the_global_result.get('computation_config', {}) or {}), computed_data=tuple(the_global_result['computed_data'].keys()))
            # else:
            #     # has none
            #     pass # 

            out_results_dict.update(active_completed_computation_result_names = tuple(a_pipeline.active_completed_computation_result_names), # ['maze1_PYR', 'maze2_PYR', 'maze_PYR']
                active_incomplete_computation_result_status_dicts= freeze(a_pipeline.active_incomplete_computation_result_status_dicts),
                computation_result_computed_data_names = freeze(comp_config_results_list)
            )
            
        return out_results_dict

    @classmethod
    def compare_pipelines(cls, lhs, rhs, debug_print=False):
        lhs_compare_dict = cls.build_pipeline_compare_dict(lhs)
        rhs_compare_dict = cls.build_pipeline_compare_dict(rhs)
        curr_diff = DiffableObject.compute_diff(lhs_compare_dict, rhs_compare_dict)
        if debug_print:
            print(f'curr_diff: {curr_diff}')
        return curr_diff


    
    # ==================================================================================================================== #
    # Properties                                                                                                           #
    # ==================================================================================================================== #

    @property
    def sess(self):
        """The sess property, accessed through the stage."""
        return self.stage.sess

    @property
    def active_sess_config(self):
        """The active_sess_config property."""
        return self.sess.config

    @property
    def session_name(self):
        """The session_name property."""
        return self.sess.name


    # Persistance and Saving _____________________________________________________________________________________________ #
    @property
    def pipeline_compare_dict(self):
        """The pipeline_compare_dict property."""
        return NeuropyPipeline.build_pipeline_compare_dict(self)

    @property
    def persistance_state(self):
        """The persistance_state property."""
        return self._persistance_state
    
    @property
    def pickle_path(self):
        """ indicates that this pipeline doesn't have a corresponding pickle file that it was loaded from"""
        if self.persistance_state is None:
            return None
        else:
            return self.persistance_state.file_path

    @property
    def has_associated_pickle(self):
        """ True if this pipeline has a corresponding pickle file that it was loaded from"""
        return (self.pickle_path is not None)

    @property
    def updated_since_last_pickle(self):
        """ True if this pipeline has a been previously loaded/saved from a pickle file and has changed since this time
        TODO: currently due to object-level comparison between configs this seems to always return True
        """
        if self.persistance_state is None:
            return True # No previous known file (indicating it's never been saved), so return True.
        return self.persistance_state.needs_save(curr_object=self)

    # Logging ____________________________________________________________________________________________________________ #
    @property
    def logger(self):
        """The logger property."""
        return self._logger

    @property
    def logger_path(self):
        """Returns the active logging path of the logger. (e.g. 'C:\\Users\\pho\\repos\\PhoPy3DPositionAnalysis2021\\EXTERNAL\\TESTING\\Logging\\debug_com.PhoHale.Spike3D.pipeline.log') """
        curr_log_handler = self._logger.logger.handlers[0]
        return curr_log_handler.baseFilename

    # Stage and Progress _________________________________________________________________________________________________ #
    @property
    def stage(self):
        """The stage property."""
        return self._stage
    @stage.setter
    def stage(self, value):
        self._stage = value
        self.sigStageChanged.emit(self._stage) # pass the new stage
    
    @property
    def last_completed_stage(self) -> PipelineStage:
        """The last_completed_stage property."""
        return self.stage.identity
    
    # Filtered Properties: _______________________________________________________________________________________________ #
    @property
    def is_filtered(self):
        """The is_filtered property."""
        return (self.stage is not None) and (isinstance(self.stage, ComputedPipelineStage))
 
    def filter_sessions(self, active_session_filter_configurations, changed_filters_ignore_list=None, debug_print=False):
        """ 
            changed_filters_ignore_list: <list> a list of names of changed filters which will be ignored if they exists
        """
        if changed_filters_ignore_list is None:
            changed_filters_ignore_list = []

        if self.is_filtered:
            # RESUSE LOADED FILTERING: If the loaded pipeline is already filtered, check to see if the filters match those that were previously applied. If they do, don't re-filter unless the user specifies to.
            prev_session_filter_configurations = {a_config_name:a_config.filter_config['filter_function'] for a_config_name, a_config in self.active_configs.items()}
            # print(f'prev_session_filter_configurations: {prev_session_filter_configurations}')
            # Check for any non-equal ones:
            is_common_filter_name = np.isin(list(active_session_filter_configurations.keys()), list(prev_session_filter_configurations.keys()))
            is_novel_filter_name = np.logical_not(is_common_filter_name)
            if debug_print:
                print(f'is_common_filter_name: {is_common_filter_name}')
                print(f'is_novel_filter_name: {is_novel_filter_name}')
            # novel_filter_names = list(active_session_filter_configurations.keys())[np.logical_not(np.isin(list(active_session_filter_configurations.keys()), list(prev_session_filter_configurations.keys())))]
            # novel_filter_names = [a_name for a_name in list(active_session_filter_configurations.keys()) if a_name not in list(prev_session_filter_configurations.keys())]
            common_filter_names = np.array(list(active_session_filter_configurations.keys()))[is_common_filter_name]
            novel_filter_names = np.array(list(active_session_filter_configurations.keys()))[is_novel_filter_name]
            if debug_print:
                print(f'common_filter_names: {common_filter_names}')
            if len(novel_filter_names) > 0:
                self.logger.info(f'novel_filter_names: {novel_filter_names}')
                print(f'novel_filter_names: {novel_filter_names}')
            ## Deal with filters with the same name, but different filter functions:
            # changed_filters_names_list = [a_config_name for a_config_name in common_filter_names if (inspect.getsource(prev_session_filter_configurations[a_config_name]) != inspect.getsource(active_session_filter_configurations[a_config_name]))] 
            changed_filters_names_list = [] # changed_filters_names_list: a list of filter names for filters that have changed but have the same name
            for a_config_name in common_filter_names:
                try:
                    prev_filter_src = inspect.getsource(prev_session_filter_configurations[a_config_name])
                    active_filter_src = inspect.getsource(active_session_filter_configurations[a_config_name])
                    if (prev_filter_src != active_filter_src):
                        if debug_print:
                            print(f'prev_filter_src != active_filter_src\nprev_filter_src:')
                            print(prev_filter_src)
                            print(f'active_filter_src:')
                            print(active_filter_src)
                        changed_filters_names_list.append(a_config_name) # if inspect works and there is a difference, add it to the changed list
                except OSError as e:
                    # OSError: source code not available
                    # if inspect fails for some reason, we should assume a difference to be safe and add it to the changed list
                    print(f'WARNING: inspect failed for {a_config_name} with error {e}. Assuming changed.')
                    changed_filters_names_list.append(a_config_name)
                except Exception as e:
                    raise e
            
            if debug_print:
                print(f'changed_filters_names_list: {changed_filters_names_list}')
            unprocessed_filters = {a_config_name:active_session_filter_configurations[a_config_name] for a_config_name in changed_filters_names_list if a_config_name not in changed_filters_ignore_list}
            ignored_changed_filters_list = [a_config_name for a_config_name in changed_filters_names_list if a_config_name in changed_filters_ignore_list]
            if len(ignored_changed_filters_list) > 0:
                print(f'WARNING: changed_filters_names_list > 0!: {changed_filters_names_list} but these filters are in the changed_filters_ignore_list: {changed_filters_ignore_list}\nignored_changed_filters_list: {ignored_changed_filters_list}')
            # assert len(changed_filters_names_list) == 0, f"WARNING: changed_filters_names_list > 0!: {changed_filters_names_list}"
            # if len(changed_filters_names_list) > 0:
            #     print(f'WARNING: changed_filters_names_list > 0!: {changed_filters_names_list}')
            for a_novel_filter_name in novel_filter_names:
                unprocessed_filters[a_novel_filter_name] = active_session_filter_configurations[a_novel_filter_name]

            ## TODO: filter for the new and changed filters here:
            self.stage.select_filters(unprocessed_filters, clear_filtered_results=False, progress_logger=self.logger) # select filters when done
    
        else:
            # Not previously filtered. Perform the filtering:
            self.stage = ComputedPipelineStage(self.stage)
            self.stage.select_filters(active_session_filter_configurations, progress_logger=self.logger) # select filters when done
       
    

    # ==================================================================================================================== #
    # Session Pickling for Loading/Saving                                                                                  #
    # ==================================================================================================================== #

    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['_logger']
        del state['_persistance_state']
        del state['_plot_object']
        #TODO 2023-06-09 12:06: - [ ] What about the display objects?

        # ## self.stage approach:
        # stage = state.get('stage', None)
        # if stage is not None:
        #     del stage['display_output'] # self.stage.display_output
        #     del stage['display_output'] # self.stage.display_output

        try:
            del state['display_output'] # self.stage.display_output
        except KeyError:
            print(f'pipeline does not have "display_output"')
            pass
        except BaseException:
            raise

        try:
            del state['render_actions'] # self.stage.display_output
        except KeyError:
            print(f'pipeline does not have "render_actions"')
            pass
        except BaseException:
            raise

        del state['_registered_output_files']
        del state['_outputs_specifier']
        # del state['_pickle_path']

        ## What about the registered functions? It seems like there are a lot of objects to drop.
        
        #TODO 2023-08-10 19:24: - [ ] Looks like the majority of the data (which is in `self.stage` aka `state['_stage']`) is pickled as a `DisplayPipelineStage` object and not a dictionary, meaning there might be considerable overhead.
        # Below was my cursory attempt to remove these extra properties.`

        # state_variable_names_to_remove = [
        #     "registered_computation_function_dict",
        #     "registered_global_computation_function_dict",
        #     "display_output",
        #     "render_actions",
        #     "registered_display_function_dict",
        #     "post_load_functions",
        #     "registered_load_function_dict"
        # ]
        
        # stage = state['_stage'] #.__dict__
        # # remove properties from the stage

        # for state_v_name in state_variable_names_to_remove:
        #     if hasattr(stage, state_v_name):
        #         delattr(stage, state_v_name) # removes the attribute from the class 
        #         # del state['_stage'][state_v_name]
        
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        self.__dict__.update(state)
        # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        super(NeuropyPipeline, self).__init__() # from 
        
        # Restore unpickable properties:
        self._logger = pipeline_module_logger
        self._logger.info(f'NeuropyPipeline.__setstate__(state="{state}")')

        self._persistance_state = None # the pickle_path has to be set manually after loading
        self._plot_object = None
        self._registered_output_files = None # for RegisteredOutputsMixin
        self._outputs_specifier = None
        
        _stage_changed_connection = self.sigStageChanged.connect(self.on_stage_changed)
         
        # Reload both the computation and display functions to get the updated values:
        self.reload_default_computation_functions()
        self.reload_default_display_functions()
        self.clear_display_outputs()
        self.clear_registered_output_files() # outputs are reset each load, should they be?



    def save_pipeline(self, saving_mode=PipelineSavingScheme.TEMP_THEN_OVERWRITE, active_pickle_filename='loadedSessPickle.pkl', override_pickle_path: Optional[Path]=None):
        """ pickles (saves) the entire pipeline to a file that can be loaded later without recomputing.

        Args:
            active_pickle_filename (str, optional): _description_. Defaults to 'loadedSessPickle.pkl'.

        Returns:
            _type_: returns the finalized save path if the file was saved, or none if saving_mode=PipelineSavingScheme.SKIP_SAVING
            
            
        TODO
        """
        #TODO 2023-06-13 17:30: - [ ] The current operation doesn't make sense for PipelineSavingScheme.TEMP_THEN_OVERWRITE: it currently saves the new pipeline to a temporary file before replacing the existing file. This means after the operation there are two identical copies of the same pipeline. 
            # should make a backup of the existing file, not the new one
        saving_mode = PipelineSavingScheme.init(saving_mode)

        if not saving_mode.shouldSave:
            print(f'WARNING: saving_mode is SKIP_SAVING so pipeline will not be saved despite calling .save_pipeline(...).')
            self.logger.warning(f'WARNING: saving_mode is SKIP_SAVING so pipeline will not be saved despite calling .save_pipeline(...).')
            return None
        else:
            ## Build Pickle Path:
            used_existing_pickle_path = False
            if (active_pickle_filename is None) and (override_pickle_path is None):
                # simplest case, use existing path because nothing is provided
                assert self.has_associated_pickle
                finalized_loaded_sess_pickle_path = self.pickle_path # get the internal pickle path that it was loaded from if none specified
                
                # get existing pickle path:
                # finalized_loaded_sess_pickle_path.stem
                used_existing_pickle_path = True

            else:
                if override_pickle_path is not None:
                    if not override_pickle_path.is_dir():
                        # a full filepath, just use that directly
                        finalized_loaded_sess_pickle_path = override_pickle_path.resolve()
                    else:
                        # default case, assumed to be a directory and we'll use the normal filename.
                        assert self.has_associated_pickle
                        # get existing pickle filename:
                        active_pickle_filename = self.pickle_path.name
                        finalized_loaded_sess_pickle_path = Path(override_pickle_path).joinpath(active_pickle_filename).resolve()                     
                else:
                    # use `self.sess.basepath`
                    finalized_loaded_sess_pickle_path = Path(self.sess.basepath).joinpath(active_pickle_filename).resolve() # Uses the './loadedSessPickle.pkl' path

                # finalized_loaded_sess_pickle_path = self.get_output_path().joinpath(active_pickle_filename).resolve() # Changed to use the 'output/loadedSessPickle.pkl' directory 
                used_existing_pickle_path = (finalized_loaded_sess_pickle_path == self.pickle_path) # used the existing path if they're the same
            
            self.logger.info(f'save_pipeline(): Attempting to save pipeline to disk...')

            print(f'finalized_loaded_sess_pickle_path: {finalized_loaded_sess_pickle_path}')
            self.logger.info(f'\tfinalized_loaded_sess_pickle_path: {finalized_loaded_sess_pickle_path}')

            new_obj_memory_usage_MB = print_object_memory_usage(self, enable_print=False)

            is_temporary_file_used:bool = False
            _desired_finalized_loaded_sess_pickle_path = None
            if finalized_loaded_sess_pickle_path.exists():
                # file already exists:
                if saving_mode.name == PipelineSavingScheme.TEMP_THEN_OVERWRITE.name:
                    ## Save under a temporary name in the same output directory, and then compare post-hoc
                    _desired_finalized_loaded_sess_pickle_path = finalized_loaded_sess_pickle_path
                    finalized_loaded_sess_pickle_path, _ = build_unique_filename(finalized_loaded_sess_pickle_path) # changes the final path to the temporary file created.
                    is_temporary_file_used = True # this is the only condition where this is true
                elif saving_mode.name == PipelineSavingScheme.OVERWRITE_IN_PLACE.name:
                    print(f'WARNING: saving_mode is OVERWRITE_IN_PLACE so {finalized_loaded_sess_pickle_path} will be overwritten even though exists.')
                    self.logger.warning(f'WARNING: saving_mode is OVERWRITE_IN_PLACE so {finalized_loaded_sess_pickle_path} will be overwritten even though exists.')
                
            # Save reloaded pipeline out to pickle for future loading
            try:
                saveData(finalized_loaded_sess_pickle_path, db=self) # Save the pipeline out to pickle.
            except Exception as e:
                raise e
            
            # If we saved to a temporary name, now see if we should overwrite or backup and then replace:
            if (is_temporary_file_used and (saving_mode.name == PipelineSavingScheme.TEMP_THEN_OVERWRITE.name)):
                assert _desired_finalized_loaded_sess_pickle_path is not None
                prev_extant_file_size_MB = print_filesystem_file_size(_desired_finalized_loaded_sess_pickle_path, enable_print=False)
                new_temporary_file_size_MB = print_filesystem_file_size(finalized_loaded_sess_pickle_path, enable_print=False)

                if (prev_extant_file_size_MB > new_temporary_file_size_MB):
                    print(f'WARNING: prev_extant_file_size_MB ({prev_extant_file_size_MB} MB) > new_temporary_file_size_MB ({new_temporary_file_size_MB} MB)! A backup will be made!')
                    self.logger.warning(f'WARNING: prev_extant_file_size_MB ({prev_extant_file_size_MB} MB) > new_temporary_file_size_MB ({new_temporary_file_size_MB} MB)! A backup will be made!')
                    # Backup old file:
                    backup_extant_file(_desired_finalized_loaded_sess_pickle_path) # only backup if the new file is smaller than the older one (meaning the older one has more info)
                
                # replace the old file with the new one:
                print(f"moving new output at '{finalized_loaded_sess_pickle_path}' -> to desired location: '{_desired_finalized_loaded_sess_pickle_path}'")
                self.logger.info(f"moving new output at '{finalized_loaded_sess_pickle_path}' -> to desired location: '{_desired_finalized_loaded_sess_pickle_path}'")
                shutil.move(finalized_loaded_sess_pickle_path, _desired_finalized_loaded_sess_pickle_path) # move the temporary file to the desired destination, overwriting it
                # Finally restore the appropriate load path:
                finalized_loaded_sess_pickle_path = _desired_finalized_loaded_sess_pickle_path

            if not used_existing_pickle_path:
                # the pickle path changed, so set it on the pipeline:
                self._persistance_state = LoadedObjectPersistanceState(finalized_loaded_sess_pickle_path, compare_state_on_load=self.pipeline_compare_dict)
            
            self.logger.info(f'\t save complete.')
            return finalized_loaded_sess_pickle_path



    # ==================================================================================================================== #
    # 2023-08-02 - Session Exporting for HDF5                                                                              #
    # ==================================================================================================================== #
    # HDF_SerializationMixin

    # HDFMixin Conformances ______________________________________________________________________________________________ #
    def _export_global_computations_to_hdf(self, file_path, key: str, **kwargs):
        """ exports the self.global_computation_results to HDF file specified by file_path, key. """
        from pyphoplacecellanalysis.General.Batch.AcrossSessionResults import AcrossSessionsResults # for build_neuron_identity_table_to_hdf
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import JonathanFiringRateAnalysisResult        
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import InstantaneousSpikeRateGroupsComputation
        from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import ExpectedVsObservedResult

        session_context = self.get_session_context() 
        session_group_key: str = "/" + session_context.get_description(separator="/", include_property_names=False) # 'kdiba/gor01/one/2006-6-08_14-26-15'
        session_uid: str = session_context.get_description(separator="|", include_property_names=False)
        ## Global Computations
        a_global_computations_group_key: str = f"{session_group_key}/global_computations"
        with tb.open_file(file_path, mode='w') as f:
            a_global_computations_group = f.create_group(session_group_key, 'global_computations', title='the result of computations that operate over many or all of the filters in the session.', createparents=True)
            
        # Handle long|short firing rate index:
        long_short_fr_indicies_analysis_results = self.global_computation_results.computed_data['long_short_fr_indicies_analysis']
        x_frs_index, y_frs_index = long_short_fr_indicies_analysis_results['x_frs_index'], long_short_fr_indicies_analysis_results['y_frs_index'] # use the all_results_dict as the computed data value
        active_context = long_short_fr_indicies_analysis_results['active_context']
        # Need to map keys of dict to an absolute dict value:
        sess_specific_aclus = list(x_frs_index.keys())
        session_ctxt_key:str = active_context.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())

        long_short_fr_indicies_analysis_results_h5_df = pd.DataFrame([(f"{session_ctxt_key}|{aclu}", session_ctxt_key, aclu, x_frs_index[aclu], y_frs_index[aclu]) for aclu in sess_specific_aclus], columns=['neuron_uid', 'session_uid', 'aclu','x_frs_index', 'y_frs_index'])
        long_short_fr_indicies_analysis_results_h5_df.to_hdf(file_path, key=f'{a_global_computations_group_key}/long_short_fr_indicies_analysis', format='table', data_columns=True)

        # long_short_post_decoding result: __________________________________________________________________________________ #
        curr_long_short_post_decoding = self.global_computation_results.computed_data['long_short_post_decoding']
        expected_v_observed_result, curr_long_short_rr = curr_long_short_post_decoding.expected_v_observed_result, curr_long_short_post_decoding.rate_remapping
        rate_remapping_df, high_remapping_cells_only = curr_long_short_rr.rr_df, curr_long_short_rr.high_only_rr_df
        # Flat_epoch_time_bins_mean, Flat_decoder_time_bin_centers, num_neurons, num_timebins_in_epoch, num_total_flat_timebins, is_short_track_epoch, is_long_track_epoch, short_short_diff, long_long_diff = expected_v_observed_result.Flat_epoch_time_bins_mean, expected_v_observed_result.Flat_decoder_time_bin_centers, expected_v_observed_result.num_neurons, expected_v_observed_result.num_timebins_in_epoch, expected_v_observed_result.num_total_flat_timebins, expected_v_observed_result.is_short_track_epoch, expected_v_observed_result.is_long_track_epoch, expected_v_observed_result.short_short_diff, expected_v_observed_result.long_long_diff


        # Rate Remapping _____________________________________________________________________________________________________ #
        rate_remapping_df = rate_remapping_df[['laps',	'replays',	'skew',	'max_axis_distance_from_center',	'distance_from_center', 	'has_considerable_remapping']]
        rate_remapping_df.to_hdf(file_path, key=f'{a_global_computations_group_key}/rate_remapping', format='table', data_columns=True)

        # jonathan_firing_rate_analysis_result _______________________________________________________________________________ #
        jonathan_firing_rate_analysis_result: JonathanFiringRateAnalysisResult = self.global_computation_results.computed_data.jonathan_firing_rate_analysis
        jonathan_firing_rate_analysis_result.to_hdf(file_path=file_path, key=f'{a_global_computations_group_key}/jonathan_fr_analysis', active_context=session_context)

        # InstantaneousSpikeRateGroupsComputation ____________________________________________________________________________ #
        try:
            inst_spike_rate_groups_result: InstantaneousSpikeRateGroupsComputation = self.global_computation_results.computed_data.long_short_inst_spike_rate_groups # = InstantaneousSpikeRateGroupsComputation(instantaneous_time_bin_size_seconds=0.01) # 10ms
            # inst_spike_rate_groups_result.compute(curr_active_pipeline=self, active_context=self.sess.get_context())
            inst_spike_rate_groups_result.to_hdf(file_path, f'{a_global_computations_group_key}/inst_fr_comps') # held up by SpikeRateTrends.inst_fr_df_list  # to HDF, don't need to split it
        except KeyError:
            print(f'long_short_inst_spike_rate_groups is missing and will be skipped')
        except BaseException:
            raise

        if not isinstance(expected_v_observed_result, ExpectedVsObservedResult):
            expected_v_observed_result = ExpectedVsObservedResult(**expected_v_observed_result.to_dict())
        
        expected_v_observed_result.to_hdf(file_path=file_path, key=f'{a_global_computations_group_key}/expected_v_observed_result', active_context=session_context) # 'output/test_ExpectedVsObservedResult.h5', '/expected_v_observed_result')


        ##TODO: remainder of global_computations
        # self.global_computation_results.to_hdf(file_path, key=f'{a_global_computations_group_key}')

        AcrossSessionsResults.build_neuron_identity_table_to_hdf(file_path, key=session_group_key, spikes_df=self.sess.spikes_df, session_uid=session_uid)


    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path
        
        Built from `build_processed_session_to_hdf(...)` on 2023-08-02
        
        """
        
        long_epoch_name, short_epoch_name, global_epoch_name = self.find_LongShortGlobal_epoch_names()

        # f.create_dataset(f'{key}/neuron_ids', data=a_sess.neuron_ids)
        # f.create_dataset(f'{key}/shank_ids', data=self.shank_ids)            
        session_context = self.get_session_context() 
        session_group_key: str = "/" + session_context.get_description(separator="/", include_property_names=False) # 'kdiba/gor01/one/2006-6-08_14-26-15'
        session_uid: str = session_context.get_description(separator="|", include_property_names=False)
        # print(f'session_group_key: {session_group_key}')

        self.sess.to_hdf(file_path=file_path, key=f"{session_group_key}/sess")

        ## Global Computations
        self._export_global_computations_to_hdf(file_path, key=key)

        # Filtered Session Results:
        for an_epoch_name in (long_epoch_name, short_epoch_name, global_epoch_name):
            filter_context_key:str = "/" + self.filtered_contexts[an_epoch_name].get_description(separator="/", include_property_names=False) # '/kdiba/gor01/one/2006-6-08_14-26-15/maze1'
            # print(f'\tfilter_context_key: {filter_context_key}')
            with tb.open_file(file_path, mode='a') as f:
                a_filter_group = f.create_group(session_group_key, an_epoch_name, title='the result of a filter function applied to the session.', createparents=True)

            filtered_session = self.filtered_sessions[an_epoch_name]
            filtered_session.to_hdf(file_path=file_path, key=f"{filter_context_key}/sess")

            a_results = self.computation_results[an_epoch_name]
            a_computed_data = a_results['computed_data']
            a_computed_data.pf1D.to_hdf(file_path=file_path, key=f"{filter_context_key}/pf1D") # damn this will be called with the `tb` still having the thingy open
            a_computed_data.pf2D.to_hdf(file_path=file_path, key=f"{filter_context_key}/pf2D")
            ## TODO: encode the rest of the computed_data



        # Done, in future could potentially return the properties that it couldn't serialize so the defaults can be tried on them.
        # or maybe returns groups? a_filter_group
        ## Actually instead of returning, I think it should call super like so:
        # super().to_hdf(self, file_path, key, **kwargs)


    @property
    def h5_export_path(self):
        hdf5_output_path: Path = self.get_output_path().joinpath('pipeline_results.h5').resolve()
        return hdf5_output_path


    def export_pipeline_to_h5(self, fail_on_exception:bool=True):
        """ Export the pipeline's HDF5 as 'pipeline_results.h5'

        TODO: check timestamp of last computed file.

        """
        hdf5_output_path: Path = self.h5_export_path # get_output_path().joinpath('pipeline_results.h5').resolve()
        print(f'pipeline hdf5_output_path: {hdf5_output_path}')
        e = None
        try:
            self.to_hdf(file_path=hdf5_output_path, key="/")
            return (hdf5_output_path, None)
        except Exception as e:
            exception_info = sys.exc_info()
            e = CapturedException(e, exception_info)
            print(f"ERROR: encountered exception {e} while trying to build the session HDF output.")
            if fail_on_exception:
                raise e.exc
            hdf5_output_path = None # set to None because it failed.
            return (hdf5_output_path, e)
        
    


    @classmethod
    def read_hdf(cls, file_path, key: str, **kwargs) -> "NeuropyPipeline":
        """ Reads the data from the key in the hdf5 file at file_path
        
        """
        raise NotImplementedError



