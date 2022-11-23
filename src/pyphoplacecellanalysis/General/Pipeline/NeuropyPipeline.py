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
from datetime import datetime

from typing import Callable, List
import inspect # used for filter_sessions(...)'s inspect.getsource to compare filters:

import numpy as np
import pandas as pd

# NeuroPy (Diba Lab Python Repo) Loading
from neuropy import core
importlib.reload(core)


from pyphocorehelpers.hashing_helpers import get_hash_tuple, freeze
from pyphocorehelpers.mixins.diffable import DiffableObject
from pyphocorehelpers.print_helpers import print_filesystem_file_size, print_object_memory_usage


from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder # hopefully this works without all the other imports
from neuropy.core.session.KnownDataSessionTypeProperties import KnownDataSessionTypeProperties

from pyphoplacecellanalysis.General.Pipeline.Stages.Computation import PipelineWithComputedPipelineStageMixin, ComputedPipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Display import PipelineWithDisplayPipelineStageMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.Filtering import FilteredPipelineMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import PipelineWithInputStage, PipelineWithLoadableStage, loadData, saveData
from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import PipelineStage

from qtpy import QtCore, QtWidgets, QtGui

# Pipeline Logging:
import logging
# from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import pipeline_module_logger
from pyphocorehelpers.print_helpers import build_module_logger
pipeline_module_logger = build_module_logger('Spike3D.pipeline')


class LoadedObjectPersistanceState(object):
    """Keeps track of the persistance state for an object that has been loaded from disk to keep track of how the object's state relates to the version on disk (the persisted version) """
    def __init__(self, file_path, compare_state_on_load):
        super(LoadedObjectPersistanceState, self).__init__()
        self.file_path = file_path
        self.load_compare_state = deepcopy(compare_state_on_load)
        
    def needs_save(self, curr_object) -> bool:
        """ compares the curr_object's state to its state when loaded from disk to see if anything changed and it needs to be re-persisted (by saving) """
        # lhs_compare_dict = NeuropyPipeline.build_pipeline_compare_dict(curr_object)
        curr_diff = DiffableObject.compute_diff(curr_object.pipeline_compare_dict, self.load_compare_state)        
        return len(curr_diff) > 0

    

class NeuropyPipeline(PipelineWithInputStage, PipelineWithLoadableStage, FilteredPipelineMixin, PipelineWithComputedPipelineStageMixin, PipelineWithDisplayPipelineStageMixin, QtCore.QObject):
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
    
    
    def __init__(self, name="pipeline", session_data_type='kdiba', basedir=None, load_function: Callable = None, post_load_functions: List[Callable] = [], parent=None, **kwargs):
        super(NeuropyPipeline, self).__init__(parent, **kwargs)
        self.pipeline_name = name
        self.session_data_type = None
        self._stage = None
        self.logger = pipeline_module_logger
        self.logger.info(f'NeuropyPipeline.__init__(name="{name}", session_data_type="{session_data_type}", basedir="{basedir}")')
        
        self._persistance_state = None # indicate that this pipeline doesn't have a corresponding pickle file that it was loaded from
        
        _stage_changed_connection = self.sigStageChanged.connect(self.on_stage_changed)
        self.set_input(name=name, session_data_type=session_data_type, basedir=basedir, load_function=load_function, post_load_functions=post_load_functions)


    def on_stage_changed(self, new_stage):
        print(f'NeuropyPipeline.on_stage_changed(new_stage="{new_stage.identity}")')
        self.logger.info(f'NeuropyPipeline.on_stage_changed(new_stage="{new_stage.identity}")')

    @classmethod
    def init_from_known_data_session_type(cls, type_name: str, known_type_properties: KnownDataSessionTypeProperties, override_basepath=None, override_post_load_functions=None):
        """ Initializes a new pipeline from a known data session type (e.g. 'bapun' or 'kdiba', which loads some defaults) """
        if override_basepath is not None:
            basepath = override_basepath
        else:
            basepath = known_type_properties.basedir
        if override_post_load_functions is not None:
            post_load_functions = override_post_load_functions
        else:
            post_load_functions = known_type_properties.post_load_functions
            
        return cls(name=f'{type_name}_pipeline', session_data_type=type_name, basedir=basepath,
            load_function=known_type_properties.load_function, post_load_functions=post_load_functions)


    @classmethod
    def try_init_from_saved_pickle_or_reload_if_needed(cls, type_name: str, known_type_properties: KnownDataSessionTypeProperties, override_basepath=None, override_post_load_functions=None, force_reload=False, active_pickle_filename='loadedSessPickle.pkl', skip_save=False):
        """ After a session has completed the loading stage prior to filtering (after all objects are built and such), it can be pickled to a file to drastically speed up future loading requests (as would have to be done when the notebook is restarted, etc) 
        Tries to find an extant pickled pipeline, and if it exists it loads and returns that. Otherwise, it loads/rebuilds the pipeline from scratch (from the initial raw data files) and then saves a pickled copy out to disk to speed up future loading attempts.
        
        # skip_save: Bool - if True, the resultant pipeline is not saved to the pickle when done
        
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
            print(f'finalized_loaded_sess_pickle_path: {finalized_loaded_sess_pickle_path}')
            try:
                loaded_pipeline = loadData(finalized_loaded_sess_pickle_path, debug_print=False)
                
            except (FileNotFoundError):
                # loading failed
                print(f'Failure loading {finalized_loaded_sess_pickle_path}.')
                loaded_pipeline = None

        else:
            # Otherwise force recompute:
            print(f'Skipping loading from pickled file because force_reload == True.')
            loaded_pipeline = None

        if loaded_pipeline is not None:
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
            pipeline_needs_resave = _ensure_unpickled_pipeline_up_to_date(curr_active_pipeline, active_data_mode_name=type_name, basedir=Path(basepath), desired_time_variable_name=desired_time_variable_name, debug_print=False)
            
            curr_active_pipeline._persistance_state = LoadedObjectPersistanceState(finalized_loaded_sess_pickle_path, compare_state_on_load=curr_active_pipeline.pipeline_compare_dict)
            ## Save out the changes to the pipeline after computation to the pickle file for easy loading in the future
            if pipeline_needs_resave:
                curr_active_pipeline.save_pipeline(active_pickle_filename=active_pickle_filename)
            else:
                print(f'property already present in pickled version. No need to save.')
    
        else:
            # Otherwise load failed, perform the fallback computation
            print(f'Must reload/rebuild.')
            curr_active_pipeline = cls.init_from_known_data_session_type(type_name, known_type_properties, override_basepath=Path(basepath), override_post_load_functions=post_load_functions)
            # Save reloaded pipeline out to pickle for future loading
            if not skip_save:
                saveData(finalized_loaded_sess_pickle_path, db=curr_active_pipeline) # 589 MB
            else:
                print('skip_save is True so resultant pipeline will not be saved to the pickle file.')
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
                    comp_config_results_list[a_name] = dict(computation_config=a_result['computation_config'], computed_data=tuple(a_result['computed_data'].keys()))
            else:
                comp_config_results_list = None

            # ## Add the global_computation_results to the comp_config_results_list:
            # if hasattr(a_pipeline, 'global_computation_results'):
            #     if comp_config_results_list is None:
            #         comp_config_results_list = {}
            #     for a_name, a_result in a_pipeline.global_computation_results.items():
            #         # ['sess', 'computation_config', 'computed_data', 'accumulated_errors']
            #         comp_config_results_list[a_name] = dict(computation_config=a_result['computation_config'], computed_data=tuple(a_result['computed_data'].keys()))


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
    
    ## Filtered Properties:
    @property
    def is_filtered(self):
        """The is_filtered property."""
        return (self.stage is not None) and (isinstance(self.stage, ComputedPipelineStage))
 
    def filter_sessions(self, active_session_filter_configurations, debug_print = False):
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
            changed_filters_names_list = [a_config_name for a_config_name in common_filter_names if (inspect.getsource(prev_session_filter_configurations[a_config_name]) != inspect.getsource(active_session_filter_configurations[a_config_name]))] # changed_filters_names_list: a list of filter names for filters that have changed but have the same name
            if debug_print:
                print(f'changed_filters_names_list: {changed_filters_names_list}')
            unprocessed_filters = {a_config_name:active_session_filter_configurations[a_config_name] for a_config_name in changed_filters_names_list}
            assert len(changed_filters_names_list) == 0, f"WARNING: changed_filters_names_list > 0!: {changed_filters_names_list}"
            # if len(changed_filters_names_list) > 0:
            #     print(f'WARNING: changed_filters_names_list > 0!: {changed_filters_names_list}')
            for a_novel_filter_name in novel_filter_names:
                unprocessed_filters[a_novel_filter_name] = active_session_filter_configurations[a_novel_filter_name]

            ## TODO: filter for the new and changed filters here:
            self.stage.select_filters(unprocessed_filters, clear_filtered_results=False, progress_logger=self.logger) # select filters when done
    
        else:
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
        del state['logger']
        del state['_persistance_state']
        # del state['_pickle_path']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        self.__dict__.update(state)
        # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        super(NeuropyPipeline, self).__init__() # from 
        
        # Restore unpickable properties:
        self.logger = pipeline_module_logger
        self.logger.info(f'NeuropyPipeline.__setstate__(state="{state}")')

        self._persistance_state = None # the pickle_path has to be set manually after loading
        
        _stage_changed_connection = self.sigStageChanged.connect(self.on_stage_changed)
         

    def save_pipeline(self, active_pickle_filename='loadedSessPickle.pkl'):
        """ pickles (saves) the entire pipeline to a file that can be loaded later without recomputing.

        Args:
            active_pickle_filename (str, optional): _description_. Defaults to 'loadedSessPickle.pkl'.

        Returns:
            _type_: _description_
        """
        ## Build Pickle Path:
        used_existing_pickle_path = False
        if active_pickle_filename is None:
            assert self.has_associated_pickle
            finalized_loaded_sess_pickle_path = self.pickle_path # get the internal pickle path that it was loaded from if none specified
            used_existing_pickle_path = True
        else:        
            finalized_loaded_sess_pickle_path = Path(self.sess.basepath).joinpath(active_pickle_filename).resolve()
            used_existing_pickle_path = (finalized_loaded_sess_pickle_path == self.pickle_path) # used the existing path if they're the same
        
        self.logger.info(f'save_pipeline(): Attempting to save pipeline to disk...')

        print(f'finalized_loaded_sess_pickle_path: {finalized_loaded_sess_pickle_path}')
        self.logger.info(f'\tfinalized_loaded_sess_pickle_path: {finalized_loaded_sess_pickle_path}')

        new_obj_memory_usage_MB = print_object_memory_usage(self, enable_print=False)

        _desired_finalized_loaded_sess_pickle_path = None
        if finalized_loaded_sess_pickle_path.exists():
            # file already exists:
            ## Save under a temporary name in the same output directory, and then compare post-hoc
            _desired_finalized_loaded_sess_pickle_path = finalized_loaded_sess_pickle_path
            finalized_loaded_sess_pickle_path, _ = _build_unique_filename(finalized_loaded_sess_pickle_path)


                

        # Save reloaded pipeline out to pickle for future loading
        saveData(finalized_loaded_sess_pickle_path, db=self) # Save the pipeline out to pickle.

        # If we saved to a temporary name, now see if we should overwrite or backup and then replace:
        if _desired_finalized_loaded_sess_pickle_path is not None:

            prev_extant_file_size_MB = print_filesystem_file_size(_desired_finalized_loaded_sess_pickle_path, enable_print=False)
            new_temporary_file_size_MB = print_filesystem_file_size(finalized_loaded_sess_pickle_path, enable_print=False)

            if (prev_extant_file_size_MB >= new_temporary_file_size_MB):
                print(f'WARNING: prev_extant_file_size_MB ({prev_extant_file_size_MB} MB) >= new_temporary_file_size_MB ({new_temporary_file_size_MB} MB)! A backup will be made!')
                # Backup old file:
                _backup_extant_file(_desired_finalized_loaded_sess_pickle_path) # only backup if the new file is smaller than the older one (meaning the older one has more info)
            
            # replace the old file with the new one:
            print(f"moving new output at '{finalized_loaded_sess_pickle_path}' -> to desired location: '{_desired_finalized_loaded_sess_pickle_path}'")
            shutil.move(finalized_loaded_sess_pickle_path, _desired_finalized_loaded_sess_pickle_path) # move the temporary file to the desired destination, overwriting it
            # Finally restore the appropriate load path:
            finalized_loaded_sess_pickle_path = _desired_finalized_loaded_sess_pickle_path

        if not used_existing_pickle_path:
            # the pickle path changed, so set it on the pipeline:
            self._persistance_state = LoadedObjectPersistanceState(finalized_loaded_sess_pickle_path, compare_state_on_load=self.pipeline_compare_dict)
        
        self.logger.info(f'\t save complete.')
        return finalized_loaded_sess_pickle_path



def _build_unique_filename(file_to_save_path, additional_postfix_extension=None):
    """ builds a unique filename for the file to be saved at file_to_save_path.
    Usage:
        from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import _build_unique_filename
        unique_save_path, unique_file_name = _build_unique_filename(curr_active_pipeline.pickle_path) # unique_file_name: '20221109173951-loadedSessPickle.pkl'
        unique_save_path # 'W:/Data/KDIBA/gor01/one/2006-6-09_1-22-43/20221109173951-loadedSessPickle.pkl'
    """
    if not isinstance(file_to_save_path, Path):
        file_to_save_path = Path(file_to_save_path)
    parent_path = file_to_save_path.parent # The location to store the backups in

    extensions = file_to_save_path.suffixes # e.g. ['.tar', '.gz']
    if additional_postfix_extension is not None:
        extensions.append(additional_postfix_extension)

    unique_file_name = f'{datetime.now().strftime("%Y%m%d%H%M%S")}-{file_to_save_path.stem}{"".join(extensions)}'
    unique_save_path = parent_path.joinpath(unique_file_name)
    # print(f"'{file_to_save_path}' backing up -> to_file: '{unique_save_path}'")
    return unique_save_path, unique_file_name



def _backup_extant_file(file_to_backup_path, MAX_BACKUP_AMOUNT=2):
    """creates a backup of an existing file that would otherwise be overwritten

    Args:
        file_to_backup_path (_type_): _description_
        MAX_BACKUP_AMOUNT (int, optional):  The maximum amount of backups to have in BACKUP_DIRECTORY. Defaults to 2.
    """
    if not isinstance(file_to_backup_path, Path):
        file_to_backup_path = Path(file_to_backup_path).resolve()
    assert file_to_backup_path.exists(), f"file at {file_to_backup_path} must already exist to be backed-up!"
    assert (not file_to_backup_path.is_dir()), f"file at {file_to_backup_path} must be a FILE, not a directory!"
    backup_extension = '.bak' # simple '.bak' file

    backup_directory_path = file_to_backup_path.parent # The location to store the backups in
    assert file_to_backup_path.exists()  # Validate the object we are about to backup exists before we continue

    # Validate the backup directory exists and create if required
    backup_directory_path.mkdir(parents=True, exist_ok=True)

    # Get the amount of past backup zips in the backup directory already
    existing_backups = [
        x for x in backup_directory_path.iterdir()
        if x.is_file() and x.suffix == backup_extension and x.name.startswith('backup-')
    ]

    # Enforce max backups and delete oldest if there will be too many after the new backup
    oldest_to_newest_backup_by_name = list(sorted(existing_backups, key=lambda f: f.name))
    while len(oldest_to_newest_backup_by_name) >= MAX_BACKUP_AMOUNT:  # >= because we will have another soon
        backup_to_delete = oldest_to_newest_backup_by_name.pop(0)
        backup_to_delete.unlink()

    # Create zip file (for both file and folder options)
    backup_file_name = f'backup-{datetime.now().strftime("%Y%m%d%H%M%S")}-{file_to_backup_path.name}{backup_extension}'
    to_file = backup_directory_path.joinpath(backup_file_name)
    print(f"'{file_to_backup_path}' backing up -> to_file: '{to_file}'")
    shutil.copy(file_to_backup_path, to_file)
    return True
    # dest = Path('dest')
    # src = Path('src')
    # dest.write_bytes(src.read_bytes()) #for binary files
    # dest.write_text(src.read_text()) #for text files