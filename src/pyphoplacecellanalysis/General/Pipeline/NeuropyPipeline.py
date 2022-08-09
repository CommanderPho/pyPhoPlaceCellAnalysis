#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
NeuropyPipeline.py
"""
import importlib
import sys
from pathlib import Path
from typing import Callable, List
import inspect # used for filter_sessions(...)'s inspect.getsource to compare filters:

import numpy as np
import pandas as pd
from pyphoplacecellanalysis.General.Pipeline.Stages.Computation import PipelineWithComputedPipelineStageMixin, ComputedPipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Display import PipelineWithDisplayPipelineStageMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.Filtering import FilteredPipelineMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import PipelineWithInputStage, PipelineWithLoadableStage, loadData, saveData

# from pyphoplacecellanalysis.General.SessionSelectionAndFiltering import batch_filter_session
from neuropy.core.session.KnownDataSessionTypeProperties import KnownDataSessionTypeProperties
from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import PipelineStage

# NeuroPy (Diba Lab Python Repo) Loading
from neuropy import core
importlib.reload(core)


class NeuropyPipeline(PipelineWithInputStage, PipelineWithLoadableStage, FilteredPipelineMixin, PipelineWithComputedPipelineStageMixin, PipelineWithDisplayPipelineStageMixin):
    """ 
    
    Exposes the active sessions via its .sess member.
    
    Stages:
    1. Input/Loading
    2. Filtering
    3. Computation
    4. Display
    
    Usage:
    > From properties:
        curr_kdiba_pipeline = NeuropyPipeline(name='kdiba_pipeline', session_data_type='kdiba', basedir=known_data_session_type_dict['kdiba'].basedir, load_function=known_data_session_type_dict['kdiba'].load_function)
    
    > From KnownDataSessionTypeProperties object:
        curr_kdiba_pipeline = NeuropyPipeline.init_from_known_data_session_type('kdiba', known_data_session_type_dict['kdiba'])

    """
    
    def __init__(self, name="pipeline", session_data_type='kdiba', basedir=None, load_function: Callable = None, post_load_functions: List[Callable] = []):
        # super(NeuropyPipeline, self).__init__()
        self.pipeline_name = name
        self.session_data_type = None
        self.stage = None
        self.set_input(name=name, session_data_type=session_data_type, basedir=basedir, load_function=load_function, post_load_functions=post_load_functions)


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
    def try_init_from_saved_pickle_or_reload_if_needed(cls, type_name: str, known_type_properties: KnownDataSessionTypeProperties, override_basepath=None, override_post_load_functions=None, force_reload=False, active_pickle_filename='loadedSessPickle.pkl'):
        """ After a session has completed the loading stage prior to filtering (after all objects are built and such), it can be pickled to a file to drastically speed up future loading requests (as would have to be done when the notebook is restarted, etc) 
        Tries to find an extant pickled pipeline, and if it exists it loads and returns that. Otherwise, it loads/rebuilds the pipeline from scratch (from the initial raw data files) and then saves a pickled copy out to disk to speed up future loading attempts.
        """
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
        print(f'finalized_loaded_sess_pickle_path: {finalized_loaded_sess_pickle_path}')

        if not force_reload:
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
        else:
            # Otherwise load failed, perform the fallback computation
            print(f'Must reload/rebuild.')
            curr_active_pipeline = cls.init_from_known_data_session_type(type_name, known_type_properties, override_basepath=Path(basepath), override_post_load_functions=post_load_functions)
            # Save reloaded pipeline out to pickle for future loading
            saveData(finalized_loaded_sess_pickle_path, db=curr_active_pipeline) # 589 MB
        # finalized_loaded_sess_pickle_path
        return curr_active_pipeline
    

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
            self.stage.select_filters(unprocessed_filters, clear_filtered_results=False) # select filters when done
    
        else:
            self.stage = ComputedPipelineStage(self.stage)
            self.stage.select_filters(active_session_filter_configurations) # select filters when done
       
    

    # ==================================================================================================================== #
    # Session Pickling for Loading/Saving                                                                                  #
    # ==================================================================================================================== #

    @staticmethod
    def try_load_pickled_pipeline_or_reload_if_needed(active_data_mode_name, active_data_mode_type_properties, basedir, override_post_load_functions=None, force_reload=False, active_pickle_filename='loadedSessPickle.pkl'):
        """ After a session has completed the loading stage prior to filtering (after all objects are built and such), it can be pickled to a file to drastically speed up future loading requests (as would have to be done when the notebook is restarted, etc) 
        Tries to find an extant pickled pipeline, and if it exists it loads and returns that. Otherwise, it loads/rebuilds the pipeline from scratch (from the initial raw data files) and then saves a pickled copy out to disk to speed up future loading attempts.
        
        """
        ## Build Pickle Path:
        finalized_loaded_sess_pickle_path = Path(basedir).joinpath(active_pickle_filename).resolve()
        print(f'finalized_loaded_sess_pickle_path: {finalized_loaded_sess_pickle_path}')

        if not force_reload:
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
            curr_active_pipeline = loaded_pipeline
        else:
            # Otherwise load failed, perform the fallback computation
            print(f'Must reload/rebuild.')
            curr_active_pipeline = NeuropyPipeline.init_from_known_data_session_type(active_data_mode_name, active_data_mode_type_properties, override_basepath=Path(basedir), override_post_load_functions=override_post_load_functions)
            # Save reloaded pipeline out to pickle for future loading
            saveData(finalized_loaded_sess_pickle_path, db=curr_active_pipeline) # 589 MB
        return curr_active_pipeline, finalized_loaded_sess_pickle_path