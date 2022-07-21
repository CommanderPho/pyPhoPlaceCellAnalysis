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

# from pyphocorehelpers.function_helpers import compose_functions

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
            curr_active_pipeline = loaded_pipeline
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
 
    def filter_sessions(self, active_session_filter_configurations):
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