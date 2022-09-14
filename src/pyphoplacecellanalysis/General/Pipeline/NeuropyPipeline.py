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

# NeuroPy (Diba Lab Python Repo) Loading
from neuropy import core
importlib.reload(core)

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
                
            ## Do patching/repair on the unpickled timeline and its sessions to ensure all correct properties are set:
            active_data_session_types_registered_classes_dict = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()
            active_data_mode_registered_class = active_data_session_types_registered_classes_dict[type_name]
            desired_time_variable_name = active_data_mode_registered_class._time_variable_name # Requires desired_time_variable_name
            pipeline_needs_resave = _ensure_unpickled_pipeline_up_to_date(curr_active_pipeline, active_data_mode_name=type_name, basedir=Path(basepath), desired_time_variable_name=desired_time_variable_name, debug_print=False)
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

    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['logger']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        self.__dict__.update(state)
        # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        super(NeuropyPipeline, self).__init__() # from 
        
        # Restore unpickable properties:
        self.logger = pipeline_module_logger
        self.logger.info(f'NeuropyPipeline.__setstate__(state="{state}")')
        
        
        _stage_changed_connection = self.sigStageChanged.connect(self.on_stage_changed)
         

    def save_pipeline(self, active_pickle_filename='loadedSessPickle.pkl'):
        ## Build Pickle Path:
        finalized_loaded_sess_pickle_path = Path(self.sess.basepath).joinpath(active_pickle_filename).resolve()
        print(f'finalized_loaded_sess_pickle_path: {finalized_loaded_sess_pickle_path}')
        # Save reloaded pipeline out to pickle for future loading
        saveData(finalized_loaded_sess_pickle_path, db=self) # Save the pipeline out to pickle.
        return finalized_loaded_sess_pickle_path
        

    @staticmethod
    def try_load_pickled_pipeline_or_reload_if_needed(active_data_mode_name, active_data_mode_type_properties, basedir, override_post_load_functions=None, force_reload=False, active_pickle_filename='loadedSessPickle.pkl', skip_save=False, debug_print=False):
        """ After a session has completed the loading stage prior to filtering (after all objects are built and such), it can be pickled to a file to drastically speed up future loading requests (as would have to be done when the notebook is restarted, etc) 
        Tries to find an extant pickled pipeline, and if it exists it loads and returns that. Otherwise, it loads/rebuilds the pipeline from scratch (from the initial raw data files) and then saves a pickled copy out to disk to speed up future loading attempts.
        
        # skip_save: Bool - if True, the resultant pipeline is not saved to the pickle when done
        
        """
        ## Build Pickle Path:
        finalized_loaded_sess_pickle_path = Path(basedir).joinpath(active_pickle_filename).resolve()
        print(f'finalized_loaded_sess_pickle_path: {finalized_loaded_sess_pickle_path}')

        if not force_reload:
            try:
                loaded_pipeline = loadData(finalized_loaded_sess_pickle_path, debug_print=debug_print)
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
            if not skip_save:
                saveData(finalized_loaded_sess_pickle_path, db=curr_active_pipeline) # 589 MB
        return curr_active_pipeline, finalized_loaded_sess_pickle_path