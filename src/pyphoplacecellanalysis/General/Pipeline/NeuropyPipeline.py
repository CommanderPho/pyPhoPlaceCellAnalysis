#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
NeuropyPipeline.py
"""
import importlib
import sys
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from pyphocorehelpers.function_helpers import compose_functions

import numpy as np
import pandas as pd
from pyphoplacecellanalysis.General.Pipeline.Computation import ComputablePipelineStage, DefaultRegisteredComputations
from pyphoplacecellanalysis.General.Pipeline.Display import DefaultDisplayFunctions, DefaultRegisteredDisplayFunctions, add_neuron_identity_info_if_needed
from pyphoplacecellanalysis.General.Pipeline.Filtering import FilterablePipelineStage
from pyphoplacecellanalysis.General.Pipeline.Loading import LoadableInput, LoadableSessionInput

from pyphoplacecellanalysis.General.SessionSelectionAndFiltering import batch_filter_session

# NeuroPy (Diba Lab Python Repo) Loading
try:
    from neuropy import core

    importlib.reload(core)
except ImportError:
    sys.path.append(r"C:\Users\Pho\repos\NeuroPy")  # Windows
    # sys.path.append('/home/pho/repo/BapunAnalysis2021/NeuroPy') # Linux
    # sys.path.append(r'/Users/pho/repo/Python Projects/NeuroPy') # MacOS
    print("neuropy module not found, adding directory to sys.path. \n >> Updated sys.path.")
    from neuropy import core

# Neuropy:
from neuropy.core.session.data_session_loader import DataSessionLoader
from neuropy.core.session.dataSession import DataSession
from neuropy.analyses.placefields import PlacefieldComputationParameters, perform_compute_placefields




# def _temp_filter_session(sess):
#     """ 
#     Usage:
#         active_session, active_epoch = _temp_filter_session(curr_bapun_pipeline.sess)
#     """
#     # curr_kdiba_pipeline.sess.epochs['maze1']
#     active_epoch = sess.epochs.get_named_timerange('maze1')

#     ## All Spikes:
#     # active_epoch_session = sess.filtered_by_epoch(active_epoch) # old
#     active_session = batch_filter_session(sess, sess.position, sess.spikes_df, active_epoch.to_Epoch())
#     return active_session, active_epoch



@dataclass
class KnownDataSessionTypeProperties(object):
    """Docstring for KnownDataSessionTypeProperties."""
    load_function: Callable
    basedir: Path
    # Optional members
    post_load_functions: List[Callable] = dataclasses.field(default_factory=list)
    filter_functions: List[Callable] = dataclasses.field(default_factory=list)
    post_compute_functions: List[Callable] = dataclasses.field(default_factory=list)
    # filter_function: Callable = None


# known_data_session_type_dict = {'kdiba':KnownDataSessionTypeProperties(load_function=(lambda a_base_dir: DataSessionLoader.kdiba_old_format_session(a_base_dir)),
#                                basedir=Path(r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53')),
#                 'bapun':KnownDataSessionTypeProperties(load_function=(lambda a_base_dir: DataSessionLoader.bapun_data_session(a_base_dir)),
#                                basedir=Path('R:\data\Bapun\Day5TwoNovel'))
#                }


# load_fn_dict = {'kdiba':(lambda a_base_dir: DataSessionLoader.kdiba_old_format_session(a_base_dir)),
#                 'bapun':(lambda a_base_dir: DataSessionLoader.bapun_data_session(a_base_dir))
#                }


@dataclass
class BaseNeuropyPipelineStage(object):
    """ BaseNeuropyPipelineStage represents a single stage of a data session processing/rendering pipeline. """
    stage_name: str = ""


@dataclass
class InputPipelineStage(LoadableInput, BaseNeuropyPipelineStage):
    """Docstring for InputPipelineStage.
    
    post_load_functions: List[Callable] a list of Callables that accept the loaded session as input and return the potentially modified session as output.
    """
    basedir: Path = Path("")
    load_function: Callable = None
    post_load_functions: List[Callable] = dataclasses.field(default_factory=list)



class LoadedPipelineStage(LoadableInput, LoadableSessionInput, BaseNeuropyPipelineStage):
    """Docstring for LoadedPipelineStage."""
    loaded_data: dict = None

    def __init__(self, input_stage: InputPipelineStage):
        self.stage_name = input_stage.stage_name
        self.basedir = input_stage.basedir
        self.loaded_data = input_stage.loaded_data
        self.post_load_functions = input_stage.post_load_functions # the functions to be called post load


    def post_load(self, debug_print=False):
        """ Called after load is complete to post-process the data """
        if (len(self.post_load_functions) > 0):
            if debug_print:
                print(f'Performing on_post_load(...) with {len(self.post_load_functions)} post_load_functions...')
            # self.sess = compose_functions(self.post_load_functions, self.sess)
            composed_post_load_function = compose_functions(*self.post_load_functions) # functions are composed left-to-right
            self.sess = composed_post_load_function(self.sess)
            
        else:
            if debug_print:
                print(f'No post_load_functions, skipping post_load.')

        
        
            


class ComputedPipelineStage(LoadableInput, LoadableSessionInput, FilterablePipelineStage, DefaultRegisteredComputations, ComputablePipelineStage, BaseNeuropyPipelineStage):
    """Docstring for ComputedPipelineStage."""

    filtered_sessions: dict = None
    filtered_epochs: dict = None
    active_configs: dict = None
    computation_results: dict = None
    
    def __init__(self, loaded_stage: LoadedPipelineStage):
        # super(ClassName, self).__init__()
        self.stage_name = loaded_stage.stage_name
        self.basedir = loaded_stage.basedir
        self.loaded_data = loaded_stage.loaded_data

        # Initialize custom fields:
        self.filtered_sessions = dict()
        self.filtered_epochs = dict()
        self.active_configs = dict() # active_config corresponding to each filtered session/epoch
        self.computation_results = dict()
        self.registered_computation_functions = list()
        self.register_default_known_computation_functions() # registers the default
        
    def register_computation(self, computation_function):
        self.registered_computation_functions.append(computation_function)
        
    def perform_registered_computations(self, previous_computation_result, debug_print=False):
        """ Called after load is complete to post-process the data """
        if (len(self.registered_computation_functions) > 0):
            if debug_print:
                print(f'Performing perform_registered_computations(...) with {len(self.registered_computation_functions)} registered_computation_functions...')            
            composed_registered_computations_function = compose_functions(*self.registered_computation_functions) # functions are composed left-to-right
            previous_computation_result = composed_registered_computations_function(previous_computation_result)
            return previous_computation_result
            
        else:
            if debug_print:
                print(f'No registered_computation_functions, skipping extended computations.')
            return previous_computation_result # just return the unaltered result
    

class DisplayPipelineStage(ComputedPipelineStage):

    """docstring for DisplayPipelineStage."""
    def __init__(self, computed_stage: ComputedPipelineStage, render_actions=dict()):
        # super(DisplayPipelineStage, self).__init__()
        # ComputedPipelineStage fields:
        self.stage_name = computed_stage.stage_name
        self.basedir = computed_stage.basedir
        self.loaded_data = computed_stage.loaded_data
        self.filtered_sessions = computed_stage.filtered_sessions
        self.filtered_epochs = computed_stage.filtered_epochs
        self.active_configs = computed_stage.active_configs # active_config corresponding to each filtered session/epoch
        self.computation_results = computed_stage.computation_results
        self.registered_computation_functions = computed_stage.registered_computation_functions

        # Initialize custom fields:
        self.render_actions = render_actions    
    
        

# class ClassName(object):
#     """docstring for ClassName."""
#     def __init__(self, arg):
#         super(ClassName, self).__init__()
#         self.arg = arg


class NeuropyPipeline():
    """ 
    
    Exposes the active sessions via its .sess member.
    
    Stages:
    1. Loading
    2. Filtering
    3. Computation
    4. Display
    
    Usage:
    > From properties:
        curr_kdiba_pipeline = NeuropyPipeline(name='kdiba_pipeline', session_data_type='kdiba', basedir=known_data_session_type_dict['kdiba'].basedir, load_function=known_data_session_type_dict['kdiba'].load_function)
    
    > From KnownDataSessionTypeProperties object:
        curr_kdiba_pipeline = NeuropyPipeline.init_from_known_data_session_type('kdiba', known_data_session_type_dict['kdiba'])

    """
    
    def __init__(self, name="pipeline", session_data_type='kdiba', basedir=None,
                 load_function: Callable = None,
                 post_load_functions: List[Callable] = []):
        # super(NeuropyPipeline, self).__init__()
        self.pipeline_name = name
        self.session_data_type = None
        self.stage = None
        self.set_input(name=name, session_data_type=session_data_type, basedir=basedir, load_function=load_function, post_load_functions=post_load_functions)


    @classmethod
    def init_from_known_data_session_type(cls, type_name: str, known_type_properties: KnownDataSessionTypeProperties):
        return cls(name=f'{type_name}_pipeline', session_data_type=type_name, basedir=known_type_properties.basedir,
            load_function=known_type_properties.load_function, post_load_functions=known_type_properties.post_load_functions)

    @property
    def is_loaded(self):
        """The is_loaded property."""
        return (self.stage is not None) and (isinstance(self.stage, LoadedPipelineStage))

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

    ## Filtered Properties:
    @property
    def is_filtered(self):
        """The is_filtered property."""
        return (self.stage is not None) and (isinstance(self.stage, ComputedPipelineStage))
    
    @property
    def filtered_epochs(self):
        """The filtered_sessions property, accessed through the stage."""
        return self.stage.filtered_epochs
    
    @property
    def filtered_sessions(self):
        """The filtered_sessions property, accessed through the stage."""
        return self.stage.filtered_sessions
    
    @property
    def active_configs(self):
        """The active_configs property corresponding to the InteractivePlaceCellConfig obtained by filtering the session. Accessed through the stage."""
        return self.stage.active_configs

    ## Computed Properties:
    @property
    def is_computed(self):
        """The is_computed property. TODO: Needs validation/Testing """
        return (self.stage is not None) and (isinstance(self.stage, ComputedPipelineStage) and (self.computation_results.values[0] is not None))

    @property
    def computation_results(self):
        """The computation_results property, accessed through the stage."""
        return self.stage.computation_results
    
    ## Display Stage Properties:
    @property
    def is_displayed(self):
        """The is_displayed property. TODO: Needs validation/Testing """
        return (self.stage is not None) and (isinstance(self.stage, DisplayPipelineStage))
    
    
    
    def set_input(self, session_data_type:str='', basedir="", load_function: Callable = None, post_load_functions: List[Callable] = [],
                  auto_load=True, **kwargs):
        """ Called to set the input stage """
        if not isinstance(basedir, Path):
            print(f"basedir is not Path. Converting...")
            active_basedir = Path(basedir)
        else:
            print(f"basedir is already Path object.")
            active_basedir = basedir

        if not active_basedir.exists():
            raise FileExistsError
        
        self.session_data_type = session_data_type
        
        # Set first pipeline stage to input:
        self.stage = InputPipelineStage(
            stage_name=f"{self.pipeline_name}_input",
            basedir=active_basedir,
            load_function=load_function,
            post_load_functions=post_load_functions
        )
        if auto_load:
            self.load()

    @classmethod
    def perform_load(cls, input_stage) -> LoadedPipelineStage:
        input_stage.load()  # perform the load operation
        return LoadedPipelineStage(input_stage)  # build the loaded stage

    def load(self):
        self.stage.load()  # perform the load operation:
        self.stage = LoadedPipelineStage(self.stage)  # build the loaded stage
        self.stage.post_load()


    def filter_sessions(self, active_session_filter_configurations):
        self.stage = ComputedPipelineStage(self.stage)
        self.stage.select_filters(active_session_filter_configurations) # select filters when done
       
    ## Computation Helpers: 
    def perform_computations(self, active_computation_params: PlacefieldComputationParameters):     
        assert isinstance(self.stage, ComputedPipelineStage), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        self.stage.single_computation(active_computation_params)
        
    def register_computation(self, computation_function):
        assert isinstance(self.stage, ComputedPipelineStage), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        self.stage.register_computation(computation_function)

    def perform_registered_computations(self, previous_computation_result, debug_print=False):
        assert isinstance(self.stage, ComputedPipelineStage), "Current self.stage must already be a ComputedPipelineStage. Call self.perform_computations to reach this step."
        self.stage.perform_registered_computations()
    
    def prepare_for_display(self):
        assert isinstance(self.stage, ComputedPipelineStage), "Current self.stage must already be a ComputedPipelineStage. Call self.perform_computations to reach this step."
        self.stage = DisplayPipelineStage(self.stage)  # build the Display stage
        # Loops through all the configs and ensure that they have the neuron identity info if they need it.
        for an_active_config_name in self.active_configs.keys():
            self.active_configs[an_active_config_name] = add_neuron_identity_info_if_needed(self.computation_results[an_active_config_name], self.active_configs[an_active_config_name])
        
    def display(self, display_function, active_session_filter_configuration: str, **kwargs):
        # active_session_filter_configuration: 'maze1'
        assert isinstance(self.stage, DisplayPipelineStage), "Current self.stage must already be a DisplayPipelineStage. Call self.prepare_for_display to reach this step."
        if display_function is None:
            display_function = DefaultDisplayFunctions._display_normal
        return display_function(self.computation_results[active_session_filter_configuration], self.active_configs[active_session_filter_configuration], **kwargs)


# class NeuropyPipeline:
#     def input(**kwargs):
#         pass

#     def compute(sess):
#         pass

#     def display(computation_results):
#         pass


