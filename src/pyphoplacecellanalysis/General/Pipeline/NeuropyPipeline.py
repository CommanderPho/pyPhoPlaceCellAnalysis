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
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import PipelineWithInputStage, PipelineWithLoadableStage

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
       
    


