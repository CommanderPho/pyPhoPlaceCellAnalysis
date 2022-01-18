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
from General.Pipeline.Stages.Computation import PipelineWithComputedPipelineStageMixin
from General.Pipeline.Stages.Display import PipelineWithDisplayPipelineStageMixin
from General.Pipeline.Stages.Filtering import PipelineWithFilteredPipelineStageMixin
from General.Pipeline.Stages.Loading import PipelineWithInputStage, PipelineWithLoadableStage

from pyphoplacecellanalysis.General.SessionSelectionAndFiltering import batch_filter_session

from pyphoplacecellanalysis.General.KnownDataSessionTypeProperties import KnownDataSessionTypeProperties


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


# known_data_session_type_dict = {'kdiba':KnownDataSessionTypeProperties(load_function=(lambda a_base_dir: DataSessionLoader.kdiba_old_format_session(a_base_dir)),
#                                basedir=Path(r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53')),
#                 'bapun':KnownDataSessionTypeProperties(load_function=(lambda a_base_dir: DataSessionLoader.bapun_data_session(a_base_dir)),
#                                basedir=Path('R:\data\Bapun\Day5TwoNovel'))
#                }


# load_fn_dict = {'kdiba':(lambda a_base_dir: DataSessionLoader.kdiba_old_format_session(a_base_dir)),
#                 'bapun':(lambda a_base_dir: DataSessionLoader.bapun_data_session(a_base_dir))
#                }




class NeuropyPipeline(PipelineWithInputStage, PipelineWithLoadableStage, PipelineWithFilteredPipelineStageMixin, PipelineWithComputedPipelineStageMixin, PipelineWithDisplayPipelineStageMixin):
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

    

 
    
    
    


