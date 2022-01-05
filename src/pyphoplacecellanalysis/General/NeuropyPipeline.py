#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
NeuropyPipeline.py
"""
import sys
import importlib
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

from pandas.core import base

# NeuroPy (Diba Lab Python Repo) Loading
try:
    from neuropy import core
    importlib.reload(core)
except ImportError:
    sys.path.append(r'C:\Users\Pho\repos\NeuroPy') # Windows
    # sys.path.append('/home/pho/repo/BapunAnalysis2021/NeuroPy') # Linux
    # sys.path.append(r'/Users/pho/repo/Python Projects/NeuroPy') # MacOS
    print('neuropy module not found, adding directory to sys.path. \n >> Updated sys.path.')
    from neuropy import core
    
from neuropy.core.session.data_session_loader import DataSessionLoader
from neuropy.core.session.dataSession import DataSession
    

@dataclass
class BaseNeuropyPipelineStage(object):
    """Docstring for InputPipelineStage."""
    stage_name: str = ''

class LoadableInput:
    
    def _check(self):
        assert (self.load_function is not None), "self.load_function must be a valid single-argument load function that isn't None!"
        assert callable(self.load_function), "self.load_function must be callable!"         

        assert (self.basedir is not None), "self.basedir must not be None!"

        assert isinstance(self.basedir, Path), "self.basedir must be a pathlib.Path type object (or a pathlib.Path subclass)"
        if not self.basedir.exists():
            raise FileExistsError
        else:
            return True
    
    
    def load(self):
        self._check()
    
        self.loaded_data = dict() 
        
        # call the internal load_function with the self.basedir.
        self.loaded_data['sess'] = self.load_function(self.basedir)
        
        # self.loaded_data['sess'] = DataSessionLoader.bapun_data_session(self.basedir)
        # self.sess = DataSessionLoader.bapun_data_session(self.basedir)
        # self.sess
        # active_sess_config = sess.config
        # session_name = sess.name
        pass

class LoadableSessionInput:
    @property
    def sess(self):
        """The sess property."""
        return self.loaded_data['sess']
    @sess.setter
    def sess(self, value):
        self.loaded_data['sess'] = value
  
    @property
    def active_sess_config(self):
        """The active_sess_config property."""
        return self.sess.config
    @active_sess_config.setter
    def active_sess_config(self, value):
        self.sess.config = value
  
    @property
    def session_name(self):
        """The session_name property."""
        return self.sess.name
    @session_name.setter
    def session_name(self, value):
        self.sess.name = value



@dataclass
class InputPipelineStage(LoadableInput, BaseNeuropyPipelineStage):
    """Docstring for InputPipelineStage."""
    basedir: Path = Path('')
    load_function: Callable = None 
     
    # @property
    # def basedir_path(self):
    #     """The basedir_path property."""
    #     return Path(self.basedir)
    
    # def __init__(self, basedir='', **kwargs):
    #     super(InputPipelineStage, self).__init__(**kwargs)
    #     # BaseNeuropyPipelineStage(**kwargs)
    #     if not isinstance(basedir, Path):
    #         print(f'basedir is not Path. Converting...')
    #         self.basedir = Path(basedir)
    #     else:
    #         print(f'basedir is already Path object.')
    #         self.basedir = basedir
        
    #     if not self.basedir.exists():
    #         raise FileExistsError
        


# @dataclass
class LoadedPipelineStage(LoadableInput, LoadableSessionInput, BaseNeuropyPipelineStage):
    """Docstring for InputPipelineStage."""
    loaded_data: dict = None
    
    def __init__(self, input_stage: InputPipelineStage):
        # super(ClassName, self).__init__()
        self.stage_name = input_stage.stage_name
        self.basedir = input_stage.basedir
        self.loaded_data = input_stage.loaded_data

  
# class ClassName(object):
#     """docstring for ClassName."""
#     def __init__(self, arg):
#         super(ClassName, self).__init__()
#         self.arg = arg
        


    
class NeuropyPipeline:
        
    @property
    def is_loaded(self):
        """The is_loaded property."""
        return (self.stage is not None) and (isinstance(self.stage, LoadedPipelineStage))

    
    def __init__(self, name='pipeline', basedir=None, load_function: Callable=None):
        # super(NeuropyPipeline, self).__init__()
        self.pipeline_name = name
        self.stage = None
        self.set_input(name=name, basedir=basedir, load_function=load_function)


    def set_input(self, basedir='', load_function: Callable=None, auto_load=True, **kwargs):
        if not isinstance(basedir, Path):
            print(f'basedir is not Path. Converting...')
            active_basedir = Path(basedir)
        else:
            print(f'basedir is already Path object.')
            active_basedir = basedir
        
        if not active_basedir.exists():
            raise FileExistsError
        # Set first pipeline stage to input:
        self.stage = InputPipelineStage(stage_name=f'{self.pipeline_name}_input', basedir=active_basedir, load_function=load_function)
        if auto_load:
            self.load()
        
    def load(self):
        self.stage.load() # perform the load operation:
        self.stage = LoadedPipelineStage(self.stage) # build the loaded stage


    def compute(self, sess):
        pass
    
    def display(self, computation_results):
        pass
    
    

# class NeuropyPipeline:
#     def input(**kwargs):
#         pass
    
#     def compute(sess):
#         pass
    
#     def display(computation_results):
#         pass