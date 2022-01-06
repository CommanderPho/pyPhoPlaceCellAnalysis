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
from pyphoplacecellanalysis.General.Pipeline.Computation import ComputablePipelineStage
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
from neuropy.core.neuron_identities import NeuronIdentity, build_units_colormap, PlotStringBrevityModeEnum



def get_neuron_identities(active_placefields, debug_print=False):
    """ 
    
    Usage:
        pf_neuron_identities, pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = get_neuron_identities(computation_result.computed_data['pf1D'])
        pf_neuron_identities, pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = get_neuron_identities(computation_result.computed_data['pf2D'])

    """
    good_placefield_neuronIDs = np.array(active_placefields.ratemap.neuron_ids) # in order of ascending ID
    good_placefield_tuple_neuronIDs = active_placefields.neuron_extended_ids

    # good_placefields_neurons_obj = active_epoch_session.neurons.get_by_id(good_placefield_neuronIDs)
    # good_placefields_neurons_obj
    if debug_print:
        np.shape(good_placefield_neuronIDs) # returns 51, why does it say that 49 are good then?
        print(f'good_placefield_neuronIDs: {good_placefield_neuronIDs}\ngood_placefield_tuple_neuronIDs: {good_placefield_tuple_neuronIDs}\n len(good_placefield_neuronIDs): {len(good_placefield_neuronIDs)}')
    
    # ## Filter by neurons with good placefields only:
    # # throwing an error because active_epoch_session's .neurons property is None. I think the memory usage from deepcopy is actually a bug, not real use.

    # # good_placefields_flattened_spiketrains = active_epoch_session.flattened_spiketrains.get_by_id(good_placefield_neuronIDs) ## Working

    # # Could alternatively build from the whole dataframe again, but prob. not needed.
    # # filtered_spikes_df = active_epoch_session.spikes_df.query("`aclu` in @good_placefield_neuronIDs")
    # # good_placefields_spk_df = good_placefields_flattened_spiketrains.to_dataframe() # .copy()
    # # good_placefields_neurons_obj = active_epoch_session.neurons.get_by_id(good_placefield_neuronIDs)
    # # good_placefields_neurons_obj = Neurons.from_dataframe(good_placefields_spk_df, active_epoch_session.recinfo.dat_sampling_rate, time_variable_name=good_placefields_spk_df.spikes.time_variable_name) # do we really want another neuron object? Should we throw out the old one?
    # good_placefields_session = active_epoch_session
    # good_placefields_session.neurons = active_epoch_session.neurons.get_by_id(good_placefield_neuronIDs)
    # good_placefields_session.flattened_spiketrains = active_epoch_session.flattened_spiketrains.get_by_id(good_placefield_neuronIDs) ## Working

    # # good_placefields_session = active_epoch_session.get_by_id(good_placefield_neuronIDs) # Filter by good placefields only, and this fetch also ensures they're returned in the order of sorted ascending index ([ 2  3  5  7  9 12 18 21 22 23 26 27 29 34 38 45 48 53 57])
    # # good_placefields_session

    pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = build_units_colormap(good_placefield_neuronIDs)
    # active_config.plotting_config.pf_sort_ind = pf_sort_ind
    # active_config.plotting_config.pf_colors = pf_colors
    # active_config.plotting_config.active_cells_colormap = pf_colormap
    # active_config.plotting_config.active_cells_listed_colormap = ListedColormap(active_config.plotting_config.active_cells_colormap)

    pf_neuron_identities = [NeuronIdentity.init_from_NeuronExtendedIdentityTuple(an_extended_identity, a_color=pf_colors[:, neuron_IDX]) for (neuron_IDX, an_extended_identity) in enumerate(good_placefield_tuple_neuronIDs)]
    # pf_neuron_identities = [NeuronIdentity.init_from_NeuronExtendedIdentityTuple(good_placefield_tuple_neuronIDs[neuron_IDX], a_color=pf_colors[:, neuron_IDX]) for neuron_IDX in np.arange(len(good_placefield_neuronIDs))]
    # pf_neuron_identities = [NeuronIdentity.init_from_NeuronExtendedIdentityTuple(an_extended_identity) for an_extended_identity in good_placefield_tuple_neuronIDs]
    return pf_neuron_identities, pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap
    
def _temp_filter_session(sess):
    """ 
    Usage:
        active_session, active_epoch = _temp_filter_session(curr_bapun_pipeline.sess)
    """
    # curr_kdiba_pipeline.sess.epochs['maze1']
    active_epoch = sess.epochs.get_named_timerange('maze1')

    ## All Spikes:
    # active_epoch_session = sess.filtered_by_epoch(active_epoch) # old
    active_session = batch_filter_session(sess, sess.position, sess.spikes_df, active_epoch.to_Epoch())
    return active_session, active_epoch



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

        
        
            


class ComputedPipelineStage(LoadableInput, LoadableSessionInput, FilterablePipelineStage, ComputablePipelineStage, BaseNeuropyPipelineStage):
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
        
    def perform_computations(self, active_computation_params: PlacefieldComputationParameters):     
        assert isinstance(self.stage, ComputedPipelineStage), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        self.stage.single_computation(active_computation_params)
        
        
    # @classmethod
    # def perform_compute(cls, loaded_stage):
    #     pass
    #     # input_stage.load() # perform the load operation
    #     # return LoadedPipelineStage(input_stage)  # build the loaded stage



    def display(self, computation_results):
        pass


# class NeuropyPipeline:
#     def input(**kwargs):
#         pass

#     def compute(sess):
#         pass

#     def display(computation_results):
#         pass


