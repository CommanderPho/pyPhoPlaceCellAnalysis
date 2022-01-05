#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
NeuropyPipeline.py
"""
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from pyphoplacecellanalysis.General.ComputationResults import ComputationResult
from pyphoplacecellanalysis.General.SessionSelectionAndFiltering import batch_filter_session

# NeuroPy (Diba Lab Python Repo) Loading
try:
    from neuropy import core

    importlib.reload(core)
except ImportError:
    sys.path.append(r"C:\Users\Pho\repos\NeuroPy")  # Windows
    # sys.path.append('/home/pho/repo/BapunAnalysis2021/NeuroPy') # Linux
    # sys.path.append(r'/Users/pho/repo/Python Projects/NeuroPy') # MacOS
    print(
        "neuropy module not found, adding directory to sys.path. \n >> Updated sys.path."
    )
    from neuropy import core

# Neuropy:
from neuropy.core.session.data_session_loader import DataSessionLoader
from neuropy.core.session.dataSession import DataSession
from neuropy.analyses.placefields import PlacefieldComputationParameters, perform_compute_placefields
from neuropy.core.neuron_identities import NeuronIdentity, build_units_colormap, PlotStringBrevityModeEnum


## Idea: Another set of computations will need to be done for each:
""" 
active_epoch_session: DataSession
computation_config: PlacefieldComputationParameters

"""

def _perform_single_computation(active_session, computation_config):
    # only requires that active_session has the .spikes_df and .position  properties
    # active_epoch_placefields1D, active_epoch_placefields2D = compute_placefields_masked_by_epochs(active_epoch_session, active_config, included_epochs=None, should_display_2D_plots=should_display_2D_plots) ## This is causing problems due to deepcopy of session.
    output_result = ComputationResult(active_session, computation_config, computed_data=dict())
    # active_epoch_placefields1D, active_epoch_placefields2D = perform_compute_placefields(active_session.spikes_df, active_session.position, computation_config, None, None, included_epochs=None, should_force_recompute_placefields=True)
    output_result.computed_data['pf1D'], output_result.computed_data['pf2D'] = perform_compute_placefields(active_session.spikes_df, active_session.position, computation_config, None, None, included_epochs=None, should_force_recompute_placefields=True)

    # Compare the results:

    # debug_print_ratemap(active_epoch_placefields1D.ratemap)
    # num_spikes_per_spiketrain = np.array([np.shape(a_spk_train)[0] for a_spk_train in active_epoch_placefields1D.spk_t])
    # num_spikes_per_spiketrain
    # print('placefield_neuronID_spikes: {}; ({} total spikes)'.format(num_spikes_per_spiketrain, np.sum(num_spikes_per_spiketrain)))
    # debug_print_placefield(active_epoch_placefields1D) #49 good
    # debug_print_placefield(output_result.computed_data['pf2D']) #51 good

    return output_result

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
    filter_function: Callable = None


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
    """Docstring for InputPipelineStage."""

    stage_name: str = ""


class LoadableInput:
    def _check(self):
        assert (
            self.load_function is not None
        ), "self.load_function must be a valid single-argument load function that isn't None!"
        assert callable(self.load_function), "self.load_function must be callable!"

        assert self.basedir is not None, "self.basedir must not be None!"

        assert isinstance(
            self.basedir, Path
        ), "self.basedir must be a pathlib.Path type object (or a pathlib.Path subclass)"
        if not self.basedir.exists():
            raise FileExistsError
        else:
            return True

    def load(self):
        self._check()

        self.loaded_data = dict()

        # call the internal load_function with the self.basedir.
        self.loaded_data["sess"] = self.load_function(self.basedir)

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
        return self.loaded_data["sess"]

    @sess.setter
    def sess(self, value):
        self.loaded_data["sess"] = value

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

    basedir: Path = Path("")
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
class LoadedPipelineStage(
    LoadableInput, LoadableSessionInput, BaseNeuropyPipelineStage
):
    """Docstring for InputPipelineStage."""

    loaded_data: dict = None

    def __init__(self, input_stage: InputPipelineStage):
        # super(ClassName, self).__init__()
        self.stage_name = input_stage.stage_name
        self.basedir = input_stage.basedir
        self.loaded_data = input_stage.loaded_data


class FilterablePipelineStage:
    
    def select_filters(self, active_session_filter_configurations):
        self.filtered_sessions = dict()
        self.filtered_epochs = dict()
        self.computation_results = dict()
        for a_select_config_name, a_select_config_filter_function in active_session_filter_configurations.items():
            print(f'Applying session filter named "{a_select_config_name}"...')
            self.filtered_sessions[a_select_config_name], self.filtered_epochs[a_select_config_name] = a_select_config_filter_function(self.sess)
            self.computation_results[a_select_config_name] = None
            
        
class ComputablePipelineStage:
    
    def single_computation(self, active_computation_params: PlacefieldComputationParameters):
        assert (len(self.filtered_sessions.keys()) > 0), "Must have at least one filtered session before calling single_computation(...). Call self.select_filters(...) first."
        # self.active_computation_results = dict()
        for a_select_config_name, a_filtered_session in self.filtered_sessions.items():
            print(f'Performing single_computation on filtered_session with filter named "{a_select_config_name}"...')
            self.computation_results[a_select_config_name] = _perform_single_computation(a_filtered_session, active_computation_params) # returns a computation result. Does this store the computation config used to compute it?
        
        # pf_neuron_identities, pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = _get_neuron_identities(computation_result.computed_data['pf1D'])
        # pf_neuron_identities, pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = _get_neuron_identities(self.active_computation_results[a_select_config_name].computed_data['pf2D'])

    

class ComputedPipelineStage(LoadableInput, LoadableSessionInput, FilterablePipelineStage, ComputablePipelineStage, BaseNeuropyPipelineStage):
    """Docstring for ComputedPipelineStage."""

    filtered_sessions: dict = None
    filtered_epochs: dict = None
    computation_results: dict = None
    
    def __init__(self, loaded_stage: LoadedPipelineStage):
        # super(ClassName, self).__init__()
        self.stage_name = loaded_stage.stage_name
        self.basedir = loaded_stage.basedir
        self.loaded_data = loaded_stage.loaded_data

        self.filtered_sessions = dict()
        self.filtered_epochs = dict()
        self.computation_results = dict()


# class ClassName(object):
#     """docstring for ClassName."""
#     def __init__(self, arg):
#         super(ClassName, self).__init__()
#         self.arg = arg


class NeuropyPipeline:
    @property
    def is_loaded(self):
        """The is_loaded property."""
        return (self.stage is not None) and (
            isinstance(self.stage, LoadedPipelineStage)
        )

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
    def computation_results(self):
        """The computation_results property, accessed through the stage."""
        return self.stage.computation_results
    

    def __init__(self, name="pipeline", session_data_type='kdiba', basedir=None, load_function: Callable = None):
        # super(NeuropyPipeline, self).__init__()
        self.pipeline_name = name
        self.session_data_type = None
        self.stage = None
        self.set_input(name=name, session_data_type=session_data_type, basedir=basedir, load_function=load_function)

    def set_input(self, session_data_type:str='', basedir="", load_function: Callable = None, auto_load=True, **kwargs):
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


