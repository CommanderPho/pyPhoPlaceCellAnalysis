from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
import pyvistaqt as pvqt


from pyphocorehelpers.indexing_helpers import interleave_elements
from pyphocorehelpers.print_helpers import WrappingMessagePrinter
from pyphocorehelpers.plotting.mixins.figure_param_text_box import add_figure_text_box # for _display_add_computation_param_text_box
from pyphocorehelpers.geometry_helpers import compute_data_extent, compute_data_aspect_ratio


from pyphoplacecellanalysis.General.Pipeline.Stages.Computation import ComputedPipelineStage
from pyphoplacecellanalysis.General.Configs.DynamicConfigs import PlottingConfig, InteractivePlaceCellConfig
from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import PipelineStage

from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
# Import Display Functions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DefaultDisplayFunctions import DefaultDisplayFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.Ratemaps import DefaultRatemapDisplayFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import DefaultDecoderDisplayFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import SpikeRastersDisplayFunctions


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
    
def add_neuron_identity_info_if_needed(computation_result, active_config):
    """ Attempts to add the neuron Identities and the color information to the active_config.plotting_config for use by my 3D classes and such. """
    try:
        len(active_config.plotting_config.pf_colors)
    except (AttributeError, KeyError):
        # add the attributes 
        active_config.plotting_config.pf_neuron_identities, active_config.plotting_config.pf_sort_ind, active_config.plotting_config.pf_colors, active_config.plotting_config.pf_colormap, active_config.plotting_config.pf_listed_colormap = get_neuron_identities(computation_result.computed_data['pf2D'])
    except Exception as e:
        # other exception
        print(f'Unexpected exception e: {e}')
        raise
    return active_config

def add_custom_plotting_options_if_needed(active_config, should_smooth_maze):
    active_config.plotting_config.use_smoothed_maze_rendering = should_smooth_maze
    return active_config

def update_figure_files_output_Format(computation_result, active_config, root_output_dir='output', debug_print=False):
    def _set_figure_save_root_day_computed_mode(plotting_config, active_session_name, active_epoch_name, root_output_dir='output', debug_print=False):
        """ Outputs to a path with the style of  """
        out_figure_save_original_root = plotting_config.get_figure_save_path('test') # 2022-01-16/
        if debug_print:
            print(f'out_figure_save_original_root: {out_figure_save_original_root}')
        # Update output figure root:
        out_day_date_folder_name = datetime.today().strftime('%Y-%m-%d') # 2022-01-16
        new_out_day_day_parent_dir = Path(root_output_dir, out_day_date_folder_name, active_session_name, active_epoch_name)
        out_figure_save_root = plotting_config.change_active_out_parent_dir(new_out_day_day_parent_dir)
        # out_figure_save_root = active_config.plotting_config.get_figure_save_path(out_day_date_folder_name, active_session_name, active_epoch_names.name) # 2022-01-16/
        if debug_print:
            print(f'out_figure_save_root: {out_figure_save_root}') # out_figure_save_root: output\2006-6-07_11-26-53\maze1\2022-01-18\2006-6-07_11-26-53\maze1
        return plotting_config
    
    
    # def _test_get_full_figure_path_components(output_root, out_day_date_folder_name, active_session_name, active_epoch_name, active_computation_config_str, active_plot_type_name, active_variant_name):
    #     return [output_root, out_day_date_folder_name, active_session_name, active_epoch_name, active_computation_config_str, active_plot_type_name, active_variant_name]
    
    
    # _test_get_full_figure_path_components('output', datetime.today().strftime('%Y-%m-%d'), active_config.active_session_config.session_name, active_config.active_epochs.name, active_config.computation_config.str_for_filename(False),
    #                                       active_plot_type_name, active_variant_name)
    
    
    
    if debug_print:
        print(f'_display_custom_user_function(computation_result, active_config, **kwargs):')
    # print(f'active_config.keys(): {list(active_config.keys())}') # active_config.keys(): ['active_session_config', 'active_epochs', 'video_output_config', 'plotting_config', 'computation_config', 'filter_config']
    # print(f'active_config.plotting_config: {active_config.plotting_config}')
    # print(f'active_config.active_session_config: {active_config.active_session_config}')
    active_session_name = active_config.active_session_config.session_name
    if debug_print:
        print(f'active_session_name: {active_session_name}')
    active_epoch_names = active_config.active_epochs
    if debug_print:
        print(f'active_epoch_names.name: {active_epoch_names.name}') # active_epoch_names: <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
    # active_epoch_names.name: maze1
    active_config.plotting_config = _set_figure_save_root_day_computed_mode(active_config.plotting_config, active_session_name, active_epoch_names.name, root_output_dir=root_output_dir, debug_print=debug_print)
    # get the output path for this figure name:
    out_figure_save_root = active_config.plotting_config.get_figure_save_path('test_plot')
    if debug_print:
        print(f'out_figure_save_root: {out_figure_save_root}')
    
    # Now convert the computation parameters for filename display:
    if debug_print:
        print(f'active_config.computation_config: {active_config.computation_config}')
    curr_computation_config_output_dir_name = active_config.computation_config.pf_params.str_for_filename(False)
    if debug_print:
        print(f'curr_computation_config_output_dir_name: {curr_computation_config_output_dir_name}')
    out_figure_save_current_computation_dir = active_config.plotting_config.get_figure_save_path(curr_computation_config_output_dir_name)
    if debug_print:
        print(f'out_figure_save_current_computation_dir: {out_figure_save_current_computation_dir}')
    # change finally to the computation config determined subdir:
    final_out_figure_save_root = active_config.plotting_config.change_active_out_parent_dir(out_figure_save_current_computation_dir)
    if debug_print:
        print(f'final_out_figure_save_root: {final_out_figure_save_root}')
    return active_config
    

class DefaultRegisteredDisplayFunctions:
    """ Simply enables specifying the default computation functions that will be defined in this file and automatically registered. 
    
    Known Uses:
        DisplayPipelineStage conforms to DefaultRegisteredDisplayFunctions to allow it to call self.register_default_known_display_functions() during its .__init__(...)
    """
    
    def register_default_known_display_functions(self):
        """ Registers all known display functions 
        
        Called in:
            DisplayPipelineStage.__init__(...): to register display functions
        """
        for (a_display_class_name, a_display_class) in DisplayFunctionRegistryHolder.get_registry().items():
            for (a_display_fn_name, a_display_fn) in a_display_class.get_all_functions(use_definition_order=False):
                self.register_display_function(a_display_fn_name, a_display_fn)
        
        # # Register the Ratemap/Placemap display functions: 
        # for (a_display_fn_name, a_display_fn) in DefaultDisplayFunctions.get_all_functions(use_definition_order=False):
        #     self.register_display_function(a_display_fn_name, a_display_fn)
            
        # # Register the Ratemap/Placemap display functions: 
        # for (a_display_fn_name, a_display_fn) in DefaultRatemapDisplayFunctions.get_all_functions(use_definition_order=False):
        #     self.register_display_function(a_display_fn_name, a_display_fn)
            
        # # Register the Bayesian decoder display functions: 
        # for (a_display_fn_name, a_display_fn) in DefaultDecoderDisplayFunctions.get_all_functions(use_definition_order=False):
        #     self.register_display_function(a_display_fn_name, a_display_fn)
            
        # # Register the spike rasters display functions: 
        # for (a_display_fn_name, a_display_fn) in SpikeRastersDisplayFunctions.get_all_functions(use_definition_order=False):
        #     self.register_display_function(a_display_fn_name, a_display_fn)
            
  
  

class PipelineWithDisplayPipelineStageMixin:
    """ To be added to the pipeline to enable conveninece access ot its pipeline stage post Display stage. """
    ## Display Stage Properties:
    @property
    def is_displayed(self):
        """The is_displayed property. TODO: Needs validation/Testing """
        return (self.stage is not None) and (isinstance(self.stage, DisplayPipelineStage))
    
    @property
    def can_display(self):
        """Whether the display functions can be performed."""
        return (self.last_completed_stage >= PipelineStage.Displayed)
    
    @property
    def registered_display_functions(self):
        """The registered_display_functions property."""
        return self.stage.registered_display_functions
        
    @property
    def registered_display_function_names(self):
        """The registered_display_function_names property."""
        return self.stage.registered_display_function_names
    
    @property
    def registered_display_function_dict(self):
        """The registered_display_function_dict property can be used to get the corresponding function from the string name."""
        return self.stage.registered_display_function_dict
    
    @property
    def registered_display_function_docs_dict(self):
        """Returns the doc strings for each registered display function. This is taken from their docstring at the start of the function defn, and provides an overview into what the function will do."""
        return {a_fn_name:a_fn.__doc__ for a_fn_name, a_fn in self.registered_display_function_dict.items()}
    
    def register_display_function(self, registered_name, display_function):
        # assert (self.can_display), "Current self.stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
        self.stage.register_display_function(registered_name, display_function)
        
    def prepare_for_display(self, root_output_dir=r'R:\data\Output', should_smooth_maze=True):
        assert (self.is_computed), "Current self.is_computed must be true. Call self.perform_computations to reach this step."
        self.stage = DisplayPipelineStage(self.stage)  # build the Display stage
        # Loops through all the configs and ensure that they have the neuron identity info if they need it.
        for an_active_config_name in self.active_configs.keys():
            # Note that there may be different numbers of neurons included in the different configs (which include different epochs/filters) so a single one-size-fits-all approach to assigning color identities won't work here.
            if an_active_config_name in self.computation_results:
                self.active_configs[an_active_config_name] = add_neuron_identity_info_if_needed(self.computation_results[an_active_config_name], self.active_configs[an_active_config_name])
                
                self.active_configs[an_active_config_name] = add_custom_plotting_options_if_needed(self.active_configs[an_active_config_name], should_smooth_maze=should_smooth_maze)
                self.active_configs[an_active_config_name] = update_figure_files_output_Format(self.computation_results[an_active_config_name], self.active_configs[an_active_config_name], root_output_dir=root_output_dir)

                    
    def display(self, display_function, active_session_filter_configuration: str, **kwargs):
        """ Called to actually perform the display. Should output a figure/widget/graphic of some kind. 
        Inputs:
            display_function: either a Callable display function (e.g. DefaultDisplayFunctions._display_1d_placefield_validations) or a str containing the name of a registered display function (e.g. '_display_1d_placefield_validations')
            active_session_filter_configuration: the string that's a key into the computation results like 'maze1' or 'maze2'.
        """
        assert self.can_display, "Current self.stage must already be a DisplayPipelineStage. Call self.prepare_for_display to reach this step."
        if display_function is None:
            display_function = DefaultDisplayFunctions._display_normal
        
        if isinstance(display_function, (str)):
            # if the display_function is a str (name of the function) instead of a callable, try to get the actual callable
            assert (display_function in self.registered_display_function_names), f"ERROR: The display function with the name {display_function} could not be found! Is it registered?"
            display_function = self.registered_display_function_dict[display_function] # find the actual function from the name
            
        assert (active_session_filter_configuration in self.computation_results), f"self.computation_results doesn't contain a key for the provided active_session_filter_configuration ('{active_session_filter_configuration}'). Did you only enable computation with enabled_filter_names in perform_computation that didn't include this key?"
        return display_function(self.computation_results[active_session_filter_configuration], self.active_configs[active_session_filter_configuration], **kwargs)


    

class DisplayPipelineStage(DefaultRegisteredDisplayFunctions, ComputedPipelineStage):
    """ The concrete pipeline stage for displaying the output computed in previous stages."""
    identity: PipelineStage = PipelineStage.Displayed
    
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
        self.registered_computation_function_dict = computed_stage.registered_computation_function_dict

        # Initialize custom fields:
        self.render_actions = render_actions    
        self.registered_display_function_dict = OrderedDict()
        self.register_default_known_display_functions() # registers the default display functions
        
    @property
    def registered_display_functions(self):
        """The registered_display_functions property."""
        return list(self.registered_display_function_dict.values()) 
        
    @property
    def registered_display_function_names(self):
        """The registered_display_function_names property."""
        return list(self.registered_display_function_dict.keys()) 
    
    
    def register_display_function(self, registered_name, display_function):
        """ registers a new custom display function"""
        self.registered_display_function_dict[registered_name] = display_function
        