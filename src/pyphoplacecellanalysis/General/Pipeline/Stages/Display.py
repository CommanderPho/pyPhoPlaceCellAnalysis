from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import numpy as np

from neuropy.core.neuron_identities import NeuronIdentity, build_units_colormap, PlotStringBrevityModeEnum
from neuropy.utils.result_context import IdentifyingContext
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters # to replace simple PlacefieldComputationParameters

from pyphoplacecellanalysis.General.Pipeline.Stages.Computation import ComputedPipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import PipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
# Import Display Functions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DefaultDisplayFunctions import DefaultDisplayFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.Ratemaps import DefaultRatemapDisplayFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import DefaultDecoderDisplayFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import SpikeRastersDisplayFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.EloyAnalysis import EloyAnalysisDisplayFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.Interactive3dDisplayFunctions import Interactive3dDisplayFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.FiringStatisticsDisplayFunctions import FiringStatisticsDisplayFunctions



def get_neuron_identities(active_placefields, debug_print=False):
    """ 
    
    Usage:
        pf_neuron_identities, pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = get_neuron_identities(computation_result.computed_data['pf1D'])
        pf_neuron_identities, pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = get_neuron_identities(computation_result.computed_data['pf2D'])

    """
    good_placefield_neuronIDs = np.array(active_placefields.ratemap.neuron_ids) # in order of ascending ID
    good_placefield_tuple_neuronIDs = active_placefields.neuron_extended_ids
    if debug_print:
        np.shape(good_placefield_neuronIDs) # returns 51, why does it say that 49 are good then?
        print(f'good_placefield_neuronIDs: {good_placefield_neuronIDs}\ngood_placefield_tuple_neuronIDs: {good_placefield_tuple_neuronIDs}\n len(good_placefield_neuronIDs): {len(good_placefield_neuronIDs)}')
    
    pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = build_units_colormap(good_placefield_neuronIDs)
    
    pf_neuron_identities = [NeuronIdentity.init_from_NeuronExtendedIdentityTuple(an_extended_identity, a_color=pf_colors[:, neuron_IDX]) for (neuron_IDX, an_extended_identity) in enumerate(good_placefield_tuple_neuronIDs)]
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

def update_figure_files_output_path(computation_result, active_config, root_output_dir='output', debug_print=False):
    """ Changes the plotting_config's output path to a path with the style of  f'{root_output_dir}/2022-01-16/{active_session_name}/{active_epoch_name}'
    Called by prepare_for_display(...) to build the output file paths for saving figures to one that dynamically contains the current day date and relevent parameters.

    Requires:
        active_config.computation_config.pf_params: to incorporate the current computation parameters into the output path
        active_config.plotting_config: to be updated

    Changes:
        active_config.plotting_config
    """
    def _set_figure_save_root_day_computed_mode(plotting_config, active_session_name, active_epoch_name, root_output_dir='output', debug_print=False):
        """ Changes the plotting_config's output path to a path with the style of  f'output/2022-01-16/{active_session_name}/{active_epoch_name}' """
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
    

    if debug_print:
        print(f'_display_custom_user_function(computation_result, active_config, **kwargs):')
    active_session_name = active_config.active_session_config.session_name
    if debug_print:
        print(f'active_session_name: {active_session_name}')
    active_epoch_names = active_config.active_epochs
    if debug_print:
        print(f'active_epoch_names.name: {active_epoch_names.name}') # active_epoch_names: <NamedTimerange: {'name': 'maze1', 'start_end_times': array([  22.26      , 1739.15336412])};>
    # active_epoch_names.name: maze1
    active_config.plotting_config = _set_figure_save_root_day_computed_mode(active_config.plotting_config, active_session_name, active_epoch_names.name, root_output_dir=root_output_dir, debug_print=debug_print)
    # get the output path for this figure name:
    
    if debug_print:
        out_figure_save_root = active_config.plotting_config.get_figure_save_path('test_plot')
        print(f'for a test plot with name "test_plot", the output path would be: {out_figure_save_root}')
    
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
    

# ==================================================================================================================== #
# PIPELINE STAGE                                                                                                       #
# ==================================================================================================================== #
class DisplayPipelineStage(ComputedPipelineStage):
    """ The concrete pipeline stage for displaying the output computed in previous stages."""
    identity: PipelineStage = PipelineStage.Displayed
    
    def __init__(self, computed_stage: ComputedPipelineStage, display_output=None, render_actions=None, filtered_contexts=None):
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
        self.display_output = display_output or DynamicParameters()
        self.render_actions = render_actions or DynamicParameters()
        self.filtered_contexts = filtered_contexts or DynamicParameters() # None by default, otherwise IdentifyingContext
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
    
    def register_default_known_display_functions(self):
        """ Registers all known display functions 
        Called in:
            DisplayPipelineStage.__init__(...): to register display functions
        """
        for (a_display_class_name, a_display_class) in DisplayFunctionRegistryHolder.get_registry().items():
            for (a_display_fn_name, a_display_fn) in a_display_class.get_all_functions(use_definition_order=False):
                self.register_display_function(a_display_fn_name, a_display_fn)

    def reload_default_display_functions(self):
        """ reloads/re-registers the default display functions after adding a new one """
        self.register_default_known_display_functions()
        
        
    def register_display_function(self, registered_name, display_function):
        """ registers a new custom display function"""
        self.registered_display_function_dict[registered_name] = display_function
        
# ==================================================================================================================== #
# PIPELINE MIXIN                                                                                                       #
# ==================================================================================================================== #
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
    def display_output(self):
        """ display_output holds the displayed figures and their acompanying helpers."""
        return self.stage.display_output
    @display_output.setter
    def display_output(self, value):
        self.stage.display_output = value
    
    @property
    def filtered_contexts(self):
        """ filtered_contexts holds the corresponding contexts for each filtered config."""
        return self.stage.filtered_contexts
    @filtered_contexts.setter
    def filtered_contexts(self, value):
        self.stage.filtered_contexts = value
    
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
        
    def reload_default_display_functions(self):
        """ reloads/re-registers the default display functions after adding a new one """
        self.stage.reload_default_display_functions() 
        
    def prepare_for_display(self, root_output_dir=r'W:\data\Output', should_smooth_maze=True):
        assert (self.is_computed), "Current self.is_computed must be true. Call self.perform_computations to reach this step."
        self.stage = DisplayPipelineStage(self.stage)  # build the Display stage
        
        # Empty the dicts:
        self.filtered_contexts = DynamicParameters()
        self.display_output = DynamicParameters()
        
        active_identifying_session_ctx = self.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
        
        # Loops through all the configs and ensure that they have the neuron identity info if they need it.
        for an_active_config_name in self.active_configs.keys():
            ## Add the filter to the active context (IdentifyingContext)
            self.filtered_contexts[an_active_config_name] = active_identifying_session_ctx.adding_context('filter', filter_name=an_active_config_name) # 'bapun_RatN_Day4_2019-10-15_11-30-06_maze'
            # self.display_output[active_identifying_session_ctx][self.filtered_contexts[an_active_config_name]] = DynamicParameters()
            self.display_output[self.filtered_contexts[an_active_config_name]] = DynamicParameters() # One display_output for each context
            
            # Note that there may be different numbers of neurons included in the different configs (which include different epochs/filters) so a single one-size-fits-all approach to assigning color identities won't work here.
            if an_active_config_name in self.computation_results:
                self.active_configs[an_active_config_name] = add_neuron_identity_info_if_needed(self.computation_results[an_active_config_name], self.active_configs[an_active_config_name])
                self.active_configs[an_active_config_name] = add_custom_plotting_options_if_needed(self.active_configs[an_active_config_name], should_smooth_maze=should_smooth_maze)
                self.active_configs[an_active_config_name] = update_figure_files_output_path(self.computation_results[an_active_config_name], self.active_configs[an_active_config_name], root_output_dir=root_output_dir)

        self.reload_default_display_functions() # reload default display functions first


    # MAIN FUNCTION ______________________________________________________________________________________________________ #
    def display(self, display_function, active_session_configuration_context, **kwargs):
        """ Called to actually perform the display. Should output a figure/widget/graphic of some kind. 
        Inputs:
            display_function: either a Callable display function (e.g. DefaultDisplayFunctions._display_1d_placefield_validations) or a str containing the name of a registered display function (e.g. '_display_1d_placefield_validations')
            active_session_filter_configuration: the string that's a key into the computation results like 'maze1' or 'maze2' that specifies which result you want to display.
        """
        assert self.can_display, "Current self.stage must already be a DisplayPipelineStage. Call self.prepare_for_display to reach this step."
        if display_function is None:
            display_function = DefaultDisplayFunctions._display_normal
        
        if isinstance(display_function, (str)):
            # if the display_function is a str (name of the function) instead of a callable, try to get the actual callable
            assert (display_function in self.registered_display_function_names), f"ERROR: The display function with the name {display_function} could not be found! Is it registered?"
            display_function = self.registered_display_function_dict[display_function] # find the actual function from the name
            
        ## Old form: active_session_filter_configuration: str
        if isinstance(active_session_configuration_context, str):
            ## Old strictly name-based version (pre 2022-09-12):
            active_session_configuration_name = active_session_configuration_context
            
            ## TODO: get the appropriate IdentifyingContext from the config name
            assert active_session_configuration_name in self.filtered_contexts, f'active_session_configuration_name: {active_session_configuration_name} is NOT in the self.filtered_contexts dict: {list(self.filtered_contexts.keys())}'
            active_session_configuration_context = self.filtered_contexts[active_session_configuration_name]
            
            
        elif isinstance(active_session_configuration_context, IdentifyingContext):
            # NEW 2022-09-12-style call with an identifying context (IdentifyingContext) object
            active_session_configuration_name = None    
        else:
            print(f'WARNING: active_session_configuration_context: {active_session_configuration_context} with type: {type(active_session_configuration_context)}')
            raise NotImplementedError
        
        
        assert isinstance(active_session_configuration_context, IdentifyingContext)
        ## Now we're certain that we have an active_session_configuration_context:
        kwargs.setdefault('active_context', active_session_configuration_context) # add 'active_context' to the kwargs for the display function if possible

        # Check if the context is filtered or at the session level:
        if not hasattr(active_session_configuration_context, 'filter_name'):
            ## Global session-level context (not filtered, so not corresponding to a specific config name):
            print(f'WARNING: .display(...) function called with GLOBAL (non-filtered) context. This should be the case only for one function (`_display_context_nested_docks`) currently. ')
            ## For a global-style display function, pass ALL of the computation_results and active_configs just to preserve the argument style.
            return display_function(self.computation_results, self.active_configs, owning_pipeline=self, active_config_name=None, **kwargs)
        
        else:
            ## The expected filtered context:
            if active_session_configuration_name is not None:
                assert active_session_configuration_context.filter_name == active_session_configuration_name
    
            active_session_configuration_name = active_session_configuration_context.filter_name
        
        ## Sanity checking:
        assert (active_session_configuration_name in self.computation_results), f"self.computation_results doesn't contain a key for the provided active_session_filter_configuration ('{active_session_configuration_name}'). Did you only enable computation with enabled_filter_names in perform_computation that didn't include this key?"
        # We pop the active_config_name parameter from the kwargs, as this was an outdated workaround to optionally get the display functions this string but now it's passed directly by the call below        
        kwarg_active_config_name = kwargs.pop('active_config_name', None)
        if kwarg_active_config_name is not None:
            assert kwarg_active_config_name == active_session_configuration_name # they better be equal or else there is a conflict.

        return display_function(self.computation_results[active_session_configuration_name], self.active_configs[active_session_configuration_name], owning_pipeline=self, active_config_name=active_session_configuration_name, **kwargs)
        

    