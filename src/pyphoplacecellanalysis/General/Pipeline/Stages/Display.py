from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, List, Dict, Union
import numpy as np
from attrs import define, field, Factory

from neuropy.core.neuron_identities import NeuronIdentity, build_units_colormap
from neuropy.utils.result_context import IdentifyingContext

from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters # to replace simple PlacefieldComputationParameters
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.programming_helpers import SourceCodeParsing

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
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.MultiContextComparingDisplayFunctions import MultiContextComparingDisplayFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import LongShortTrackComparingDisplayFunctions

from pyphoplacecellanalysis.General.Mixins.ExportHelpers import FileOutputManager, FigureOutputLocation, ContextToPathMode	


def has_good_str_value(a_str_val) -> bool:
    return ((a_str_val is not None) and (len(a_str_val) > 0))

@define(slots=False)
class DisplayFunctionItem:
    """ for helping to render a UI display function tree.

    display_function_items = {a_fn_name:DisplayFunctionItem.init_from_fn_object(a_fn) for a_fn_name, a_fn in curr_active_pipeline.registered_display_function_dict.items()}
    display_function_items


    """
    name: str = field()
    fn_callable: Callable = field()

    is_global: bool = field()
    short_name: str = field()
    docs: str = field()
    icon_path: Optional[str] = field()
    vscode_jump_link: Optional[str] = field()


    @classmethod
    def init_from_fn_object(cls, a_fn, icon_path=None):
        _obj = cls(name=a_fn.__name__, fn_callable=a_fn, is_global=getattr(a_fn,'is_global', False), short_name=(getattr(a_fn,'short_name', a_fn.__name__) or a_fn.__name__),
            docs=a_fn.__doc__, icon_path=icon_path, vscode_jump_link=SourceCodeParsing.build_vscode_jump_link(a_fcn_handle=a_fn))
        
        ## try to get the jump link:
        # vscode_jump_link: str = SourceCodeParsing.build_vscode_jump_link(a_fcn_handle=a_fn)
        # _obj.vscode_jump_link = vscode_jump_link
        
        return _obj

    @property
    def best_display_name(self) -> str:
        """ returns the best name for display """
        if has_good_str_value(self.short_name):
            return self.short_name
        else:
            return self.name
        

    @property
    def longform_description(self) -> str:
        """The longform_description property."""
        out_str_arr = []

        if has_good_str_value(self.short_name):
            out_str_arr.append(f"short_name: {self.short_name}") # short name first, then
            out_str_arr.append(f"name: {self.name}") # full name
        else:
            out_str_arr.append(f"name: {self.name}") # just name

        if has_good_str_value(self.docs):
            out_str_arr.append(f"docs: {self.docs}")
            
        if has_good_str_value(self.vscode_jump_link):
            out_str_arr.append(f"link: {self.vscode_jump_link}")
            
        out_str = '\n'.join(out_str_arr)
        return out_str
    
    @property
    def longform_description_formatted_html(self) -> str:
        """HTML-formatted (with bold labels) longform text for use in QTextBrowser via .setHtml(...) 
        
        # <b style='color:red;'>bold and red</b>

        """
        out_str_arr = []

        if has_good_str_value(self.short_name):
            out_str_arr.append(f"<b style='color:white;'>short_name</b>: {self.short_name}") # short name first, then
            out_str_arr.append(f"<b style='color:white;'>name</b>: {self.name}") # full name
        else:
            out_str_arr.append(f"<b style='color:white;'>name</b>: {self.name}") # just name

        if has_good_str_value(self.docs):
            out_str_arr.append(f"<b style='color:white;'>docs</b>: {self.docs}")
            
        if has_good_str_value(self.vscode_jump_link):
            # Create the HTML-formatted link
            # html_link: str = f'<a href="{self.vscode_jump_link}">Open display fcn in VSCode</a>'
            html_link: str = f'<a href="{self.vscode_jump_link}">{self.vscode_jump_link}</a>'
            out_str_arr.append(f"<b style='color:white;'>link</b>: {html_link}")
            
        out_str = '<br>'.join(out_str_arr) # linebreaks with HTML's <br>
        return out_str
    



    


class Plot:
    """a member dot accessor for display functions.

    2022-12-13

    Can call like:
        `curr_active_pipeline.plot._display_1d_placefields`


    ## Set in `reload_default_display_functions()` 
        self._plot_object = None
        self._plot_object = Plot(self)
    """
    def __init__(self, curr_active_pipeline):
        super(Plot, self).__init__()
        self._pipeline_reference = curr_active_pipeline

    @property
    def display_function_items(self) -> Dict[str,DisplayFunctionItem]:
        return {a_fn_name:DisplayFunctionItem.init_from_fn_object(a_fn) for a_fn_name, a_fn in self._pipeline_reference.registered_display_function_dict.items()}


    def __dir__(self):
        return self._pipeline_reference.registered_display_function_names # ['area', 'perimeter', 'location']
    
    def __getattr__(self, k):
        if '__getstate__' in k: # a trick to make spyder happy when inspecting dotdict
            def _dummy():
                pass
            return _dummy
        # Check if arguments are passed
        def display_wrapper(*args, **kwargs):
            if len(args) == 0:
                # if no args passed, get global context, otherwise assume first arg is context:
                active_session_configuration_context = kwargs.pop('active_session_configuration_context', list(self._pipeline_reference.filtered_contexts.values())[-1])
            else:
                # otherwise assume first arg is context:
                active_session_configuration_context = args[0]
                args = args[1:]

            if isinstance(active_session_configuration_context, str):
                    # if first arg is context, remove it from args:
                    active_session_configuration_context = self._pipeline_reference.filtered_contexts[active_session_configuration_context] # if the passed argument is a string (like 'maze1'), find it in the filtered contexts dict
            return self._pipeline_reference.display(display_function=k, active_session_configuration_context=active_session_configuration_context, *args, **kwargs)
        return display_wrapper 
        # Return display_wrapper as a property, allowing use without parentheses if desired
        # return property(display_wrapper)





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
    
    # pf_neuron_identities = [NeuronIdentity.init_from_NeuronExtendedIdentityTuple(an_extended_identity, a_color=pf_colors[:, neuron_IDX]) for (neuron_IDX, an_extended_identity) in enumerate(good_placefield_tuple_neuronIDs)]
    pf_neuron_identities = [NeuronIdentity.init_from_NeuronExtendedIdentity(an_extended_identity, a_color=pf_colors[:, neuron_IDX]) for (neuron_IDX, an_extended_identity) in enumerate(good_placefield_tuple_neuronIDs)]
    return pf_neuron_identities, pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap
    
def add_neuron_identity_info_if_needed(computation_result, active_config):
    """ Attempts to add the neuron Identities and the color information to the active_config.plotting_config for use by my 3D classes and such. """
    try:
        len(active_config.plotting_config.pf_colors)
    except (AttributeError, KeyError, TypeError):
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
        out_figure_save_original_root = plotting_config.get_figure_save_path('test', enable_creating_directory=False) # 2022-01-16/
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
        out_figure_save_root = active_config.plotting_config.get_figure_save_path('test_plot', enable_creating_directory=False)
        print(f'for a test plot with name "test_plot", the output path would be: {out_figure_save_root}')
    
    # Now convert the computation parameters for filename display:
    if debug_print:
        print(f'active_config.computation_config: {active_config.computation_config}')
    curr_computation_config_output_dir_name = active_config.computation_config.pf_params.str_for_filename(False)
    if debug_print:
        print(f'curr_computation_config_output_dir_name: {curr_computation_config_output_dir_name}')
    out_figure_save_current_computation_dir = active_config.plotting_config.get_figure_save_path(curr_computation_config_output_dir_name, enable_creating_directory=False)
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
@define(slots=False, repr=False)
class DisplayPipelineStage(ComputedPipelineStage):
    """ The concrete pipeline stage for displaying the output computed in previous stages."""
    @classmethod
    def get_stage_identity(cls) -> PipelineStage:
        return PipelineStage.Displayed

    identity: PipelineStage = field(default=PipelineStage.Displayed)

    display_output: Optional[DynamicParameters] = field(default=None)
    render_actions: Optional[DynamicParameters] = field(default=None)

    registered_display_function_dict: OrderedDict = field(default=Factory(OrderedDict))


    @classmethod
    def init_from_previous_stage(cls, computed_stage: ComputedPipelineStage, display_output=None, render_actions=None, override_filtered_contexts=None):
        _obj = cls()
        _obj.stage_name = computed_stage.stage_name
        _obj.basedir = computed_stage.basedir
        _obj.loaded_data = computed_stage.loaded_data
        _obj.filtered_sessions = computed_stage.filtered_sessions
        _obj.filtered_epochs = computed_stage.filtered_epochs
        _obj.filtered_contexts = override_filtered_contexts or computed_stage.filtered_contexts
        _obj.active_configs = computed_stage.active_configs # active_config corresponding to each filtered session/epoch
        _obj.computation_results = computed_stage.computation_results
        _obj.global_computation_results = computed_stage.global_computation_results
        _obj.registered_computation_function_dict = computed_stage.registered_computation_function_dict
        _obj.registered_global_computation_function_dict = computed_stage.registered_global_computation_function_dict

        # Initialize custom fields:
        _obj.display_output = display_output or DynamicParameters()
        _obj.render_actions = render_actions or DynamicParameters()
        # self.filtered_contexts = override_filtered_contexts or DynamicParameters() # None by default, otherwise IdentifyingContext
        _obj.registered_display_function_dict = OrderedDict()
        _obj.register_default_known_display_functions() # registers the default display functions

        return _obj

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
        display_function.is_global = getattr(display_function, 'is_global', False) # sets the 'is_global' property on the function with its current value if it has one, otherwise it assume that it's not global and sets False
        self.registered_display_function_dict[registered_name] = display_function
        

    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        if 'registered_load_function_dict' in state:
            del state['registered_load_function_dict']
        del state['registered_computation_function_dict']
        del state['registered_global_computation_function_dict']
        del state['display_output']
        del state['render_actions']
        del state['registered_display_function_dict']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        if 'identity' not in state:
            print(f'unpickling from old NeuropyPipelineStage')
            state['identity'] = None
            state['identity'] = type(self).get_stage_identity()

        self.__dict__.update(state)
        # Call the superclass __init__() (from https://stackoverflow.com/a/48325758)
        # super(LoadedPipelineStage, self).__init__() # from 

        self.registered_load_function_dict = {}
        self.register_default_known_load_functions() # registers the default load functions
        
        self.registered_computation_function_dict = OrderedDict()
        self.registered_global_computation_function_dict = OrderedDict()
        self.reload_default_computation_functions() # registers the default

        # Initialize custom fields:
        self.display_output = DynamicParameters()
        self.render_actions = DynamicParameters()
        self.registered_display_function_dict = OrderedDict()
        self.register_default_known_display_functions() # registers the default display functions



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
    def display_output_history_list(self):
        """The list of contexts in the order they were added to the self.display_output (as the keys)."""
        return list(self.display_output.keys())

    @property
    def display_output_last_added_context(self):
        """The last context added to the self.display_output (as the key)."""
        return self.display_output_history_list[-1]

    @property
    def last_added_display_output(self):
        """The last_added_display_output value."""
        last_added_display_output = self.display_output[self.display_output_last_added_context]
        return last_added_display_output

    @property
    def plot(self) -> Plot:
        """An interactive accessor object for display functions."""
        return self._plot_object
    

    ## *_functions
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
        
        try:
            self.stage.reload_default_display_functions()
        except AttributeError as e:
            print(f'ERROR: attribute error encountered, sazved pipeline before advancing to the display stage. Attemping to fix.')
            # self.prepare_for_display()
            raise NotImplementedError(f'always advance the pipeline to the display stage before pickling!')
        
        except Exception as e:
            raise e


        # rebuilds the convenience plot object:
        self._plot_object = None
        self._plot_object = Plot(self)



    def clear_display_outputs(self):
        """ removes any display outputs 
        # display_output_history_list is derived from self.display_output, so we only need to clear one.
        # seems that .clear() doesn't work for DynamicParameters for some reason. Doesn't seem to change anything.
        """
        ## Clear any hanging display outputs:
        # do I need to close them before I just remove them?
        # self.display_output.clear()
        self.display_output = DynamicParameters() # drop all
        # Loops through all the configs and ensure that they have the neuron identity info if they need it.
        for an_active_config_name in self.active_configs.keys():
            # ## Add the filter to the active context (IdentifyingContext)
            self.display_output[self.filtered_contexts[an_active_config_name]] = DynamicParameters() # One display_output for each context
        
        # for a_display_output_key in self.display_output_history_list:
        #     # a_display_output.close()
        #     del self.display_output[a_display_output_key]
        # self.display_output_history_list.clear()
        # assert len(self.display_output_history_list) == 0 # should be empty now


    def prepare_for_display(self, root_output_dir=r'W:\data\Output', should_smooth_maze=True):
        assert (self.is_computed), "Current self.is_computed must be true. Call self.perform_computations to reach this step."
        # self.stage = DisplayPipelineStage(self.stage)  # build the Display stage
        self.stage = DisplayPipelineStage.init_from_previous_stage(self.stage)  # build the Display stage
        
        # Empty the dicts:
        # self.filtered_contexts = DynamicParameters()
        self.display_output = DynamicParameters()
        
        # active_identifying_session_ctx = self.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
        
        # Loops through all the configs and ensure that they have the neuron identity info if they need it.
        for an_active_config_name in self.active_configs.keys():
            # ## Add the filter to the active context (IdentifyingContext)
            # self.filtered_contexts[an_active_config_name] = active_identifying_session_ctx.adding_context('filter', filter_name=an_active_config_name) # 'bapun_RatN_Day4_2019-10-15_11-30-06_maze'
            # self.display_output[active_identifying_session_ctx][self.filtered_contexts[an_active_config_name]] = DynamicParameters()
            self.display_output[self.filtered_contexts[an_active_config_name]] = DynamicParameters() # One display_output for each context
            
            # Note that there may be different numbers of neurons included in the different configs (which include different epochs/filters) so a single one-size-fits-all approach to assigning color identities won't work here.
            if an_active_config_name in self.computation_results:
                self.active_configs[an_active_config_name] = add_neuron_identity_info_if_needed(self.computation_results[an_active_config_name], self.active_configs[an_active_config_name])
                self.active_configs[an_active_config_name] = add_custom_plotting_options_if_needed(self.active_configs[an_active_config_name], should_smooth_maze=should_smooth_maze)
                self.active_configs[an_active_config_name] = update_figure_files_output_path(self.computation_results[an_active_config_name], self.active_configs[an_active_config_name], root_output_dir=root_output_dir)
        
        self.reload_default_display_functions() # reload default display functions first


    # MAIN FUNCTION ______________________________________________________________________________________________________ #
    def display(self, display_function: Optional[Union[str, Callable]], active_session_configuration_context: Optional[Union[str, IdentifyingContext]]=None, **kwargs):
        """ Called to actually perform the display. Should output a figure/widget/graphic of some kind. 
        Inputs:
            display_function: either a Callable display function (e.g. DefaultDisplayFunctions._display_1d_placefield_validations) or a str containing the name of a registered display function (e.g. '_display_1d_placefield_validations')
            active_session_filter_configuration: the string that's a key into the computation results like 'maze1' or 'maze2' that specifies which result you want to display.

        # Display Context:
        The final display context needs to include all aspects that make it unique, e.g. the contexts being plotted, the results being used to generate the context, the computation fcn, etc.
        Currently I think the display_outputs system uses the a simple display context (consisting of only the display function name) as the key, meaning it needs to be refined so that multiple version of the same figure can be produced.

        """
        from neuropy.utils.result_context import IdentifyingContext
        
        assert self.can_display, "Current self.stage must already be a DisplayPipelineStage. Call self.prepare_for_display to reach this step."
        debug_print = kwargs.get('debug_print', False)
        
        if display_function is None:
            # Default display function is `._display_normal`
            display_function = DefaultDisplayFunctions._display_normal
        
        if isinstance(display_function, (str)):
            # if the display_function is a str (name of the function) instead of a callable, try to get the actual callable
            assert (display_function in self.registered_display_function_names), f"ERROR: The display function with the name {display_function} could not be found! Is it registered?"
            display_function = self.registered_display_function_dict[display_function] # find the actual function from the name
        

        # Determine whether the `active_session_configuration_context` passed was really a context or str which should be used as `active_session_configuration_name` (and the real context must be extracted from `self.filtered_contexts`):
        active_session_configuration_name: Optional[str] = None
        
        ## Old form: active_session_filter_configuration: str compared to updated 2022-09-12-style call with an identifying context (IdentifyingContext) object
        # After this we will have both: `active_session_configuration_name`, `active_session_configuration_name` properly set
        if active_session_configuration_context is None:
            # No context specified is only allowed for global functions:
            assert getattr(display_function, 'is_global', False), f"display_function must be global if `active_session_configuration_context` is not specified but it is not! {display_function}"
            assert not hasattr(active_session_configuration_context, 'filter_name'), f"global functions should NOT have filter_name specified in their contexts: \n\tdisplay_function:{display_function}\n\tactive_session_configuration_context: {active_session_configuration_context}"
            active_session_configuration_context = self.sess.get_context() # get the appropriate context for global display functions
            active_session_configuration_name = None # config name is None for a session-level context. 
            # Now have both `active_session_configuration_name`, `active_session_configuration_name`
        elif isinstance(active_session_configuration_context, str):
            ## Old strictly name-based version (pre 2022-09-12). Extract the actual context from self.filtered_contexts:
            active_session_configuration_name = active_session_configuration_context # `active_session_configuration_context` was actually the name (`active_session_configuration_name`)
            # Get the context:
            assert active_session_configuration_name in self.filtered_contexts, f'active_session_configuration_name: {active_session_configuration_name} is NOT in the self.filtered_contexts dict: {list(self.filtered_contexts.keys())}'
            active_session_configuration_context = self.filtered_contexts[active_session_configuration_name]
            # Now have both `active_session_configuration_name`, `active_session_configuration_name`

        elif isinstance(active_session_configuration_context, IdentifyingContext):
            # Passed a context directly. Need to extract the `active_session_configuration_name`

            # Check if the context is filtered or at the session level:
            if not hasattr(active_session_configuration_context, 'filter_name'):
                ## Global session-level context (not filtered, so not corresponding to a specific config name):
                active_session_configuration_name = None
            else:
                ## Non-global (filtered) context (most common):
                # if active_session_configuration_context.has_keys(['lap_dir'])[0]:
                #     # directional laps version:
                #     active_session_configuration_name = active_session_configuration_context.get_subset(['filter_name','lap_dir']).get_description()
                #     # retired on 2023-11-29 after changing the 
                # else:
                #     # typical (non-directional laps) version:
                #     active_session_configuration_name = active_session_configuration_context.filter_name

                # # typical (non-directional laps) version:
                active_session_configuration_name = active_session_configuration_context.filter_name

            # Now have both `active_session_configuration_name`, `active_session_configuration_name`
        else:
            raise NotImplementedError(f"type(active_session_configuration_context): {type(active_session_configuration_context)}, active_session_configuration_context: {active_session_configuration_context}")
            pass # hope that it's an IdentifyingContext, but we'll check soon.
        

        ## Sets the active_context kwarg that's passed in to the display function:
        assert isinstance(active_session_configuration_context, IdentifyingContext)
        ## Now we're certain that we have an active_session_configuration_context:
        kwargs.setdefault('active_context', active_session_configuration_context) # add 'active_context' to the kwargs for the display function if possible

        if debug_print:
            print(f'active_session_configuration_name: "{active_session_configuration_name}", active_session_configuration_context: {active_session_configuration_context}')

        # Remove obsolite kwarg: We pop the active_config_name parameter from the kwargs, as this was an outdated workaround to optionally get the display functions this string but now it's passed directly by the call below        
        kwarg_active_config_name = kwargs.pop('active_config_name', None)
        if kwarg_active_config_name is not None:
            assert kwarg_active_config_name == active_session_configuration_name # they better be equal or else there is a conflict.
            raise PendingDeprecationWarning(f"2023-11-29- We pop the active_config_name parameter from the kwargs, as this was an outdated workaround to optionally get the display functions this string but now it's passed directly by the call below")


        # Check if the context is filtered or at the session level:
        if not hasattr(active_session_configuration_context, 'filter_name'):
            ## Global session-level context (not filtered, so not corresponding to a specific config name):
            ## For a global-style display function, pass ALL of the computation_results and active_configs just to preserve the argument style.
            # NOTE: global-style display functions have re-arranged arguments of the form (owning_pipeline_reference, global_computation_results, computation_results, active_configs, **kwargs). This differs from standard ones.
            assert getattr(display_function, 'is_global', False), f"display_function must be global if `active_session_configuration_context` does not have a `filter_name` property, but it is not!\n\tdisplay_function:{display_function}\n\tactive_session_configuration_context: {active_session_configuration_context}"
            curr_display_output = display_function(self, self.global_computation_results, self.computation_results, self.active_configs, active_config_name=None, **kwargs) # CALL GLOBAL DISPLAY FUNCTION
        
        else:
            ## Non-global (filtered) context:
            # Should be a display functions: The expected filtered context:

            ## Sanity checking:
            assert active_session_configuration_name is not None # not true for global contexts
            if (active_session_configuration_context.filter_name != active_session_configuration_name):
                print(f'WARN: active_session_configuration_context.filter_name != active_session_configuration_name: {active_session_configuration_context.filter_name} != {active_session_configuration_name}. This used to be an assert but to enable directional pfs it was reduced to a warning.')
            # assert active_session_configuration_context.filter_name == active_session_configuration_name
            assert (active_session_configuration_name in self.computation_results), f"self.computation_results doesn't contain a key for the provided active_session_filter_configuration ('{active_session_configuration_name}'). Did you only enable computation with enabled_filter_names in perform_computation that didn't include this key?"


            curr_display_output = display_function(self.computation_results[active_session_configuration_name], self.active_configs[active_session_configuration_name], owning_pipeline=self, active_config_name=active_session_configuration_name, **kwargs)
            
    
        ## Build the final display context: 
        found_display_fcn_index = self.registered_display_functions.index(display_function)
        display_fn_name = self.registered_display_function_names[found_display_fcn_index]
        active_display_fn_identifying_ctx = active_session_configuration_context.adding_context_if_missing(display_fn_name=display_fn_name) # display_fn_name should be like '_display_1d_placefields'

        # Add the display outputs to the active context. Each display function should return a structure like: dict(fig=active_figure, ax=ax_pf_1D)
        # owning_pipeline.display_output[active_display_fn_identifying_ctx] = (active_figure, ax_pf_1D)
        self.display_output[active_display_fn_identifying_ctx] = curr_display_output # sets the internal display reference to that item

        return curr_display_output



# ==================================================================================================================== #
# Figure Saving and Outputs                                                                                            #
# ==================================================================================================================== #
class PipelineWithDisplaySavingMixin:
    """ provides functionality for saving figures to file.
    
    from pyphoplacecellanalysis.General.Pipeline.Stages.Display import PipelineWithDisplaySavingMixin
    
    """
    
    def build_display_context_for_session(self, display_fn_name:str, **kwargs) -> "IdentifyingContext":
        """ builds a new display context for the session out of kwargs 
        Usage:
            curr_active_pipeline.build_display_context_for_session(display_fn_name='DecodedEpochSlices', epochs='replays', decoder='long_results_obj')
        """
        assert isinstance(display_fn_name, str), '"display_fn_name" must be provided as a string.'
        active_identifying_session_ctx = self.sess.get_context()
        display_subcontext = IdentifyingContext(display_fn_name=display_fn_name, **kwargs)
        return active_identifying_session_ctx.merging_context('display_', display_subcontext)
    

    
    def build_display_context_for_filtered_session(self, filtered_session_name:str, display_fn_name:str, **kwargs) -> "IdentifyingContext":
        """ builds a new display context for a filtered session out of kwargs 
        Usage:
            curr_active_pipeline.build_display_context_for_session(display_fn_name='DecodedEpochSlices', epochs='replays', decoder='long_results_obj')
        """
        assert isinstance(display_fn_name, str), '"display_fn_name" must be provided as a string.'
        active_identifying_session_ctx = self.filtered_contexts[filtered_session_name]
        display_subcontext = IdentifyingContext(display_fn_name=display_fn_name, **kwargs)
        return active_identifying_session_ctx.merging_context('display_', display_subcontext)

    @function_attributes(short_name=None, tags=['save','figure'], input_requires=[], output_provides=[], uses=['build_and_write_to_file'], used_by=[], creation_date='2023-06-14 19:26', related_items=[])
    def output_figure(self, final_context: IdentifyingContext, fig, context_tuple_join_character='_', write_vector_format:bool=False, write_png:bool=True, debug_print=True, override_fig_man: Optional[FileOutputManager]=None, **kwargs):
        """ outputs the figure using the provided context. 
        
        Usage:
            active_out_figure_paths, final_context = 
        """
        from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_and_write_to_file
        
        if override_fig_man is None:
            fig_man = self.get_output_manager()
        else:
            # use custom figure manager
            fig_man = override_fig_man

        # fig_man: FileOutputManager = self.get_output_manager() # get the output manager
        # figures_parent_out_path, fig_save_basename = fig_man.get_figure_output_parent_and_basename(final_context, make_folder_if_needed=True)
        # active_out_figure_paths = perform_write_to_file(fig, final_context, figures_parent_out_path=figures_parent_out_path, write_vector_format=write_vector_format, write_png=write_png, register_output_file_fn=self.register_output_file)
        # final_context = final_context.adding_context_if_missing(self.sess.get_context()) # add the session context if it's missing
        active_out_figure_paths = build_and_write_to_file(fig, final_context, fig_man=fig_man, context_tuple_join_character=context_tuple_join_character, write_vector_format=write_vector_format, write_png=write_png, register_output_file_fn=self.register_output_file, **kwargs)
        return active_out_figure_paths, final_context


    @classmethod
    def conform(cls, obj):
        """ makes the object conform to this mixin by adding its properties. 
        Usage:
            from pyphoplacecellanalysis.General.Pipeline.Stages.Computation import PipelineWithComputedPipelineStageMixin, ComputedPipelineStage
            from pyphoplacecellanalysis.General.Pipeline.Stages.Display import PipelineWithDisplayPipelineStageMixin, PipelineWithDisplaySavingMixin
            from pyphoplacecellanalysis.General.Pipeline.Stages.Filtering import FilteredPipelineMixin
            from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import PipelineWithInputStage, PipelineWithLoadableStage
            from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import PipelineStage
            from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline

            PipelineWithDisplaySavingMixin.conform(curr_active_pipeline)

        """
        def conform_to_implementing_method(func):
            """ captures 'obj', 'cls'"""
            setattr(type(obj), func.__name__, func)
        
        conform_to_implementing_method(cls.build_display_context_for_session)
        conform_to_implementing_method(cls.build_display_context_for_filtered_session)
        # conform_to_implementing_method(cls.write_figure_to_daily_programmatic_session_output_path)
        # conform_to_implementing_method(cls.write_figure_to_output_path)
        conform_to_implementing_method(cls.output_figure)
        



