import sys
from pyphocorehelpers.function_helpers import compose_functions

# NeuroPy (Diba Lab Python Repo) Loading
try:
    from neuropy import core
except ImportError:
    sys.path.append(r"C:\Users\Pho\repos\NeuroPy")  # Windows
    # sys.path.append('/home/pho/repo/BapunAnalysis2021/NeuroPy') # Linux
    # sys.path.append(r'/Users/pho/repo/Python Projects/NeuroPy') # MacOS
    print("neuropy module not found, adding directory to sys.path. \n >> Updated sys.path.")
    from neuropy import core

from neuropy.analyses.placefields import PlacefieldComputationParameters, perform_compute_placefields


from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import BaseNeuropyPipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Filtering import FilterablePipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import LoadableInput, LoadableSessionInput, LoadedPipelineStage    
from pyphoplacecellanalysis.General.ComputationResults import ComputationResult


class ComputablePipelineStage:
    """ Designates that a pipeline stage is computable. """
        
    @classmethod
    def _perform_single_computation(cls, active_session, computation_config):
        """Conceptually, a single computation consists of a specific active_session and a specific computation_config object
        Args:
            active_session (DataSession): [description]
            computation_config (PlacefieldComputationParameters): [description]

        Returns:
            [type]: [description]
        """
        # only requires that active_session has the .spikes_df and .position  properties
        # active_epoch_placefields1D, active_epoch_placefields2D = compute_placefields_masked_by_epochs(active_epoch_session, active_config, included_epochs=None, should_display_2D_plots=should_display_2D_plots) ## This is causing problems due to deepcopy of session.
        output_result = ComputationResult(active_session, computation_config, computed_data=dict())
        # active_epoch_placefields1D, active_epoch_placefields2D = perform_compute_placefields(active_session.spikes_df, active_session.position, computation_config, None, None, included_epochs=None, should_force_recompute_placefields=True)
        
        # Test to see if included_epochs is set, if not, set it to None.
        
        
        output_result.computed_data['pf1D'], output_result.computed_data['pf2D'] = perform_compute_placefields(active_session.spikes_df, active_session.position, computation_config, None, None, included_epochs=computation_config.computation_epochs, should_force_recompute_placefields=True)

        # Compare the results:

        # debug_print_ratemap(active_epoch_placefields1D.ratemap)
        # num_spikes_per_spiketrain = np.array([np.shape(a_spk_train)[0] for a_spk_train in active_epoch_placefields1D.spk_t])
        # num_spikes_per_spiketrain
        # print('placefield_neuronID_spikes: {}; ({} total spikes)'.format(num_spikes_per_spiketrain, np.sum(num_spikes_per_spiketrain)))
        # debug_print_placefield(active_epoch_placefields1D) #49 good
        # debug_print_placefield(output_result.computed_data['pf2D']) #51 good

        return output_result

    def single_computation(self, active_computation_params: PlacefieldComputationParameters):
        """ Takes its filtered_session and applies the provided active_computation_params to it. The results are stored in self.computation_results under the same key as the filtered session. """
        assert (len(self.filtered_sessions.keys()) > 0), "Must have at least one filtered session before calling single_computation(...). Call self.select_filters(...) first."
        # self.active_computation_results = dict()
        for a_select_config_name, a_filtered_session in self.filtered_sessions.items():
            print(f'Performing single_computation on filtered_session with filter named "{a_select_config_name}"...')
            self.active_configs[a_select_config_name].computation_config = active_computation_params #TODO: if more than one computation config is passed in, the active_config should be duplicated for each computation config.
            self.computation_results[a_select_config_name] = ComputablePipelineStage._perform_single_computation(a_filtered_session, active_computation_params) # returns a computation result. Does this store the computation config used to compute it?
        
            # call to perform any registered computations:
            self.computation_results[a_select_config_name] = self.perform_registered_computations(self.computation_results[a_select_config_name], debug_print=True)

        # pf_neuron_identities, pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = _get_neuron_identities(computation_result.computed_data['pf1D'])
        # pf_neuron_identities, pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = _get_neuron_identities(self.active_computation_results[a_select_config_name].computed_data['pf2D'])


"""-------------- Specific Computation Functions to be registered --------------"""

from pyphoplacecellanalysis.Analysis.reconstruction import BayesianPlacemapPositionDecoder

def _perform_position_decoding_computation(computation_result: ComputationResult):
    """ Builds the 2D Placefield Decoder """
    def position_decoding_computation(active_session, computation_config, prev_output_result: ComputationResult):
        prev_output_result.computed_data['pf2D_Decoder'] = BayesianPlacemapPositionDecoder(computation_config.time_bin_size, prev_output_result.computed_data['pf2D'], active_session.spikes_df.copy(), debug_print=False)
        # %timeit pho_custom_decoder.compute_all():  18.8 s ± 149 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        prev_output_result.computed_data['pf2D_Decoder'].compute_all() #  --> n = self.
        return prev_output_result

    return position_decoding_computation(computation_result.sess, computation_result.computation_config, computation_result)



class DefaultRegisteredComputations:
    """ Simply enables specifying the default computation functions that will be defined in this file and automatically registered. """
    def register_default_known_computation_functions(self):
        self.register_computation(_perform_position_decoding_computation)
    


class PipelineWithComputedPipelineStageMixin:
    """ To be added to the pipeline to enable conveninece access ot its pipeline stage post Computed stage. """
    ## Computed Properties:
    @property
    def is_computed(self):
        """The is_computed property. TODO: Needs validation/Testing """
        return (self.stage is not None) and (isinstance(self.stage, ComputedPipelineStage) and (self.computation_results.values[0] is not None))

    @property
    def computation_results(self):
        """The computation_results property, accessed through the stage."""
        return self.stage.computation_results
    
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
        # self.register_default_known_computation_functions() # registers the default
        
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
    