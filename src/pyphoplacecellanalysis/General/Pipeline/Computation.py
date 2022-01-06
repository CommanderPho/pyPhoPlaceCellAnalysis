import sys
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
        output_result.computed_data['pf1D'], output_result.computed_data['pf2D'] = perform_compute_placefields(active_session.spikes_df, active_session.position, computation_config, None, None, included_epochs=None, should_force_recompute_placefields=True)

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
        
        # pf_neuron_identities, pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = _get_neuron_identities(computation_result.computed_data['pf1D'])
        # pf_neuron_identities, pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = _get_neuron_identities(self.active_computation_results[a_select_config_name].computed_data['pf2D'])

    