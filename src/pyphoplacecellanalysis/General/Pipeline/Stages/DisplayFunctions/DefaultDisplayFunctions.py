import matplotlib.pyplot as plt

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
from pyphoplacecellanalysis.General.Mixins.DisplayHelpers import _display_add_computation_param_text_box, _save_displayed_figure_if_needed

from neuropy.core.neuron_identities import NeuronIdentity, build_units_colormap, PlotStringBrevityModeEnum
from neuropy.plotting.placemaps import plot_all_placefields
from neuropy.plotting.ratemaps import enumTuningMap2DPlotVariables # for getting the variant name from the dict
from neuropy.utils.mixins.unwrap_placefield_computation_parameters import unwrap_placefield_computation_parameters

from pyphoplacecellanalysis.PhoPositionalData.plotting.placefield import plot_1d_placecell_validations


class DefaultDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    
    def _display_1d_placefield_validations(computation_result, active_config, **kwargs):
        """ Renders all of the flat 1D place cell validations with the yellow lines that trace across to their horizontally drawn placefield (rendered on the right of the plot) """
        out_figures_list = plot_1d_placecell_validations(computation_result.computed_data['pf1D'], active_config.plotting_config, **({'modifier_string': 'lap_only', 'should_save': False} | kwargs))
        return out_figures_list

    def _display_2d_placefield_result_plot_raw(computation_result, active_config, **kwargs):
        """ produces a stupid figure """
        out_figures_list = computation_result.computed_data['pf2D'].plot_raw(**({'label_cells': True} | kwargs)); # Plots an overview of each cell all in one figure
        return out_figures_list

    def _display_2d_placefield_result_plot_ratemaps_2D(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ displays 2D placefields in a MATPLOTLIB window 
        
        Internally wraps `PfND.plot_ratemaps_2D` which itself wraps `neuropy.plotting.ratemaps.plot_ratemap_2D`
        
            optionally shows peak firing rates
        
        """
        computation_result.computed_data['pf2D'].plot_ratemaps_2D(**({'subplots': (None, 3), 'resolution_multiplier': 1.0, 'enable_spike_overlay': False, 'brev_mode': PlotStringBrevityModeEnum.MINIMAL} | kwargs))
        
        # plot_variable_name = ({'plot_variable': None} | kwargs)
        plot_variable_name = kwargs.get('plot_variable', enumTuningMap2DPlotVariables.TUNING_MAPS).name
        active_figure = plt.gcf()
        
        active_pf_computation_params = unwrap_placefield_computation_parameters(active_config.computation_config)
        _display_add_computation_param_text_box(active_figure, active_pf_computation_params) # Adds the parameters text.
        
        active_pf_2D_figures = [active_figure]            
        
        # Save the figure out to disk if we need to:
        should_save_to_disk = enable_saving_to_disk
        if should_save_to_disk:
            _save_displayed_figure_if_needed(active_config.plotting_config, plot_type_name='_display_2d_placefield_result_plot_ratemaps_2D', active_variant_name=plot_variable_name, active_figures=active_pf_2D_figures)
        
        return active_pf_2D_figures
    
    # def _display_2d_placefield_result(computation_result, active_config):
    #     """ Renders the red trajectory info as the first figure, and then the ratemaps as the second. """
    #     active_config = add_neuron_identity_info_if_needed(computation_result, active_config)
    #     computation_result.computed_data['pf2D'].plot_raw(label_cells=True); # Plots an overview of each cell all in one figure
    #     computation_result.computed_data['pf2D'].plot_ratemaps_2D(resolution_multiplier=2.5, brev_mode=PlotStringBrevityModeEnum.MINIMAL)


    def _display_normal(computation_result, active_config, **kwargs):
        """
        
        Internally wraps `neuropy.plotting.placemaps.plot_all_placefields` which itself wraps `PfND.plot_ratemaps_2D` which itself wraps `neuropy.plotting.ratemaps.plot_ratemap_2D`
        
        Usage:
            _display_normal(curr_kdiba_pipeline.computation_results['maze1'], curr_kdiba_pipeline.active_configs['maze1'])
        """
        if active_config.computation_config is None:
            active_config.computation_config = computation_result.computation_config
        ax_pf_1D, occupancy_fig, active_pf_2D_figures, active_pf_2D_gs = plot_all_placefields(None, computation_result.computed_data['pf2D'], active_config, **({'should_save_to_disk': False} | kwargs))
        
        return occupancy_fig, active_pf_2D_figures
        
