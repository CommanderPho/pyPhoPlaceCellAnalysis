import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import pyqtplot_plot_image_array, pyqtplot_common_setup
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters

## Required for _display_2d_placefield_result_plot_ratemaps_2D and _display_normal
from pyphoplacecellanalysis.General.Mixins.DisplayHelpers import _display_add_computation_param_text_box, _save_displayed_figure_if_needed

from neuropy.core.neuron_identities import NeuronIdentity, build_units_colormap, PlotStringBrevityModeEnum
from neuropy.plotting.placemaps import plot_all_placefields
from neuropy.utils.matplotlib_helpers import enumTuningMap2DPlotVariables # for getting the variant name from the dict
from neuropy.utils.mixins.unwrap_placefield_computation_parameters import unwrap_placefield_computation_parameters



class DefaultRatemapDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ Functions related to visualizing Bayesian Decoder performance. """
    
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
        
    def _display_placemaps_pyqtplot_2D(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots the prediction error for the two_step decoder at each point in time.
            Based off of "_temp_debug_two_step_plots_animated_imshow"
        """
        # Get the decoders from the computation result:
        active_one_step_decoder = computation_result.computed_data['pf2D_Decoder'] # doesn't actually require the Decoder, could just use computation_result.computed_data['pf2D']            
        # Get flat list of images:
        images = active_one_step_decoder.ratemap.normalized_tuning_curves # (43, 63, 63)
        # images = active_one_step_decoder.ratemap.normalized_tuning_curves[0:40,:,:] # (43, 63, 63)
        occupancy = active_one_step_decoder.ratemap.occupancy
        app, parent_root_widget, root_render_widget, plot_array, img_item_array, other_components_array = pyqtplot_plot_image_array(active_one_step_decoder.xbin, active_one_step_decoder.ybin, images, occupancy, 
                                                                                app=kwargs.get('app',None), parent_root_widget=kwargs.get('parent_root_widget',None), root_render_widget=kwargs.get('root_render_widget',None))
        # win.show()
        display_outputs = DynamicParameters(root_render_widget=root_render_widget, plot_array=plot_array, img_item_array=img_item_array, other_components_array=other_components_array)
        # return app, parent_root_widget, root_render_widget
        return app, parent_root_widget, display_outputs


