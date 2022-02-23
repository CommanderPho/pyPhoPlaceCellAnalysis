from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
import pyvistaqt as pvqt


from pyphoplacecellanalysis.General.Pipeline.Stages.Computation import ComputedPipelineStage
from pyphoplacecellanalysis.General.Configs.DynamicConfigs import PlottingConfig, InteractivePlaceCellConfig
from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import PipelineStage
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.Ratemaps import DefaultRatemapDisplayFunctions
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import DefaultDecoderDisplayFunctions
from pyphoplacecellanalysis.General.Mixins.AllFunctionEnumeratingMixin import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Mixins.DisplayHelpers import _display_add_computation_param_text_box, _save_displayed_figure_if_needed

from neuropy.core.neuron_identities import NeuronIdentity, build_units_colormap, PlotStringBrevityModeEnum
from neuropy.plotting.placemaps import plot_all_placefields
from neuropy.plotting.ratemaps import enumTuningMap2DPlotVariables # for getting the variant name from the dict

from PhoGui.InteractivePlotter.Mixins.ImagePlaneRendering import ImagePlaneRendering

import PhoGui
from PhoGui.InteractivePlotter.PhoInteractivePlotter import PhoInteractivePlotter
from PhoGui.InteractivePlotter.shared_helpers import InteractivePyvistaPlotterBuildIfNeededMixin
from PhoGui.InteractivePlotter.InteractivePlaceCellDataExplorer import InteractivePlaceCellDataExplorer

from PhoGui.InteractivePlotter.InteractiveCustomDataExplorer import InteractiveCustomDataExplorer

from pyphoplacecellanalysis.GUI.Panel.panel_placefield import build_panel_interactive_placefield_visibility_controls, build_all_placefield_output_panels, SingleEditablePlacefieldDisplayConfiguration, ActivePlacefieldsPlottingPanel
from PhoGui.InteractivePlotter.InteractivePlaceCellTuningCurvesDataExplorer import InteractivePlaceCellTuningCurvesDataExplorer

from PhoPositionalData.plotting.placefield import plot_1d_placecell_validations



class DefaultDisplayFunctions(AllFunctionEnumeratingMixin):
    
    def _display_1d_placefield_validations(computation_result, active_config, **kwargs):
        """ Renders all of the flat 1D place cell validations with the yellow lines that trace across to their horizontally drawn placefield (rendered on the right of the plot) """
        out_figures_list = plot_1d_placecell_validations(computation_result.computed_data['pf1D'], active_config.plotting_config, **({'modifier_string': 'lap_only', 'should_save': False} | kwargs))
        return out_figures_list



    def _display_2d_placefield_result_plot_raw(computation_result, active_config, **kwargs):
        out_figures_list = computation_result.computed_data['pf2D'].plot_raw(**({'label_cells': True} | kwargs)); # Plots an overview of each cell all in one figure
        return out_figures_list

    def _display_2d_placefield_result_plot_ratemaps_2D(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
         # Build the unique identifier config for this result:
        # combined_id_config = UniqueCombinedConfigIdentifier(filter_name, active_config, variant_identifier_label=variant_identifier_label)
    
        # active_plot_type_name = '_display_2d_placefield_result_plot_ratemaps_2D' 
        # active_variant_name = None
        # if active_variant_name is not None:
        #     active_plot_filename = '-'.join([active_plot_type_name, active_variant_name])
        # else:
        #     active_plot_filename = active_plot_type_name
        # active_plot_filepath = active_config.plotting_config.get_figure_save_path(active_plot_filename).with_suffix('.png')
        # print(f'active_plot_filepath: {active_plot_filepath}')
        
        
        
        # active_pf_2D_output_filepath = active_config.plotting_config.get_figure_save_path(common_parent_foldername, common_basename).with_suffix('.png')
        # with WrappingMessagePrinter('Saving 2D Placefield image out to "{}"...'.format(active_pf_2D_output_filepath), begin_line_ending='...', finished_message='done.'):
        #     for aFig in active_pf_2D_figures:
        #         aFig.savefig(active_pf_2D_output_filepath)
        computation_result.computed_data['pf2D'].plot_ratemaps_2D(**({'subplots': (None, 3), 'resolution_multiplier': 1.0, 'enable_spike_overlay': False, 'brev_mode': PlotStringBrevityModeEnum.MINIMAL} | kwargs))
        
        # plot_variable_name = ({'plot_variable': None} | kwargs)
        plot_variable_name = kwargs.get('plot_variable', enumTuningMap2DPlotVariables.TUNING_MAPS).name
        active_figure = plt.gcf()
        _display_add_computation_param_text_box(active_figure, active_config.computation_config) # Adds the parameters text.
        
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
        Usage:
            _display_normal(curr_kdiba_pipeline.computation_results['maze1'], curr_kdiba_pipeline.active_configs['maze1'])
        """
        # print(f'active_config: {active_config}')
        # active_config = computation_result.sess.config
        if active_config.computation_config is None:
            active_config.computation_config = computation_result.computation_config

        # ax_pf_1D, occupancy_fig, active_pf_2D_figures, active_pf_2D_gs = plot_all_placefields(computation_result.computed_data['pf1D'], computation_result.computed_data['pf2D'], active_config, should_save_to_disk=False)
        ax_pf_1D, occupancy_fig, active_pf_2D_figures, active_pf_2D_gs = plot_all_placefields(None, computation_result.computed_data['pf2D'], active_config, **({'should_save_to_disk': False} | kwargs))
        
        return occupancy_fig, active_pf_2D_figures
        

    ## Tuning Curves 3D Plot:
    def _display_3d_interactive_tuning_curves_plotter(computation_result, active_config, **kwargs):
        """ 
        Inputs: {'extant_plotter': None} 
        Outputs: {'ipcDataExplorer', 'pActiveTuningCurvesPlotter', 'pane'}
        """
        # try: pActiveTuningCurvesPlotter
        # except NameError: pActiveTuningCurvesPlotter = None # Checks variable p's existance, and sets its value to None if it doesn't exist so it can be checked in the next step

        pActiveTuningCurvesPlotter = kwargs.get('extant_plotter', None)
        # if pActiveTuningCurvesPlotter is None:
        #     print('_display_3d_interactive_tuning_curves_plotter: No extant plotter provided!')
        # else:
        #     print(f"_display_3d_interactive_tuning_curves_plotter: extant_plotter: {kwargs['extant_plotter']}")

        ipcDataExplorer = InteractivePlaceCellTuningCurvesDataExplorer(active_config, computation_result.sess, computation_result.computed_data['pf2D'], active_config.plotting_config.pf_colors, **({'extant_plotter':None} | kwargs))
        pActiveTuningCurvesPlotter = ipcDataExplorer.plot(pActiveTuningCurvesPlotter) # [2, 17449]
        ### Build Dynamic Panel Interactive Controls for configuring Placefields:
        pane = build_panel_interactive_placefield_visibility_controls(ipcDataExplorer)
        # return pane
        return {'ipcDataExplorer': ipcDataExplorer, 'pActiveTuningCurvesPlotter': pActiveTuningCurvesPlotter, 'pane': pane}
            
        

    ## Interactive 3D Spike and Behavior Browser: 
    def _display_3d_interactive_spike_and_behavior_browser(computation_result, active_config, **kwargs):
        """ 
        Inputs: {'extant_plotter': None} 
        Outputs: {'ipspikesDataExplorer', 'pActiveInteractivePlaceSpikesPlotter'}
        """
        active_config.plotting_config.show_legend = True        
        active_session = computation_result.sess # this is unfiltered, shouldn't be used... actually, it should be filtered. Don't know what's wrong here.

        print(f'computation_result.sess: {computation_result.sess}')
        # try: pActiveInteractivePlaceSpikesPlotter
        # except NameError: pActiveInteractivePlaceSpikesPlotter = None # Checks variable p's existance, and sets its value to None if it doesn't exist so it can be checked in the next step
        pActiveInteractivePlaceSpikesPlotter = kwargs.get('extant_plotter', None)
        ipspikesDataExplorer = InteractivePlaceCellDataExplorer(active_config, computation_result.sess, **({'extant_plotter':None} | kwargs))
        pActiveInteractivePlaceSpikesPlotter = ipspikesDataExplorer.plot(pActivePlotter=pActiveInteractivePlaceSpikesPlotter)
        return {'ipspikesDataExplorer': ipspikesDataExplorer, 'pActiveInteractivePlaceSpikesPlotter': pActiveInteractivePlaceSpikesPlotter}

    ## CustomDataExplorer 3D Plotter:
    def _display_3d_interactive_custom_data_explorer(computation_result, active_config, **kwargs):
        """ 
        Inputs: {'extant_plotter': None} 
        Outputs: {'iplapsDataExplorer', 'pActiveInteractiveLapsPlotter'}
        """
        active_laps_config = InteractivePlaceCellConfig(active_session_config=computation_result.sess.config, active_epochs=None, video_output_config=None, plotting_config=None) # '3|1    
        active_laps_config.plotting_config = PlottingConfig(output_subplots_shape='1|5', output_parent_dir=Path('output', computation_result.sess.config.session_name, 'custom_laps'))
        # try: pActiveInteractiveLapsPlotter
        # except NameError: pActiveInteractiveLapsPlotter = None # Checks variable p's existance, and sets its value to None if it doesn't exist so it can be checked in the next step
        pActiveInteractiveLapsPlotter = kwargs.get('extant_plotter', None)
        iplapsDataExplorer = InteractiveCustomDataExplorer(active_laps_config, computation_result.sess, **({'extant_plotter':None} | kwargs))
        pActiveInteractiveLapsPlotter = iplapsDataExplorer.plot(pActivePlotter=pActiveInteractiveLapsPlotter)
        return {'iplapsDataExplorer': iplapsDataExplorer, 'pActiveInteractiveLapsPlotter': pActiveInteractiveLapsPlotter}

    def _display_3d_image_plotter(computation_result, active_config, **kwargs):
        """ 
        Inputs: {'extant_plotter': None} 
        Outputs: {'pActiveImageTestPlotter'}
        """
        def plot_3d_image_plotter(active_epoch_placefields2D, image_file=r'output\2006-6-07_11-26-53\maze\speedThresh_0.00-gridBin_5.00_3.00-smooth_0.00_0.00-frateThresh_0.10\pf2D-Occupancy-maze-odd_laps-speedThresh_0.00-gridBin_5.00_3.00-smooth_0.00_0.00-frateThresh_0.png'):
            loaded_image_tex = pv.read_texture(image_file)
            pActiveImageTestPlotter = pvqt.BackgroundPlotter()
            return ImagePlaneRendering.plot_3d_image(pActiveImageTestPlotter, active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.occupancy, loaded_image_tex=loaded_image_tex)
            
        # Texture from file:
        image_file = r'output\2006-6-07_11-26-53\maze\speedThresh_0.00-gridBin_5.00_3.00-smooth_0.00_0.00-frateThresh_0.10\pf2D-Occupancy-maze-odd_laps-speedThresh_0.00-gridBin_5.00_3.00-smooth_0.00_0.00-frateThresh_0.png'
        pActiveImageTestPlotter = plot_3d_image_plotter(computation_result.computed_data['pf2D'], image_file=image_file)
        return {'pActiveImageTestPlotter': pActiveImageTestPlotter}
