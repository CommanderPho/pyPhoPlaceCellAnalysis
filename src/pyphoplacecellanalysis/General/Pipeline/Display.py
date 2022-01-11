from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv
import pyvistaqt as pvqt
from PhoPositionalData.analysis.interactive_placeCell_config import InteractivePlaceCellConfig, PlottingConfig

from neuropy.core.neuron_identities import NeuronIdentity, build_units_colormap, PlotStringBrevityModeEnum
from neuropy.plotting.placemaps import plot_all_placefields
from PhoGui.InteractivePlotter.Mixins.ImagePlaneRendering import ImagePlaneRendering

import PhoGui
from PhoGui.InteractivePlotter.PhoInteractivePlotter import PhoInteractivePlotter
from PhoGui.InteractivePlotter.shared_helpers import InteractivePyvistaPlotterBuildIfNeededMixin
from PhoGui.InteractivePlotter.InteractivePlaceCellDataExplorer import InteractivePlaceCellDataExplorer

from PhoGui.InteractivePlotter.InteractiveCustomDataExplorer import InteractiveCustomDataExplorer

from pyphoplacecellanalysis.GUI.Panel.panel_placefield import build_panel_interactive_placefield_visibility_controls, build_all_placefield_output_panels, SingleEditablePlacefieldDisplayConfiguration, ActivePlacefieldsPlottingPanel
from PhoGui.InteractivePlotter.InteractivePlaceCellTuningCurvesDataExplorer import InteractivePlaceCellTuningCurvesDataExplorer
    

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
    
class DefaultDisplayFunctions:
    def _display_normal(computation_result, active_config):
        """
        Usage:
            _display_normal(curr_kdiba_pipeline.computation_results['maze1'], curr_kdiba_pipeline.active_configs['maze1'])
        """
        
        # from neuropy.core.neuron_identities import build_units_colormap
        # print(f'active_config: {active_config}')
        # active_config = computation_result.sess.config
        if active_config.computation_config is None:
            active_config.computation_config = computation_result.computation_config

        pf_neuron_identities, pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = get_neuron_identities(computation_result.computed_data['pf2D'])
        # ax_pf_1D, occupancy_fig, active_pf_2D_figures, active_pf_2D_gs = plot_all_placefields(computation_result.computed_data['pf1D'], computation_result.computed_data['pf2D'], active_config, should_save_to_disk=False)
        ax_pf_1D, occupancy_fig, active_pf_2D_figures, active_pf_2D_gs = plot_all_placefields(None, computation_result.computed_data['pf2D'], active_config, should_save_to_disk=False)
        
        
    ## Tuning Curves 3D Plot:
    def _display_3d_interactive_tuning_curves_plotter(computation_result, active_config):
        # try: pActiveTuningCurvesPlotter
        # except NameError: pActiveTuningCurvesPlotter = None # Checks variable p's existance, and sets its value to None if it doesn't exist so it can be checked in the next step
        pActiveTuningCurvesPlotter = None
        try:
            len(active_config.plotting_config.pf_colors)
        except AttributeError as e:
            # add the attributes 
            active_config.plotting_config.pf_neuron_identities, active_config.plotting_config.pf_sort_ind, active_config.plotting_config.pf_colors, active_config.plotting_config.pf_colormap, active_config.plotting_config.pf_listed_colormap = get_neuron_identities(computation_result.computed_data['pf2D'])
        
        ipcDataExplorer = InteractivePlaceCellTuningCurvesDataExplorer(active_config, computation_result.sess, computation_result.computed_data['pf2D'], active_config.plotting_config.pf_colors, extant_plotter=pActiveTuningCurvesPlotter)
        pActiveTuningCurvesPlotter = ipcDataExplorer.plot(pActiveTuningCurvesPlotter) # [2, 17449]
        ### Build Dynamic Panel Interactive Controls for configuring Placefields:
        pane = build_panel_interactive_placefield_visibility_controls(ipcDataExplorer)
        pane



    ## Interactive 3D Spike and Behavior Browser: 
    def _display_3d_interactive_spike_and_behavior_browser(computation_result, active_config):
        active_config.plotting_config.show_legend = True
        # try: pActiveInteractivePlaceSpikesPlotter
        # except NameError: pActiveInteractivePlaceSpikesPlotter = None # Checks variable p's existance, and sets its value to None if it doesn't exist so it can be checked in the next step
        pActiveInteractivePlaceSpikesPlotter = None
        ipspikesDataExplorer = InteractivePlaceCellDataExplorer(active_config, computation_result.sess, extant_plotter=pActiveInteractivePlaceSpikesPlotter)
        pActiveInteractivePlaceSpikesPlotter = ipspikesDataExplorer.plot(pActivePlotter=pActiveInteractivePlaceSpikesPlotter)


    ## CustomDataExplorer 3D Plotter:
    def _display_3d_interactive_custom_data_explorer(computation_result, active_config):
        active_laps_config = InteractivePlaceCellConfig(active_session_config=computation_result.sess.config, active_epochs=None, video_output_config=None, plotting_config=None) # '3|1    
        active_laps_config.plotting_config = PlottingConfig(output_subplots_shape='1|5', output_parent_dir=Path('output', computation_result.sess.config.session_name, 'custom_laps'))
        # try: pActiveInteractiveLapsPlotter
        # except NameError: pActiveInteractiveLapsPlotter = None # Checks variable p's existance, and sets its value to None if it doesn't exist so it can be checked in the next step
        pActiveInteractiveLapsPlotter = None
        iplapsDataExplorer = InteractiveCustomDataExplorer(active_laps_config, computation_result.sess, extant_plotter=pActiveInteractiveLapsPlotter)
        pActiveInteractiveLapsPlotter = iplapsDataExplorer.plot(pActivePlotter=pActiveInteractiveLapsPlotter)




    def _display_3d_image_plotter(computation_result, active_config):
        def plot_3d_image_plotter(active_epoch_placefields2D, image_file=r'output\2006-6-07_11-26-53\maze\speedThresh_0.00-gridBin_5.00_3.00-smooth_0.00_0.00-frateThresh_0.10\pf2D-Occupancy-maze-odd_laps-speedThresh_0.00-gridBin_5.00_3.00-smooth_0.00_0.00-frateThresh_0.png'):
            loaded_image_tex = pv.read_texture(image_file)
            pActiveImageTestPlotter = pvqt.BackgroundPlotter()
            return ImagePlaneRendering.plot_3d_image(pActiveImageTestPlotter, active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.occupancy, loaded_image_tex=loaded_image_tex)
            
        # Texture from file:
        image_file = r'output\2006-6-07_11-26-53\maze\speedThresh_0.00-gridBin_5.00_3.00-smooth_0.00_0.00-frateThresh_0.10\pf2D-Occupancy-maze-odd_laps-speedThresh_0.00-gridBin_5.00_3.00-smooth_0.00_0.00-frateThresh_0.png'
        pActiveImageTestPlotter = plot_3d_image_plotter(computation_result.computed_data['pf2D'], image_file=image_file)



class DefaultRegisteredDisplayFunctions:
    """ Simply enables specifying the default computation functions that will be defined in this file and automatically registered. """
    def register_default_known_display_functions(self):
        self.register_display_function(DefaultDisplayFunctions._display_normal)
        self.register_display_function(DefaultDisplayFunctions._display_3d_interactive_tuning_curves_plotter)
        self.register_display_function(DefaultDisplayFunctions._display_3d_interactive_spike_and_behavior_browser)
        self.register_display_function(DefaultDisplayFunctions._display_3d_interactive_custom_data_explorer)
        self.register_display_function(DefaultDisplayFunctions._display_3d_image_plotter)
  