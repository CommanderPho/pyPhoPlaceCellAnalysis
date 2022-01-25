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
from pyphoplacecellanalysis.General.Decoder.decoder_result import DecoderResultDisplayingPlot2D    


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
    curr_computation_config_output_dir_name = active_config.computation_config.str_for_filename(False)
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
    
    
def _save_displayed_figure_if_needed(plotting_config, plot_type_name='plot', active_variant_name=None, active_figures=list(), debug_print=False):
    if active_variant_name is not None:
        active_plot_filename = '-'.join([plot_type_name, active_variant_name])
    else:
        active_plot_filename = plot_type_name
    active_plot_filepath = plotting_config.get_figure_save_path(active_plot_filename).with_suffix('.png')
    if debug_print:
        print(f'active_plot_filepath: {active_plot_filepath}')
    with WrappingMessagePrinter('Saving 2D Placefield image out to "{}"...'.format(active_plot_filepath), begin_line_ending='...', finished_message='done.'):
        for aFig in active_figures:
            aFig.savefig(active_plot_filepath)
    
    
# Post plotting figure helpers:
def _display_add_computation_param_text_box(fig, computation_config):
    """ Adds a small box containing the computation parmaters to the matplotlib figure. 
    Usage:
        _display_add_computation_param_text_box(plt.gcf(), active_session_computation_config)
    """
    if fig is None:
        fig = plt.gcf()
    render_text = computation_config.str_for_attributes_list_display(key_val_sep_char=':')
    return add_figure_text_box(fig, render_text=render_text)

                
class DefaultDisplayFunctions:

    def _display_1d_placefield_validations(computation_result, active_config, **kwargs):
        """ Renders all of the flat 1D place cell validations with the yellow lines that trace across to their horizontally drawn placefield (rendered on the right of the plot) """
        active_config = add_neuron_identity_info_if_needed(computation_result, active_config)
        out_figures_list = plot_1d_placecell_validations(computation_result.computed_data['pf1D'], active_config.plotting_config, **({'modifier_string': 'lap_only', 'should_save': False} | kwargs))


    def _display_2d_placefield_result_plot_raw(computation_result, active_config, **kwargs):
        active_config = add_neuron_identity_info_if_needed(computation_result, active_config)
        computation_result.computed_data['pf2D'].plot_raw(**({'label_cells': True} | kwargs)); # Plots an overview of each cell all in one figure


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
        
        # Save the figure out to disk if we need to:
        should_save_to_disk = enable_saving_to_disk
        if should_save_to_disk:
            active_pf_2D_figures = [active_figure]            
            _save_displayed_figure_if_needed(active_config.plotting_config, plot_type_name='_display_2d_placefield_result_plot_ratemaps_2D', active_variant_name=plot_variable_name, active_figures=active_pf_2D_figures)
        
 
    # def _display_2d_placefield_result(computation_result, active_config):
    #     """ Renders the red trajectory info as the first figure, and then the ratemaps as the second. """
    #     active_config = add_neuron_identity_info_if_needed(computation_result, active_config)
    #     computation_result.computed_data['pf2D'].plot_raw(label_cells=True); # Plots an overview of each cell all in one figure
    #     computation_result.computed_data['pf2D'].plot_ratemaps_2D(resolution_multiplier=2.5, brev_mode=PlotStringBrevityModeEnum.MINIMAL)

    
    def _display_decoder_result(computation_result, active_config):
        renderer = DecoderResultDisplayingPlot2D(computation_result.computed_data['pf2D_Decoder'], computation_result.sess.position.to_dataframe())
        def animate(i):
            # print(f'animate({i})')
            return renderer.display(i)
        
        
        # interact(animate, i=(0, computation_result.computed_data['pf2D_Decoder'].num_time_windows, 10))
  

    
    def _display_plot_most_likely_position_comparisons(computation_result, active_config, **kwargs):
        """ renders the computed posterior for the position from the Bayesian decoder and overlays the animal's actual position over the top. """
        def plot_most_likely_position_comparsions(pho_custom_decoder, position_df, show_posterior=True, show_one_step_most_likely_positions_plots=True, debug_print=False):
            """
            Usage:
                fig, axs = plot_most_likely_position_comparsions(pho_custom_decoder, sess.position.to_dataframe())
            """
            # xmin, xmax, ymin, ymax, tmin, tmax = compute_data_extent(position_df['x'].to_numpy(), position_df['y'].to_numpy(), position_df['t'].to_numpy())
            
            with plt.ion():
                overlay_mode = True
                if overlay_mode:
                    nrows=2
                else:
                    nrows=4
                fig, axs = plt.subplots(ncols=1, nrows=nrows, figsize=(15,15), clear=True, sharex=True, sharey=False, constrained_layout=False)
                # active_window = pho_custom_decoder.active_time_windows[window_idx] # a tuple with a start time and end time
                # active_p_x_given_n = np.squeeze(pho_custom_decoder.p_x_given_n[:,:,window_idx]) # same size as occupancy
                # Actual Position Plots:
                axs[0].plot(position_df['t'].to_numpy(), position_df['x'].to_numpy(), label='measured x', color='#ff0000ff')
                axs[0].set_title('x')
                axs[1].plot(position_df['t'].to_numpy(), position_df['y'].to_numpy(), label='measured y', color='#ff0000ff')
                axs[1].set_title('y')
                # # Most likely position plots:
                # axs[2].plot(pho_custom_decoder.active_time_window_centers, np.squeeze(pho_custom_decoder.most_likely_positions[:,0]), lw=0.5) # (Num windows x 2)
                # axs[2].set_title('most likely positions x')
                # axs[3].plot(pho_custom_decoder.active_time_window_centers, np.squeeze(pho_custom_decoder.most_likely_positions[:,1]), lw=0.5) # (Num windows x 2)
                # axs[3].set_title('most likely positions y')
                
                if show_posterior:
                    active_posterior = pho_custom_decoder.p_x_given_n
                    # active_posterior = pho_custom_decoder['p_x_given_n_and_x_prev'] # dict-style
                    
                    # re-normalize each marginal after summing
                    marginal_posterior_y = np.squeeze(np.sum(active_posterior, 0)) # sum over all x. Result should be [y_bins x time_bins]
                    marginal_posterior_y = marginal_posterior_y / np.sum(marginal_posterior_y, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)
                    marginal_posterior_x = np.squeeze(np.sum(active_posterior, 1)) # sum over all y. Result should be [x_bins x time_bins]
                    marginal_posterior_x = marginal_posterior_x / np.sum(marginal_posterior_x, axis=0) # sum over all positions for each time_bin (so there's a normalized distribution at each timestep)

                    # Compute extents foir imshow:
                    
                    # im = ax.imshow(curr_p_x_given_n, **main_plot_kwargs) # add the curr_px_given_n image
                    main_plot_kwargs = {
                        'origin': 'lower',
                        'vmin': 0,
                        'vmax': 1,
                        'cmap': 'turbo',
                        'interpolation':'nearest',
                        'aspect':'auto',
                    }
                        
                    # Posterior distribution heatmaps at each point.
                    # X
                    xmin, xmax, ymin, ymax = (pho_custom_decoder.active_time_window_centers[0], pho_custom_decoder.active_time_window_centers[-1], pho_custom_decoder.xbin[0], pho_custom_decoder.xbin[-1])           
                    # xmin, xmax = axs[0].get_xlim()
                    x_first_extent = (xmin, xmax, ymin, ymax)
                    # y_first_extent = (ymin, ymax, xmin, xmax)
                    active_extent = x_first_extent
                    im_posterior_x = axs[0].imshow(marginal_posterior_x, extent=active_extent, **main_plot_kwargs)
                    # axs[0].axis("off")
                    axs[0].set_xlim((xmin, xmax))
                    axs[0].set_ylim((ymin, ymax))
                    # Y
                    xmin, xmax, ymin, ymax = (pho_custom_decoder.active_time_window_centers[0], pho_custom_decoder.active_time_window_centers[-1], pho_custom_decoder.ybin[0], pho_custom_decoder.ybin[-1])
                    x_first_extent = (xmin, xmax, ymin, ymax)
                    # y_first_extent = (ymin, ymax, xmin, xmax)
                    active_extent = x_first_extent
                    im_posterior_y = axs[1].imshow(marginal_posterior_y, extent=active_extent, **main_plot_kwargs)
                    # axs[1].axis("off")
                    axs[1].set_xlim((xmin, xmax))
                    axs[1].set_ylim((ymin, ymax))
                    
                    
                if show_one_step_most_likely_positions_plots:
                    # Most likely position plots:
                    active_time_window_variable = pho_custom_decoder.active_time_window_centers
                    active_most_likely_positions_x = pho_custom_decoder.most_likely_positions[:,0].T
                    active_most_likely_positions_y = pho_custom_decoder.most_likely_positions[:,1].T
                    
                    # Enable drawing flat lines for each time bin interval instead of just displaying the single point in the middle:
                    #   build separate points for the start and end of each bin interval, and the repeat every element of the x and y values to line them up.
                    active_half_time_bin_seconds = pho_custom_decoder.time_bin_size / 2.0
                    active_time_window_start_points = np.expand_dims(pho_custom_decoder.active_time_window_centers - active_half_time_bin_seconds, axis=1)
                    active_time_window_end_points = np.expand_dims(pho_custom_decoder.active_time_window_centers + active_half_time_bin_seconds, axis=1)
                    active_time_window_start_end_points = interleave_elements(active_time_window_start_points, active_time_window_end_points) # from pyphocorehelpers.indexing_helpers import interleave_elements
                    
                    if debug_print:
                        print(f'np.shape(active_time_window_end_points): {np.shape(active_time_window_end_points)}\nnp.shape(active_time_window_start_end_points): {np.shape(active_time_window_start_end_points)}') 
                        # np.shape(active_time_window_end_points): (5783, 1)
                        # np.shape(active_time_window_start_end_points): (11566, 1)

                    active_time_window_variable = active_time_window_start_end_points
                    active_most_likely_positions_x = np.repeat(active_most_likely_positions_x, 2, axis=0) # repeat each element twice
                    active_most_likely_positions_y = np.repeat(active_most_likely_positions_y, 2, axis=0) # repeat each element twice    
                    
                    axs[0].plot(active_time_window_variable, active_most_likely_positions_x, lw=1.0, color='gray', alpha=0.4, label='1-step: most likely positions x') # (Num windows x 2)
                    # axs[0].set_title('most likely positions x')
                    axs[1].plot(active_time_window_variable, active_most_likely_positions_y, lw=1.0, color='gray', alpha=0.4, label='1-step: most likely positions y') # (Num windows x 2)
                    # axs[1].set_title('most likely positions y')
                    
                    
                fig.suptitle(f'Decoded Position data component comparison')
                return fig, axs
        
        # Call the plot function with the decoder result.
        fig, axs = plot_most_likely_position_comparsions(computation_result.computed_data['pf2D_Decoder'], computation_result.sess.position.to_dataframe(), **({'show_posterior':True, 'show_one_step_most_likely_positions_plots':True}|kwargs) )
        
        # show_two_step_most_likely_positions_plots=True
        
        active_two_step_decoder = computation_result.computed_data.get('pf2D_TwoStepDecoder', None)
        if active_two_step_decoder is not None:
            # have valid two_step_decoder, plot those predictions as well:
            # active_two_step_decoder['most_likely_positions'][:, time_window_bin_idx]
            active_time_window_variable = computation_result.computed_data['pf2D_Decoder'].active_time_window_centers
            active_most_likely_positions_x = active_two_step_decoder['most_likely_positions'][0,:]
            active_most_likely_positions_y = active_two_step_decoder['most_likely_positions'][1,:]
            two_step_options_dict = {
                'color':'#00ff7f99',
                'face_color':'#55ff0099',
                'edge_color':'#00aa0099'
            }
            # marker_style: 'circle', marker_size:0.25
            axs[0].plot(active_time_window_variable, active_most_likely_positions_x, lw=1.0, color='#00ff7f99', alpha=0.6, label='2-step: most likely positions x') # (Num windows x 2)
            # axs[0].set_title('most likely positions x')
            axs[1].plot(active_time_window_variable, active_most_likely_positions_y, lw=1.0, color='#00ff7f99', alpha=0.6, label='2-step: most likely positions y') # (Num windows x 2)
            # axs[1].set_title('most likely positions y')

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
        
        
    ## Tuning Curves 3D Plot:
    def _display_3d_interactive_tuning_curves_plotter(computation_result, active_config):
        # try: pActiveTuningCurvesPlotter
        # except NameError: pActiveTuningCurvesPlotter = None # Checks variable p's existance, and sets its value to None if it doesn't exist so it can be checked in the next step
        pActiveTuningCurvesPlotter = None
        ipcDataExplorer = InteractivePlaceCellTuningCurvesDataExplorer(active_config, computation_result.sess, computation_result.computed_data['pf2D'], active_config.plotting_config.pf_colors, extant_plotter=pActiveTuningCurvesPlotter)
        pActiveTuningCurvesPlotter = ipcDataExplorer.plot(pActiveTuningCurvesPlotter) # [2, 17449]
        ### Build Dynamic Panel Interactive Controls for configuring Placefields:
        pane = build_panel_interactive_placefield_visibility_controls(ipcDataExplorer)
        return pane
        



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
        self.register_display_function(DefaultDisplayFunctions._display_1d_placefield_validations)
        self.register_display_function(DefaultDisplayFunctions._display_2d_placefield_result_plot_raw)
        self.register_display_function(DefaultDisplayFunctions._display_2d_placefield_result_plot_ratemaps_2D)
        self.register_display_function(DefaultDisplayFunctions._display_normal)
        
        self.register_display_function(DefaultDisplayFunctions._display_decoder_result)
        self.register_display_function(DefaultDisplayFunctions._display_plot_most_likely_position_comparisons)
        
        self.register_display_function(DefaultDisplayFunctions._display_3d_interactive_tuning_curves_plotter)
        self.register_display_function(DefaultDisplayFunctions._display_3d_interactive_spike_and_behavior_browser)
        self.register_display_function(DefaultDisplayFunctions._display_3d_interactive_custom_data_explorer)
        self.register_display_function(DefaultDisplayFunctions._display_3d_image_plotter)
  
  

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
    
    
    def prepare_for_display(self, root_output_dir=r'R:\data\Output'):
        assert (self.is_computed), "Current self.is_computed must be true. Call self.perform_computations to reach this step."
        self.stage = DisplayPipelineStage(self.stage)  # build the Display stage
        # Loops through all the configs and ensure that they have the neuron identity info if they need it.
        for an_active_config_name in self.active_configs.keys():
            self.active_configs[an_active_config_name] = add_neuron_identity_info_if_needed(self.computation_results[an_active_config_name], self.active_configs[an_active_config_name])
            self.active_configs[an_active_config_name] = update_figure_files_output_Format(self.computation_results[an_active_config_name], self.active_configs[an_active_config_name], root_output_dir=root_output_dir)
            
        
    def display(self, display_function, active_session_filter_configuration: str, **kwargs):
        # active_session_filter_configuration: 'maze1'
        assert self.can_display, "Current self.stage must already be a DisplayPipelineStage. Call self.prepare_for_display to reach this step."
        if display_function is None:
            display_function = DefaultDisplayFunctions._display_normal
        return display_function(self.computation_results[active_session_filter_configuration], self.active_configs[active_session_filter_configuration], **kwargs)



    

class DisplayPipelineStage(ComputedPipelineStage):
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
        self.registered_computation_functions = computed_stage.registered_computation_functions

        # Initialize custom fields:
        self.render_actions = render_actions    
    
        
        
        