from copy import deepcopy
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Neuropy:
from neuropy.core.position import build_position_df_resampled_to_time_windows # used in DecoderResultDisplayingPlot2D.setup()
from neuropy.analyses.placefields import PfND
# from neuropy.utils.debug_helpers import debug_print_placefield, debug_print_subsession_neuron_differences, debug_print_ratemap, debug_print_spike_counts, debug_plot_2d_binning, print_aligned_columns
# from neuropy.utils.debug_helpers import parameter_sweeps, _plot_parameter_sweep, compare_placefields_info
from neuropy.core.epoch import Epoch
from neuropy.utils.dynamic_container import DynamicContainer

from pyphocorehelpers.indexing_helpers import find_neighbours

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BayesianPlacemapPositionDecoder # perform_leave_one_aclu_out_decoding_analysis

# ==================================================================================================================== #
# DecoderResultDisplaying* Classes                                                                                     #
# ==================================================================================================================== #

class DecoderResultDisplayingBaseClass:
    """ Initialize by passing in a decoder of type BayesianPlacemapPositionDecoder 
    Responsible for displaying the decoded positions. 
    
    """
    def __init__(self, decoder: BayesianPlacemapPositionDecoder):
        super(DecoderResultDisplayingBaseClass, self).__init__()
        self.decoder = decoder
        self.setup()

    def setup(self):
        raise NotImplementedError
        
    def display(self, i):
        raise NotImplementedError

    # decoder properties:
    @property
    def num_time_windows(self):
        """The num_time_windows property."""
        return self.decoder.num_time_windows

    @property
    def active_time_windows(self):
        return self.decoder.active_time_windows

    @property
    def p_x_given_n(self):
        return self.decoder.p_x_given_n
    
    # placefield result variable:
    @property
    def pf(self):
        return self.decoder.pf

    # placefield properties:
    @property
    def ratemap(self):
        return self.pf.ratemap
    
    @property
    def occupancy(self):
        return self.pf.occupancy
            
    # ratemap properties (xbin & ybin)  
    @property
    def xbin(self):
        return self.ratemap.xbin
    @property
    def ybin(self):
        return self.ratemap.ybin
    @property
    def xbin_centers(self):
        return self.ratemap.xbin_centers
    @property
    def ybin_centers(self):
        return self.ratemap.ybin_centers
    
    @property
    def most_likely_positions(self):
        return self.decoder.most_likely_positions

class DecoderResultDisplayingPlot2D(DecoderResultDisplayingBaseClass):
    """ Displays the decoder for 2D position. """
    debug_print = False
    
    def __init__(self, decoder: BayesianPlacemapPositionDecoder, position_df):
        self.position_df = position_df
        super(DecoderResultDisplayingPlot2D, self).__init__(decoder)
        
    def setup(self):
        # make a new figure
        # self.fig = plt.figure(figsize=(15,15))
        self.fig, self.axs = plt.subplots(ncols=1, nrows=1, figsize=(15,15), clear=True, constrained_layout=True)
        # self.title_string = f'2D Decoded Positions'
        # self.fig.suptitle(self.title_string)
        self.index = 0
        active_window, active_p_x_given_n, active_most_likely_x_position, active_nearest_measured_position = self.get_data(self.index) # get the first item
        self.active_im = DecoderResultDisplayingPlot2D.plot_single_decoder_result(self.xbin, self.ybin, active_p_x_given_n, drop_below_threshold=None, final_string_components=[f'Decoder Result[i: {self.index}]: time window: {active_window}'], ax=self.axs)
        
        # self.active_most_likely_pos = self.axs.plot([], [], lw=2) # how to use a plot(...)
        
        if self.position_df is not None:
            # Setup data:
            self.position_df = build_position_df_resampled_to_time_windows(self.position_df, time_bin_size=self.decoder.time_bin_size)
            # Build Scatter plot when done:
            self.active_nearest_measured_pos_plot = self.axs.scatter([], [], label='actual_recorded_position', color='w')
    
        self.active_most_likely_pos_plot = self.axs.scatter([], [], label='most_likely_position', color='k') # How to initialize a scatter(...). see https://stackoverflow.com/questions/42722691/python-matplotlib-update-scatter-plot-from-a-function
        # line, = ax.plot([], [], lw=2)
        
        # self.fig.xticks()
        
        # Animation:
        # self.ani = FuncAnimation(self.fig, self.update, frames=2, interval=100, repeat=True)
        
        
    def get_data(self, window_idx):
        active_window = self.decoder.active_time_windows[window_idx] # a tuple with a start time and end time
        active_p_x_given_n = np.squeeze(self.decoder.p_x_given_n[:,:,window_idx]) # same size as occupancy
        
        active_most_likely_x_indicies = self.decoder.most_likely_position_indicies[:,window_idx]
        active_most_likely_x_position = (self.xbin_centers[active_most_likely_x_indicies[0]], self.ybin_centers[active_most_likely_x_indicies[1]])
        
        if self.position_df is not None:
            active_window_start = active_window[0]
            active_window_end = active_window[1]
            active_window_midpoint = active_window_start + ((active_window_end - active_window_start) / 2.0)
            [lowerneighbor_ind, upperneighbor_ind] = find_neighbours(active_window_midpoint, self.position_df, 't')
            active_nearest_measured_position = self.position_df.loc[lowerneighbor_ind, ['x','y']].to_numpy()
        else:
            active_nearest_measured_position = None
        
        return active_window, active_p_x_given_n, active_most_likely_x_position, active_nearest_measured_position

    @staticmethod
    def prepare_data_for_plotting(p_x_given_n, drop_below_threshold: float=0.0000001):
        curr_p_x_given_n = p_x_given_n.copy()
        if drop_below_threshold is not None:
            curr_p_x_given_n[np.where(curr_p_x_given_n < drop_below_threshold)] = np.nan # null out the p_x_given_n below certain values
            
        ## 2022-09-15 - Testing
        curr_p_x_given_n = np.nan_to_num(curr_p_x_given_n, nan=0.0, posinf=1.0, neginf=0.0) # IDK if this is smart or not
    
        ## Seems to work:
        curr_p_x_given_n = np.rot90(curr_p_x_given_n, k=-1)
        curr_p_x_given_n = np.fliplr(curr_p_x_given_n)
        return curr_p_x_given_n

    @staticmethod
    def plot_single_decoder_result(xbin, ybin, p_x_given_n, drop_below_threshold: float=0.0000001, final_string_components=[], ax=None):
        """Plots a single decoder posterior Heatmap
        """
        if ax is None:
            ax = plt.gca()

        curr_p_x_given_n = DecoderResultDisplayingPlot2D.prepare_data_for_plotting(p_x_given_n, drop_below_threshold=drop_below_threshold)
        
        """ https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html """
        xmin, xmax, ymin, ymax = (xbin[0], xbin[-1], ybin[0], ybin[-1])
        # The extent keyword arguments controls the bounding box in data coordinates that the image will fill specified as (left, right, bottom, top) in data coordinates, the origin keyword argument controls how the image fills that bounding box, and the orientation in the final rendered image is also affected by the axes limits.
        extent = (xmin, xmax, ymin, ymax)
        # print(f'extent: {extent}')
        # extent = None
        # We'll also create a black background into which the pixels will fade
        # background_black = np.full((*curr_p_x_given_n.shape, 3), 0, dtype=np.uint8)

        imshow_shared_kwargs = {
            'origin': 'lower',
            'extent': extent,
        }

        main_plot_kwargs = imshow_shared_kwargs | {
            # 'vmax': vmax,
            'vmin': 0,
            'vmax': 1,
            'cmap': 'jet',
        }

        # ax.imshow(background_black, **imshow_shared_kwargs) # add black background image
        im = ax.imshow(curr_p_x_given_n, **main_plot_kwargs) # add the curr_px_given_n image
        ax.axis("off")

        # conventional way:
        final_title = '\n'.join(final_string_components)
        ax.set_title(final_title) # f"Cell {ratemap.neuron_ids[cell]} - {ratemap.get_extended_neuron_id_string(neuron_i=cell)} \n{round(np.nanmax(pfmap),2)} Hz"

        return im

    def update(self, i):
        if DecoderResultDisplayingPlot2D.debug_print:
            print(f'update(i: {i})')
            
        self.index = i
        active_window, active_p_x_given_n, active_most_likely_x_position, active_nearest_measured_position = self.get_data(self.index)
        if DecoderResultDisplayingPlot2D.debug_print:
            print(f'active_window: {active_window}, active_p_x_given_n: {active_p_x_given_n}, active_most_likely_x_position: {active_most_likely_x_position}, active_nearest_measured_position: {active_nearest_measured_position}')
        
        # Update only:
        self.active_im.set_array(DecoderResultDisplayingPlot2D.prepare_data_for_plotting(active_p_x_given_n, drop_below_threshold=None))
        self.active_most_likely_pos_plot.set_offsets(np.c_[active_most_likely_x_position[0], active_most_likely_x_position[1]]) # method for updating a scatter_plot
        
        if self.position_df is not None:
            self.active_nearest_measured_pos_plot.set_offsets(np.c_[active_nearest_measured_position[0], active_nearest_measured_position[1]]) # method for updating a scatter_plot
        
        self.axs.set_title(f'Decoder Result[i: {self.index}]: time window: {active_window}')  # update title
        return (self.active_im, self.active_most_likely_pos_plot)
        
        
    def display(self, i):
        updated_plots_tuple = self.update(i) # calls update
        # anim = animation.FuncAnimation(figure, func=update_figure, fargs=(bar_rects, iteration), frames=generator, interval=100, repeat=False)
        # return (self.active_im,)
        return self.fig # returns fig


def perform_leave_one_aclu_out_decoding_analysis(spikes_df, active_pos_df, active_filter_epochs, original_all_included_decoder=None, filter_epoch_description_list=None, decoding_time_bin_size=0.025):
    """2023-03-03 - Performs a "leave-one-out" decoding analysis where we leave out each neuron one at a time and see how the decoding degrades (which serves as an indicator of the importance of that neuron on the decoding performance).

    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import perform_leave_one_aclu_out_decoding_analysis
    """

    def _build_one_left_out_decoders(original_all_included_decoder):
        """ "Leave-one-out" decoding
        WARNING: this might suck up a ton of memory! 
        """
        original_neuron_ids = np.array(original_all_included_decoder.pf.ratemap.neuron_ids) # original_pf.included_neuron_IDs
        one_left_out_decoder_dict = {}
        for aclu_to_omit in original_neuron_ids:
            subset_included_neuron_ids = np.array([aclu for aclu in original_neuron_ids if aclu != aclu_to_omit]) # get all but the omitted neuron
            one_left_out_decoder_dict[aclu_to_omit] = original_all_included_decoder.get_by_id(subset_included_neuron_ids, defer_compute_all=True) # skip computations
            
        return one_left_out_decoder_dict


    spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_type('pyramidal') ## get only the pyramidal spikes
    active_pos = active_pos_df.position.to_Position_obj() # convert back to a full position object
 
    ## Build placefield for the decoder to use:
    if original_all_included_decoder is None:
        original_decoder_pf1D = PfND(deepcopy(spikes_df), deepcopy(active_pos.linear_pos_obj)) # all other settings default
        ## Build the new decoder:
        original_all_included_decoder = BayesianPlacemapPositionDecoder(decoding_time_bin_size, original_decoder_pf1D, original_decoder_pf1D.filtered_spikes_df.copy(), debug_print=False)
    else:
        print(f'USING EXISTING original_1D_decoder.')

    all_included_filter_epochs_decoder_result = original_all_included_decoder.decode_specific_epochs(spikes_df, filter_epochs=active_filter_epochs, decoding_time_bin_size=decoding_time_bin_size, debug_print=False)

    # pretty dang inefficient, as there are 70 cells:
    one_left_out_decoder_dict = _build_one_left_out_decoders(original_all_included_decoder)
    ## `decode_specific_epochs` for each of the decoders:
    one_left_out_filter_epochs_decoder_result_dict = {}
    ### Loop through and perform the decoding for each epoch. This is the slow part.
    for left_out_aclu, curr_aclu_omitted_decoder in one_left_out_decoder_dict.items():
        filter_epochs_decoder_result = curr_aclu_omitted_decoder.decode_specific_epochs(spikes_df, filter_epochs=active_filter_epochs, decoding_time_bin_size=decoding_time_bin_size, debug_print=False)
        # if filter_epoch_description_list is None:
        #     filter_epoch_description_list = [f'Epoch {i}' for i in range(len(filter_epochs_decoder_result.epoch_description_list))]
        filter_epochs_decoder_result.epoch_description_list = deepcopy(filter_epoch_description_list) # PLOT_ONLY
        one_left_out_filter_epochs_decoder_result_dict[left_out_aclu] = filter_epochs_decoder_result

    """ Returns:
        original_1D_decoder: original decoder with all aclu values included
        all_included_filter_epochs_decoder_result: the decoder result for the original decoder with all aclu values included
        one_left_out_decoder_dict: a dictionary of decoders, where each decoder has one less aclu than the original decoder. The key is the aclu that was omitted from the decoder.
        one_left_out_filter_epochs_decoder_result_dict: a dictionary of decoder results for each of the decoders in one_left_out_decoder_dict. The key is the aclu that was omitted from the decoder.
    """
    return original_all_included_decoder, all_included_filter_epochs_decoder_result, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict


# ==================================================================================================================== #
# 2023-03-17 Surprise Analysis                                                                                         #
# ==================================================================================================================== #
from attrs import define, field
# import cattrs 

@define
class SurpriseAnalysisResult:
    """ 

    Built with:
        from pyphocorehelpers.general_helpers import GeneratedClassDefinitionType, CodeConversion
        CodeConversion.convert_dictionary_to_class_defn(long_results_dict, class_name='SurpriseAnalysisResult', class_definition_mode=GeneratedClassDefinitionType.DATACLASS)
    n_neurons, n_epochs * n_timebins_for_epoch_i
    {'n_neurons':67, 'n_epochs':625, 'n_total_time_bins':6855}
    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import SurpriseAnalysisResult
    """
    active_filter_epochs: Epoch
    original_1D_decoder: BayesianPlacemapPositionDecoder
    all_included_filter_epochs_decoder_result: DynamicContainer
    flat_all_epochs_measured_cell_spike_counts: np.ndarray = field(metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    flat_all_epochs_measured_cell_firing_rates: np.ndarray = field(metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    flat_all_epochs_decoded_epoch_time_bins: np.ndarray = field(metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    flat_all_epochs_computed_surprises: np.ndarray = field(metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    flat_all_epochs_computed_expected_cell_firing_rates: np.ndarray = field(metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    flat_all_epochs_difference_from_expected_cell_spike_counts: np.ndarray = field(metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    flat_all_epochs_difference_from_expected_cell_firing_rates: np.ndarray = field(metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    all_epochs_decoded_epoch_time_bins_mean: np.ndarray = field(metadata={'shape': ('n_epochs', 'n_neurons')})
    all_epochs_computed_cell_surprises_mean: np.ndarray = field(metadata={'shape': ('n_epochs', 'n_neurons')})
    all_epochs_all_cells_computed_surprises_mean: np.ndarray = field(metadata={'shape': ('n_epochs',)})
    flat_all_epochs_computed_one_left_out_to_global_surprises: np.ndarray = field(metadata={'shape': ('n_neurons', 'n_total_time_bins')})
    all_epochs_computed_cell_one_left_out_to_global_surprises_mean: np.ndarray = field(metadata={'shape': ('n_epochs', 'n_neurons')})
    all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean: np.ndarray = field(metadata={'shape': ('n_epochs',)})
    one_left_out_omitted_aclu_distance_df: pd.core.frame.DataFrame = field(metadata={'shape': ('n_neurons', 3)})
    most_contributing_aclus: np.ndarray = field(metadata={'shape': ('n_neurons',)})

    # def __attrs_post_init__(self):
    #     self.z = self.x + self.y

    # def sliced_by_aclus(self, aclus):
    #     """ returns a copy of itself sliced by the aclus provided. """
    #     from attrs import asdict, fields, evolve
    #     aclu_is_included = np.isin(self.original_1D_decoder.neuron_IDs, aclus)  #.shape # (104, 63)
    #     def _filter_obj_attribute(an_attr, attr_value):
    #         """ return attributes only if they have n_neurons in their shape metadata """
    #         return ('n_neurons' in an_attr.metadata.get('shape', ()))            
    #     _temp_obj_dict = asdict(self, filter=_filter_obj_attribute)
    #     # Find all fields that contain a 'n_neurons':
    #     neuron_indexed_attributes = [a_field for a_field in fields(type(self)) if ('n_neurons' in a_field.metadata.get('shape', ()))]
    #     # neuron_shape_index_for_attributes = [a_field.metadata['shape'].index('n_neurons') for a_field in neuron_indexed_attributes]
    #     neuron_shape_index_for_attribute_name_dict = {a_field.name:a_field.metadata['shape'].index('n_neurons') for a_field in neuron_indexed_attributes} # need the actual attributes so that we can get the .metadata['shape'] from them and find the n_neurons index location
    #     _temp_obj_dict = {k:v.take(indices=aclu_is_included, axis=neuron_shape_index_for_attribute_name_dict[k]) for k, v in _temp_obj_dict.items()} # filter the n_neurons axis containing items to get a reduced dictionary
    #     return evolve(self, **_temp_obj_dict)



    # @staticmethod
    # def _build_results_dict(a_results_tuple):
    #     active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, flat_all_epochs_measured_cell_spike_counts, flat_all_epochs_measured_cell_firing_rates, flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates, flat_all_epochs_difference_from_expected_cell_spike_counts, flat_all_epochs_difference_from_expected_cell_firing_rates, all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_all_cells_computed_surprises_mean = a_results_tuple
    #     return {'active_filter_epochs':active_filter_epochs, 'original_1D_decoder':original_1D_decoder, 'all_included_filter_epochs_decoder_result':all_included_filter_epochs_decoder_result, 'flat_all_epochs_measured_cell_spike_counts':flat_all_epochs_measured_cell_spike_counts, 'flat_all_epochs_measured_cell_firing_rates':flat_all_epochs_measured_cell_firing_rates, 'flat_all_epochs_decoded_epoch_time_bins':flat_all_epochs_decoded_epoch_time_bins, 'flat_all_epochs_computed_surprises':flat_all_epochs_computed_surprises, 'flat_all_epochs_computed_expected_cell_firing_rates':flat_all_epochs_computed_expected_cell_firing_rates, 'flat_all_epochs_difference_from_expected_cell_spike_counts':flat_all_epochs_difference_from_expected_cell_spike_counts, 'flat_all_epochs_difference_from_expected_cell_firing_rates':flat_all_epochs_difference_from_expected_cell_firing_rates, 'all_epochs_decoded_epoch_time_bins_mean':all_epochs_decoded_epoch_time_bins_mean, 'all_epochs_computed_cell_surprises_mean':all_epochs_computed_cell_surprises_mean, 'all_epochs_all_cells_computed_surprises_mean':all_epochs_all_cells_computed_surprises_mean}


def perform_full_session_leave_one_out_decoding_analysis(sess, original_1D_decoder=None, decoding_time_bin_size = 0.02, cache_suffix = ''):
    """ 2023-03-17 - Performs a full session leave one out decoding analysis.

    Args:
        sess: a Session object
        original_1D_decoder: an optional `BayesianPlacemapPositionDecoder` object to decode with. If None, the decoder will be created from the session. If you pass a decoder, you want to provide the global session and not a particular filtered one.
        decoding_time_bin_size: the time bin size for the decoder
        cache_suffix: a suffix to add to the cache file name, or None to not cache
    Returns:
        decoder_result: the decoder result for the original decoder with all aclu values included

    Calls:
        perform_leave_one_aclu_out_decoding_analysis(...)
        _subfn_compute_leave_one_out_analysis(...)


    Usage:
        from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import perform_full_session_leave_one_out_decoding_analysis
    """
    from neuropy.core.epoch import Epoch
    # for caching/saving:
    from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import loadData, saveData
    from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import _analyze_leave_one_out_decoding_results

    if cache_suffix is not None:
        ### Build a folder to store the temporary outputs:
        output_data_folder = sess.get_output_path()

    ## Get testing variables from `sess`
    spikes_df = sess.spikes_df
    pyramidal_only_spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_type('pyramidal') ## get only the pyramidal spikes
    active_pos = sess.position
    active_pos_df = active_pos.to_dataframe()

    # --------------
    # ### active_filter_epochs: sess.laps: 
    # laps_copy = deepcopy(sess.laps)
    # active_filter_epochs = laps_copy.filtered_by_lap_flat_index(np.arange(20)).as_epoch_obj() # epoch object

    ### active_filter_epochs: sess.replay: 
    active_filter_epochs = sess.replay.epochs.get_valid_df().epochs.get_epochs_longer_than(minimum_duration=5.0*decoding_time_bin_size).epochs.get_non_overlapping_df()
    if not 'stop' in active_filter_epochs.columns:
        # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
        active_filter_epochs['stop'] = active_filter_epochs['end'].copy()
    if not 'label' in active_filter_epochs.columns:
        # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
        active_filter_epochs['label'] = active_filter_epochs['flat_replay_idx'].copy()
    active_filter_epochs = Epoch(active_filter_epochs)

    if original_1D_decoder is None:
        ## Build the new decoder (if not provided):
        active_pos = active_pos_df.position.to_Position_obj() # convert back to a full position object
        original_decoder_pf1D = PfND(deepcopy(pyramidal_only_spikes_df), deepcopy(active_pos.linear_pos_obj)) # all other settings default
        ## Build the new decoder:
        original_1D_decoder = BayesianPlacemapPositionDecoder(decoding_time_bin_size, original_decoder_pf1D, original_decoder_pf1D.filtered_spikes_df.copy(), debug_print=False)
    else:
        print(f'reusing extant decoder.')

    # -- Part 1 -- perform the decoding:
    original_1D_decoder, all_included_filter_epochs_decoder_result, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict = perform_leave_one_aclu_out_decoding_analysis(pyramidal_only_spikes_df, active_pos_df, active_filter_epochs, original_all_included_decoder=original_1D_decoder, decoding_time_bin_size=decoding_time_bin_size)
    # one_left_out_omitted_aclu_distance_df, most_contributing_aclus = _temp_analyze(active_pos_df, one_left_out_filter_epochs_decoder_result_dict)


    # Save to file:
    if cache_suffix is not None:
        leave_one_out_result_pickle_path = output_data_folder.joinpath(f'leave_one_out_results{cache_suffix}.pkl').resolve()
        print(f'leave_one_out_result_pickle_path: {leave_one_out_result_pickle_path}')
        saveData(leave_one_out_result_pickle_path, (active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict))

    # -- Part 2 -- perform the analysis on the decoder results:
    flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates, flat_all_epochs_computed_one_left_out_to_global_surprises, all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_computed_cell_one_left_out_to_global_surprises_mean, all_epochs_all_cells_computed_surprises_mean, all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean, one_left_out_omitted_aclu_distance_df, most_contributing_aclus = _analyze_leave_one_out_decoding_results(active_pos_df, active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, one_left_out_decoder_dict, one_left_out_filter_epochs_decoder_result_dict)

    ## Flatten the measured spike counts over the time bins within all epochs to get something of the same shape as `flat_all_epochs_decoded_epoch_time_bins`:
    flat_all_epochs_measured_cell_spike_counts = np.hstack(all_included_filter_epochs_decoder_result.spkcount) # .shape (65, 4584) -- (n_neurons, n_epochs * n_timebins_for_epoch_i), combines across all time_bins within all epochs
    assert flat_all_epochs_computed_expected_cell_firing_rates.shape == flat_all_epochs_measured_cell_spike_counts.shape, f"{flat_all_epochs_measured_cell_spike_counts.shape = } != {flat_all_epochs_computed_expected_cell_firing_rates.shape =}"
    ## Get the time bins where each cell is firing (has more than one spike):
    is_cell_firing_time_bin = (flat_all_epochs_measured_cell_spike_counts > 0)

    ## Convert spike counts to firing rates by dividing by the time bin size:
    flat_all_epochs_measured_cell_firing_rates = flat_all_epochs_measured_cell_spike_counts / original_1D_decoder.time_bin_size
    ## Convert the expected firing rates to spike counts by multiplying by the time bin size (NOTE: there can be fractional expected spikes):
    flat_all_epochs_computed_expected_cell_spike_counts = flat_all_epochs_computed_expected_cell_firing_rates * original_1D_decoder.time_bin_size

    ## Compute the difference from the expected firing rate observed for each cell (in each time bin):
    flat_all_epochs_difference_from_expected_cell_spike_counts = flat_all_epochs_computed_expected_cell_spike_counts - flat_all_epochs_measured_cell_spike_counts
    flat_all_epochs_difference_from_expected_cell_firing_rates = flat_all_epochs_computed_expected_cell_firing_rates - flat_all_epochs_measured_cell_firing_rates
    # flat_all_epochs_difference_from_expected_cell_firing_rates

    if cache_suffix is not None:
        # Save to file to cache in case we crash:
        leave_one_out_surprise_result_pickle_path = output_data_folder.joinpath(f'leave_one_out_surprise_results{cache_suffix}.pkl').resolve()
        print(f'leave_one_out_surprise_result_pickle_path: {leave_one_out_surprise_result_pickle_path}')
        saveData(leave_one_out_surprise_result_pickle_path, (active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, 
                                                            flat_all_epochs_measured_cell_spike_counts, flat_all_epochs_measured_cell_firing_rates, 
                                                            flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates,
                                                            flat_all_epochs_difference_from_expected_cell_spike_counts, flat_all_epochs_difference_from_expected_cell_firing_rates,
                                                            all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_all_cells_computed_surprises_mean,
                                                            flat_all_epochs_computed_one_left_out_to_global_surprises, all_epochs_computed_cell_one_left_out_to_global_surprises_mean, all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean,
                                                            one_left_out_omitted_aclu_distance_df, most_contributing_aclus))


    # (active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, flat_all_epochs_measured_cell_spike_counts, flat_all_epochs_measured_cell_firing_rates, flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates, flat_all_epochs_difference_from_expected_cell_spike_counts, flat_all_epochs_difference_from_expected_cell_firing_rates, all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_all_cells_computed_surprises_mean, flat_all_epochs_computed_one_left_out_to_global_surprises, all_epochs_computed_cell_one_left_out_to_global_surprises_mean, all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean, one_left_out_omitted_aclu_distance_df, most_contributing_aclus)


    return (active_filter_epochs, original_1D_decoder, all_included_filter_epochs_decoder_result, 
                                                            flat_all_epochs_measured_cell_spike_counts, flat_all_epochs_measured_cell_firing_rates, 
                                                            flat_all_epochs_decoded_epoch_time_bins, flat_all_epochs_computed_surprises, flat_all_epochs_computed_expected_cell_firing_rates,
                                                            flat_all_epochs_difference_from_expected_cell_spike_counts, flat_all_epochs_difference_from_expected_cell_firing_rates,
                                                            all_epochs_decoded_epoch_time_bins_mean, all_epochs_computed_cell_surprises_mean, all_epochs_all_cells_computed_surprises_mean,
                                                            flat_all_epochs_computed_one_left_out_to_global_surprises, all_epochs_computed_cell_one_left_out_to_global_surprises_mean, all_epochs_all_cells_computed_one_left_out_to_global_surprises_mean,
                                                            one_left_out_omitted_aclu_distance_df, most_contributing_aclus)