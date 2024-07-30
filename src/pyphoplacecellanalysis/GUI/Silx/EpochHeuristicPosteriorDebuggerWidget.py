from copy import deepcopy
import numpy as np
from pathlib import Path
import pandas as pd
from attrs import astuple, asdict, field, define # used in `UnpackableMixin`
from silx.gui import qt
from silx.gui.plot import Plot2D, Plot1D
from silx.gui.colors import Colormap
from silx.gui.plot.items import ImageBase
from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicScoresTuple

import numpy.ma as ma # used in `most_likely_directional_rank_order_shuffling`
from PIL import Image
from pyphocorehelpers.plotting.media_output_helpers import get_array_as_image
from scipy.signal import convolve2d

from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import neuropy.utils.type_aliases as types

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult

from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicReplayScoring, compute_local_peak_probabilities, get_peaks_mask, expand_peaks_mask, InversionCount, is_valid_sequence_index, _compute_sequences_spanning_ignored_intrusions
from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import _compute_diffusion_value, HeuristicScoresTuple


## Uses: xbin, t_start, pos_bin_size, time_bin_size

@define(slots=False)
class EpochHeuristicDebugger:
    """ 
    Displays a Silx-based heatmap that renders a 1D posterior across space and time
    
    Usage:
    
        from pyphoplacecellanalysis.GUI.Silx.EpochHeuristicPosteriorDebuggerWidget import EpochHeuristicDebugger
        
        
        a_decoder_decoded_epochs_result: DecodedFilterEpochsResult = deepcopy(directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict['long_LR'])
        
        dbgr = EpochHeuristicDebugger(p_x_given_n_masked=deepcopy(p_x_given_n_masked))
        dbgr.build_ui()

        slider = widgets.IntSlider(value=12, min=0, max=(a_decoder_decoded_epochs_result.num_filter_epochs-1))
        slider.observe(dbgr.on_slider_change, names='value')
        display(slider)
    
    """
    active_decoder_decoded_epochs_result: DecodedFilterEpochsResult = field(default=None)
    active_single_epoch_result: SingleEpochDecodedResult = field(default=None)
    p_x_given_n_masked: NDArray = field(default=None) # deepcopy(p_x_given_n_masked) # .T
    heuristic_scores: HeuristicScoresTuple = field(default=None)
    debug_print: bool = field(default=False)

    xbin: NDArray = field(default=None)
    xbin_centers: NDArray = field(default=None)
    pos_bin_size: float = field(default=None)
    time_bin_size: float = field(default=None)
    time_bin_centers: NDArray = field(default=None)

    ## Widgets/Plots:
    main_widget: qt.QWidget = field(default=None)
    main_layout: qt.QVBoxLayout = field(default=None)
    
    # a_cmap = Colormap(name="viridis", vmin=0, vmax=1)
    a_cmap: Colormap = field(factory=(lambda *args, **kwargs: Colormap(name="viridis", vmin=0))) # , vmax=1    
    plot: Plot2D = field(factory=Plot2D)
    
    plot_position: Plot1D = field(factory=Plot1D)
    plot_velocity: Plot1D = field(factory=Plot1D)
    plot_acceleration: Plot1D = field(factory=Plot1D)
    plot_extra: Plot1D = field(factory=Plot1D)



    # most_likely_position_indicies
    @property
    def active_most_likely_position_indicies(self) -> NDArray:
        """The most_likely_position_indicies property."""
        assert len(self.active_single_epoch_result.most_likely_position_indicies) == 1, f" the [0] is to handle the fact that for some reason the list is double-wrapped: [[37  0 28 52 56 28 55]]"
        return self.active_single_epoch_result.most_likely_position_indicies[0] # the [0] is to handle the fact that for some reason the list is double-wrapped: [[37  0 28 52 56 28 55]]


    def build_ui(self):
        
        ## Build Image:
        img_origin = (0.0, 0.0)
        # img_origin = (t_start, xbin[0]) # (origin X, origin Y)
        # img_origin = (xbin[0], t_start) # (origin X, origin Y)
        img_scale = (1.0, 1.0)
        # img_scale = ((1.0/(t_end - t_start)), (1.0/(xbin[-1] - xbin[0])))
        # img_scale = (pos_bin_size, time_bin_size) # ??
        # img_scale = (1.0/float(pos_bin_size), 1.0/float(time_bin_size))
        # 

        print(f'img_origin: {img_origin}')
        print(f'img_scale: {img_scale}')

        # label_kwargs = dict(xlabel='t (sec)', ylabel='x (cm)')
        label_kwargs = dict(xlabel='t (bin)', ylabel='x (bin)')
        self.plot.addImage(self.p_x_given_n_masked, legend='p_x_given_n', replace=True, colormap=self.a_cmap, origin=img_origin, scale=img_scale, **label_kwargs, resetzoom=True) # , colormap="viridis", vmin=0, vmax=1
        prev_img: ImageBase = self.plot.getImage('p_x_given_n')


        empty_arr = np.array([], dtype='int64')
        
        # Add data to the plots
        self.plot_position.addCurve(empty_arr, empty_arr, legend="Position", xlabel='t (tbin)', ylabel='x (bin)', replace=True)
        

        
        self.plot_velocity.addCurve(empty_arr, empty_arr, legend="Velocity", xlabel='t (tbin)', ylabel='velocity (bin/tbin)', replace=True)
        self.plot_acceleration.addCurve(empty_arr, empty_arr, legend="Acceleration", xlabel='t (tbin)', ylabel='accel. (bin/tbin^2)', replace=True)
        self.plot_extra.addCurve(empty_arr, empty_arr, legend="Extra", xlabel='t (tbin)', ylabel='Extra', replace=True)

        # Create a main widget and set a vertical layout
        self.main_widget = qt.QWidget()
        self.main_layout = qt.QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)

        # Add the plots to the layout
        self.main_layout.addWidget(self.plot)
        self.main_layout.addWidget(self.plot_position)
        self.main_layout.addWidget(self.plot_velocity)
        self.main_layout.addWidget(self.plot_acceleration)
        self.main_layout.addWidget(self.plot_extra)

        # Show the main widget
        self.main_widget.show()
        # self.plot.show()
        

    # def recompute(self):
    
    def update_active_epoch(self, active_epoch_idx: int):
        """ called after the time-bin is updated.
        
        TODO: this could be greatly optimized.
        
        requires: self.active_decoder_decoded_epochs_result
        
        """
        print(f'update_active_epoch(active_epoch_idx={active_epoch_idx})')
        assert self.active_decoder_decoded_epochs_result is not None
        
        # Data Update ________________________________________________________________________________________________________ #
        # active_captured_single_epoch_result = a_decoder_decoded_epochs_result.get_result_for_epoch(active_epoch_idx=active_epoch_idx) 
        # self.p_x_given_n_masked = _get_epoch_posterior(active_epoch_idx=active_epoch_idx)

        self.time_bin_size = self.active_decoder_decoded_epochs_result.decoding_time_bin_size
        self.active_single_epoch_result = self.active_decoder_decoded_epochs_result.get_result_for_epoch(active_epoch_idx=active_epoch_idx) # gets the SingleEpochDecodedResult for this epoch

        self.time_bin_centers = deepcopy(self.active_single_epoch_result.time_bin_container.centers)
        t_start, t_end = self.active_single_epoch_result.time_bin_edges[0], self.active_single_epoch_result.time_bin_edges[-1]

        p_x_given_n = deepcopy(self.active_single_epoch_result.p_x_given_n)
        most_likely_positions = deepcopy(self.active_single_epoch_result.most_likely_positions)
        most_likely_positionIndicies = deepcopy(self.active_single_epoch_result.most_likely_position_indicies)

        ## Convert from a probability matrix to a cost matrix by computing (1.0 - P), so the most probable have the lowest values
        costs_matrix = 1.0 - deepcopy(p_x_given_n)
        # costs_matrix
        uniform_diffusion_prob: float = _compute_diffusion_value(p_x_given_n) # single bin diffusion probability
        if self.debug_print:
            print(f'uniform_diffusion_prob: {uniform_diffusion_prob}')
        is_higher_than_diffusion = (p_x_given_n > uniform_diffusion_prob)

        self.p_x_given_n_masked = ma.masked_array(p_x_given_n, mask=np.logical_not(is_higher_than_diffusion), fill_value=np.nan)
        

        self.heuristic_scores = HeuristicReplayScoring.compute_pho_heuristic_replay_scores(a_result=self.active_decoder_decoded_epochs_result, an_epoch_idx=self.active_single_epoch_result.epoch_data_index, debug_print=False)
        # longest_sequence_length, longest_sequence_length_ratio, direction_change_bin_ratio, congruent_dir_bins_ratio, total_congruent_direction_change, total_variation, integral_second_derivative, stddev_of_diff, position_derivatives_df = self.heuristic_scores
        # np.diff(active_captured_single_epoch_result.most_likely_position_indicies)
        if self.debug_print:
            # print(f'heuristic_scores: {astuple(self.heuristic_scores)[:-1]}')
            print(f"heuristic_scores: {asdict(self.heuristic_scores, filter=(lambda an_attr, attr_value: an_attr.name not in ['position_derivatives_df']))}")

        # Plottings __________________________________________________________________________________________________________ #
        prev_img: ImageBase = self.plot.getImage('p_x_given_n')
        prev_img.setData(self.p_x_given_n_masked)
        # prev_img._setYLabel(f'epoch[{active_epoch_idx}: x (bin)')

        max_path = np.nanargmax(self.p_x_given_n_masked, axis=0) # returns the x-bins that maximize the path
        assert len(max_path) == len(self.time_bin_centers)
        # _curve_x = time_bin_centers
        _curve_x = np.arange(len(max_path)) + 0.5 # move forward by a half bin

        # a_track_length: float = 170.0
        # effectively_same_location_size = 0.1 * a_track_length # 10% of the track length
        # effectively_same_location_num_bins: int = np.rint(effectively_same_location_size)
        effectively_same_location_num_bins: int = 4
        _max_path_Curve = self.plot.addCurve(x=_curve_x, y=max_path, color='r', symbol='s', legend='max_path', replace=True, yerror=effectively_same_location_num_bins)
        # _max_path_Curve
        

        ## Update position plots:
        _curve_t = np.arange(len(self.active_most_likely_position_indicies)) # + 0.5 # move forward by a half bin
        pos = deepcopy(self.active_most_likely_position_indicies)
        
        _curve_vel_t = _curve_t[1:] + 0.5 # move forward by a half bin
        vel = np.diff(pos)
        _curve_accel_t = _curve_t[2:] + 0.5 # move forward by a half bin
        
        accel = np.diff(vel)
        
        if self.debug_print:
            print(f'_curve_t: {_curve_t}')
            print(f'pos: {pos}')
            print(f'vel: {vel}')
            print(f'accel: {accel}')
        
        # self.plot_position.getCurve("Position").setData(_curve_t, pos, xlabel='t (tbin)', ylabel='x (bin)')
        # self.plot_velocity.getCurve("Velocity").setData(_curve_vel_t, vel, xlabel='t (tbin)', ylabel='velocity (bin/tbin)')
        # self.plot_acceleration.getCurve("Acceleration").setData(_curve_accel_t, accel, xlabel='t (tbin)', ylabel='accel. (bin/tbin^2)')
        # self.plot_extra.getCurve("Extra").setData(_curve_accel_t, accel)

        self.plot_position.addCurve(_curve_t, pos, legend="Position", xlabel='t (tbin)', ylabel='x (bin)', replace=True)
        self.plot_velocity.addCurve(_curve_vel_t, vel, legend="Velocity", xlabel='t (tbin)', ylabel='velocity (bin/tbin)', replace=True)
        self.plot_acceleration.addCurve(_curve_accel_t, accel, legend="Acceleration", xlabel='t (tbin)', ylabel='accel. (bin/tbin^2)', replace=True)
        self.plot_extra.addCurve(_curve_accel_t, accel, legend="Extra", xlabel='t (tbin)', ylabel='Extra', replace=True)
                


    def on_slider_change(self, change):
        """Updates the active epoch via a slider:
        
        """

        # print("Slider value:", change.new)
        active_epoch_idx: int = int(change.new)
        if self.debug_print:
            print(f'epoch[{active_epoch_idx}]')
    
        self.update_active_epoch(active_epoch_idx=active_epoch_idx)
        

