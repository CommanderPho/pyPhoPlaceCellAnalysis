from copy import deepcopy
import numpy as np
from pathlib import Path
import pandas as pd
from functools import partial
from attrs import astuple, asdict, field, define, Factory # used in `UnpackableMixin`
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

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DecoderDecodedEpochsResult

from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import HeuristicReplayScoring, compute_local_peak_probabilities, get_peaks_mask, expand_peaks_mask, InversionCount, is_valid_sequence_index, _compute_sequences_spanning_ignored_intrusions
from pyphoplacecellanalysis.Analysis.Decoder.heuristic_replay_scoring import _compute_diffusion_value, HeuristicScoresTuple

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.GUI.Qt.Widgets.ScrollBarWithSpinBox.ScrollBarWithSpinBox import ScrollBarWithSpinBox
from pyphocorehelpers.gui.Qt.pandas_model import SimplePandasModel, create_tabbed_table_widget
from pyphoplacecellanalysis.GUI.Qt.Widgets.LogViewerTextEdit import LogViewer
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons
from pyphoplacecellanalysis.Resources.icon_helpers import try_get_icon
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer


from pyphoplacecellanalysis.Analysis.position_derivatives import PositionDerivativesContainer

        
@function_attributes(short_name=None, tags=['Silx', 'grid', 'matrix'], input_requires=[], output_provides=[], uses=[], used_by=['EpochHeuristicDebugger'], creation_date='2024-08-13 06:06', related_items=[])
def setup_plot_grid_ticks(a_plot: Union[Plot1D, Plot2D], minor_ticks:bool=False):
    """ Updates the grid-size for the rendered grid:
    Requires that Silx be using a matpltolib-based backend
    
    Usage:
    
        from pyphoplacecellanalysis.GUI.Silx.EpochHeuristicPosteriorDebuggerWidget import setup_plot_grid_ticks

        # x_ticks_obj_list, y_ticks_obj_list = setup_plot_grid_ticks(a_plot=dbgr.plot)
        x_ticks_obj_list, y_ticks_obj_list = setup_plot_grid_ticks(a_plot=dbgr.plot_position, minor_ticks=False)

    """
    pos_x_range = a_plot.getXAxis().getLimits()
    pos_y_range = a_plot.getYAxis().getLimits()

    pos_x_range = (int(pos_x_range[0]), int(pos_x_range[1]))
    pos_y_range = (int(pos_y_range[0]), int(pos_y_range[1]))

    x_ticks = np.arange(pos_x_range[0], pos_x_range[-1], 1)
    y_ticks = np.arange(pos_y_range[0], pos_y_range[-1], 1)

    an_ax = a_plot.getBackend().ax # matplotlib ax (matplotlib.axes._axes.Axes)
    x_ticks_obj_list: List = an_ax.set_xticks(x_ticks, minor=minor_ticks) # List[matplotlib.axis.XTick]
    y_ticks_obj_list: List = an_ax.set_yticks(y_ticks, minor=minor_ticks) # List[matplotlib.axis.YTick]

    a_plot.setGraphGrid(which='major')
    return x_ticks_obj_list, y_ticks_obj_list


@function_attributes(short_name=None, tags=['Silx', 'remove', 'toolbar', 'gui'], input_requires=[], output_provides=[], uses=[], used_by=['EpochHeuristicDebugger'], creation_date='2024-08-13 06:07', related_items=[])
def remove_all_plot_toolbars(a_plot: Union[Plot1D, Plot2D]):
    """ removes the default plot-customization toolbars from the Plot*Ds """
    _plot_toolbars = [a_plot.toolBar(), a_plot.getOutputToolBar(), a_plot.getInteractiveModeToolBar()]
    for a_toolbar in _plot_toolbars:
        a_plot.removeToolBar(a_toolbar)



## Uses: xbin, t_start, pos_bin_size, time_bin_size

# Define the partial function above the class
plot1d_factory = partial(Plot1D)
# plot1d_factory = partial(Plot1D, toolbar=False)


@metadata_attributes(short_name=None, tags=['Silx', 'gui'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-13 06:06', related_items=[])
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
    bin_by_bin_heuristic_scores: Dict = field(default=Factory(dict))
    debug_print: bool = field(default=False)

    xbin: NDArray = field(default=None)
    xbin_centers: NDArray = field(default=None)
    pos_bin_size: float = field(default=None)
    time_bin_size: float = field(default=None)
    decoder_track_length: float = field(default=None)
    time_bin_centers: NDArray = field(default=None)

    position_derivatives: PositionDerivativesContainer = field(default=None)    

    ## Widgets/Plots:
    ui: PhoUIContainer = field(default=None)    
    main_widget: qt.QWidget = field(default=None)
    main_layout: qt.QVBoxLayout = field(default=None)
    
    # a_cmap = Colormap(name="viridis", vmin=0, vmax=1)
    a_cmap: Colormap = field(factory=(lambda *args, **kwargs: Colormap(name="viridis", vmin=0))) # , vmax=1    
    plot: Plot2D = field(factory=Plot2D)
    
    plot_position: Plot1D = field(factory=plot1d_factory)
    plot_velocity: Plot1D = field(factory=plot1d_factory)
    plot_acceleration: Plot1D = field(factory=plot1d_factory)
    plot_extra: Plot1D = field(factory=plot1d_factory)
    

    programmatic_plots_config_dict: Dict[str, Dict] = field(default=Factory(dict)) ## empty dict by default
    programmatic_plots_dict: Dict[str, Plot1D] = field(default=Factory(dict)) ## empty dict by default
    
    
    use_bin_units_instead_of_realworld: bool = field(default=True, metadata={'notes': 'if False, uses the real-world units (cm/seconds). If True, uses nbin units (n_posbins/n_timebins)'})
    

    # Computed Properties ________________________________________________________________________________________________ #
    @property
    def n_epochs(self) -> int:
        return self.active_decoder_decoded_epochs_result.num_filter_epochs
    

    @property
    def active_epoch_index(self) -> int:
        return self.active_single_epoch_result.epoch_data_index
    
        
    @property
    def filter_epochs(self) -> pd.DataFrame:
        return self.active_decoder_decoded_epochs_result.filter_epochs
    
    # @property
    # def active_most_likely_position_indicies(self) -> NDArray:
    #     """The most_likely_position_indicies property."""
    #     assert len(self.active_single_epoch_result.most_likely_position_indicies) == 1, f" for some reason the list should be double-wrapped: [[37  0 28 52 56 28 55]], meaning it has length 1"
    #     return self.active_single_epoch_result.most_likely_position_indicies[0] # the [0] is to handle the fact that for some reason the list is double-wrapped: [[37  0 28 52 56 28 55]]

    @property
    def active_most_likely_position_arr(self) -> NDArray:
        """The most_likely_position_indicies property."""
        if self.use_bin_units_instead_of_realworld:
            ## bin units
            assert len(self.active_single_epoch_result.most_likely_position_indicies) == 1, f" for some reason the list should be double-wrapped: [[37  0 28 52 56 28 55]], meaning it has length 1"
            active_active_most_likely_position_arr = deepcopy(self.active_single_epoch_result.most_likely_position_indicies[0]) # the [0] is to handle the fact that for some reason the list is double-wrapped: [[37  0 28 52 56 28 55]]
            # active_active_most_likely_position_arr = deepcopy(self.active_most_likely_position_indicies)
        else:
            ## real-world units
            # assert len(self.active_single_epoch_result.most_likely_positions) == 1, f" for some reason the list should be double-wrapped: [[37  0 28 52 56 28 55]], meaning it has length 1"
            # active_active_most_likely_position_arr = deepcopy(self.active_single_epoch_result.most_likely_positions[0]) # the [0] is to handle the fact that for some reason the list is double-wrapped: [[37  0 28 52 56 28 55]]
            active_active_most_likely_position_arr = deepcopy(self.active_single_epoch_result.most_likely_positions)            

        # assert len(active_active_most_likely_position_arr) == 1, f" the [0] is to handle the fact that for some reason the list is double-wrapped: [[37  0 28 52 56 28 55]]"
        return active_active_most_likely_position_arr




    @property
    def plot_configs_dict(self) -> Dict[str, Dict]:
        """ convenince access to the dict of position plots 
        """
        common_plot_config_dict = dict(symbol='o', linestyle=':', color=(0.0, 0.0, 1.0, 0.2,)) # , fillColor="rgba(0, 0, 255, 50)"

        return {"Position": dict(legend="Position", xlabel='t (tbin)',
                                #   ylabel='x_pos (bin)',
                                ylabel='x_pos (cm)',
                                **common_plot_config_dict),
            "Velocity": dict(legend="Velocity", xlabel='t (tbin)', ylabel='velocity (bin/tbin)', baseline=0.0, fill=True, **common_plot_config_dict),
            "Acceleration": dict(legend="Acceleration", xlabel='t (tbin)', ylabel='accel. (bin/tbin^2)', baseline=0.0, fill=True, **common_plot_config_dict),
            "Extra": dict(legend="Extra", xlabel='t (tbin)', ylabel='Extra', baseline=0.0, fill=True, **common_plot_config_dict),
        }
        
    @property
    def position_plots_dict(self) -> Dict[str, Plot1D]:
        """ convenince access to the dict of position plots 
        """
        position_plots_list = [self.plot_position, self.plot_velocity, self.plot_acceleration, self.plot_extra]
        return dict(zip(['Position', 'Velocity', 'Acceleration', 'Extra'], position_plots_list))


    @property
    def plot_configs_dict(self) -> Dict[str, Dict]:
        """ convenince access to the dict of position plots 
        
        dict(legend="Extra", xlabel='t (tbin)', ylabel='Extra', baseline=0.0, fill=True, symbol='o', linestyle=':', color=(0.0, 0.0, 1.0, 0.2,))
        
        
        """
        common_plot_config_dict = dict(symbol='o', linestyle=':', color=(0.0, 0.0, 1.0, 0.2,)) # , fillColor="rgba(0, 0, 255, 50)"

        return {"Position": dict(legend="Position", xlabel='t (tbin)',
                                #   ylabel='x_pos (bin)',
                                ylabel='x_pos (cm)',
                                is_visible=True,
                                **common_plot_config_dict),
            "Velocity": dict(legend="Velocity", xlabel='t (tbin)', ylabel='velocity (bin/tbin)', baseline=0.0, fill=True, is_visible=False, **common_plot_config_dict),
            "Acceleration": dict(legend="Acceleration", xlabel='t (tbin)', ylabel='accel. (bin/tbin^2)', baseline=0.0, fill=True, is_visible=False, **common_plot_config_dict),
            "Extra": dict(legend="Extra", xlabel='t (tbin)', ylabel='Extra', baseline=0.0, fill=True, is_visible=False, **common_plot_config_dict),
        }
    


    @classmethod
    def init_from_epoch_idx(cls, a_decoder_decoded_epochs_result: DecodedFilterEpochsResult, active_epoch_idx: int=0, **kwargs) -> "EpochHeuristicDebugger":
        """ initializes to a specific epoch_idx
        
        """        
        _obj = cls(active_decoder_decoded_epochs_result=deepcopy(a_decoder_decoded_epochs_result), **kwargs)
        if _obj.active_single_epoch_result is None:
            active_captured_single_epoch_result: SingleEpochDecodedResult = a_decoder_decoded_epochs_result.get_result_for_epoch(active_epoch_idx=active_epoch_idx)
            _obj.active_single_epoch_result = deepcopy(active_captured_single_epoch_result)

        if _obj.time_bin_size is None:
             _obj.time_bin_size = a_decoder_decoded_epochs_result.decoding_time_bin_size
             
        if _obj.time_bin_centers is None:
             _obj.time_bin_centers = deepcopy(_obj.active_single_epoch_result.time_bin_container.centers)
               
        _obj.update_active_epoch_data(active_epoch_idx=active_epoch_idx)
        _obj.build_ui()
        _obj.update_active_epoch(active_epoch_idx=active_epoch_idx)
        return _obj
        

    def _build_image_scale_and_origin(self):
        """ updates: self.plot,
        
        """
        n_global_xbin_pos_bins: int = len(self.xbin_centers)
        print(f'n_global_xbin_pos_bins: {n_global_xbin_pos_bins}')
        xmin, xmax = self.xbin[0], self.xbin[-1]
        n_pos_bins, n_tbins = np.shape(self.p_x_given_n_masked)
        print(f'n_tbins: {n_tbins}, n_pos_bins: {n_pos_bins}')
        assert n_global_xbin_pos_bins == n_pos_bins, f"n_global_xbin_pos_bins: {n_global_xbin_pos_bins} != n_pos_bins: {n_pos_bins} but it should!"
        
        ## Build Image:
        if self.use_bin_units_instead_of_realworld:
            img_origin = (0.0, 0.0)
            img_scale = (1.0, 1.0)
            img_bounds = ((0, n_tbins,), (0, n_pos_bins,))
        else:
            x_range: float = xmax - xmin
            img_height_scale: float = x_range / float(n_pos_bins) # nope
            # img_height_scale: float = float(n_pos_bins) / x_range # nope
            print(f'x_range: {x_range}, img_height_scale: {img_height_scale}')
            # img_origin = (0.0, 0.0)
            img_origin = (0.0, xmin) # start at height xmin
            # img_scale = (1.0, 1.0) # height should be x_range
            img_scale = (1.0, img_height_scale) # height should be x_range
            img_bounds = ((0, n_tbins,), (xmin, xmax,))
            
        print(f'img_origin: {img_origin}')
        print(f'img_scale: {img_scale}')
        print(f'img_bounds: {img_bounds}')
        return img_scale, img_origin, img_bounds
    

    @function_attributes(short_name=None, tags=['init', 'plots'], input_requires=[], output_provides=[], uses=['_build_image_scale_and_origin'], used_by=[], creation_date='2024-11-25 18:54', related_items=[])
    def _build_plots(self):
        """ updates: self.plot,
        
        """
        xmin, xmax = self.xbin[0], self.xbin[-1]
        img_scale, img_origin, img_bounds = self._build_image_scale_and_origin()        
        if self.use_bin_units_instead_of_realworld:
            ## bin units
            label_kwargs = dict(xlabel='t (tbin)', ylabel='x_pos (bin)')

        else:
            ## real-world units    
            label_kwargs = dict(xlabel='t (sec)', ylabel='x (cm)')

        empty_arr = np.array([], dtype='int64')
        
        def _subfn_helper_setup_new_plot(a_plot_name: str, a_plot, a_plot_config_dict):
            """ captures: empty_arr, xmin, xmax
            
            """
            is_visible: bool = a_plot_config_dict.pop('is_visible', True)
            remove_all_plot_toolbars(a_plot)
            ## add curves
            a_plot.addCurve(empty_arr, empty_arr, **a_plot_config_dict, replace=True)            
            ## Update plot properties:
            a_plot.setActiveCurve(a_plot_name)
            a_plot.setGraphGrid(which=True) # good
            a_plot.getXAxis().setLabel(a_plot_config_dict["xlabel"])
            a_plot.getYAxis().setLabel(a_plot_config_dict["ylabel"])
            a_plot.setXAxisAutoScale(flag=False) # then turn off
            if a_plot_name == 'Position':
                a_plot.setYAxisAutoScale(flag=False) # position y-axis is fixed to the total bins
                if not self.use_bin_units_instead_of_realworld:
                    a_plot.getYAxis().setLimits(xmin, xmax)

            else:
                a_plot.setYAxisAutoScale(flag=True)
            a_plot.setHidden((not is_visible))


        self.plot.addImage(self.p_x_given_n_masked, legend='p_x_given_n', replace=True, colormap=self.a_cmap, origin=img_origin, scale=img_scale, **label_kwargs, resetzoom=True) # , colormap="viridis", vmin=0, vmax=1
        pos_x_range = self.plot.getXAxis().getLimits()
        pos_y_range = self.plot.getYAxis().getLimits()
        curr_img_bounds = (pos_x_range, pos_y_range)
        print(f'curr_img_bounds: {curr_img_bounds}')

        self.plot.getXAxis().setLimits(img_bounds[0][0], img_bounds[0][1])
        self.plot.getYAxis().setLimits(img_bounds[1][0], img_bounds[1][1])

        prev_img: ImageBase = self.plot.getImage('p_x_given_n')
        ## Setup grid:            
        pos_x_range = self.plot.getXAxis().getLimits()
        pos_y_range = self.plot.getYAxis().getLimits()
        curr_img_bounds = (pos_x_range, pos_y_range)
        print(f'curr_img_bounds: {curr_img_bounds}')
        
        # pos_x_range = (int(pos_x_range[0]), int(pos_x_range[1]))
        # pos_y_range = (int(pos_y_range[0]), int(pos_y_range[1]))
        self.plot.setGraphGrid(which=True)
        # remove_all_plot_toolbars(self.plot)

        # Position Derivative Plots:

        # Position plots specifically: _______________________________________________________________________________________ #
        plot_configs_dict = self.plot_configs_dict
        position_plots_list = [self.plot_position, self.plot_velocity, self.plot_acceleration, self.plot_extra]
        position_plots_dict = dict(zip(list(plot_configs_dict.keys()), position_plots_list))

        # Add data to the plots:
        for a_plot_name, a_plot in position_plots_dict.items():
            a_plot_config_dict = plot_configs_dict[a_plot_name]
            _subfn_helper_setup_new_plot(a_plot_name=a_plot_name, a_plot=a_plot, a_plot_config_dict=a_plot_config_dict)


        # Extra Custom/Programmatic Plots ____________________________________________________________________________________ #
        # self.bin_by_bin_heuristic_scores = {} # new array
        # self.programmatic_plots_config_dict = {}
        self.programmatic_plots_dict = {}
        for k, a_bin_by_bin_values in self.bin_by_bin_heuristic_scores.items():
            a_plot = self.programmatic_plots_dict.get(k, None)
            a_plot_config_dict = self.programmatic_plots_config_dict.get(k, None)
            if a_plot_config_dict is None:
                # build a new one
                a_plot_config_dict = dict(legend=f"{k}", xlabel='t (tbin)', ylabel=f"{k}", baseline=0.0, fill=True, is_visible=True, symbol='o', linestyle=':', color=(0.0, 0.0, 1.0, 0.2,))
                self.programmatic_plots_config_dict[k] = deepcopy(a_plot_config_dict) # store the config dict after creation
            if a_plot is None:
                # build a new plot
                a_plot = Plot1D() ## new plot
                self.programmatic_plots_dict[k] = a_plot # stor ethe plot in the dict after creation
            assert a_plot is not None                
            _subfn_helper_setup_new_plot(a_plot_name=k, a_plot=a_plot, a_plot_config_dict=a_plot_config_dict)



    @function_attributes(short_name=None, tags=['update'], input_requires=[], output_provides=[], uses=['_build_image_scale_and_origin'], used_by=['update_active_epoch'], creation_date='2024-11-25 18:54', related_items=[])
    def _update_active_plots(self, debug_print=False):
        """ updates: self.plot,
        Uses: self.xbin, self.p_x_given_n_masked
        
        """
        n_global_xbin_pos_bins: int = len(self.xbin_centers)
        print(f'n_global_xbin_pos_bins: {n_global_xbin_pos_bins}')
        xmin, xmax = self.xbin[0], self.xbin[-1]
        n_pos_bins, n_tbins = np.shape(self.p_x_given_n_masked)
        print(f'n_tbins: {n_tbins}, n_pos_bins: {n_pos_bins}')
        assert n_global_xbin_pos_bins == n_pos_bins, f"n_global_xbin_pos_bins: {n_global_xbin_pos_bins} != n_pos_bins: {n_pos_bins} but it should!"
                

        prev_img: ImageBase = self.plot.getImage('p_x_given_n')
        prev_img.setData(self.p_x_given_n_masked)
        img_scale, img_origin, img_bounds = self._build_image_scale_and_origin()        
        prev_img.setOrigin(img_origin)
        prev_img.setScale(img_scale)
        self.plot.getXAxis().setLimits(img_bounds[0][0], img_bounds[0][1])
        self.plot.getYAxis().setLimits(img_bounds[1][0], img_bounds[1][1])
        
        # prev_img._setYLabel(f'epoch[{active_epoch_idx}: x (bin)')

        max_path = np.nanargmax(self.p_x_given_n_masked, axis=0) # returns the x-bins that maximize the path
        assert len(max_path) == len(self.time_bin_centers)
        # _curve_x = time_bin_centers
        _curve_x = np.arange(len(max_path)) + 0.5 # move forward by a half bin
        
        ## offset appropriately to match the image bounds
        max_path = (max_path * float(img_scale[1])) + float(img_origin[1])

        # a_track_length: float = 170.0
        # effectively_same_location_size = 0.1 * a_track_length # 10% of the track length
        # effectively_same_location_num_bins: int = np.rint(effectively_same_location_size)
        # effectively_same_location_num_bins: int = 4
        # _max_path_Curve = self.plot.addCurve(x=_curve_x, y=max_path, color='r', symbol='s', legend='max_path', replace=True, yerror=effectively_same_location_num_bins)
        # _max_path_Curve = self.plot.addCurve(x=_curve_x, y=max_path, color=(1.0, 0.0, 0.0, 0.25,), symbol='s', legend='max_path', replace=True, yerror=effectively_same_location_num_bins)
        _max_path_Curve = self.plot.addCurve(x=_curve_x, y=max_path, color=(1.0, 0.0, 0.0, 0.25,), symbol='s', replace=True, yerror=None) ## last working
        
        
        # _max_path_Curve
        
        ## Update position plots:
        # _curve_pos_t = np.arange(len(self.active_most_likely_position_indicies)) + 0.5 # move forward by a half bin
        # pos = deepcopy(self.active_most_likely_position_indicies)
        # _curve_vel_t = _curve_pos_t[1:] # + 0.25 # move forward by a half bin
        # vel = np.diff(pos)
        # _curve_accel_t = _curve_pos_t[2:] # + 0.125 # move forward by a half bin
        # accel = np.diff(vel)
        
        # Update position plots
        
        # if self.debug_print:
        #     print(f'_curve_t: {_curve_pos_t}')
        #     print(f'pos: {self.position_derivatives.pos}')
        #     print(f'vel: {vel}')
        #     print(f'accel: {accel}')


        src_plot = self.plot # main plot (Plot2D) is the source plot
        t_range = src_plot.getXAxis().getLimits()
        pos_y_range = src_plot.getYAxis().getLimits()
        
        self.plot_position.getCurve("Position").setData(self.position_derivatives._curve_pos_t, self.position_derivatives.pos)
        if not self.use_bin_units_instead_of_realworld:
            if debug_print:
                print(f'xmin, xmax: {(xmin, xmax)}')        
            data_xmin, data_xmax = np.nanmin(self.position_derivatives.pos), np.nanmax(self.position_derivatives.pos)
            if debug_print:
                print(f'data_xmin, data_xmax: {(data_xmin, data_xmax)}')
            self.plot_position.getYAxis().setLimits(xmin, xmax)
            
        self.plot_velocity.getCurve("Velocity").setData(self.position_derivatives._curve_vel_t, self.position_derivatives.vel)
        self.plot_acceleration.getCurve("Acceleration").setData(self.position_derivatives._curve_accel_t, self.position_derivatives.accel)
        # self.plot_extra.getCurve("Extra").setData(self.position_derivatives._curve_accel_t, self.position_derivatives.accel)
        # self.plot_extra.getCurve("Extra").setData(self.position_derivatives._curve_vel_t, self.position_derivatives.kinetic_energy)
        # self.plot_extra.getCurve("Extra").setData(self.position_derivatives._curve_accel_t, self.position_derivatives.applied_forces)      

        # if len(self.bin_by_bin_heuristic_scores) > 0:
        #     for k, v in self.bin_by_bin_heuristic_scores.items():
        #         ## plot these somehow
        #         assert len(self.position_derivatives._curve_pos_t) == len(v), f"for k: '{k}' - len(self.position_derivatives._curve_pos_t): {len(self.position_derivatives._curve_pos_t)}, len(v): {len(v)}"
        #         self.plot_extra.getCurve("Extra").setData(self.position_derivatives._curve_pos_t, v)    
        #         # self.plot_extra.getCurve("Extra").set
        #         # self.plot_extra.setGraphGrid(which=True) # good
        #         # self.plot_extra.getXAxis().setLabel(a_plot_config_dict["xlabel"])
        #         self.plot_extra.getYAxis().setLabel(f"{k}")
        #         self.plot_extra.setXAxisAutoScale(flag=True)
                

        for k, a_bin_by_bin_values in self.bin_by_bin_heuristic_scores.items():
            assert len(self.position_derivatives._curve_pos_t) == len(a_bin_by_bin_values), f"for k: '{k}' - len(self.position_derivatives._curve_pos_t): {len(self.position_derivatives._curve_pos_t)}, len(a_bin_by_bin_values): {len(a_bin_by_bin_values)}"
            a_plot = self.programmatic_plots_dict.get(k, None)
            # a_plot_config_dict = self.programmatic_plots_config_dict.get(k, None)
            assert a_plot is not None                
            a_plot.getCurve(k).setData(self.position_derivatives._curve_pos_t, a_bin_by_bin_values)   
            # self.plot_extra.getCurve("Extra").set
            # self.plot_extra.setGraphGrid(which=True) # good
            # self.plot_extra.getXAxis().setLabel(a_plot_config_dict["xlabel"])
            # a_plot.getYAxis().setLabel(f"{k}")
            a_plot.setXAxisAutoScale(flag=True) 
        
                

        ## Update the limits:       
        for a_plot_name, a_plot in self.position_plots_dict.items():
            ## Update plot properties:
            a_plot.getXAxis().setLimits(*t_range)
            if a_plot_name == 'Position':
                if self.use_bin_units_instead_of_realworld:
                    a_plot.getYAxis().setLimits(*pos_y_range)
                else:
                    a_plot.getYAxis().setLimits(xmin, xmax)
            else:
                a_plot.resetZoom() # reset zoom resets the y-axis only

        # programmatic_plots_dict ____________________________________________________________________________________________ #
        for a_plot_name, a_plot in self.programmatic_plots_dict.items():
            ## Update plot properties:
            a_plot.getXAxis().setLimits(*t_range)
            if a_plot_name in ['Position', ]:
                if self.use_bin_units_instead_of_realworld:
                    a_plot.getYAxis().setLimits(*pos_y_range)
                else:
                    a_plot.getYAxis().setLimits(xmin, xmax)
            else:
                a_plot.resetZoom() # reset zoom resets the y-axis only
                

    @function_attributes(short_name=None, tags=['init', 'ui'], input_requires=[], output_provides=[], uses=['_build_plots', '_build_utility_controls',], used_by=['cls.init_from_epoch_idx'], creation_date='2024-11-25 18:55', related_items=[])
    def build_ui(self):
        """ builds the ui and plots. Called only once on startup.
        """
        self.ui = PhoUIContainer()
        
        self._build_plots()
        
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
        
        ## add the extra plots:
        for a_plot_name, a_plot in self.programmatic_plots_dict.items():
            self.main_layout.addWidget(a_plot) ## add the plot to the main_layout

        ## add the debugging controls
        ui_dict = self._build_utility_controls(main_layout=self.main_layout)
        self.ui = PhoUIContainer(**ui_dict) ## update with the ui_dict
        
        # Show the main widget
        self.main_widget.show()
        

    # ==================================================================================================================== #
    # Update Functions                                                                                                     #
    # ==================================================================================================================== #
    @function_attributes(short_name=None, tags=['update', 'data'], input_requires=[], output_provides=[], uses=['HeuristicReplayScoring.compute_pho_heuristic_replay_scores'], used_by=['update_active_epoch'], creation_date='2024-07-30 15:08', related_items=[])
    def update_active_epoch_data(self, active_epoch_idx: int):
        """ Data Update only - called after the time-bin is updated.
        
        TODO: this could be greatly optimized.

        Updates: self.p_x_given_n_masked, self.heuristic_scores, self.bin_by_bin_heuristic_scores, self.position_derivatives
        """
        if self.debug_print:
            print(f'update_active_epoch_data(active_epoch_idx={active_epoch_idx})')
        assert self.active_decoder_decoded_epochs_result is not None
        # Data Update Only ________________________________________________________________________________________________________ #
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
        
        self.heuristic_scores = HeuristicReplayScoring.compute_pho_heuristic_replay_scores(a_result=self.active_decoder_decoded_epochs_result, an_epoch_idx=self.active_single_epoch_result.epoch_data_index, debug_print=False, use_bin_units_instead_of_realworld=self.use_bin_units_instead_of_realworld)
        # a_bin_by_bin_jump_time_window_centers, a_bin_by_bin_jump_distance = HeuristicReplayScoring.bin_by_bin_jump_distance(a_result=self.active_decoder_decoded_epochs_result, an_epoch_idx=self.active_single_epoch_result.epoch_data_index, a_decoder_track_length=self.decoder_track_length)

        # Extra Custom/Programmatic Plots ____________________________________________________________________________________ #
        self.bin_by_bin_heuristic_scores = {} # new dict
        for k, a_bin_by_bin_fn in HeuristicReplayScoring.build_all_bin_by_bin_computation_fn_dict().items():
            a_bin_by_bin_fn_time_window_centers, a_bin_by_bin_values = a_bin_by_bin_fn(a_result=self.active_decoder_decoded_epochs_result, an_epoch_idx=self.active_single_epoch_result.epoch_data_index, a_decoder_track_length=self.decoder_track_length)
            if a_bin_by_bin_values is not None:
                self.bin_by_bin_heuristic_scores[k] = a_bin_by_bin_values
                


        # longest_sequence_length, longest_sequence_length_ratio, direction_change_bin_ratio, congruent_dir_bins_ratio, total_congruent_direction_change, total_variation, integral_second_derivative, stddev_of_diff, position_derivatives_df = self.heuristic_scores
        # np.diff(active_captured_single_epoch_result.most_likely_position_indicies)
        if self.debug_print:
            # print(f'heuristic_scores: {astuple(self.heuristic_scores)[:-1]}')
            print(f"heuristic_scores: {asdict(self.heuristic_scores, filter=(lambda an_attr, attr_value: an_attr.name not in ['position_derivatives_df']))}")

        # Update position data:
        active_active_most_likely_position_arr = deepcopy(self.active_most_likely_position_arr)
        self.position_derivatives = PositionDerivativesContainer(pos=active_active_most_likely_position_arr)
                


    @function_attributes(short_name=None, tags=['update'], input_requires=[], output_provides=[], uses=['update_active_epoch_data', '_update_active_plots', '_update_active_tables'], used_by=['on_slider_change'], creation_date='2024-07-30 15:09', related_items=[])
    def update_active_epoch(self, active_epoch_idx: int):
        """ called after the time-bin is updated.
        
        requires: self.active_decoder_decoded_epochs_result
        
        """
        if self.debug_print:
            print(f'update_active_epoch(active_epoch_idx={active_epoch_idx})')
        assert self.active_decoder_decoded_epochs_result is not None
        xmin, xmax = self.xbin[0], self.xbin[-1]
        
        # Data Update ________________________________________________________________________________________________________ #
        self.update_active_epoch_data(active_epoch_idx=active_epoch_idx)
        
        # Plottings __________________________________________________________________________________________________________ #
        self._update_active_plots()
        
        ## Update the tables:
        self._update_active_tables()
        


    def _update_active_tables(self):
        """ Updates the epoch-dependent tables after active_epoch_idx is changed.
                
        requires up-to-date `self.heuristic_scores`
        Updates:
            self.ui.models_dict[table_id_name]
            self.ui.views_dict[table_id_name]'s models
            Table appearance
            
        """
        from pyphocorehelpers.gui.Qt.pandas_model import SimplePandasModel

        heuristic_scores_dict = asdict(self.heuristic_scores, filter=lambda a, v: a.name not in ['position_derivatives_df'])
        df = pd.DataFrame(list(heuristic_scores_dict.values()), index=heuristic_scores_dict.keys(), columns=['Values'])

        ## INPUTS: df

        # Update the table:
        table_id_name: str = 'combined_epoch_stats'
        self.ui.models_dict[table_id_name] = SimplePandasModel(df.copy())

        # Update the view's model:
        table_view = self.ui.views_dict[table_id_name]
        table_view.setModel(self.ui.models_dict[table_id_name])
        # Adjust the column widths to fit the contents
        table_view.resizeColumnsToContents()


    def on_slider_change(self, change):
        """ Callback for the integrated slider that allows selecting the active epoch:
        
        """

        # print("Slider value:", change.new)
        active_epoch_idx: int = int(change.new)
        if self.debug_print:
            print(f'epoch[{active_epoch_idx}]')
    
        self.update_active_epoch(active_epoch_idx=active_epoch_idx)
        

    def programmatically_update_active_epoch(self, active_epoch_idx: int):
        """ called after the time-bin is updated.
        
        requires: self.active_decoder_decoded_epochs_result
        
        """
        if self.debug_print:
            print(f'programmatically_update_active_epoch(active_epoch_idx={active_epoch_idx})')
        active_epoch_scrollbar_ctrl_widget: ScrollBarWithSpinBox = self.ui['ctrls_widget']
        active_epoch_scrollbar_ctrl_widget.setValue(active_epoch_idx)
        active_epoch_scrollbar_ctrl_widget.emitChanged()
            
            
    # Data Indexing Helpers ______________________________________________________________________________________________ #
    def find_data_indicies_from_epoch_times(self, epoch_times: NDArray) -> NDArray:
        subset = deepcopy(self.active_decoder_decoded_epochs_result)
        if not isinstance(subset.filter_epochs, pd.DataFrame):
            subset.filter_epochs = subset.filter_epochs.to_dataframe()
        return subset.filter_epochs.epochs.find_data_indicies_from_epoch_times(epoch_times=epoch_times)

    def find_epoch_times_to_data_indicies_map(self, epoch_times: NDArray, atol:float=1e-3, t_column_names=None) -> Dict[Union[float, Tuple[float, float]], Union[int, NDArray]]:
        """ returns the a Dict[Union[float, Tuple[float, float]], Union[int, NDArray]] matching data indicies corresponding to the epoch [start, stop] times 
        epoch_times: S x 2 array of epoch start/end times
        Returns: (S, ) array of data indicies corresponding to the times.

        Uses:
            epoch_time_to_index_map = deepcopy(dbgr.active_decoder_decoded_epochs_result).filter_epochs.epochs.find_epoch_times_to_data_indicies_map(epoch_times=[epoch_start_time, ])
        
        """
        subset = deepcopy(self.active_decoder_decoded_epochs_result)
        if not isinstance(subset.filter_epochs, pd.DataFrame):
            subset.filter_epochs = subset.filter_epochs.to_dataframe()
        return subset.filter_epochs.epochs.find_epoch_times_to_data_indicies_map(epoch_times=epoch_times, atol=atol, t_column_names=t_column_names)
           


    # Utility ____________________________________________________________________________________________________________ #
    def _build_utility_controls(self, main_layout):
        """ Build the utility controls at the bottom """
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, get_utility_dock_colors
        
        ctrls_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=get_utility_dock_colors, showCloseButton=False)

        active_epoch_scrollbar_ctrl_widget = ScrollBarWithSpinBox()
        active_epoch_scrollbar_ctrl_widget.setObjectName("ctrls_widget")
        active_epoch_scrollbar_ctrl_widget.update_range(0, (self.n_epochs-1))
        active_epoch_scrollbar_ctrl_widget.setValue(self.active_epoch_index)

        def valueChanged(new_val:int):
            # if self.debug_print:
            #     print(f'ScrollBarWithSpinBox valueChanged(new_val: {new_val})')
            self.update_active_epoch(active_epoch_idx=int(new_val))
            

        ctrls_widget_connection = active_epoch_scrollbar_ctrl_widget.sigValueChanged.connect(valueChanged)
        ctrl_layout_widget = pg.LayoutWidget()
        ctrl_layout_widget.addWidget(active_epoch_scrollbar_ctrl_widget, row=1, rowspan=1, col=1, colspan=2)
        ctrl_widgets_dict = dict(ctrls_widget=active_epoch_scrollbar_ctrl_widget, ctrls_widget_connection=ctrls_widget_connection)

        position_derivatives_df: pd.DataFrame = deepcopy(self.heuristic_scores.position_derivatives_df)
        active_epochs_df: pd.DataFrame = self.filter_epochs

        # Tabbled table widget:
        tab_widget, views_dict, models_dict = create_tabbed_table_widget(dataframes_dict={'epochs': active_epochs_df.copy(),
                                                                                                        'position_derivatives': position_derivatives_df.copy(), 
                                                                                                        'combined_epoch_stats': pd.DataFrame()})
        ctrl_widgets_dict['tables_tab_widget'] = tab_widget
        ctrl_widgets_dict['views_dict'] = views_dict
        ctrl_widgets_dict['models_dict'] = models_dict

        # Add the tab widget to the layout
        ctrl_layout_widget.addWidget(tab_widget, row=2, rowspan=1, col=1, colspan=2)
        tab_widget.setMinimumHeight(400)
        
        # logTextEdit = LogViewer() # QTextEdit subclass
        # logTextEdit.setReadOnly(True)
        # logTextEdit.setObjectName("logTextEdit")
        # # logTextEdit.setSizePolicy(pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Expanding)

        # ctrl_layout_widget.addWidget(logTextEdit, row=3, rowspan=1, col=1, colspan=2)
        # ctrl_widgets_dict['logTextEdit'] = logTextEdit
        
        # _out_dock_widgets['bottom_controls'] = root_dockAreaWindow.add_display_dock(identifier='bottom_controls', widget=ctrl_layout_widget, dockSize=(600,200), dockAddLocationOpts=['bottom'], display_config=ctrls_dock_config)
        # ctrls_dock_widgets_dict = {}
        # ctrls_dock_widgets_dict['bottom_controls'] = root_dockAreaWindow.add_display_dock(identifier='bottom_controls', widget=ctrl_layout_widget, dockSize=(600,200), dockAddLocationOpts=['bottom'], display_config=ctrls_dock_config)

        ## Add to main layout:
        main_layout.addWidget(ctrl_layout_widget)
    
        ui_dict = dict(ctrl_layout=ctrl_layout_widget, **ctrl_widgets_dict, on_valueChanged=valueChanged)
        return ui_dict