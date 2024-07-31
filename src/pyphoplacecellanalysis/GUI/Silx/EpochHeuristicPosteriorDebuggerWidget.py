from copy import deepcopy
import numpy as np
from pathlib import Path
import pandas as pd
from functools import partial
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


@define(slots=False)
class PositionDerivativesContainer:
    """
    Holds 
    
    from pyphoplacecellanalysis.GUI.Silx.EpochHeuristicPosteriorDebuggerWidget import PositionDerivativesContainer
    
    
    """
    pos: NDArray = field()
    mass: float = field(default=1.0)
    
    ## Computable:
    _curve_pos_t: NDArray = field(default=None)
    vel: NDArray = field(default=None)
    _curve_vel_t: NDArray = field(default=None)
    accel: NDArray = field(default=None)
    _curve_accel_t: NDArray = field(default=None)
    kinetic_energy: NDArray = field(default=None)
    total_energy: float = field(default=None)

    applied_forces: NDArray = field(default=None)
    total_applied_force: float = field(default=None)
    
    def __attrs_post_init__(self):
        # Recompute all:
        self.compute()
        

    def compute(self):
        """ called to recompute all computed properties after updating self.pos
        """
        self._curve_pos_t = np.arange(len(self.pos)) + 0.5  # Move forward by a half bin


        # Compute velocity
        self.vel = np.diff(self.pos)
        self._curve_vel_t = self._curve_pos_t[:-1] + 0.5  # Center between the original position x-values

        # Compute acceleration
        self.accel = np.diff(self.vel)
        self._curve_accel_t = self._curve_vel_t[:-1] + 0.5  # Center between the velocity x-values

        # Compute kinetic energy at each time step
        self.kinetic_energy: NDArray = 0.5 * self.mass * self.vel**2

        # Total energy needed to move the particle along the trajectory
        self.total_energy: float = np.sum(self.kinetic_energy)
        
        ## Forces
        self.applied_forces = self.accel * self.mass
        self.total_applied_force = np.sum(self.applied_forces)
        
        

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


def remove_all_plot_toolbars(a_plot: Union[Plot1D, Plot2D]):
    _plot_toolbars = [a_plot.toolBar(), a_plot.getOutputToolBar(), a_plot.getInteractiveModeToolBar()]
    for a_toolbar in _plot_toolbars:
        a_plot.removeToolBar(a_toolbar)



## Uses: xbin, t_start, pos_bin_size, time_bin_size

# Define the partial function above the class
plot1d_factory = partial(Plot1D)
# plot1d_factory = partial(Plot1D, toolbar=False)

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
    
    # most_likely_position_indicies
    @property
    def active_most_likely_position_indicies(self) -> NDArray:
        """The most_likely_position_indicies property."""
        assert len(self.active_single_epoch_result.most_likely_position_indicies) == 1, f" the [0] is to handle the fact that for some reason the list is double-wrapped: [[37  0 28 52 56 28 55]]"
        return self.active_single_epoch_result.most_likely_position_indicies[0] # the [0] is to handle the fact that for some reason the list is double-wrapped: [[37  0 28 52 56 28 55]]


    @property
    def position_plots_dict(self) -> Dict[str, Plot1D]:
        """ convenince access to the dict of position plots 
        """
        position_plots_list = [self.plot_position, self.plot_velocity, self.plot_acceleration, self.plot_extra]
        return dict(zip(['Position', 'Velocity', 'Acceleration', 'Extra'], position_plots_list))


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
        
    def build_ui(self):
        """ builds the ui and plots. Called only once on startup.
        """
        self.ui = PhoUIContainer()
        
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
        label_kwargs = dict(xlabel='t (tbin)', ylabel='x_pos (bin)')
        self.plot.addImage(self.p_x_given_n_masked, legend='p_x_given_n', replace=True, colormap=self.a_cmap, origin=img_origin, scale=img_scale, **label_kwargs, resetzoom=True) # , colormap="viridis", vmin=0, vmax=1
        prev_img: ImageBase = self.plot.getImage('p_x_given_n')
        ## Setup grid:            
        pos_x_range = self.plot.getXAxis().getLimits()
        pos_y_range = self.plot.getYAxis().getLimits()
        
        pos_x_range = (int(pos_x_range[0]), int(pos_x_range[1]))
        pos_y_range = (int(pos_y_range[0]), int(pos_y_range[1]))
        self.plot.setGraphGrid(which=True)
        # remove_all_plot_toolbars(self.plot)

        # Position Derivative Plots:
        empty_arr = np.array([], dtype='int64')

        common_plot_config_dict = dict(symbol='o', linestyle=':', color='blue')

        plot_configs_dict = {"Position": dict(legend="Position", xlabel='t (tbin)', ylabel='x_pos (bin)', **common_plot_config_dict),
            "Velocity": dict(legend="Velocity", xlabel='t (tbin)', ylabel='velocity (bin/tbin)', baseline=0.0, fill=True, **common_plot_config_dict),
            "Acceleration": dict(legend="Acceleration", xlabel='t (tbin)', ylabel='accel. (bin/tbin^2)', baseline=0.0, fill=True, **common_plot_config_dict),
            "Extra": dict(legend="Extra", xlabel='t (tbin)', ylabel='Extra', baseline=0.0, fill=True, **common_plot_config_dict),
        }

        position_plots_list = [self.plot_position, self.plot_velocity, self.plot_acceleration, self.plot_extra]
        position_plots_dict = dict(zip(list(plot_configs_dict.keys()), position_plots_list))

        # Add data to the plots:
        for a_plot_name, a_plot in position_plots_dict.items():
            a_plot_config_dict = plot_configs_dict[a_plot_name]
            remove_all_plot_toolbars(a_plot)
            
            ## add curves
            a_plot.addCurve(empty_arr, empty_arr, **a_plot_config_dict, replace=True)
            
            ## Update plot properties:
            a_plot.setActiveCurve(a_plot_name)
            a_plot.setGraphGrid(which=True) # good
            a_plot.getXAxis().setLabel(a_plot_config_dict["xlabel"])
            a_plot.getYAxis().setLabel(a_plot_config_dict["ylabel"])
            a_plot.setXAxisAutoScale(flag=False)
            if a_plot_name == 'Position':
                a_plot.setYAxisAutoScale(flag=False) # position y-axis is fixed to the total bins
            else:
                a_plot.setYAxisAutoScale(flag=True)

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

        ## add the debugging controls
        ui_dict = self._build_utility_controls(main_layout=self.main_layout)
        self.ui = PhoUIContainer(**ui_dict) ## update with the ui_dict
        
        # Show the main widget
        self.main_widget.show()
        

    # def recompute(self):
    @function_attributes(short_name=None, tags=['update'], input_requires=[], output_provides=[], uses=['HeuristicReplayScoring.compute_pho_heuristic_replay_scores'], used_by=['update_active_epoch'], creation_date='2024-07-30 15:08', related_items=[])
    def update_active_epoch_data(self, active_epoch_idx: int):
        """ Data Update only - called after the time-bin is updated.
        
        TODO: this could be greatly optimized.

        Updates: self.p_x_given_n_masked, self.heuristic_scores
        """
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
        
        self.heuristic_scores = HeuristicReplayScoring.compute_pho_heuristic_replay_scores(a_result=self.active_decoder_decoded_epochs_result, an_epoch_idx=self.active_single_epoch_result.epoch_data_index, debug_print=False)
        # longest_sequence_length, longest_sequence_length_ratio, direction_change_bin_ratio, congruent_dir_bins_ratio, total_congruent_direction_change, total_variation, integral_second_derivative, stddev_of_diff, position_derivatives_df = self.heuristic_scores
        # np.diff(active_captured_single_epoch_result.most_likely_position_indicies)
        if self.debug_print:
            # print(f'heuristic_scores: {astuple(self.heuristic_scores)[:-1]}')
            print(f"heuristic_scores: {asdict(self.heuristic_scores, filter=(lambda an_attr, attr_value: an_attr.name not in ['position_derivatives_df']))}")

        # Update position data:
        self.position_derivatives = PositionDerivativesContainer(pos=deepcopy(self.active_most_likely_position_indicies))
        


    @function_attributes(short_name=None, tags=['update'], input_requires=[], output_provides=[], uses=['update_active_epoch_data'], used_by=['on_slider_change'], creation_date='2024-07-30 15:09', related_items=[])
    def update_active_epoch(self, active_epoch_idx: int):
        """ called after the time-bin is updated.
        
        requires: self.active_decoder_decoded_epochs_result
        
        """
        print(f'update_active_epoch(active_epoch_idx={active_epoch_idx})')
        assert self.active_decoder_decoded_epochs_result is not None
        
        # Data Update ________________________________________________________________________________________________________ #
        self.update_active_epoch_data(active_epoch_idx=active_epoch_idx)
        
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
        
        self.plot_position.getCurve("Position").setData(self.position_derivatives._curve_pos_t, self.position_derivatives.pos)
        self.plot_velocity.getCurve("Velocity").setData(self.position_derivatives._curve_vel_t, self.position_derivatives.vel)
        self.plot_acceleration.getCurve("Acceleration").setData(self.position_derivatives._curve_accel_t, self.position_derivatives.accel)
        # self.plot_extra.getCurve("Extra").setData(self.position_derivatives._curve_accel_t, self.position_derivatives.accel)
        # self.plot_extra.getCurve("Extra").setData(self.position_derivatives._curve_vel_t, self.position_derivatives.kinetic_energy)
        self.plot_extra.getCurve("Extra").setData(self.position_derivatives._curve_accel_t, self.position_derivatives.applied_forces)        

        ## Update the limits:                
        src_plot = self.plot # main plot (Plot2D) is the source plot
        x_range = src_plot.getXAxis().getLimits()
        pos_y_range = src_plot.getYAxis().getLimits()
        
        for a_plot_name, a_plot in self.position_plots_dict.items():
            # a_plot.addCurve(empty_arr, empty_arr, **a_plot_config_dict, replace=True)
            
            ## Update plot properties:
            a_plot.getXAxis().setLimits(*x_range)
            if a_plot_name == 'Position':
                a_plot.getYAxis().setLimits(*pos_y_range)
            else:
                a_plot.resetZoom() # reset zoom resets the y-axis only



    def on_slider_change(self, change):
        """Updates the active epoch via a slider:
        
        """

        # print("Slider value:", change.new)
        active_epoch_idx: int = int(change.new)
        if self.debug_print:
            print(f'epoch[{active_epoch_idx}]')
    
        self.update_active_epoch(active_epoch_idx=active_epoch_idx)
        

    def _build_utility_controls(self, main_layout):
        """ Build the utility controls at the bottom """
        from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, get_utility_dock_colors
        
        ctrls_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=get_utility_dock_colors, showCloseButton=False)

        ctrls_widget = ScrollBarWithSpinBox()
        ctrls_widget.setObjectName("ctrls_widget")
        ctrls_widget.update_range(0, (self.n_epochs-1))
        ctrls_widget.setValue(self.active_epoch_index)

        def valueChanged(new_val:int):
            print(f'ScrollBarWithSpinBox valueChanged(new_val: {new_val})')
            self.update_active_epoch(active_epoch_idx=int(new_val))
            

        ctrls_widget_connection = ctrls_widget.sigValueChanged.connect(valueChanged)
        ctrl_layout_widget = pg.LayoutWidget()
        ctrl_layout_widget.addWidget(ctrls_widget, row=1, rowspan=1, col=1, colspan=2)
        ctrl_widgets_dict = dict(ctrls_widget=ctrls_widget, ctrls_widget_connection=ctrls_widget_connection)

        # Step 4: Create DataFrame and QTableView
        # df =  selected active_selected_spikes_df # pd.DataFrame(...)  # Replace with your DataFrame
        # model = PandasModel(df)
        # pandasDataFrameTableModel = SimplePandasModel(active_epochs_df.copy())

        # tableView = pg.QtWidgets.QTableView()
        # tableView.setModel(pandasDataFrameTableModel)
        # tableView.setObjectName("pandasTablePreview")
        # # tableView.setSizePolicy(pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Expanding)

        # ctrl_widgets_dict['pandasDataFrameTableModel'] = pandasDataFrameTableModel
        # ctrl_widgets_dict['tableView'] = tableView

        # # Step 5: Add TableView to LayoutWidget
        # ctrl_layout_widget.addWidget(tableView, row=2, rowspan=1, col=1, colspan=1)

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