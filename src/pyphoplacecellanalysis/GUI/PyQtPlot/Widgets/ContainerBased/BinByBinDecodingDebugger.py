# ==================================================================================================================== #
# 2025-01-21 - Bin-by-bin decoding examples                                                                            #
# ==================================================================================================================== #
import numpy as np
import pandas as pd
import pyqtgraph as pg
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
import pyphoplacecellanalysis.General.type_aliases as types
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from attrs import field, Factory, define
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.print_helpers import strip_type_str_to_classname
from neuropy.utils.mixins.AttrsClassHelpers import keys_only_repr

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.DataStructure.RenderPlots.PyqtgraphRenderPlots import PyqtgraphRenderPlots
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool import GenericPyQtGraphContainer
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots # PyqtgraphRenderPlots
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, SingleEpochDecodedResult
from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
from neuropy.utils.indexing_helpers import PandasHelpers

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import get_proper_global_spikes_df


@define(slots=False, eq=False)
class BinByBinDebuggingData:
    """ Holds the data needed for using the BinByBinDecodingDebugger
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.BinByBinDecodingDebugger import BinByBinDebuggingData
    
    
    ## INPUTS: neuron_IDs, (active_global_spikes_df, active_window_decoded_epochs_df, active_aclu_spike_counts_dict_list)
    ## INPUTS: active_window_slice_idxs, active_window_time_bin_edges, active_p_x_given_n
    plots_container = PyqtgraphRenderPlots(name='PhoTest', root_plot=None) # Create a new one
    plots_data = RenderPlotsData(name=f'epoch[Test]', spikes_df=active_global_spikes_df, active_aclus=neuron_IDs)
    win, out_pf1D_decoder_template_objects, (plots_container, plots_data) = BinByBinDecodingDebugger._perform_build_time_binned_decoder_debug_plots(a_decoder=a_decoder, time_bin_edges=active_window_time_bin_edges, p_x_given_n=active_p_x_given_n, active_epoch_active_aclu_spike_counts_list=active_epoch_active_aclu_spike_counts_list,
                                                                                                                                plots_data=plots_data, plots_container=plots_container,
                                                                                                                                debug_print=False)
                                                                                                                                
    """
    decoding_time_bin_size: float = field(repr=True)
    a_decoder: BasePositionDecoder = field(repr=False)
    global_spikes_df: pd.DataFrame = field(repr=False)
    decoding_bins_epochs_df: pd.DataFrame = field(repr=keys_only_repr)
    time_bin_edges: NDArray = field(repr=keys_only_repr)
    p_x_given_n: NDArray = field(repr=keys_only_repr)
    n_max_debugged_time_bins: Optional[int] = field(default=20)

    @classmethod
    def init_from_single_continuous_result(cls, a_decoder, global_spikes_df: pd.DataFrame, single_continuous_result: SingleEpochDecodedResult, decoding_time_bin_size: float, n_max_debugged_time_bins: Optional[int] = 20):
        """
        decoding_time_bin_size: float = a_decoded_result.decoding_time_bin_size
        single_continuous_result: SingleEpochDecodedResult = a_decoded_result.get_result_for_epoch(0) # SingleEpochDecodedResult
        a_decoder = deepcopy(results1D.decoders[a_decoder_name])
        neuron_IDs = deepcopy(a_decoder.neuron_IDs)
        global_spikes_df = get_proper_global_spikes_df(curr_active_pipeline).spikes.sliced_by_neuron_id(neuron_IDs) ## only get the relevant spikes
        
        
        bin_by_bin_data: BinByBinDebuggingData = BinByBinDebuggingData.init_from_single_continuous_result(a_decoder=a_decoder, global_spikes_df=global_spikes_df, single_continuous_result=single_continuous_result, decoding_time_bin_size=decoding_time_bin_size)
        
        """

        decoding_bins_epochs_df: pd.DataFrame = single_continuous_result.build_pseudo_epochs_df_from_decoding_bins().epochs.get_valid_df()
        time_bin_edges = deepcopy(single_continuous_result.time_bin_edges)
        p_x_given_n: NDArray = deepcopy(single_continuous_result.p_x_given_n)

        global_spikes_df = deepcopy(global_spikes_df)
        global_spikes_df = global_spikes_df.spikes.add_binned_time_column(time_window_edges=time_bin_edges, time_window_edges_binning_info=single_continuous_result.time_bin_container.edge_info) # "binned_time" column added
        global_spikes_df = global_spikes_df.dropna(axis='index', how='any', subset=['binned_time'], inplace=False)
        
        global_spikes_df['binned_time'] = global_spikes_df['binned_time'].astype(int) - 1 # convert to 0-indexed
        # global_spikes_df
        # a_decoded_result.filter_epochs
        # a_decoded_result.spkcount

        ## OUTPUTS: decoding_bins_epochs_df, time_bin_edges, p_x_given_n, decoding_time_bin_size
        ## get the list of all 'aclu' values in the dataframe when groupnig on the 'binned_time' columns:
        # global_spikes_df['binned_time']
        # Get unique aclu values per time bin
        unique_aclus_per_bin = global_spikes_df.groupby('binned_time')['aclu'].unique()
        # unique_aclus_per_bin

        ## OUTPUTS: unique_aclus_per_bin
        # assert len(decoding_bins_epochs_df) == len(unique_aclus_per_bin), f"len(decoding_bins_epochs_df): {len(decoding_bins_epochs_df)}, len(unique_aclus_per_bin): {len(unique_aclus_per_bin)}"
        decoding_bins_epochs_df['active_aclus'] = unique_aclus_per_bin
        decoding_bins_epochs_df['active_aclus'] = decoding_bins_epochs_df['active_aclus'].apply(lambda x: [] if np.any(pd.isna(x)) else x) ## fill the np.NaNs with empty lists
        # spike_counts_per_bin = global_spikes_df.groupby('binned_time')['aclu'].value_counts().unstack(fill_value=0) ## global:

        ## OUTPUTS: spike_counts_per_bin
        decoding_bins_epochs_df
        _obj = cls(decoding_time_bin_size=decoding_time_bin_size, a_decoder=a_decoder, global_spikes_df=global_spikes_df, decoding_bins_epochs_df=decoding_bins_epochs_df, time_bin_edges=time_bin_edges, p_x_given_n=p_x_given_n, n_max_debugged_time_bins=n_max_debugged_time_bins)
        return _obj
    

    def sliced_to_current_window(self, active_window_t_start, active_window_t_end, debug_print:bool=True):
        """ captures: global_spikes_df, decoding_bins_epochs_df, time_bin_edges, p_x_given_n
        
        """
        ## INPUTS: global_spikes_df, decoding_bins_epochs_df
        ## INPUTS: decoding_bins_epochs_df, time_bin_edges, p_x_given_n
        ## Slice to current window:
        
        print(f'active_window_t_start: {active_window_t_start}, active_window_t_end: {active_window_t_end}')

        active_global_spikes_df = deepcopy(self.global_spikes_df).spikes.time_sliced(t_start=active_window_t_start, t_stop=active_window_t_end)
        active_window_decoded_epochs_df = deepcopy(self.decoding_bins_epochs_df).time_slicer.time_slice(t_start=active_window_t_start, t_stop=active_window_t_end)
        active_window_decoded_epochs_df['rel_epoch_idx'] = active_window_decoded_epochs_df.index.to_numpy().astype(int) ## add the ''rel_epoch_idx' column
        ## constrain plot to just self.n_max_debugged_time_bins time bins:
        if self.n_max_debugged_time_bins is not None:
            min_time_bin_idx: int = active_window_decoded_epochs_df['label'].astype(int).min()
            max_time_bin_idx: int = (min_time_bin_idx + self.n_max_debugged_time_bins)-1
            print(f'self.n_max_debugged_time_bins: {self.n_max_debugged_time_bins}, min_time_bin_idx: {min_time_bin_idx}, max_time_bin_idx: {max_time_bin_idx}')
            max_time_stop_sec: float = active_window_decoded_epochs_df[(active_window_decoded_epochs_df['label'].astype(int) <= max_time_bin_idx)]['stop'].max()
            print(f'max_time_stop_sec: {max_time_stop_sec}')
            # active_global_spikes_df, active_window_decoded_epochs_df, active_epoch_active_aclu_spike_counts_list, (active_window_slice_idxs, active_window_time_bin_edges, active_p_x_given_n) = self.sliced_to_current_window(active_window_t_start, max_time_stop_sec)
            # active_window_decoded_epochs_df
            active_global_spikes_df = deepcopy(self.global_spikes_df).spikes.time_sliced(t_start=active_window_t_start, t_stop=max_time_stop_sec)
            active_window_decoded_epochs_df = deepcopy(self.decoding_bins_epochs_df).time_slicer.time_slice(t_start=active_window_t_start, t_stop=max_time_stop_sec)
            active_window_decoded_epochs_df['rel_epoch_idx'] = active_window_decoded_epochs_df.index.to_numpy().astype(int) ## add the ''rel_epoch_idx' column
            

        ## INPUTS: decoding_bins_epochs_df, time_bin_edges, p_x_given_n
        active_window_slice_idxs: NDArray = active_window_decoded_epochs_df['label'].to_numpy().astype(int) ## get indicies required to slice
        active_window_time_bin_edges = self.time_bin_edges[active_window_slice_idxs]
        if np.ndim(self.p_x_given_n) == 3:
            active_p_x_given_n = self.p_x_given_n[:, :, active_window_slice_idxs]
        else:
            active_p_x_given_n = self.p_x_given_n[:, active_window_slice_idxs]

        n_epoch_time_bins = len(active_window_time_bin_edges) - 1
        if debug_print:
            print(f'n_epoch_time_bins: {n_epoch_time_bins}')
            
        ## OUTPUTS: active_window_slice_idxs, active_window_time_bin_edges, active_p_x_given_n


        # active_spike_counts_per_bin_df: pd.DataFrame = active_global_spikes_df.groupby('binned_time')['aclu'].value_counts().unstack(fill_value=0)
        active_spike_counts_per_bin_df: pd.DataFrame = active_global_spikes_df.groupby('binned_time')['aclu'].value_counts().unstack(fill_value=0)
        # active_spike_counts_per_bin_df.index # Int64Index([7957, 7958, 7959, 7960, 7963, 7964, 7966, 7967], dtype='int64', name='binned_time')
        ## Find missing index values and insert rows containing all zeros - active_window_slice_idxs: array([7958, 7959, 7960, 7961, 7962, 7963, 7964, 7965, 7966, 7967])
        # Get the expected range of bin indices
        expected_bin_indices = active_window_slice_idxs
        # Reindex the dataframe to include all expected bins, filling missing ones with 0
        active_spike_counts_per_bin_df = active_spike_counts_per_bin_df.reindex(expected_bin_indices, fill_value=0)


        ## INPUTS: active_spike_counts_per_bin
        # active_spike_counts_per_bin.columns
        active_aclu_spike_counts_dict_list: List[Dict[types.aclu_index, int]] = active_spike_counts_per_bin_df.to_dict(orient='record')
        # active_aclu_spike_counts_dict_list: List[Dict[types.aclu_index, int]] = [{k:v for k, v in an_aclu_to_spike_count_dict.items() if v > 0} for an_aclu_to_spike_count_dict in active_aclu_spike_counts_dict_list] ## only include the aclus with non-zero num spikes
        active_aclu_spike_counts_dict_list: List[Dict[types.aclu_index, int]] = [{k:v for k, v in an_aclu_to_spike_count_dict.items() if v > 0} for an_aclu_to_spike_count_dict in active_aclu_spike_counts_dict_list] ## only include the aclus with non-zero num spikes
        # aclu_columns = active_spike_counts_per_bin.columns.to_numpy().astype(int)
        active_epoch_active_aclu_spike_counts_list = active_aclu_spike_counts_dict_list
        ## OUTPUTS: active_aclu_spike_counts_dict_list
        

        ## OUTPUTS: active_global_spikes_df, active_window_decoded_epochs_df, active_epoch_active_aclu_spike_counts_list
        # active_epoch_active_aclu_spike_counts_list
        return active_global_spikes_df, active_window_decoded_epochs_df, active_epoch_active_aclu_spike_counts_list, (active_window_slice_idxs, active_window_time_bin_edges, active_p_x_given_n)



@metadata_attributes(short_name=None, tags=['pyqtgraph'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-24 17:22', related_items=[])
@define(slots=False, eq=False)
class BinByBinDecodingDebugger(GenericPyQtGraphContainer):
    """ handles displaying the process of debugging decoding for each time bin
    
    Usage 1 > Plotting Laps:    
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import LayoutScrollability, pyqtplot_build_image_bounds_extent
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import BaseTemplateDebuggingMixin, build_pf1D_heatmap_with_labels_and_peaks, TrackTemplates
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.BinByBinDecodingDebugger import BinByBinDecodingDebugger 

        # Example usage:
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping() 
        global_laps_epochs_df = global_laps.to_dataframe()
        global_laps_epochs_df

        ## INPUTS: 
        time_bin_size: float = 0.250
        a_lap_id: int = 9
        a_decoder_name = 'long_LR'
        epoch_id_col_name = 'lap_id'
        ## COMPUTED: 
        a_decoder_idx: int = track_templates.get_decoder_names().index(a_decoder_name)
        a_decoder = deepcopy(track_templates.long_LR_decoder)
        (_out_decoded_time_bin_edges, _out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists, _out_decoded_active_p_x_given_n, _out_decoded_active_plots_data) = BinByBinDecodingDebugger.build_spike_counts_and_decoder_outputs(a_decoder=a_decoder, epochs_df=global_laps_epochs_df, spikes_df=global_spikes_df, epoch_id_col_name=epoch_id_col_name, time_bin_size=time_bin_size)
        win, out_pf1D_decoder_template_objects, (_out_decoded_active_plots, _out_decoded_active_plots_data) = BinByBinDecodingDebugger.build_time_binned_decoder_debug_plots(a_decoder=a_decoder, an_epoch_id=a_lap_id, _out_decoded_time_bin_edges=_out_decoded_time_bin_edges, _out_decoded_active_p_x_given_n=_out_decoded_active_p_x_given_n,
                                                                                                                                                                            _out_decoded_active_unit_lists=_out_decoded_active_unit_lists, _out_decoded_active_plots_data=_out_decoded_active_plots_data, debug_print=True)

                                                                                                                                                                                

    Usage 2:
        ## All-in-one mode:
        win, out_pf1D_decoder_template_objects, (_out_decoded_active_plots, _out_decoded_active_plots_data) = BinByBinDecodingDebugger.plot_bin_by_bin_decoding_example(curr_active_pipeline=curr_active_pipeline, a_decoder=a_decoder, time_bin_size=time_bin_size, an_epoch_id=an_epoch_id)


    """
    # ==================================================================================================================== #
    # GenericPyQtGraphContainer Conformance                                                                                #
    # ==================================================================================================================== #
    name: str = field(default='binByBinDecodingDebugger')
    plots: PyqtgraphRenderPlots = field(default=Factory(PyqtgraphRenderPlots, 'binByBinDecodingDebugger'))
    plot_data: RenderPlotsData = field(default=Factory(RenderPlotsData, 'binByBinDecodingDebugger'))
    ui: PhoUIContainer = field(default=Factory(PhoUIContainer, 'binByBinDecodingDebugger'))
    params: VisualizationParameters = field(default=Factory(VisualizationParameters, 'binByBinDecodingDebugger'), repr=keys_only_repr)
    
    # time_bin_size: float = field(default=0.500) # 500ms
    # spikes_df: pd.DataFrame = field()
    # global_laps_epochs_df: pd.DataFrame = field()
    

    @classmethod
    def _compute_active_units_for_time_bins(cls, unit_specific_time_binned_spike_counts: np.ndarray, neuron_IDs: np.ndarray, n_time_bins: int) -> List[Dict[int, float]]:
        """Computes the active units and their spike counts for each time bin.
        
        Args:
            unit_specific_time_binned_spike_counts (np.ndarray): Array of spike counts per unit per time bin
            neuron_IDs (np.ndarray): Array of neuron IDs
            n_time_bins (int): Number of time bins
            
        Returns:
            List[Dict[int, float]]: List of dictionaries mapping active unit IDs to their spike counts for each time bin
        """
        active_units_list = []
        for a_time_bin_idx in np.arange(n_time_bins):
            unit_spike_counts = np.squeeze(unit_specific_time_binned_spike_counts[:, a_time_bin_idx])
            active_unit_idxs = np.where(unit_spike_counts > 0)[0]
            active_units = neuron_IDs[active_unit_idxs]
            active_aclu_spike_counts_dict = dict(zip(active_units, unit_spike_counts[active_unit_idxs]))
            active_units_list.append(active_aclu_spike_counts_dict)
        return active_units_list



    @classmethod
    def build_spike_counts_and_decoder_outputs(cls, a_decoder, epochs_df, spikes_df, time_bin_size=0.500, epoch_id_col_name: str='lap_id', debug_print=False):
        """ 
        
        - Subsets the spikes_df by the decoder's neuron_IDs.
        - Uses the decoder to decode posteriors at the specified `time_bin_size`
        
            a_decoder_name: types.DecoderName = 'long_LR'
            
        """
        ## Get a specific decoder
        
        # a_decoder_idx: int = track_templates.get_decoder_names().index(a_decoder_name)
        # a_decoder = deepcopy(track_templates.long_LR_decoder)

        neuron_IDs = deepcopy(a_decoder.neuron_IDs)
        spikes_df = deepcopy(spikes_df).spikes.sliced_by_neuron_id(neuron_IDs)
        unique_units = np.unique(spikes_df['aclu'])
        _out_decoded_time_bin_edges = {}
        _out_decoded_unit_specific_time_binned_spike_counts = {}
        _out_decoded_active_unit_lists = {}
        _out_decoded_active_p_x_given_n = {}
        _out_decoded_active_plots_data: Dict[str, RenderPlotsData]  = {}

        for a_row in epochs_df.itertuples():
            t_start = a_row.start
            t_end = a_row.stop
            time_bin_edges = np.arange(t_start, (t_end + time_bin_size), time_bin_size)
            n_time_bins = len(time_bin_edges) - 1
            assert n_time_bins > 0
            a_row_epoch_id = getattr(a_row, epoch_id_col_name) # a_row.lap_id
            
            _out_decoded_time_bin_edges[a_row_epoch_id] = time_bin_edges
            unit_specific_time_binned_spike_counts = np.array([
                np.histogram(spikes_df.loc[spikes_df['aclu'] == unit, 't_rel_seconds'], bins=time_bin_edges)[0]
                for unit in unique_units
            ])
            all_epoch_active_units_list = []
            active_units_list = []
            for a_time_bin_idx in np.arange(n_time_bins):
                unit_spike_counts = np.squeeze(unit_specific_time_binned_spike_counts[:, a_time_bin_idx])
                # normalized_unit_spike_counts = (unit_spike_counts / np.sum(unit_spike_counts))
                active_unit_idxs = np.where(unit_spike_counts > 0)[0]
                active_units = neuron_IDs[active_unit_idxs]
                active_aclu_spike_counts_dict = dict(zip(active_units, unit_spike_counts[active_unit_idxs]))
                active_units_list.append(active_aclu_spike_counts_dict)
                all_epoch_active_units_list.extend(active_units)

            _out_decoded_active_unit_lists[a_row_epoch_id] = active_units_list
            _out_decoded_unit_specific_time_binned_spike_counts[a_row_epoch_id] = unit_specific_time_binned_spike_counts

            # all_lap_active_units_list = np.unique(list(Set(all_lap_active_units_list)))
            all_epoch_active_units_list = np.unique(all_epoch_active_units_list)
            if debug_print:
                print(f'all_epoch_active_units_list: {all_epoch_active_units_list}')
            epoch_specific_spikes_df = deepcopy(spikes_df).spikes.time_sliced(t_start=t_start, t_stop=t_end).spikes.sliced_by_neuron_id(all_epoch_active_units_list)
            epoch_specific_spikes_df, neuron_id_to_new_IDX_map = epoch_specific_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()  # rebuild the fragile indicies afterwards
            _decoded_pos_outputs = a_decoder.decode(unit_specific_time_binned_spike_counts=unit_specific_time_binned_spike_counts, time_bin_size=time_bin_size, output_flat_versions=True, debug_print=False)
            _out_decoded_active_p_x_given_n[a_row_epoch_id] = _decoded_pos_outputs
            _out_decoded_active_plots_data[a_row_epoch_id] = RenderPlotsData(name=f'epoch[{a_row_epoch_id}]', spikes_df=epoch_specific_spikes_df, active_aclus=all_epoch_active_units_list)

        return (_out_decoded_time_bin_edges, _out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists, _out_decoded_active_p_x_given_n, _out_decoded_active_plots_data)


    @classmethod
    def _helper_simply_plot_posterior_in_pyqtgraph_plotitem(cls, curr_plot, image, xbin_edges, ybin_edges, debug_print:bool=False):
        """ builds the plotItems
        """
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import pyqtplot_build_image_bounds_extent
        pg.setConfigOptions(imageAxisOrder='row-major')
        pg.setConfigOptions(antialias=True)
        cmap = pg.colormap.get('jet','matplotlib')
        image_bounds_extent, x_range, y_range = pyqtplot_build_image_bounds_extent(xbin_edges, ybin_edges, margin=0.0, debug_print=debug_print)
        curr_plot.hideButtons()
        img_item = pg.ImageItem(image=image, levels=(0,1))
        curr_plot.addItem(img_item, defaultPadding=0.0)
        img_item.setImage(image, rect=image_bounds_extent, autoLevels=False)
        img_item.setLookupTable(cmap.getLookupTable(nPts=256), update=False)
        curr_plot.setRange(xRange=x_range, yRange=y_range, padding=0.0, update=False, disableAutoRange=True)
        curr_plot.setLimits(xMin=x_range[0], xMax=x_range[-1], yMin=y_range[0], yMax=y_range[-1])
        return img_item


    @classmethod
    def _perform_build_time_binned_decoder_debug_plots(cls, a_decoder, time_bin_edges, p_x_given_n, active_epoch_active_aclu_spike_counts_list, plots_data: Optional[RenderPlotsData]=None, plots_container: Optional[RenderPlots]=None, debug_print=False, name_suffix: str = 'unknown'):
        """ Builds the time-binned decoder debug plots for visualizing decoding results.
            
            Builds a multi-row plot layout containing:
                1. A spike raster plot showing active spikes across time
                2. A posterior probability plot showing P(x|n) across time bins
                3. Individual plots for each time bin showing active neuron templates
            
            Args:
                a_decoder: The decoder object used for decoding
                an_epoch_id: The ID of the epoch to plot
                _out_decoded_time_bin_edges: Dict mapping epoch IDs to time bin edges
                _out_decoded_active_p_x_given_n: Dict mapping epoch IDs to decoded posterior probabilities
                _out_decoded_unit_specific_time_binned_spike_counts: Dict mapping epoch IDs to spike counts per unit per time bin
                _out_decoded_active_unit_lists: Dict mapping epoch IDs to lists of active units per time bin
                _out_decoded_active_plots_data: Dict mapping epoch IDs to plot data objects
                debug_print: Whether to print debug information
                
            Returns:
                tuple: (window, decoder_template_objects, (active_plots, active_plots_data))
                    - window: The PyQtGraph window containing the plots
                    - decoder_template_objects: List of decoder template visualization objects
                    - active_plots: Dict mapping epoch IDs to plot containers
                    - active_plots_data: Dict mapping epoch IDs to plot data
                    
                
        Usage:
            from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.BinByBinDecodingDebugger import BinByBinDebuggingData

            a_decoder_name = 'long'

            # ## 2D:
            # ## INPUTS: results2D
            # a_decoded_result: DecodedFilterEpochsResult = deepcopy(results2D.continuous_results[a_decoder_name])
            # a_decoder = deepcopy(results2D.decoders[a_decoder_name])

            ## 1D: 
            ## INPUTS: results1D
            a_decoded_result: DecodedFilterEpochsResult = deepcopy(results1D.continuous_results[a_decoder_name])
            a_decoder = deepcopy(results1D.decoders[a_decoder_name])
            neuron_IDs = deepcopy(a_decoder.neuron_IDs)
            global_spikes_df = get_proper_global_spikes_df(curr_active_pipeline).spikes.sliced_by_neuron_id(neuron_IDs) ## only get the relevant spikes
            ## OUTPUTS: neuron_IDs, global_spikes_df, active_window_time_bins

            decoding_time_bin_size: float = a_decoded_result.decoding_time_bin_size
            single_continuous_result: SingleEpochDecodedResult = a_decoded_result.get_result_for_epoch(0) # SingleEpochDecodedResult
            decoding_bins_epochs_df: pd.DataFrame = single_continuous_result.build_pseudo_epochs_df_from_decoding_bins().epochs.get_valid_df()
            bin_by_bin_data: BinByBinDebuggingData = BinByBinDebuggingData.init_from_single_continuous_result(a_decoder=a_decoder, global_spikes_df=global_spikes_df, single_continuous_result=single_continuous_result, decoding_time_bin_size=decoding_time_bin_size, n_max_debugged_time_bins=20)
            ## OUTPUTS: bin_by_bin_data


            ## INPUTS: active_spikes_window, global_spikes_df, decoding_bins_epochs_df
            ## Slice to current window:
            active_window_t_start, active_window_t_end = active_spikes_window.active_time_window
            print(f'active_window_t_start: {active_window_t_start}, active_window_t_end: {active_window_t_end}')
            active_global_spikes_df, active_window_decoded_epochs_df, active_epoch_active_aclu_spike_counts_list, (active_window_slice_idxs, active_window_time_bin_edges, active_p_x_given_n) = bin_by_bin_data.sliced_to_current_window(active_window_t_start, active_window_t_end)

            ## OUTPUTS: active_window_slice_idxs, active_window_time_bin_edges, active_p_x_given_n

            ## OUTPUTS: active_global_spikes_df, active_window_decoded_epochs_df, active_epoch_active_aclu_spike_counts_list

            ## INPUTS: neuron_IDs, (active_global_spikes_df, active_window_decoded_epochs_df, active_aclu_spike_counts_dict_list)
            ## INPUTS: active_window_slice_idxs, active_window_time_bin_edges, active_p_x_given_n
            plots_container = PyqtgraphRenderPlots(name='PhoTest', root_plot=None) # Create a new one
            plots_data = RenderPlotsData(name=f'epoch[Test]', spikes_df=active_global_spikes_df, a_decoder=a_decoder, active_aclus=neuron_IDs, bin_by_bin_data=bin_by_bin_data)
            win, out_pf1D_decoder_template_objects, (plots_container, plots_data) = BinByBinDecodingDebugger._perform_build_time_binned_decoder_debug_plots(a_decoder=a_decoder, time_bin_edges=active_window_time_bin_edges, p_x_given_n=active_p_x_given_n, active_epoch_active_aclu_spike_counts_list=active_epoch_active_aclu_spike_counts_list,
                                                                                                                                        plots_data=plots_data, plots_container=plots_container,
                                                                                                                                        debug_print=False)
        """
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import BaseTemplateDebuggingMixin
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import new_plot_raster_plot

        # ==================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                  #
        # ==================================================================================================================== #
        
        n_epoch_time_bins = len(time_bin_edges) - 1
        if debug_print:
            print(f'n_epoch_time_bins: {n_epoch_time_bins}')

        win = pg.GraphicsLayoutWidget()
        plots = []
        out_pf1D_decoder_template_objects: List[BaseTemplateDebuggingMixin] = []

        ## Initialize a new plots_container as needed:
        if plots_container is None:
            plots_container = RenderPlots(name='_perform_build_time_binned_decoder_debug_plots', root_plot=None) # [a_lap_id]

        assert plots_data is not None,  f"`plots_data` is None, but required for access to `plots_data.spikes_df` and `plots_data.active_aclus`."

        # Epoch Active Spikes, takes up a row _______________________________________________________________ #
        spanning_spikes_raster_plot = win.addPlot(title=f"spikes_raster Plot - {name_suffix}", row=0, rowspan=1, col=0, colspan=n_epoch_time_bins)
        spanning_spikes_raster_plot.setTitle(f"spikes_raster Plot - {name_suffix}")
        plots_container.root_plot = spanning_spikes_raster_plot
        app, raster_win, plots_container, plots_data = new_plot_raster_plot(plots_data.spikes_df, plots_data.active_aclus, scatter_plot_kwargs=None, win=spanning_spikes_raster_plot, plots_data=plots_data, plots=plots_container,
                                                            scatter_app_name=f'epoch_specific_spike_raster', defer_show=True, active_context=None, add_debug_header_label=False) # RasterPlotSetupTuple
        
        win.nextRow()

        # Decoded Epoch Posterior (bin-by-bin), takes up a row _______________________________________________________________ #
        spanning_posterior_plot = win.addPlot(title="P_x_given_n Plot", row=1, rowspan=1, col=0, colspan=n_epoch_time_bins)
        spanning_posterior_plot.setTitle(f"P_x_given_n Plot - Decoded over epoch[{name_suffix}]")

        flat_p_x_given_n = deepcopy(p_x_given_n)
        cls._helper_simply_plot_posterior_in_pyqtgraph_plotitem(curr_plot=spanning_posterior_plot, image=flat_p_x_given_n, xbin_edges=np.arange(n_epoch_time_bins+1), ybin_edges=deepcopy(a_decoder.xbin))
        win.nextRow()

        # Bin-by-bin active spike templates/pf1D fields ______________________________________________________________________ #
        for a_time_bin_idx in np.arange(n_epoch_time_bins):
            active_bin_active_aclu_spike_counts_dict = active_epoch_active_aclu_spike_counts_list[a_time_bin_idx]
            active_bin_active_aclu_spike_count_values = np.array(list(active_bin_active_aclu_spike_counts_dict.values()))
            active_bin_active_aclu_bin_normalized_spike_count_values = active_bin_active_aclu_spike_count_values / np.sum(active_bin_active_aclu_spike_count_values)

            aclu_override_alpha_weights = 0.8 + (0.2 * active_bin_active_aclu_bin_normalized_spike_count_values)
            active_bin_aclus = np.array(list(active_bin_active_aclu_spike_counts_dict.keys()))
            active_solo_override_num_spikes_weights = dict(zip(active_bin_aclus, active_bin_active_aclu_bin_normalized_spike_count_values))
            active_aclu_override_alpha_weights_dict = dict(zip(active_bin_aclus, aclu_override_alpha_weights))
            if debug_print:
                print(f'a_time_bin_idx: {a_time_bin_idx}/{n_epoch_time_bins} - active_bin_aclus: {active_bin_aclus}')

            plot = win.addPlot(title=f"Plot {a_time_bin_idx+1}", row=2, rowspan=1, col=a_time_bin_idx, colspan=1)
            plot.getViewBox().setBorder(color=(200, 200, 200), width=1)
            spanning_posterior_plot.showGrid(x=True, y=True)
            x_axis = spanning_posterior_plot.getAxis('bottom')
            x_axis.setTickSpacing(major=5, minor=1)

            plots.append(plot)
            _obj: BaseTemplateDebuggingMixin = BaseTemplateDebuggingMixin.init_from_decoder(a_decoder=a_decoder, win=plot, title_str=f't={a_time_bin_idx}')
            _obj.update_base_decoder_debugger_data(
                included_neuron_ids=active_bin_aclus,
                solo_override_alpha_weights=active_aclu_override_alpha_weights_dict,
                solo_override_num_spikes_weights=active_solo_override_num_spikes_weights
            )
            
            out_pf1D_decoder_template_objects.append(_obj)

        win.nextRow()
        win.setWindowTitle(f'BinByBinDecodingDebugger - {name_suffix}')
        win.show()
        return win, out_pf1D_decoder_template_objects, (plots_container, plots_data)
    

    @classmethod
    def perform_update_time_binned_decoder_debug_plots(cls, win, decoder_template_objects, plots_container, plots_data, new_time_bin_edges, new_p_x_given_n, new_active_aclu_spike_counts_list, debug_print=False):
        """Updates an existing bin-by-bin decoder debug plot with new data.
        
        Args:
            win: PyQtGraph GraphicsLayoutWidget containing the plots
            decoder_template_objects: List of BaseTemplateDebuggingMixin objects
            plots_container: RenderPlots container with plot references
            plots_data: RenderPlotsData containing spike data
            new_time_bin_edges: Updated time bin edges array
            new_p_x_given_n: Updated posterior probabilities array
            new_active_aclu_spike_counts_list: List of dicts mapping active unit IDs to spike counts
            debug_print: Whether to print debug info
            
        Returns:
            tuple: (win, decoder_template_objects, (plots_container, plots_data))
            
            
        Usage:
        
        
            # Later when data changes:
            ## INPUTS: active_spikes_window, global_spikes_df, decoding_bins_epochs_df
            ## Slice to current window:
            active_window_t_start, active_window_t_end = active_spikes_window.active_time_window
            print(f'active_window_t_start: {active_window_t_start}, active_window_t_end: {active_window_t_end}')
            active_global_spikes_df, active_window_decoded_epochs_df, active_epoch_active_aclu_spike_counts_list, (active_window_slice_idxs, active_window_time_bin_edges, active_p_x_given_n) = bin_by_bin_data.sliced_to_current_window(active_window_t_start, active_window_t_end)


            win, template_objs, (plots_container, plots_data) = BinByBinDecodingDebugger.update_time_binned_decoder_debug_plots(
                win, out_pf1D_decoder_template_objects, plots_container, plots_data,
                new_time_bin_edges=active_window_time_bin_edges, new_p_x_given_n=active_p_x_given_n, new_active_aclu_spike_counts_list=active_epoch_active_aclu_spike_counts_list
            )

        """
        from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.SpikeRasters import new_plot_raster_plot
        
        # 1. Update the spike raster plot (first row)
        spanning_spikes_raster_plot = win.getItem(row=0, col=0)
        plots_container.root_plot = spanning_spikes_raster_plot
        # Clear and replot raster
        spanning_spikes_raster_plot.clear()
        app, raster_win, plots_container, plots_data = new_plot_raster_plot(
            plots_data.spikes_df, plots_data.active_aclus, 
            scatter_plot_kwargs=None, win=spanning_spikes_raster_plot,
            plots_data=plots_data, plots=plots_container,
            scatter_app_name='epoch_specific_spike_raster',
            defer_show=True, active_context=None, 
            add_debug_header_label=False
        )

        # 2. Update posterior plot (second row) 
        spanning_posterior_plot = win.getItem(row=1, col=0)
        spanning_posterior_plot.clear()
        n_epoch_time_bins = len(new_time_bin_edges) - 1
        flat_p_x_given_n = deepcopy(new_p_x_given_n)
        cls._helper_simply_plot_posterior_in_pyqtgraph_plotitem(
            curr_plot=spanning_posterior_plot,
            image=flat_p_x_given_n,
            xbin_edges=np.arange(n_epoch_time_bins+1),
            ybin_edges=plots_data.a_decoder.xbin
        )

        # 3. Update individual time bin template plots (third row)
        for time_bin_idx, (template_obj, active_bin_spike_counts) in enumerate(zip(decoder_template_objects, new_active_aclu_spike_counts_list)):
            if debug_print:
                print(f'Updating time bin {time_bin_idx}')
                
            active_bin_spike_count_values = np.array(list(active_bin_spike_counts.values()))
            normalized_spike_counts = active_bin_spike_count_values / np.sum(active_bin_spike_count_values)
            alpha_weights = 0.8 + (0.2 * normalized_spike_counts)
            
            active_aclus = np.array(list(active_bin_spike_counts.keys()))
            template_obj.update_base_decoder_debugger_data(
                included_neuron_ids=active_aclus,
                solo_override_alpha_weights=dict(zip(active_aclus, alpha_weights)),
                solo_override_num_spikes_weights=dict(zip(active_aclus, normalized_spike_counts))
            )

        return win, decoder_template_objects, (plots_container, plots_data)


    def update_time_binned_decoder_debug_plots(self, new_time_bin_edges, new_p_x_given_n, new_active_aclu_spike_counts_list, debug_print=False, **kwargs):
            """Updates an existing bin-by-bin decoder debug plot with new data.
            
            Args:
                win: PyQtGraph GraphicsLayoutWidget containing the plots
                decoder_template_objects: List of BaseTemplateDebuggingMixin objects
                plots_container: RenderPlots container with plot references
                plots_data: RenderPlotsData containing spike data
                new_time_bin_edges: Updated time bin edges array
                new_p_x_given_n: Updated posterior probabilities array
                new_active_aclu_spike_counts_list: List of dicts mapping active unit IDs to spike counts
                debug_print: Whether to print debug info
                
            Returns:
                tuple: (win, decoder_template_objects, (plots_container, plots_data))
                
                
            Usage:
            
            
                # Later when data changes:
                ## INPUTS: active_spikes_window, global_spikes_df, decoding_bins_epochs_df
                ## Slice to current window:
                active_window_t_start, active_window_t_end = active_spikes_window.active_time_window
                print(f'active_window_t_start: {active_window_t_start}, active_window_t_end: {active_window_t_end}')
                active_global_spikes_df, active_window_decoded_epochs_df, active_epoch_active_aclu_spike_counts_list, (active_window_slice_idxs, active_window_time_bin_edges, active_p_x_given_n) = bin_by_bin_data.sliced_to_current_window(active_window_t_start, active_window_t_end)


                win, template_objs, (plots_container, plots_data) = BinByBinDecodingDebugger.update_time_binned_decoder_debug_plots(
                    win, out_pf1D_decoder_template_objects, plots_container, plots_data,
                    new_time_bin_edges=active_window_time_bin_edges, new_p_x_given_n=active_p_x_given_n, new_active_aclu_spike_counts_list=active_epoch_active_aclu_spike_counts_list
                )

            """
            # raise NotImplementedError(f'#TODO 2025-08-22 12:14: - [ ] Not done')
            _update_output = self.perform_update_time_binned_decoder_debug_plots(win=self.ui.win, decoder_template_objects=self.plot_data.pf1D_decoder_template_objects, plots_container=self.plots, plots_data=self.plot_data,
                                                                        new_time_bin_edges=new_time_bin_edges, new_p_x_given_n=new_p_x_given_n, new_active_aclu_spike_counts_list=new_active_aclu_spike_counts_list, debug_print=debug_print,
                                                                        **kwargs)
            ## just in case it doesn't modify in place, we need to unpack and assign:
            # win, out_pf1D_decoder_template_objects, (plots_container, plots_data) = _update_output ## unpack the output
            # self.ui.win, self.plot_data.pf1D_decoder_template_objects, (self.plots, self.plot_data) = _update_output ## unpack the output
            
            return _update_output
    


    @function_attributes(short_name=None, tags=['MAIN', 'plot', 'GUI'], input_requires=[], output_provides=[], uses=['new_plot_raster_plot', 'pyqtplot_build_image_bounds_extent', 'BaseTemplateDebuggingMixin'], used_by=['cls.plot_bin_by_bin_decoding_example'], creation_date='2025-02-24 12:20', related_items=[])
    @classmethod
    def build_time_binned_decoder_debug_plots(cls, a_decoder, an_epoch_id, _out_decoded_time_bin_edges, _out_decoded_active_p_x_given_n, _out_decoded_active_unit_lists, _out_time_bin_decoded_active_plots_data, debug_print=False):
        """ Builds the time-binned decoder debug plots for visualizing decoding results.
            
            Builds a multi-row plot layout containing:
                1. A spike raster plot showing active spikes across time
                2. A posterior probability plot showing P(x|n) across time bins
                3. Individual plots for each time bin showing active neuron templates
            
            Args:
                a_decoder: The decoder object used for decoding
                an_epoch_id: The ID of the epoch to plot
                _out_decoded_time_bin_edges: Dict mapping epoch IDs to time bin edges
                _out_decoded_active_p_x_given_n: Dict mapping epoch IDs to decoded posterior probabilities
                _out_decoded_unit_specific_time_binned_spike_counts: Dict mapping epoch IDs to spike counts per unit per time bin
                _out_decoded_active_unit_lists: Dict mapping epoch IDs to lists of active units per time bin
                _out_decoded_active_plots_data: Dict mapping epoch IDs to plot data objects
                debug_print: Whether to print debug information
                
            Returns:
                tuple: (window, decoder_template_objects, (active_plots, active_plots_data))
                    - window: The PyQtGraph window containing the plots
                    - decoder_template_objects: List of decoder template visualization objects
                    - active_plots: Dict mapping epoch IDs to plot containers
                    - active_plots_data: Dict mapping epoch IDs to plot data
        """
        
        # ==================================================================================================================== #
        # BEGIN FUNCTION BODY                                                                                                  #
        # ==================================================================================================================== #
        time_bin_edges = _out_decoded_time_bin_edges[an_epoch_id]
        n_epoch_time_bins = len(time_bin_edges) - 1
        if debug_print:
            print(f'an_epoch_id: {an_epoch_id}, n_epoch_time_bins: {n_epoch_time_bins}')

        _out_time_bin_decoded_active_plots = {}

        active_epoch_active_aclu_spike_counts_list = _out_decoded_active_unit_lists[an_epoch_id]
        time_bin_edges = _out_decoded_time_bin_edges[an_epoch_id]
        most_likely_positions, p_x_given_n, most_likely_position_indicies, flat_outputs_container = _out_decoded_active_p_x_given_n[an_epoch_id]
        plots_data = _out_time_bin_decoded_active_plots_data[an_epoch_id]
        plots_container = RenderPlots(name=an_epoch_id, root_plot=None) # Create a new one
        # plots_data = RenderPlotsData(name=f'epoch[{an_epoch_id}]', spikes_df=epoch_specific_spikes_df, active_aclus=all_lap_active_units_list)
        win, out_pf1D_decoder_template_objects, (plots_container, plots_data) = cls._perform_build_time_binned_decoder_debug_plots(a_decoder=a_decoder, time_bin_edges=time_bin_edges, p_x_given_n=p_x_given_n, active_epoch_active_aclu_spike_counts_list=active_epoch_active_aclu_spike_counts_list,
                                                                                                                                    plots_data=plots_data, plots_container=plots_container,
                                                                                                                                    debug_print=False)
        ## Assign the outputs:
        _out_time_bin_decoded_active_plots[an_epoch_id] = plots_container
        _out_time_bin_decoded_active_plots_data[an_epoch_id] = plots_data


        # win.nextRow()
        win.setWindowTitle('BinByBinDecodingDebugger')
        win.show()
        return win, out_pf1D_decoder_template_objects, (_out_time_bin_decoded_active_plots, _out_time_bin_decoded_active_plots_data)


    @function_attributes(short_name=None, tags=['private', 'plot'], input_requires=[], output_provides=[], uses=['cls.build_spike_counts_and_decoder_outputs', 'cls.build_time_binned_decoder_debug_plots'], used_by=[], creation_date='2025-02-24 12:19', related_items=[])
    @classmethod
    def plot_bin_by_bin_decoding_example(cls, curr_active_pipeline, a_decoder: BasePositionDecoder, time_bin_size: float = 0.250, epoch_id_col_name: str='lap_id', an_epoch_id: int = 9, name_suffix: str=None):
        """
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_bin_by_bin_decoding_example

                ## INPUTS: time_bin_size
        time_bin_size: float = 0.500 # 500ms

        ## any (generic) directionald decoder
        # neuron_IDs = deepcopy(track_templates.any_decoder_neuron_IDs) # array([  2,   5,   8,  10,  14,  15,  23,  24,  25,  26,  31,  32,  33,  41,  49,  50,  51,  55,  58,  64,  69,  70,  73,  74,  75,  76,  78,  82,  83,  85,  86,  90,  92,  93,  96, 109])

        ## Get a specific decoder
        a_decoder_name: types.DecoderName = 'long_LR'
        a_decoder_idx: int = track_templates.get_decoder_names().index(a_decoder_name)
        a_decoder = deepcopy(track_templates.long_LR_decoder)

        ## Build the plotter:
        win, out_pf1D_decoder_template_objects, (_out_decoded_active_plots, _out_decoded_active_plots_data) = BinByBinDecodingDebugger.plot_bin_by_bin_decoding_example(curr_active_pipeline=curr_active_pipeline, track_templates=track_templates, time_bin_size=time_bin_size, a_lap_id=a_lap_id, a_decoder_name=a_decoder_name)
        
        
        
        """
        from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import LayoutScrollability, pyqtplot_build_image_bounds_extent
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.TemplateDebugger import BaseTemplateDebuggingMixin, build_pf1D_heatmap_with_labels_and_peaks, TrackTemplates

        if name_suffix is None:
            name_suffix = f'Epoch[{epoch_id_col_name}={an_epoch_id}]'
            
        neuron_IDs = deepcopy(a_decoder.neuron_IDs)
        ## OUTPUTS: neuron_IDs, global_spikes_df, active_window_time_bins
        
        # Example usage:
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        # global_spikes_df = get_proper_global_spikes_df(curr_active_pipeline).spikes.sliced_by_neuron_id(neuron_IDs) ## only get the relevant spikes
        global_spikes_df = deepcopy(curr_active_pipeline.computation_results[global_epoch_name]['computed_data'].pf1D.spikes_df)
        global_laps = deepcopy(curr_active_pipeline.filtered_sessions[global_epoch_name].laps).trimmed_to_non_overlapping()
        global_laps_epochs_df = global_laps.to_dataframe()
        global_laps_epochs_df

        ## COMPUTED: 
        a_decoder = deepcopy(a_decoder)
        (_out_decoded_time_bin_edges, _out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists, _out_decoded_active_p_x_given_n, _out_decoded_active_plots_data) = cls.build_spike_counts_and_decoder_outputs(a_decoder=a_decoder, epochs_df=global_laps_epochs_df, epoch_id_col_name=epoch_id_col_name, spikes_df=global_spikes_df, time_bin_size=time_bin_size)
        win, out_pf1D_decoder_template_objects, (_out_decoded_active_plots, _out_decoded_active_plots_data) = cls.build_time_binned_decoder_debug_plots(a_decoder=a_decoder, an_epoch_id=an_epoch_id, _out_decoded_time_bin_edges=_out_decoded_time_bin_edges, _out_decoded_active_p_x_given_n=_out_decoded_active_p_x_given_n, 
                                                                                                                                                        _out_decoded_active_unit_lists=_out_decoded_active_unit_lists, _out_time_bin_decoded_active_plots_data=_out_decoded_active_plots_data, debug_print=True)
        print(f"Returned window: {win}")
        print(f"Returned decoder objects: {out_pf1D_decoder_template_objects}")

        # plots_container = PyqtgraphRenderPlots(name=f'PhoTest_{name_suffix}', root_plot=None) # Create a new one
        # plots_data = RenderPlotsData(name=f'epoch[{name_suffix}]', spikes_df=global_spikes_df, a_decoder=a_decoder, active_aclus=neuron_IDs, bin_by_bin_data=bin_by_bin_data)
        # win, out_pf1D_decoder_template_objects, (plots_container, plots_data) = BinByBinDecodingDebugger._perform_build_time_binned_decoder_debug_plots(a_decoder=a_decoder, time_bin_edges=active_window_time_bin_edges, p_x_given_n=active_p_x_given_n, active_epoch_active_aclu_spike_counts_list=active_epoch_active_aclu_spike_counts_list,
        #                                                                                                                             plots_data=plots_data, plots_container=plots_container,
        #                                                                                                                             debug_print=False, name_suffix=name_suffix)
        # bin_by_bin_debugger: BinByBinDecodingDebugger = BinByBinDecodingDebugger.init_from_builder_classmethod(win=win, pf1D_decoder_template_objects=out_pf1D_decoder_template_objects, plots_container=plots_container, plot_data=plots_data)
        
        # return bin_by_bin_debugger 
        return win, out_pf1D_decoder_template_objects, (_out_decoded_active_plots, _out_decoded_active_plots_data)
    


    # @classmethod
    # def init_from_track_templates(cls, track_templates):
    #     raise NotImplementedError(f'#TODO 2025-02-24 12:30: - [ ] Does not yet return any kind of object.')


    @classmethod
    def init_from_decoded_reusult(cls, track_templates):
        raise NotImplementedError(f'#TODO 2025-02-24 12:30: - [ ] Does not yet return any kind of object.')




    @classmethod
    def init_from_builder_classmethod(cls, win, pf1D_decoder_template_objects, plots_container, plot_data, name_suffix: str='test') -> "BinByBinDecodingDebugger":
        _obj_dict = dict()
        if plots_container is None:
            plots_container = PyqtgraphRenderPlots(name=f'PhoTest_{name_suffix}', root_plot=None) # Create a new one
            
        _obj_dict['plots'] = plots_container

        if plot_data is None:
            plot_data = RenderPlotsData(name=f'epoch[{name_suffix}]', pf1D_decoder_template_objects=pf1D_decoder_template_objects) # , spikes_df=active_global_spikes_df, a_decoder=a_decoder, active_aclus=neuron_IDs, bin_by_bin_data=bin_by_bin_data
        if isinstance(plot_data, dict):
            # looks like a separate entry for each time bin? Weird
            plot_data = RenderPlotsData(name=f'epoch[{name_suffix}]', pf1D_decoder_template_objects=pf1D_decoder_template_objects)
            

        _obj_dict['plot_data'] = plot_data
        _obj = cls(**_obj_dict)
        _obj.plot_data.pf1D_decoder_template_objects = pf1D_decoder_template_objects
        _obj.ui.win = win
        _obj.params.on_update_fcn = None
        return _obj 



    @classmethod
    def init_from_plot_bin_by_bin_decoding(cls, win, pf1D_decoder_template_objects, _out_decoded_active_plots, _out_decoded_active_plots_data, name_suffix: str='test') -> "BinByBinDecodingDebugger":
        """
        
        """
        _obj_dict = dict()
        plots_container = PyqtgraphRenderPlots(name=f'PhoTest_{name_suffix}', root_plot=None, _out_decoded_active_plots=_out_decoded_active_plots) # Create a new one
        _obj_dict['plots'] = plots_container

        plot_data = RenderPlotsData(name=f'epoch[{name_suffix}]', pf1D_decoder_template_objects=pf1D_decoder_template_objects, _out_decoded_active_plots_data=_out_decoded_active_plots_data) # , spikes_df=active_global_spikes_df, a_decoder=a_decoder, active_aclus=neuron_IDs, bin_by_bin_data=bin_by_bin_data
        _obj_dict['plot_data'] = plot_data
        _obj = cls(**_obj_dict)
        _obj.plot_data.pf1D_decoder_template_objects = pf1D_decoder_template_objects
        _obj.ui.win = win
        _obj.params.on_update_fcn = None
        return _obj 



    @function_attributes(short_name=None, tags=['USEFUL', 'unused', 'debug', 'visualizztion', 'SpikeRasterWindow'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-05-14 14:01', related_items=[])
    @classmethod
    def plot_attached_BinByBinDecodingDebugger(cls, spike_raster_window, curr_active_pipeline, a_decoder: BasePositionDecoder, a_decoded_result: Union[DecodedFilterEpochsResult, SingleEpochDecodedResult], n_max_debugged_time_bins:int=25, name_suffix: str = 'unknoown'):
        """ 
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import plot_attached_BinByBinDecodingDebugger

        ## INPUTS: a_decoder, a_decoded_result
            
        a_decoder_name: str = 'long_LR'
        a_decoder = all_directional_pf1D_Decoder_dict[a_decoder_name]
        a_decoded_result = a_continuously_decoded_dict[a_decoder_name]

        ## INPUTS: a_decoder, a_decoded_result
        bin_by_bin_debugger, win, out_pf1D_decoder_template_objects, (plots_container, plots_data), _on_update_fcn = BinByBinDecodingDebugger.plot_attached_BinByBinDecodingDebugger(spike_raster_window, curr_active_pipeline, a_decoder=a_decoder, a_decoded_result=a_decoded_result)

        
        """
        from pyphocorehelpers.DataStructure.RenderPlots.PyqtgraphRenderPlots import PyqtgraphRenderPlots
        from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.BinByBinDecodingDebugger import BinByBinDebuggingData, BinByBinDecodingDebugger


        neuron_IDs = deepcopy(a_decoder.neuron_IDs)
        global_spikes_df = get_proper_global_spikes_df(curr_active_pipeline).spikes.sliced_by_neuron_id(neuron_IDs) ## only get the relevant spikes
        ## OUTPUTS: neuron_IDs, global_spikes_df, active_window_time_bins
        active_2d_plot = spike_raster_window.spike_raster_plt_2d
        active_spikes_window = active_2d_plot.spikes_window

        if isinstance(a_decoded_result, SingleEpochDecodedResult):
            single_continuous_result = a_decoded_result ## already have this
            decoding_time_bin_size: float = single_continuous_result.time_bin_container.edge_info.step
        else:
            ## extract it
            single_continuous_result: SingleEpochDecodedResult = a_decoded_result.get_result_for_epoch(0) # SingleEpochDecodedResult            
            decoding_time_bin_size: float = a_decoded_result.decoding_time_bin_size


        # decoding_bins_epochs_df: pd.DataFrame = single_continuous_result.build_pseudo_epochs_df_from_decoding_bins().epochs.get_valid_df()
        bin_by_bin_data: BinByBinDebuggingData = BinByBinDebuggingData.init_from_single_continuous_result(a_decoder=a_decoder, global_spikes_df=global_spikes_df, single_continuous_result=single_continuous_result, decoding_time_bin_size=decoding_time_bin_size, n_max_debugged_time_bins=n_max_debugged_time_bins)
        ## OUTPUTS: bin_by_bin_data

        ## INPUTS: active_spikes_window, global_spikes_df, decoding_bins_epochs_df
        ## Slice to current window:
        active_window_t_start, active_window_t_end = active_spikes_window.active_time_window
        print(f'active_window_t_start: {active_window_t_start}, active_window_t_end: {active_window_t_end}')
        active_global_spikes_df, active_window_decoded_epochs_df, active_epoch_active_aclu_spike_counts_list, (active_window_slice_idxs, active_window_time_bin_edges, active_p_x_given_n) = bin_by_bin_data.sliced_to_current_window(active_window_t_start, active_window_t_end)

        ## OUTPUTS: active_window_slice_idxs, active_window_time_bin_edges, active_p_x_given_n

        ## OUTPUTS: active_global_spikes_df, active_window_decoded_epochs_df, active_epoch_active_aclu_spike_counts_list

        ## INPUTS: neuron_IDs, (active_global_spikes_df, active_window_decoded_epochs_df, active_aclu_spike_counts_dict_list)
        ## INPUTS: active_window_slice_idxs, active_window_time_bin_edges, active_p_x_given_n
        plots_container = PyqtgraphRenderPlots(name=f'PhoTest_{name_suffix}', root_plot=None) # Create a new one
        plots_data = RenderPlotsData(name=f'epoch[{name_suffix}]', spikes_df=active_global_spikes_df, a_decoder=a_decoder, active_aclus=neuron_IDs, bin_by_bin_data=bin_by_bin_data)
        win, out_pf1D_decoder_template_objects, (plots_container, plots_data) = BinByBinDecodingDebugger._perform_build_time_binned_decoder_debug_plots(a_decoder=a_decoder, time_bin_edges=active_window_time_bin_edges, p_x_given_n=active_p_x_given_n, active_epoch_active_aclu_spike_counts_list=active_epoch_active_aclu_spike_counts_list,
                                                                                                                                    plots_data=plots_data, plots_container=plots_container,
                                                                                                                                    debug_print=False, name_suffix=name_suffix)
        bin_by_bin_debugger: BinByBinDecodingDebugger = BinByBinDecodingDebugger.init_from_builder_classmethod(win=win, pf1D_decoder_template_objects=out_pf1D_decoder_template_objects, plots_container=plots_container, plot_data=plots_data)
        

        

        # Later when data changes:
        def _on_update_fcn(*args, **kwargs):
            """ captures: active_spikes_window, bin_by_bin_debugger, bin_by_bin_data 
            """
            ## INPUTS: active_spikes_window, global_spikes_df, decoding_bins_epochs_df
            ## Slice to current window:
            active_window_t_start, active_window_t_end = active_spikes_window.active_time_window
            print(f'active_window_t_start: {active_window_t_start}, active_window_t_end: {active_window_t_end}')
            active_global_spikes_df, active_window_decoded_epochs_df, active_epoch_active_aclu_spike_counts_list, (active_window_slice_idxs, active_window_time_bin_edges, active_p_x_given_n) = bin_by_bin_data.sliced_to_current_window(active_window_t_start, active_window_t_end)
            # win, out_pf1D_decoder_template_objects, (plots_container, plots_data) = BinByBinDecodingDebugger.perform_update_time_binned_decoder_debug_plots(win, out_pf1D_decoder_template_objects, plots_container, plots_data, new_time_bin_edges=active_window_time_bin_edges, new_p_x_given_n=active_p_x_given_n, new_active_aclu_spike_counts_list=active_epoch_active_aclu_spike_counts_list)
            _update_output = bin_by_bin_debugger.update_time_binned_decoder_debug_plots(new_time_bin_edges=active_window_time_bin_edges, new_p_x_given_n=active_p_x_given_n, new_active_aclu_spike_counts_list=active_epoch_active_aclu_spike_counts_list)
            # win, out_pf1D_decoder_template_objects, (plots_container, plots_data) = _update_output

        bin_by_bin_debugger.params.on_update_fcn = _on_update_fcn

        ## connect the update event

        # Perform Initial (one-time) update from source -> controlled:
        # active_matplotlib_view_widget.on_window_changed(active_spikes_window.active_window_start_time, active_spikes_window.active_window_end_time)
        # sync_connection = active_2d_plot.window_scrolled.connect(active_matplotlib_view_widget.on_window_changed)
        # active_2d_plot.ui.connections[identifier] = sync_connection # add the connection to the connections array

        ## idk if this will work:
        _on_update_fcn(active_spikes_window.active_window_start_time, active_spikes_window.active_window_end_time)
        sync_connection = active_2d_plot.window_scrolled.connect(_on_update_fcn)
        active_2d_plot.ui.connections['bin_by_bin_debugger'] = sync_connection # add the connection to the connections array

        ## END def _on_update_fcn()...
        return bin_by_bin_debugger, win, out_pf1D_decoder_template_objects, (plots_container, plots_data), _on_update_fcn




    ## OUTPUTS: _out_decoded_time_bin_edges, _out_decoded_unit_specific_time_binned_spike_counts, _out_decoded_active_unit_lists, _out_decoded_active_p_x_given_n



