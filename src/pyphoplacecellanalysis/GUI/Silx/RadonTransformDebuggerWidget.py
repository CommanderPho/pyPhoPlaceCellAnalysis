import functools
from pathlib import Path
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
# pd.options.mode.dtype_backend = 'pyarrow' # use new pyarrow backend instead of numpy
from attrs import define, field, fields, Factory
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
import nptyping as ND
from nptyping import NDArray
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy
import matplotlib as mpl

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from neuropy.core.epoch import ensure_dataframe
from neuropy.analyses.decoders import RadonTransformDebugValue

from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult, SingleEpochDecodedResult
from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import get_radon_transform
from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import FixedCustomColormaps
from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PhoPublicationFigureHelper

from silx.gui import qt
from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.plot import PlotWindow, ImageView
from silx.gui.plot.Profile import ProfileToolBar

from silx.gui.plot.tools.roi import RegionOfInterestManager
from silx.gui.plot.tools.roi import RegionOfInterestTableWidget
from silx.gui.plot.tools.roi import RoiModeSelectorAction
from silx.gui.plot.items.roi import RectangleROI, BandROI, LineROI
from silx.gui.plot.items import LineMixIn, SymbolMixIn, FillMixIn
from silx.gui.plot.actions import control as control_actions

from silx.gui.plot.ROIStatsWidget import ROIStatsWidget
from silx.gui.plot.StatsWidget import UpdateModeWidget
from silx.gui.plot import Plot2D
from silx.gui.plot.items import Curve
from silx.gui.plot.items import ImageData
from silx.gui.colors import Colormap
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, ScalarFormatter

""" 

Uses Silx


"""

@metadata_attributes(short_name=None, tags=[''], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2024-08-13 00:00', related_items=[])
@define(slots=False)
class RadonDebugValue:
    """ Values for a single epoch. Class to hold debugging information for a transformation process """
    # p_x_given_n: NDArray = field()
    # epoch_info_tuple: Tuple = field()	

    active_decoded_epoch_container: SingleEpochDecodedResult = field()
    active_debug_info: RadonTransformDebugValue = field()
    
    score: float = field()
    velocity: float = field()
    intercept: float = field()

    active_num_neighbors: int = field(default=None)
    active_neighbors_arr: List = field(default=None)

    start_point: Tuple[float, float] = field(default=None)
    end_point: Tuple[float, float] = field(default=None)
    band_width: float = field(default=None)

    @property
    def p_x_given_n(self) -> NDArray:
        """The  p_x_given_n: NDArray property."""
        return self.active_decoded_epoch_container.p_x_given_n
    @p_x_given_n.setter
    def  p_x_given_n(self, value):
        self.active_decoded_epoch_container.p_x_given_n = value
    
    @property
    def epoch_info_tuple(self) -> Tuple:
        """The  p_x_given_n: NDArray property."""
        return self.active_decoded_epoch_container.epoch_info_tuple
    @epoch_info_tuple.setter
    def  epoch_info_tuple(self, value):
        self.active_decoded_epoch_container.epoch_info_tuple = value
    
    @property
    def epoch_data_index(self) -> int:
        """The epoch_data_index for the computed epoch."""
        return self.active_decoded_epoch_container.epoch_data_index
    

    
    
def compute_score(arr, y_line):
    n_lines = 1
    y_line = np.rint(y_line).astype("int") # round to nearest integer
    
    t = np.arange(arr.shape[1])
    n_t = arr.shape[1]
    # tmid = (nt + 1) / 2 - 1

    pos = np.arange(arr.shape[0])
    n_pos = len(pos)
    # pmid = (npos + 1) / 2 - 1

    # t_mat = np.tile(t, (n_lines, 1))
    posterior = np.zeros((n_lines, n_t))

    # if line falls outside of array in a given bin, replace that with median posterior value of that bin across all positions
    t_out = np.where((y_line < 0) | (y_line > n_pos - 1))
    t_in = np.where((y_line >= 0) & (y_line <= n_pos - 1))
    posterior[t_out] = np.median(arr[:, t_out[1]], axis=0)
    posterior[t_in] = arr[y_line[t_in], t_in[1]]

    old_settings = np.seterr(all="ignore")
    posterior_mean = np.nanmean(posterior, axis=1)
    return posterior_mean


def roi_radon_transform_score(arr):
    """ a stats function that takes the ROI and returns the radon transform score """
    # print(f'np.shape(arr): {np.shape(arr)}')
    # return np.nanmean(arr, axis=1)
    # print(f'np.sum(np.isnan(arr)): {np.sum(np.isnan(arr))}')
    column_medians = np.nanmedian(arr, axis=0)
    filled_arr = [arr[:,i].filled(column_medians[i]) for i in np.arange(np.shape(arr)[1])]
    return np.nanmean(filled_arr)



# decoder_laps_radon_transform_df_dict
# │   ├── decoder_laps_radon_transform_df_dict: dict
# 	│   ├── long_LR: pandas.core.frame.DataFrame (children omitted) - (84, 4)
# 	│   ├── long_RL: pandas.core.frame.DataFrame (children omitted) - (84, 4)
# 	│   ├── short_LR: pandas.core.frame.DataFrame (children omitted) - (84, 4)
# 	│   ├── short_RL: pandas.core.frame.DataFrame (children omitted) - (84, 4)
# │   ├── decoder_laps_radon_transform_extras_dict: dict
# 	│   ├── long_LR: list - (1, 1, 2, 84)
# 	│   ├── long_RL: list - (1, 1, 2, 84)
# 	│   ├── short_LR: list - (1, 1, 2, 84)
# 	│   ├── short_RL: list - (1, 1, 2, 84)

# decoder_ripple_radon_transform_df_dict 
# a_radon_transform_output = np.squeeze(deepcopy(decoder_laps_radon_transform_extras_dict['long_LR'])) # collapse singleton dimensions with np.squeeze: (1, 1, 2, 84) -> (2, 84) # (2, n_epochs)


# np.shape(a_radon_transform_output)

# np.squeeze(a_radon_transform_output).shape
# len(a_radon_transform_output)


# ---------------------------------------------------------------------------- #
#                            Widgets/Visual Classes                            #
# ---------------------------------------------------------------------------- #

# ==================================================================================================================== #
# Main Conainer Object                                                                                                 #
# ==================================================================================================================== #

# Define a simple on_setattr hook
# def always_capitalize(instance, attribute, new_value):
#     if isinstance(new_value, str):
#         return new_value.capitalize()
#     return new_value

from pyphoplacecellanalysis.GUI.Silx.silx_helpers import AutoHideToolBar, _RoiStatsDisplayExWindow, _RoiStatsWidget

def on_set_active_decoder_name_changed(instance, attribute, new_value):
    print(f'on_set_active_decoder_name_changed(new_value: {new_value})')
    # if isinstance(new_value, str):
    #     return new_value.capitalize()
    is_valid_name: bool = new_value in instance.decoder_filter_epochs_decoder_result_dict.keys()
    if not is_valid_name:
        print(f'\tname: "{new_value}" is not a valid decoder name. valid names: {list(instance.decoder_filter_epochs_decoder_result_dict.keys())}. not changing')
        return instance.active_decoder_name # return existing value to prevent update
    
    return new_value


def on_set_active_epoch_idx_changed(instance, attribute, new_value):
    print(f'on_set_epoch_idx_changed(new_value: {new_value})')
    new_epoch_idx: int = int(new_value)
    _ = instance.update_epoch_idx(active_epoch_idx=new_epoch_idx) ## change the index
    instance.refresh_overlays()
    print(f'\tdone.')
    return new_value


@metadata_attributes(short_name=None, tags=['radon', 'debugger', 'gui', 'Silx'], input_requires=[], output_provides=[], uses=['Silx'], used_by=[], creation_date='2024-08-13 00:00', related_items=[])
@define(slots=False, repr=False)
class RadonTransformDebugger:
    """ interactive debugger of Radon Transforms computed on Posteriors using Silx
    
    from pyphoplacecellanalysis.GUI.Silx.RadonTransformDebuggerWidget import RadonTransformDebugger, RadonDebugValue

    """
    pos_bin_size: float = field()
    decoder_filter_epochs_decoder_result_dict: Dict = field()
    decoder_radon_transform_extras_dict: Dict = field()
    
    active_decoder_name: str = field(default='long_LR') # , on_setattr=on_set_active_decoder_name_changed
    _active_epoch_idx: int = field(default=3) # , on_setattr=on_set_active_epoch_idx_changed
    _active_epoch_radon_values: Optional[RadonDebugValue] = field(default=None)

    window: _RoiStatsDisplayExWindow = field(default=None)
    _band_roi: BandROI = field(default=None)

    xbin: NDArray = field(default=None)
    xbin_centers: NDArray = field(default=None)

    epoch_comments: Dict[Tuple[str, float], str] = field(default=Factory(dict))
    _comment_line_edit: Optional[Any] = field(default=None, eq=False)
    _comment_dock: Optional[Any] = field(default=None, eq=False)
    _comment_loading: bool = field(default=False, eq=False)
    _last_comment_key: Optional[Tuple[str, float]] = field(default=None, eq=False)

    posterior_heatmap_imshow_kwargs: Dict = field(default=Factory(lambda: dict(
        cmap=FixedCustomColormaps.get_custom_greyscale_with_low_values_dropped_cmap(low_value_cutoff=0.01, full_opacity_threshold=0.25),
    )))
    overlay_label_color: str = field(default='white')
    radon_debugging_labels: bool = field(default=True)
    should_draw_time_bin_boundaries: bool = field(default=True)
    time_bin_edges_display_kwargs: Dict = field(default=Factory(lambda: dict(color='grey', alpha=0.5, linewidth=1.5)))


    @property
    def active_epoch_idx(self):
        """The active_epoch_idx property."""
        return self._active_epoch_idx
    @active_epoch_idx.setter
    def active_epoch_idx(self, value):
        # value = on_set_active_epoch_idx_changed(self, None, new_value=value)
        self._active_epoch_idx = value
        # if self.window is not None:
        #     self.update_GUI() # update the GUI, hopefuly it exists


    @property
    def result(self) -> DecodedFilterEpochsResult:
        return self.decoder_filter_epochs_decoder_result_dict[self.active_decoder_name]


    @property
    def active_filter_epochs(self) -> pd.DataFrame:
        return ensure_dataframe(self.result.active_filter_epochs)


    @property
    def active_epoch_start_t(self) -> float:
        """ Start time of the currently plotted epoch (seconds). """
        return float(self.active_filter_epochs['start'].iloc[self.active_epoch_idx])


    @property
    def time_bin_size(self) -> float:
        return float(self.result.decoding_time_bin_size)


    def _radon_transform_extras_tuple(self):
        """ Unwrap stored radon extras to `(num_neighbours, neighbors_arr, ...)`.

        Export stores `df, *extras = compute_radon_transforms(...)`, which nests as `[[(num_neighbours, neighbors_arr, debug_info)]]` — lengths `(1, 1, 3, n_epochs)`.
        `np.squeeze` cannot build one array from that because `neighbors_arr` and `debug_info` are inhomogeneous across epochs.
        """
        payload = self.decoder_radon_transform_extras_dict[self.active_decoder_name]
        while isinstance(payload, (list, tuple)) and (len(payload) == 1) and isinstance(payload[0], (list, tuple)) and (not isinstance(payload[0], np.ndarray)):
            payload = payload[0]
        if isinstance(payload, np.ndarray):
            payload = np.squeeze(payload) # older homogeneous extras, shape (1, 1, 2, n_epochs) -> (2, n_epochs)
        return payload


    @property
    def num_neighbours(self) -> NDArray:
        return self._radon_transform_extras_tuple()[0]
    
    @property
    def neighbors_arr(self) -> NDArray:
        return self._radon_transform_extras_tuple()[1]
    
    @property
    def stats_measures(self) -> List[Tuple]:
        """define stats to display."""
        return [
            # ('sum', np.sum),
            # ('mean', np.mean),
            ('shape', np.shape),
            ('score', roi_radon_transform_score),
            ('prev_score', (lambda arr: self.active_radon_values.epoch_info_tuple.score)),
            ('prev_shape', (lambda arr: np.shape(self.active_radon_values.active_neighbors_arr))),
        ]


    @property
    def active_radon_values(self) -> RadonDebugValue:
        """ value for current index """
        # a_posterior, (start_point, end_point, band_width), (active_num_neighbors, active_neighbors_arr) = self.on_update_epoch_idx(active_epoch_idx=self.active_epoch_idx)
        # return RadonDebugValue(a_posterior=a_posterior, active_epoch_info_tuple=active_epoch_info_tuple, start_point=start_point, end_point=end_point, band_width=band_width, active_num_neighbors=active_num_neighbors, active_neighbors_arr=active_neighbors_arr)
        if (self._active_epoch_radon_values is not None) and (self._active_epoch_radon_values.epoch_data_index == self._active_epoch_idx):
            # recompute not needed. Return the existing `self._active_epoch_radon_values`
            return self._active_epoch_radon_values
        else:
            # needs a recompute:
            self._active_epoch_radon_values = self.update_epoch_idx(active_epoch_idx=self.active_epoch_idx) ## update to the new value
            assert ((self._active_epoch_radon_values is not None) and (self._active_epoch_radon_values.epoch_data_index == self._active_epoch_idx)), f"self._active_epoch_radon_values.epoch_data_index: {self._active_epoch_radon_values.epoch_data_index} != self._active_epoch_idx: {self._active_epoch_idx}"
            return self._active_epoch_radon_values


    @classmethod
    def matplotlib_cmap_to_silx_colormap(cls, mpl_cmap, vmin: float = 0, vmax: Optional[float] = None, n_colors: int = 256) -> Colormap:
        """ Convert a matplotlib colormap (or silx Colormap / name string) into a silx Colormap with optional vmin/vmax. """
        if isinstance(mpl_cmap, Colormap):
            if vmin is not None:
                mpl_cmap.setVMin(vmin)
            if vmax is not None:
                mpl_cmap.setVMax(vmax)
            return mpl_cmap
        if isinstance(mpl_cmap, str):
            return Colormap(name=mpl_cmap, vmin=vmin, vmax=vmax)
        # matplotlib Colormap / LinearSegmentedColormap: sample RGBA LUT for silx (preserves alpha)
        lut = np.asarray(mpl_cmap(np.linspace(0.0, 1.0, int(n_colors))), dtype=float)
        return Colormap(colors=lut, vmin=vmin, vmax=vmax)


    @classmethod
    def perform_add_real_space_posterior(cls, a_plot, p_x_given_n: NDArray, active_time_bin_edges: NDArray, xbin: NDArray, time_bin_size: float, pos_bin_size: float, legend_key:str='p_x_given_n', resetzoom: bool=True, debug_print=False, posterior_heatmap_imshow_kwargs: Optional[Dict]=None):
        """ 
        
        active_time_bin_edges = deepcopy(dbgr.result.time_bin_edges[dbgr.active_epoch_idx])
        p_x_given_n = deepcopy(dbgr.active_radon_values.p_x_given_n)
        new_image = perform_add_real_space_posterior(a_plot=new_plot, p_x_given_n=p_x_given_n, active_time_bin_edges=active_time_bin_edges, xbin=xbin, time_bin_size=time_bin_size, pos_bin_size=pos_bin_size)

        """
        if posterior_heatmap_imshow_kwargs is None:
            posterior_heatmap_imshow_kwargs = {}
        ## END if posterior_heatmap_imshow_kwargs is None...
        default_cmap = FixedCustomColormaps.get_custom_greyscale_with_low_values_dropped_cmap(low_value_cutoff=0.01, full_opacity_threshold=0.25)
        mpl_cmap = posterior_heatmap_imshow_kwargs.get('cmap', default_cmap)
        vmin = posterior_heatmap_imshow_kwargs.get('vmin', 0)
        vmax = posterior_heatmap_imshow_kwargs.get('vmax', None)
        a_cmap = cls.matplotlib_cmap_to_silx_colormap(mpl_cmap=mpl_cmap, vmin=vmin, vmax=vmax)
        img_origin = (active_time_bin_edges[0], xbin[0]) # (origin X, origin Y)
        img_scale = (time_bin_size, pos_bin_size) # ??
        if debug_print:
            print(f'img_origin: {img_origin}')
            print(f'img_scale: {img_scale}')

        label_kwargs = dict(xlabel='t (sec)', ylabel='x (cm)')
        # label_kwargs = dict(xlabel='t (bin)', ylabel='x (bin)')

        # replace=False: silx replace=True deletes ALL other images (not just this legend).
        new_image: ImageData = a_plot.addImage(p_x_given_n, legend=legend_key, replace=False, z=0, colormap=a_cmap, origin=img_origin, scale=img_scale, **label_kwargs, resetzoom=resetzoom) # , colormap="viridis", vmin=0, vmax=1
        return new_image


    def add_real_space_posterior(self, a_plot, legend_key:str='p_x_given_n', resetzoom: bool=True, debug_print=False):
        """
        new_image_data: ImageData = dbgr.add_real_space_posterior(a_plot=new_plot)

        """
        active_time_bin_edges = deepcopy(self.result.time_bin_edges[self.active_epoch_idx])
        p_x_given_n = deepcopy(self.active_radon_values.p_x_given_n)
        return self.perform_add_real_space_posterior(a_plot=a_plot, p_x_given_n=p_x_given_n, active_time_bin_edges=active_time_bin_edges, xbin=self.xbin, time_bin_size=self.time_bin_size, pos_bin_size=self.pos_bin_size, legend_key=legend_key, resetzoom=resetzoom, debug_print=debug_print, posterior_heatmap_imshow_kwargs=self.posterior_heatmap_imshow_kwargs)


    def _get_image_origin_scale(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """ Same origin/scale used by the posterior image: (time_bin_edges[0], xbin[0]) and (dt, dx). """
        active_time_bin_edges = deepcopy(self.result.time_bin_edges[self.active_epoch_idx])
        img_origin = (float(active_time_bin_edges[0]), float(self.xbin[0]))
        img_scale = (float(self.time_bin_size), float(self.pos_bin_size))
        return img_origin, img_scale


    @classmethod
    def build_scoring_band_mask(cls, best_y_line_idxs: NDArray, n_pos: int, n_neighbours: int) -> NDArray:
        """ Vertical scoring window used by radon_transform: rows `best_y_line_idxs[ci] ± n_neighbours`, clipped.

        Out-of-bounds columns (line index outside [0, n_pos)) are left as NaN — those columns use the median fill in compute_score.
        """
        best_y_line_idxs = np.asarray(best_y_line_idxs).astype(int)
        n_t: int = len(best_y_line_idxs)
        mask = np.full((n_pos, n_t), np.nan, dtype=float)
        for ci in np.arange(n_t):
            ri: int = int(best_y_line_idxs[ci])
            if (ri < 0) or (ri > (n_pos - 1)):
                continue
            lo: int = max(0, ri - int(n_neighbours))
            hi: int = min(n_pos - 1, ri + int(n_neighbours))
            mask[lo:(hi + 1), ci] = 1.0
        ## END for ci in np.arange(n_t)....

        return mask


    @classmethod
    def iter_scoring_band_polygons(cls, mask: NDArray, origin: Tuple[float, float], scale: Tuple[float, float]):
        """ Yield stair-step (x, y) polygons covering contiguous in-band columns (bin edges in real space). """
        ox, oy = float(origin[0]), float(origin[1])
        sx, sy = float(scale[0]), float(scale[1])
        n_pos, n_t = int(mask.shape[0]), int(mask.shape[1])
        ci: int = 0
        while ci < n_t:
            rows = np.where(np.isfinite(mask[:, ci]))[0]
            if len(rows) == 0:
                ci += 1
                continue
            run_cols: List[Tuple[int, int, int]] = []
            while ci < n_t:
                rows = np.where(np.isfinite(mask[:, ci]))[0]
                if len(rows) == 0:
                    break
                run_cols.append((ci, int(rows[0]), int(rows[-1])))
                ci += 1
            ## END while ci < n_t....

            lower_x: List[float] = []
            lower_y: List[float] = []
            upper_x: List[float] = []
            upper_y: List[float] = []
            for col_i, lo, hi in run_cols:
                x0 = ox + (col_i * sx)
                x1 = ox + ((col_i + 1) * sx)
                y0 = oy + (lo * sy)
                y1 = oy + ((hi + 1) * sy)
                lower_x.extend([x0, x1])
                lower_y.extend([y0, y0])
                upper_x.extend([x0, x1])
                upper_y.extend([y1, y1])
            ## END for col_i, lo, hi in run_cols....

            xs = np.asarray(lower_x + upper_x[::-1], dtype=float)
            ys = np.asarray(lower_y + upper_y[::-1], dtype=float)
            yield xs, ys
        ## END while ci < n_t....


    def add_time_bin_xgrid(self, a_plot, legend_key: str = 'time_bin_edge', debug_print=False):
        """ Draw vertical lines at each time-bin edge (DecodedEpochSlices-style xgrid) as silx Curves so they survive replot/export. """
        # Drop previous edge curves (same legend-prefix cleanup as scoring_band).
        for item in list(a_plot.getItems()):
            name = item.getName() if hasattr(item, 'getName') else ''
            if isinstance(name, str) and name.startswith(legend_key):
                a_plot.removeItem(item)
        ## END for item in list(a_plot.getItems())....

        if not self.should_draw_time_bin_boundaries:
            return []

        time_bin_edges = np.asarray(self.result.time_bin_edges[self.active_epoch_idx], dtype=float)
        y0: float = float(self.xbin[0])
        y1: float = float(self.xbin[-1])
        display_kwargs = dict(self.time_bin_edges_display_kwargs)
        line_color = display_kwargs.get('color', 'grey')
        line_alpha = float(display_kwargs.get('alpha', 0.5))
        line_width = float(display_kwargs.get('linewidth', 1.5))
        if debug_print:
            print(f'time_bin_xgrid: n_edges={len(time_bin_edges)}, y=[{y0}, {y1}], color={line_color}, alpha={line_alpha}')

        curves = []
        for edge_idx, edge_t in enumerate(time_bin_edges):
            edge_curve: Curve = a_plot.addCurve(x=np.array([edge_t, edge_t], dtype=float), y=np.array([y0, y1], dtype=float), legend=f'{legend_key}_{edge_idx}', color=line_color, linestyle='-', linewidth=line_width, symbol=None, replace=False, z=0.5)
            edge_curve.setAlpha(alpha=line_alpha)
            curves.append(edge_curve)
        ## END for edge_idx, edge_t in enumerate(time_bin_edges)....

        return curves


    def add_scoring_band_overlay(self, a_plot, legend_key: str = 'scoring_band', debug_print=False):
        """ Overlay the vertical neighbor window that enters the radon score.

        Drawn as filled stair-step shape(s) on top of the posterior. An ImageRgba overlay is unreliable with the
        matplotlib backend (often invisible on top of ImageData); shapes composite correctly.
        """
        a_debug_info: RadonTransformDebugValue = self.active_radon_values.active_debug_info
        n_neighbours: int = int(self.active_radon_values.active_num_neighbors)
        mask = self.build_scoring_band_mask(best_y_line_idxs=a_debug_info.best_y_line_idxs, n_pos=int(a_debug_info.n_pos), n_neighbours=n_neighbours)
        img_origin, img_scale = self._get_image_origin_scale()
        if debug_print:
            print(f'scoring_band mask finite cells: {np.sum(np.isfinite(mask))}, n_neighbours: {n_neighbours}, origin: {img_origin}, scale: {img_scale}')

        # Drop previous band shapes (ROIStatsWidget ignores Shape removals — no 'key not recognized' warnings).
        for item in list(a_plot.getItems()):
            name = item.getName() if hasattr(item, 'getName') else ''
            if isinstance(name, str) and name.startswith(legend_key):
                a_plot.removeItem(item)
        ## END for item in list(a_plot.getItems())....

        shapes = []
        for poly_idx, (xs, ys) in enumerate(self.iter_scoring_band_polygons(mask=mask, origin=img_origin, scale=img_scale)):
            # Orange fill matching the reference figure's scoring ROI; overlay=True keeps it above the image.
            shape_item = a_plot.addShape(xs, ys, legend=f'{legend_key}_{poly_idx}', shape='polygon', color=(1.0, 0.65, 0.0, 0.45), fill=True, overlay=True, z=1, linestyle='-', linewidth=1.5)
            shapes.append(shape_item)
        ## END for poly_idx, (xs, ys) in enumerate(self.iter_scoring_band_polygons(mask=mask, origin=img_origin, scale=img_scale))....

        return shapes


    def add_real_space_curve(self, a_plot, legend_key: str = 'y(t)=velocity*t+intercept', debug_print=False):
        """ Plot the geometric radon line from active_debug_info.y_line (internal slope, not the negated returned velocity). """
        a_debug_info: RadonTransformDebugValue = self.active_radon_values.active_debug_info
        real_line_t = np.asarray(a_debug_info.t, dtype=float)
        # When enable_return_neighbors_arr=True, y_line is already the 1d best-line geometry (velocity*t + intercept) before sign flip.
        y_line = np.asarray(a_debug_info.y_line, dtype=float)
        if np.ndim(y_line) > 1:
            y_line = np.squeeze(y_line[a_debug_info.best_line_idx, :])
        if debug_print:
            print(f'y_line t range: [{real_line_t[0]}, {real_line_t[-1]}], x range: [{y_line[0]}, {y_line[-1]}]')

        # replace=False: silx replace=True deletes ALL other curves (would wipe rho_phi).
        real_space_curve: Curve = a_plot.addCurve(x=real_line_t, y=y_line, legend=legend_key, color=(1.0, 0.0, 0.0, 0.7), linestyle='-', linewidth=3, symbol=None, replace=False, z=2)
        real_space_curve.setAlpha(alpha=0.7)
        return real_space_curve


    def add_rho_phi_overlay(self, a_plot, legend_key: str = 'rho_phi', debug_print=False):
        """ Draw the index-space rho normal from (ci_mid, ri_mid) to the foot, converted to seconds/centimeters. """
        a_debug_info: RadonTransformDebugValue = self.active_radon_values.active_debug_info
        dt: float = float(self.time_bin_size)
        dx: float = float(self.pos_bin_size)
        t0: float = float(a_debug_info.t[0])
        x0: float = float(a_debug_info.pos[0])
        ci_mid: float = float(a_debug_info.ci_mid)
        ri_mid: float = float(a_debug_info.ri_mid)
        best_rho: float = float(a_debug_info.best_rho)
        best_phi: float = float(a_debug_info.best_phi)

        # Foot of the perpendicular in index space: center + rho * (cos phi, sin phi)
        ci_foot: float = ci_mid + (best_rho * np.cos(best_phi))
        ri_foot: float = ri_mid + (best_rho * np.sin(best_phi))

        # t = ci * dt + t0, x = ri * dx + x0  (same mapping as radon_transform)
        t_center: float = (ci_mid * dt) + t0
        x_center: float = (ri_mid * dx) + x0
        t_foot: float = (ci_foot * dt) + t0
        x_foot: float = (ri_foot * dx) + x0
        if debug_print:
            print(f'rho/phi center=({t_center}, {x_center}), foot=({t_foot}, {x_foot}), rho={best_rho}, phi={best_phi}')

        # Default white for contrast against dark viridis; override via overlay_label_color (e.g. 'black' on Greys)
        rho_curve: Curve = a_plot.addCurve(x=np.array([t_center, t_foot], dtype=float), y=np.array([x_center, x_foot], dtype=float), legend=legend_key, color=self.overlay_label_color, linestyle='--', linewidth=2, symbol=None, replace=False, z=3)
        rho_curve.setAlpha(alpha=0.95)
        a_plot.addMarker(x=t_center, y=x_center, legend=f'{legend_key}_center', text=f'ρ={best_rho:.3g}\nφ={best_phi:.3g}', color=self.overlay_label_color, symbol='o', selectable=False, draggable=False)
        a_plot.addMarker(x=t_foot, y=x_foot, legend=f'{legend_key}_foot', text='', color=self.overlay_label_color, symbol='+', selectable=False, draggable=False)
        return rho_curve


    def add_score_label(self, a_plot, legend_key: str = 'radon_score', debug_print=False):
        """ Place a text marker with the epoch's final radon score near the top-left of the posterior (cm/s frame). """
        a_debug_info: RadonTransformDebugValue = self.active_radon_values.active_debug_info
        score: float = float(self.active_radon_values.score)
        t_label: float = float(a_debug_info.t[0])
        x_label: float = float(a_debug_info.pos[-1])
        if debug_print:
            print(f'radon score label: score={score}, at=({t_label}, {x_label})')

        return a_plot.addMarker(x=t_label, y=x_label, legend=legend_key, text=f'radon={score:.3f}', color=self.overlay_label_color, symbol='', selectable=False, draggable=False)


    def _clear_radon_debugging_labels(self, a_plot, legend_keys: Optional[List[str]] = None):
        """ Remove rho/phi debug overlays (used when radon_debugging_labels is False). Score label is separate. """
        if legend_keys is None:
            legend_keys = ['rho_phi', 'rho_phi_center', 'rho_phi_foot']
        legend_key_set = set(legend_keys)
        for item in list(a_plot.getItems()):
            name = item.getName() if hasattr(item, 'getName') else ''
            if isinstance(name, str) and (name in legend_key_set):
                a_plot.removeItem(item)
        ## END for item in list(a_plot.getItems())....


    def _comment_key_for_current_epoch(self) -> Tuple[str, float]:
        """ Identity for the plotted epoch: (active_decoder_name, active_epoch_start_t). """
        return (self.active_decoder_name, self.active_epoch_start_t)


    def _save_comment_from_field(self, key: Optional[Tuple[str, float]] = None):
        """ Write the Comment QLineEdit into epoch_comments under `key` (default: current epoch). """
        if (self._comment_line_edit is None) or self._comment_loading:
            return
        if key is None:
            key = self._comment_key_for_current_epoch()
        self.epoch_comments[key] = str(self._comment_line_edit.text())


    def _load_comment_into_field(self):
        """ Load epoch_comments for the current key into the Comment field; save any pending edit under the previous key first. """
        if self._comment_line_edit is None:
            return
        new_key: Tuple[str, float] = self._comment_key_for_current_epoch()
        if (self._last_comment_key is not None) and (self._last_comment_key != new_key):
            self._save_comment_from_field(key=self._last_comment_key)
        self._comment_loading = True
        try:
            self._comment_line_edit.setText(self.epoch_comments.get(new_key, ''))
        finally:
            self._comment_loading = False
        self._last_comment_key = new_key


    def _ensure_comment_dock(self):
        """ Install bottom Comment: QLineEdit dock once on the window (safe across re-build_GUI). """
        if self.window is None:
            return
        if self._comment_dock is not None:
            return
        comment_widget = qt.QWidget(self.window)
        comment_layout = qt.QHBoxLayout(comment_widget)
        comment_layout.setContentsMargins(6, 4, 6, 4)
        comment_label = qt.QLabel('Comment:', comment_widget)
        self._comment_line_edit = qt.QLineEdit(comment_widget)
        self._comment_line_edit.setPlaceholderText('Add a comment/description for this epoch…')
        self._comment_line_edit.editingFinished.connect(self._save_comment_from_field)
        comment_layout.addWidget(comment_label)
        comment_layout.addWidget(self._comment_line_edit, stretch=1)
        self._comment_dock = qt.QDockWidget('Comment', self.window)
        self._comment_dock.setWidget(comment_widget)
        self.window.addDockWidget(qt.Qt.BottomDockWidgetArea, self._comment_dock)
        self._load_comment_into_field()


    def update_epoch_idx(self, active_epoch_idx: int, debug_print=False):
        """ Called when the active_epoch_idx is updated to recompute the required RadonTransform values and update the GUI/ROIs
        Usage:
            a_posterior, (start_point, end_point, band_width), (active_num_neighbors, active_neighbors_arr) = on_update_epoch_idx(active_epoch_idx=5)
        
        captures: pos_bin_size, time_bin_size """
        ## ON UPDATE: active_epoch_idx — save comment under the previous key before switching
        self._save_comment_from_field()
        self.active_epoch_idx = active_epoch_idx ## update the index
        
        ## INPUTS: pos_bin_size
        a_posterior = self.result.p_x_given_n_list[active_epoch_idx].copy()

        # num_neighbours # (84,)
        # np.shape(neighbors_arr) # (84,)

        # neighbors_arr[0].shape # (57, 66)
        # neighbors_arr[1].shape # (57, 66)

        # for a_neighbors_arr in neighbors_arr:
        # 	print(f'np.shape(a_neighbors_arr): {np.shape(a_neighbors_arr)}') # np.shape(a_neighbors_arr): (57, N[epoch_idx]) - where N[epoch_idx] = result.nbins[epoch_idx]

        active_num_neighbors: int = self.num_neighbours[self.active_epoch_idx]
        active_neighbors_arr = self.neighbors_arr[self.active_epoch_idx].copy()

        # n_arr_v = (2 * num_neighbours[0] + 1)
        # print(f"n_arr_v: {n_arr_v}")

        # flat_neighbors_arr = np.array(neighbors_arr)
        # np.shape(flat_neighbors_arr)


        ## OUTPUTS: active_num_neighbors, active_neighbors_arr, a_posterior
        # decoder_laps_radon_transform_df: pd.DataFrame = decoder_laps_radon_transform_df_dict[active_decoder_name].copy()
        # decoder_laps_radon_transform_df

        # active_filter_epochs[active_filter_epochs[''
        active_epoch_info_tuple = tuple(self.active_filter_epochs.itertuples(name='EpochTuple'))[self.active_epoch_idx]
        # active_epoch_info_tuple
        # (active_epoch_info_tuple.velocity, active_epoch_info_tuple.intercept)

        ## build the ROI properties:
        # start_point = (0.0, active_epoch_info_tuple.intercept)
        # end_point = (active_epoch_info_tuple.duration, (active_epoch_info_tuple.duration * active_epoch_info_tuple.velocity))
        # band_width = pos_bin_size * float(active_num_neighbors)

        only_compute_current_active_epoch_time_bins: bool = True
        
        NP: int = [np.shape(p)[0] for p in self.result.p_x_given_n_list][0] # just get the first one, they're all the same
        NT: NDArray = np.array([np.shape(p)[1] for p in self.result.p_x_given_n_list]) # These are all different, depends on the length of the epoch.
        if only_compute_current_active_epoch_time_bins:
            NT = NT[self.active_epoch_idx] # an int


        if debug_print:
            print(f'NP: {NP}, NT: {NT}')

        # 1-indexed: this was what the author provided, but it seems to be 1-indexed.
        # index_space_t_mid = ((NT + 1) / 2)
        # index_space_x_mid = ((NP+1)/2)

        # 0-indexed
        index_space_t_mid = ((NT) / 2)
        index_space_x_mid = ((NP) / 2)

        if debug_print:
            print(f'index_space_t_mid: {index_space_t_mid}, index_space_x_mid: {index_space_x_mid}')


        active_time_window_centers = deepcopy(self.result.time_window_centers[self.active_epoch_idx]) # will need this either way later

        if only_compute_current_active_epoch_time_bins:
            ## only active index's bin:    
            real_space_t_mid = ((active_time_window_centers[0]+active_time_window_centers[-1]) / 2)
        else:
            ## all bins:
            real_space_t_mid = np.array([((active_time_window_centers[0]+active_time_window_centers[-1]) / 2) for active_time_window_centers in self.result.time_window_centers])

        real_space_x_mid = ((self.xbin[-1]+self.xbin[0])/2.0)


        ## Conversion functions:
        convert_real_space_x_to_index_space_ri = lambda x: (((x - real_space_x_mid)/self.pos_bin_size) + index_space_x_mid)

        # ## WORKING NOW:
        # convert_real_space_x_to_index_space_ri(dbgr.xbin)
        # convert_real_space_x_to_index_space_ri(dbgr.xbin_centers)

        convert_real_time_t_to_index_time_ci = lambda t: (((t - real_space_t_mid)/self.time_bin_size) + index_space_t_mid)

        ## index space
        

        ## Get the values computed by the original Radon Transform computation that was saved out:
        # start_point = [0.0, active_epoch_info_tuple.intercept]
        # end_point = [active_epoch_info_tuple.duration, (active_epoch_info_tuple.duration * active_epoch_info_tuple.velocity)]
        # band_width = self.pos_bin_size * float(active_num_neighbors)

        start_point = [active_time_window_centers[0], active_epoch_info_tuple.intercept]
        end_point = [active_time_window_centers[-1], (active_epoch_info_tuple.intercept + (active_epoch_info_tuple.duration * active_epoch_info_tuple.velocity))]
        band_width = self.pos_bin_size * float(active_num_neighbors)

        ## REMAINING QUESTION: is `.intercept` calculated at the first time bin_center? Or the first time_bin_edge?

        if debug_print:
            print(f'position-frame line info:')
            print(f'\tstart_point: {start_point},\t end_point: {end_point},\t band_width: {band_width}')


        ## Start converions:
        start_point[0] = convert_real_time_t_to_index_time_ci(start_point[0]) # not right because `t` is supposed to be absolute times anchored in the middle of the time bins. I need active time windows
        end_point[0] = convert_real_time_t_to_index_time_ci(end_point[0])

        start_point[1] = convert_real_space_x_to_index_space_ri(start_point[1])
        end_point[1] = convert_real_space_x_to_index_space_ri(end_point[1])

        # band_width = float(active_num_neighbors)

        # ## convert time (x) coordinates:
        # time_bin_size: float = float(self.result.decoding_time_bin_size)
        # start_point[0] = (start_point[0]/time_bin_size)
        # end_point[0] = (end_point[0]/time_bin_size)
        # # end_point[1] = (end_point[1]/time_bin_size) # not sure about this one

        # ## convert from position (cm) units to y-bins:
        # pos_bin_size: float = float(self.pos_bin_size) # passed directly
        # start_point[1] = (start_point[1]/pos_bin_size)
        # end_point[1] = (end_point[1]/pos_bin_size) # not sure about this one
        # band_width = float(active_num_neighbors)

        if debug_print:
            print(f'index-frame line info:')
            print(f'\tstart_point: {start_point},\t end_point: {end_point},\t band_width: {band_width}')

        ## OUTPUTS: a_posterior, (start_point, end_point, band_width), (active_num_neighbors, active_neighbors_arr)
        # Initialize an instance of TransformDebugger using the variables as keyword arguments
        # transform_debug_instance = RadonDebugValue(a_posterior=a_posterior, start_point=start_point, end_point=end_point, band_width=band_width, active_num_neighbors=active_num_neighbors, active_neighbors_arr=active_neighbors_arr)

        single_epoch_result: SingleEpochDecodedResult = self.result.get_result_for_epoch(active_epoch_idx=self.active_epoch_idx)

        # Entirely new radon computation: ____________________________________________________________________________________ #
        active_time_window_centers = deepcopy(self.result.time_window_centers[self.active_epoch_idx]) # will need this either way later
        score, velocity, intercept, (num_neighbours, neighbors_arr, debug_info) = get_radon_transform(posterior=a_posterior,
                    decoding_time_bin_duration=self.time_bin_size, pos_bin_size=self.pos_bin_size,
                    nlines=8192, n_jobs=1,
                    margin=None, n_neighbours=active_num_neighbors,
                    enable_return_neighbors_arr=True,
                    t0=active_time_window_centers[0],
                    x0=self.xbin_centers[0])
        score = score[0]
        velocity = velocity[0]
        intercept = intercept[0]
        num_neighbours = num_neighbours[0]
        neighbors_arr = neighbors_arr[0]
        a_debug_info: RadonTransformDebugValue = debug_info[0]

        ## Geometric line endpoints from y_line (internal slope*t + intercept); band_width is the vertical scoring window height in cm.
        real_line_t = np.asarray(a_debug_info.t, dtype=float)
        y_line = np.asarray(a_debug_info.y_line, dtype=float)
        if np.ndim(y_line) > 1:
            y_line = np.squeeze(y_line[a_debug_info.best_line_idx, :])
        start_point = [float(real_line_t[0]), float(y_line[0])]
        end_point = [float(real_line_t[-1]), float(y_line[-1])]
        band_width = float((2 * int(num_neighbours) + 1) * self.pos_bin_size)
        
        ## upgrade to RadonDebugValue:
        result = RadonDebugValue(active_decoded_epoch_container=single_epoch_result, active_debug_info=a_debug_info, score=score, velocity=velocity, intercept=intercept,
                            # active_num_neighbors=num_neighbours, active_neighbors_arr=neighbors_arr,
                            active_num_neighbors=active_num_neighbors, active_neighbors_arr=active_neighbors_arr,
                            start_point=start_point, end_point=end_point, band_width=band_width)
        self._load_comment_into_field()
        return result


    def _configure_plot_display(self, a_plot):
        """ Show epoch/decoder in the title; sparse non-scientific x ticks (bin edges come from the xgrid, not every label). """
        a_plot.setGraphTitle(f'epoch_idx={self.active_epoch_idx} | decoder={self.active_decoder_name}')
        an_ax = a_plot.getBackend().ax  # matplotlib.axes._axes.Axes
        an_ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        an_ax.xaxis.set_major_formatter(fmt)


    def _default_export_suffix(self) -> str:
        return f'{self.active_decoder_name}_epoch{self.active_epoch_idx}'


    def _save_publication_pdf(self, save_path: Path) -> Path:
        """ Write the current silx plot to `save_path` under publication matplotlib defaults.

        Flushes the Qt event loop and forces a matplotlib redraw first — required for programmatic
        exports right after refresh_overlays (the GUI Export button works because QFileDialog already pumped events).
        """
        assert self.window is not None, "build_GUI() must be called before exporting."
        save_path = Path(save_path)
        if save_path.suffix.lower() != '.pdf':
            save_path = save_path.with_suffix('.pdf')
        save_path.parent.mkdir(parents=True, exist_ok=True)

        a_plot = self.window.plot
        # Ensure deferred silx/matplotlib paints from the latest refresh_overlays are applied.
        qt.QApplication.processEvents()
        if hasattr(a_plot, 'replot'):
            a_plot.replot()
        backend = a_plot.getBackend() if hasattr(a_plot, 'getBackend') else None
        if (backend is not None) and hasattr(backend, 'fig'):
            backend.fig.canvas.draw()
            if hasattr(backend.fig.canvas, 'flush_events'):
                backend.fig.canvas.flush_events()
        qt.QApplication.processEvents()
        # Re-apply locator/formatter after replot so export PDFs keep sparse non-scientific x ticks.
        self._configure_plot_display(a_plot)

        with mpl.rc_context(PhoPublicationFigureHelper.rc_context_kwargs(prepare_for_publication=True)):
            ok = a_plot.saveGraph(str(save_path), fileFormat='pdf')
        assert ok, f"silx saveGraph failed for path: {save_path}"
        print(f'export_for_publication: saved "{save_path}"')
        return save_path


    @function_attributes(short_name=None, tags=['export', 'pdf', 'publication', 'figure'], input_requires=[], output_provides=[], uses=['PhoPublicationFigureHelper.rc_context_kwargs', 'Plot2D.saveGraph'], used_by=[], creation_date='2026-10-07 00:00', related_items=[])
    def export_for_publication(self, figures_parent_folder: Path, export_suffix: Optional[str] = None) -> Dict[str, Path]:
        """ Export the current debugger figure to a publication-style PDF.

        Usage:
            _out_paths = dbgr.export_for_publication(figures_parent_folder=Path('output/figures'))
        """
        if export_suffix is None:
            export_suffix = self._default_export_suffix()
        figures_parent_folder = Path(figures_parent_folder)
        # Avoid RadonTransform_RadonTransform_... when callers already include the prefix.
        if str(export_suffix).startswith('RadonTransform_'):
            image_save_path = figures_parent_folder.joinpath(f'{export_suffix}.pdf')
        else:
            image_save_path = figures_parent_folder.joinpath(f'RadonTransform_{export_suffix}.pdf')
        return {'pdf': self._save_publication_pdf(image_save_path)}


    def _on_export_pdf_clicked(self):
        """ QFileDialog → exact-path publication PDF write. """
        default_name = f'RadonTransform_{self._default_export_suffix()}.pdf'
        chosen_path, _ = qt.QFileDialog.getSaveFileName(self.window, 'Export PDF', default_name, 'PDF (*.pdf)')
        if not chosen_path:
            return
        self._save_publication_pdf(Path(chosen_path))


    def _ensure_export_toolbar(self):
        """ Install Export PDF toolbar once on the window (safe across re-build_GUI). """
        if self.window is None:
            return
        if getattr(self, '_export_toolbar', None) is not None:
            return
        toolbar = self.window.addToolBar('Export')
        export_action = qt.QAction('Export PDF', self.window)
        export_action.setToolTip('Export current figure to a publication-style PDF')
        export_action.triggered.connect(self._on_export_pdf_clicked)
        toolbar.addAction(export_action)
        self._export_toolbar = toolbar


    def refresh_overlays(self, resetzoom: bool = False):
        """ Redraw posterior, scoring band, geometric line, and rho/phi from active_radon_values when the window exists.

        All addImage/addCurve calls use replace=False. In silx, replace=True means delete *all* other
        images/curves (not update-by-legend), which was wiping P_x_given_n when the band was added.
        """
        if self.window is None:
            return
        a_plot = self.window.plot
        self.add_real_space_posterior(a_plot=a_plot, legend_key='P_x_given_n', resetzoom=resetzoom)
        self.add_time_bin_xgrid(a_plot=a_plot)
        self.add_scoring_band_overlay(a_plot=a_plot)
        self.add_real_space_curve(a_plot=a_plot)
        if self.radon_debugging_labels:
            self.add_rho_phi_overlay(a_plot=a_plot)
        else:
            self._clear_radon_debugging_labels(a_plot=a_plot)
        ## END if self.radon_debugging_labels...
        self.add_score_label(a_plot=a_plot)
        # Keep the posterior as the active image so the colorbar matches p_x_given_n, not the band.
        a_plot.setActiveImage('P_x_given_n')
        self._configure_plot_display(a_plot)
        self._load_comment_into_field()


    def build_GUI(self):
        ## Get the current data for this index:
        # an_epoch_debug_value = self.on_update_epoch_idx(active_epoch_idx=5)

        # No default BandROI — the scoring window is drawn as a vertical mask matching compute_score.
        self.band_roi = None
        if self.window is None:
            self.window = _RoiStatsDisplayExWindow()
        else:
            self.window.plot.clear()

        # Create the thread that calls submitToQtMainThread
        # updateThread = UpdateThread(window.plot)
        # updateThread.start()  # Start updating the plot

        # define some image and curve
        # self.window.plot.addImage(self.active_radon_values.p_x_given_n, legend='P_x_given_n', replace=True, xlabel='time bins', ylabel='pos_bins', selectable=False, draggable=False)

        self._ensure_comment_dock()
        self.refresh_overlays(resetzoom=True)

        # window.plot.addImage(numpy.random.random(10000).reshape(100, 100), legend='img2', origin=(0, 100))
        self.window.setStats(self.stats_measures)

        update_mode: str = 'auto'
        self.window.setUpdateMode(update_mode)

        self._ensure_export_toolbar()

        self.window.show()
        # app.exec()
        # updateThread.stop()  # Stop updating the plot


    def _perform_update_band_ROI(self, start_point: Tuple[float, float] = None, end_point: Tuple[float, float] = None, band_width: float = None):
        """ Refresh geometric overlays for the active epoch (BandROI no longer used for the scoring window). """
        self.refresh_overlays(resetzoom=False)


    def update_ROI(self):
        print(f'update_ROI()\n\tactive_epoch_idx: {self.active_epoch_idx})')
        self.refresh_overlays(resetzoom=False)
        print(f'\tdone.')


    def update_GUI(self):
        print(f'update_GUI()\n\tactive_epoch_idx: {self.active_epoch_idx})')
        # posterior_identifier_str: str = f"Posterior Epoch[{self.active_epoch_idx}]"
        # self.window.plot.addImage(self.active_radon_values.a_posterior, replace=True, resetzoom=True, copy=True, legend='P_x_given_n', ylabel=posterior_identifier_str)

        if self.window is not None:
            self.window.plot.clear()
        self.build_GUI()
        
        # image = self.window.plot.getImage('P_x_given_n')  # Retrieve the image
        # image.setData(self.active_radon_values.a_posterior)  # Update the displayed data

        # self.update_ROI()
        print(f'\tdone.')


