"""Standalone vispy rendering for the predictive decoding central view (posterior heatmap, time bins, centroids, contours).

This module does not import from PredictiveDecodingComputations to avoid circular imports.
Callers must pass all views and mutable lists via _update_dict. After raster plotting, _update_dict may
gain vispy_multi_raster_plot (VispyMultiRasterPlotTuple) alongside time_bin_raster (host view).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from nptyping import NDArray

from vispy import scene
import vispy.scene.visuals as vz

from pyphocorehelpers.assertion_helpers import Assert
from pyphoplacecellanalysis.Pho2D.vispy.predictive_time_colormap import predictive_time_bin_rgba
from pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers import ContourItem, contours_from_masks, create_contour_line_visuals
from pyphoplacecellanalysis.Pho2D.vispy.vispy_raster import VispyRasterVisual, plot_multiple_raster_plot_vispy, VispyMultiRasterPlotTuple


def render_central_view(p_x_given_n: np.ndarray, posterior_2d: np.ndarray, time_bin_colors: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float, new_epoch_idx: int, epoch_start_t: Optional[float], epoch_end_t: Optional[float], *, epoch_flat_mask_future_past_result: Optional[List[Any]] = None,
                            curr_position_df: Optional[pd.DataFrame] = None, current_traj_seconds_pre_post_extension: float = 0.75, num_epochs: int = 1, max_time_bins_to_show: int = 12, fallback_mask_2d_for_shape: Optional[np.ndarray] = None,
                            use_new_centroid_arrows: bool = True, use_single_arrows_object: bool = False, _update_dict: Optional[Dict[str, Any]] = None, needs_clear_owned_views: bool = True,
                            actIve_filter_epochs_spikes_df: Optional[pd.DataFrame]=None, active_epochs_df: Optional[pd.DataFrame] = None, active_aclus: Optional[NDArray]=None, time_bin_edges: Optional[NDArray]=None,
                        ) -> Dict[str, Any]:
    """Update the center view with posteriors, time bins, centroid dots/arrows, current position line, and contours.

    _update_dict must contain the vispy views and mutable lists (e.g. posterior_2d_view, time_bin_grid,
    time_bin_raster for spike rasters, past_view, future_view, centroid_dots, centroid_arrows,
    current_position_line, trajectory_arrows, epoch_info_text, time_bin_views, time_bin_labels, time_bin_images,
    past_mask_contours, posterior_mask_contours, future_mask_contours). When a raster is drawn,
    time_bin_raster and vispy_multi_raster_plot are written back. It is updated in place and returned.

    epoch_flat_mask_future_past_result: optional list of duck-typed epoch results. Each element must have
    centroids_df (DataFrame with x, y, segment_idx; optionally segment_Vp_deg, dt) and
    epoch_t_bins_high_prob_pos_mask for contour rendering.

    Returns the updated _update_dict.
    """
    if _update_dict is None:
        _update_dict = {}

    if posterior_2d is None or posterior_2d.size == 0:
        if fallback_mask_2d_for_shape is not None and fallback_mask_2d_for_shape.size > 0:
            img_height, img_width = fallback_mask_2d_for_shape.T.shape
        else:
            img_height, img_width = 1, 1
    else:
        img_height, img_width = posterior_2d.T.shape
    x_scale = (x_max - x_min) / img_width
    y_scale = (y_max - y_min) / img_height

    posterior_img = None
    posterior_2d_view = _update_dict.get('posterior_2d_view', None)

    if posterior_2d is not None and posterior_2d.size > 0 and posterior_2d_view is not None:
        img_data: NDArray = np.ascontiguousarray(posterior_2d.T, dtype=np.float32)
        posterior_img = vz.Image(img_data, cmap='viridis', parent=posterior_2d_view.scene, name=f'posterior_img')
        posterior_img.transform = scene.STTransform(scale=(x_scale, y_scale), translate=(x_min, y_min))
        posterior_img.order = 0
        posterior_img.opacity = 0.8
        posterior_img.set_gl_state(blend=True, depth_test=True, blend_func=('src_alpha', 'one_minus_src_alpha'))

    _update_dict.update(posterior_img=posterior_img)

    centroid_dots = _update_dict.get('centroid_dots', [])
    centroid_arrows = _update_dict.get('centroid_arrows', [])

    if epoch_flat_mask_future_past_result is not None and new_epoch_idx < len(epoch_flat_mask_future_past_result) and posterior_2d_view is not None:
        epoch_result = epoch_flat_mask_future_past_result[new_epoch_idx]
        if epoch_result is not None and hasattr(epoch_result, 'centroids_df') and epoch_result.centroids_df is not None and 'x' in epoch_result.centroids_df.columns and 'y' in epoch_result.centroids_df.columns and 'segment_idx' in epoch_result.centroids_df.columns:
            centroids_df = epoch_result.centroids_df
            centroid_x_values = np.asarray(centroids_df['x'].to_numpy(), dtype=np.float64)
            centroid_y_values = np.asarray(centroids_df['y'].to_numpy(), dtype=np.float64)
            valid_mask = np.isfinite(centroid_x_values) & np.isfinite(centroid_y_values)
            if np.any(valid_mask):
                x_pixel = centroid_x_values[valid_mask]
                y_pixel = centroid_y_values[valid_mask]
                x_centroids = x_min + x_pixel * x_scale
                y_centroids = y_min + y_pixel * y_scale
                original_indices = np.where(valid_mask)[0]
                n_centroids: int = len(x_centroids)
                n_time_bin_slots: int = max(len(time_bin_colors), int(np.max(original_indices)) + 1) if len(original_indices) > 0 else len(time_bin_colors)
                color_by_time_bin = np.zeros((n_time_bin_slots, 4), dtype=np.float32)
                color_by_time_bin[0:len(time_bin_colors)] = time_bin_colors
                if n_time_bin_slots > len(time_bin_colors):
                    color_by_time_bin[len(time_bin_colors):] = (1.0, 1.0, 1.0, 0.8)
                centroid_colors = color_by_time_bin[original_indices]
                centroid_pos = np.column_stack([x_centroids, y_centroids])
                centroid_markers = vz.Markers(pos=centroid_pos, face_color=centroid_colors, size=8, edge_width=0, parent=posterior_2d_view.scene, name=f'centroid_markers')
                centroid_markers.order = 8
                centroid_dots.append(centroid_markers)

                data_scale: float = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
                arrow_head_size: float = data_scale * 0.05
                arrow_length: float = arrow_head_size * 0.3

                if use_new_centroid_arrows:
                    pos_col_names = ['x_start', 'y_start']
                    arrow_centroids_df: pd.DataFrame = deepcopy(centroids_df[valid_mask]).rename(columns={'x': 'x_pixel', 'y': 'y_pixel'}, inplace=False)
                    arrow_centroids_df['x_start'] = x_centroids
                    arrow_centroids_df['y_start'] = y_centroids
                    arrow_centroids_df['original_indices'] = original_indices
                    arrow_centroids_df[['x_start', 'y_start']] = arrow_centroids_df[pos_col_names]
                    arrow_centroids_df[['x_end', 'y_end']] = arrow_centroids_df[pos_col_names].shift(-1)
                    arrow_centroids_df['original_index_start'] = arrow_centroids_df['original_indices'].astype(int)
                    arrow_centroids_df[['original_index_end']] = arrow_centroids_df[['original_index_start']].shift(-1)
                    arrow_centroids_df = arrow_centroids_df.dropna(axis='index', how='any', subset=['x_end', 'y_end', 'original_index_end'], inplace=False).reset_index(drop=True)
                    arrow_centroids_df['original_index_end'] = arrow_centroids_df['original_index_end'].astype(int)

                    if len(arrow_centroids_df) > 0:
                        arrow_centroids_df[['dx', 'dy']] = (arrow_centroids_df[['x_end', 'y_end']].to_numpy() - arrow_centroids_df[['x_start', 'y_start']].to_numpy())
                        if 'dt' in arrow_centroids_df.columns:
                            distances_spatial = np.sqrt((arrow_centroids_df['dx'].to_numpy())**2 + (arrow_centroids_df['dt'].to_numpy())**2)
                        else:
                            distances_spatial = np.sqrt((arrow_centroids_df['dx'].to_numpy())**2 + (arrow_centroids_df['dy'].to_numpy())**2)
                        arrow_centroids_df['dxdy_len'] = distances_spatial
                        arrow_centroids_df = arrow_centroids_df[np.isfinite(arrow_centroids_df['dxdy_len']) & (arrow_centroids_df['dxdy_len'] > 0.0)].copy()
                        if len(arrow_centroids_df) > 0:
                            arrow_centroids_df[['unit_dx', 'unit_dy']] = arrow_centroids_df[['dx', 'dy']].to_numpy() / arrow_centroids_df['dxdy_len'].to_numpy()[:, None]
                            arrow_centroids_df[['x_mid', 'y_mid']] = (arrow_centroids_df[['x_start', 'y_start']].to_numpy() + arrow_centroids_df[['dx', 'dy']].to_numpy())

                            pos: NDArray = np.vstack([arrow_centroids_df[['x_start', 'y_start']].to_numpy(), arrow_centroids_df[['x_end', 'y_end']].to_numpy()])
                            arrows: NDArray = arrow_centroids_df[['x_start', 'y_start', 'x_mid', 'y_mid']].to_numpy()

                            _safe_color_map_fn = lambda t_idx: tuple(color_by_time_bin[t_idx]) if (0 <= t_idx < n_time_bin_slots) else (1.0, 1.0, 1.0, 0.8)
                            _original_index_start_colors_list = arrow_centroids_df['original_index_start'].map(_safe_color_map_fn).to_list()
                            _original_index_end_colors_list = arrow_centroids_df['original_index_end'].map(_safe_color_map_fn).to_list()
                            vertex_point_color: NDArray = np.vstack([np.stack([v0, v1]) for v0, v1 in zip(_original_index_start_colors_list, _original_index_end_colors_list)])
                            Assert.same_length(vertex_point_color, pos)
                            arrow_color: NDArray = np.vstack(_original_index_start_colors_list)
                            Assert.same_length(arrow_color, arrows)

                            if use_single_arrows_object:
                                arrow: vz.Arrow = vz.Arrow(pos=pos, arrows=arrows, arrow_color=arrow_color, arrow_type='triangle_30', arrow_size=arrow_head_size, color=vertex_point_color, width=3.0,
                                                            method='gl', connect='segments', parent=posterior_2d_view.scene, name=f'arrow')
                                arrow.order = 7
                                centroid_arrows.append(arrow)
                            else:
                                for i, a_row in enumerate(arrow_centroids_df.itertuples(index=True)):
                                    t_idx = original_indices[i]
                                    a_row_dict = a_row._asdict()
                                    x_center = a_row_dict['x_start']
                                    y_center = a_row_dict['y_start']
                                    unit_dx = a_row_dict['unit_dx']
                                    unit_dy = a_row_dict['unit_dy']
                                    x_start, y_start = x_center, y_center
                                    x_end = x_center + (unit_dx * arrow_length)
                                    y_end = y_center + (unit_dy * arrow_length)
                                    an_arrow_color = tuple(color_by_time_bin[t_idx]) if (0 <= t_idx < n_time_bin_slots) else (1.0, 1.0, 1.0, 0.8)
                                    a_pos = np.array([[x_start, y_start], [x_end, y_end]])
                                    an_arrows = np.array([[x_start, y_start, x_end, y_end]])
                                    a_pos = np.asarray(a_pos, dtype=np.float32)
                                    an_arrows = np.asarray(an_arrows, dtype=np.float32)
                                    arrow = vz.Arrow(pos=a_pos, arrows=an_arrows, arrow_type='triangle_30', arrow_size=arrow_head_size, color=an_arrow_color, arrow_color=an_arrow_color, width=3.0, method='agg', parent=posterior_2d_view.scene, name=f'centroids_arrow[{t_idx}]')
                                    arrow.order = 7
                                    centroid_arrows.append(arrow)

                else:
                    if 'segment_Vp_deg' in centroids_df.columns:
                        segment_Vp_deg = np.asarray(centroids_df['segment_Vp_deg'].to_numpy(), dtype=np.float64)[valid_mask]
                        valid_angle_mask = np.isfinite(segment_Vp_deg)
                        if np.any(valid_angle_mask):
                            angles_rad = np.deg2rad(segment_Vp_deg[valid_angle_mask])
                            x_centroids_valid = x_centroids[valid_angle_mask]
                            y_centroids_valid = y_centroids[valid_angle_mask]
                            arrow_centroid_indices = np.where(valid_angle_mask)[0]
                            for i in range(len(x_centroids_valid)):
                                x_center = x_centroids_valid[i]
                                y_center = y_centroids_valid[i]
                                angle = angles_rad[i]
                                x_start, y_start = x_center, y_center
                                x_end = x_center + arrow_length * np.cos(angle)
                                y_end = y_center + arrow_length * np.sin(angle)
                                centroid_idx = arrow_centroid_indices[i]
                                t_idx = original_indices[centroid_idx]
                                arrow_color = tuple(color_by_time_bin[t_idx]) if (0 <= t_idx < n_time_bin_slots) else (1.0, 1.0, 1.0, 0.8)
                                pos = np.array([[x_start, y_start], [x_end, y_end]])
                                pos = np.asarray(pos, dtype=np.float32)
                                arrows = np.array([[x_start, y_start, x_end, y_end]])
                                arrows = np.asarray(arrows, dtype=np.float32)
                                arrow = vz.Arrow(pos=pos, arrows=arrows, arrow_type='triangle_30', arrow_size=arrow_head_size, color=arrow_color, arrow_color=arrow_color, width=3.0, method='agg', parent=posterior_2d_view.scene, name=f'centroids_arrow[{i}]')
                                arrow.order = 7
                                centroid_arrows.append(arrow)

    _update_dict.update(centroid_dots=centroid_dots, centroid_arrows=centroid_arrows)

    current_position_line = _update_dict.get('current_position_line', None)
    trajectory_arrows = _update_dict.get('trajectory_arrows', [])
    epoch_info_text = _update_dict.get('epoch_info_text', None)

    if curr_position_df is not None and epoch_start_t is not None and epoch_end_t is not None and 't' in curr_position_df.columns and 'x' in curr_position_df.columns and 'y' in curr_position_df.columns and posterior_2d_view is not None:
        extended_start_t = epoch_start_t - current_traj_seconds_pre_post_extension
        extended_end_t = epoch_end_t + current_traj_seconds_pre_post_extension
        extended_mask = (curr_position_df['t'] >= extended_start_t) & (curr_position_df['t'] <= extended_end_t)
        extended_positions = curr_position_df[extended_mask]
        if len(extended_positions) > 0:
            x_coords = np.asarray(extended_positions['x'].to_numpy(), dtype=np.float64)
            y_coords = np.asarray(extended_positions['y'].to_numpy(), dtype=np.float64)
            t_coords = np.asarray(extended_positions['t'].to_numpy(), dtype=np.float64)
            valid_mask = np.isfinite(x_coords) & np.isfinite(y_coords) & np.isfinite(t_coords)
            if np.count_nonzero(valid_mask) >= 2:
                x_valid = x_coords[valid_mask]
                y_valid = y_coords[valid_mask]
                t_valid = t_coords[valid_mask]
                within_epoch_mask = (t_valid >= epoch_start_t) & (t_valid <= epoch_end_t)
                n_points = len(x_valid)
                colors = np.ones((n_points, 4), dtype=np.float32)
                colors[:, :3] = 0.7
                colors[:, 3] = np.where(within_epoch_mask, 1.0, 0.2)
                current_pos = np.ascontiguousarray(np.column_stack([x_valid, y_valid]), dtype=np.float32)
                current_colors = np.ascontiguousarray(colors, dtype=np.float32)
                if (current_pos.shape[0] >= 2) and (current_colors.shape[0] == current_pos.shape[0]) and np.all(np.isfinite(current_pos)) and np.all(np.isfinite(current_colors)):
                    if current_position_line is None:
                        current_position_line = vz.Line(pos=current_pos, color=current_colors, width=3, method='gl', parent=posterior_2d_view.scene)
                        current_position_line.order = 5
                    else:
                        current_position_line.set_data(pos=current_pos, color=current_colors)
                elif current_position_line is not None:
                    current_position_line.set_data(pos=np.array([], dtype=np.float32).reshape(0, 2), color=np.array([], dtype=np.float32).reshape(0, 4))
            else:
                if current_position_line is not None:
                    current_position_line.set_data(pos=np.array([], dtype=np.float32).reshape(0, 2), color=np.array([], dtype=np.float32).reshape(0, 4))
        else:
            if current_position_line is not None:
                current_position_line.set_data(pos=np.array([], dtype=np.float32).reshape(0, 2), color=np.array([], dtype=np.float32).reshape(0, 4))
            if needs_clear_owned_views:
                for arrow in trajectory_arrows:
                    if arrow is not None:
                        arrow.parent = None
                trajectory_arrows.clear()
            else:
                print('WARN: cannot clean arrows')
    else:
        if current_position_line is not None:
            current_position_line.set_data(pos=np.array([], dtype=np.float32).reshape(0, 2), color=np.array([], dtype=np.float32).reshape(0, 4))
        if needs_clear_owned_views:
            for arrow in trajectory_arrows:
                if arrow is not None:
                    arrow.parent = None
            trajectory_arrows.clear()
        else:
            print('WARN: cannot clean arrows')

    _update_dict.update(current_position_line=current_position_line, trajectory_arrows=trajectory_arrows)

    if epoch_start_t is not None and epoch_end_t is not None and posterior_2d_view is not None:
        epoch_info_str = f'Epoch {new_epoch_idx + 1}/{num_epochs} | start_t: {epoch_start_t:.2f}s | end_t: {epoch_end_t:.2f}s | duration: {epoch_end_t - epoch_start_t:.2f}s'
        text_y_pos = y_max + (y_max - y_min) * 0.15
        text_x_pos = (x_min + x_max) / 2
        epoch_info_text = vz.Text(epoch_info_str, pos=(text_x_pos, text_y_pos), color='white', font_size=10, bold=False, anchor_x='center', anchor_y='bottom', parent=posterior_2d_view.scene)
        y_range = y_max - y_min
        posterior_2d_view.camera.set_range(x=(x_min, x_max), y=(y_min - y_range * 0.05, y_max + y_range * 0.2))

    _update_dict.update(epoch_info_text=epoch_info_text)

    time_bin_views = _update_dict.get('time_bin_views', [])
    time_bin_labels = _update_dict.get('time_bin_labels', [])
    time_bin_images = _update_dict.get('time_bin_images', [])

    time_bin_grid = _update_dict.get('time_bin_grid', None)

    if p_x_given_n is not None and p_x_given_n.size > 0 and time_bin_grid is not None:
        n_time_bins = p_x_given_n.shape[2]
        n_bins_to_show = min(n_time_bins, max_time_bins_to_show)
        view_time_bin_colors = predictive_time_bin_rgba(n_bins_to_show, alpha=1.0)[:, :3]
        vol_min, vol_max = p_x_given_n.min(), p_x_given_n.max()

        _grid_widgets = getattr(time_bin_grid, '_grid_widgets', {})
        needs_rebuild_time_bin_views = (len(time_bin_views) != n_bins_to_show) or (needs_clear_owned_views and len(_grid_widgets) != n_bins_to_show)
        if needs_rebuild_time_bin_views:
            if needs_clear_owned_views:
                ## time_bin_views first:
                for view in time_bin_views:
                    if view is not None and hasattr(view, 'parent'):
                        view.parent = None
                for child in list(time_bin_grid.children):
                    time_bin_grid.remove_widget(child)


                # remove_widget must run before parent=None; detached views are not in .children, so a loop over
                # time_bin_grid.children never unregister them and leaves phantom columns in the layout solver.
                for _slot in list(_grid_widgets.values()):
                    _w = _slot[-1]
                    time_bin_grid.remove_widget(_w)
                    _w.parent = None
                time_bin_views.clear()
            for t_idx in range(n_bins_to_show):
                t_bin_border_color = view_time_bin_colors[t_idx] if t_idx < len(view_time_bin_colors) else (0.5, 0.5, 0.5)
                view = time_bin_grid.add_view(row=0, col=t_idx, border_color=t_bin_border_color)
                view.camera = scene.PanZoomCamera(aspect=1)
                view.camera.set_range(x=(x_min, x_max), y=(y_min, y_max))
                time_bin_views.append(view)

        for t_idx in range(n_bins_to_show):
            slice_2d = p_x_given_n[:, :, t_idx].T.astype(np.float32)
            if vol_max > vol_min:
                slice_2d = (slice_2d - vol_min) / (vol_max - vol_min)
            slice_2d = np.ascontiguousarray(slice_2d, dtype=np.float32)
            view = time_bin_views[t_idx]
            slice_img = vz.Image(slice_2d, cmap='viridis', parent=view.scene)
            slice_img.order = 0
            img_height_s, img_width_s = slice_2d.shape
            scale_x_img = (x_max - x_min) / img_width_s if img_width_s > 0 else 1
            scale_y_img = (y_max - y_min) / img_height_s if img_height_s > 0 else 1
            slice_img.transform = scene.STTransform(scale=(scale_x_img, scale_y_img), translate=(x_min, y_min))
            time_bin_images.append(slice_img)
            label_y_pos = y_max + (y_max - y_min) * 0.08
            label = vz.Text(f't={t_idx}', pos=((x_min + x_max) / 2, label_y_pos), color='white', font_size=10, anchor_x='center', anchor_y='bottom', parent=view.scene)
            time_bin_labels.append(label)
            view.camera.set_range(x=(x_min, x_max), y=(y_min, y_max + (y_max - y_min) * 0.1))

    _update_dict.update(time_bin_views=time_bin_views, time_bin_labels=time_bin_labels, time_bin_images=time_bin_images)

    contour_render_kwargs = dict(fill=True)
    past_view = _update_dict.get('past_view', None)
    future_view = _update_dict.get('future_view', None)
    posterior_2d_view = _update_dict.get('posterior_2d_view', None)
    posterior_mask_contours = _update_dict.get('posterior_mask_contours', [])
    list_names = ['past_mask_contours', 'posterior_mask_contours', 'future_mask_contours']
    active_posterior_contours_dict_list = [(past_view, _update_dict.get('past_mask_contours', [])), (posterior_2d_view, _update_dict.get('posterior_mask_contours', [])), (future_view, _update_dict.get('future_mask_contours', []))]
    views_and_lists_to_draw = [(v, c) for v, c in active_posterior_contours_dict_list if v is not None]

    if epoch_flat_mask_future_past_result is not None and new_epoch_idx < len(epoch_flat_mask_future_past_result):
        epoch_result_for_contours = epoch_flat_mask_future_past_result[new_epoch_idx]
        if epoch_result_for_contours is not None and hasattr(epoch_result_for_contours, 'epoch_t_bins_high_prob_pos_mask') and epoch_result_for_contours.epoch_t_bins_high_prob_pos_mask is not None:
            per_t_bin_mask = epoch_result_for_contours.epoch_t_bins_high_prob_pos_mask
            n_mask_t_bins = per_t_bin_mask.shape[2]
            masks = [per_t_bin_mask[:, :, t_idx].T for t_idx in range(n_mask_t_bins)]
            contour_time_bin_colors = predictive_time_bin_rgba(n_mask_t_bins, alpha=0.7)
            colors = [tuple(contour_time_bin_colors[i]) for i in range(n_mask_t_bins)]
            contour_data_per_mask = contours_from_masks(masks, x_bounds=(x_min, x_max), y_bounds=(y_min, y_max), colors=colors, level=0.5, return_per_mask=True)
            contour_data_flat: List[ContourItem] = [item for sublist in contour_data_per_mask for item in sublist]
            contour_order_main = 25
            for view, cont_list in views_and_lists_to_draw:
                scene_parent = view.scene if view is not None else None
                if scene_parent is not None:
                    lines, polygon_fills = create_contour_line_visuals(contour_data_flat, parent=scene_parent, line_width=2.0, order=contour_order_main, **contour_render_kwargs, name=f'ContourMain')
                    cont_list.extend(polygon_fills)
                    cont_list.extend(lines)

            contour_order_t_bins = 25
            for t_idx in range(min(len(contour_data_per_mask), len(time_bin_views))):
                scene_parent = time_bin_views[t_idx].scene if time_bin_views[t_idx] is not None else None
                if scene_parent is not None:
                    per_mask_contours: List[ContourItem] = contour_data_per_mask[t_idx]
                    lines, polygon_fills = create_contour_line_visuals(per_mask_contours, parent=scene_parent, line_width=2.0, order=contour_order_t_bins, **contour_render_kwargs, name=f'Contours[{t_idx}]')
                    posterior_mask_contours.extend(polygon_fills)
                    posterior_mask_contours.extend(lines)

            _update_dict.update(posterior_mask_contours=posterior_mask_contours)
            for (a_name, (a_view, a_cont_list)) in zip(list_names, active_posterior_contours_dict_list):
                if (a_view is not None) and (a_cont_list is not None):
                    _update_dict[a_name] = a_cont_list


    time_bin_raster = _update_dict.get('time_bin_raster', None)
    can_plot_raster: bool = (active_epochs_df is not None) and (active_aclus is not None) and (actIve_filter_epochs_spikes_df is not None) and (time_bin_raster is not None)
    if can_plot_raster:
        num_epoch_time_bins_for_raster: Optional[int] = None
        if p_x_given_n is not None and getattr(p_x_given_n, 'size', 0) > 0 and p_x_given_n.ndim >= 3:
            num_epoch_time_bins_for_raster = int(min(int(p_x_given_n.shape[2]), int(max_time_bins_to_show)))
        raster_time_bin_edges: Optional[np.ndarray] = None
        if time_bin_edges is not None:
            te = np.asarray(time_bin_edges, dtype=np.float64).ravel()
            if p_x_given_n is not None and getattr(p_x_given_n, 'size', 0) > 0 and p_x_given_n.ndim >= 3:
                k_vis = int(min(int(p_x_given_n.shape[2]), int(max_time_bins_to_show)))
                if te.size >= k_vis + 1:
                    raster_time_bin_edges = te[: k_vis + 1].astype(np.float32, copy=False)
            elif te.size >= 2:
                raster_time_bin_edges = te.astype(np.float32, copy=False)
        num_bins_for_raster_lines: Optional[int] = None if raster_time_bin_edges is not None else num_epoch_time_bins_for_raster
        raster_edge_vu: Optional[np.ndarray] = None
        if raster_time_bin_edges is not None:
            m = int(np.asarray(raster_time_bin_edges, dtype=np.float32).size)
            if m >= 2:
                raster_edge_vu = np.linspace(0.0, 1.0, m, dtype=np.float32)
        # new_all_aclus_sort_indicies = None
        # pen = {'color': 'white', 'width': 1}
        # override_scatter_plot_kwargs = dict(pxMode=False, symbol='vbar', size=5, pen=None) ## small
        override_scatter_plot_kwargs = dict(pxMode=False, symbol='vbar', size=6, pen=None) ## mid
        # override_scatter_plot_kwargs = dict(pxMode=False, symbol='vbar', size=10, pen=None) ## big
        # override_scatter_plot_kwargs = dict(pxMode=True, symbol='vbar', size=0.001, pen=None)

        _out_vispy_raster: VispyMultiRasterPlotTuple = plot_multiple_raster_plot_vispy(filter_epochs_df=active_epochs_df, spikes_df=actIve_filter_epochs_spikes_df,
                                                            included_neuron_ids=active_aclus,
                                                            # unit_sort_order=new_all_aclus_sort_indicies, unit_colors_list=unit_colors_list_L, 
                                                            scatter_plot_kwargs=override_scatter_plot_kwargs,
                                            epoch_id_key_name='replay_epoch_id', scatter_app_name=f'Decoded example replays (epoch {new_epoch_idx + 1}/{num_epochs})', defer_show=True,
                                            active_context=None, time_bin_raster_view=time_bin_raster, clear_host_scene=needs_clear_owned_views, bgcolor='black',
                                            time_bin_edges=raster_time_bin_edges, num_epoch_time_bins=num_bins_for_raster_lines, time_bin_edge_vu=raster_edge_vu, time_bin_edge_line_alpha=0.65)
        _update_dict.update(time_bin_raster=time_bin_raster, vispy_multi_raster_plot=_out_vispy_raster)


    return _update_dict
