import pyphoplacecellanalysis
import pyphoplacecellanalysis.External.pyqtgraph as pg
import pyphoplacecellanalysis.External.pyqtgraph.opengl as gl # for 3D raster plot

class BaseGrid3DTimeCurvesHelper:
    """ 
        2022-06-07: Add the 3D Curves baseline grid axes (straight lines below each time series to make it easier to see what value they represent:
        TODO: only works for PyQtGraph-type plotters (no vedo support)
    """

    @classmethod
    def init_3D_time_curves_baseline_grid_mesh(cls, active_curve_plotter_3d):
        """ time_curves baseline grid mesh properties:
            time_curves_enable_baseline_grid (default: True): whether to enable drawing a grid at the baseline of the 3D curves that helps visually align each curve with its neuron/spikes.
            time_curves_baseline_grid_color (default:'White'): the color of the baseline grid.
            time_curves_baseline_grid_alpha (default: 0.5): the alpha (opacity) of the baseline grid.
        """
        active_curve_plotter_3d.params.setdefault('time_curves_enable_baseline_grid', True)
        active_curve_plotter_3d.params.setdefault('time_curves_baseline_grid_color', 'White')
        active_curve_plotter_3d.params.setdefault('time_curves_baseline_grid_alpha', 0.5)

    @classmethod
    def add_3D_time_curves_baseline_grid_mesh(cls, active_curve_plotter_3d):
        # TODO: needs to be updated on .on_adjust_temporal_spatial_mapping(...)
        if active_curve_plotter_3d.params.setdefault('time_curves_enable_baseline_grid', True):
            if 'time_curve_helpers' not in active_curve_plotter_3d.plots:
                # no plots.time_curve_helpers variables, create it:
                active_curve_plotter_3d.plots.time_curve_helpers = dict()

            if 'plots_grid_3dCurveBaselines_Grid' not in active_curve_plotter_3d.plots.time_curve_helpers:
                # no plots.time_curve_helpers.plots_grid_3dCurveBaselines_Grid variable, create it:
                active_color = pg.mkColor(active_curve_plotter_3d.params.setdefault('time_curves_baseline_grid_color', 'White'))
                active_color.setAlphaF(active_curve_plotter_3d.params.setdefault('time_curves_baseline_grid_alpha', 0.5))
                active_curve_plotter_3d.plots.time_curve_helpers['plots_grid_3dCurveBaselines_Grid'] = gl.GLGridItem(antialias=True, color=active_color)
                
                # Add the item to the plotter:
                active_curve_plotter_3d.ui.main_gl_widget.addItem(active_curve_plotter_3d.plots.time_curve_helpers['plots_grid_3dCurveBaselines_Grid'])
            else:
                # otherwise we already have it, update it if needed and return it:
                active_curve_plotter_3d.plots.time_curve_helpers['plots_grid_3dCurveBaselines_Grid'].resetTransform() # only need to reset the transform if it had already been transformed by a previous operation:
            
            # Update it either way:
            cls.update_3D_time_curves_baseline_grid_mesh(active_curve_plotter_3d)

            return active_curve_plotter_3d.plots.time_curve_helpers['plots_grid_3dCurveBaselines_Grid'] # return the GLGridItem
        else:
            print(f'add_3D_time_curves_baseline_mesh(...): active_curve_plotter_3d.params.time_curves_enable_baseline_grid is False, so no baseline mesh will be added.')
            return False

    @classmethod
    def update_3D_time_curves_baseline_grid_mesh(cls, active_curve_plotter_3d):
        # Update it:
        active_curve_plotter_3d.plots.time_curve_helpers['plots_grid_3dCurveBaselines_Grid'].setSpacing(active_curve_plotter_3d.params.axes_planes_floor_fixed_y_spacing, 1) # (y-axis, x-axis)
        active_curve_plotter_3d.plots.time_curve_helpers['plots_grid_3dCurveBaselines_Grid'].translate(0, 0, active_curve_plotter_3d.params.time_curves_z_baseline) # Shift up in the z-dir
        active_curve_plotter_3d.plots.time_curve_helpers['plots_grid_3dCurveBaselines_Grid'].setSize(active_curve_plotter_3d.temporal_axis_length, active_curve_plotter_3d.n_full_cell_grid) # (y-axis, x-axis)

    @classmethod
    def remove_3D_time_curves_baseline_grid_mesh(cls, active_curve_plotter_3d):
        if 'time_curve_helpers' not in active_curve_plotter_3d.plots:
            return False # nothing to remove
        if 'plots_grid_3dCurveBaselines_Grid' not in active_curve_plotter_3d.plots.time_curve_helpers:
            return False # nothing to remove
        active_item = active_curve_plotter_3d.plots.time_curve_helpers['plots_grid_3dCurveBaselines_Grid']
        active_curve_plotter_3d.ui.main_gl_widget.removeItem(active_item) # remove the item from the main_gl_widget
        active_curve_plotter_3d.plots.time_curve_helpers['plots_grid_3dCurveBaselines_Grid'] = None
        del active_curve_plotter_3d.plots.time_curve_helpers['plots_grid_3dCurveBaselines_Grid']
        return active_item


class Render3DTimeCurvesBaseGridMixin:
    """ a thin wrapper around BaseGrid3DTimeCurvesHelper's static methods:

    """
    def init_3D_time_curves_baseline_grid_mesh(self):
        BaseGrid3DTimeCurvesHelper.init_3D_time_curves_baseline_grid_mesh(self)

    def add_3D_time_curves_baseline_grid_mesh(self):
        # TODO: needs to be updated on .on_adjust_temporal_spatial_mapping(...)
        return BaseGrid3DTimeCurvesHelper.add_3D_time_curves_baseline_grid_mesh(self)

    def update_3D_time_curves_baseline_grid_mesh(self):
        BaseGrid3DTimeCurvesHelper.update_3D_time_curves_baseline_grid_mesh(self)

    def remove_3D_time_curves_baseline_grid_mesh(self):
        return BaseGrid3DTimeCurvesHelper.remove_3D_time_curves_baseline_grid_mesh(self)

