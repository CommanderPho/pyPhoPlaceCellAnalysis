import numpy as np
import pyvista as pv
from pyvista.plotting.plotting import Plotter
# from pyvista.core.composite import MultiBlock
from pyvistaqt import BackgroundPlotter
from pyvistaqt.plotting import MultiPlotter

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

from pyphoplacecellanalysis.Pho3D.PyVista.spikeAndPositions import point_location_circle, point_location_trail_circle # Required for InteractivePyvistaPlotter_PointAndPathPlottingMixin

class InteractivePyvistaPlotterBuildIfNeededMixin:
    """ allows the implementor to build a new plotter if it needs, or re-use the existing one if it already exists. """
    @staticmethod
    def build_new_plotter_if_needed(pActiveTuningCurvesPlotter=None, plotter_type='BackgroundPlotter', **kwargs):
        """[summary]

        Args:
            pActiveTuningCurvesPlotter ([type], optional): [description]. Defaults to None.
            plotter_type (str, optional): [description]. ['BackgroundPlotter', 'MultiPlotter'] Defaults to 'BackgroundPlotter'.

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        if (pActiveTuningCurvesPlotter is not None):
            if isinstance(pActiveTuningCurvesPlotter, BackgroundPlotter):
                if pActiveTuningCurvesPlotter.app_window.isHidden():
                    print('No open BackgroundPlotter')
                    pActiveTuningCurvesPlotter.close() # Close it to start over fresh
                    pActiveTuningCurvesPlotter = None
                    needs_create_new_backgroundPlotter = True
                else:
                    print('BackgroundPlotter already open, reusing it.. NOT Forcing creation of a new one!')
                    pActiveTuningCurvesPlotter.close() # Close it to start over fresh
                    pActiveTuningCurvesPlotter = None
                    needs_create_new_backgroundPlotter = True
                    
            else:
                print(f'No open BackgroundPlotter, p is a {type(pActiveTuningCurvesPlotter)} object')
                pActiveTuningCurvesPlotter.close()
                pActiveTuningCurvesPlotter = None
                needs_create_new_backgroundPlotter = True
        else:
            print('No extant BackgroundPlotter')
            needs_create_new_backgroundPlotter = True

        if needs_create_new_backgroundPlotter:
            print(f'Creating a new {plotter_type}')
            # pActiveTuningCurvesPlotter = BackgroundPlotter(window_size=(1920, 1080), shape=(1,1), off_screen=False) # Use just like you would a pv.Plotter() instance
            if plotter_type == 'BackgroundPlotter':
                pActiveTuningCurvesPlotter = BackgroundPlotter(**({'window_size':(1920, 1080), 'shape':(1,1), 'off_screen':False} | kwargs)) # Use just like you would a pv.Plotter() instance 
            elif plotter_type == 'MultiPlotter':
                pActiveTuningCurvesPlotter = MultiPlotter(**({'window_size':(1920, 1080), 'shape':(1,1), 'off_screen':False} | kwargs))
            else:
                print(f'plotter_type is of unknown type {plotter_type}')
                raise ValueError
                
        return pActiveTuningCurvesPlotter


    @property
    def is_multiplotter(self) -> bool:
        return (hasattr(self.p, '__getitem__') and hasattr(self.p, '_nrows') and hasattr(self.p, '_ncols'))
    

class InteractivePyvistaPlotter_ObjectManipulationMixin:
    """ Has a self.plots dict that uses string keys to access named plots
        This mixin adds functions that enables interactive manipulation of plotted objects post-hoc
    """
    ## Plot Manipulation Helpers:
    @property
    def get_plot_objects_list(self):
        """ a list of all valid plot objects """
        return list(self.plots.keys())

    @staticmethod
    def __toggle_visibility(mesh):
        new_vis = not bool(mesh.GetVisibility())
        mesh.SetVisibility(new_vis)
        # return new_vis

    def safe_get_plot(self, plot_key):
        a_plot = self.plots.get(plot_key, None)
        if a_plot is not None:
            return a_plot
        else:
            raise IndexError

    def set_plot_visibility(self, plot_key, is_visibie):
        self.safe_get_plot(plot_key).SetVisibility(is_visibie)

    def toggle_plot_visibility(self, plot_key):
        return InteractivePyvistaPlotter_ObjectManipulationMixin.__toggle_visibility(self.safe_get_plot(plot_key))


class InteractivePyvistaPlotter_PointAndPathPlottingMixin:
    """ Implementor can render location points and paths/trails in the plotter
    
    Requires (Implementor Must Provide):
        p
        plots
        plots_data
    
    Provides:
        Provided Properties:
            None
        Provided Methods:
            perform_plot_location_point(...)
            perform_plot_location_trail(...)
    
    """
    def perform_plot_location_point(self, plot_name, curr_animal_point, render=True, circle_circumference_scale:float=0.25, **kwargs):
        """ will render a flat indicator of a single point like is used for the animal's current location. 
        Updates the existing plot if the same plot_name is reused. 

        curr_animal_point = [self.x[active_included_all_window_position_indicies[-1]], self.y[active_included_all_window_position_indicies[-1]], self.z_fixed[-1]]
        self.perform_plot_location_point('animal_current_location_point', curr_animal_point, render=False)

        """
        ## COMPAT: merge operator '|'requires Python 3.9
        pdata_current_point = pv.PolyData(curr_animal_point) # a mesh

        if circle_circumference_scale is None:
            scale = False
        else:
            scale = float(circle_circumference_scale)        

        pc_current_point = pdata_current_point.glyph(scale=scale, geom=point_location_circle)
        
        self.plots_data[plot_name] = {'pdata_current_point':pdata_current_point, 'pc_current_point':pc_current_point}
        self.plots[plot_name] = self.p.add_mesh(pc_current_point, name=plot_name, render=render, **({'color':'green', 'ambient':0.6, 'opacity':0.5,
                        'show_edges':True, 'edge_color':[0.05, 0.8, 0.08], 'line_width':3.0, 'nan_opacity':0.0, 'render_lines_as_tubes':True,
                        'show_scalar_bar':False, 'use_transparency':True, 'reset_camera':False} | kwargs))
        return self.plots[plot_name], self.plots_data[plot_name]


    def perform_plot_location_trail(self, plot_name, arr_x, arr_y, arr_z, render=True, trail_fade_values=None, trail_point_size_values=None, **kwargs):
        """ will render a series of points as a trajectory/path given arr_x, arr_y, and arr_z vectors of the same length.
        indicator of a single point like is used for the animal's current location. 
        Updates the existing plot if the same plot_name is reused. """
        point_cloud_fixedSegements_positionTrail = np.column_stack((arr_x, arr_y, arr_z))
        pdata_positionTrail = pv.PolyData(point_cloud_fixedSegements_positionTrail.copy()) # a mesh
        active_num_samples = len(arr_x) # get the number of samples to be plotted so that the trail_fade_values and trail_point_size_values may be cut down to only the most recent (the last active_num_samples values) if there are fewer points than maximum
        if trail_fade_values is not None:
            pdata_positionTrail.point_data['pho_fade_values'] = trail_fade_values[-active_num_samples:]
            scalars_arg = 'pho_fade_values'
        else:
            scalars_arg = None
        if trail_point_size_values is not None:
            pdata_positionTrail.point_data['pho_size_values'] = trail_point_size_values[-active_num_samples:]
            point_size_scale_arg = 'pho_size_values'
        else:
            point_size_scale_arg = None
        
        # create many spheres from the point cloud
        pc_positionTrail = pdata_positionTrail.glyph(scale=point_size_scale_arg, geom=point_location_trail_circle)
        self.plots_data[plot_name] = {'point_cloud_fixedSegements_positionTrail':point_cloud_fixedSegements_positionTrail, 'pdata_positionTrail':pdata_positionTrail, 'pc_positionTrail':pc_positionTrail}
        self.plots[plot_name] = self.p.add_mesh(pc_positionTrail, name=plot_name, render=render, **({'ambient':0.6, 'opacity':'linear_r', 'scalars':scalars_arg, 'nan_opacity':0.0,
                                                'show_edges':False, 'render_lines_as_tubes':True, 'show_scalar_bar':False, 'use_transparency':True, 'reset_camera':False} | kwargs))
        return self.plots[plot_name], self.plots_data[plot_name]
            
            
@metadata_attributes(short_name=None, tags=['pyvista','box','grid_bin_bounds'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-09-18 14:37', related_items=[])
class InteractivePyvistaPlotter_BoxPlottingMixin:
    """ Implementor can render rects and boxes in the plotter
    
    Requires (Implementor Must Provide):
        p
        plots
        plots_data
    
    Provides:
        Provided Properties:
            None
        Provided Methods:
            perform_plot_grid_bin_bounds_box(...)
    
    """
    def perform_plot_grid_bin_bounds_box(self, grid_bin_bounds, render=True, **kwargs):
        """ will render a flat box showing the grid_bin_bounds considered for computations. 
            TODO: does NOT do this - Updates the existing plot if the same plot_name is reused. 

        perform_plot_grid_bin_bounds_box(grid_bin_bounds=long_pf1D.config.grid_bin_bounds, render=True, **kwargs)

        
        """
        plot_name, self.plots[plot_name], grid_bin_bounds_mesh = self.add_grid_bin_bounds_box(self.p, grid_bin_bounds)
        self.plots_data[plot_name] = {'grid_bin_bounds_mesh':grid_bin_bounds_mesh, 'grid_bin_bounds':grid_bin_bounds}
        return plot_name, self.plots[plot_name], self.plots_data[plot_name]


    @classmethod
    def add_grid_bin_bounds_box(cls, p, grid_bin_bounds, label='grid_bin_bounds'):
        """ Render the grid_bin_bounds as a flat box under the maze: 
        Usage:
             plot_name, grid_bin_bounds_mesh_actor, grid_bin_bounds_mesh = add_grid_bin_bounds_box(pActiveSpikesBehaviorPlotter, long_pf1D.config.grid_bin_bounds)
            
        Remove via:
            pActiveSpikesBehaviorPlotter.remove_actor(grid_bin_bounds_mesh_actor)
        """
        plot_name:str = f"{label}_bg"
        add_mesh_kwargs = dict(show_edges=True, line_width=5, color='#fffb0086', name=plot_name, label=label)  
        ((xMin, xMax), (yMin, yMax)) = grid_bin_bounds
        grid_bin_bounds_mesh = pv.Box(bounds=(xMin, xMax, yMin, yMax, -2, -0.1), level=0)
        grid_bin_bounds_mesh_actor = p.add_mesh(grid_bin_bounds_mesh, **add_mesh_kwargs)
        return plot_name, grid_bin_bounds_mesh_actor, grid_bin_bounds_mesh


