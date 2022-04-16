import vedo
from vedo import Mesh, Cone, Plotter, printc, Glyph
from vedo import Rectangle, Lines, Plane, Axes, merge, colorMap # for StaticVedo_3DRasterHelper
from vedo import Volume, ProgressBar, show, settings


class VedoPlotterHelpers:
    """docstring for VedoHelpers.
    
    Import with:
    
        from pyphoplacecellanalysis.GUI.Vedo.VedoMeshManipulatable import VedoPlotterHelpers
    
    """
    
    @classmethod
    def vedo_create_if_needed(cls, plotter, item_key_name, render_item, defer_render=False):
        """ Creates a new mesh if it exists in spike_raster_plt_3d_vedo
        plotter: Spike3DRaster_Vedo the main 3d plotter.
        item_key_name: str - like 'new_active_axes'
        render_item: a valid vedo object that can be added to the plotter, like a Mesh or Text3D
        defer_render: bool - whether to immediately render after removing an item or not.
        
        Requirements:
            plotter must have the properties: 
                   plotter.ui.plt
                plotter.plots.meshes
        Usage:
            vedo_create_if_needed(spike_raster_plt_3d_vedo, 'active_window_only_axes', new_mesh)
        
        """
        found_item = plotter.plots.meshes.get(item_key_name, None)
        if found_item is None:
            plotter.plots.meshes[item_key_name] = render_item
            plotter.ui.plt += render_item
            found_item = render_item # set the found_item to be returned to the new item
            if not defer_render:
                plotter.ui.plt.render() # render after adding the new item.
    
        # Either way, returns the found item after ensuring it has been added to the .plots.meshes dict with the correct key, and added to the plotter.
        return found_item
        
        
    
    @classmethod
    def vedo_remove_if_exists(cls, plotter, item_key_name, defer_render=False):
        """ Removes a mesh if it exists in spike_raster_plt_3d_vedo
        plotter: Spike3DRaster_Vedo the main 3d plotter.
        item_key_name: str - like 'new_active_axes'
        defer_render: bool - whether to immediately render after removing an item or not.
        
        Requirements:
            plotter must have the properties: 
                   plotter.ui.plt
                plotter.plots.meshes
        Usage:
            vedo_remove_if_exists(spike_raster_plt_3d_vedo, 'active_window_only_axes')
            vedo_remove_if_exists(spike_raster_plt_3d_vedo, 'active_window_only_axes')
            vedo_remove_if_exists(spike_raster_plt_3d_vedo, 'all_data_axes')
            vedo_remove_if_exists(spike_raster_plt_3d_vedo, 'rect_meshes')
        
        """
        # found_item = spike_raster_plt_3d_vedo.plots.meshes.get(item_key_name, None)
        found_item = plotter.plots.meshes.pop(item_key_name, None) # .pop('key', None) also removes the key from the dict if it was there.
        if found_item is not None:
            plotter.ui.plt.remove(found_item) # remove extant item. 
            # Delete the key from the dictionary:
            # del pike_raster_plt_3d_vedo.plots.meshes[item_key_name] # not needed if using .pop(...) version
            if not defer_render:
                plotter.ui.plt.render() # render after removing the old one.
            return True
        else:
            # Nothing was removed
            return False
        
        