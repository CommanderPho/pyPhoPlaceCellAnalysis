import vedo
from vedo import Mesh, Cone, Plotter, printc, Glyph
from vedo import Rectangle, Lines, Plane, Axes, merge, colorMap # for StaticVedo_3DRasterHelper
from vedo import Volume, ProgressBar, show, settings

# from pyphocorehelpers.plotting.vedo_qt_helpers import MainVedoPlottingWindow

from pyphoplacecellanalysis.GUI.Vedo.VedoMeshManipulatable import VedoPlotterHelpers


class VedoPlotterHelpers:
    """docstring for VedoHelpers."""
    
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
        
        