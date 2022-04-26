import numpy as np
import pandas as pd

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import pyqtplot_plot_image_array, pyqtplot_common_setup


class DefaultRatemapDisplayFunctions(AllFunctionEnumeratingMixin):
    """ Functions related to visualizing Bayesian Decoder performance. """
    
    def _display_placemaps_pyqtplot_2D(computation_result, active_config, enable_saving_to_disk=False, **kwargs):
        """ Plots the prediction error for the two_step decoder at each point in time.
            Based off of "_temp_debug_two_step_plots_animated_imshow"
        """
        # Get the decoders from the computation result:
        active_one_step_decoder = computation_result.computed_data['pf2D_Decoder'] # doesn't actually require the Decoder, could just use computation_result.computed_data['pf2D']            
        # Get flat list of images:
        images = active_one_step_decoder.ratemap.normalized_tuning_curves # (43, 63, 63)
        # images = active_one_step_decoder.ratemap.normalized_tuning_curves[0:40,:,:] # (43, 63, 63)
        occupancy = active_one_step_decoder.ratemap.occupancy
        app, parent_root_widget, root_render_widget, plot_array, img_item_array = pyqtplot_plot_image_array(active_one_step_decoder.xbin, active_one_step_decoder.ybin, images, occupancy, 
                                                                                app=kwargs.get('app',None), parent_root_widget=kwargs.get('parent_root_widget',None), root_render_widget=kwargs.get('root_render_widget',None))
        # win.show()
        return app, parent_root_widget, root_render_widget


