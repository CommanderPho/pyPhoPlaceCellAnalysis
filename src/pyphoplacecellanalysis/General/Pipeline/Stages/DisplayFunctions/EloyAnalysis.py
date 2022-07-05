import numpy as np
import pandas as pd

from pyphocorehelpers.mixins.member_enumerating import AllFunctionEnumeratingMixin
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
# from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import pyqtplot_plot_image_array, pyqtplot_common_setup

import pyphoplacecellanalysis.External.pyqtgraph as pg


class EloyAnalysisDisplayFunctions(AllFunctionEnumeratingMixin, metaclass=DisplayFunctionRegistryHolder):
    """ Functions related to visualizing results related to Pho's 2022 Analysis of Placefield Density and Animal Speed for Eloy """
    
    # def _display_speed_vs_PFoverlapDensity_plots(computation_result, active_config, enable_saving_to_disk=False, app=None, parent_root_widget=None, root_render_widget=None, debug_print=False, **kwargs):
    def _display_speed_vs_PFoverlapDensity_plots(computation_result, active_config, enable_saving_to_disk=False, debug_print=False, **kwargs):
        """ Plot the 1D and 2D sorted avg_speed_per_pos and PFoverlapDensity to reveal any trends
        """
        active_eloy_analysis = computation_result.computed_data.get('EloyAnalysis', None)
        # root_render_widget, parent_root_widget, app = pyqtplot_common_setup(f'_display_speed_vs_PFoverlapDensity_plots', app=app, parent_root_widget=parent_root_widget, root_render_widget=root_render_widget)

        ## 1D:
        ## Plot the sorted avg_speed_per_pos and PFoverlapDensity to reveal any trends:
        out_plot_1D = pg.plot(active_eloy_analysis.sorted_1D_avg_speed_per_pos, active_eloy_analysis.sorted_PFoverlapDensity_1D, pen=None, symbol='o', title='Sorted 1D AVG Speed per Pos vs. Sorted 1D PFOverlapDensity', left='Sorted 1D PFOverlapDensity', bottom='Sorted 1D AVG Speed per Pos bin (x)') ## setting pen=None disables line drawing
        # out_plot_1D = root_render_widget.addPlot(row=curr_row, col=curr_col, name=curr_plot_identifier_string, title=curr_cell_identifier_string)
        
        ## 2D:
        ## Plot the sorted avg_speed_per_pos and PFoverlapDensity to reveal any trends:
        out_plot_2D = pg.plot(active_eloy_analysis.sorted_avg_2D_speed_per_pos, active_eloy_analysis.sorted_PFoverlapDensity_2D, pen=None, symbol='o', title='Sorted AVG 2D Speed per Pos vs. Sorted 2D PFOverlapDensity', left='Sorted 2D PFOverlapDensity', bottom='Sorted AVG 2D Speed per Pos bin (x,y)') ## setting pen=None disables line drawing
        
        return out_plot_1D, out_plot_2D
        # return app, parent_root_widget, root_render_widget


