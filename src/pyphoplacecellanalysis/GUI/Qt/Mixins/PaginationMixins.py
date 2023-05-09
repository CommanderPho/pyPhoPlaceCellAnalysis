# From plot_paginated_decoded_epoch_slices
from pyphocorehelpers.indexing_helpers import Paginator
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

from pyphoplacecellanalysis.External.pyqtgraph import QtCore
from pyphoplacecellanalysis.GUI.Qt.Widgets.PaginationCtrl.PaginationControlWidget import PaginationControlWidget


class PaginatedFigureController(QtCore.QObject):
    """2023-05-08 - Holds the current state for real-time pagination 
    
    Potential existing similar implementations
        - that Spike3DWindow event jump utility used to jump to next/prev/specific event (like replays, etc)
        - create_new_figure_if_needed(a_name) or something similar
        - that tabbed matplotlib figure implementation
        - docking widgets in figures
        - from pyphoplacecellanalysis.GUI.Qt.PlaybackControls.Spike3DRasterBottomPlaybackControlBarWidget import Spike3DRasterBottomPlaybackControlBar, on_jump_left
        
	Usage:
    
    from pyphoplacecellanalysis.GUI.Qt.Mixins.PaginationMixins import PaginatedFigureController
        
    """
    params: VisualizationParameters
    plots_data: RenderPlotsData
    plots: RenderPlots
    ui: PhoUIContainer

    @property
    def current_page_idx(self):
        """The curr_page_index property."""
        return self.ui.mw.ui.paginator_controller_widget.current_page_idx

    def __init__(self, params, plots_data, plots, ui, parent=None):
        super(PaginatedFigureController, self).__init__(parent=parent)
        self.params, self.plots_data, self.plots, self.ui = params, plots_data, plots, ui

    def configure(self, **kwargs):
        """ assigns and computes needed variables for rendering. """
        pass

    def initialize(self, **kwargs):
        """ sets up Figures """
        # self.fig, self.axs = plt.subplots(nrows=len(rr_replays))
        pass

    def update(self, **kwargs):
        """ called to specifically render data on the figure. """
        pass

    def on_close(self):
        """ called when the figure is closed. """
        pass
    
    @staticmethod
    def _subfn_helper_add_pagination_control_widget(a_paginator: Paginator, mw, defer_render=True):
        """ Add the PaginationControlWidget to the bottom of the figure """
        # LIMITATION: only works on non-scrollable figures:
        mw.ui.paginator_controller_widget = PaginationControlWidget(n_pages=a_paginator.num_pages)
        mw.ui.root_vbox.addWidget(mw.ui.paginator_controller_widget) # add the pagination control widget
        mw.ui.paginator_controller_widget.setMinimumHeight(38) # Set minimum height so it doesn't disappear
        if not defer_render:
            mw.draw()
            mw.show()