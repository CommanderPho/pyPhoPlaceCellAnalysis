import numpy as np

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

    ## Computed properties:
    @property
    def paginator(self):
        """The paginator property."""
        return self.plots_data.paginator

    @property
    def current_page_idx(self):
        """The curr_page_index property."""
        return self.ui.mw.ui.paginator_controller_widget.current_page_idx

    @property
    def total_number_of_items_to_show(self):
        """The total number of items (subplots usually) to be shown across all pages)."""
        return self.paginator.nItemsToShow

    def __init__(self, params, plots_data, plots, ui, parent=None):
        super(PaginatedFigureController, self).__init__(parent=parent)
        self.params, self.plots_data, self.plots, self.ui = params, plots_data, plots, ui

    def configure(self, **kwargs):
        """ assigns and computes needed variables for rendering. """
        self._subfn_helper_setup_selectability()

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
    

    # Selectability/Interactivity Helpers ________________________________________________________________________________ #
    def on_click(self, event):
        """ called when an axis is clicked to toggle the selection. """
        # Get the clicked Axes object
        ax = event.inaxes
        # Find the axes
        found_index = safe_find_index_in_list(self.plots.axs, ax) # find the index on the page of the ax that was clicked
        # print(f'{found_index = }')
        current_page_idx = self.current_page_idx
        curr_page_data_indicies = self.paginator.get_page_data(page_idx=current_page_idx)[0] # the [0] returns only the indicies and not the data
        found_data_index = curr_page_data_indicies[found_index]
        # print(f'{current_page_idx = }, {found_data_index =}')
        # Toggle the selection status of the clicked Axes
        self.is_selected[found_data_index] = not self.is_selected.get(found_data_index, False) # if never set before, assume that it's not selected
        ## Update visual apperance of axis:
        self.perform_update_ax_selected_state(ax=ax, is_selected=self.is_selected[found_data_index])

        # Redraw the figure to show the updated selection
        event.canvas.draw()

    def perform_update_ax_selected_state(self, ax, is_selected: bool):
        """ simply updates the visual appearance of the provided ax to indicate whether it's selected. """
        # Set the face color of the clicked Axes based on its selection status
        if is_selected:
            ax.patch.set_facecolor('gray')
        else:
            ax.patch.set_facecolor('white')

    def perform_update_selections(self, defer_render:bool = False):
        """ called to update the selection when the page is changed or something else happens. """        
        current_page_idx = self.current_page_idx
        curr_page_data_indicies = self.paginator.get_page_data(page_idx=current_page_idx)[0] # the [0] returns only the indicies and not the data
        assert len(self.plots.axs) == len(curr_page_data_indicies), f"len(plots.axs): {len(self.plots.axs)}, len(curr_page_data_indicies): {len(curr_page_data_indicies)}"
        for ax, found_data_idx in zip(self.plots.axs, list(curr_page_data_indicies)): # TODO: might fail for the last page?
            # print(f'found_data_idx: {found_data_idx}')
            # found_data_index = curr_page_data_indicies[found_index]
            # print(f'{current_page_idx = }, {found_data_index =}')
            is_selected = self.is_selected.get(found_data_idx, False)
            self.perform_update_ax_selected_state(ax=ax, is_selected=is_selected)
                
        # Redraw the figure to show the updated selection
        if not defer_render:
            self.fig.canvas.draw()

    def _subfn_helper_setup_selectability(self):
        """ sets up selectability of items. 
        Requires: `self.paginator.included_combined_indicies_pages`
        Sets: `self.params.flat_all_data_indicies`, `self.params.is_selected`
        """
        ## Recover all the data indicies for the all pages combined (`flat_all_data_indicies`):
        curr_all_pages_lists = self.paginator.included_combined_indicies_pages # [page_idx]
        flat_all_data_indicies = []
        for a_single_page_list in curr_all_pages_lists:
            included_page_data_indicies = np.array([curr_included_data_index for (a_linear_index, curr_row, curr_col, curr_included_data_index) in a_single_page_list]) # a list of the data indicies on this page
            flat_all_data_indicies.append(included_page_data_indicies)

        self.params.flat_all_data_indicies = np.hstack(flat_all_data_indicies)
        ## Initialize `params.is_selected` to False for each item:
        self.params.is_selected = dict(zip(self.params.flat_all_data_indicies, np.repeat(False, self.total_number_of_items_to_show))) # Repeat "False" for each item



    # Context and titles _________________________________________________________________________________________________ #
    def perform_update_titles_from_context(self, page_idx:int, included_page_data_indicies, **kwargs):
        """ Tries to update ethe figure suptitle and the window title from the self.params.active_identifying_figure_ctx if it's set

        Requires: `self.params.active_identifying_figure_ctx`, `self.paginator.num_pages`
        """
        if self.params.get('active_identifying_figure_ctx', None) is not None:
            active_identifying_ctx = self.params.active_identifying_figure_ctx.adding_context(**kwargs, page=f'{page_idx+1}of{self.paginator.num_pages}', aclus=f"{included_page_data_indicies}")
            final_context = active_identifying_ctx # Display/Variable context mode
            active_identifying_ctx_string = final_context.get_description(separator='|') # Get final discription string
            print(f'active_identifying_ctx_string: "{active_identifying_ctx_string}"')
            # active_figure_save_basename = build_figure_basename_from_display_context(final_context)
            self.update_titles(active_identifying_ctx_string)
        else:
            active_identifying_ctx = None
            self.update_titles("no context set!")

        return active_identifying_ctx

    def update_titles(self, window_title: str, suptitle: str = None):
        """ sets the titles for the figure """
        if suptitle is None:
            suptitle = window_title # same as window title
        # Set the window title:
        self.ui.mw.setWindowTitle(window_title)
        self.ui.mw.fig.suptitle(suptitle) # set the plot suptitle
        self.ui.mw.draw()


    # ==================================================================================================================== #
    # Static Methods                                                                                                       #
    # ==================================================================================================================== #

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