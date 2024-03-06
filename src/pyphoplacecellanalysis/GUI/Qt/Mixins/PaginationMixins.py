from typing import Any, Dict
import numpy as np
from nptyping import NDArray
import pandas as pd
from attrs import define, field, Factory
from benedict import benedict # https://github.com/fabiocaccamo/python-benedict#usage

from neuropy.utils.mixins.dict_representable import SubsettableDictRepresentable

from pyphocorehelpers.indexing_helpers import Paginator
from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.gui.Qt.connections_container import ConnectionsContainer
from pyphocorehelpers.indexing_helpers import safe_find_index_in_list

from pyphoplacecellanalysis.External.pyqtgraph import QtCore
from pyphoplacecellanalysis.GUI.Qt.Widgets.PaginationCtrl.PaginationControlWidget import PaginationControlWidget
from pyphoplacecellanalysis.Pho2D.matplotlib.MatplotlibTimeSynchronizedWidget import MatplotlibTimeSynchronizedWidget


""" refactored to avoid:

TypeError: super(type, obj): obj must be an instance or subtype of type

"""

@define(slots=False, eq=False)
class SelectionsObject(SubsettableDictRepresentable):
    """ represents a selection of epochs in an interactive paginated viewer.
    """
    global_epoch_start_t: float
    global_epoch_end_t: float
    variable_name: str
    figure_ctx: "IdentifyingContext" = field(alias='active_identifying_figure_ctx')

    flat_all_data_indicies: NDArray
    is_selected: NDArray
    epoch_labels: NDArray


    @property
    def selected_indicies(self):
        """The selected_indicies property."""
        return self.flat_all_data_indicies[self.is_selected]
    
    def to_dataframe(self) -> pd.DataFrame:
        # to dataframe:
        dict_repr = self.to_dict()
        selection_epochs_df = pd.DataFrame(dict_repr.subset(['epoch_labels', 'flat_all_data_indicies']))
        selection_epochs_df['is_selected'] = dict_repr['is_selected'].values()
        return selection_epochs_df


    @classmethod
    def init_from_visualization_params(cls, params: VisualizationParameters):
        active_params_dict: benedict = benedict(params.to_dict())
        active_params_dict = active_params_dict.subset(['global_epoch_start_t', 'global_epoch_end_t', 'variable_name', 'active_identifying_figure_ctx', 'flat_all_data_indicies', 'epoch_labels', 'is_selected'])
        active_params_dict['is_selected'] = np.array(list(active_params_dict['is_selected'].values())) # dump the keys
        return cls(**active_params_dict)
        

    def update_selections_from_annotations(self, user_annotations_dict:dict, debug_print=True):
        """ 

        saved_selection_L.is_selected


        saved_selection_L = pagination_controller_L.save_selection()
        saved_selection_S = pagination_controller_S.save_selection()

        saved_selection_L.update_selections_from_annotations(user_annotations_dict=user_annotations)
        saved_selection_S.update_selections_from_annotations(user_annotations_dict=user_annotations)
                
        ## re-apply the selections:
        pagination_controller_L.restore_selections(saved_selection_L)
        pagination_controller_S.restore_selections(saved_selection_S)

        History:
        
            Old Usage:
                ## Capture current user selection
                saved_selection_L = pagination_controller_L.save_selection()
                saved_selection_S = pagination_controller_S.save_selection()

                saved_selection_L = UserAnnotationsManager.update_selections_from_annotations(saved_selection_L, user_annotations)
                saved_selection_S = UserAnnotationsManager.update_selections_from_annotations(saved_selection_S, user_annotations)
        """
        final_figure_context = self.figure_ctx
        was_annotation_found = False
        # try to find a matching user_annotation for the final_context_L
        for a_ctx, selections_array in user_annotations_dict.items():
            an_item_diff = a_ctx.diff(final_figure_context)
            if debug_print:
                print(an_item_diff)
                print(f'\t{len(an_item_diff)}')
            if an_item_diff == {('user_annotation', 'selections')}:
                print(f'item found: {a_ctx}\nselections_array: {selections_array}')
                was_annotation_found = True
                self.is_selected = np.isin(self.flat_all_data_indicies, selections_array) # update the is_selected
                break # done looking
            
            # print(IdentifyingContext.subtract(a_ctx, final_context_L))
        if not was_annotation_found:
            print(f'WARNING: no matching context found in {len(user_annotations_dict)} annotations. `saved_selection` will be returned unaltered.')
        return self



@define(slots=False, eq=False) # eq=False makes hashing and equality by identity, which is appropriate for this type of object
class PaginatedFigureBaseController:
    params: VisualizationParameters = VisualizationParameters(name='PaginatedFigureBaseController')
    plots_data: RenderPlotsData = RenderPlotsData(name='PaginatedFigureBaseController')
    plots: RenderPlots = RenderPlots(name='PaginatedFigureBaseController')
    ui: PhoUIContainer = PhoUIContainer(name='PaginatedFigureBaseController', connections=ConnectionsContainer())

    ## Computed properties:
    @property
    def paginator(self) -> Paginator:
        """The paginator property."""
        return self.plots_data.paginator


    @property
    def paginator_controller_widget(self) -> PaginationControlWidget:
        """ the widget that goes left and right by pages in the bottom of the left plot. """
        return self.ui.mw.ui.paginator_controller_widget
    
    @property
    def plot_widget(self) -> "MatplotlibTimeSynchronizedWidget":
        """ the list of plotting child widgets. """
        return self.ui.mw


    @property
    def current_page_idx(self) -> int:
        """The curr_page_index property."""
        return self.ui.mw.ui.paginator_controller_widget.current_page_idx

    @property
    def total_number_of_items_to_show(self) -> int:
        """The total number of items (subplots usually) to be shown across all pages)."""
        return self.paginator.nItemsToShow

    # def __attrs_pre_init__(self):
    #     super().__init__(parent=None)

    # Selectability/Interactivity Helpers ________________________________________________________________________________ #
    @property
    def is_selected(self) -> NDArray:
        """The selected_indicies property."""
        return np.array(list(self.params.is_selected.values()))
    
    @property
    def selected_indicies(self):
        """The selected_indicies property. 
        `self.params.flat_all_data_indicies` is built using `._subfn_helper_setup_selectability(...)`

        """
        return self.params.flat_all_data_indicies[self.is_selected]


    def save_selection(self) -> SelectionsObject:
        # active_params_backup: VisualizationParameters = _out_pagination_controller.params
        # list(_out_pagination_controller.params.keys())
        # active_params_dict: benedict = benedict(active_params_backup.to_dict())
        # active_params_dict = active_params_dict.subset(['global_epoch_start_t', 'global_epoch_end_t', 'variable_name', 'active_identifying_figure_ctx', 'flat_all_data_indicies', 'epoch_labels', 'is_selected'])
        # active_params_dict['is_selected'] = active_params_dict['is_selected'].values() # dump
        active_selections_object = SelectionsObject.init_from_visualization_params(self.params)
        return active_selections_object

    def restore_selections(self, selections: SelectionsObject, defer_render=False):
        # if not isinstance(selections_dict, benedict):
        # 	selections_dict = benedict(selections_dict)
        # Validate the restore by making sure that we're restoring onto the valid objects
        assert self.params.active_identifying_figure_ctx == selections.figure_ctx
        assert self.params.variable_name == selections.variable_name
        # were_any_updated = False
        for a_selected_index in selections.selected_indicies:
            assert a_selected_index in self.params.flat_all_data_indicies, f"a_selected_index: {a_selected_index} is not in flat_all_data_indicies: {self.params.flat_all_data_indicies}"
            self.params.is_selected[a_selected_index] = True
        # Post:
        self.perform_update_selections(defer_render=defer_render)
        
    def on_click(self, event):
        """ called when an axis is clicked to toggle the selection. """
        if self.params.debug_print:
            print(f'PaginatedFigureBaseController.on_click(...):')

        # Get the clicked Axes object
        ax = event.inaxes
        # Find the axes
        found_index = safe_find_index_in_list(self.plots.axs, ax) # find the index on the page of the ax that was clicked
        if found_index is not None:
            # print(f'{found_index = }')
            current_page_idx = self.current_page_idx
            curr_page_data_indicies = self.paginator.get_page_data(page_idx=current_page_idx)[0] # the [0] returns only the indicies and not the data
            found_data_index = curr_page_data_indicies[found_index]
            print(f'{current_page_idx = }, {found_data_index =}')
            # Toggle the selection status of the clicked Axes
            self.params.is_selected[found_data_index] = (not self.params.is_selected.get(found_data_index, False)) # if never set before, assume that it's not selected
            ## Update visual apperance of axis:
            self.perform_update_ax_selected_state(ax=ax, is_selected=self.params.is_selected[found_data_index])

            # Redraw the figure to show the updated selection
            # event.canvas.draw()
            event.canvas.draw_idle()
        else:
            print(f'\tcould not find the clicked ax: {ax} in the list of axes: {self.plots.axs}')

    def perform_update_ax_selected_state(self, ax, is_selected: bool):
        """ simply updates the visual appearance of the provided ax to indicate whether it's selected. """
        if self.params.debug_print:
            print(f'PaginatedFigureBaseController.perform_update_ax_selected_state(...):')

        # Set the face color of the clicked Axes based on its selection status
        if is_selected:
            ax.patch.set_facecolor('gray')
        else:
            ax.patch.set_facecolor('white')

        selection_artists_dict = self.plots.get('selection_artists_dict', {})
        a_selection_artists = selection_artists_dict.get(ax, {})
        
        # ## Most general version (but unimplemented/unfinished) based off of the new `selection_artists_dict`
        # for an_artist_key, an_artist_dict in selection_artists_dict.items():
        #     if an_artist_dict is not None:
        #         an_artist = an_artist_dict.get(ax, None)
        #         if an_artist is not None:
        #             ## TODO: call the uddate handler here:
        #             an_artist.set_visible(is_selected)


        # Update the selection rectangles for this ax if we have them:
        if a_selection_rect is not None:
            a_selection_rect.set_visible(is_selected)

        # action_button_list_dict = selection_artists_dict.get('action_buttons', {})
        # an_action_buttons_list = action_button_list_dict.get(ax, None)
        an_action_buttons_list = a_selection_rect.get('action_buttons', None)
        if an_action_buttons_list is not None:
            ## TODO: do something here?
            # an_action_buttons_list
            pass

    def perform_update_selections(self, defer_render:bool = False):
        """ called to update the selection when the page is changed or something else happens. """
        if self.params.debug_print:
            print(f'PaginatedFigureBaseController.perform_update_selections(...):')
        
        current_page_idx = self.current_page_idx
        curr_page_data_indicies = self.paginator.get_page_data(page_idx=current_page_idx)[0] # the [0] returns only the indicies and not the data
        assert len(self.plots.axs) == len(curr_page_data_indicies), f"len(plots.axs): {len(self.plots.axs)}, len(curr_page_data_indicies): {len(curr_page_data_indicies)}"
        ## This seems uneeeded, but we'll see:
        self._subfn_build_selectibility_rects_if_needed(self.plots.axs, list(curr_page_data_indicies))
        
        for ax, found_data_idx in zip(self.plots.axs, list(curr_page_data_indicies)): # TODO: might fail for the last page?
            # print(f'found_data_idx: {found_data_idx}')
            # found_data_index = curr_page_data_indicies[found_index]
            # print(f'{current_page_idx = }, {found_data_index =}')
            is_selected = self.params.is_selected.get(found_data_idx, False)
            self.perform_update_ax_selected_state(ax=ax, is_selected=is_selected)
                
        # Redraw the figure to show the updated selection
        if not defer_render:
            self.plots.fig.canvas.draw_idle()
            ax.get_figure().canvas.draw()

    def _subfn_build_selectibility_rects_if_needed(self, axs, curr_page_data_indicies):
        """ adds the selectibility rectangles (patches), one for each axis, to the matplotlib axes provided
        
        Updates: self.plots.selection_artists_dict,
             self.plots.selection_artists_dict.rectangles, self.plots.selection_artists_dict.action_buttons
        Uses: self.params.is_selected
        """
        from matplotlib.patches import Rectangle
        from neuropy.utils.matplotlib_helpers import add_selection_patch

        if not self.plots.has_attr('selection_artists_dict'):
            ## Create the dict if needed and it doesn't exist:
            if self.params.debug_print:
                print(f'building new self.plots.selection_artists_dict as one does not exist')
            self.plots.selection_artists_dict = {}
    
        ## INPUTS: curr_page_data_indicies, axs
        assert len(axs) == len(curr_page_data_indicies), f"len(plots.axs): {len(axs)}, len(curr_page_data_indicies): {len(curr_page_data_indicies)}"
        for ax, found_data_idx in zip(axs, list(curr_page_data_indicies)):
            ## First get the ax
            a_selection_artists_dict = self.plots.selection_artists_dict.get(ax, None)
            if a_selection_artists_dict is None:
                # create a new one
                if self.params.debug_print:
                    print(f'needed a new a_selection_artists_dict.')
                       
                # Define an event for toggling the properties with check buttons
                def on_toggle_action_button(label):
                    # Here you would toggle the associated property
                    print(f"Property '{label}' toggled")
                
                button_properties = [dict(name='is_bad', value=False),
                                     dict(name='needs_review', value=False),
                                     dict(name='is_excluded', value=False),
                                    ] # , action_handler_fn=on_toggle_action_button

                a_selection_rect, action_buttons_list = add_selection_patch(ax, selection_color='green', alpha=0.6, zorder=-1, action_button_configs=button_properties, defer_draw=True)
                # Connect the event handler
                action_buttons_list.on_clicked(on_toggle_action_button)

                self.plots.selection_artists_dict[ax] = {
                                    'rectangles': a_selection_rect,
                                    'action_buttons': action_buttons_list,
                                    }
    
            else:
                a_selection_rect = a_selection_artists_dict['rectangles']
                action_buttons_list = a_selection_artists_dict['action_buttons']
            is_selected = self.params.is_selected.get(found_data_idx, False)
            a_selection_rect.set_visible(is_selected)
            # action_buttons_list # do something

            ## END if a_selection_artists_dict is None:


    def _subfn_clear_selectability_rects(self):
        from matplotlib.widgets import Widget
        import matplotlib.artist as martist
        print(f'_subfn_clear_selectability_rects(): removing...')
        for an_ax, a_selection_artists in self.plots.selection_artists_dict.items():
            if a_selection_artists is not None:
                for sub_k, a_renderable in a_selection_artists.items():
                    if a_renderable is not None:
                        if isinstance(a_renderable, martist.Artist):
                            a_renderable.remove()
                        elif isinstance(a_renderable, Widget):
                            # Widget removal mechanism.
                            for child in a_renderable.ax.get_children():
                                child.remove()
                            a_renderable.disconnect_events()
                        elif isinstance(a_renderable, (list, tuple)) and all(isinstance(item, martist.Artist) for item in a_renderable):
                            for an_artist_part in a_renderable:
                                if an_artist_part is not None:
                                    an_artist_part.remove()
                        else:
                            raise UserWarning(f"sub_k: {sub_k} -- Neither an artist nor a list of artists")
                            pass

                                                        
        self.plots['selection_artists_dict'] = {}
        del self.plots['selection_artists_dict']

        # for ax, a_selection_rect in self.plots.selection_rectangles_dict.items():
        #     a_selection_rect.remove()
        #     a_buttons = self.plots.action_buttons_dict.get(ax, None)
        #     if a_buttons is not None:
        #         for a_btn in a_buttons:
        #             if a_btn is not None:
        #                 a_btn.remove()
        # self.plots[ 'rectangles'] = {}
        # self.plots['action_buttons'] = {}
        # del self.plots[ 'rectangles']
        # del self.plots['action_buttons']


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


    def _perform_clear_all_selections(self):
        """ Resets all selections (setting is_selected to False) internally, not graphical.
        """
        self._subfn_helper_setup_selectability()

    # Context and titles _________________________________________________________________________________________________ #
    def perform_update_titles_from_context(self, page_idx:int, included_page_data_indicies, **kwargs):
        """ Tries to update the figure suptitle and the window title from the self.params.active_identifying_figure_ctx if it's set

        Requires: `self.params.active_identifying_figure_ctx`, `self.paginator.num_pages`
        """
        if self.params.get('active_identifying_figure_ctx', None) is not None:
            collision_prefix = kwargs.pop('collision_prefix', '_DecodedEpochSlices_plot_test_')
            context_kwargs = kwargs
            if (self.paginator.num_pages > 1):
                 # ideally wouldn't include page number unless (self.paginator.num_pages > 1)
                 context_kwargs['page'] = f'{page_idx+1}of{self.paginator.num_pages}'
            context_kwargs['aclus'] = f"{included_page_data_indicies}" # BUG: these aren't aclus when plotting epochs or something else.
            # Build the context:
            active_identifying_ctx = self.params.active_identifying_figure_ctx
            if len(context_kwargs) > 0:
                active_identifying_ctx = active_identifying_ctx.adding_context(collision_prefix, **context_kwargs) 

            final_context = active_identifying_ctx # Display/Variable context mode
            active_identifying_ctx_string = final_context.get_description(separator='|') # Get final discription string
            if kwargs.get('debug_print', False):
                print(f'active_identifying_ctx_string: "{active_identifying_ctx_string}"')
            self.update_titles(active_identifying_ctx_string)
        else:
            active_identifying_ctx = None
            self.update_titles("no context set!")

        return active_identifying_ctx


    def update_titles(self, window_title: str, suptitle: str = None):
        """ sets the suptitle and window title for the figure """
        if suptitle is None:
            suptitle = window_title # same as window title
        # Set the window title:
        self.ui.mw.setWindowTitle(window_title)
        self.ui.mw.fig.suptitle(suptitle, wrap=True) # set the plot suptitle
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
        mw.ui.paginator_controller_widget.setFixedHeight(21)
        # mw.ui.paginator_controller_widget.setMinimumHeight(24) # Set minimum height so it doesn't disappear
        if not defer_render:
            mw.draw()
            mw.show()
            

@define(slots=False, eq=False) # eq=False makes hashing and equality by identity, which is appropriate for this type of object
class PaginatedFigureController(PaginatedFigureBaseController):
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

    # ==================================================================================================================== #
    # Common lifecycle that inheritors can override                                                                        #
    # ==================================================================================================================== #
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
    


    
# ==================================================================================================================== #
# Pagination Data Providers                                                                                            #
# ==================================================================================================================== #

class PaginatedPlotDataProvider:
    """ Provides auxillary and optional data to paginated plots, currently of decoded posteriors. 

    from pyphoplacecellanalysis.GUI.Qt.Mixins.PaginationMixins import PaginatedPlotDataProvider

    
    
    """
    provided_params: Dict[str, Any] = dict(enable_weighted_correlation_info = True)
    provided_plots_data: Dict[str, Any] = {'weighted_corr_data': None}
    provided_plots: Dict[str, Any] = {'weighted_corr': {}}

    @classmethod
    def get_provided_params(cls) -> Dict[str, Any]:
        return cls.provided_params

    @classmethod
    def get_provided_plots_data(cls) -> Dict[str, Any]:
        return cls.provided_plots_data
    
    @classmethod
    def get_provided_plots(cls) -> Dict[str, Any]:
        return cls.provided_plots

    @classmethod
    def get_provided_callbacks(cls) -> Dict[str, Dict]:
        """ override """
        return {'on_render_page_callbacks': 
                {'plot_wcorr_data': cls._callback_update_curr_single_epoch_slice_plot}
        }


    @classmethod
    def add_data_to_pagination_controller(cls, a_pagination_controller, *provided_data, update_controller_on_apply:bool=False):
        """ should be general I think.

        Adds the required information to the pagination_controller's .params, .plots, .plots_data, .ui

        Uses: cls.provided_params

        """
        ## Add the .params:
        for a_key, a_value in cls.provided_params.items():
            if not a_pagination_controller.params.has_attr(a_key):
                a_pagination_controller.params[a_key] = a_value

        ## Add the .plots_data:
        assert len(provided_data) == 1
        # weighted_corr_data = provided_data[1]
        assert len(provided_data) == len(cls.provided_plots_data), f"len(provided_data): {len(provided_data)} != len(cls.provided_plots_data): {len(cls.provided_plots_data)}"
        active_plots_data = {k:(provided_data[i] or default_class_value) for i, (k, default_class_value) in enumerate(cls.provided_plots_data.items())}

        for a_key, a_value in active_plots_data.items():
            a_pagination_controller.plots_data[a_key] = a_value

        ## Add the .plots:
        for a_key, a_value in cls.provided_plots.items():
            a_pagination_controller.plots[a_key] = a_value

        ## Add the callbacks
        for a_callback_type, a_callback_dict in cls.get_provided_callbacks().items():
            # a_callback_type: like 'on_render_page_callbacks'
            pagination_controller_callbacks_dict = a_pagination_controller.params.get(a_callback_type, None)
            if pagination_controller_callbacks_dict is None:
                a_pagination_controller.params[a_callback_type] = {} # allocate a new dict to hold callbacks
            # register the specific callbacks of this type:
            for a_callback_id, a_callback_fn in a_callback_dict.items():
                a_pagination_controller.params[a_callback_type][a_callback_id] = a_callback_fn

        # Trigger the update
        if update_controller_on_apply:
            a_pagination_controller.on_paginator_control_widget_jump_to_page(0)
        

    # @classmethod
    # def _callback_update_curr_single_epoch_slice_plot(cls, curr_ax, params: "VisualizationParameters", plots_data: "RenderPlotsData", plots: "RenderPlots", ui: "PhoUIContainer",
    #                                                    data_idx:int, curr_time_bins, *args, epoch_slice=None, curr_time_bin_container=None, **kwargs): # curr_posterior, curr_most_likely_positions, debug_print:bool=False
    #     """ 
    #     Called with:

    #         self.params, self.plots_data, self.plots, self.ui = a_callback(curr_ax, self.params, self.plots_data, self.plots, self.ui, curr_slice_idxs, curr_time_bins, curr_posterior, curr_most_likely_positions, debug_print=self.params.debug_print)

    #     Data:
    #         plots_data.weighted_corr_data
    #     Plots:
    #         plots['weighted_corr']

    #     """
    #     from neuropy.utils.matplotlib_helpers import add_inner_title # plot_decoded_epoch_slices_paginated
    #     from matplotlib.offsetbox import AnchoredText

    #     debug_print = kwargs.pop('debug_print', True)
    #     if debug_print:
    #         print(f'WeightedCorrelationPaginatedPlotDataProvider._callback_update_curr_single_epoch_slice_plot(..., data_idx: {data_idx}, curr_time_bins: {curr_time_bins})')
        
    #     if epoch_slice is not None:
    #         if debug_print:
    #             print(f'\tepoch_slice: {epoch_slice}')
    #         assert len(epoch_slice) == 2
    #         epoch_start_t, epoch_end_t = epoch_slice # unpack
    #         if debug_print:
    #             print(f'\tepoch_start_t: {epoch_start_t}, epoch_end_t: {epoch_end_t}')
    #     else:
    #         raise NotImplementedError(f'epoch_slice is REQUIRED to index into the wcorr_data dict, but is None!')
        
    #     raise NotImplementedError(f"inheriting classes should be overriding this method!")

    #     if debug_print:
    #         print(f'\t success!')
    #     return params, plots_data, plots, ui




