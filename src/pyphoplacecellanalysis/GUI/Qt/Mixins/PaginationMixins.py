from copy import deepcopy
from logging import warning
from typing import Any, Dict, List
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
from pyphoplacecellanalysis.GUI.Qt.Widgets.PaginationCtrl.PaginationControlWidget import PaginationControlWidget, PaginationControlWidgetState
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

# ==================================================================================================================== #
# Pagination Data Providers                                                                                            #
# ==================================================================================================================== #

class PaginatedPlotDataProvider:
    """ Provides auxillary and optional data to paginated plots, currently of decoded posteriors. 

    from pyphoplacecellanalysis.GUI.Qt.Mixins.PaginationMixins import PaginatedPlotDataProvider

    Implmentors must provide:

        plots_group_identifier_key: str = 'weighted_corr'

        @classmethod
        def get_provided_callbacks(cls) -> Dict[str, Dict]:
            return {'on_render_page_callbacks': 
                    {'plot_wcorr_data': cls._callback_update_curr_single_epoch_slice_plot}
            }


        @classmethod
        def _callback_update_curr_single_epoch_slice_plot(cls, curr_ax, params: "VisualizationParameters", plots_data: "RenderPlotsData", plots: "RenderPlots", ui: "PhoUIContainer", data_idx:int, curr_time_bins, *args, epoch_slice=None, curr_time_bin_container=None, **kwargs):
            pass
    
    
    """
    provided_params: Dict[str, Any] = dict(enable_weighted_correlation_info = True)
    provided_plots_data: Dict[str, Any] = {'weighted_corr_data': None}
    provided_plots: Dict[str, Any] = {'weighted_corr': {}}
    column_names: List[str] = []
    

    @classmethod
    def get_provided_params(cls) -> Dict[str, Any]:
        return deepcopy(cls.provided_params)

    @classmethod
    def get_provided_plots_data(cls) -> Dict[str, Any]:
        return deepcopy(cls.provided_plots_data)
    
    @classmethod
    def get_provided_plots(cls) -> Dict[str, Any]:
        return deepcopy(cls.provided_plots)

    @classmethod
    def get_provided_callbacks(cls) -> Dict[str, Dict]:
        """ override """
        return deepcopy({'on_render_page_callbacks': 
                {'plot_wcorr_data': cls._callback_update_curr_single_epoch_slice_plot}
        })

    @classmethod
    def add_data_to_pagination_controller(cls, a_pagination_controller, *provided_data, update_controller_on_apply:bool=False):
        """ should be general I think.

        Adds the required information to the pagination_controller's .params, .plots, .plots_data, .ui
        Also initializes the callbacks
        
        
        Uses: cls.provided_params

        """
        ## Add the .params:
        for a_key, a_value in cls.get_provided_params().items():
            if not a_pagination_controller.params.has_attr(a_key):
                a_pagination_controller.params[a_key] = a_value # DEEPCOPY ALT

        ## Add the .plots_data:
        assert len(provided_data) == 1
        # weighted_corr_data = provided_data[1]
        assert len(provided_data) == len(cls.get_provided_plots_data()), f"len(provided_data): {len(provided_data)} != len(cls.get_provided_plots_data()): {len(cls.get_provided_plots_data())}"
        active_plots_data = {k:(deepcopy(provided_data[i]) or default_class_value) for i, (k, default_class_value) in enumerate(cls.get_provided_plots_data().items())}

        for a_key, a_value in active_plots_data.items():
            a_pagination_controller.plots_data[a_key] = a_value # DEEPCOPY ALT

        ## Add the .plots:
        for a_key, a_value in cls.get_provided_plots().items():
            a_pagination_controller.plots[a_key] = a_value # DEEPCOPY ALT

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
        


    @classmethod
    def remove_data_from_pagination_controller(cls, a_pagination_controller, should_remove_params:bool=False, update_controller_on_apply:bool=False):
        """ Removes the added plots from the pagination_controler

        """
        ## Remove the callbacks
        for a_callback_type, a_callback_dict in cls.get_provided_callbacks().items():
            pagination_controller_callbacks_dict = a_pagination_controller.params.get(a_callback_type, None)
            if pagination_controller_callbacks_dict is not None:
                for a_callback_id in a_callback_dict.keys():
                    if a_callback_id in pagination_controller_callbacks_dict:
                        del pagination_controller_callbacks_dict[a_callback_id]

        ## Remove the .plots:
        for a_key in cls.get_provided_plots().keys():
            if a_key in a_pagination_controller.plots:
                del a_pagination_controller.plots[a_key]
                
        ## Remove the .plots_data:
        for a_key, default_class_value in cls.get_provided_plots_data().items():
            if a_key in a_pagination_controller.plots_data:
                del a_pagination_controller.plots_data[a_key]
                    
        ## Remove the .params:
        if should_remove_params:
            for a_key in cls.get_provided_params().keys():
                if a_pagination_controller.params.has_attr(a_key):
                    del a_pagination_controller.params[a_key]

        # Trigger the update
        if update_controller_on_apply:
            a_pagination_controller.on_paginator_control_widget_jump_to_page(0)

    





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
    def plot_widget(self) -> "MatplotlibTimeSynchronizedWidget":
        """ the list of plotting child widgets. """
        return self.ui.mw


    # MODE(isPaginatorControlWidgetBackedMode) == True: paginator_controller_widget (PaginationControlWidget) backed-mode (default) __________________________________________________________________ #
    @property
    def paginator_controller_widget(self) -> PaginationControlWidget:
        """ the widget that goes left and right by pages in the bottom of the left plot. """
        assert self.params.get('isPaginatorControlWidgetBackedMode', True)
        return self.ui.mw.ui.paginator_controller_widget
    
    # MODE(isPaginatorControlWidgetBackedMode) == False: Proposed state-backed (PaginationControlWidgetState) mode without `paginator_controller_widget` (2024-03-06) ______________________________________ #
    # self isn't QtObject-based so can't emit signals
    # this section is entirely copied from the interface provided by `PaginationControlWidget` as a compatibility replacement

    def get_total_pages(self) -> int:
        if self.params.get('isPaginatorControlWidgetBackedMode', True):
            # MODE(isPaginatorControlWidgetBackedMode) == True: paginator_controller_widget (PaginationControlWidget) backed-mode (default)
            return self.paginator_controller_widget.get_total_pages()
        else:
            # MODE(isPaginatorControlWidgetBackedMode) == False: Proposed state-backed (PaginationControlWidgetState) mode without `paginator_controller_widget` (2024-03-06)
            return self.pagination_state.n_pages

    @property
    def pagination_state(self) -> PaginationControlWidgetState:
        """ Used only when isPaginatorControlWidgetBackedMode == False. The state normally owned by PaginationControlWidget """
        assert (not self.params.get('isPaginatorControlWidgetBackedMode', False))
        return self.params.state
        

    @property
    def current_page_idx(self) -> int:
        """ the 0-based index of the current page. """
        if self.params.get('isPaginatorControlWidgetBackedMode', True):
            # MODE(isPaginatorControlWidgetBackedMode) == True: paginator_controller_widget (PaginationControlWidget) backed-mode (default)
            return self.paginator_controller_widget.current_page_idx
        else:
            # MODE(isPaginatorControlWidgetBackedMode) == False: Proposed state-backed (PaginationControlWidgetState) mode without `paginator_controller_widget` (2024-03-06)
            return self.pagination_state.current_page_idx
    
    def go_to_page(self, page_number: int):
        """ one-based page_number """
        if self.params.get('isPaginatorControlWidgetBackedMode', True):
            # MODE(isPaginatorControlWidgetBackedMode) == True: paginator_controller_widget (PaginationControlWidget) backed-mode (default)
            return self.paginator_controller_widget.go_to_page(page_number)
        else:
            # MODE(isPaginatorControlWidgetBackedMode) == False: Proposed state-backed (PaginationControlWidgetState) mode without `paginator_controller_widget` (2024-03-06)
            if page_number > 0 and page_number <= self.get_total_pages():
                updated_page_idx = page_number - 1 # convert the page number to a page index
                self.pagination_state.current_page_idx = updated_page_idx ## update the state

    def update_page_idx(self, updated_page_idx: int):
        """ this value is safe to bind to. """
        return self.programmatically_update_page_idx(updated_page_idx=updated_page_idx, block_signals=False)

    def programmatically_update_page_idx(self, updated_page_idx: int, block_signals:bool=False) -> bool:
        """ Programmatically updates the spinBoxPage with the zero-based page_number 
        page number (1-based) is always one greater than the page_index (0-based)
        """
        updated_page_number = updated_page_idx + 1 # page number (1-based) is always one greater than the page_index (0-based)
        assert ((updated_page_number > 0) and (updated_page_number <= self.get_total_pages())), f"programmatically_update_page_idx(updated_page_idx: {updated_page_idx}) is invalid! updated_page_number: {updated_page_number}, total_pages: {self.get_total_pages()}"
        if self.params.get('isPaginatorControlWidgetBackedMode', True):
            # MODE(isPaginatorControlWidgetBackedMode) == True: paginator_controller_widget (PaginationControlWidget) backed-mode (default)
            did_change: bool = (self.paginator_controller_widget.current_page_idx != updated_page_idx)
            self.paginator_controller_widget.programmatically_update_page_idx(updated_page_idx, block_signals=block_signals) # updates the embedded pagination widget
        else:
            # MODE(isPaginatorControlWidgetBackedMode) == False: Proposed state-backed (PaginationControlWidgetState) mode without `paginator_controller_widget` (2024-03-06)
            did_change: bool = (self.pagination_state.current_page_idx != updated_page_idx)
            self.pagination_state.current_page_idx = updated_page_idx ## update the state
            # since there are no signals that will be emited from the paginator_controller_widget to trigger updates of self, need to manually call the function that's usually bound
            if not block_signals:
                self.on_jump_to_page(page_idx=updated_page_idx)

        return did_change


    # MODE(isPaginatorControlWidgetBackedMode): Common ___________________________________________________________________ #


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
        a_selection_rect = a_selection_artists.get('rectangles', None)
        if a_selection_rect is not None:
            a_selection_rect.set_visible(is_selected)

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

        enable_per_epoch_action_buttons: bool = self.params.get('enable_per_epoch_action_buttons', False)

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
                       
                
                if enable_per_epoch_action_buttons:
                    # Define an event for toggling the properties with check buttons
                    def on_toggle_action_button(label):
                        # Here you would toggle the associated property
                        print(f"Property '{label}' toggled")
                    
                    button_properties = [dict(name='is_bad', value=False),
                                        dict(name='needs_review', value=False),
                                        dict(name='is_excluded', value=False),
                                        ] # , action_handler_fn=on_toggle_action_button
                else:
                    button_properties = None

                a_selection_rect, action_buttons_list = add_selection_patch(ax, selection_color='green', alpha=0.6, zorder=-1, action_button_configs=button_properties, defer_draw=True)

                self.plots.selection_artists_dict[ax] = {
                                    'rectangles': a_selection_rect,
                                    # 'action_buttons': action_buttons_list,
                                    }
                

                if action_buttons_list is not None:
                    # Connect the event handler
                    action_buttons_list.on_clicked(on_toggle_action_button)
                    self.plots.selection_artists_dict[ax]['action_buttons'] = action_buttons_list

            else:
                a_selection_rect = a_selection_artists_dict['rectangles']
                action_buttons_list = a_selection_artists_dict.get('action_buttons', None)

            if found_data_idx is None:
                is_selected = False
            else:
                is_selected = self.params.is_selected.get(found_data_idx, False)
                
            a_selection_rect.set_visible(is_selected)

            if (action_buttons_list is not None) and (not enable_per_epoch_action_buttons):
                ##TODO: remove the action buttons or hide them
                action_buttons_list.set_visible(is_selected)

            if (enable_per_epoch_action_buttons and (action_buttons_list is None)):
                print(f'WARN: enable_per_epoch_action_buttons: true but action buttons are only created on the first run of `_subfn_build_selectibility_rects_if_needed`. If you want them you can remove all selctions stuff using the clear function and re-add')


            ## Do something with the action buttons
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


    def _helper_setup_pagination_control(self):
        """ Add the PaginationControlWidget or setup the PaginationControlWidgetState depending on isPaginatorControlWidgetBackedMode. 
        """
        ## 
        isPaginatorControlWidgetBackedMode = self.params.get('isPaginatorControlWidgetBackedMode', None)
        if isPaginatorControlWidgetBackedMode is None:
            isPaginatorControlWidgetBackedMode = True
            self.params.isPaginatorControlWidgetBackedMode = isPaginatorControlWidgetBackedMode
        
        if isPaginatorControlWidgetBackedMode:
            self._subfn_helper_add_pagination_control_widget(self.paginator, self.ui.mw, defer_render=False) # minimum height is 21
        else:
            self.params.state = PaginationControlWidgetState(n_pages=self.paginator.num_pages, current_page_idx=0)

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
    

    def on_jump_to_page(self, page_idx: int):
        """ Called when the page index is changed to update the figure
    
        """
        warning(f"WARN: Implementors should override this function to handle updates when the page is changed.")
        if self.params.debug_print:
            self.ui.print(f'PaginatedFigureController.on_jump_to_page(page_idx: {page_idx})') # for page_idx == max_index this is called but doesn't continue
        included_page_data_indicies, (curr_page_active_filter_epochs, curr_page_epoch_labels, curr_page_time_bin_containers, curr_page_posterior_containers) = self.plots_data.paginator.get_page_data(page_idx=page_idx)
        if self.params.debug_print:
            self.ui.print(f'\tincluded_page_data_indicies: {included_page_data_indicies}')



    



