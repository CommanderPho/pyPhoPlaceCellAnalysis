import sys
from copy import deepcopy
from typing import List, Any, Tuple, Optional, Callable
from attrs import define, field, Factory, asdict
import numpy as np
import pandas as pd

from neuropy.utils.result_context import IdentifyingContext

# ==================================================================================================================== #
# 2023-06-21 User Annotations                                                                      #
# ==================================================================================================================== #

@define(slots=False)
class UserAnnotationsManager:
    """ class for holding User Annotations of the data. Performed interactive by the user, and then saved to disk for later use. An example are the selected replays to be used as examples. 
    
    Usage:
        from pyphoplacecellanalysis.General.Model.user_annotations import UserAnnotationsManager
        
    """
    @staticmethod
    def get_user_annotations():
        """ hardcoded user annotations
        

        New Entries can be generated like:
            from pyphoplacecellanalysis.GUI.Qt.Mixins.PaginationMixins import SelectionsObject
            from pyphoplacecellanalysis.General.Model.user_annotations import UserAnnotationsManager
            from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import DecodedEpochSlicesPaginatedFigureController

            ## Stacked Epoch Plot
            example_stacked_epoch_graphics = curr_active_pipeline.display('_display_long_and_short_stacked_epoch_slices', defer_render=False, save_figure=True)
            pagination_controller_L, pagination_controller_S = example_stacked_epoch_graphics.plot_data['controllers']
            ax_L, ax_S = example_stacked_epoch_graphics.axes
            final_figure_context_L, final_context_S = example_stacked_epoch_graphics.context

            user_annotations = UserAnnotationsManager.get_user_annotations()

            ## Capture current user selection
            saved_selection_L: SelectionsObject = pagination_controller_L.save_selection()
            saved_selection_S: SelectionsObject = pagination_controller_S.save_selection()
            final_L_context = saved_selection_L.figure_ctx.adding_context_if_missing(user_annotation='selections')
            final_S_context = saved_selection_S.figure_ctx.adding_context_if_missing(user_annotation='selections')
            user_annotations[final_L_context] = saved_selection_L.flat_all_data_indicies[saved_selection_L.is_selected]
            user_annotations[final_S_context] = saved_selection_S.flat_all_data_indicies[saved_selection_S.is_selected]
            # Updates the context. Needs to generate the code.

            ## Generate code to insert int user_annotations:
            print('Add the following code to UserAnnotationsManager.get_user_annotations() function body:')
            print(f"user_annotations[{final_L_context.get_initialization_code_string()}] = np.array({list(saved_selection_L.flat_all_data_indicies[saved_selection_L.is_selected])})")
            print(f"user_annotations[{final_S_context.get_initialization_code_string()}] = np.array({list(saved_selection_S.flat_all_data_indicies[saved_selection_S.is_selected])})")


        Usage:
            user_anootations = get_user_annotations()
            user_anootations

        """
        user_annotations = {}

        ## IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([13,  14,  15,  25,  27,  28,  31,  37,  42,  45,  48,  57,  61,  62,  63,  76,  79,  82,  89,  90, 111, 112, 113, 115])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([  9,  11,  13,  14,  15,  20,  22,  25,  37,  40,  45,  48,  61, 62,  76,  79,  84,  89,  90,  93,  94, 111, 112, 113, 115, 121])

        ## IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([5,  13,  15,  17,  20,  21,  24,  31,  33,  43,  44,  49,  63, 64,  66,  68,  70,  71,  74,  76,  77,  78,  84,  90,  94,  95, 104, 105, 122, 123])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([ 12,  13,  15,  17,  20,  24,  30,  31,  32,  33,  41,  43,  49, 54,  55,  68,  70,  71,  73,  76,  77,  78,  84,  89,  94, 100, 104, 105, 111, 114, 115, 117, 118, 122, 123, 131])

        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([10, 11, 12, 17, 18, 22])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([10, 11, 12, 16, 18, 19, 23])



        # IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([13, 23, 41, 46])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([4, 7, 10, 15, 21, 23, 41])

        # ## TODO:-
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([2, 6])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([2, 5, 9, 10])
        
        # ## TODO:
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([3, 4, 5])
        
        # ## TODO:
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([])
        # user_annotations[IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([])

        # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([3, 13, 16, 18, 19, 20, 23, 24, 27, 28, 36, 38, 40, 43, 44, 47, 48, 52, 55, 64, 65])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([3, 10, 13, 16, 18, 19, 24, 27, 28, 36, 40, 43, 44, 47, 48, 50, 55, 60, 64, 65])

        # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([2, 13, 18, 23, 25, 27, 32])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([2, 8, 9, 13, 16, 18, 25, 27, 28, 32, 33])

        # IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40')
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='long_results_obj',user_annotation='selections')] = np.array([4, 22, 24, 28, 30, 38, 42, 50, 55, 60, 67, 70, 76, 83, 85, 100, 103, 107, 108, 113, 118, 121, 122, 131, 140, 142, 149, 153, 170, 171])
        user_annotations[IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40',display_fn_name='DecodedEpochSlices',epochs='replays',decoder='short_results_obj',user_annotation='selections')] = np.array([2, 7, 11, 17, 20, 22, 30, 34, 38, 39, 41, 43, 47, 49, 55, 59, 60, 69, 70, 75, 77, 80, 83, 85, 86, 100, 107, 110, 113, 114, 115, 118, 120, 121, 122, 126, 130, 131, 138, 140, 142, 149, 157, 160, 168, 170])


        return user_annotations
    
    @classmethod
    def has_user_annotation(cls, test_context):
        user_anootations = cls.get_user_annotations()
        was_annotation_found: bool = False
        # try to find a matching user_annotation for the final_context_L
        for a_ctx, selections_array in user_anootations.items():
            an_item_diff = a_ctx.diff(test_context)
            if an_item_diff == {('user_annotation', 'selections')}:
                was_annotation_found = True
                break # done looking
        return was_annotation_found

    @staticmethod
    def update_selections_from_annotations(saved_selection, user_anootations, debug_print=True):
        """ 
        
        saved_selection_L.is_selected
        
        
        saved_selection_L = update_selections_from_annotations(saved_selection_L, user_anootations)
        saved_selection_S = update_selections_from_annotations(saved_selection_S, user_anootations)
        ## re-apply the selections:
        pagination_controller_L.restore_selections(saved_selection_L)
        pagination_controller_S.restore_selections(saved_selection_S)

        """
        final_figure_context = saved_selection.figure_ctx
        was_annotation_found = False
        # try to find a matching user_annotation for the final_context_L
        for a_ctx, selections_array in user_anootations.items():
            an_item_diff = a_ctx.diff(final_figure_context)
            if debug_print:
                print(an_item_diff)
                print(f'\t{len(an_item_diff)}')
            if an_item_diff == {('user_annotation', 'selections')}:
                print(f'item found: {a_ctx}\nselections_array: {selections_array}')
                was_annotation_found = True
                saved_selection.is_selected = np.isin(saved_selection.flat_all_data_indicies, selections_array) # update the is_selected
                break # done looking
            
            # print(IdentifyingContext.subtract(a_ctx, final_context_L))
        if not was_annotation_found:
            print(f'WARNING: no matching context found in {len(user_anootations)} annotations. `saved_selection` will be returned unaltered.')
        return saved_selection


    def interactive_good_epoch_selections(self, curr_active_pipeline):
        # Allows the user to interactively select good epochs and generate hardcoded user_annotation entries from the results:
        ## Stacked Epoch Plot
        from pyphoplacecellanalysis.GUI.Qt.Mixins.PaginationMixins import SelectionsObject
        from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import DecodedEpochSlicesPaginatedFigureController

        ## Stacked Epoch Plot
        example_stacked_epoch_graphics = curr_active_pipeline.display('_display_long_and_short_stacked_epoch_slices', defer_render=False, save_figure=False)
        pagination_controller_L, pagination_controller_S = example_stacked_epoch_graphics.plot_data['controllers']
        ax_L, ax_S = example_stacked_epoch_graphics.axes
        figure_context_L, figure_context_S = example_stacked_epoch_graphics.context


        # user_annotations = UserAnnotationsManager.get_user_annotations()
        user_annotations = self.get_user_annotations()

        ## Capture current user selection
        saved_selection_L: SelectionsObject = pagination_controller_L.save_selection()
        saved_selection_S: SelectionsObject = pagination_controller_S.save_selection()
        final_L_context = saved_selection_L.figure_ctx.adding_context_if_missing(user_annotation='selections')
        final_S_context = saved_selection_S.figure_ctx.adding_context_if_missing(user_annotation='selections')
        user_annotations[final_L_context] = saved_selection_L.flat_all_data_indicies[saved_selection_L.is_selected]
        user_annotations[final_S_context] = saved_selection_S.flat_all_data_indicies[saved_selection_S.is_selected]
        # Updates the context. Needs to generate the code.

        ## Generate code to insert int user_annotations:
        print('Add the following code to `pyphoplacecellanalysis.General.Model.user_annotations.UserAnnotationsManager.get_user_annotations()` function body:')
        print(f"user_annotations[{final_L_context.get_initialization_code_string()}] = np.array({list(saved_selection_L.flat_all_data_indicies[saved_selection_L.is_selected])})")
        print(f"user_annotations[{final_S_context.get_initialization_code_string()}] = np.array({list(saved_selection_S.flat_all_data_indicies[saved_selection_S.is_selected])})")

