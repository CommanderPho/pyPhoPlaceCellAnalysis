import numpy as np

class KeysListAccessingMixin:
    """ Provides a helper function to get the keys from a variety of different data-types. Used for combo-boxes."""
    @classmethod
    def get_keys_list(cls, data):
        if isinstance(data, dict):
            keys = list(data.keys())
        elif isinstance(data, list) or isinstance(data, tuple):
            keys = data
        elif isinstance(data, np.ndarray) or isinstance(data, np.void):
            keys = data.dtype.names
        else:
            print("get_keys_list(data): Unknown data type:", type(data), data)
            raise
        return keys
    
    

class ComboBoxCtrlOwnerMixin:
    """ Implementor owns a combo-box control """

    @classmethod
    def replace_combo_items(cls, curr_combo_box, updated_list, debug_print=False):
        ## clear previous old items:
        curr_combo_box.clear()
        num_known_types = len(updated_list)
        if debug_print:
            print(f'num_known_types: {num_known_types}')
        # adding list of items to combo box
        curr_combo_box.addItems(updated_list)
        
    @classmethod
    def get_current_combo_item_selection(cls, curr_combo_box, debug_print=False):
        ## Capture the previous selection:
        selected_index = curr_combo_box.currentIndex()
        selected_item_text = curr_combo_box.itemText(selected_index)  # Get the text at index.
        if debug_print:
            print(f'selected_index: {selected_index}, selected_item_text: {selected_item_text}')
        return (selected_index, selected_item_text)

        
    @classmethod
    def try_select_combo_item_with_text(cls, curr_combo_box, search_text, debug_print=False):
        found_desired_index = curr_combo_box.findText(search_text)
        if debug_print:
            print(f'search_text: {search_text}, found_desired_index: {found_desired_index}')
        ## Re-select the previously selected item:
        curr_combo_box.setCurrentIndex(found_desired_index)
        return found_desired_index
    
    
    

class CheckTableCtrlOwnerMixin:
    """ Implementor owns a CheckTable control """
    
    @classmethod
    def get_current_check_table_row_checkedstate(cls, curr_checktable, debug_print=False):
        """Gets the list of filters for which to do filtering for from the current selections in the checkbox table UI. Returns a list of filter names that are enabled."""
        rows_state = curr_checktable.saveState()['rows']
        # print(f'\t {rows_state}') # [['row[0]', True, False], ['row[1]', False, False]]
        # enabled_filter_names = []
        # known_item_state_dict = {}
        # for a_row in rows_state:
        #     # ['row[0]', True, False]
        #     row_config_name = a_row[0]
        #     row_include_state = a_row[1]
        #     known_item_state_dict[row_config_name] = row_include_state
        return {row_config_name:row_include_state for row_config_name, row_include_state in rows_state}              
    
        
    @classmethod
    def try_check_check_table_row_from_state_dict(cls, curr_checktable, known_checked_state_dict, debug_print=False):
        found_desired_index = curr_checktable.findText(known_checked_state_dict)
        if debug_print:
            print(f'search_text: {known_checked_state_dict}, found_desired_index: {found_desired_index}')
        ## Re-select the previously selected item:
        curr_checktable.setCurrentIndex(found_desired_index)
        return found_desired_index
    
    @classmethod
    def try_check_at_least_one_check_table_row(cls, curr_checktable, debug_print=False):
        if (len(curr_checktable.saveState()['rows']) > 1):
            # if we have one or more rows (columns are assumed to be fixed), set at least the first entry by default
            curr_checktable.set_value(0,0,True)
         