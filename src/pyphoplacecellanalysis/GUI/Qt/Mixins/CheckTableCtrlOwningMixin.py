import numpy as np

class CheckTableCtrlOwningMixin:
    """ Implementor owns a CheckTable control
    
    Required Properties:
        self.configRows = []
        self.ctrls['included_configs_table']
    
    """

    @property
    def check_table_ctrl(self):
        """ The checktable control """
        return self.ctrls['included_configs_table']


    @property
    def check_table_sibling_ctrls(self):
        """check_table_sibling_ctrls are controls that will have .blockSignals(True), ..., .blockSignals(False) called on them when programmtically updating the check_table."""
        return self.ctrls.values()


    @property
    def check_table_row_checkedstate(self):
        """ The checktable control """
        return self.get_current_check_table_row_checkedstate(self.check_table_ctrl) # {'maze1': True, 'maze2': False, 'maze': False}
    
    @property
    def all_filters(self):
        """Gets the list of all possible filters for which to do filtering for from the current selections in the checkbox table UI. Returns a list of filter names."""
        return self.configRows
    
    @property
    def enabled_filters(self):
        """Gets the list of filters for which to do filtering for from the current selections in the checkbox table UI. Returns a list of filter names that are enabled."""
        rows_state = self.check_table_ctrl.saveState()['rows']
        # print(f'\t {rows_state}') # [['row[0]', True, False], ['row[1]', False, False]]
        enabled_filter_names = []
        for a_row in rows_state:
            # ['row[0]', True, False]
            row_config_name = a_row[0]
            row_include_state = a_row[1]
            if row_include_state:
                enabled_filter_names.append(row_config_name)
        return enabled_filter_names      
    
        
    @property
    def is_action_enabled(self):
        """The is_action_enabled property."""
        return (len(self.enabled_filters) > 0) # if we have one or more enabled filter the action can be performed. Otherwise it's disabled.
    
    
    def updateConfigRows(self, data):
        """ updates the self.configRows and the accompanying controls from the data """
        if isinstance(data, dict):
            keys = list(data.keys())
        elif isinstance(data, list) or isinstance(data, tuple):
            keys = data
        elif isinstance(data, np.ndarray) or isinstance(data, np.void):
            keys = data.dtype.names
        else:
            print("Unknown data type:", type(data), data)
            return
            
        for c in self.check_table_sibling_ctrls:
            c.blockSignals(True)
        #for c in [self.ctrls['included_configs'], self.ctrls['y'], self.ctrls['size']]:
        for c in [self.check_table_ctrl]:
            c.updateRows(keys) # update the rows with the config rows

        for c in self.check_table_sibling_ctrls:
            c.blockSignals(False)
        # Update the self.keys value:
        self.configRows = keys
        self.ui_update()

    def selectFirstConfigRow(self):
        """ convenience method to programmatically select the first config row if there is one """
        if len(self.all_filters) > 0:
            self.try_check_at_least_one_check_table_row(self.check_table_ctrl)
        return self.enabled_filters
    
    def selectAllConfigRows(self):
        """ convenience method to programmatically select all known filters """
        all_enabled_dict = {a_config_name:True for a_config_name in self.all_filters}
        self.try_check_check_table_row_from_state_dict(self.check_table_ctrl, all_enabled_dict)
        return self.enabled_filters
    
    def clearAllSelectedConfigRows(self):
        """ convenience method to programmatically deselect all known filters """
        all_enabled_dict = {a_config_name:False for a_config_name in self.all_filters}
        self.try_check_check_table_row_from_state_dict(self.check_table_ctrl, all_enabled_dict)
        return self.enabled_filters
    
    def ui_update(self):
        """ called to update the ctrls depending on its properties. """
        pass
    
    
        
    @classmethod
    def get_current_check_table_row_checkedstate(cls, curr_checktable, debug_print=False):
        """Gets the list of filters for which to do filtering for from the current selections in the checkbox table UI. Returns a list of filter names that are enabled."""
        rows_state = curr_checktable.saveState()['rows']
        return {row_config_name:row_include_state for row_config_name, row_include_state in rows_state}              
    
        
    @classmethod
    def try_check_check_table_row_from_state_dict(cls, curr_checktable, known_checked_state_dict, debug_print=False):
        """ checks rows of the checktable according to the values passed in known_checked_state_dict
        TODO: currently only works for 1 column (multiple row) check tables since col_idx is hardcoded to be 0
        
        Usage:
            pipeline_filter_node.try_check_check_table_row_from_state_dict(pipeline_filter_node.check_table_ctrl, {'maze1': True, 'maze2': True, 'maze': True})
            
        """
        rows_state = curr_checktable.saveState()['rows']
        rowsMap = curr_checktable.rowsMap
        for row_config_name, row_include_state in rows_state:
            found_state = known_checked_state_dict.get(row_config_name, None)
            if found_state is not None and (found_state != row_include_state):
                # Set the state for this row:
                curr_checktable.set_value(row_idx=rowsMap[row_config_name], col_idx=0, value=found_state)
        return cls.get_current_check_table_row_checkedstate(curr_checktable)
    
    @classmethod
    def try_check_at_least_one_check_table_row(cls, curr_checktable, debug_print=False):
        if (len(curr_checktable.saveState()['rows']) > 1):
            # if we have one or more rows (columns are assumed to be fixed), set at least the first entry by default
            curr_checktable.set_value(0,0,True)
         
         
         