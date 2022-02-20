from typing import OrderedDict
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from pyqtgraph.widgets.CheckTable import CheckTable

__all__ = ['ExtendedCheckTable']


# ExtendedCheckTable


class ExtendedCheckTable(CheckTable):
    """CheckTable allows easily updating the rows using c.updateRows(keys), but requires columns to be set at initialization. 

    """
    def __init__(self, columns):
        CheckTable.__init__(self, columns)

    @property
    def checked_state(self):
        """The checked_state property.
            # OrderedDict([('maze1', [False, False]), ('maze2', [False, False])])
        """
        curr_state = self.saveState()
        """ {'cols': ['filter', 'compute'],
            'rows': [['maze1', False, False], ['maze2', False, False]]}
        """
        rows = curr_state['rows'] # print(f'\t {rows_state}') # [['row[0]', True, False], ['row[1]', False, False]]
        # enabled_filter_names = []
        out_dict = OrderedDict()
        for (row_config_name, *row_include_states_list) in rows:
            # ['row[0]', True, False]
            # row_config_name = a_row[0]
            # row_include_state = a_row[1]
            out_dict[row_config_name] = row_include_states_list
            # if row_include_state:
            # 	enabled_filter_names.append(row_config_name)
    
        return out_dict

    def get_column_values(self, col_name):
        # assert col_name in self.columnNames
        # col_index = self.columnNames.index(col_name)
        col_index = self.columnsMap[col_name]
        # rows = []
        row_vals = []
        for i in range(len(self.rowNames)):
            # row = [c.isChecked() for c in self.rowWidgets[i][1:]] # gets the whole row
            # row_col_value = row[col_index]
            row_col_value = self.rowWidgets[i][1+col_index].isChecked()
            # rows.append(row)
            row_vals.append(row_col_value)
        return row_vals

    # def set_column_values(self, col_name):
    #     # assert col_name in self.columnNames
    #     # col_index = self.columnNames.index(col_name)
    #     col_index = self.columnsMap[col_name]
    #     # rows = []
    #     row_vals = []
    #     for i in range(len(self.rowNames)):
    #         # row = [c.isChecked() for c in self.rowWidgets[i][1:]] # gets the whole row
    #         # row_col_value = row[col_index]
    #         row_col_value = self.rowWidgets[i][1+col_index].isChecked()
    #         # rows.append(row)
    #         row_vals.append(row_col_value)
    #     return row_vals
    
    def set_value(self, row_idx, col_idx, value):
        # rowNum = row_idx + 1
        self.rowWidgets[row_idx][1+col_idx].setChecked(value)
    
    # @checked_state.setter
    # def checked_state(self, value):
    # 	self._checked_state = value
    