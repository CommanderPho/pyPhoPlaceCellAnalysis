from typing import OrderedDict
import numpy as np
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.widgets.CheckTable import CheckTable

__all__ = ['ExtendedCheckTable']


# ExtendedCheckTable


class ExtendedCheckTable(CheckTable):
    """CheckTable allows easily updating the rows using c.updateRows(keys), but requires columns to be set at initialization. 

    """
    def __init__(self, columns):
        CheckTable.__init__(self, columns)

    @property
    def rowsMap(self):
        """The mapping of the row names to their index, analagous to columnsMap
            # OrderedDict([('maze1', 0), ('maze2', 1), ('maze', 2)])
        """
        return OrderedDict({a_row_name:row_idx for row_idx, a_row_name in enumerate(self.rowNames)}) # {'maze1': 0, 'maze2': 1, 'maze': 2}
        
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
        out_dict = OrderedDict()
        for (row_config_name, *row_include_states_list) in rows:
            out_dict[row_config_name] = row_include_states_list
    
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
    