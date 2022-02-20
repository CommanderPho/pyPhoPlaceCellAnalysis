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
		"""The checked_state property."""
		curr_state = self.saveState()
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

	# @checked_state.setter
	# def checked_state(self, value):
	# 	self._checked_state = value
	