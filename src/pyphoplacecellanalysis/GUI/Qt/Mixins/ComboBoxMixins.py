import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from nptyping import NDArray

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
    
    

class ComboBoxCtrlOwningMixin:
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
    
    
from PyQt5 import QtWidgets

## Ideal Interface:
def build_combo_box(label: str, options: List[str], currentIndexChangedCallback=None) -> QtWidgets.QComboBox:
    """ Instantiates a working PyQt5 QComboBox inline with a list of options, a label that's rendered to the left of the comobo box to indicate what the options mean.
    
        from pyphoplacecellanalysis.GUI.Qt.Mixins.ComboBoxMixins import build_combo_box
    
        new_combo_box = build_combo_box(label='Format', options=['Numpy','Png', 'Svg'])
        global_thin_button_bar_widget.horizontalLayout.addWidget(new_combo_box) # add the pagination control widget
        # new_combo_box.currentIndexChanged.connect(self.on_selected_context_index_changed)
        
    """
    layout = QtWidgets.QHBoxLayout()
    layout.setContentsMargins(0, -1, -1, -1)
    layout.setSpacing(2)
    
    label_widget = QtWidgets.QLabel(label)
    combo_box = QtWidgets.QComboBox()
    combo_box.addItems(options)
    
    if currentIndexChangedCallback:
        combo_box.currentIndexChanged.connect(currentIndexChangedCallback)
        
    layout.addWidget(label_widget)
    layout.addWidget(combo_box)
    
    container_widget = QtWidgets.QWidget()
    container_widget.setContentsMargins(0, 0, 0, 0)
    container_widget.setLayout(layout)
    
    combo_box.setObjectName(f'cmb{label}')
    
    return container_widget



