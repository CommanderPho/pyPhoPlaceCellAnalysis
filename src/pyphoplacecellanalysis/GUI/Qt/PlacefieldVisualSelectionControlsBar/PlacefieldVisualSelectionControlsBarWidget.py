# pyuic5 PlacefieldVisualSelectionControlsBarWidget.ui -o PlacefieldVisualSelectionControlsBarWidget.py -x
# PlacefieldVisualSelectionControlsBarWidgetBase
# pyuic5 PlacefieldVisualSelectionControlsBarWidgetBase.ui -o PlacefieldVisualSelectionControlsBarWidgetBase.py -x

import numpy as np

import pyphoplacecellanalysis.External.pyqtgraph as pg
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp

from matplotlib.colors import to_hex # required for QColor conversion to hex


from pyphoplacecellanalysis.GUI.Qt.PlacefieldVisualSelectionControlsBar.PlacefieldVisualSelectionControlsBarWidgetBase import Ui_rootForm # Generated file from .ui

# For compatibility with the panel ui version:
# from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.general_plotting_mixins import SingleNeuronPlottingExtended
from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.general_plotting_mixins import SingleNeuronPlottingExtended



class PlacefieldVisualSelectionControlsBarWidget(QtWidgets.QWidget):
    """docstring for PlacefieldVisualSelectionControlsBarWidget."""
 
    # spike_config_changed = QtCore.pyqtSignal(list, list, bool) # change_unit_spikes_included(self, cell_IDXs=None, cell_IDs=None, are_included=True)
    # tuning_curve_display_config_changed = QtCore.pyqtSignal(list, list) # on_update_tuning_curve_display_config(self, updated_config_indicies, updated_configs)
    
    # update_signal = QtCore.pyqtSignal(list, list, float, float, list, list, list, list)
    # finish_signal = QtCore.pyqtSignal(float, float)
 
    desired_full_panel_width = 1200
    desired_full_panel_height = 200
    
    enable_debug_print = False
    
    def __init__(self, *args, **kwargs):
        super(PlacefieldVisualSelectionControlsBarWidget, self).__init__(*args, **kwargs)
        self.ui = Ui_rootForm()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        
        self.desired_full_panel_width = PlacefieldVisualSelectionControlsBarWidget.desired_full_panel_width
        self.desired_full_panel_height = PlacefieldVisualSelectionControlsBarWidget.desired_full_panel_height
        
        # Final UI Refinements:
        self.initUI()
        
        # initialize member variables
        self.enable_debug_print = PlacefieldVisualSelectionControlsBarWidget.enable_debug_print
        
        
    @property
    def groupBox(self):
        return self.ui.placefieldControlsGroupbox
    
    @property
    def pf_layout(self):
        return self.ui.pf_layout
    
    @property
    def show_all_buttons_list(self):
        return (self.ui.btnShowAll_Pfs, self.ui.btnShowAll_Spikes, self.ui.btnShowAll_Both)
    @property
    def hide_all_buttons_list(self):
        return (self.ui.btnHideAll_Pfs, self.ui.btnHideAll_Spikes, self.ui.btnHideAll_Both)
        
        
    def initUI(self):
        # self.setObjectName('placefieldControlsContainer')
        
        # self.ui.batchControlPanel.hide()
         
        
        self.resize(self.desired_full_panel_width, self.desired_full_panel_height)
        
        # self.ui.placefieldControlsGroupbox = self.ui.placefieldControlsGroupbox
        # self.ui.horizontalLayout = self.ui.horizontalLayout
        
        ## Groupbox and Scrollarea:
        # groupBox.setLayout(pf_layout) # set the groupBox's layout to the one containing the widgets
        # self.setLayout(self.ui.horizontalLayout)

        # # Add a horizontal scroll area (so the placefield controls can be scrolled horizontally:
        # self.ui.scroll_area = QtWidgets.QScrollArea()
        # self.ui.scroll_area.resize(self.desired_full_panel_width, 150)
        # self.ui.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # self.ui.scroll_area.setSizeAdjustPolicy(QtGui.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        # self.ui.scroll_area.setWidget(self) # set the contents widget of the scrollarea to be the groupBox
        # # scroll_area.setWidget(groupBox) # set the contents widget of the scrollarea to be the groupBox
        # self.ui.scroll_area.setWidgetResizable(True)
        # scroll_area.setWidgetResizable(False) # This really breaks it for some reason. Oh, I guess because it's dynamically trying to resize the widget instead of creating more room.
        self.ui.scroll_area.setFixedHeight(150)
        
        # self.ui.outer_scroll_layout = QtWidgets.QVBoxLayout()
        # self.ui.outer_scroll_layout.setSpacing(0)
        # self.ui.outer_scroll_layout.setContentsMargins(0, 0, 0, 0)
        # self.ui.outer_scroll_layout.setObjectName("outerLayout")
        # self.ui.outer_scroll_layout.addWidget(self.ui.scroll_area)
        # Set the root widget's layout to the outer_scroll_layout
        # placefieldControlsContainerWidget.setLayout(outer_scroll_layout)
        # self.ui.placefieldControlsGroupbox.setLayout(self.ui.outer_scroll_layout)
        
        # Create a new layout to add the widgets to:
        self.ui.pf_layout = QtWidgets.QHBoxLayout()
        self.ui.pf_layout.setSpacing(0)
        self.ui.pf_layout.setContentsMargins(0, 0, 0, 0)
        self.ui.pf_layout.setObjectName("horizontalLayout")
        
        ## TODO: add the widgets here:
        
        
        ## Once the widgets are added to pf_layout, set the container to the layout:
        self.ui.placefieldControlsContainer.setLayout(self.ui.pf_layout)



## Start Qt event loop
if __name__ == '__main__':
    app = mkQApp("PlacefieldVisualSelectionControlsBarWidget Example")
    widget = PlacefieldVisualSelectionControlsBarWidget()
    widget.show()
    pg.exec()

