# LauncherWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\MainApplicationWindows\LauncherWidget\LauncherWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import traceback
import types
import os
from typing import Optional, List, Dict
from functools import wraps

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem, QTextBrowser
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

from pyphoplacecellanalysis.External.pyqtgraph import QtCore, QtGui
from pyphoplacecellanalysis.General.Pipeline.Stages.Display import Plot, DisplayFunctionItem
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphoplacecellanalysis.GUI.Qt.Mixins.PipelineOwningMixin import PipelineOwningMixin
from neuropy.utils.result_context import IdentifyingContext
from pyphoplacecellanalysis.GUI.Qt.Widgets.IdentifyingContextSelector.IdentifyingContextSelectorWidget import IdentifyingContextSelectorWidget

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'LauncherWidget.ui')

## hardcoded icons to use for each display function. Eventually will be added to the display function metadata decorator
icon_table = {'_display_spike_rasters_pyqtplot_2D':':/Icons/Icons/SpikeRaster2DIcon.ico',
'_display_spike_rasters_pyqtplot_3D':':/Icons/Icons/SpikeRaster3DIcon.ico',
'_display_spike_rasters_vedo_3D':':/Icons/Icons/SpikeRaster3D_VedoIcon.ico',
'_display_directional_laps_overview':':/Render/Icons/Icon/SimplePlot/Laps.png',
'_display_directional_merged_pfs':':/Render/Icons/Icon/Pseudo2D.png',
'_display_1d_placefield_validations':':/Graphics/Icons/graphics/Spikes.png',
'_display_1d_placefields':':/Graphics/Icons/graphics/Spikes.png',
'_display_1d_placefield_occupancy':':/Render/Icons/Icon/Occupancy.png',
'_display_2d_placefield_occupancy':':/Render/Icons/Icon/Occupancy.png',
'_display_placemaps_pyqtplot_2D':':/Render/Icons/Icon/Heatmap.png',
'_display_rank_order_debugger':':/Icons/Icons/visualizations/rank_order_raster_debugger.ico',
'_display_directional_template_debugger':':/Icons/Icons/visualizations/template_1D_debugger.ico',
'_display_3d_interactive_tuning_curves_plotter':':/Icons/Icons/TuningMapDataExplorerIconWithLabel.ico',
'_display_3d_interactive_spike_and_behavior_browser':':/Icons/Icons/InteractivePlaceCellDataExplorerIconWithLabel.ico',
'_display_2d_placefield_result_plot_ratemaps_2D':':/Render/Icons/Icon/HeatmapUgly.png',
}

class LauncherWidget(PipelineOwningMixin, QWidget):
    """ a programmatic launcher widget that displays a tree that can be programmatically updated.
    
    
    Currently trying to use it for programmatic access to the display functions.
    
    Usage:    
        from pyphoplacecellanalysis.External.pyqtgraph import QtWidgets, QtCore, QtGui
        from pyphoplacecellanalysis.GUI.Qt.MainApplicationWindows.LauncherWidget.LauncherWidget import LauncherWidget

        widget = LauncherWidget()
        treeWidget = widget.mainTreeWidget # QTreeWidget
        widget.build_for_pipeline(curr_active_pipeline=curr_active_pipeline)
        widget.show()

        
    TODO:
        curr_fcn = curr_active_pipeline.registered_display_function_dict['_display_2d_placefield_result_plot_ratemaps_2D']
        print(str(curr_fcn.__code__.co_varnames)) # PyFunction_GetCode # ('computation_result', 'active_config', 'enable_saving_to_disk', 'kwargs', 'display_outputs', 'plot_variable_name', 'active_figure', 'active_pf_computation_params', 'session_identifier', 'fig_label', 'active_pf_2D_figures', 'should_save_to_disk')

     """
    _curr_active_pipeline_ref = None #  : Optional[Plot]
    debug_print = False

    @property
    def treeWidget(self) -> QtWidgets.QTreeWidget:
        """The treeWidget property."""
        return self.mainTreeWidget # QTreeWidget
    
    
    # @property
    # def displayContextSelectorWidget(self) -> IdentifyingContextSelectorWidget:
    #     """The treeWidget property."""
    #     # return self.ui.displayContextSelectorWidget # IdentifyingContextSelectorWidget
    #     return self._displayContextSelectorWidget # IdentifyingContextSelectorWidget

    @property
    def curr_active_pipeline(self):
        """The treeWidget property."""
        return self._curr_active_pipeline_ref
    
    @property
    def owning_pipeline(self):
        """ PipelineOwningMixin: The owning_pipeline property."""
        return self._curr_active_pipeline_ref


    @property
    def docPanelTextBrowser(self) -> QTextBrowser:
        return self.ui.textBrowser
    
    # @property
    # def display_function_items(self) -> Dict[str,DisplayFunctionItem]:
    #     return {a_fn_name:DisplayFunctionItem.init_from_fn_object(a_fn) for a_fn_name, a_fn in self._pipeline_reference.registered_display_function_dict.items()}

    @property
    def selected_context(self) -> Optional[IdentifyingContext]:
        """The selected_context property."""
        if not hasattr(self, 'displayContextSelectorWidget'):
            return None
        if self.displayContextSelectorWidget is None:
            return None
        return self.displayContextSelectorWidget.current_selected_context

    @property
    def has_selected_context(self) -> bool:
        """The selected_context property."""
        return (self.selected_context is not None)


    # __init__ fcn _______________________________________________________________________________________________________ #
    def __init__(self, debug_print=False, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file
        self._curr_active_pipeline_ref = None
        self.debug_print = debug_print
        # self._displayContextSelectorWidget = None
        # self.ui.displayContextSelectorWidget = None

        self.initUI()
        self.show() # Show the GUI

    def initUI(self):
        # Connect the itemDoubleClicked signal to the on_tree_item_double_clicked slot
        self.treeWidget.itemDoubleClicked.connect(self.on_tree_item_double_clicked)
        # Enable mouse tracking to receive itemEntered events
        self.treeWidget.setMouseTracking(True)

        # Signal: itemEntered - fired when mouse hovers over an item
        self.treeWidget.itemEntered.connect(self.on_tree_item_hovered)

        self.displayContextSelectorWidget.sigContextChanged.connect(self.on_selected_context_changed)

        # self.displayContextSelectorWidget._owning_pipeline = self.curr_active_pipeline
        # self.ui.displayContextSelectorWidget

        # self.ui.displayContextSelectorWidget = None
        # self.ui.displayContextSelectorWidgetContainer.layout.add


    def get_display_function_items(self) -> Dict[str,DisplayFunctionItem]:
        assert self._curr_active_pipeline_ref is not None
        return {a_fn_name:DisplayFunctionItem.init_from_fn_object(a_fn, icon_path=icon_table.get(a_fn_name, None)) for a_fn_name, a_fn in self.curr_active_pipeline.registered_display_function_dict.items()}

    def get_display_function_item(self, a_fn_name: str) -> Optional[DisplayFunctionItem]:
        return self.get_display_function_items().get(a_fn_name, None)
    

    def _perform_get_display_function_code(self, a_fcn_name: str):
        """ gets the actual display function to be executed"""
        a_disp_fn_item = self.get_display_function_item(a_fn_name=a_fcn_name) #display_function_items[item_data]
        if self.debug_print:
            print(f'\ta_disp_fn_item: {a_disp_fn_item}')
        assert a_disp_fn_item is not None, f'\t WARN: a_disp_fn_item is None for key: "{a_fcn_name}"'
        # a_fn_handle = self.curr_active_pipeline.plot.__getattr__(item_data) # find matching item in the pipleine's .plot attributes
        # a_fn_handle = self.curr_active_pipeline.plot.__getattr__(a_disp_fn_item.fn_callable)
        a_fn_handle = self.curr_active_pipeline.plot.__getattr__(a_disp_fn_item.name)
        return a_fn_handle
    


    def _perform_execute_display_function(self, a_fcn_name: str, *args, **kwargs):
        """ gets the display function to execute and executes it """
        a_fn_handle = self._perform_get_display_function_code(a_fcn_name=a_fcn_name)
        assert a_fn_handle is not None
        # args = []
        # kwargs = {}
        a_disp_fn_item = self.get_display_function_item(a_fn_name=a_fcn_name)
        assert a_disp_fn_item is not None, f"a_disp_fn_item is None! for a_fn_name='{a_fcn_name}'"
        if a_disp_fn_item.is_global:
            return self.curr_active_pipeline.display(display_function=a_fcn_name, active_session_configuration_context=None, *args, **kwargs)
        else:
            # non-global, needs a context:
            current_selected_context = self.displayContextSelectorWidget.current_selected_context
            if current_selected_context is not None:
                # args = list(args) ## convert to list if a tuple
                # args.insert(0, current_selected_context)
                return self.curr_active_pipeline.display(display_function=a_fcn_name, active_session_configuration_context=current_selected_context, *args, **kwargs)
            else:
                return None

        # return a_fn_handle(*args, **kwargs)
    




    # Define a function to be executed when a tree widget item is double-clicked
    # @QtCore.Slot(object, int)
    @pyqtExceptionPrintingSlot(object)
    def on_tree_item_double_clicked(self, item, column):
        if self.debug_print:
            print(f"Item double-clicked: {item}, column: {column}\n\t", item.text(column))
        # print(f'\titem.data: {item.data}')
        # raise NotImplementedError
        # item_data = item.data(column, 0) # ItemDataRole 
        item_data = item.data(column, 0) # ItemDataRole 
        if self.debug_print:
            print(f'\titem_data: {item_data}')
        assert item_data is not None, f"item_Data is None"
        assert isinstance(item_data, str), f"item_data is not a string! type(item_data): {type(item_data)}, item_data: {item_data}"
        a_fcn_name: str = item_data
        return self._perform_execute_display_function(a_fcn_name=a_fcn_name)


    @pyqtExceptionPrintingSlot(object, int)
    def on_tree_item_hovered(self, item, column):
        if self.debug_print:
            print(f"Item hovered: {item}, column: {column}\n\t", item.text(column))
        # print(f'\titem.data: {item.data}')
        # raise NotImplementedError
        # item_data = item.data(column, 0) # ItemDataRole 
        item_data = item.data(column, 0) # ItemDataRole 
        if self.debug_print:
            print(f'\titem_data: {item_data}')
        assert item_data is not None
        assert isinstance(item_data, str)
        a_fcn_name: str = item_data

        a_disp_fn_item = self.get_display_function_item(a_fn_name=a_fcn_name)
        if self.debug_print:
            print(f'\ta_disp_fn_item: {a_disp_fn_item}')
        # tooltip_text = item.toolTip(column)
        if a_disp_fn_item is not None:
            # self.docPanelTextBrowser.setText(a_disp_fn_item.longform_description)
            self.docPanelTextBrowser.setHtml(a_disp_fn_item.longform_description_formatted_html)

        else:
            print(f'\t WARN: a_disp_fn_item is None for key: "{a_fcn_name}"')
        

    @pyqtExceptionPrintingSlot(object, object)
    def on_selected_context_changed(self, new_key: str, new_context: IdentifyingContext):
        print(f'on_selected_context_changed(new_key: {new_key}, new_context: {new_context})')
        self.update_local_display_items_are_enabled()
        

    def update_local_display_items_are_enabled(self):
        """ sets the local display items as disabled if no context is selected. 
        """
        it = QtWidgets.QTreeWidgetItemIterator(self.treeWidget)
        while it.value():
            item = it.value()
            is_global_fn = item.data(1, QtCore.Qt.UserRole)
            if (is_global_fn is not None) and (is_global_fn == False):
                item.setDisabled(not self.has_selected_context)
            it += 1  # Move to the next item in the iterator
            

    def build_for_pipeline(self, curr_active_pipeline):
        self._curr_active_pipeline_ref = curr_active_pipeline
        curr_active_pipeline.reload_default_display_functions()
        
        self.displayContextSelectorWidget.build_for_pipeline(curr_active_pipeline)

        self.ui.displayFunctionTreeItem_global_fns = None
        self.ui.displayFunctionTreeItem_non_global_fns = None


        # Add root item
        displayFunctionTreeItem = QtWidgets.QTreeWidgetItem(["Display Functions"])
        # si.gitem = 
        # self.treeWidget.addTopLevelItem(displayFunctionTreeItem)
        display_function_items = self.get_display_function_items() # {a_fn_name:DisplayFunctionItem.init_from_fn_object(a_fn) for a_fn_name, a_fn in curr_active_pipeline.registered_display_function_dict.items()}
        # display_function_items

        # displayFunctionTreeItem_prefab_fns = QtWidgets.QTreeWidgetItem(["Prefab Functions"])
        self.ui.displayFunctionTreeItem_global_fns = QtWidgets.QTreeWidgetItem(["Global Functions"])
        self.ui.displayFunctionTreeItem_non_global_fns = QtWidgets.QTreeWidgetItem(["Non-Global Functions"])

        # self.treeWidget.addTopLevelItem(displayFunctionTreeItem_prefab_fns)
        self.treeWidget.addTopLevelItem(self.ui.displayFunctionTreeItem_global_fns)
        self.treeWidget.addTopLevelItem(self.ui.displayFunctionTreeItem_non_global_fns)

        # for a_fcn_name, a_fcn in curr_active_pipeline.registered_display_function_dict.items():
        for a_fcn_name, a_disp_fn_item in display_function_items.items():
            # extract the info from the function:
            # if hasattr(a_fcn, 'short_name') and a_fcn.short_name is not None:
            #     active_name = a_fcn.short_name or a_fcn_name
            # else:
            #     active_name = a_fcn_name

            # active_name: str = a_disp_fn_item.name


            should_use_nice_display_names: bool = False # currently broken

            if should_use_nice_display_names:
                active_name: str = a_disp_fn_item.best_display_name
            else:
                active_name: str = a_disp_fn_item.name # function name


            if self.debug_print:
                print(f'adding {active_name}')
            childDisplayFunctionTreeItem = QtWidgets.QTreeWidgetItem([active_name])
            # childDisplayFunctionTreeItem.setText(0, active_name)
            # Set the tooltip for the item
            # active_tooltip_text = curr_active_pipeline.registered_display_function_docs_dict.get(a_fcn_name, "No tooltip")
                        
            active_tooltip_text = (a_disp_fn_item.docs or "No tooltip")
            childDisplayFunctionTreeItem.setToolTip(0, active_tooltip_text)
            childDisplayFunctionTreeItem.setData(0, QtCore.Qt.UserRole, a_fcn_name) # "Child 1 custom data"
            # childDisplayFunctionTreeItem.setData(0, QtCore.Qt.UserRole, a_disp_fn_item) # "Child 1 custom data"
            childDisplayFunctionTreeItem.setData(1, QtCore.Qt.UserRole, a_disp_fn_item.is_global)

            if (a_disp_fn_item.icon_path is not None) and (len(a_disp_fn_item.icon_path) > 0):
                # has valid iconpath
                print(f'setting icon to "{a_disp_fn_item.icon_path}"...')
                # childDisplayFunctionTreeItem.setIcon(0, QtGui.QIcon("child_1_icon.png"))
                childDisplayFunctionTreeItem.setIcon(0, QtGui.QIcon(a_disp_fn_item.icon_path))

            # childDisplayFunctionTreeItem
            # displayFunctionTreeItem.addChild(childDisplayFunctionTreeItem)
            # self.treeWidget.addTopLevelItem(childDisplayFunctionTreeItem) # add top level

            if a_disp_fn_item.is_global:
                self.ui.displayFunctionTreeItem_global_fns.addChild(childDisplayFunctionTreeItem)
            else:
                # non-global
                self.ui.displayFunctionTreeItem_non_global_fns.addChild(childDisplayFunctionTreeItem)

        self.update_local_display_items_are_enabled()
        
        # self.treeWidget.addTopLevelItem(displayFunctionTreeItem)
        self.ui.displayFunctionTreeItem_global_fns.setExpanded(True)
        self.ui.displayFunctionTreeItem_non_global_fns.setExpanded(True)
        


## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = LauncherWidget()
    widget.show()
    sys.exit(app.exec_())
