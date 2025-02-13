# LauncherWidget.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\MainApplicationWindows\LauncherWidget\LauncherWidget.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import subprocess
import traceback
import types
import os
from typing import Optional, List, Dict
from functools import wraps

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem, QTextBrowser
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QTreeWidget, QTreeWidgetItem, QWidget, QHeaderView, QMenu
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon, QContextMenuEvent
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir, QUrl, QTimer

from pyphocorehelpers.programming_helpers import copy_to_clipboard
from pyphoplacecellanalysis.External.pyqtgraph import QtCore, QtGui
from pyphoplacecellanalysis.General.Pipeline.Stages.Display import Plot, DisplayFunctionItem
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot
from pyphoplacecellanalysis.GUI.Qt.Mixins.PipelineOwningMixin import PipelineOwningMixin
from neuropy.utils.result_context import IdentifyingContext
from pyphoplacecellanalysis.GUI.Qt.Widgets.IdentifyingContextSelector.IdentifyingContextSelectorWidget import IdentifyingContextSelectorWidget
# from pyphocorehelpers.programming_helpers import SourceCodeParsing
from pyphocorehelpers.Filesystem.path_helpers import open_vscode_link
from pyphocorehelpers.gui.Qt.widget_positioning_helpers import WidgetPositioningHelpers


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
# '_display_trial_to_trial_reliability':':/Render/Icons/graphics/TrialByTrialReliabilityImageArray.png',
# '_display_directional_merged_pf_decoded_epochs':':/Render/Icons/graphics/yellow_blue_plot_icon.png',
'_display_trial_to_trial_reliability':':/Graphics/Icons/graphics/TrialByTrialReliabilityImageArray.png',
'_display_directional_merged_pf_decoded_epochs':':/Graphics/Icons/graphics/yellow_blue_plot_icon.png',
'_display_directional_track_template_pf1Ds':':/Graphics/Icons/graphics/directional_track_template_pf1Ds.png',
# '_display_directional_track_template_pf1Ds':':/Render/Icons/Icon/SimplePlot/Laps.png',
'_display_long_short_laps':':/Graphics/Icons/graphics/long_short_laps.png',
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

        
        
    self.ui.btnCopySelectedDisplayFunctionCode
    
    
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
    def __init__(self, debug_print=False, should_use_nice_display_names: bool = True, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file
        self._curr_active_pipeline_ref = None
        self.debug_print = debug_print
        self.should_use_nice_display_names = should_use_nice_display_names
        if should_use_nice_display_names:
            self.best_display_name_to_function_name_map = {}
        else:
            self.best_display_name_to_function_name_map = None

        # self._displayContextSelectorWidget = None
        # self.ui.displayContextSelectorWidget = None

        self.initUI()
        self.show() # Show the GUI


    def initUI(self):
        # Connect the itemDoubleClicked signal to the on_tree_item_double_clicked slot
        self.treeWidget.itemDoubleClicked.connect(self.on_tree_item_double_clicked)
        self.treeWidget.itemClicked.connect(self.on_tree_item_single_clicked)
        # Enable mouse tracking to receive itemEntered events
        self.treeWidget.setMouseTracking(True)

        # Signal: itemEntered - fired when mouse hovers over an item
        self.treeWidget.itemEntered.connect(self.on_tree_item_hovered)
        # self.treeWidget.selectionChanged
        # self.treeWidget.itemSelectionChanged.connect(self.on_tree_item_selection_changed) # not working
        
        # self.treeWidget.selectedItems
        # self.treeWidget.selection
        self.displayContextSelectorWidget.sigContextChanged.connect(self.on_selected_context_changed)

        # self.displayContextSelectorWidget._owning_pipeline = self.curr_active_pipeline
        # self.ui.displayContextSelectorWidget

        # self.ui.displayContextSelectorWidget = None
        # self.ui.displayContextSelectorWidgetContainer.layout.add
        
        # ## Enable custom context menu:
        # self.treeWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        # self.setContextMenuPolicy(Qt.CustomContextMenu)
        # self.treeWidget.customContextMenuRequested.connect(self.contextMenuEvent)
        # self.customContextMenuRequested.connect(self.contextMenuEvent)
        # Set context menu policy and connect the signal
        self.treeWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.treeWidget.customContextMenuRequested.connect(self.show_custom_context_menu)

        ## setup link handling:    
        self.docPanelTextBrowser.setOpenLinks(False) 
        self.docPanelTextBrowser.setOpenExternalLinks(False) # Disable automatic opening of external links
        self.docPanelTextBrowser.anchorClicked.connect(self.handle_link_click) # Connect the link click handler
    
        self.ui.btnCopySelectedDisplayFunctionCode.clicked.connect(self.handle_copy_code_button_clicked)

    # ==================================================================================================================== #
    # Item Access Methods                                                                                                  #
    # ==================================================================================================================== #
    def get_display_function_items(self) -> Dict[str,DisplayFunctionItem]:
        assert self._curr_active_pipeline_ref is not None
        return {a_fn_name:DisplayFunctionItem.init_from_fn_object(a_fn, icon_path=icon_table.get(a_fn_name, None)) for a_fn_name, a_fn in self.curr_active_pipeline.registered_display_function_dict.items()}

    def get_display_function_item(self, a_fn_name: str) -> Optional[DisplayFunctionItem]:
        if self.should_use_nice_display_names:
            an_active_fn_name: str = self.best_display_name_to_function_name_map.get(a_fn_name, a_fn_name)
            return self.get_display_function_items().get(an_active_fn_name, None)
        else:
            return self.get_display_function_items().get(a_fn_name, None)
    
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
            

    def _rebuild_tree(self):
        """ rebuilds the entire tree using the items provided by `self.get_display_function_items()`
        
        """
        if self.should_use_nice_display_names:
            self.best_display_name_to_function_name_map = {}

        self.ui.displayFunctionTreeItem_global_fns = None
        self.ui.displayFunctionTreeItem_non_global_fns = None
        
         # currently broken

        # Add root item
        displayFunctionTreeItem = QtWidgets.QTreeWidgetItem(["Display Functions"])
        # si.gitem = 
        # self.treeWidget.addTopLevelItem(displayFunctionTreeItem)
        display_function_items = self.get_display_function_items() # {a_fn_name:DisplayFunctionItem.init_from_fn_object(a_fn) for a_fn_name, a_fn in curr_active_pipeline.registered_display_function_dict.items()}
        # display_function_items

        # displayFunctionTreeItem_prefab_fns = QtWidgets.QTreeWidgetItem(["Prefab Functions"])
        self.ui.displayFunctionTreeItem_global_fns = QtWidgets.QTreeWidgetItem(["Global Functions"])
        self.ui.displayFunctionTreeItem_non_global_fns = QtWidgets.QTreeWidgetItem(["Non-Global Functions"])

        self.treeWidget.clear() ## clear all existing items
        
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
            if self.should_use_nice_display_names:
                active_name: str = a_disp_fn_item.best_display_name
                self.best_display_name_to_function_name_map[a_disp_fn_item.best_display_name] = a_disp_fn_item.name # function name
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
            childDisplayFunctionTreeItem.setData(1, QtCore.Qt.UserRole, a_disp_fn_item.is_global)
            childDisplayFunctionTreeItem.setData(2, QtCore.Qt.UserRole, a_disp_fn_item.name)

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
        


    def build_for_pipeline(self, curr_active_pipeline):
        self._curr_active_pipeline_ref = curr_active_pipeline
        curr_active_pipeline.reload_default_display_functions()
        ## update window title/etc
        session_id_str: str = self._curr_active_pipeline_ref.get_complete_session_identifier_string()
        self.setWindowTitle(f'Spike3D Launcher: {session_id_str}')
        self.displayContextSelectorWidget.build_for_pipeline(curr_active_pipeline) ## update the context list
        self._rebuild_tree() ## call self._rebuild_tree()

        
        
    # Handle link click
    def handle_link_click(self, url: QUrl):
        """ handle link (anchor) clicked events. Calls `open_vscode_link(...)` to open the link in vscode. """
        a_link_str: str = url.toString()
        print(f"Link clicked: {a_link_str}")
        open_vscode_link(a_vscode_link_str=a_link_str, open_in_background=True)
        # Refocus the widget after opening the link
        # self.activateWindow()  # Ensure the window is activated
        # self.raise_()  # Bring the window to the top
        # Use QTimer to delay refocusing the widget
        QTimer.singleShot(500, self.refocus_widget)
        
    def refocus_widget(self):
        """ Brings the widget back into focus after a delay. """
        print(f'.refocus_widget()')
        WidgetPositioningHelpers.qt_win_to_foreground(self)
        
        # self.activateWindow()  # Ensure the window is activated
        # self.raise_()  # Bring the window to the top
    

    def update_fn_documentation_panel(self, a_fcn_name: str):
        a_disp_fn_item = self.get_display_function_item(a_fn_name=a_fcn_name)
        if self.debug_print:
            print(f'\ta_disp_fn_item: {a_disp_fn_item}')
        # tooltip_text = item.toolTip(column)
        if a_disp_fn_item is not None:
            # self.docPanelTextBrowser.setText(a_disp_fn_item.longform_description)
            # QTextBrowser
            # self.docPanelTextBrowser.setOpenLinks(False) 
            # self.docPanelTextBrowser.setOpenExternalLinks(False) # Disable automatic opening of external links
            # self.docPanelTextBrowser.anchorClicked.connect(self.handle_link_click) # Connect the link click handler
            self.docPanelTextBrowser.setHtml(a_disp_fn_item.longform_description_formatted_html)
        else:
            print(f'\t WARN: a_disp_fn_item is None for key: "{a_fcn_name}"')
            


    def handle_copy_code_button_clicked(self, *args, **kwargs):
        print(f'handle_copy_code_button_clicked(...)')
        selected_items = self.treeWidget.selectedItems() # List[QTreeWidgetItem]
        all_out_code: List[str] = []
        for item in selected_items:
            # print(item.text(0), "-", item.text(1))  # Print data from column 0 and column 1
            item_data = item.data(0, 0) # ItemDataRole 
            # print(f'\titem_data: {item_data}')
            assert item_data is not None
            assert isinstance(item_data, str)
            a_fcn_name: str = item_data
            all_out_code.append(self.build_display_function_run_code(a_fcn_name=a_fcn_name)) 
        final_out_code: str = '\n'.join(all_out_code)
        print(final_out_code)
        ## copy to clipboard
        copy_to_clipboard(code_str=final_out_code, message_print=True)
        


    # ==================================================================================================================== #
    # Events                                                                                                               #
    # ==================================================================================================================== #
    @pyqtExceptionPrintingSlot(object)
    def on_tree_item_single_clicked(self, item, column):
        if self.debug_print:
            print(f"Item single-clicked: {item}, column: {column}\n\t", item.text(column))
        item_data = item.data(column, 0) # ItemDataRole 
        # item_data = item.data(column, 2)
        if self.debug_print:
            print(f'\titem_data: {item_data}')
        assert item_data is not None, f"item_Data is None"
        assert isinstance(item_data, str), f"item_data is not a string! type(item_data): {type(item_data)}, item_data: {item_data}"
        a_fcn_name: str = item_data
        self.update_fn_documentation_panel(a_fcn_name=a_fcn_name)


    # Define a function to be executed when a tree widget item is double-clicked
    @pyqtExceptionPrintingSlot(object)
    def on_tree_item_double_clicked(self, item, column):
        if self.debug_print:
            print(f"Item double-clicked: {item}, column: {column}\n\t", item.text(column))
        # print(f'\titem.data: {item.data}')
        # raise NotImplementedError
        item_data = item.data(column, 0) # ItemDataRole 
        # item_data = item.data(column, 2)
        if self.debug_print:
            print(f'\titem_data: {item_data}')
        assert item_data is not None, f"item_Data is None"
        assert isinstance(item_data, str), f"item_data is not a string! type(item_data): {type(item_data)}, item_data: {item_data}"
        a_fcn_name: str = item_data
        return self._perform_execute_display_function(a_fcn_name=a_fcn_name)


    @pyqtExceptionPrintingSlot(object, int)
    def on_tree_item_hovered(self, item, column):
        """ called when hovering a tree item """
        if self.debug_print:
            print(f"Item hovered: {item}, column: {column}\n\t", item.text(column))
        item_data = item.data(column, 0) # ItemDataRole 
        # item_data = item.data(column, 2)
        if self.debug_print:
            print(f'\titem_data: {item_data}')
        assert item_data is not None
        assert isinstance(item_data, str)
        a_fcn_name: str = item_data
        # self.update_fn_documentation_panel(a_fcn_name=a_fcn_name) ## do no do for selection mode
        

    @pyqtExceptionPrintingSlot(object, int)
    def on_tree_item_selection_changed(self, item, column):
        """ called when selection changes for the tree """
        if self.debug_print:
            print(f"on_tree_item_selection_changed: {item}, column: {column}\n\t", item.text(column))
        item_data = item.data(column, 0) # ItemDataRole 
        # item_data = item.data(column, 2)
        if self.debug_print:
            print(f'\titem_data: {item_data}')
        assert item_data is not None
        assert isinstance(item_data, str)
        a_fcn_name: str = item_data
        self.update_fn_documentation_panel(a_fcn_name=a_fcn_name)
        


    @pyqtExceptionPrintingSlot(object, object)
    def on_selected_context_changed(self, new_key: str, new_context: IdentifyingContext):
        print(f'on_selected_context_changed(new_key: {new_key}, new_context: {new_context})')
        self.update_local_display_items_are_enabled()
        

    ## Custom context menu:
    def show_custom_context_menu(self, position: QPoint):
        """ used for tree item's context menus """
        # Get the item at the clicked position
        print(f"show_custom_context_menu(): position: {position}")
        # Get the item at the clicked position
        item = self.treeWidget.itemAt(position)
        if item is not None:
            item_data = item.data(0, 0) # ItemDataRole 
            assert item_data is not None
            assert isinstance(item_data, str)
            a_fcn_name: str = item_data
            a_disp_fn_item = self.get_display_function_item(a_fn_name=a_fcn_name)
            assert a_disp_fn_item is not None, f"a_disp_fn_item is None for a_fcn_name: {a_fcn_name}"
            menu = QtWidgets.QMenu(self)
            # action_edit = menu.addAction("Edit")
            # action_delete = menu.addAction("Delete")
            
            # action_edit.triggered.connect(lambda: self.edit_item(item))
            # action_delete.triggered.connect(lambda: self.delete_item(item))
        
            action_run_item = menu.addAction(f"Run {item.text(0)}")
            action_show_code_jumplink = menu.addAction("Show Code in Editor...")
            
            # action1.triggered.connect(lambda: print(f"Run Action triggered for {item.text(0)}"))
            action_run_item.triggered.connect(lambda: self._perform_execute_display_function(a_fcn_name=a_fcn_name))
            # action_show_code_jumplink.triggered.connect(lambda: print("Show Code in Editor action triggered"))
            action_show_code_jumplink.triggered.connect(lambda: self._perform_get_display_function_code_jumplink(a_fcn_name=a_fcn_name))
            
            menu.exec_(self.treeWidget.viewport().mapToGlobal(position))
            



    # @pyqtExceptionPrintingSlot(object)
    # def contextMenuEvent(self, event: QContextMenuEvent):
    #     print(f'contextMenuEvent(event: {event})')
    #     # Create the context menu
    #     menu = QMenu(self)
    #     # Add actions
    #     action1 = menu.addAction("Action 1")
    #     action2 = menu.addAction("Action 2")
    #     # Connect actions
    #     action1.triggered.connect(self.action1_triggered)
    #     action2.triggered.connect(self.action2_triggered)
    #     # Show the menu at the global position
    #     menu.exec_(event.globalPos())


    def action1_triggered(self):
        print("Action 1 selected")

    def action2_triggered(self):
        print("Action 2 selected")


        
    # ==================================================================================================================== #
    # Display Item Actions                                                                                                 #
    # ==================================================================================================================== #
    


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
    

    def _perform_get_display_function_code_jumplink(self, a_fcn_name: str):
        """ gets the display function vscode jumplink """
        a_disp_fn_item = self.get_display_function_item(a_fn_name=a_fcn_name)
        assert a_disp_fn_item is not None
        vscode_jump_link: str = a_disp_fn_item.vscode_jump_link
        assert vscode_jump_link is not None, f"vscode_jump_link is None for a_disp_fn_item: {a_disp_fn_item}!"
        # a_fn_handle = self._perform_get_display_function_code(a_fcn_name=a_fcn_name)
        # a_fn_handle = a_disp_fn_item.fn_callable
        # assert a_fn_handle is not None
        # vscode_jump_link: str = SourceCodeParsing.build_vscode_jump_link(a_fcn_handle=a_fn_handle) # <function Plot.__getattr__.<locals>.display_wrapper at 0x000002199E176EE0>
        print(f'vscode_jump_link: {vscode_jump_link}')
        return vscode_jump_link
    
    def _perform_execute_display_function(self, a_fcn_name: str, *args, **kwargs):
        """ gets the display function to execute and executes it """
        a_fn_handle = self._perform_get_display_function_code(a_fcn_name=a_fcn_name)
        assert a_fn_handle is not None
        # args = []
        # kwargs = {}
        a_disp_fn_item = self.get_display_function_item(a_fn_name=a_fcn_name)
        assert a_disp_fn_item is not None, f"a_disp_fn_item is None! for a_fn_name='{a_fcn_name}'"
        if a_disp_fn_item.is_global:
            return self.curr_active_pipeline.display(display_function=a_disp_fn_item.name, active_session_configuration_context=None, *args, **kwargs)
        else:
            # non-global, needs a context:
            current_selected_context = self.displayContextSelectorWidget.current_selected_context
            if current_selected_context is not None:
                # args = list(args) ## convert to list if a tuple
                # args.insert(0, current_selected_context)
                return self.curr_active_pipeline.display(display_function=a_disp_fn_item.name, active_session_configuration_context=current_selected_context, *args, **kwargs)
            else:
                return None

        # return a_fn_handle(*args, **kwargs)
        


    def build_display_function_run_code(self, a_fcn_name: str, include_initial_define_line=True) -> str:
        """ btnCopySelectedDisplayFunctionCode
        """
        code_out: str = ""
        if include_initial_define_line:
            code_out = f"{code_out}_out = dict()\n"

        a_fn_handle = self._perform_get_display_function_code(a_fcn_name=a_fcn_name)
        assert a_fn_handle is not None
        # args = []
        # kwargs = {}
        a_disp_fn_item: DisplayFunctionItem = self.get_display_function_item(a_fn_name=a_fcn_name)
        assert a_disp_fn_item is not None, f"a_disp_fn_item is None! for a_fn_name='{a_fcn_name}'"
        if a_disp_fn_item.is_global:
            code_out = f"{code_out}_out['{a_disp_fn_item.name}'] = curr_active_pipeline.display(display_function='{a_disp_fn_item.name}', active_session_configuration_context=None) # {a_disp_fn_item.name}\n" # , *{args}, **{kwargs}
        else:
            # non-global, needs a context:
            current_selected_context = self.displayContextSelectorWidget.current_selected_context
            if current_selected_context is not None:
                # args = list(args) ## convert to list if a tuple
                # args.insert(0, current_selected_context)
                code_out = f"{code_out}_out['{a_disp_fn_item.name}'] = curr_active_pipeline.display(display_function='{a_disp_fn_item.name}', active_session_configuration_context={current_selected_context.get_initialization_code_string()}) # {a_disp_fn_item.name}\n" # , *{args}, **{kwargs}
            else:
                return None
            
        return code_out
        # return f""" 
        #     _out = dict()
        #     _out['{a_fcn_name}'] = curr_active_pipeline.display('{a_fcn_name}') # {a_fcn_name}
        # """






## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    widget = LauncherWidget()
    widget.show()
    sys.exit(app.exec_())
