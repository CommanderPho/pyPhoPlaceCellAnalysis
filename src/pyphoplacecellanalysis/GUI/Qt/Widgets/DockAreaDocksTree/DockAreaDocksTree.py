# DockAreaDocksTree.py
# Generated from c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\DockAreaDocksTree\DockAreaDocksTree.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import numpy as np
import pandas as pd
from functools import partial # used in `CustomHeaderTableView`

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

import pyphoplacecellanalysis.External.pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHeaderView, QPushButton, QWidget, QSizePolicy, QLabel
from PyQt5.QtCore import QAbstractTableModel, Qt

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QTreeWidgetItemIterator
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
from pyphoplacecellanalysis.Resources import GuiResources, ActionIcons
from pyphoplacecellanalysis.Resources.icon_helpers import try_get_icon
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig, DockDisplayColors
        

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'DockAreaDocksTree.ui')

import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QFormLayout, QCheckBox, QLineEdit, QComboBox, 
                            QGroupBox, QLabel)
from PyQt5.QtCore import Qt

# Import the DockDisplayConfig class
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import DockDisplayConfig
from pyphoplacecellanalysis.GUI.Qt.Widgets.DockAreaDocksTree.DockConfigEditorWidget import DockConfigEditor

  


@metadata_attributes(short_name=None, tags=['build_dock_area_managing_tree_widget'], input_requires=[], output_provides=[], uses=[], used_by=['build_dock_area_managing_tree_widget'], creation_date='2025-01-28 07:32', related_items=['build_dock_area_managing_tree_widget'])
class DockAreaDocksTree(QWidget):
    """ Display an interactive pyqt5 widget for manager Docks
    
    from pyphoplacecellanalysis.GUI.Qt.Widgets.DockAreaDocksTree.DockAreaDocksTree import DockAreaDocksTree
    
    
    dynamic_docked_widget_container = active_2d_plot.ui.dynamic_docked_widget_container # NestedDockAreaWidget
    dock_tree_list, group_meta_item_dict = dynamic_docked_widget_container.get_dockGroup_dock_tree_dict()
    a_dock_area_docks_tree_widget: DockAreaDocksTree = DockAreaDocksTree()
    a_dock_area_docks_tree_widget.rebuild_dock_tree_items(dock_tree_list=dock_tree_list)
    a_dock_area_docks_tree_widget.show()

    """
    sigDockConfigChanged = pg.QtCore.Signal(object)

    # @property
    # def mainTreeWidget(self): #  -> pg.QtWidgets.QTreeWidget
    #     """The mainTreeWidget property."""
    #     return self.ui.mainTreeWidget

    # @pg.QtCore.Property(pg.QtWidgets.QTreeWidget) # Note that this ia *pyqt*Property, meaning it's available to pyqt
    # @pg.QtCore.pyqtProperty(pg.QtWidgets.QTreeWidget) # Note that this ia *pyqt*Property, meaning it's available to pyqt
    # def mainTreeWidget(self): #  -> 
    #     """The mainTreeWidget property."""
    #     return self.ui.mainTreeWidget

    def get_main_tree(self) -> pg.QtWidgets.QTreeWidget:
        return self.ui.mainTreeWidget


    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file

        self.initUI()
        self.show() # Show the GUI

    def initUI(self):
        # Connect context menu signal
        self.ui.mainTreeWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.mainTreeWidget.customContextMenuRequested.connect(self.show_context_menu)    

    def rebuild_dock_tree_items(self, dock_tree_list):
        """ rebuilds the tree items from a list 
        """
        main_docks_tree = self.get_main_tree()
        main_docks_tree.clear()

        group_icon_path = ':/Icons/Icons/Icon/application-sub.png'
        dock_item_icon_path = ':/Icons/Icons/Icon/applications-blue.png'

        TreeItem_ungrouped_root = pg.QtWidgets.QTreeWidgetItem(["Ungrouped Docks"])
        an_icon = try_get_icon(icon_path=group_icon_path)
        if an_icon is not None:
            TreeItem_ungrouped_root.setIcon(0, an_icon)
        main_docks_tree.addTopLevelItem(TreeItem_ungrouped_root)


        for a_dock_or_dock_group in dock_tree_list:
            if not isinstance(a_dock_or_dock_group, dict):
                a_dock = a_dock_or_dock_group
                dock_name: str = a_dock.name()
                # an_item = pg.QtWidgets.QTreeWidgetItem(main_docks_tree, [str(dock_name)])
                an_item = pg.QtWidgets.QTreeWidgetItem([str(dock_name)])
                an_item.setData(1, 0, a_dock) # column=0, role=0, value=context
                an_icon = try_get_icon(icon_path=dock_item_icon_path)
                if an_icon is not None:
                    an_item.setIcon(0, an_icon)

                TreeItem_ungrouped_root.addChild(an_item)
            else:
                # a dock-group
                for a_dock_group_name, a_dock_group_list in a_dock_or_dock_group.items():
                    dock_name: str = a_dock_group_name
                    # an_group_item = pg.QtWidgets.QTreeWidgetItem(main_docks_tree, [str(dock_name)])
                    an_group_item = pg.QtWidgets.QTreeWidgetItem([str(dock_name)])
                    an_group_item.setData(1, 0, a_dock_group_list) # column=0, role=0, value=context
                    an_icon = try_get_icon(icon_path=group_icon_path)
                    assert an_icon is not None
                    if an_icon is not None:
                        an_group_item.setIcon(0, an_icon)
                                    
                    main_docks_tree.addTopLevelItem(an_group_item)
            
                    for a_sub_dock in a_dock_group_list:
                        assert not isinstance(a_sub_dock, Dict)
                        a_dock = a_sub_dock
                        dock_name: str = a_dock.name()
                        # an_item = pg.QtWidgets.QTreeWidgetItem(main_docks_tree, [str(dock_name)])
                        an_item = pg.QtWidgets.QTreeWidgetItem([str(dock_name)]) ## set `an_group_item` to parent item
                        an_item.setData(1, 0, a_dock) # column=0, role=0, value=context
                        an_icon = try_get_icon(icon_path=dock_item_icon_path)
                        if an_icon is not None:
                            an_item.setIcon(0, an_icon)
                            
                        an_group_item.addChild(an_item)


        main_docks_tree.expandAll()

    # ==================================================================================================================== #
    # Custom Tree Item Actions                                                                                             #
    # ==================================================================================================================== #
    def show_context_menu(self, position):
        """Shows a custom context menu for leaf tree items"""
        # Get the item at clicked position
        item = self.ui.mainTreeWidget.itemAt(position)
        if item is None:
            return
            
        # Only show menu for leaf nodes (items without children)
        if item.childCount() == 0:
            menu = QtWidgets.QMenu()
            
            # Add menu actions
            show_info_action = menu.addAction("Show Info")
            rename_action = menu.addAction("Rename")
            close_action = menu.addAction("Close dock")
            detach_action = menu.addAction("Detatch dock")
            collapse_action = menu.addAction("Collapse dock")
            toggle_visibility_action = menu.addAction("Toggle dock visibility")
            
            # Connect actions
            show_info_action.triggered.connect(lambda: self.show_item_info(item))
            rename_action.triggered.connect(lambda: self.rename_item(item))
            close_action.triggered.connect(lambda: self.close_dock(item))
            detach_action.triggered.connect(lambda: self.detach_dock(item))
            collapse_action.triggered.connect(lambda: self.collapse_dock(item))
            toggle_visibility_action.triggered.connect(lambda: self.toggle_dock_visibility(item))
            
            # Show the menu at cursor position
            menu.exec_(self.ui.mainTreeWidget.viewport().mapToGlobal(position))

    def show_item_info(self, item):
        """Handler for Show Info action"""
        dock = item.data(1, 0)  # Get the stored dock from column 1
        a_config: CustomDockDisplayConfig = dock.config
        _custom_display_config = CustomDockDisplayConfig(showCloseButton=False, corner_radius='0px') # custom_get_colors_callback_fn=get_utility_dock_colors,
        modal_dock_config_editor: DockConfigEditor = DockConfigEditor(config=a_config)
        
        def _on_modal_dock_config_edited(a_dock_config_editor):
            print(f'_on_modal_dock_config_edited(a_dock....):')
            print(f'\ta_dock_config_editor.config: {a_dock_config_editor.config}')
            

        _a_modal_dock_conn = modal_dock_config_editor.sigDockConfigChanged.connect(lambda x: _on_modal_dock_config_edited(x))
        
        modal_dock_config_editor.show()
        
        
        # # width = self.ui.spinBox_Width.value
        # # height = self.ui.spinBox_Height.value
        # # _geometry = self.geometry()
        # width = int(self.width())
        # height = int(self.height())

        # a_hex_bg_color: str = self.color.name(pg.QtGui.QColor.HexRgb)
        # _custom_display_config.custom_get_colors_dict = {False: DockDisplayColors(fg_color='#fff', bg_color=a_hex_bg_color, border_color=a_hex_bg_color),
        #         True: DockDisplayColors(fg_color='#aaa', bg_color=a_hex_bg_color, border_color=a_hex_bg_color),
        # }

        # # missing keys: dict(widget=self, dockAddLocationOpts=None)
        # # return dict(identifier=self.identifier, widget=self, dockSize=(width,height), dockAddLocationOpts=None, display_config=_custom_display_config, autoOrientation=False)
        # return dict(identifier=self.identifier, dockSize=(width,height), display_config=_custom_display_config, autoOrientation=False)

        info_text = f"Dock Name: {dock.name()}\nType: {type(dock).__name__}"
        from pyphoplacecellanalysis.General.Mixins.DisplayHelpers import debug_widget_geometry


        # QtWidgets.QMessageBox.information(self, "Dock Info", info_text)



    def rename_item(self, item):
        """Handler for Rename action"""
        dock = item.data(1, 0)
        current_name = dock.name()
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Dock", 
            "Enter new name:", 
            QtWidgets.QLineEdit.Normal,
            current_name
        )
        if ok and new_name:
            item.setText(0, new_name)
            # Emit signal to notify of name change
            self.sigDockConfigChanged.emit({"dock": dock, "new_name": new_name})

    def close_dock(self, item):
        """Handler for Close dock action"""
        dock = item.data(1, 0)
        dock.close()

    def detach_dock(self, item):
        """Handler for Detatch dock action"""
        dock = item.data(1, 0)
        dock.float()

    def collapse_dock(self, item):
        """Handler for Collapse dock action"""
        dock = item.data(1, 0)
        dock.collapse()

    def toggle_dock_visibility(self, item):
        """Handler for Toggle dock visibility action"""
        dock = item.data(1, 0)
        if dock.isVisible():
            dock.hide()
        else:
            dock.show()

## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    widget = DockAreaDocksTree()
    widget.show()
    sys.exit(app.exec_())
