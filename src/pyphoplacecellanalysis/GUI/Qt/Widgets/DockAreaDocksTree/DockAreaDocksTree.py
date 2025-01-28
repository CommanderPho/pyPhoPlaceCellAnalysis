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

## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'DockAreaDocksTree.ui')

class DockAreaDocksTree(QWidget):
    """ 
    
    from pyphoplacecellanalysis.GUI.Qt.Widgets.DockAreaDocksTree.DockAreaDocksTree import DockAreaDocksTree
    
    
    dynamic_docked_widget_container = active_2d_plot.ui.dynamic_docked_widget_container # NestedDockAreaWidget
    dock_tree_list = dynamic_docked_widget_container.get_dockGroup_dock_tree_dict()
    a_dock_area_docks_tree_widget: DockAreaDocksTree = DockAreaDocksTree()
    a_dock_area_docks_tree_widget.rebuild_dock_tree_items(dock_tree_list=dock_tree_list)
    a_dock_area_docks_tree_widget.show()

    """

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
        pass
    

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


## Start Qt event loop
if __name__ == '__main__':
    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    widget = DockAreaDocksTree()
    widget.show()
    sys.exit(app.exec_())
