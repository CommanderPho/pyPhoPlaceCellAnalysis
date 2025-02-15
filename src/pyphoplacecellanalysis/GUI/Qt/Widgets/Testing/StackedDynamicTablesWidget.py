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

from pyphocorehelpers.gui.Qt.pandas_model import SimplePandasModel # , GroupedHeaderView #, create_tabbed_table_widget

## For dock widget
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.NestedDockAreaWidget import NestedDockAreaWidget
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig

from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView, QMenu, QAction, QVBoxLayout, QWidget
from PyQt5.QtCore import QAbstractTableModel, Qt

_table_stylesheet: str = '''
/* General table styling */
QTableView {
    background-color: #2e2e2e;   /* Background of the table */
    color: #ffffff;             /* Text color */
    border: 1px solid #444444;  /* Border around the table */
    gridline-color: #555555;    /* Gridline color */
    selection-background-color: #3c0058; /* Background of selected items */
    selection-color: #f6d1ff;   /* Text color of selected items */
}

/* Header styling */
QHeaderView::section {
    background-color: #444444;  /* Background of header sections */
    color: #ffffff;             /* Text color in the header */
    padding: 4px;               /* Padding inside header sections */
    border: 1px solid #555555;  /* Border for each header section */
}

# QHeaderView::section {
#     background-color: lightblue;
#     color: black;
#     border: 1px solid gray;
#     font-weight: bold;
# }

            
/* Alternating row colors */
QTableView::item {
    background-color: #3a3a3a;  /* Default row background */
}

QTableView::item:alternate {
    background-color: #414141;  /* Alternating row background */
}

/* Focused item styling */
QTableView::item:focus {
    background-color: #666666;  /* Background of focused item */
    border: 1px solid #888888;  /* Border for the focused item */
}

QTableView::item:selected {
	background-color: rgb(85, 0, 127);
	color: rgb(227, 225, 255);
}
QTableView::item:selected:active {
	background-color: rgb(85, 0, 127);
	color: rgb(227, 225, 255);
}


# /* Scrollbar styling */
# QScrollBar:vertical {
#     background: #2e2e2e;        /* Background of the vertical scrollbar */
#     width: 15px;                /* Width of the scrollbar */
#     margin: 15px 0 15px 0;      /* Margins around the scrollbar */
# }

# QScrollBar::handle:vertical {
#     background: #555555;        /* Scroll handle color */
#     min-height: 20px;           /* Minimum height for the handle */
# }

# QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
#     background: #444444;        /* Arrow buttons background */
#     border: 1px solid #555555;  /* Border for arrow buttons */
# }

# QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
#     background: none;           /* Space above and below the handle */
# }

# QScrollBar:horizontal {
#     background: #2e2e2e;        /* Background of the horizontal scrollbar */
#     height: 15px;               /* Height of the scrollbar */
#     margin: 0 15px 0 15px;      /* Margins around the scrollbar */
# }

# QScrollBar::handle:horizontal {
#     background: #555555;        /* Scroll handle color */
#     min-width: 20px;            /* Minimum width for the handle */
# }

# QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
#     background: #444444;        /* Arrow buttons background */
#     border: 1px solid #555555;  /* Border for arrow buttons */
# }

# QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
#     background: none;           /* Space left and right of the handle */
# }

/* Corner widget styling (if present) */
QTableCornerButton::section {
    background-color: #444444;  /* Corner button background */
    border: 1px solid #555555;  /* Border for the corner button */
}

'''

@metadata_attributes(short_name=None, tags=['epochs', 'tables', 'ui'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-08 16:40', related_items=[])
class CustomHeaderTableView(pg.QtWidgets.QTableView):
    """ QTableView with custom header and context menu for column visibility. """
    def __init__(self, model=None, visible_columns=None):
        super().__init__()
        self._column_visibility_menu = QMenu(self)
        self.column_actions = []  # Stores actions for each column
        self.visible_columns = visible_columns  # List of visible column indices or names
        if model is not None:
            self.setModel(model)

    def setModel(self, model):
        """ Override setModel to attach additional functionality when a model is set. """
        super().setModel(model)
        if model is not None:
            self.initCustomHeaders()

    def initCustomHeaders(self):
        """ Add context menu functionality to the horizontal header. """
        # Clear existing actions to prevent duplication
        self._column_visibility_menu.clear()
        self.column_actions = []

        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self.showColumnContextMenu)

        num_cols: int = self.model().columnCount()
        if (self.visible_columns is None) or (len(self.visible_columns) == 0):
            ## set default columns
            self.visible_columns = np.arange(num_cols) ## all visible

        # Initialize column actions and visibility
        for col in range(num_cols):
            # Check if the column should be visible
            is_visible = True
            if (self.visible_columns is not None):
                if isinstance(self.visible_columns[0], int):  # Indices
                    is_visible = col in self.visible_columns
                elif isinstance(self.visible_columns[0], str):  # Column names
                    column_name = self.model().headerData(col, Qt.Horizontal, Qt.DisplayRole)
                    is_visible = column_name in self.visible_columns

            # Apply initial visibility
            self.setColumnHidden(col, not is_visible)

            # Create context menu action
            action = QAction(self.model().headerData(col, Qt.Horizontal, Qt.DisplayRole), self)
            action.setCheckable(True)
            action.setChecked(is_visible)
            action.triggered.connect(partial(self.toggle_column, col))
            self.column_actions.append(action)
            self._column_visibility_menu.addAction(action)

    def showColumnContextMenu(self, position):
        """ Show context menu for column visibility at the requested position. """
        header = self.horizontalHeader()
        global_position = header.mapToGlobal(position)
        self._column_visibility_menu.exec_(global_position)

    def toggle_column(self, column, visible):
        """ Toggle visibility of the specified column. """
        self.setColumnHidden(column, not visible)


@metadata_attributes(short_name=None, tags=['table', 'manager', 'ui'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-08 16:41', related_items=[])
class TableManager:
    """ Manages a dynamically updating dict of tables, rendered as docked widgets
    
    visible_columns_dict: specifies the table columns to show for a given dataseries name
    
    
    """
    def __init__(self, parent_widget, visible_columns_dict: Dict[str, List[str]]=None):
        if visible_columns_dict is None:
            visible_columns_dict = {} 

        self.visible_columns_dict = visible_columns_dict

        ## Setup the widgets:
        self.parent_widget = parent_widget
        
        # Create the dynamic docked widget container
        self.dynamic_docked_widget_container = NestedDockAreaWidget()
        self.dynamic_docked_widget_container.setObjectName("dynamic_docked_widget_container")
        
        # Create a layout for the wrapper
        self.wrapper_layout = QVBoxLayout(parent_widget)
        self.wrapper_layout.setSpacing(0)
        self.wrapper_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add the container to the wrapper layout
        self.wrapper_layout.addWidget(self.dynamic_docked_widget_container)
        
        self.dock_items = {}  # Store dock items
        self.models = {}  # dict of SimplePandasModel objects

    def update_tables(self, data_sources: Dict[str, pd.DataFrame]):
        # Remove old dock items no longer present
        to_remove = []
        for name in self.dock_items.keys():
            if name not in data_sources:
                to_remove.append(name)
        for name in to_remove:
            self.remove_table_dock(name)

        # Add or update tables
        for name, df in data_sources.items():
            if name not in self.dock_items:
                self.add_table_dock(name, df)
            else:
                # Update existing table
                self._update_table(self.dock_items[name], df)

    def add_table_dock(self, name: str, df: pd.DataFrame, dockSize=(500,100)):
        """Creates a new docked table widget

            Uses: `self.visible_columns_dict` to try and determine what columns to display for this dataseries name
        """
        display_config = CustomDockDisplayConfig(showCloseButton=True, orientation='horizontal')
        
        # Create table widget
        visible_columns = self.visible_columns_dict.get(name, ['start', 'delta_aligned_start_t', 'label', 'unique_active_aclus'])
        
        included_visible_columns = [col for col in df.columns if col in visible_columns]
        table, model = self._create_table(df, visible_columns=included_visible_columns)

        # No extant table widget and display_dock currently, create a new one:
        dDisplayItem = self.dynamic_docked_widget_container.find_display_dock(identifier=name) # Dock
        assert dDisplayItem is None
        
        # Add to dynamic dock container 
        _, dDisplayItem = self.dynamic_docked_widget_container.add_display_dock(name, dockSize=dockSize, display_config=display_config, widget=table, dockAddLocationOpts=['bottom'], autoOrientation=False)

        dDisplayItem.setOrientation('horizontal', force=True)
        dDisplayItem.updateStyle()
        dDisplayItem.update()
        
        self.dock_items[name] = dDisplayItem
        self.models[name] = model
        return dDisplayItem

    def remove_table_dock(self, name: str):
        """Removes a docked table widget"""
        if name in self.dock_items:
            dock_item = self.dock_items.pop(name)
            self.models.pop(name)
            self.dynamic_docked_widget_container.remove_display_dock(name)

    def find_table(self, name: str):
        """ returns all related components for a table
        
        table, dDisplayItem, model = manager.find_table(name='Table_3')
        """
        dDisplayItem = self.dynamic_docked_widget_container.find_display_dock(identifier=name) # Dock
        dock_children_widgets = dDisplayItem.widgets
        assert len(dock_children_widgets) == 1, f"dock_children_widgets: {dock_children_widgets}, dock_item: {dDisplayItem}"
        table = dock_children_widgets[0]
        model = self.models[name]
        return table, dDisplayItem, model

    def highlight_row(self, name: str, row_index: int):
        # row_index = 2  # Row to highlight (0-based index)
        table, dDisplayItem, model = self.find_table(name=name)
        table.selectRow(row_index)

    def scroll_to_row(self, name: str, row_index: int):
        # row_index = 4  # Row to scroll to (0-based index)
        table, dDisplayItem, model = self.find_table(name=name)
        table.scrollTo(model.index(row_index, 0), QTableView.PositionAtTop)


    # ==================================================================================================================== #
    # Private Methods                                                                                                      #
    # ==================================================================================================================== #
    def _create_table(self, df: pd.DataFrame, visible_columns=None):
        headers: List[str] = [str(col) for col in df.columns]
        
        # table = pg.QtWidgets.QTableView()
        table = CustomHeaderTableView(visible_columns=visible_columns)
        model = self._fill_table(table, df)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        table.setSelectionBehavior(table.SelectRows)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.setStyleSheet(_table_stylesheet)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().hide()
        ## enable menu for showing/hiding columns:
        table.column_visibility_menu = QMenu("Columns", table)
        table.column_actions = []
        for i, header in enumerate(headers):
            action = QAction(header, table, checkable=True)
            action.setChecked(True)
            action.triggered.connect(lambda checked, col=i: table.toggle_column(col, checked))
            table.column_visibility_menu.addAction(action)
            table.column_actions.append(action)

        # menu_bar = self.dynamic_docked_widget_container.window().menuBar()
        # menu_bar.addMenu(table.column_visibility_menu)

        return (table, model)

    def _update_table(self, dock_item, df: pd.DataFrame):
        dock_children_widgets = dock_item.widgets
        assert len(dock_children_widgets) == 1, f"dock_children_widgets: {dock_children_widgets}, dock_item: {dock_item}"
        table = dock_children_widgets[0]
        model = self._fill_table(table, df)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        return model

    def _fill_table(self, table, df: pd.DataFrame) -> SimplePandasModel:
        curr_model = SimplePandasModel(df.copy())
        table.setModel(curr_model)
        return curr_model



def test_TableManager(n_tables: int = 4):
    """ 

    from pyphoplacecellanalysis.GUI.Qt.Widgets.Testing.StackedDynamicTablesWidget import test_TableManager, CustomHeaderTableView, TableManager

    """
    import random
    import pandas as pd

    # Helper function to generate random data
    def generate_random_data(avg_n_rows: int = 5):
        rows = random.randint((avg_n_rows-2), (avg_n_rows+10))  # Random number of rows
        cols = random.randint(2, 6)   # Random number of columns
        data = [[random.randint(1, 100) for _ in range(cols)] for _ in range(rows)]
        columns = [f'Col_{i}' for i in range(cols)]
        return pd.DataFrame(data, columns=columns)


    # Main Window
    window = QWidget()
    # layout = QVBoxLayout(window)

    # Create TableManager
    manager = TableManager(window)

    # Initial random data sources
    data_sources = [generate_random_data() for _ in range(n_tables-1)]
    
    # Insert a small dataframe
    small_df = pd.DataFrame([[0, 0], [1, 1]], columns=['A', 'B'])
    data_sources.insert(1, small_df)
    
    # Create dictionary with names for the dataframes
    named_data_sources = {f'Table_{i}': df for i, df in enumerate(data_sources)}
    manager.update_tables(named_data_sources)

    # Buttons to test functionality
    def add_table():
        new_df = pd.DataFrame([[14, 15, 16], [17, 18, 19]], columns=['A', 'B', 'C'])
        named_data_sources[f'Table_{len(named_data_sources)}'] = new_df
        manager.update_tables(named_data_sources)

    def remove_table():
        if named_data_sources:
            named_data_sources.pop(list(named_data_sources.keys())[-1])
            manager.update_tables(named_data_sources)

    add_button = QPushButton("Add Table")
    remove_button = QPushButton("Remove Table")
    add_button.clicked.connect(add_table)
    remove_button.clicked.connect(remove_table)

    manager.wrapper_layout.addWidget(add_button)
    manager.wrapper_layout.addWidget(remove_button)

    window.setWindowTitle('Stacked Dynamics Tables Widget (test, Pho)')
    

    return manager, named_data_sources, (window, add_button, remove_button)

if __name__ == "__main__":
        
    app = QApplication([])
        
    manager, named_data_sources, (window, add_button, remove_button) = test_TableManager()
    # Show the window
    window.show()

    app.exec_()


