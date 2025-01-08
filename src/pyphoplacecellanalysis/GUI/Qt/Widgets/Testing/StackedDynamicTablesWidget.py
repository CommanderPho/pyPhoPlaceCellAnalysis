from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import pandas as pd

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

import pyphoplacecellanalysis.External.pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHeaderView, QPushButton, QWidget, QSizePolicy, QLabel

from pyphocorehelpers.gui.Qt.pandas_model import SimplePandasModel # , GroupedHeaderView #, create_tabbed_table_widget

## For dock widget
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.NestedDockAreaWidget import NestedDockAreaWidget
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig


class TableManager:
    """ Manages a dynamically updating dict of tables, rendered as a vertical stack of tables
    """
    def __init__(self, layout):
        self.layout = layout
        self.dock_items = {}  # Store dock items instead of separate tables/labels
        self.models = {}  # dict of SimplePandasModel objects
        self.layout.setSpacing(2)

    def update_tables(self, data_sources: Dict[str, pd.DataFrame]):
        # Remove old dock items no longer present
        to_remove = []
        for name in self.dock_items.keys():
            if name not in data_sources:
                to_remove.append(name)
        for name in to_remove:
            dock_item = self.dock_items.pop(name)
            self.layout.removeWidget(dock_item)
            dock_item.deleteLater()
            self.models.pop(name).deleteLater()

        # Add or update tables
        for name, df in data_sources.items():
            if name not in self.dock_items:
                # Create new dock item containing label and table
                dock_item = pg.DockItem(name=name)
                dock_layout = QVBoxLayout()
                
                # Add label
                label = QLabel(name)
                label.setStyleSheet("font-weight: bold; margin: 0px;")
                dock_layout.addWidget(label)
                
                # Create and add table
                table, model = self._create_table(df)
                dock_layout.addWidget(table)
                
                dock_item.setLayout(dock_layout)
                self.dock_items[name] = dock_item
                self.models[name] = model
                self.layout.addWidget(dock_item)
            else:
                # Update existing table
                dock_item = self.dock_items[name]
                table = dock_item.layout().itemAt(1).widget()  # Get table widget
                self.models[name] = self._update_table(table, df)

    def _create_table(self, df: pd.DataFrame):
        table = pg.QtWidgets.QTableView()
        model = self._fill_table(table, df)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.setMaximumHeight(self._calculate_table_height(table, len(df)))
        table.setStyleSheet("""
            QHeaderView::section {
                background-color: lightblue;
                color: black;
                border: 1px solid gray;
                font-weight: bold;
            }
        """)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        return (table, model)

    def _update_table(self, table, df: pd.DataFrame):
        model = self._fill_table(table, df)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.setMaximumHeight(self._calculate_table_height(table, len(df)))
        return model

    def _calculate_table_height(self, table, row_count: int):
        row_height = table.sizeHintForRow(0) if row_count > 0 else 0
        header_height = table.horizontalHeader().height()
        vertical_header_width = table.verticalHeader().width()
        scrollbar_height = table.horizontalScrollBar().height()
        # Some padding for borders or margins
        return ((row_count + 1) * row_height) + header_height + vertical_header_width + 4 + scrollbar_height

    def _fill_table(self, table, df: pd.DataFrame) -> SimplePandasModel:
        curr_model = SimplePandasModel(df.copy())
        table.setModel(curr_model)
        return curr_model





if __name__ == "__main__":

    import random
    import pandas as pd

    # Helper function to generate random data
    def generate_random_data():
        rows = random.randint(2, 10)  # Random number of rows
        cols = random.randint(2, 6)   # Random number of columns
        data = [[random.randint(1, 100) for _ in range(cols)] for _ in range(rows)]
        columns = [f'Col_{i}' for i in range(cols)]
        return pd.DataFrame(data, columns=columns)

    app = QApplication([])

    # Main Window
    window = QWidget()
    layout = QVBoxLayout(window)

    # Create TableManager
    manager = TableManager(layout)

    # Initial random data sources
    data_sources = [generate_random_data() for _ in range(3)]
    
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

    layout.addWidget(add_button)
    layout.addWidget(remove_button)

    window.setWindowTitle('Stacked Dynamics Tables Widget (test, Pho)')
    
    # Show the window
    window.show()
    app.exec_()


