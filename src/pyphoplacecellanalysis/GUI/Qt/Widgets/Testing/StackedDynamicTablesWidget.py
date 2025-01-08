from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import pandas as pd

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

import pyphoplacecellanalysis.External.pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHeaderView, QPushButton, QWidget, QSizePolicy, QLabel

from pyphocorehelpers.gui.Qt.pandas_model import SimplePandasModel # , GroupedHeaderView #, create_tabbed_table_widget


class TableManager:
    """ Manages a dynamically updating dict of tables, rendered as a vertical stack of tables
    """
    def __init__(self, layout):
        """
        Initialize the TableManager with a given QVBoxLayout.
        :param layout: The QVBoxLayout to manage tables.
        """
        self.layout = layout
        self.tables = {}
        self.labels = {}  # Store labels for each table
        self.models = {}  # dict of `SimplePandasModel` objects
        self.layout.setSpacing(2)  # tighter spacing between items
        

    def update_tables(self, data_sources: Dict[str, pd.DataFrame]):
        # Remove old tables/labels no longer present
        to_remove = []
        for name in self.tables.keys():
            if name not in data_sources:
                to_remove.append(name)
        for name in to_remove:
            table = self.tables.pop(name)
            self.layout.removeWidget(table)
            table.deleteLater()
            self.models.pop(name).deleteLater()
            # Remove its label too
            label = self.labels.pop(name, None)
            if label is not None:
                self.layout.removeWidget(label)
                label.deleteLater()

        # Add or update tables
        for name, df in data_sources.items():
            if name not in self.tables:
                # Create and insert label above the table
                label = QLabel(name)
                label.setStyleSheet("font-weight: bold; margin: 0px;")  # Customize as needed
                self.labels[name] = label
                self.layout.addWidget(label)

                # Create table
                table, model = self._create_table(df)
                self.tables[name] = table
                self.models[name] = model
                self.layout.addWidget(table)
            else:
                self.models[name] = self._update_table(self.tables[name], df)

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



import random

# Helper function to generate random data
def generate_random_data():
    rows = random.randint(2, 10)  # Random number of rows
    cols = random.randint(2, 6)   # Random number of columns
    return [[random.randint(1, 100) for _ in range(cols)] for _ in range(rows)]


if __name__ == "__main__":


    app = QApplication([])

    # Main Window
    window = QWidget()
    layout = QVBoxLayout(window)

    # Create TableManager
    # manager = TableManager()
    # layout.addWidget(manager.stacked_widget)
    
    # Create TableManager
    manager = TableManager(layout)

    # Sample data sources
    data_sources = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11], [12, 13]]
    ]
    
    # Initial random data sources
    data_sources = [generate_random_data() for _ in range(3)]
    
    data_sources.insert(1, [[0, 0], [1, 1]]) ## small entry (only 2 rows)
    manager.update_tables(data_sources)

    # Buttons to test functionality
    def add_table():
        data_sources.append([[14, 15, 16], [17, 18, 19]])
        manager.update_tables(data_sources)

    def remove_table():
        if data_sources:
            data_sources.pop()
            manager.update_tables(data_sources)

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


