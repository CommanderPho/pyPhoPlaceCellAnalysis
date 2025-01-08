from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
import pandas as pd

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

import pyphoplacecellanalysis.External.pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHeaderView, QPushButton, QWidget, QSizePolicy

from pyphocorehelpers.gui.Qt.pandas_model import SimplePandasModel # , GroupedHeaderView #, create_tabbed_table_widget


class TableManager:
    """ Manages a dynamically updating dict of tables, rendered as a vertical stack of tables
    
    
        from pyphoplacecellanalysis.GUI.Qt.Widgets.Testing.StackedDynamicTablesWidget import TableManager
    """
    def __init__(self, layout):
        """
        Initialize the TableManager with a given QVBoxLayout.
        :param layout: The QVBoxLayout to manage tables.
        """
        self.layout = layout
        self.tables = {}
        self.models = {} ## dict of `SimplePandasModel` objects

    def update_tables(self, data_sources: Dict[str, pd.DataFrame]):
        """
        Synchronize the displayed tables with the provided data sources.
        :param data_sources: List of 2D data arrays, each representing a table's data.

        Updates:
            self.tables[a_dataseries_name]
            self.models[a_dataseries_name]
        """
        # Remove extra tables ________________________________________________________________________________________________ #
        # while len(self.tables) > len(data_sources):
        for a_dataseries_name, a_table in self.tables.items():
            if a_dataseries_name not in data_sources:
                table = self.tables.pop(a_dataseries_name)
                self.layout.removeWidget(table)
                table.deleteLater()
                model = self.models.pop(a_dataseries_name) ## remove the model
                model.deleteLater()


        # Add or update tables
        # for i, data in enumerate(data_sources):
        for i, (a_dataseries_name, df) in enumerate(data_sources.items()):
            # if i >= len(self.tables):
            if (a_dataseries_name not in self.tables):
                ## need a new table and model:
                table, model = self._create_table(df)
                self.tables[a_dataseries_name] = table
                self.models[a_dataseries_name] = model
                # self.tables.append(table)
                self.layout.addWidget(table)
            else:
                ## table/model should already exist -- update them
                self.models[a_dataseries_name] = self._update_table(self.tables[a_dataseries_name], df)

    def _create_table(self, df: pd.DataFrame):
        """
        Create a new QTableWidget based on the given data.
        :param data: 2D list of data for the table.
        :return: QTableWidget instance.
        """
        # table = QTableWidget(len(df), len(df[0]) if df else 0)
        table = pg.QtWidgets.QTableView()
        model = self._fill_table(table, df)
        # Set size policy to shrink based on content
        # table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()        
        table.setMaximumHeight(self._calculate_table_height(table, len(df)))
        # Apply stylesheet to change header background color
        table.setStyleSheet("""
        QHeaderView::section {
            background-color: lightblue;
            color: black;  /* Optional: Change text color */
            border: 1px solid gray;  /* Optional: Add border */
            font-weight: bold;  /* Optional: Make text bold */
        }
        """)
        # table.setStyleSheet("QHeaderView::section { background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(255, 255, 255, 255)) }")

        # Optional: Resize to fit contents
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)


        return (table, model)

    def _update_table(self, table, df: pd.DataFrame):
        """
        Update an existing QTableWidget with new data.
        :param table: QTableWidget instance.
        :param data: 2D list of data for the table.
        """
        # table.setRowCount(len(df))
        # table.setColumnCount(len(df[0]) if df else 0)
        model = self._fill_table(table, df)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.setMaximumHeight(self._calculate_table_height(table, len(df)))


    def _calculate_table_height(self, table, row_count: int):
        """
        Calculate the appropriate height for the table based on its content, including headers.
        :param table: QTableWidget instance.
        :param row_count: Number of rows in the table.
        :return: Calculated height in pixels.
        """
        row_height = table.sizeHintForRow(0) if row_count > 0 else 0
        header_height = table.horizontalHeader().height()
        vertical_header_width = table.verticalHeader().width()
        
        # Get the heights and widths
        vertical_scrollbar_height = table.horizontalScrollBar().height()
        # horizontal_scrollbar_width = table.verticalScrollBar().width()

        return ((row_count+1) * row_height) + header_height + vertical_header_width + 4 + vertical_scrollbar_height  # 4 for borders/padding


    def _fill_table(self, table, df: pd.DataFrame) -> SimplePandasModel:
        """
        Fill a QTableWidget with data.
        :param table: QTableWidget instance.
        :param data: 2D list of data for the table.
        """
        
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


