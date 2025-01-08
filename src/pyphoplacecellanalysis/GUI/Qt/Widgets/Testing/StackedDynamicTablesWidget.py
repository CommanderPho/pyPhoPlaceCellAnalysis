from PyQt5.QtWidgets import QApplication, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtWidgets import QStackedWidget, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QPushButton, QWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QWidget, QSizePolicy


# class TableManager:
#     def __init__(self, parent=None):
#         # self.stacked_widget = QStackedWidget(parent)
#         self.layout = layout
#         self.tables = []
#         self.data_sources = []  # Persistent data source list

#     def update_tables(self):
#         """
#         Synchronize tables with the stored data sources.
#         """
#         print(f'update_tables()')
#         while len(self.tables) > len(self.data_sources):
#             print(f'removing...')
#             table = self.tables.pop()
#             self.stacked_widget.removeWidget(table)
#             table.deleteLater()

#         for i, data in enumerate(self.data_sources):
#             print(f'{i}, {data} ...')
#             if i >= len(self.tables):
#                 table = self._create_table(data)
#                 self.tables.append(table)
#                 self.stacked_widget.addWidget(table)
#             else:
#                 self._update_table(self.tables[i], data)

#     def add_table(self, data):
#         """
#         Add a new table and update the display.
#         """
#         self.data_sources.append(data)
#         self.update_tables()

#     def remove_table(self, index):
#         """
#         Remove a table by index and update the display.
#         """
#         print(f'remove_table(self, index: {index})')
#         if 0 <= index < len(self.data_sources):
#             self.data_sources.pop(index)
#             self.update_tables()

#     def _create_table(self, data):
#         print(f'create_table(self, data: {data})')
#         table = QTableWidget(len(data), len(data[0]) if data else 0)
#         self._fill_table(table, data)
#         return table

#     def _update_table(self, table, data):
#         table.setRowCount(len(data))
#         table.setColumnCount(len(data[0]) if data else 0)
#         self._fill_table(table, data)

#     def _fill_table(self, table, data):
#         for i, row in enumerate(data):
#             for j, value in enumerate(row):
#                 table.setItem(i, j, QTableWidgetItem(str(value)))

#     def set_current_table(self, index):
#         if 0 <= index < len(self.tables):
#             self.stacked_widget.setCurrentIndex(index)


class TableManager:
    def __init__(self, layout):
        """
        Initialize the TableManager with a given QVBoxLayout.
        :param layout: The QVBoxLayout to manage tables.
        """
        self.layout = layout
        self.tables = []

    def update_tables(self, data_sources):
        """
        Synchronize the displayed tables with the provided data sources.
        :param data_sources: List of 2D data arrays, each representing a table's data.
        """
        # Remove extra tables
        while len(self.tables) > len(data_sources):
            table = self.tables.pop()
            self.layout.removeWidget(table)
            table.deleteLater()

        # Add or update tables
        for i, data in enumerate(data_sources):
            if i >= len(self.tables):
                table = self._create_table(data)
                self.tables.append(table)
                self.layout.addWidget(table)
            else:
                self._update_table(self.tables[i], data)

    def _create_table(self, data):
        """
        Create a new QTableWidget based on the given data.
        :param data: 2D list of data for the table.
        :return: QTableWidget instance.
        """
        table = QTableWidget(len(data), len(data[0]) if data else 0)
        self._fill_table(table, data)
        # Set size policy to shrink based on content
        table.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()        
        table.setMaximumHeight(self._calculate_table_height(table, len(data)))
        return table

    def _update_table(self, table, data):
        """
        Update an existing QTableWidget with new data.
        :param table: QTableWidget instance.
        :param data: 2D list of data for the table.
        """
        table.setRowCount(len(data))
        table.setColumnCount(len(data[0]) if data else 0)
        self._fill_table(table, data)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.setMaximumHeight(self._calculate_table_height(table, len(data)))


    def _calculate_table_height(self, table, row_count):
        """
        Calculate the appropriate height for the table based on its content, including headers.
        :param table: QTableWidget instance.
        :param row_count: Number of rows in the table.
        :return: Calculated height in pixels.
        """
        row_height = table.sizeHintForRow(0) if row_count > 0 else 0
        header_height = table.horizontalHeader().height()
        vertical_header_width = table.verticalHeader().width()
        return row_count * row_height + header_height + vertical_header_width + 2  # 2 for borders/padding


    def _fill_table(self, table, data):
        """
        Fill a QTableWidget with data.
        :param table: QTableWidget instance.
        :param data: 2D list of data for the table.
        """
        for i, row in enumerate(data):
            for j, value in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(str(value)))


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


