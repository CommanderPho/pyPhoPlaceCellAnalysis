from PyQt5.QtWidgets import QApplication, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtWidgets import QStackedWidget, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QPushButton, QWidget, QTableWidgetItem



class TableManager:
    def __init__(self, parent=None):
        self.stacked_widget = QStackedWidget(parent)
        self.tables = []
        self.data_sources = []  # Persistent data source list

    def update_tables(self):
        """
        Synchronize tables with the stored data sources.
        """
        print(f'update_tables()')
        while len(self.tables) > len(self.data_sources):
            print(f'removing...')
            table = self.tables.pop()
            self.stacked_widget.removeWidget(table)
            table.deleteLater()

        for i, data in enumerate(self.data_sources):
            print(f'{i}, {data} ...')
            if i >= len(self.tables):
                table = self._create_table(data)
                self.tables.append(table)
                self.stacked_widget.addWidget(table)
            else:
                self._update_table(self.tables[i], data)

    def add_table(self, data):
        """
        Add a new table and update the display.
        """
        self.data_sources.append(data)
        self.update_tables()

    def remove_table(self, index):
        """
        Remove a table by index and update the display.
        """
        print(f'remove_table(self, index: {index})')
        if 0 <= index < len(self.data_sources):
            self.data_sources.pop(index)
            self.update_tables()

    def _create_table(self, data):
        print(f'create_table(self, data: {data})')
        table = QTableWidget(len(data), len(data[0]) if data else 0)
        self._fill_table(table, data)
        return table

    def _update_table(self, table, data):
        table.setRowCount(len(data))
        table.setColumnCount(len(data[0]) if data else 0)
        self._fill_table(table, data)

    def _fill_table(self, table, data):
        for i, row in enumerate(data):
            for j, value in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(str(value)))

    def set_current_table(self, index):
        if 0 <= index < len(self.tables):
            self.stacked_widget.setCurrentIndex(index)



if __name__ == "__main__":

    app = QApplication([])

    # Main Window
    window = QWidget()
    layout = QVBoxLayout(window)

    # Create TableManager
    manager = TableManager()
    layout.addWidget(manager.stacked_widget)

    # Sample initial data sources
    manager.data_sources = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11], [12, 13]]
    ]
    manager.update_tables()

    # Buttons to test functionality
    def add_table():
        manager.add_table([[14, 15, 16], [17, 18, 19]])

    def remove_table():
        if manager.data_sources:
            manager.remove_table(len(manager.data_sources) - 1)

    def next_table():
        current_index = manager.stacked_widget.currentIndex()
        manager.set_current_table((current_index + 1) % len(manager.tables))

    add_button = QPushButton("Add Table")
    remove_button = QPushButton("Remove Table")
    next_button = QPushButton("Next Table")
    add_button.clicked.connect(add_table)
    remove_button.clicked.connect(remove_table)
    next_button.clicked.connect(next_table)

    layout.addWidget(add_button)
    layout.addWidget(remove_button)
    layout.addWidget(next_button)

    # Show the window
    window.show()
    app.exec_()


