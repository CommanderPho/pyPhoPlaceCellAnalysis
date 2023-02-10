"""
Create a pyqt5 class that creates the following gui, described from top to bottom:
1. Contains a label that says "Epochs:"
2. Contains a dynamic listbox containing the list of Epochs current displayed. Each row contains their title, a button to toggle their visibility, and a button to show an options window.
"""
import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class EpochsWindow(QWidget):
    def __init__(self, parent=None):
        super(EpochsWindow, self).__init__(parent)

        self.epoch_list = []

        self.setWindowTitle("Epochs")

        self.initUI()

    def initUI(self):
        # Create the label
        self.label = QLabel("Epochs:")

        # Create the listbox
        self.listbox = QListWidget()
        self.listbox.setSelectionMode(QAbstractItemView.NoSelection)

        # Create the buttons
        self.toggle_btn = QPushButton("Toggle")
        self.options_btn = QPushButton("Options")

        # Create the layout and add the widgets
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.listbox)
        self.layout.addWidget(self.toggle_btn)
        self.layout.addWidget(self.options_btn)

        # Connect the buttons to their respective slots
        self.toggle_btn.clicked.connect(self.toggle_epoch)
        self.options_btn.clicked.connect(self.show_epoch_options)

        # Set the layout
        self.setLayout(self.layout)

    def add_epoch(self, title):
        self.epoch_list.append(title)

        # Add the epoch to the listbox
        item = QListWidgetItem()
        item.setText(title)
        self.listbox.addItem(item)

    def toggle_epoch(self):
        # Get the selected item from the listbox
        item = self.listbox.selectedItems()[0]

        # Toggle the visibility of the epoch
        self.epoch_list[item.text()]["visible"] = not self.epoch_list[item.text()]["visible"]

    def show_epoch_options(self):
        # Get the selected item from the listbox
        item = self.listbox.selectedItems()[0]

        # Show the options window for the epoch
        self.epoch_list[item.text()]["options"].show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EpochsWindow()
    window.show()
    sys.exit(app.exec_())