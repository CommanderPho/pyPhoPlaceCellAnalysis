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

        self.epoch_list = {}

        self.setWindowTitle("Epochs")

        self.initUI()

    def initUI(self):
        # Create the label
        self.label = QLabel("Epochs:")

        # Create the listbox
        self.listbox = QListWidget()
        self.listbox.setSelectionMode(QAbstractItemView.NoSelection)

        # Create the layout and add the widgets
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.listbox)


        # Set the layout
        self.setLayout(self.layout)

    def add_epoch(self, title):
        self.epoch_list[title] = {"visible": True, "options": None}

        # Create the buttons
        label = QLabel(title)
        toggle_btn = QPushButton("Toggle")
        options_btn = QPushButton("Options")

        # Create the layout and add the buttons
        layout = QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(toggle_btn)
        layout.addWidget(options_btn)

        # Create the widget and set the layout
        widget = QWidget()
        widget.setLayout(layout)

        # Create the list item and set the widget
        item = QListWidgetItem(self.listbox)
        item.setSizeHint(widget.sizeHint())
        self.listbox.addItem(item)
        self.listbox.setItemWidget(item, widget)

        # Connect the buttons to their respective slots
        toggle_btn.clicked.connect(lambda: self.toggle_epoch(title))
        options_btn.clicked.connect(lambda: self.show_epoch_options(title))

    def toggle_epoch(self, title):
        # Toggle the visibility of the epoch
        self.epoch_list[title]["visible"] = not self.epoch_list[title]["visible"]

    def show_epoch_options(self, title):
        # Show the options window for the epoch
        self.epoch_list[title]["options"].show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EpochsWindow()
    window.show()
    ## Test adding epochs:
    for an_epoch_name in ["Epoch 1", "Epoch 2", "Epoch 3"]:
        window.add_epoch(an_epoch_name)

    sys.exit(app.exec_())