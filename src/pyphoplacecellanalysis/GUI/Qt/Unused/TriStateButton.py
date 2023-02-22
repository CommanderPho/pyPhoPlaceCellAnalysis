"""
Create a pyqt5 button class that allows tri-state selection.
"""

from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

import pyphoplacecellanalysis.External.pyqtgraph as pg

# class TriStateButton(QCheckBox):
#     """
#     A button that can be in three states:
#     - Unchecked
#     - Checked
#     - Partially checked
#     """
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setCheckable(True)
#         self.setCheckState(Qt.Unchecked)
#         self.setIcon(QIcon(":/icons/checkbox_unchecked.png"))
#         self.setIconSize(self.iconSize())
#         self.clicked.connect(self.on_clicked)

#     def on_clicked(self):
#         """
#         When the button is clicked, change the state.
#         """
#         if self.checkState() == Qt.Unchecked:
#             self.setCheckState(Qt.Checked)
#             self.setIcon(QIcon(":/icons/checkbox_checked.png"))
#         elif self.checkState() == Qt.Checked:
#             self.setCheckState(Qt.PartiallyChecked)
#             self.setIcon(QIcon(":/icons/checkbox_partially_checked.png"))
#         elif self.checkState() == Qt.PartiallyChecked:
#             self.setCheckState(Qt.Unchecked)
#             self.setIcon(QIcon(":/icons/checkbox_unchecked.png"))

from PyQt5.QtWidgets import QPushButton

class TriStateButton(QPushButton):
    """
    A QPushButton that supports three states: 
    - 'Off' (default state)
    - 'On'
    - 'Indeterminate'
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = 'Off'
        self.setCheckable(True)
        self.clicked.connect(self._updateState)

    def _updateState(self):
        if self.isChecked():
            if self._state == 'On':
                self._state = 'Indeterminate'
            elif self._state == 'Indeterminate':
                self._state = 'Off'
            else:
                self._state = 'On'
        else:
            self._state = 'Off'

    def getState(self):
        return self._state



def main():
    app = pg.mkQApp("RadialMenu Test")
	# app.setStyleSheet(stylesheet_data_stream.readAll())
	# app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5()) # QDarkStyle version

    test_widget = TriStateButton()
    test_widget.show()
    pg.exec()
        
if __name__ == '__main__':
    main()
