from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtGui, QtCore

class PhoPipelineSecondaryWindow(QtGui.QWidget):
    """
    This "window" is a QWidget. If it has no parent,
    it will appear as a free-floating window.
    """
    def __init__(self, contents):
        super().__init__()
        layout = QtGui.QVBoxLayout()
        for a_widget in contents:
            layout.addWidget(a_widget)
        self.setLayout(layout)

