from qtpy import QtWidgets, QtCore

class CustomGridLayout(QtWidgets.QVBoxLayout):
    """ A replacement for QGridLayout that allows insert/deletion of rows into the layout at runtime to overcome the issue of being unable to set it
    https://stackoverflow.com/questions/42084879/how-to-insert-qwidgets-in-the-middle-of-a-layout
    https://stackoverflow.com/a/42147532
    Credit to K. Muller 
    
    """
    def __init__(self):
        super(CustomGridLayout, self).__init__()
        self.setAlignment(QtCore.Qt.AlignTop)  # !!!
        self.setSpacing(20)


    def addWidget(self, widget, row, col):
        # 1. How many horizontal layouts (rows) are present?
        horLaysNr = self.count()

        # 2. Add rows if necessary
        if row < horLaysNr:
            pass
        else:
            while row >= horLaysNr:
                lyt = QtWidgets.QHBoxLayout()
                lyt.setAlignment((QtCore.Qt.AlignLeft)
                self.addLayout(lyt)
                horLaysNr = self.count()
            ###
        ###

        # 3. Insert the widget at specified column
        self.itemAt(row).insertWidget(col, widget)

    ''''''

    def insertRow(self, row):
        lyt = QtWidgets.QHBoxLayout()
        lyt.setAlignment((QtCore.Qt.AlignLeft)
        self.insertLayout(row, lyt)

    ''''''

    def deleteRow(self, row):
        for j in reversed(range(self.itemAt(row).count())):
            self.itemAt(row).itemAt(j).widget().setParent(None)
        ###
        self.itemAt(row).setParent(None)

    def clear(self):
        for i in reversed(range(self.count())):
            for j in reversed(range(self.itemAt(i).count())):
                self.itemAt(i).itemAt(j).widget().setParent(None)
            ###
        ###
        for i in reversed(range(self.count())):
            self.itemAt(i).setParent(None)
        ###

    ''''''
