from typing import OrderedDict
from ..Qt import QtCore, QtWidgets
from . import VerticalLabel

__all__ = ['CheckTable']

class CheckTable(QtWidgets.QWidget):
    
    sigStateChanged = QtCore.Signal(object, object, object) # (row, col, state)
    
    def __init__(self, columns):
        QtWidgets.QWidget.__init__(self)
        self.layout = QtWidgets.QGridLayout()
        self.layout.setSpacing(0)
        self.setLayout(self.layout)
        self.headers = []
        self.columnNames = []
        self.columnsMap = OrderedDict()
        self.updateColumns(columns)
        
        self.rowNames = []
        self.rowWidgets = []
        self.oldRows = {}  ## remember settings from removed rows; reapply if they reappear.
        
    ## Columns:
    def updateColumns(self, columns):
        # add only:
        self.headers = []
        self.columnNames = columns # no columns
        self.columnsMap = OrderedDict()
        # self.columnNames = [] # no columns
        # col = 1
        for c in columns:
            # label = VerticalLabel.VerticalLabel(c, orientation='vertical')
            # self.headers.append(label)
            # self.layout.addWidget(label, 0, col)
            self.addColumn(c)
            # col += 1
            
    def addColumn(self, name):
        label = VerticalLabel.VerticalLabel(name, orientation='vertical')
        self.headers.append(label)
        # col = len(self.columnNames)+1
        col = len(self.headers) # since this occurs after we add the header widget, it's correct
        self.layout.addWidget(label, 0, col)
        self.columnsMap[name] = (len(self.headers) - 1) # set the index for that column in the columnsMap
        # # TODO: clear make checkmarks for all rows. Currently requires clearing all rows first before adding columns:
        # row = len(self.rowNames)+1
        # checks = []
        # col = 1
        # for c in self.columnNames:
        #     check = QtWidgets.QCheckBox('')
        #     check.col = c
        #     check.row = name
        #     self.layout.addWidget(check, row, col)
        #     checks.append(check)
        #     if name in self.oldRows:
        #         check.setChecked(self.oldRows[name][col])
        #     col += 1
        #     #QtCore.QObject.connect(check, QtCore.SIGNAL('stateChanged(int)'), self.checkChanged)
        #     check.stateChanged.connect(self.checkChanged)
        
    def removeColumn(self, name):
        col = self.columnNames.index(name)
        self.columnNames.pop(col)
        # remove the header widget:
        self.headers[col].setParent(None)
        self.headers.pop(col)
        # for i in range(col, len(self.rowNames)):
        # Remove all of the columns row's widgets:
        # remove this column's widget in each row:
        for i in range(0, len(self.rowNames)):
            widgets = self.rowWidgets[i]
            widgets[col].setParent(None)
            if isinstance(widgets[col], QtWidgets.QCheckBox):
                widgets[col].stateChanged.disconnect(self.checkChanged)
            self.rowWidgets[i].pop(col) # remove it from the list

        # TODO: remove it from the old rows as well?
        
                
                
                
    ## Rows:
    def updateRows(self, rows):
        for r in self.rowNames[:]:
            if r not in rows:
                self.removeRow(r)
        for r in rows:
            if r not in self.rowNames:
                self.addRow(r)

    def addRow(self, name):
        label = QtWidgets.QLabel(name)
        row = len(self.rowNames)+1
        self.layout.addWidget(label, row, 0)
        checks = []
        col = 1
        for c in self.columnNames:
            check = QtWidgets.QCheckBox('')
            check.col = c
            check.row = name
            self.layout.addWidget(check, row, col)
            checks.append(check)
            if name in self.oldRows:
                check.setChecked(self.oldRows[name][col])
            col += 1
            #QtCore.QObject.connect(check, QtCore.SIGNAL('stateChanged(int)'), self.checkChanged)
            check.stateChanged.connect(self.checkChanged)
        self.rowNames.append(name)
        self.rowWidgets.append([label] + checks)
        
    def removeRow(self, name):
        row = self.rowNames.index(name)
        self.oldRows[name] = self.saveState()['rows'][row]  ## save for later
        self.rowNames.pop(row)
        for w in self.rowWidgets[row]:
            w.setParent(None)
            #QtCore.QObject.disconnect(w, QtCore.SIGNAL('stateChanged(int)'), self.checkChanged)
            if isinstance(w, QtWidgets.QCheckBox):
                w.stateChanged.disconnect(self.checkChanged)
        self.rowWidgets.pop(row)
        for i in range(row, len(self.rowNames)):
            widgets = self.rowWidgets[i]
            for j in range(len(widgets)):
                widgets[j].setParent(None)
                self.layout.addWidget(widgets[j], i+1, j)

    def checkChanged(self, state):
        check = QtCore.QObject.sender(self)
        #self.emit(QtCore.SIGNAL('stateChanged'), check.row, check.col, state)
        self.sigStateChanged.emit(check.row, check.col, state)
        
    def saveState(self):
        rows = []
        for i in range(len(self.rowNames)):
            row = [self.rowNames[i]] + [c.isChecked() for c in self.rowWidgets[i][1:]]
            rows.append(row)
        return {'cols': self.columnNames, 'rows': rows}
        
    def restoreState(self, state):
        rows = [r[0] for r in state['rows']]
        self.updateRows(rows)
        for r in state['rows']:
            rowNum = self.rowNames.index(r[0])
            for i in range(1, len(r)):
                self.rowWidgets[rowNum][i].setChecked(r[i])
            
