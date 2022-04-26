# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PlacefieldVisualSelectionWidgetBase.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_rootForm(object):
    def setupUi(self, rootForm):
        rootForm.setObjectName("rootForm")
        rootForm.resize(100, 116)
        rootForm.setStyleSheet("background-color: rgb(71, 58, 46);\n"
"border-color: rgb(207, 207, 207);\n"
"background-color: rgba(71, 65, 60, 180);\n"
"color: rgb(244, 244, 244);\n"
"border-color: rgb(0, 0, 0);")
        self.gridLayout = QtWidgets.QGridLayout(rootForm)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox = QtWidgets.QGroupBox(rootForm)
        self.groupBox.setMaximumSize(QtCore.QSize(160, 160))
        self.groupBox.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.groupBox.setFlat(False)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setContentsMargins(2, 0, 2, 4)
        self.verticalLayout_2.setSpacing(2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.btnTitle = QtWidgets.QPushButton(self.groupBox)
        self.btnTitle.setObjectName("btnTitle")
        self.verticalLayout_2.addWidget(self.btnTitle)
        self.btnColorButton = ColorButton(self.groupBox)
        self.btnColorButton.setMinimumSize(QtCore.QSize(24, 24))
        self.btnColorButton.setText("")
        self.btnColorButton.setObjectName("btnColorButton")
        self.verticalLayout_2.addWidget(self.btnColorButton)
        self.chkbtnPlacefield = QtWidgets.QToolButton(self.groupBox)
        self.chkbtnPlacefield.setCheckable(True)
        self.chkbtnPlacefield.setChecked(False)
        self.chkbtnPlacefield.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.chkbtnPlacefield.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.chkbtnPlacefield.setObjectName("chkbtnPlacefield")
        self.verticalLayout_2.addWidget(self.chkbtnPlacefield)
        self.chkbtnSpikes = QtWidgets.QToolButton(self.groupBox)
        self.chkbtnSpikes.setCheckable(True)
        self.chkbtnSpikes.setChecked(False)
        self.chkbtnSpikes.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.chkbtnSpikes.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.chkbtnSpikes.setObjectName("chkbtnSpikes")
        self.verticalLayout_2.addWidget(self.chkbtnSpikes)
        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)

        self.retranslateUi(rootForm)
        QtCore.QMetaObject.connectSlotsByName(rootForm)

    def retranslateUi(self, rootForm):
        _translate = QtCore.QCoreApplication.translate
        rootForm.setWindowTitle(_translate("rootForm", "Form"))
        self.groupBox.setTitle(_translate("rootForm", "pf[i]"))
        self.btnTitle.setText(_translate("rootForm", "pf[i]"))
        self.chkbtnPlacefield.setText(_translate("rootForm", "pf"))
        self.chkbtnSpikes.setText(_translate("rootForm", "spikes"))
from pyphoplacecellanalysis.External.pyqtgraph.widgets.ColorButton import ColorButton


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    rootForm = QtWidgets.QWidget()
    ui = Ui_rootForm()
    ui.setupUi(rootForm)
    rootForm.show()
    sys.exit(app.exec_())
