# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\DockPlanningHelperWidget\TinyDockPlanningHelperWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(502, 25)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        Form.setMinimumSize(QtCore.QSize(0, 25))
        Form.setBaseSize(QtCore.QSize(500, 25))
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(2)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btnAddWidgetBelow = QtWidgets.QToolButton(Form)
        self.btnAddWidgetBelow.setMinimumSize(QtCore.QSize(16, 23))
        self.btnAddWidgetBelow.setBaseSize(QtCore.QSize(23, 23))
        self.btnAddWidgetBelow.setArrowType(QtCore.Qt.DownArrow)
        self.btnAddWidgetBelow.setObjectName("btnAddWidgetBelow")
        self.horizontalLayout.addWidget(self.btnAddWidgetBelow)
        self.btnAddWidgetAbove = QtWidgets.QToolButton(Form)
        self.btnAddWidgetAbove.setMinimumSize(QtCore.QSize(23, 22))
        self.btnAddWidgetAbove.setToolTip("Add DockPlanningHelperWidget above (tabbed)")
        self.btnAddWidgetAbove.setArrowType(QtCore.Qt.UpArrow)
        self.btnAddWidgetAbove.setObjectName("btnAddWidgetAbove")
        self.horizontalLayout.addWidget(self.btnAddWidgetAbove)
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.txtDockTitle = QtWidgets.QLineEdit(Form)
        self.txtDockTitle.setObjectName("txtDockTitle")
        self.horizontalLayout.addWidget(self.txtDockTitle)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.spinBox_Width = QtWidgets.QSpinBox(Form)
        self.spinBox_Width.setToolTip("DockWidth")
        self.spinBox_Width.setStatusTip("DockWidth")
        self.spinBox_Width.setFrame(False)
        self.spinBox_Width.setReadOnly(True)
        self.spinBox_Width.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_Width.setKeyboardTracking(False)
        self.spinBox_Width.setMinimum(1)
        self.spinBox_Width.setMaximum(2000)
        self.spinBox_Width.setSingleStep(100)
        self.spinBox_Width.setProperty("value", 600)
        self.spinBox_Width.setObjectName("spinBox_Width")
        self.horizontalLayout.addWidget(self.spinBox_Width)
        self.spinBox_Height = QtWidgets.QSpinBox(Form)
        self.spinBox_Height.setToolTip("Height")
        self.spinBox_Height.setStatusTip("Height")
        self.spinBox_Height.setFrame(False)
        self.spinBox_Height.setReadOnly(True)
        self.spinBox_Height.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinBox_Height.setKeyboardTracking(False)
        self.spinBox_Height.setMinimum(1)
        self.spinBox_Height.setMaximum(8000)
        self.spinBox_Height.setProperty("value", 300)
        self.spinBox_Height.setObjectName("spinBox_Height")
        self.horizontalLayout.addWidget(self.spinBox_Height)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.btnColorButton = ColorButton(Form)
        self.btnColorButton.setMinimumSize(QtCore.QSize(24, 24))
        self.btnColorButton.setBaseSize(QtCore.QSize(24, 24))
        self.btnColorButton.setText("")
        self.btnColorButton.setObjectName("btnColorButton")
        self.horizontalLayout.addWidget(self.btnColorButton)
        self.btnAddWidgetRight = QtWidgets.QToolButton(Form)
        self.btnAddWidgetRight.setMinimumSize(QtCore.QSize(16, 23))
        self.btnAddWidgetRight.setBaseSize(QtCore.QSize(23, 23))
        self.btnAddWidgetRight.setArrowType(QtCore.Qt.RightArrow)
        self.btnAddWidgetRight.setObjectName("btnAddWidgetRight")
        self.horizontalLayout.addWidget(self.btnAddWidgetRight)
        self.horizontalLayout.setStretch(3, 1)
        self.horizontalLayout.setStretch(4, 1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.label.setBuddy(self.txtDockTitle)
        self.label_3.setBuddy(self.spinBox_Width)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        Form.setTabOrder(self.txtDockTitle, self.spinBox_Width)
        Form.setTabOrder(self.spinBox_Width, self.spinBox_Height)
        Form.setTabOrder(self.spinBox_Height, self.btnAddWidgetBelow)
        Form.setTabOrder(self.btnAddWidgetBelow, self.btnAddWidgetRight)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.btnAddWidgetBelow.setToolTip(_translate("Form", "Add DockPlanningHelperWidget below"))
        self.btnAddWidgetBelow.setText(_translate("Form", "..."))
        self.btnAddWidgetAbove.setText(_translate("Form", "..."))
        self.label.setText(_translate("Form", "Title"))
        self.txtDockTitle.setText(_translate("Form", "Position Decoder"))
        self.txtDockTitle.setPlaceholderText(_translate("Form", "Position Decoder"))
        self.label_3.setText(_translate("Form", "dockSize"))
        self.label_2.setText(_translate("Form", "color"))
        self.btnAddWidgetRight.setToolTip(_translate("Form", "Add DockPlanningHelperWidget right"))
        self.btnAddWidgetRight.setText(_translate("Form", "..."))
from pyqtgraph.widgets.ColorButton import ColorButton
