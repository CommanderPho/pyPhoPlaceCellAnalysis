# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\ThinButtonBar\ExtraButtonsWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ExtraButtonsWidget(object):
    def setupUi(self, ExtraButtonsWidget):
        ExtraButtonsWidget.setObjectName("ExtraButtonsWidget")
        ExtraButtonsWidget.resize(692, 21)
        ExtraButtonsWidget.setWindowTitle("ExtraButtonsWidget")
        self.gridLayout = QtWidgets.QGridLayout(ExtraButtonsWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.thinButtonBarWidget = QtWidgets.QFrame(ExtraButtonsWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.thinButtonBarWidget.sizePolicy().hasHeightForWidth())
        self.thinButtonBarWidget.setSizePolicy(sizePolicy)
        self.thinButtonBarWidget.setMinimumSize(QtCore.QSize(120, 21))
        self.thinButtonBarWidget.setMaximumSize(QtCore.QSize(16777215, 21))
        self.thinButtonBarWidget.setBaseSize(QtCore.QSize(120, 21))
        self.thinButtonBarWidget.setObjectName("thinButtonBarWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.thinButtonBarWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.horizontalLayout_ButtonContainer = QtWidgets.QHBoxLayout()
        self.horizontalLayout_ButtonContainer.setContentsMargins(0, -1, -1, -1)
        self.horizontalLayout_ButtonContainer.setSpacing(2)
        self.horizontalLayout_ButtonContainer.setObjectName("horizontalLayout_ButtonContainer")
        self.btnRefresh = QtWidgets.QToolButton(self.thinButtonBarWidget)
        self.btnRefresh.setToolTip("Refresh")
        self.btnRefresh.setStatusTip("Refresh")
        self.btnRefresh.setText("Refresh")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/png/gui/icons/view-refresh.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnRefresh.setIcon(icon)
        self.btnRefresh.setAutoRaise(True)
        self.btnRefresh.setObjectName("btnRefresh")
        self.horizontalLayout_ButtonContainer.addWidget(self.btnRefresh)
        self.toolButton_Clipboard = QtWidgets.QToolButton(self.thinButtonBarWidget)
        self.toolButton_Clipboard.setToolTip("Clipboard")
        self.toolButton_Clipboard.setText("Clipboard")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/png/gui/icons/clipboard.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_Clipboard.setIcon(icon1)
        self.toolButton_Clipboard.setObjectName("toolButton_Clipboard")
        self.horizontalLayout_ButtonContainer.addWidget(self.toolButton_Clipboard)
        self.btnCopySelectedEpochs = QtWidgets.QToolButton(self.thinButtonBarWidget)
        self.btnCopySelectedEpochs.setEnabled(True)
        self.btnCopySelectedEpochs.setToolTip("Copy Selections")
        self.btnCopySelectedEpochs.setStatusTip("CopySelections")
        self.btnCopySelectedEpochs.setText("Copy Selections")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/png/gui/icons/edit-copy.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnCopySelectedEpochs.setIcon(icon2)
        self.btnCopySelectedEpochs.setAutoRaise(True)
        self.btnCopySelectedEpochs.setObjectName("btnCopySelectedEpochs")
        self.horizontalLayout_ButtonContainer.addWidget(self.btnCopySelectedEpochs)
        self.toolButton_Printer = QtWidgets.QToolButton(self.thinButtonBarWidget)
        self.toolButton_Printer.setToolTip("Printer")
        self.toolButton_Printer.setText("Printer")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/png/gui/icons/document-print.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_Printer.setIcon(icon3)
        self.toolButton_Printer.setAutoRaise(True)
        self.toolButton_Printer.setObjectName("toolButton_Printer")
        self.horizontalLayout_ButtonContainer.addWidget(self.toolButton_Printer)
        spacerItem1 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_ButtonContainer.addItem(spacerItem1)
        self.toolButton_Brush = QtWidgets.QToolButton(self.thinButtonBarWidget)
        self.toolButton_Brush.setToolTip("Brush")
        self.toolButton_Brush.setText("Brush")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/png/gui/icons/draw-brush.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_Brush.setIcon(icon4)
        self.toolButton_Brush.setCheckable(True)
        self.toolButton_Brush.setAutoExclusive(True)
        self.toolButton_Brush.setObjectName("toolButton_Brush")
        self.buttonGroup_ActiveTool = QtWidgets.QButtonGroup(ExtraButtonsWidget)
        self.buttonGroup_ActiveTool.setObjectName("buttonGroup_ActiveTool")
        self.buttonGroup_ActiveTool.addButton(self.toolButton_Brush)
        self.horizontalLayout_ButtonContainer.addWidget(self.toolButton_Brush)
        self.toolButton_Pencil = QtWidgets.QToolButton(self.thinButtonBarWidget)
        self.toolButton_Pencil.setToolTip("Pencil")
        self.toolButton_Pencil.setText("Pencil")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/png/gui/icons/draw-pencil.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_Pencil.setIcon(icon5)
        self.toolButton_Pencil.setCheckable(True)
        self.toolButton_Pencil.setAutoExclusive(True)
        self.toolButton_Pencil.setObjectName("toolButton_Pencil")
        self.buttonGroup_ActiveTool.addButton(self.toolButton_Pencil)
        self.horizontalLayout_ButtonContainer.addWidget(self.toolButton_Pencil)
        self.toolButton_Eraser = QtWidgets.QToolButton(self.thinButtonBarWidget)
        self.toolButton_Eraser.setToolTip("Eraser")
        self.toolButton_Eraser.setText("Eraser")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/png/gui/icons/draw-rubber.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_Eraser.setIcon(icon6)
        self.toolButton_Eraser.setCheckable(True)
        self.toolButton_Eraser.setAutoExclusive(True)
        self.toolButton_Eraser.setObjectName("toolButton_Eraser")
        self.buttonGroup_ActiveTool.addButton(self.toolButton_Eraser)
        self.horizontalLayout_ButtonContainer.addWidget(self.toolButton_Eraser)
        self.horizontalLayout.addLayout(self.horizontalLayout_ButtonContainer)
        self.horizontalLayout.setStretch(0, 1)
        self.gridLayout.addWidget(self.thinButtonBarWidget, 0, 0, 1, 1)

        self.retranslateUi(ExtraButtonsWidget)
        QtCore.QMetaObject.connectSlotsByName(ExtraButtonsWidget)

    def retranslateUi(self, ExtraButtonsWidget):
        pass
from pyphoplacecellanalysis.Resources import silx_resources_rc