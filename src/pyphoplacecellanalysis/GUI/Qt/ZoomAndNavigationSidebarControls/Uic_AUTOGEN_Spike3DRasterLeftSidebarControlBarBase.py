# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\ZoomAndNavigationSidebarControls\Spike3DRasterLeftSidebarControlBarBase.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_leftSideToolbarWidget(object):
    def setupUi(self, leftSideToolbarWidget):
        leftSideToolbarWidget.setObjectName("leftSideToolbarWidget")
        leftSideToolbarWidget.resize(100, 840)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(leftSideToolbarWidget.sizePolicy().hasHeightForWidth())
        leftSideToolbarWidget.setSizePolicy(sizePolicy)
        leftSideToolbarWidget.setMinimumSize(QtCore.QSize(100, 0))
        leftSideToolbarWidget.setBaseSize(QtCore.QSize(100, 0))
        leftSideToolbarWidget.setStyleSheet("background-color: rgb(71, 58, 46);\n"
"border-color: rgb(207, 207, 207);\n"
"background-color: rgba(71, 65, 60, 180);\n"
"color: rgb(244, 244, 244);\n"
"border-color: rgb(0, 0, 0);")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(leftSideToolbarWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(2, 4, 2, 4)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_4 = QtWidgets.QLabel(leftSideToolbarWidget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.spinAnimationTimeStep = SpinBox(leftSideToolbarWidget)
        self.spinAnimationTimeStep.setSingleStep(0.01)
        self.spinAnimationTimeStep.setProperty("value", 0.1)
        self.spinAnimationTimeStep.setObjectName("spinAnimationTimeStep")
        self.verticalLayout.addWidget(self.spinAnimationTimeStep)
        self.label_5 = QtWidgets.QLabel(leftSideToolbarWidget)
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5)
        self.spinTemporalZoomFactor = SpinBox(leftSideToolbarWidget)
        self.spinTemporalZoomFactor.setMinimum(0.1)
        self.spinTemporalZoomFactor.setMaximum(100.0)
        self.spinTemporalZoomFactor.setSingleStep(0.1)
        self.spinTemporalZoomFactor.setProperty("value", 1.0)
        self.spinTemporalZoomFactor.setObjectName("spinTemporalZoomFactor")
        self.verticalLayout.addWidget(self.spinTemporalZoomFactor)
        self.label_6 = QtWidgets.QLabel(leftSideToolbarWidget)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.spinRenderWindowDuration = SpinBox(leftSideToolbarWidget)
        self.spinRenderWindowDuration.setReadOnly(True)
        self.spinRenderWindowDuration.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.spinRenderWindowDuration.setKeyboardTracking(False)
        self.spinRenderWindowDuration.setSingleStep(0.5)
        self.spinRenderWindowDuration.setProperty("value", 1.0)
        self.spinRenderWindowDuration.setObjectName("spinRenderWindowDuration")
        self.verticalLayout.addWidget(self.spinRenderWindowDuration)
        self.verticalScrollBar = QtWidgets.QScrollBar(leftSideToolbarWidget)
        self.verticalScrollBar.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.verticalScrollBar.sizePolicy().hasHeightForWidth())
        self.verticalScrollBar.setSizePolicy(sizePolicy)
        self.verticalScrollBar.setOrientation(QtCore.Qt.Vertical)
        self.verticalScrollBar.setObjectName("verticalScrollBar")
        self.verticalLayout.addWidget(self.verticalScrollBar)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem)
        self.verticalSliderZoom = QtWidgets.QSlider(leftSideToolbarWidget)
        self.verticalSliderZoom.setEnabled(False)
        self.verticalSliderZoom.setMinimum(0)
        self.verticalSliderZoom.setMaximum(1000)
        self.verticalSliderZoom.setProperty("value", 500)
        self.verticalSliderZoom.setOrientation(QtCore.Qt.Vertical)
        self.verticalSliderZoom.setInvertedAppearance(False)
        self.verticalSliderZoom.setInvertedControls(False)
        self.verticalSliderZoom.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.verticalSliderZoom.setTickInterval(100)
        self.verticalSliderZoom.setObjectName("verticalSliderZoom")
        self.verticalLayout.addWidget(self.verticalSliderZoom, 0, QtCore.Qt.AlignHCenter)
        self.btnToggleCollapseExpand = QtWidgets.QToolButton(leftSideToolbarWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnToggleCollapseExpand.sizePolicy().hasHeightForWidth())
        self.btnToggleCollapseExpand.setSizePolicy(sizePolicy)
        self.btnToggleCollapseExpand.setText("")
        self.btnToggleCollapseExpand.setCheckable(True)
        self.btnToggleCollapseExpand.setChecked(False)
        self.btnToggleCollapseExpand.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.btnToggleCollapseExpand.setArrowType(QtCore.Qt.LeftArrow)
        self.btnToggleCollapseExpand.setObjectName("btnToggleCollapseExpand")
        self.verticalLayout.addWidget(self.btnToggleCollapseExpand)
        self.verticalLayout.setStretch(8, 1)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(leftSideToolbarWidget)
        QtCore.QMetaObject.connectSlotsByName(leftSideToolbarWidget)

    def retranslateUi(self, leftSideToolbarWidget):
        _translate = QtCore.QCoreApplication.translate
        self.label_4.setText(_translate("leftSideToolbarWidget", "anim. time step"))
        self.spinAnimationTimeStep.setSuffix(_translate("leftSideToolbarWidget", "s"))
        self.label_5.setText(_translate("leftSideToolbarWidget", "zoom factor"))
        self.spinTemporalZoomFactor.setSuffix(_translate("leftSideToolbarWidget", "x"))
        self.label_6.setText(_translate("leftSideToolbarWidget", "win. dur."))
        self.spinRenderWindowDuration.setSuffix(_translate("leftSideToolbarWidget", "s"))
        self.btnToggleCollapseExpand.setToolTip(_translate("leftSideToolbarWidget", "Collapse Sidebar"))
from pyphoplacecellanalysis.External.pyqtgraph.widgets.SpinBox import SpinBox
