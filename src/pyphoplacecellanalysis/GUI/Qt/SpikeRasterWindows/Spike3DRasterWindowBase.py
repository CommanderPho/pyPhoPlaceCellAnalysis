# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\SpikeRasterWindows\Spike3DRasterWindowBase.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_RootWidget(object):
    def setupUi(self, RootWidget):
        RootWidget.setObjectName("RootWidget")
        RootWidget.resize(1663, 1112)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/Icons/Icons/SpikeRaster3DIcon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        RootWidget.setWindowIcon(icon)
        RootWidget.setStyleSheet("background-color: rgb(71, 58, 46);\n"
"border-color: rgb(207, 207, 207);\n"
"background-color: rgba(71, 65, 60, 180);\n"
"color: rgb(244, 244, 244);\n"
"border-color: rgb(0, 0, 0);")
        self.gridLayout = QtWidgets.QGridLayout(RootWidget)
        self.gridLayout.setContentsMargins(0, 2, 0, 0)
        self.gridLayout.setVerticalSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.bottomPlaybackControlBarWidget = Spike3DRasterBottomPlaybackControlBar(RootWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bottomPlaybackControlBarWidget.sizePolicy().hasHeightForWidth())
        self.bottomPlaybackControlBarWidget.setSizePolicy(sizePolicy)
        self.bottomPlaybackControlBarWidget.setMinimumSize(QtCore.QSize(0, 70))
        self.bottomPlaybackControlBarWidget.setMaximumSize(QtCore.QSize(16777215, 70))
        self.bottomPlaybackControlBarWidget.setObjectName("bottomPlaybackControlBarWidget")
        self.gridLayout.addWidget(self.bottomPlaybackControlBarWidget, 2, 0, 1, 3)
        self.leftSideToolbarWidget = Spike3DRasterLeftSidebarControlBar(RootWidget)
        self.leftSideToolbarWidget.setMinimumSize(QtCore.QSize(52, 1040))
        self.leftSideToolbarWidget.setMaximumSize(QtCore.QSize(52, 16777215))
        self.leftSideToolbarWidget.setBaseSize(QtCore.QSize(52, 0))
        self.leftSideToolbarWidget.setObjectName("leftSideToolbarWidget")
        self.gridLayout.addWidget(self.leftSideToolbarWidget, 0, 0, 2, 1)
        self.mainContentFrame = QtWidgets.QFrame(RootWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mainContentFrame.sizePolicy().hasHeightForWidth())
        self.mainContentFrame.setSizePolicy(sizePolicy)
        self.mainContentFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.mainContentFrame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.mainContentFrame.setObjectName("mainContentFrame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.mainContentFrame)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 2)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.splitter = QtWidgets.QSplitter(self.mainContentFrame)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setHandleWidth(10)
        self.splitter.setObjectName("splitter")
        self.mainSpike3DRasterWidget = QtWidgets.QWidget(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mainSpike3DRasterWidget.sizePolicy().hasHeightForWidth())
        self.mainSpike3DRasterWidget.setSizePolicy(sizePolicy)
        self.mainSpike3DRasterWidget.setMinimumSize(QtCore.QSize(400, 400))
        self.mainSpike3DRasterWidget.setBaseSize(QtCore.QSize(0, 400))
        self.mainSpike3DRasterWidget.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.mainSpike3DRasterWidget.setObjectName("mainSpike3DRasterWidget")
        self.secondarySpikeRasterControlWidget = QtWidgets.QWidget(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.secondarySpikeRasterControlWidget.sizePolicy().hasHeightForWidth())
        self.secondarySpikeRasterControlWidget.setSizePolicy(sizePolicy)
        self.secondarySpikeRasterControlWidget.setMinimumSize(QtCore.QSize(0, 200))
        self.secondarySpikeRasterControlWidget.setBaseSize(QtCore.QSize(0, 200))
        self.secondarySpikeRasterControlWidget.setObjectName("secondarySpikeRasterControlWidget")
        self.gridLayout_2.addWidget(self.splitter, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.mainContentFrame, 0, 1, 1, 2)
        self.gridLayout.setRowStretch(0, 1)

        self.retranslateUi(RootWidget)
        QtCore.QMetaObject.connectSlotsByName(RootWidget)

    def retranslateUi(self, RootWidget):
        _translate = QtCore.QCoreApplication.translate
        RootWidget.setWindowTitle(_translate("RootWidget", "Spike 3D Raster Window"))
from pyphoplacecellanalysis.GUI.Qt.PlaybackControls.Spike3DRasterBottomPlaybackControlBarWidget import Spike3DRasterBottomPlaybackControlBar
from pyphoplacecellanalysis.GUI.Qt.ZoomAndNavigationSidebarControls.Spike3DRasterLeftSidebarControlBarWidget import Spike3DRasterLeftSidebarControlBar

## CUSTOM: DO NOT OVERWRITE:
import pyphoplacecellanalysis.Resources.GuiResources





if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    rootForm = QtWidgets.QWidget()
    ui = Ui_RootWidget()
    ui.setupUi(rootForm)
    rootForm.show()
    sys.exit(app.exec_())


