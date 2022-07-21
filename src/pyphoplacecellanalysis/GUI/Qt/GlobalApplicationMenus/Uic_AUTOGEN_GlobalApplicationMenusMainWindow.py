# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\GlobalApplicationMenus\GlobalApplicationMenusMainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_GlobalApplicationMenusMainWindow(object):
    def setupUi(self, GlobalApplicationMenusMainWindow):
        GlobalApplicationMenusMainWindow.setObjectName("GlobalApplicationMenusMainWindow")
        GlobalApplicationMenusMainWindow.resize(796, 600)
        self.centralwidget = QtWidgets.QWidget(GlobalApplicationMenusMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        GlobalApplicationMenusMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(GlobalApplicationMenusMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 796, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuAnalyses = QtWidgets.QMenu(self.menubar)
        self.menuAnalyses.setObjectName("menuAnalyses")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuWindows = QtWidgets.QMenu(self.menubar)
        self.menuWindows.setObjectName("menuWindows")
        self.menuVisualizations = QtWidgets.QMenu(self.menubar)
        self.menuVisualizations.setObjectName("menuVisualizations")
        GlobalApplicationMenusMainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(GlobalApplicationMenusMainWindow)
        self.statusbar.setObjectName("statusbar")
        GlobalApplicationMenusMainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(GlobalApplicationMenusMainWindow)
        self.toolBar.setObjectName("toolBar")
        GlobalApplicationMenusMainWindow.addToolBar(QtCore.Qt.BottomToolBarArea, self.toolBar)
        self.actionLoad = QtWidgets.QAction(GlobalApplicationMenusMainWindow)
        self.actionLoad.setObjectName("actionLoad")
        self.actionSave = QtWidgets.QAction(GlobalApplicationMenusMainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_As = QtWidgets.QAction(GlobalApplicationMenusMainWindow)
        self.actionSave_As.setObjectName("actionSave_As")
        self.actionSave_As_2 = QtWidgets.QAction(GlobalApplicationMenusMainWindow)
        self.actionSave_As_2.setObjectName("actionSave_As_2")
        self.actionQuit_App = QtWidgets.QAction(GlobalApplicationMenusMainWindow)
        self.actionQuit_App.setObjectName("actionQuit_App")
        self.actionAbout = QtWidgets.QAction(GlobalApplicationMenusMainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionQuit_Application = QtWidgets.QAction(GlobalApplicationMenusMainWindow)
        self.actionQuit_Application.setObjectName("actionQuit_Application")
        self.actionNew_3D_Raster_PyQtGraph = QtWidgets.QAction(GlobalApplicationMenusMainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/Icons/Icons/SpikeRaster3DIcon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionNew_3D_Raster_PyQtGraph.setIcon(icon)
        self.actionNew_3D_Raster_PyQtGraph.setObjectName("actionNew_3D_Raster_PyQtGraph")
        self.actionNew_Tuning_Curves_Explorer_ipcDataExplorer = QtWidgets.QAction(GlobalApplicationMenusMainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/Icons/Icons/TuningMapDataExplorerIconWithLabel.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionNew_Tuning_Curves_Explorer_ipcDataExplorer.setIcon(icon1)
        self.actionNew_Tuning_Curves_Explorer_ipcDataExplorer.setObjectName("actionNew_Tuning_Curves_Explorer_ipcDataExplorer")
        self.actionNew_Spikes_Behavior_Explorer_ipspikesDataExplorer = QtWidgets.QAction(GlobalApplicationMenusMainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/Icons/Icons/InteractivePlaceCellDataExplorerIconWithLabel.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionNew_Spikes_Behavior_Explorer_ipspikesDataExplorer.setIcon(icon2)
        self.actionNew_Spikes_Behavior_Explorer_ipspikesDataExplorer.setObjectName("actionNew_Spikes_Behavior_Explorer_ipspikesDataExplorer")
        self.actionNew_3D_Raster_Vedo = QtWidgets.QAction(GlobalApplicationMenusMainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/Icons/Icons/SpikeRaster3D_VedoIcon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionNew_3D_Raster_Vedo.setIcon(icon3)
        self.actionNew_3D_Raster_Vedo.setObjectName("actionNew_3D_Raster_Vedo")
        self.actionNew_2D_Raster_PyQtGraph = QtWidgets.QAction(GlobalApplicationMenusMainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/Icons/Icons/SpikeRaster2DIcon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionNew_2D_Raster_PyQtGraph.setIcon(icon4)
        self.actionNew_2D_Raster_PyQtGraph.setObjectName("actionNew_2D_Raster_PyQtGraph")
        self.menuFile.addAction(self.actionLoad)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit_Application)
        self.menuHelp.addAction(self.actionAbout)
        self.menuVisualizations.addAction(self.actionNew_2D_Raster_PyQtGraph)
        self.menuVisualizations.addAction(self.actionNew_3D_Raster_PyQtGraph)
        self.menuVisualizations.addAction(self.actionNew_3D_Raster_Vedo)
        self.menuVisualizations.addSeparator()
        self.menuVisualizations.addAction(self.actionNew_Tuning_Curves_Explorer_ipcDataExplorer)
        self.menuVisualizations.addAction(self.actionNew_Spikes_Behavior_Explorer_ipspikesDataExplorer)
        self.menuVisualizations.addSeparator()
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuAnalyses.menuAction())
        self.menubar.addAction(self.menuWindows.menuAction())
        self.menubar.addAction(self.menuVisualizations.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(GlobalApplicationMenusMainWindow)
        QtCore.QMetaObject.connectSlotsByName(GlobalApplicationMenusMainWindow)

    def retranslateUi(self, GlobalApplicationMenusMainWindow):
        _translate = QtCore.QCoreApplication.translate
        GlobalApplicationMenusMainWindow.setWindowTitle(_translate("GlobalApplicationMenusMainWindow", "GlobalApplicationMenusMainWindow"))
        self.menuFile.setTitle(_translate("GlobalApplicationMenusMainWindow", "File"))
        self.menuAnalyses.setTitle(_translate("GlobalApplicationMenusMainWindow", "Analyses"))
        self.menuHelp.setTitle(_translate("GlobalApplicationMenusMainWindow", "Help"))
        self.menuWindows.setTitle(_translate("GlobalApplicationMenusMainWindow", "Windows"))
        self.menuVisualizations.setTitle(_translate("GlobalApplicationMenusMainWindow", "Visualizations"))
        self.toolBar.setWindowTitle(_translate("GlobalApplicationMenusMainWindow", "toolBar"))
        self.actionLoad.setText(_translate("GlobalApplicationMenusMainWindow", "Load..."))
        self.actionSave.setText(_translate("GlobalApplicationMenusMainWindow", "Save"))
        self.actionSave_As.setText(_translate("GlobalApplicationMenusMainWindow", "Save As.."))
        self.actionSave_As_2.setText(_translate("GlobalApplicationMenusMainWindow", "Save As.."))
        self.actionQuit_App.setText(_translate("GlobalApplicationMenusMainWindow", "Quit App"))
        self.actionAbout.setText(_translate("GlobalApplicationMenusMainWindow", "About..."))
        self.actionQuit_Application.setText(_translate("GlobalApplicationMenusMainWindow", "Quit Application"))
        self.actionNew_3D_Raster_PyQtGraph.setText(_translate("GlobalApplicationMenusMainWindow", "New 3D Raster (PyQtGraph)"))
        self.actionNew_Tuning_Curves_Explorer_ipcDataExplorer.setText(_translate("GlobalApplicationMenusMainWindow", "New Tuning Curves Explorer (ipcDataExplorer)"))
        self.actionNew_Spikes_Behavior_Explorer_ipspikesDataExplorer.setText(_translate("GlobalApplicationMenusMainWindow", "New Spikes+Behavior Explorer (ipspikesDataExplorer)"))
        self.actionNew_3D_Raster_Vedo.setText(_translate("GlobalApplicationMenusMainWindow", "New 3D Raster (Vedo)"))
        self.actionNew_2D_Raster_PyQtGraph.setText(_translate("GlobalApplicationMenusMainWindow", "New 2D Raster (PyQtGraph)"))
import GuiResources_rc
import breeze_rc
