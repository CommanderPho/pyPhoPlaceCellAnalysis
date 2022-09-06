# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\GlobalApplicationMenus\LocalMenus_AddRenderable.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets



class Ui_LocalMenus_AddRenderable(object):
    def setupUi(self, LocalMenus_AddRenderable):
        LocalMenus_AddRenderable.setObjectName("LocalMenus_AddRenderable")
        LocalMenus_AddRenderable.resize(703, 126)
        LocalMenus_AddRenderable.setWindowTitle("LocalMenus")
        self.centralwidget = QtWidgets.QWidget(LocalMenus_AddRenderable)
        self.centralwidget.setObjectName("centralwidget")
        LocalMenus_AddRenderable.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(LocalMenus_AddRenderable)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 703, 22))
        self.menubar.setObjectName("menubar")
        self.menuAdd_Renderable = QtWidgets.QMenu(self.menubar)
        self.menuAdd_Renderable.setTitle("Add Renderable")
        self.menuAdd_Renderable.setObjectName("menuAdd_Renderable")
        self.menuAddRenderable_Time_Curves = QtWidgets.QMenu(self.menuAdd_Renderable)
        self.menuAddRenderable_Time_Curves.setTitle("Add Time Curves...")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/Render/Icons/actions/chart-up-color.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.menuAddRenderable_Time_Curves.setIcon(icon)
        self.menuAddRenderable_Time_Curves.setObjectName("menuAddRenderable_Time_Curves")
        self.menuAddRenderable_Time_Intervals = QtWidgets.QMenu(self.menuAdd_Renderable)
        self.menuAddRenderable_Time_Intervals.setTitle("Add Time Intervals...")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/Render/Icons/actions/spectrum-emission.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.menuAddRenderable_Time_Intervals.setIcon(icon1)
        self.menuAddRenderable_Time_Intervals.setObjectName("menuAddRenderable_Time_Intervals")
        LocalMenus_AddRenderable.setMenuBar(self.menubar)
        self.actionAddTimeCurves_Position = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeCurves_Position.setText("Position")
        self.actionAddTimeCurves_Position.setObjectName("actionAddTimeCurves_Position")
        self.actionAddTimeCurves_Custom = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeCurves_Custom.setText("Custom...")
        self.actionAddTimeCurves_Custom.setObjectName("actionAddTimeCurves_Custom")
        self.actionAddCustomRenderable = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddCustomRenderable.setText("Add Custom...")
        self.actionAddCustomRenderable.setObjectName("actionAddCustomRenderable")
        self.actionAddTimeIntervals_PBEs = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeIntervals_PBEs.setText("PBEs")
        self.actionAddTimeIntervals_PBEs.setObjectName("actionAddTimeIntervals_PBEs")
        self.actionAddTimeIntervals_Session_Epochs = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeIntervals_Session_Epochs.setText("Session Epochs")
        self.actionAddTimeIntervals_Session_Epochs.setObjectName("actionAddTimeIntervals_Session_Epochs")
        self.actionAddTimeIntervals_Laps = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeIntervals_Laps.setText("Laps")
        self.actionAddTimeIntervals_Laps.setObjectName("actionAddTimeIntervals_Laps")
        self.actionAddTimeIntervals_Custom = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeIntervals_Custom.setText("Custom...")
        self.actionAddTimeIntervals_Custom.setObjectName("actionAddTimeIntervals_Custom")
        self.actionAddTimeCurves_Random = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeCurves_Random.setText("Random")
        self.actionAddTimeCurves_Random.setObjectName("actionAddTimeCurves_Random")
        self.menuAddRenderable_Time_Curves.addAction(self.actionAddTimeCurves_Position)
        self.menuAddRenderable_Time_Curves.addAction(self.actionAddTimeCurves_Random)
        self.menuAddRenderable_Time_Curves.addAction(self.actionAddTimeCurves_Custom)
        self.menuAddRenderable_Time_Intervals.addAction(self.actionAddTimeIntervals_PBEs)
        self.menuAddRenderable_Time_Intervals.addAction(self.actionAddTimeIntervals_Session_Epochs)
        self.menuAddRenderable_Time_Intervals.addAction(self.actionAddTimeIntervals_Laps)
        self.menuAddRenderable_Time_Intervals.addAction(self.actionAddTimeIntervals_Custom)
        self.menuAdd_Renderable.addAction(self.menuAddRenderable_Time_Curves.menuAction())
        self.menuAdd_Renderable.addAction(self.menuAddRenderable_Time_Intervals.menuAction())
        self.menuAdd_Renderable.addAction(self.actionAddCustomRenderable)
        self.menubar.addAction(self.menuAdd_Renderable.menuAction())

        self.retranslateUi(LocalMenus_AddRenderable)
        QtCore.QMetaObject.connectSlotsByName(LocalMenus_AddRenderable)

    def retranslateUi(self, LocalMenus_AddRenderable):
        pass
# import ActionIcons_rc
# import GuiResources_rc
# import breeze_rc

## CUSTOM: DO NOT OVERWRITE:
import pyphoplacecellanalysis.Resources.ActionIcons
import pyphoplacecellanalysis.Resources.GuiResources