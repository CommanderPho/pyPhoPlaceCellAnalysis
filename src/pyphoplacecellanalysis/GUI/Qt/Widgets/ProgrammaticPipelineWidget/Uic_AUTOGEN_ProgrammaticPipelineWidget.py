# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\ProgrammaticPipelineWidget\ProgrammaticPipelineWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(553, 554)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/Icons/Icons/ProcessIcon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form.setWindowIcon(icon)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 12, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 3, 0, 1, 1)
        self.contextSelectorWidget = IdentifyingContextSelectorWidget(Form)
        self.contextSelectorWidget.setMinimumSize(QtCore.QSize(0, 61))
        self.contextSelectorWidget.setStyleSheet("border-color: rgb(170, 0, 127);")
        self.contextSelectorWidget.setObjectName("contextSelectorWidget")
        self.gridLayout.addWidget(self.contextSelectorWidget, 0, 0, 1, 1)
        self.widget = QtWidgets.QWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setMinimumSize(QtCore.QSize(0, 0))
        self.widget.setBaseSize(QtCore.QSize(0, 150))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.VisualizationsGroup = QtWidgets.QGroupBox(self.widget)
        self.VisualizationsGroup.setEnabled(True)
        self.VisualizationsGroup.setFlat(True)
        self.VisualizationsGroup.setObjectName("VisualizationsGroup")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.VisualizationsGroup)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.toolButton_4 = QtWidgets.QToolButton(self.VisualizationsGroup)
        self.toolButton_4.setEnabled(False)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/Icons/Icons/InteractivePlaceCellDataExplorerIconWithLabel.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_4.setIcon(icon1)
        self.toolButton_4.setIconSize(QtCore.QSize(30, 30))
        self.toolButton_4.setObjectName("toolButton_4")
        self.buttonGroup_Outer_Vis = QtWidgets.QButtonGroup(Form)
        self.buttonGroup_Outer_Vis.setObjectName("buttonGroup_Outer_Vis")
        self.buttonGroup_Outer_Vis.setExclusive(False)
        self.buttonGroup_Outer_Vis.addButton(self.toolButton_4)
        self.gridLayout_2.addWidget(self.toolButton_4, 0, 3, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 0, 8, 1, 1)
        self.toolButton_3 = QtWidgets.QToolButton(self.VisualizationsGroup)
        self.toolButton_3.setEnabled(False)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/Icons/Icons/TuningMapDataExplorerIconWithLabel.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_3.setIcon(icon2)
        self.toolButton_3.setIconSize(QtCore.QSize(30, 30))
        self.toolButton_3.setObjectName("toolButton_3")
        self.buttonGroup_Outer_Vis.addButton(self.toolButton_3)
        self.gridLayout_2.addWidget(self.toolButton_3, 0, 2, 1, 1)
        self.toolButton_2 = QtWidgets.QToolButton(self.VisualizationsGroup)
        self.toolButton_2.setEnabled(False)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/Icons/Icons/SpikeRaster3D_VedoIcon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_2.setIcon(icon3)
        self.toolButton_2.setIconSize(QtCore.QSize(30, 30))
        self.toolButton_2.setObjectName("toolButton_2")
        self.buttonGroup_Outer_Vis.addButton(self.toolButton_2)
        self.gridLayout_2.addWidget(self.toolButton_2, 0, 1, 1, 1)
        self.toolButton_10 = QtWidgets.QToolButton(self.VisualizationsGroup)
        self.toolButton_10.setEnabled(False)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/Render/Icons/Icon/Heatmap.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_10.setIcon(icon4)
        self.toolButton_10.setIconSize(QtCore.QSize(30, 30))
        self.toolButton_10.setObjectName("toolButton_10")
        self.gridLayout_2.addWidget(self.toolButton_10, 0, 4, 1, 1)
        self.toolButton = QtWidgets.QToolButton(self.VisualizationsGroup)
        self.toolButton.setEnabled(False)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/Icons/Icons/SpikeRaster3DIcon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton.setIcon(icon5)
        self.toolButton.setIconSize(QtCore.QSize(30, 30))
        self.toolButton.setObjectName("toolButton")
        self.buttonGroup_Outer_Vis.addButton(self.toolButton)
        self.gridLayout_2.addWidget(self.toolButton, 0, 0, 1, 1)
        self.btnProgrammaticDisplayConfig_2 = QtWidgets.QPushButton(self.VisualizationsGroup)
        self.btnProgrammaticDisplayConfig_2.setMinimumSize(QtCore.QSize(0, 36))
        self.btnProgrammaticDisplayConfig_2.setBaseSize(QtCore.QSize(0, 36))
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/Render/Icons/actions/chart--pencil.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnProgrammaticDisplayConfig_2.setIcon(icon6)
        self.btnProgrammaticDisplayConfig_2.setIconSize(QtCore.QSize(30, 30))
        self.btnProgrammaticDisplayConfig_2.setFlat(False)
        self.btnProgrammaticDisplayConfig_2.setObjectName("btnProgrammaticDisplayConfig_2")
        self.gridLayout_2.addWidget(self.btnProgrammaticDisplayConfig_2, 0, 9, 1, 1)
        self.verticalLayout.addWidget(self.VisualizationsGroup)
        self.toolBox = QtWidgets.QToolBox(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.toolBox.sizePolicy().hasHeightForWidth())
        self.toolBox.setSizePolicy(sizePolicy)
        self.toolBox.setMinimumSize(QtCore.QSize(0, 200))
        self.toolBox.setBaseSize(QtCore.QSize(0, 200))
        self.toolBox.setFrameShape(QtWidgets.QFrame.Box)
        self.toolBox.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.toolBox.setMidLineWidth(1)
        self.toolBox.setObjectName("toolBox")
        self.tooltab_Display = QtWidgets.QWidget()
        self.tooltab_Display.setGeometry(QtCore.QRect(0, 0, 511, 260))
        self.tooltab_Display.setObjectName("tooltab_Display")
        self.tooltab_Display_Layout = QtWidgets.QHBoxLayout(self.tooltab_Display)
        self.tooltab_Display_Layout.setObjectName("tooltab_Display_Layout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(4, -1, 4, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.btnProgrammaticDisplayConfig = QtWidgets.QPushButton(self.tooltab_Display)
        self.btnProgrammaticDisplayConfig.setIcon(icon6)
        self.btnProgrammaticDisplayConfig.setFlat(False)
        self.btnProgrammaticDisplayConfig.setObjectName("btnProgrammaticDisplayConfig")
        self.verticalLayout_2.addWidget(self.btnProgrammaticDisplayConfig, 0, QtCore.Qt.AlignTop)
        self.btnProgrammaticDisplay = QtWidgets.QPushButton(self.tooltab_Display)
        self.btnProgrammaticDisplay.setObjectName("btnProgrammaticDisplay")
        self.verticalLayout_2.addWidget(self.btnProgrammaticDisplay, 0, QtCore.Qt.AlignTop)
        self.tooltab_Display_Layout.addLayout(self.verticalLayout_2)
        self.tooltab_ProgrammaticDisplayLayout = QtWidgets.QVBoxLayout()
        self.tooltab_ProgrammaticDisplayLayout.setObjectName("tooltab_ProgrammaticDisplayLayout")
        self.line = QtWidgets.QFrame(self.tooltab_Display)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.tooltab_ProgrammaticDisplayLayout.addWidget(self.line)
        self.tooltab_Display_Layout.addLayout(self.tooltab_ProgrammaticDisplayLayout)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/Render/Icons/actions/category-item.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolBox.addItem(self.tooltab_Display, icon7, "")
        self.tooltab_Visualization = QtWidgets.QWidget()
        self.tooltab_Visualization.setGeometry(QtCore.QRect(0, 0, 405, 40))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tooltab_Visualization.sizePolicy().hasHeightForWidth())
        self.tooltab_Visualization.setSizePolicy(sizePolicy)
        self.tooltab_Visualization.setObjectName("tooltab_Visualization")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tooltab_Visualization)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.VisualizationsContent = QtWidgets.QFrame(self.tooltab_Visualization)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.VisualizationsContent.sizePolicy().hasHeightForWidth())
        self.VisualizationsContent.setSizePolicy(sizePolicy)
        self.VisualizationsContent.setObjectName("VisualizationsContent")
        self.tooltab_Visualization_Layout = QtWidgets.QHBoxLayout(self.VisualizationsContent)
        self.tooltab_Visualization_Layout.setContentsMargins(2, 2, 2, 2)
        self.tooltab_Visualization_Layout.setObjectName("tooltab_Visualization_Layout")
        self.toolButton_9 = QtWidgets.QToolButton(self.VisualizationsContent)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/Icons/Icons/SpikeRaster2DIcon.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_9.setIcon(icon8)
        self.toolButton_9.setIconSize(QtCore.QSize(30, 30))
        self.toolButton_9.setObjectName("toolButton_9")
        self.buttonGroup_Vis_Raster = QtWidgets.QButtonGroup(Form)
        self.buttonGroup_Vis_Raster.setObjectName("buttonGroup_Vis_Raster")
        self.buttonGroup_Vis_Raster.setExclusive(False)
        self.buttonGroup_Vis_Raster.addButton(self.toolButton_9)
        self.tooltab_Visualization_Layout.addWidget(self.toolButton_9)
        self.toolButton_5 = QtWidgets.QToolButton(self.VisualizationsContent)
        self.toolButton_5.setIcon(icon5)
        self.toolButton_5.setIconSize(QtCore.QSize(30, 30))
        self.toolButton_5.setObjectName("toolButton_5")
        self.buttonGroup_Vis_Raster.addButton(self.toolButton_5)
        self.tooltab_Visualization_Layout.addWidget(self.toolButton_5)
        self.toolButton_6 = QtWidgets.QToolButton(self.VisualizationsContent)
        self.toolButton_6.setIcon(icon3)
        self.toolButton_6.setIconSize(QtCore.QSize(30, 30))
        self.toolButton_6.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.toolButton_6.setObjectName("toolButton_6")
        self.buttonGroup_Vis_Raster.addButton(self.toolButton_6)
        self.tooltab_Visualization_Layout.addWidget(self.toolButton_6)
        self.toolButton_8 = QtWidgets.QToolButton(self.VisualizationsContent)
        self.toolButton_8.setIcon(icon2)
        self.toolButton_8.setIconSize(QtCore.QSize(30, 30))
        self.toolButton_8.setObjectName("toolButton_8")
        self.buttonGroup_Vis_Raster.addButton(self.toolButton_8)
        self.tooltab_Visualization_Layout.addWidget(self.toolButton_8)
        self.toolButton_7 = QtWidgets.QToolButton(self.VisualizationsContent)
        self.toolButton_7.setIcon(icon1)
        self.toolButton_7.setIconSize(QtCore.QSize(30, 30))
        self.toolButton_7.setObjectName("toolButton_7")
        self.buttonGroup_Vis_Raster.addButton(self.toolButton_7)
        self.tooltab_Visualization_Layout.addWidget(self.toolButton_7)
        spacerItem2 = QtWidgets.QSpacerItem(8, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.tooltab_Visualization_Layout.addItem(spacerItem2)
        self.toolButton_11 = QtWidgets.QToolButton(self.VisualizationsContent)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/Render/Icons/Icon/HeatmapUgly.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_11.setIcon(icon9)
        self.toolButton_11.setIconSize(QtCore.QSize(30, 30))
        self.toolButton_11.setObjectName("toolButton_11")
        self.buttonGroup_Vis_Maps = QtWidgets.QButtonGroup(Form)
        self.buttonGroup_Vis_Maps.setObjectName("buttonGroup_Vis_Maps")
        self.buttonGroup_Vis_Maps.setExclusive(False)
        self.buttonGroup_Vis_Maps.addButton(self.toolButton_11)
        self.tooltab_Visualization_Layout.addWidget(self.toolButton_11)
        self.toolButton_14 = QtWidgets.QToolButton(self.VisualizationsContent)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(":/Render/Icons/Icon/Occupancy.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_14.setIcon(icon10)
        self.toolButton_14.setIconSize(QtCore.QSize(30, 30))
        self.toolButton_14.setObjectName("toolButton_14")
        self.buttonGroup_Vis_Maps.addButton(self.toolButton_14)
        self.tooltab_Visualization_Layout.addWidget(self.toolButton_14)
        self.toolButton_13 = QtWidgets.QToolButton(self.VisualizationsContent)
        self.toolButton_13.setIcon(icon4)
        self.toolButton_13.setIconSize(QtCore.QSize(30, 30))
        self.toolButton_13.setObjectName("toolButton_13")
        self.buttonGroup_Vis_Maps.addButton(self.toolButton_13)
        self.tooltab_Visualization_Layout.addWidget(self.toolButton_13)
        self.toolButton_15 = QtWidgets.QToolButton(self.VisualizationsContent)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap(":/Render/Icons/Icon/Decode.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_15.setIcon(icon11)
        self.toolButton_15.setIconSize(QtCore.QSize(30, 30))
        self.toolButton_15.setObjectName("toolButton_15")
        self.buttonGroup_Vis_Maps.addButton(self.toolButton_15)
        self.tooltab_Visualization_Layout.addWidget(self.toolButton_15)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.tooltab_Visualization_Layout.addItem(spacerItem3)
        self.gridLayout_3.addWidget(self.VisualizationsContent, 0, 0, 1, 1)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap(":/Render/Icons/actions/categories.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolBox.addItem(self.tooltab_Visualization, icon12, "")
        self.tooltab_Utilities = QtWidgets.QWidget()
        self.tooltab_Utilities.setGeometry(QtCore.QRect(0, 0, 98, 28))
        self.tooltab_Utilities.setObjectName("tooltab_Utilities")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tooltab_Utilities)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.UtilitiesContent = QtWidgets.QFrame(self.tooltab_Utilities)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.UtilitiesContent.sizePolicy().hasHeightForWidth())
        self.UtilitiesContent.setSizePolicy(sizePolicy)
        self.UtilitiesContent.setObjectName("UtilitiesContent")
        self.tooltab_Utilities_Layout = QtWidgets.QHBoxLayout(self.UtilitiesContent)
        self.tooltab_Utilities_Layout.setContentsMargins(2, 2, 2, 2)
        self.tooltab_Utilities_Layout.setObjectName("tooltab_Utilities_Layout")
        self.toolButton_12 = QtWidgets.QToolButton(self.UtilitiesContent)
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap(":/Graphics/Icons/graphics/Pho Symbol.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_12.setIcon(icon13)
        self.toolButton_12.setIconSize(QtCore.QSize(30, 30))
        self.toolButton_12.setObjectName("toolButton_12")
        self.tooltab_Utilities_Layout.addWidget(self.toolButton_12)
        spacerItem4 = QtWidgets.QSpacerItem(188, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.tooltab_Utilities_Layout.addItem(spacerItem4)
        self.gridLayout_4.addWidget(self.UtilitiesContent, 0, 0, 1, 1)
        self.toolBox.addItem(self.tooltab_Utilities, icon13, "")
        self.verticalLayout.addWidget(self.toolBox)
        self.gridLayout.addWidget(self.widget, 1, 0, 2, 1)
        self.actionLoad = QtWidgets.QAction(Form)
        self.actionLoad.setObjectName("actionLoad")
        self.actionSave = QtWidgets.QAction(Form)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_As = QtWidgets.QAction(Form)
        self.actionSave_As.setObjectName("actionSave_As")
        self.actionQuit_App = QtWidgets.QAction(Form)
        self.actionQuit_App.setObjectName("actionQuit_App")
        self.actionAbout = QtWidgets.QAction(Form)
        self.actionAbout.setObjectName("actionAbout")
        self.actionQuit_Application = QtWidgets.QAction(Form)
        self.actionQuit_Application.setObjectName("actionQuit_Application")
        self.actionNew_3D_Raster_PyQtGraph = QtWidgets.QAction(Form)
        self.actionNew_3D_Raster_PyQtGraph.setIcon(icon5)
        self.actionNew_3D_Raster_PyQtGraph.setObjectName("actionNew_3D_Raster_PyQtGraph")
        self.actionNew_Tuning_Curves_Explorer_ipcDataExplorer = QtWidgets.QAction(Form)
        self.actionNew_Tuning_Curves_Explorer_ipcDataExplorer.setIcon(icon2)
        self.actionNew_Tuning_Curves_Explorer_ipcDataExplorer.setObjectName("actionNew_Tuning_Curves_Explorer_ipcDataExplorer")
        self.actionNew_Spikes_Behavior_Explorer_ipspikesDataExplorer = QtWidgets.QAction(Form)
        self.actionNew_Spikes_Behavior_Explorer_ipspikesDataExplorer.setIcon(icon1)
        self.actionNew_Spikes_Behavior_Explorer_ipspikesDataExplorer.setObjectName("actionNew_Spikes_Behavior_Explorer_ipspikesDataExplorer")
        self.actionNew_3D_Raster_Vedo = QtWidgets.QAction(Form)
        self.actionNew_3D_Raster_Vedo.setIcon(icon3)
        self.actionNew_3D_Raster_Vedo.setObjectName("actionNew_3D_Raster_Vedo")
        self.actionNew_2D_Raster_PyQtGraph = QtWidgets.QAction(Form)
        self.actionNew_2D_Raster_PyQtGraph.setIcon(icon8)
        self.actionNew_2D_Raster_PyQtGraph.setObjectName("actionNew_2D_Raster_PyQtGraph")
        self.actionAddTimeCurves_Position = QtWidgets.QAction(Form)
        self.actionAddTimeCurves_Position.setObjectName("actionAddTimeCurves_Position")
        self.actionAddTimeCurves_Custom = QtWidgets.QAction(Form)
        self.actionAddTimeCurves_Custom.setObjectName("actionAddTimeCurves_Custom")
        self.actionAdd_Custom = QtWidgets.QAction(Form)
        self.actionAdd_Custom.setObjectName("actionAdd_Custom")
        self.actionAddTimeIntervals_PBEs = QtWidgets.QAction(Form)
        self.actionAddTimeIntervals_PBEs.setObjectName("actionAddTimeIntervals_PBEs")
        self.actionAddTimeIntervals_Session_Epochs = QtWidgets.QAction(Form)
        self.actionAddTimeIntervals_Session_Epochs.setObjectName("actionAddTimeIntervals_Session_Epochs")
        self.actionAddTimeIntervals_Laps = QtWidgets.QAction(Form)
        self.actionAddTimeIntervals_Laps.setObjectName("actionAddTimeIntervals_Laps")
        self.actionAddTimeIntervals_Custom = QtWidgets.QAction(Form)
        self.actionAddTimeIntervals_Custom.setObjectName("actionAddTimeIntervals_Custom")
        self.actionAddTimeCurves_Random = QtWidgets.QAction(Form)
        self.actionAddTimeCurves_Random.setObjectName("actionAddTimeCurves_Random")
        self.actionNew_1D_Ratemaps_Plot = QtWidgets.QAction(Form)
        icon14 = QtGui.QIcon()
        icon14.addPixmap(QtGui.QPixmap(":/Graphics/Icons/graphics/Spikes.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionNew_1D_Ratemaps_Plot.setIcon(icon14)
        self.actionNew_1D_Ratemaps_Plot.setObjectName("actionNew_1D_Ratemaps_Plot")
        self.actionProgrammaticDisplayConfiguration = QtWidgets.QAction(Form)
        self.actionProgrammaticDisplayConfiguration.setObjectName("actionProgrammaticDisplayConfiguration")

        self.retranslateUi(Form)
        self.toolBox.setCurrentIndex(0)
        self.btnProgrammaticDisplay.clicked.connect(Form.onProgrammaticDisplay)
        self.btnProgrammaticDisplayConfig.clicked.connect(Form.onShowProgrammaticDisplayConfig)
        self.btnProgrammaticDisplayConfig_2.clicked.connect(Form.onShowProgrammaticDisplayConfig)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Active Pipeline"))
        self.VisualizationsGroup.setTitle(_translate("Form", "Visualizations"))
        self.toolButton_4.setText(_translate("Form", "..."))
        self.toolButton_3.setText(_translate("Form", "..."))
        self.toolButton_2.setText(_translate("Form", "..."))
        self.toolButton_10.setText(_translate("Form", "..."))
        self.toolButton.setText(_translate("Form", "..."))
        self.btnProgrammaticDisplayConfig_2.setText(_translate("Form", "Configuration"))
        self.btnProgrammaticDisplayConfig.setText(_translate("Form", "Configuration"))
        self.btnProgrammaticDisplay.setText(_translate("Form", "Programmatic Display"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.tooltab_Display), _translate("Form", "Display"))
        self.toolButton_9.setText(_translate("Form", "..."))
        self.toolButton_5.setText(_translate("Form", "..."))
        self.toolButton_6.setText(_translate("Form", "..."))
        self.toolButton_8.setText(_translate("Form", "..."))
        self.toolButton_7.setText(_translate("Form", "..."))
        self.toolButton_11.setText(_translate("Form", "..."))
        self.toolButton_14.setText(_translate("Form", "..."))
        self.toolButton_13.setText(_translate("Form", "..."))
        self.toolButton_15.setText(_translate("Form", "..."))
        self.toolBox.setItemText(self.toolBox.indexOf(self.tooltab_Visualization), _translate("Form", "Visualizations"))
        self.toolButton_12.setText(_translate("Form", "..."))
        self.toolBox.setItemText(self.toolBox.indexOf(self.tooltab_Utilities), _translate("Form", "Utilities"))
        self.actionLoad.setText(_translate("Form", "Load..."))
        self.actionSave.setText(_translate("Form", "Save"))
        self.actionSave_As.setText(_translate("Form", "Save As.."))
        self.actionQuit_App.setText(_translate("Form", "Quit App"))
        self.actionAbout.setText(_translate("Form", "About..."))
        self.actionQuit_Application.setText(_translate("Form", "Quit Application"))
        self.actionNew_3D_Raster_PyQtGraph.setText(_translate("Form", "New 3D Raster (PyQtGraph)"))
        self.actionNew_Tuning_Curves_Explorer_ipcDataExplorer.setText(_translate("Form", "New Tuning Curves Explorer (ipcDataExplorer)"))
        self.actionNew_Spikes_Behavior_Explorer_ipspikesDataExplorer.setText(_translate("Form", "New Spikes+Behavior Explorer (ipspikesDataExplorer)"))
        self.actionNew_3D_Raster_Vedo.setText(_translate("Form", "New 3D Raster (Vedo)"))
        self.actionNew_2D_Raster_PyQtGraph.setText(_translate("Form", "New 2D Raster (PyQtGraph)"))
        self.actionAddTimeCurves_Position.setText(_translate("Form", "Position"))
        self.actionAddTimeCurves_Custom.setText(_translate("Form", "Custom..."))
        self.actionAdd_Custom.setText(_translate("Form", "Add Custom..."))
        self.actionAddTimeIntervals_PBEs.setText(_translate("Form", "PBEs"))
        self.actionAddTimeIntervals_Session_Epochs.setText(_translate("Form", "Session Epochs"))
        self.actionAddTimeIntervals_Laps.setText(_translate("Form", "Laps"))
        self.actionAddTimeIntervals_Custom.setText(_translate("Form", "Custom..."))
        self.actionAddTimeCurves_Random.setText(_translate("Form", "Random"))
        self.actionNew_1D_Ratemaps_Plot.setText(_translate("Form", "New 1D Ratemaps Plot"))
        self.actionProgrammaticDisplayConfiguration.setText(_translate("Form", "Programmatic Display Configuration"))
        self.actionProgrammaticDisplayConfiguration.setToolTip(_translate("Form", "Show Programmatic Display Config GUI"))
from pyphoplacecellanalysis.GUI.Qt.Widgets.IdentifyingContextSelector.IdentifyingContextSelectorWidget import IdentifyingContextSelectorWidget
from pyphoplacecellanalysis.Resources import ActionIcons
from pyphoplacecellanalysis.Resources import GuiResources
