# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Menus\LocalMenus_AddRenderable\LocalMenus_AddRenderable.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
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
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/Render/Icons/actions/categories.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.menuAdd_Renderable.setIcon(icon)
        self.menuAdd_Renderable.setObjectName("menuAdd_Renderable")
        self.menuAddRenderable_Time_Curves = QtWidgets.QMenu(self.menuAdd_Renderable)
        self.menuAddRenderable_Time_Curves.setTitle("Add Time Curves...")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/Render/Icons/actions/chart-up-color.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.menuAddRenderable_Time_Curves.setIcon(icon1)
        self.menuAddRenderable_Time_Curves.setObjectName("menuAddRenderable_Time_Curves")
        self.menuAddRenderable_Time_Intervals = QtWidgets.QMenu(self.menuAdd_Renderable)
        self.menuAddRenderable_Time_Intervals.setTitle("Add Time Intervals...")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/Render/Icons/actions/spectrum-emission.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.menuAddRenderable_Time_Intervals.setIcon(icon2)
        self.menuAddRenderable_Time_Intervals.setObjectName("menuAddRenderable_Time_Intervals")
        self.menuAddRenderable_Matplotlib_Plot = QtWidgets.QMenu(self.menuAdd_Renderable)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/Graphics/Icons/graphics/ic_multiline_chart_48px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.menuAddRenderable_Matplotlib_Plot.setIcon(icon3)
        self.menuAddRenderable_Matplotlib_Plot.setObjectName("menuAddRenderable_Matplotlib_Plot")
        self.menuCreate_Paired_Widget = QtWidgets.QMenu(self.menubar)
        self.menuCreate_Paired_Widget.setTitle("Create Paired Widget")
        self.menuCreate_Paired_Widget.setObjectName("menuCreate_Paired_Widget")
        self.menuDocked_Widgets = QtWidgets.QMenu(self.menubar)
        self.menuDocked_Widgets.setObjectName("menuDocked_Widgets")
        self.menuAdd_Docked_Widget = QtWidgets.QMenu(self.menuDocked_Widgets)
        self.menuAdd_Docked_Widget.setObjectName("menuAdd_Docked_Widget")
        self.menuDecoder = QtWidgets.QMenu(self.menubar)
        self.menuDecoder.setObjectName("menuDecoder")
        self.menuDecoded_Epoch_Slices = QtWidgets.QMenu(self.menuDecoder)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/Graphics/Icons/graphics/Rectangles.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.menuDecoded_Epoch_Slices.setIcon(icon4)
        self.menuDecoded_Epoch_Slices.setObjectName("menuDecoded_Epoch_Slices")
        self.menuStandaloneWindows = QtWidgets.QMenu(self.menubar)
        self.menuStandaloneWindows.setObjectName("menuStandaloneWindows")
        LocalMenus_AddRenderable.setMenuBar(self.menubar)
        self.actionAddTimeCurves_Position = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeCurves_Position.setText("Position")
        self.actionAddTimeCurves_Position.setObjectName("actionAddTimeCurves_Position")
        self.actionAddTimeCurves_Custom = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeCurves_Custom.setEnabled(True)
        self.actionAddTimeCurves_Custom.setText("Custom...")
        self.actionAddTimeCurves_Custom.setObjectName("actionAddTimeCurves_Custom")
        self.actionAddCustomRenderable = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddCustomRenderable.setEnabled(False)
        self.actionAddCustomRenderable.setText("Add Custom...")
        self.actionAddCustomRenderable.setObjectName("actionAddCustomRenderable")
        self.actionAddTimeIntervals_PBEs = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeIntervals_PBEs.setText("PBEs")
        self.actionAddTimeIntervals_PBEs.setObjectName("actionAddTimeIntervals_PBEs")
        self.actionAddTimeIntervals_SessionEpochs = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeIntervals_SessionEpochs.setText("Session Epochs")
        self.actionAddTimeIntervals_SessionEpochs.setObjectName("actionAddTimeIntervals_SessionEpochs")
        self.actionAddTimeIntervals_Laps = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeIntervals_Laps.setText("Laps")
        self.actionAddTimeIntervals_Laps.setObjectName("actionAddTimeIntervals_Laps")
        self.actionAddTimeIntervals_Custom = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeIntervals_Custom.setText("Custom...")
        self.actionAddTimeIntervals_Custom.setObjectName("actionAddTimeIntervals_Custom")
        self.actionAddTimeCurves_Random = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeCurves_Random.setText("Random")
        self.actionAddTimeCurves_Random.setObjectName("actionAddTimeCurves_Random")
        self.actionClear_all_Time_Curves = QtWidgets.QAction(LocalMenus_AddRenderable)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/Render/Icons/actions/chart--minus.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionClear_all_Time_Curves.setIcon(icon5)
        self.actionClear_all_Time_Curves.setObjectName("actionClear_all_Time_Curves")
        self.actionClear_all_Time_Intervals = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionClear_all_Time_Intervals.setIcon(icon5)
        self.actionClear_all_Time_Intervals.setObjectName("actionClear_all_Time_Intervals")
        self.actionClear_all_Renderables = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionClear_all_Renderables.setObjectName("actionClear_all_Renderables")
        self.actionAddTimeIntervals_Ripples = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeIntervals_Ripples.setObjectName("actionAddTimeIntervals_Ripples")
        self.actionAddTimeIntervals_Replays = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeIntervals_Replays.setObjectName("actionAddTimeIntervals_Replays")
        self.actionCreate_paired_time_synchronized_widget = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionCreate_paired_time_synchronized_widget.setObjectName("actionCreate_paired_time_synchronized_widget")
        self.actionTimeSynchronizedOccupancyPlotter = QtWidgets.QAction(LocalMenus_AddRenderable)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/Render/Icons/actions/bar-chart_2@1x.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionTimeSynchronizedOccupancyPlotter.setIcon(icon6)
        self.actionTimeSynchronizedOccupancyPlotter.setText("TimeSynchronizedOccupancyPlotter")
        self.actionTimeSynchronizedOccupancyPlotter.setObjectName("actionTimeSynchronizedOccupancyPlotter")
        self.actionTimeSynchronizedPlacefieldsPlotter = QtWidgets.QAction(LocalMenus_AddRenderable)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/Render/Icons/actions/wifi-channel_2@1x.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionTimeSynchronizedPlacefieldsPlotter.setIcon(icon7)
        self.actionTimeSynchronizedPlacefieldsPlotter.setText("TimeSynchronizedPlacefieldsPlotter")
        self.actionTimeSynchronizedPlacefieldsPlotter.setObjectName("actionTimeSynchronizedPlacefieldsPlotter")
        self.actionCombineTimeSynchronizedPlotterWindow = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionCombineTimeSynchronizedPlotterWindow.setIcon(icon3)
        self.actionCombineTimeSynchronizedPlotterWindow.setText("Combined Time Syncrhonized Plotter")
        self.actionCombineTimeSynchronizedPlotterWindow.setObjectName("actionCombineTimeSynchronizedPlotterWindow")
        self.actionTimeSynchronizedDecoderPlotter = QtWidgets.QAction(LocalMenus_AddRenderable)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/Render/Icons/actions/area-chart_2@1x.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionTimeSynchronizedDecoderPlotter.setIcon(icon8)
        self.actionTimeSynchronizedDecoderPlotter.setText("TimeSynchronizedDecoderPlotter")
        self.actionTimeSynchronizedDecoderPlotter.setObjectName("actionTimeSynchronizedDecoderPlotter")
        self.actionAddMatplotlibPlot_DecodedPosition = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddMatplotlibPlot_DecodedPosition.setEnabled(False)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/Graphics/Icons/graphics/ic_blur_linear_48px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAddMatplotlibPlot_DecodedPosition.setIcon(icon9)
        self.actionAddMatplotlibPlot_DecodedPosition.setObjectName("actionAddMatplotlibPlot_DecodedPosition")
        self.actionClear_all_Matplotlib_Plots = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionClear_all_Matplotlib_Plots.setIcon(icon5)
        self.actionClear_all_Matplotlib_Plots.setObjectName("actionClear_all_Matplotlib_Plots")
        self.actionAddMatplotlibPlot_Custom = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddMatplotlibPlot_Custom.setEnabled(False)
        self.actionAddMatplotlibPlot_Custom.setObjectName("actionAddMatplotlibPlot_Custom")
        self.actionMatplotlib_View = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionMatplotlib_View.setObjectName("actionMatplotlib_View")
        self.actionCustom = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionCustom.setObjectName("actionCustom")
        self.actionTest = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionTest.setObjectName("actionTest")
        self.actionContext_Nested_Docks = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionContext_Nested_Docks.setObjectName("actionContext_Nested_Docks")
        self.actionDecoded_Epoch_Slices_Laps = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionDecoded_Epoch_Slices_Laps.setIcon(icon4)
        self.actionDecoded_Epoch_Slices_Laps.setObjectName("actionDecoded_Epoch_Slices_Laps")
        self.actionDecoded_Epoch_Slices_PBEs = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionDecoded_Epoch_Slices_PBEs.setObjectName("actionDecoded_Epoch_Slices_PBEs")
        self.actionDecoded_Epoch_Slices_Ripple = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionDecoded_Epoch_Slices_Ripple.setObjectName("actionDecoded_Epoch_Slices_Ripple")
        self.actionDecoded_Epoch_Slices_Replay = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionDecoded_Epoch_Slices_Replay.setObjectName("actionDecoded_Epoch_Slices_Replay")
        self.actionDecoded_Epoch_Slices_Custom = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionDecoded_Epoch_Slices_Custom.setEnabled(False)
        self.actionDecoded_Epoch_Slices_Custom.setObjectName("actionDecoded_Epoch_Slices_Custom")
        self.actionAddTimeCurves_Velocity = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeCurves_Velocity.setObjectName("actionAddTimeCurves_Velocity")
        self.actionAddTimeIntervals_Bursts = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeIntervals_Bursts.setObjectName("actionAddTimeIntervals_Bursts")
        self.actionAddTimeCurves_RelativeEntropySurprise = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeCurves_RelativeEntropySurprise.setObjectName("actionAddTimeCurves_RelativeEntropySurprise")
        self.actionSpike3DLauncher = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionSpike3DLauncher.setObjectName("actionSpike3DLauncher")
        self.actionAddTimeIntervals_NonPBEs = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionAddTimeIntervals_NonPBEs.setObjectName("actionAddTimeIntervals_NonPBEs")
        self.actionDecoded_Epoch_Slices_NonPBEs = QtWidgets.QAction(LocalMenus_AddRenderable)
        self.actionDecoded_Epoch_Slices_NonPBEs.setObjectName("actionDecoded_Epoch_Slices_NonPBEs")
        self.menuAddRenderable_Time_Curves.addAction(self.actionAddTimeCurves_Position)
        self.menuAddRenderable_Time_Curves.addAction(self.actionAddTimeCurves_Velocity)
        self.menuAddRenderable_Time_Curves.addAction(self.actionAddTimeCurves_Random)
        self.menuAddRenderable_Time_Curves.addAction(self.actionAddTimeCurves_RelativeEntropySurprise)
        self.menuAddRenderable_Time_Curves.addAction(self.actionAddTimeCurves_Custom)
        self.menuAddRenderable_Time_Curves.addSeparator()
        self.menuAddRenderable_Time_Curves.addAction(self.actionClear_all_Time_Curves)
        self.menuAddRenderable_Time_Intervals.addAction(self.actionAddTimeIntervals_SessionEpochs)
        self.menuAddRenderable_Time_Intervals.addAction(self.actionAddTimeIntervals_Laps)
        self.menuAddRenderable_Time_Intervals.addAction(self.actionAddTimeIntervals_PBEs)
        self.menuAddRenderable_Time_Intervals.addAction(self.actionAddTimeIntervals_Ripples)
        self.menuAddRenderable_Time_Intervals.addAction(self.actionAddTimeIntervals_Replays)
        self.menuAddRenderable_Time_Intervals.addAction(self.actionAddTimeIntervals_Bursts)
        self.menuAddRenderable_Time_Intervals.addAction(self.actionAddTimeIntervals_NonPBEs)
        self.menuAddRenderable_Time_Intervals.addAction(self.actionAddTimeIntervals_Custom)
        self.menuAddRenderable_Time_Intervals.addSeparator()
        self.menuAddRenderable_Time_Intervals.addAction(self.actionClear_all_Time_Intervals)
        self.menuAddRenderable_Matplotlib_Plot.addAction(self.actionAddMatplotlibPlot_DecodedPosition)
        self.menuAddRenderable_Matplotlib_Plot.addAction(self.actionAddMatplotlibPlot_Custom)
        self.menuAddRenderable_Matplotlib_Plot.addSeparator()
        self.menuAddRenderable_Matplotlib_Plot.addAction(self.actionClear_all_Matplotlib_Plots)
        self.menuAdd_Renderable.addAction(self.menuAddRenderable_Time_Curves.menuAction())
        self.menuAdd_Renderable.addAction(self.menuAddRenderable_Time_Intervals.menuAction())
        self.menuAdd_Renderable.addAction(self.menuAddRenderable_Matplotlib_Plot.menuAction())
        self.menuAdd_Renderable.addAction(self.actionCreate_paired_time_synchronized_widget)
        self.menuAdd_Renderable.addAction(self.actionAddCustomRenderable)
        self.menuAdd_Renderable.addSeparator()
        self.menuAdd_Renderable.addAction(self.actionClear_all_Renderables)
        self.menuCreate_Paired_Widget.addAction(self.actionTimeSynchronizedOccupancyPlotter)
        self.menuCreate_Paired_Widget.addAction(self.actionTimeSynchronizedPlacefieldsPlotter)
        self.menuCreate_Paired_Widget.addAction(self.actionTimeSynchronizedDecoderPlotter)
        self.menuCreate_Paired_Widget.addSeparator()
        self.menuCreate_Paired_Widget.addAction(self.actionCombineTimeSynchronizedPlotterWindow)
        self.menuAdd_Docked_Widget.addAction(self.actionMatplotlib_View)
        self.menuAdd_Docked_Widget.addAction(self.actionContext_Nested_Docks)
        self.menuAdd_Docked_Widget.addAction(self.actionCustom)
        self.menuDocked_Widgets.addAction(self.menuAdd_Docked_Widget.menuAction())
        self.menuDocked_Widgets.addAction(self.actionTest)
        self.menuDecoded_Epoch_Slices.addAction(self.actionDecoded_Epoch_Slices_Laps)
        self.menuDecoded_Epoch_Slices.addAction(self.actionDecoded_Epoch_Slices_PBEs)
        self.menuDecoded_Epoch_Slices.addAction(self.actionDecoded_Epoch_Slices_Ripple)
        self.menuDecoded_Epoch_Slices.addAction(self.actionDecoded_Epoch_Slices_Replay)
        self.menuDecoded_Epoch_Slices.addAction(self.actionDecoded_Epoch_Slices_NonPBEs)
        self.menuDecoded_Epoch_Slices.addSeparator()
        self.menuDecoded_Epoch_Slices.addAction(self.actionDecoded_Epoch_Slices_Custom)
        self.menuDecoder.addAction(self.menuDecoded_Epoch_Slices.menuAction())
        self.menuStandaloneWindows.addAction(self.actionSpike3DLauncher)
        self.menubar.addAction(self.menuAdd_Renderable.menuAction())
        self.menubar.addAction(self.menuCreate_Paired_Widget.menuAction())
        self.menubar.addAction(self.menuDocked_Widgets.menuAction())
        self.menubar.addAction(self.menuDecoder.menuAction())
        self.menubar.addAction(self.menuStandaloneWindows.menuAction())

        self.retranslateUi(LocalMenus_AddRenderable)
        QtCore.QMetaObject.connectSlotsByName(LocalMenus_AddRenderable)

    def retranslateUi(self, LocalMenus_AddRenderable):
        _translate = QtCore.QCoreApplication.translate
        self.menuAddRenderable_Matplotlib_Plot.setTitle(_translate("LocalMenus_AddRenderable", "Add Matplotlib Plot..."))
        self.menuDocked_Widgets.setTitle(_translate("LocalMenus_AddRenderable", "Docked Widgets"))
        self.menuAdd_Docked_Widget.setTitle(_translate("LocalMenus_AddRenderable", "Add Docked Widget"))
        self.menuDecoder.setTitle(_translate("LocalMenus_AddRenderable", "Decoder"))
        self.menuDecoded_Epoch_Slices.setTitle(_translate("LocalMenus_AddRenderable", "Decoded Epoch Slices"))
        self.menuStandaloneWindows.setTitle(_translate("LocalMenus_AddRenderable", "Standalone Windows"))
        self.actionClear_all_Time_Curves.setText(_translate("LocalMenus_AddRenderable", "Clear all Time Curves"))
        self.actionClear_all_Time_Intervals.setText(_translate("LocalMenus_AddRenderable", "Clear all Time Intervals"))
        self.actionClear_all_Renderables.setText(_translate("LocalMenus_AddRenderable", "Clear all Renderables"))
        self.actionAddTimeIntervals_Ripples.setText(_translate("LocalMenus_AddRenderable", "Ripples"))
        self.actionAddTimeIntervals_Replays.setText(_translate("LocalMenus_AddRenderable", "Replays"))
        self.actionCreate_paired_time_synchronized_widget.setText(_translate("LocalMenus_AddRenderable", "Create paired time synchronized widget"))
        self.actionAddMatplotlibPlot_DecodedPosition.setText(_translate("LocalMenus_AddRenderable", "Add Position Decoding"))
        self.actionAddMatplotlibPlot_DecodedPosition.setStatusTip(_translate("LocalMenus_AddRenderable", "Use from \"Create Connected Widget\" menu"))
        self.actionClear_all_Matplotlib_Plots.setText(_translate("LocalMenus_AddRenderable", "Clear all Matplotlib Plots"))
        self.actionAddMatplotlibPlot_Custom.setText(_translate("LocalMenus_AddRenderable", "Custom Matplotlib Plot..."))
        self.actionMatplotlib_View.setText(_translate("LocalMenus_AddRenderable", "Matplotlib View"))
        self.actionMatplotlib_View.setToolTip(_translate("LocalMenus_AddRenderable", "<html><head/><body><p>Matplotlib View<img src=\":/Graphics/Icons/graphics/ic_timeline_48px.png\"/></p></body></html>"))
        self.actionCustom.setText(_translate("LocalMenus_AddRenderable", "Custom..."))
        self.actionTest.setText(_translate("LocalMenus_AddRenderable", "Test"))
        self.actionContext_Nested_Docks.setText(_translate("LocalMenus_AddRenderable", "Context Nested Docks"))
        self.actionDecoded_Epoch_Slices_Laps.setText(_translate("LocalMenus_AddRenderable", "Laps"))
        self.actionDecoded_Epoch_Slices_PBEs.setText(_translate("LocalMenus_AddRenderable", "PBEs"))
        self.actionDecoded_Epoch_Slices_Ripple.setText(_translate("LocalMenus_AddRenderable", "Ripple"))
        self.actionDecoded_Epoch_Slices_Replay.setText(_translate("LocalMenus_AddRenderable", "Replay"))
        self.actionDecoded_Epoch_Slices_Custom.setText(_translate("LocalMenus_AddRenderable", "Custom..."))
        self.actionAddTimeCurves_Velocity.setText(_translate("LocalMenus_AddRenderable", "Velocity"))
        self.actionAddTimeIntervals_Bursts.setText(_translate("LocalMenus_AddRenderable", "Bursts"))
        self.actionAddTimeCurves_RelativeEntropySurprise.setText(_translate("LocalMenus_AddRenderable", "Relative Entropy Surprise"))
        self.actionSpike3DLauncher.setText(_translate("LocalMenus_AddRenderable", "Spike3D Launcher"))
        self.actionAddTimeIntervals_NonPBEs.setText(_translate("LocalMenus_AddRenderable", "Non-PBEs"))
        self.actionDecoded_Epoch_Slices_NonPBEs.setText(_translate("LocalMenus_AddRenderable", "Non-PBEs"))
from pyphoplacecellanalysis.Resources import ActionIcons
from pyphoplacecellanalysis.Resources import GuiResources
from pyphoplacecellanalysis.Resources import breeze
