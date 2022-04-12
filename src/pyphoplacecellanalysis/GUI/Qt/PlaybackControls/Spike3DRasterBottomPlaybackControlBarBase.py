# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\PlaybackControls\Spike3DRasterBottomPlaybackControlBarBase.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_RootWidget(object):
    def setupUi(self, RootWidget):
        RootWidget.setObjectName("RootWidget")
        RootWidget.resize(1663, 70)
        RootWidget.setStyleSheet("background-color: rgb(71, 58, 46);\n"
"border-color: rgb(207, 207, 207);\n"
"background-color: rgba(71, 65, 60, 180);\n"
"color: rgb(244, 244, 244);\n"
"border-color: rgb(0, 0, 0);")
        self.gridLayout = QtWidgets.QGridLayout(RootWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.frame_media_control = QtWidgets.QFrame(RootWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_media_control.sizePolicy().hasHeightForWidth())
        self.frame_media_control.setSizePolicy(sizePolicy)
        self.frame_media_control.setMinimumSize(QtCore.QSize(0, 70))
        self.frame_media_control.setMaximumSize(QtCore.QSize(16777215, 70))
        self.frame_media_control.setStyleSheet("background:transparent;")
        self.frame_media_control.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_media_control.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_media_control.setLineWidth(1)
        self.frame_media_control.setObjectName("frame_media_control")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_media_control)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.button_full_screen = QtWidgets.QPushButton(self.frame_media_control)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_full_screen.sizePolicy().hasHeightForWidth())
        self.button_full_screen.setSizePolicy(sizePolicy)
        self.button_full_screen.setObjectName("button_full_screen")
        self.gridLayout_2.addWidget(self.button_full_screen, 1, 12, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 1, 13, 1, 1)
        self.btnHelp = QtWidgets.QPushButton(self.frame_media_control)
        self.btnHelp.setObjectName("btnHelp")
        self.gridLayout_2.addWidget(self.btnHelp, 1, 20, 1, 1)
        self.button_mark_end = QtWidgets.QPushButton(self.frame_media_control)
        self.button_mark_end.setObjectName("button_mark_end")
        self.gridLayout_2.addWidget(self.button_mark_end, 1, 10, 1, 1)
        self.btnSkipLeft = QtWidgets.QToolButton(self.frame_media_control)
        self.btnSkipLeft.setObjectName("btnSkipLeft")
        self.gridLayout_2.addWidget(self.btnSkipLeft, 1, 14, 1, 1)
        self.btnLeft = QtWidgets.QToolButton(self.frame_media_control)
        self.btnLeft.setArrowType(QtCore.Qt.LeftArrow)
        self.btnLeft.setObjectName("btnLeft")
        self.gridLayout_2.addWidget(self.btnLeft, 1, 15, 1, 1)
        self.button_play_pause = ToggleButton(self.frame_media_control)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_play_pause.sizePolicy().hasHeightForWidth())
        self.button_play_pause.setSizePolicy(sizePolicy)
        self.button_play_pause.setCheckable(False)
        self.button_play_pause.setObjectName("button_play_pause")
        self.gridLayout_2.addWidget(self.button_play_pause, 1, 0, 1, 1)
        self.doubleSpinBoxPlaybackSpeed = QtWidgets.QDoubleSpinBox(self.frame_media_control)
        self.doubleSpinBoxPlaybackSpeed.setMinimum(0.2)
        self.doubleSpinBoxPlaybackSpeed.setMaximum(6.0)
        self.doubleSpinBoxPlaybackSpeed.setProperty("value", 1.0)
        self.doubleSpinBoxPlaybackSpeed.setObjectName("doubleSpinBoxPlaybackSpeed")
        self.gridLayout_2.addWidget(self.doubleSpinBoxPlaybackSpeed, 1, 4, 1, 1)
        self.button_mark_start = QtWidgets.QPushButton(self.frame_media_control)
        self.button_mark_start.setObjectName("button_mark_start")
        self.gridLayout_2.addWidget(self.button_mark_start, 1, 9, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 1, 11, 1, 1)
        self.slider_progress = HighlightedJumpSlider(self.frame_media_control)
        self.slider_progress.setEnabled(False)
        self.slider_progress.setMaximum(9999)
        self.slider_progress.setPageStep(1)
        self.slider_progress.setOrientation(QtCore.Qt.Horizontal)
        self.slider_progress.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.slider_progress.setObjectName("slider_progress")
        self.gridLayout_2.addWidget(self.slider_progress, 0, 0, 1, 21)
        self.button_slow_down = QtWidgets.QPushButton(self.frame_media_control)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_slow_down.sizePolicy().hasHeightForWidth())
        self.button_slow_down.setSizePolicy(sizePolicy)
        self.button_slow_down.setObjectName("button_slow_down")
        self.gridLayout_2.addWidget(self.button_slow_down, 1, 3, 1, 1)
        self.spinBoxFrameJumpMultiplier = QtWidgets.QSpinBox(self.frame_media_control)
        self.spinBoxFrameJumpMultiplier.setMinimum(1)
        self.spinBoxFrameJumpMultiplier.setMaximum(1000)
        self.spinBoxFrameJumpMultiplier.setObjectName("spinBoxFrameJumpMultiplier")
        self.gridLayout_2.addWidget(self.spinBoxFrameJumpMultiplier, 1, 16, 1, 1)
        self.btnRight = QtWidgets.QToolButton(self.frame_media_control)
        self.btnRight.setArrowType(QtCore.Qt.RightArrow)
        self.btnRight.setObjectName("btnRight")
        self.gridLayout_2.addWidget(self.btnRight, 1, 17, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem2, 1, 2, 1, 1)
        self.button_speed_up = QtWidgets.QPushButton(self.frame_media_control)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_speed_up.sizePolicy().hasHeightForWidth())
        self.button_speed_up.setSizePolicy(sizePolicy)
        self.button_speed_up.setObjectName("button_speed_up")
        self.gridLayout_2.addWidget(self.button_speed_up, 1, 6, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem3, 1, 8, 1, 1)
        self.btnSkipRight = QtWidgets.QToolButton(self.frame_media_control)
        self.btnSkipRight.setObjectName("btnSkipRight")
        self.gridLayout_2.addWidget(self.btnSkipRight, 1, 18, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem4, 1, 19, 1, 1)
        self.toolButton_SpeedBurstEnabled = QtWidgets.QToolButton(self.frame_media_control)
        self.toolButton_SpeedBurstEnabled.setEnabled(False)
        self.toolButton_SpeedBurstEnabled.setCheckable(True)
        self.toolButton_SpeedBurstEnabled.setChecked(True)
        self.toolButton_SpeedBurstEnabled.setObjectName("toolButton_SpeedBurstEnabled")
        self.gridLayout_2.addWidget(self.toolButton_SpeedBurstEnabled, 1, 5, 1, 1)
        self.button_reverse = QtWidgets.QPushButton(self.frame_media_control)
        self.button_reverse.setMinimumSize(QtCore.QSize(30, 25))
        self.button_reverse.setObjectName("button_reverse")
        self.gridLayout_2.addWidget(self.button_reverse, 1, 1, 1, 1)
        self.gridLayout.addWidget(self.frame_media_control, 0, 0, 1, 1)

        self.retranslateUi(RootWidget)
        QtCore.QMetaObject.connectSlotsByName(RootWidget)

    def retranslateUi(self, RootWidget):
        _translate = QtCore.QCoreApplication.translate
        RootWidget.setWindowTitle(_translate("RootWidget", "Form"))
        self.button_full_screen.setToolTip(_translate("RootWidget", "Set the video to full screen"))
        self.button_full_screen.setText(_translate("RootWidget", "Full Screen"))
        self.btnHelp.setText(_translate("RootWidget", "Help"))
        self.button_mark_end.setToolTip(_translate("RootWidget", "Mark the end of the entry"))
        self.button_mark_end.setText(_translate("RootWidget", "Mark End"))
        self.btnSkipLeft.setToolTip(_translate("RootWidget", "Skip Frames Left"))
        self.btnSkipLeft.setText(_translate("RootWidget", "<-"))
        self.btnSkipLeft.setShortcut(_translate("RootWidget", "Ctrl+Left"))
        self.btnLeft.setToolTip(_translate("RootWidget", "Step Frames Left"))
        self.btnLeft.setText(_translate("RootWidget", "<"))
        self.btnLeft.setShortcut(_translate("RootWidget", "Left"))
        self.button_play_pause.setToolTip(_translate("RootWidget", "Toggle Play/Pause"))
        self.button_play_pause.setText(_translate("RootWidget", "Play/Pause"))
        self.button_mark_start.setToolTip(_translate("RootWidget", "Mark the start of the entry"))
        self.button_mark_start.setText(_translate("RootWidget", "Mark Start"))
        self.button_slow_down.setToolTip(_translate("RootWidget", "Slow down the video"))
        self.button_slow_down.setText(_translate("RootWidget", "Slow Down"))
        self.button_slow_down.setShortcut(_translate("RootWidget", "Ctrl+S"))
        self.btnRight.setToolTip(_translate("RootWidget", "Step Frames Right"))
        self.btnRight.setText(_translate("RootWidget", ">"))
        self.btnRight.setShortcut(_translate("RootWidget", "Right"))
        self.button_speed_up.setToolTip(_translate("RootWidget", "Speed up the video"))
        self.button_speed_up.setText(_translate("RootWidget", "Speed Up"))
        self.button_speed_up.setShortcut(_translate("RootWidget", "Ctrl+="))
        self.btnSkipRight.setToolTip(_translate("RootWidget", "Skip Frames Right"))
        self.btnSkipRight.setText(_translate("RootWidget", "->"))
        self.btnSkipRight.setShortcut(_translate("RootWidget", "Ctrl+Right"))
        self.toolButton_SpeedBurstEnabled.setStatusTip(_translate("RootWidget", "Trigger speedburst with the hotkey"))
        self.toolButton_SpeedBurstEnabled.setWhatsThis(_translate("RootWidget", "Trigger speedburst with the hotkey"))
        self.toolButton_SpeedBurstEnabled.setText(_translate("RootWidget", "SpeedBurst"))
        self.button_reverse.setText(_translate("RootWidget", "Reverse"))
from pyphocorehelpers.gui.Qt.HighlightedJumpSlider import HighlightedJumpSlider
from pyphocorehelpers.gui.Qt.ToggleButton import ToggleButton




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    rootForm = QtWidgets.QWidget()
    ui = Ui_RootWidget()
    ui.setupUi(rootForm)
    rootForm.show()
    sys.exit(app.exec_())
