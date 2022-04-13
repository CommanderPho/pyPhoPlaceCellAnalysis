# Spike3DRasterBottomPlaybackControlBar.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\PlaybackControls\Spike3DRasterBottomPlaybackControlBar.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
from datetime import datetime, timezone, timedelta
import numpy as np
from enum import Enum

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
# from ...pyPhoPlaceCellAnalysis.src.pyphoplacecellanalysis.GUI.Qt.PlaybackControls import Spike3DRasterBottomPlaybackControlBar

from pyphoplacecellanalysis.GUI.Qt.PlaybackControls.Spike3DRasterBottomPlaybackControlBarBase import Ui_RootWidget  # Generated file from .ui

# Custom Widget classes
from pyphocorehelpers.gui.Qt.HighlightedJumpSlider import HighlightedJumpSlider
from pyphocorehelpers.gui.Qt.ToggleButton import ToggleButtonModel, ToggleButton

# For extra button symbols:
import qtawesome as qta


class Spike3DRasterBottomPlaybackControlBar(QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        self.ui = Ui_RootWidget()
        self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        
        self.initUI()
        self.show() # Show the GUI



    def initUI(self):
        self.ui.slider_progress.hide()
        
        # Setup Button: Play/Pause
        self.ui.button_play_pause.setMinimumHeight(25)
        self.ui.button_play_pause.setMinimumWidth(30)
        self.ui.play_pause_model = ToggleButtonModel(None, self)
        self.ui.play_pause_model.setStateMap(
            {
                True: {
                    "text": "",
                    "icon": qta.icon("fa.play", scale_factor=0.7, color='white', color_active='orange')
                },
                False: {
                    "text": "",
                    "icon": qta.icon("fa.pause", scale_factor=0.7, color='white', color_active='orange')
                }
            }
        )
        self.ui.button_play_pause.setModel(self.ui.play_pause_model)
        self.ui.button_play_pause.clicked.connect(self.play_pause)
        
        
        self.is_playback_reversed = self.ui.button_reverse.isChecked()
        if self.ui.button_reverse.isChecked():
            # setting background color to light-blue
            self.ui.button_reverse.setStyleSheet("background-color : lightblue")
  
        # if it is unchecked
        else:
            # set background color back to light-grey
            self.ui.button_reverse.setStyleSheet("background-color : lightgrey")
        self.ui.button_reverse.clicked.connect(self.on_reverse_held)
        
        self.ui.btnLeft.clicked.connect(self.on_jump_left)
        self.ui.btnRight.clicked.connect(self.on_jump_right)
        
        # self.ui.doubleSpinBoxPlaybackSpeed.setStyleSheet("alternate-background-color: rgb(255, 0, 0); selection-background-color: rgb(75, 83, 65);")
                    
        # self.ui.doubleSpinBoxPlaybackSpeed.setStyleSheet("QDoubleSpinBox"
        #                     "{"
        #                     "border : 2px solid black;"
        #                     "background : rgb(62, 57, 52);"
        #                     "}"
        #                     "QDoubleSpinBox::hover"
        #                     "{"
        #                     "border : 2px solid green;"
        #                     "background : lightgreen;"
        #                     "}"                            
        #                     )
        # "QDoubleSpinBox::up-arrow"
        #                     "{"
        #                     "border : 1px solid black;"
        #                     "background : blue;"
        #                     "}"
        #                     "QDoubleSpinBox::down-arrow"
        #                     "{"
        #                     "border : 1px solid black;"
        #                     "background : red;"
        #                     "}"
        # rgb(75, 83, 65)
        
        # self.ui.spinBoxFrameJumpMultiplier.setStyleSheet("alternate-background-color: rgb(255, 0, 0); selection-background-color: rgb(75, 83, 65);")
        # self.ui.spinBoxFrameJumpMultiplier.setStyleSheet("QSpinBox"
        # "{"
        # "background: rgb(62, 57, 52);"
        # "}"
        # "QSpinBox::hover"
        # "{"
        # "background: rgb(75, 83, 65);"
        # "}"
        # )
        
        
        ## Remove Extra Buttons:
        self.ui.btnSkipLeft.hide()
        self.ui.btnSkipRight.hide()
        
        # Remove Full Screen Button
        self.ui.button_full_screen.hide()
        self.ui.button_mark_start.hide()
        self.ui.button_mark_end.hide()
        
        
        
    def __str__(self):
         return 

    
    # Called when the play/pause button is clicked:
    def play_pause(self):
        # THIS IS NOT TRUE: the play_pause_model uses inverted logic, so we negate the current state value to determine if is_playing
        is_playing = self.ui.play_pause_model.getState()
        # print(f'is_playing: {is_playing}')
        
        if (not is_playing):
            # if self.slidebar_val == 1:
            #     # self.ui.slider.setValue(0)
            #     pass
            # # self.play_pause_model.setState(not is_playing)
            # self.animationThread.start()
            pass

        else:
            # self.play_pause_model.setState(not is_playing)
            # self.animationThread.terminate()
            pass
                    
        # self.ui.play_pause_model.blockSignals(True)
        # self.ui.play_pause_model.setState(not is_playing)
        # self.ui.play_pause_model.blockSignals(False)
        
        # if self.ui.btn_slide_run.tag == "paused" or self.slidebar_val == 1:
        #     if self.slidebar_val == 1:
        #         self.ui.slider.setValue(0)
            
        #     self.ui.btn_slide_run.setText("||")
        #     self.ui.btn_slide_run.tag = "running"
        #     self.animationThread.start()

        # elif self.ui.btn_slide_run.tag == "running":
        #     self.ui.btn_slide_run.setText(">")
        #     self.ui.btn_slide_run.tag = "paused"
        #     self.animationThread.terminate()


    

    def on_jump_left(self):
        # Skip back some frames
        # self.shift_animation_frame_val(-5)
        pass
        
    def on_jump_right(self):
        # Skip forward some frames
        # self.shift_animation_frame_val(5)
        pass        

    def on_reverse_held(self):
        # Change the direction of playback by changing the sign of the updating.
        # if button is checked    
        self.is_playback_reversed = self.ui.button_reverse.isChecked()
        if self.ui.button_reverse.isChecked():
            # setting background color to light-blue
            self.ui.button_reverse.setStyleSheet("background-color : lightblue")
  
        # if it is unchecked
        else:
            # set background color back to light-grey
            self.ui.button_reverse.setStyleSheet("background-color : lightgrey")



## Start Qt event loop
# if __name__ == '__main__':
#     app = mkQApp("PlacefieldVisualSelectionWidget Example")
#     widget = PlacefieldVisualSelectionWidget()
#     widget.show()
#     pg.exec()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    testWidget = Spike3DRasterBottomPlaybackControlBar()
    # testWidget.show()
    sys.exit(app.exec_())
