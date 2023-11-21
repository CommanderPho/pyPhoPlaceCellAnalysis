# Spike3DRasterBottomPlaybackControlBar.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\PlaybackControls\Spike3DRasterBottomPlaybackControlBar.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
from datetime import datetime, timezone, timedelta
import numpy as np
from enum import Enum

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir

## IMPORTS:
# from ...pyPhoPlaceCellAnalysis.src.pyphoplacecellanalysis.GUI.Qt.PlaybackControls import Spike3DRasterBottomPlaybackControlBar

# from pyphoplacecellanalysis.GUI.Qt.PlaybackControls.Spike3DRasterBottomPlaybackControlBarBase import Ui_RootWidget  # Generated file from .ui
# from pyphoplacecellanalysis.GUI.Qt.PlaybackControls.Uic_AUTOGEN_Spike3DRasterBottomPlaybackControlBarBase import Ui_RootWidget


# Custom Widget classes
from pyphocorehelpers.gui.Qt.HighlightedJumpSlider import HighlightedJumpSlider
from pyphocorehelpers.gui.Qt.ToggleButton import ToggleButtonModel, ToggleButton

# For extra button symbols:
import qtawesome as qta

from pyphoplacecellanalysis.GUI.Qt.Mixins.ComboBoxMixins import KeysListAccessingMixin, ComboBoxCtrlOwningMixin
from pyphoplacecellanalysis.GUI.Qt.PlaybackControls.Spike3DRasterBottomPlaybackControlBarBase import Spike3DRasterBottomPlaybackControlBarBase

""" TODO: Refactor from pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\Mixins\RenderWindowControlsMixin.py

RenderWindowControlsMixin
RenderPlaybackControlsMixin




btnJumpToPrevious
comboActiveJumpTargetSeries
btnJumpToNext

# TODO:

btnCurrentIntervals_Remove
btnCurrentIntervals_Customize

"""


class Spike3DRasterBottomPlaybackControlBar(ComboBoxCtrlOwningMixin, Spike3DRasterBottomPlaybackControlBarBase):
    """ A playback bar with buttons loaded from a Qt .ui file. """
    
    play_pause_toggled = QtCore.pyqtSignal(bool) # returns bool indicating whether is_playing
    jump_left = QtCore.pyqtSignal()
    jump_right = QtCore.pyqtSignal()
    reverse_toggled = QtCore.pyqtSignal(bool) # returns bool indicating whether is_reversed
        
    # Jump Target Items
    jump_target_left = QtCore.pyqtSignal(str)
    jump_target_right = QtCore.pyqtSignal(str)
    jump_series_selection_changed = QtCore.pyqtSignal(str)
    
    # Series Target Actions
    series_remove_pressed = QtCore.pyqtSignal(str)    
    series_customize_pressed = QtCore.pyqtSignal(str)

    series_clear_all_pressed = QtCore.pyqtSignal()
    series_add_pressed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent) # Call the inherited classes __init__ method
        
        # # Auto
        # self.ui = Ui_RootWidget()
        # self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        
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
        self._format_button_reversed()
        self.ui.button_reverse.clicked.connect(self.on_reverse_held)
        
        self.ui.btnLeft.clicked.connect(self.on_jump_left)
        self.ui.btnRight.clicked.connect(self.on_jump_right)

        ## Remove Extra Buttons:
        self.ui.btnSkipLeft.hide()
        self.ui.btnSkipRight.hide()
        
        # Remove Full Screen Button
        self.ui.button_full_screen.hide()
        self.ui.button_mark_start.hide()
        self.ui.button_mark_end.hide()
        
        # Jump-to:
        self.ui.btnJumpToUnused.hide()
        self.ui.btnJumpToPrevious.pressed.connect(self.on_jump_prev_series_item)
        self.ui.btnJumpToNext.pressed.connect(self.on_jump_next_series_item)
        self.ui.comboActiveJumpTargetSeries.currentTextChanged.connect(self.on_jump_combo_series_changed)

        self.ui.btnCurrentIntervals_Customize.pressed.connect(self.on_series_customize_button_pressed)
        self.ui.btnCurrentIntervals_Remove.pressed.connect(self.on_series_remove_button_pressed)

        ## Setup Extra Button:
        # self.ui.btnCurrentIntervals_Extra.pressed.connect(self.on_series_customize_button_pressed)
        self.ui._series_extra_menu = QtWidgets.QMenu()
        self.ui._series_extra_menu.addAction('Clear all', self.on_action_clear_all_pressed)
        self.ui._series_extra_menu.addAction('Add Intervals...', self.on_action_request_add_intervals_pressed)
        self.ui.btnCurrentIntervals_Extra.setMenu(self.ui._series_extra_menu)
        self._update_series_action_buttons(self.has_valid_current_target_series_name) # should disable the action buttons to start

        
    # def __str__(self):
    #      return 
    
    # Called when the play/pause button is clicked:
    def play_pause(self):
        # THIS IS NOT TRUE: the play_pause_model uses inverted logic, so we negate the current state value to determine if is_playing
        is_playing = self.ui.play_pause_model.getState()
        # print(f'is_playing: {is_playing}')
        self.play_pause_toggled.emit(is_playing)

    def on_jump_left(self):
        # Skip back some frames
        self.jump_left.emit()
        # self.shift_animation_frame_val(-5)
        pass
        
    def on_jump_right(self):
        # Skip forward some frames
        self.jump_right.emit()
        # self.shift_animation_frame_val(5)
        pass        

    def on_reverse_held(self):
        # Change the direction of playback by changing the sign of the updating.
        # if button is checked    
        self.is_playback_reversed = self.ui.button_reverse.isChecked()
        self._format_button_reversed()

        self.reverse_toggled.emit(self.is_playback_reversed)

    def _format_button_reversed(self):
        """ formats the reverse button based on whether it is checked or not """
        if self.ui.button_reverse.isChecked():
            # setting background color to an energized blue with white text
            self.ui.button_reverse.setStyleSheet("color : rgb(255, 255, 255); background-color : rgb(85, 170, 255);")
  
        # if it is unchecked
        else:
            # set background color back to default (dark grey) with grey text
            self.ui.button_reverse.setStyleSheet("color : rgb(180, 180, 180); background-color : rgb(65, 60, 54)")


    # ==================================================================================================================== #
    # Jump Target Controls                                                                                                 #
    # ==================================================================================================================== #
    # @property
    # def combo_jump_target_series(self):
    #     return self.ui.comboActiveJumpTargetSeries

    @property
    def current_selected_jump_target_series_name(self):
        """The current_selected_jump_target_series_name property."""
        selected_index, selected_item_text = self.get_current_jump_target_series_selection()
        return selected_item_text
    @current_selected_jump_target_series_name.setter
    def current_selected_jump_target_series_name(self, value):
        self.try_select_combo_item_with_text(self.ui.comboActiveJumpTargetSeries, search_text=value, debug_print=False)

    @property
    def has_valid_current_target_series_name(self):
        """True if there is currently a valid current_target_series selected in the combo box."""
        selected_index, selected_item_text = self.get_current_jump_target_series_selection()
        return (selected_index > -1)

    def get_current_jump_target_series_selection(self):
        """ gets the currently selected jump-target series """
        ## Capture the previous selection:
        selected_index, selected_item_text = self.get_current_combo_item_selection(self.ui.comboActiveJumpTargetSeries, debug_print=False)
        return (selected_index, selected_item_text)


    def update_jump_target_series_options(self, new_options):
        """ called to update the list of jump-target options """
        return self._tryUpdateComboItemsUi(self.ui.comboActiveJumpTargetSeries, new_options)

    def _tryUpdateComboItemsUi(self, curr_combo_box, new_options):
        updated_list = KeysListAccessingMixin.get_keys_list(new_options)
        # self.replace_combo_items(self.ui.comboActiveJumpTargetSeries, updated_list)
        ## Update Combo box items:
        # curr_combo_box = self.ui.comboActiveJumpTargetSeries # QComboBox 

        ## Freeze signals:
        curr_combo_box.blockSignals(True)
        
        ## Capture the previous selection:
        selected_index, selected_item_text = self.get_current_combo_item_selection(curr_combo_box)
        had_previous_selected_item = (selected_item_text is not None)

        ## Perform the replacement:
        self.replace_combo_items(curr_combo_box, updated_list)
        
        ## Re-select the previously selected item if possible:
        if not had_previous_selected_item:
            # no previously selected item. Instead, select the first item.
            # self._trySelectFirstComboItem()
            self._trySelectFirstComboItem(self.ui.comboActiveJumpTargetSeries, updated_list)
        found_desired_index = self.try_select_combo_item_with_text(curr_combo_box, selected_item_text)
        ## Unblock the signals:
        curr_combo_box.blockSignals(False)

    def _trySelectFirstComboItem(self, curr_combo_box, current_list):
        """ tries to select the first item (index 0) if possible. Otherwise, fails gracefully.
        Internally calls self.try_select_combo_item_with_text(...)
         """
        # no previously selected item. Instead, select the first item.
        if (len(current_list) > 0):
            selected_item_text = current_list[0] # get the first item text to try and select.
            found_desired_index = self.try_select_combo_item_with_text(curr_combo_box, selected_item_text)
        else:
            print(f'WARNING: could not select any default items because the list was empty.')
            found_desired_index = None
        return found_desired_index


    def _update_series_action_buttons(self, has_valid_series_selection: bool):
        """ conditionally update whether the buttons are enabled based on whether we have a valid series selection. """
        self.ui.btnJumpToPrevious.setEnabled(has_valid_series_selection)
        self.ui.btnJumpToNext.setEnabled(has_valid_series_selection)
        self.ui.btnCurrentIntervals_Remove.setEnabled(has_valid_series_selection)
        self.ui.btnCurrentIntervals_Customize.setEnabled(has_valid_series_selection)
        # self.ui.btnCurrentIntervals_Extra.setEnabled(has_valid_series_selection)


    @QtCore.pyqtSlot(object)
    def on_rendered_intervals_list_changed(self, interval_list_owning_object):
        """ called when the list of rendered intervals changes """
        self.update_jump_target_series_options(interval_list_owning_object.list_all_rendered_intervals(debug_print=False))

    @QtCore.pyqtSlot(str)
    def on_jump_combo_series_changed(self, series_name):
        print(f'on_jump_combo_series_changed(series_name: {series_name})')
        curr_jump_series_name = self.current_selected_jump_target_series_name # 'PBEs'
        self._update_series_action_buttons(self.has_valid_current_target_series_name) # enable/disable the action buttons
        self.jump_series_selection_changed.emit(curr_jump_series_name)

    @QtCore.pyqtSlot()
    def on_jump_next_series_item(self):
        """ seeks the current active_time_Window to the start of the next epoch event (for the epoch event series specified in the bottom bar) 

            By default, snap the start of the active_time_window to the start of the next epoch event
        """
        curr_jump_series_name = self.current_selected_jump_target_series_name # 'PBEs'
        print(f'on_jump_next_series_item(): curr_jump_series_name: {curr_jump_series_name}')
        self.jump_target_right.emit(curr_jump_series_name)

    @QtCore.pyqtSlot()
    def on_jump_prev_series_item(self):
        """ seeks the current active_time_Window to the start of the next epoch event (for the epoch event series specified in the bottom bar) 

            By default, snap the start of the active_time_window to the start of the next epoch event
        """
        curr_jump_series_name = self.current_selected_jump_target_series_name # 'PBEs'
        print(f'on_jump_prev_series_item(): curr_jump_series_name: {curr_jump_series_name}')
        self.jump_target_left.emit(curr_jump_series_name)


    @QtCore.pyqtSlot()
    def on_series_remove_button_pressed(self):
        """ 
        """
        curr_series_name = self.current_selected_jump_target_series_name # 'PBEs'
        print(f'on_series_remove_button_pressed(): curr_series_name: {curr_series_name}')
        self.series_remove_pressed.emit(curr_series_name)

    @QtCore.pyqtSlot()
    def on_series_customize_button_pressed(self):
        """ 
        """
        curr_series_name = self.current_selected_jump_target_series_name # 'PBEs'
        print(f'on_series_customize_button_pressed(): curr_series_name: {curr_series_name}')
        self.series_customize_pressed.emit(curr_series_name)


    @QtCore.pyqtSlot()
    def on_action_clear_all_pressed(self):
        """ 
        """
        print(f'on_action_clear_all_pressed()')
        self.series_clear_all_pressed.emit()

    @QtCore.pyqtSlot()
    def on_action_request_add_intervals_pressed(self):
        """ 
        """
        print(f'on_action_request_add_intervals_pressed()')
        self.series_add_pressed.emit()


    # @QtCore.pyqtSlot()
    # def on_series_extra_button_pressed(self):
    #     """ 
    #     """
    #     print(f'on_series_extra_button_pressed()')
    #     # self.series_remove_pressed.emit(curr_series_name)

    
class SpikeRasterBottomFrameControlsMixin:
    """ renders the UI controls for the Spike3DRaster_Vedo class 
        Follows Conventions outlined in ModelViewMixin Conventions.md
        
        Currently Unused after being removed from Spike3DRaster_Vedo
    """
    
    @QtCore.pyqtSlot()
    def SpikeRasterBottomFrameControlsMixin_on_init(self):
        """ perform any parameters setting/checking during init """
        pass

    @QtCore.pyqtSlot()
    def SpikeRasterBottomFrameControlsMixin_on_setup(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        pass
    
    @QtCore.pyqtSlot()
    def SpikeRasterBottomFrameControlsMixin_connectSignals(self, bottom_bar_controls):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        bottom_bar_connections = []
        bottom_bar_connections.append(bottom_bar_controls.play_pause_toggled.connect(self.play_pause))
        bottom_bar_connections.append(bottom_bar_controls.jump_left.connect(self.on_jump_left))
        bottom_bar_connections.append(bottom_bar_controls.jump_right.connect(self.on_jump_right))
        bottom_bar_connections.append(bottom_bar_controls.reverse_toggled.connect(self.on_reverse_held))
        return bottom_bar_connections
        
            
    @QtCore.pyqtSlot()
    def SpikeRasterBottomFrameControlsMixin_on_buildUI(self):
        """ perfrom setup/creation of widget/graphical/data objects. Only the core objects are expected to exist on the implementor (root widget, etc) """
        # CALLED:
        
        # controls_frame = QtWidgets.QFrame()
        # controls_layout = QtWidgets.QHBoxLayout() # H-box layout
        
        # # controls_layout = QtWidgets.QGridLayout()
        # # controls_layout.setContentsMargins(0, 0, 0, 0)
        # # controls_layout.setVerticalSpacing(0)
        # # controls_layout.setHorizontalSpacing(0)
        # # controls_layout.setStyleSheet("background : #1B1B1B; color : #727272")
        
        # # Set-up the rest of the Qt window
        # button = QtWidgets.QPushButton("My Button makes the cone red")
        # button.setToolTip('This is an example button')
        # button.clicked.connect(self.onClick)
        # controls_layout.addWidget(button)
        
        # button2 = QtWidgets.QPushButton("<")
        # button2.setToolTip('<')
        # # button2.clicked.connect(self.onClick)
        # controls_layout.addWidget(button2)
        
        # button3 = QtWidgets.QPushButton(">")
        # button3.setToolTip('>')
        # controls_layout.addWidget(button3)
        
        # # Set Final Layouts:
        # controls_frame.setLayout(controls_layout)
        
        controls_frame = Spike3DRasterBottomPlaybackControlBar() # Initialize new controls class from the Spike3DRasterBottomPlaybackControlBar class.
        controls_layout = controls_frame.layout() # Get the layout
        
        self.SpikeRasterBottomFrameControlsMixin_connectSignals(controls_frame)

        
        return controls_frame, controls_layout


    @QtCore.pyqtSlot()
    def SpikeRasterBottomFrameControlsMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        # TODO: NOT CALLED
        pass

    @QtCore.pyqtSlot(float, float)
    def SpikeRasterBottomFrameControlsMixin_on_window_update(self, new_start=None, new_end=None):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc. """
        # TODO: NOT CALLED
        pass
    
    
    # ## Update Functions:
    # @QtCore.pyqtSlot(bool)
    # def play_pause(self, is_playing):
    #     print(f'SpikeRasterBottomFrameControlsMixin.play_pause(is_playing: {is_playing})')
    #     if (not is_playing):
    #         self.animationThread.start()
    #     else:
    #         self.animationThread.terminate()

    # @QtCore.pyqtSlot()
    # def on_jump_left(self):
    #     # Skip back some frames
    #     print(f'SpikeRasterBottomFrameControlsMixin.on_jump_left()')
    #     self.shift_animation_frame_val(-5)
        
    # @QtCore.pyqtSlot()
    # def on_jump_right(self):
    #     # Skip forward some frames
    #     print(f'SpikeRasterBottomFrameControlsMixin.on_jump_right()')
    #     self.shift_animation_frame_val(5)
        

    # @QtCore.pyqtSlot(bool)
    # def on_reverse_held(self, is_reversed):
    #     print(f'SpikeRasterBottomFrameControlsMixin.on_reverse_held(is_reversed: {is_reversed})')
    #     pass





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
