# Spike3DRasterBottomPlaybackControlBar.py
# Generated from c:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\PlaybackControls\Spike3DRasterBottomPlaybackControlBar.ui automatically by PhoPyQtClassGenerator VSCode Extension
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import numpy as np
from enum import Enum

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QToolTip, QStackedWidget, QHBoxLayout, QVBoxLayout, QSplitter, QFormLayout, QLabel, QFrame, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QHeaderView
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QRect, QObject, QEvent, pyqtSignal, pyqtSlot, QSize, QDir, QTime

## IMPORTS:
# from ...pyPhoPlaceCellAnalysis.src.pyphoplacecellanalysis.GUI.Qt.PlaybackControls import Spike3DRasterBottomPlaybackControlBar

# from pyphoplacecellanalysis.GUI.Qt.PlaybackControls.Spike3DRasterBottomPlaybackControlBarBase import Ui_RootWidget  # Generated file from .ui
# from pyphoplacecellanalysis.GUI.Qt.PlaybackControls.Uic_AUTOGEN_Spike3DRasterBottomPlaybackControlBarBase import Ui_RootWidget


# Custom Widget classes
from pyphocorehelpers.gui.Qt.HighlightedJumpSlider import HighlightedJumpSlider
from pyphocorehelpers.gui.Qt.ToggleButton import ToggleButtonModel, ToggleButton
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot

# For extra button symbols:
import qtawesome as qta

from pyphocorehelpers.DataStructure.logging_data_structures import LoggingBaseClass
from pyphoplacecellanalysis.GUI.Qt.Mixins.ComboBoxMixins import KeysListAccessingMixin, ComboBoxCtrlOwningMixin
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes
from pyphoplacecellanalysis.GUI.Qt.Widgets.Testing.LoggingOutputWidget.LoggingOutputWidget import LoggingOutputWidget
from pyphocorehelpers.DataStructure.logging_data_structures import LoggingBaseClass, LoggingBaseClassLoggerOwningMixin

from pyphocorehelpers.DataStructure.general_parameter_containers import DebugHelper, VisualizationParameters, RenderPlots, RenderPlotsData
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.gui.Qt.connections_container import ConnectionsContainer


""" TODO: Refactor from pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\Mixins\RenderWindowControlsMixin.py

RenderWindowControlsMixin
RenderPlaybackControlsMixin




btnJumpToPrevious
comboActiveJumpTargetSeries - Combo box that allows the user to specify the "jump to" target
btnJumpToNext

# TODO:

btnCurrentIntervals_Remove
btnCurrentIntervals_Customize


btnJumpToSpecifiedTime
jumpToHourMinSecTimeEdit



# Edit Double Fields _________________________________________________________________________________________________ #
btnEditNumberField_Revert
btnEditNumberField_Toggle
doubleSpinBox_ActiveWindowStartTime
doubleSpinBox_ActiveWindowEndTime



## Logging Widgets:
txtLogLine -- shows a preview of the last log entries
btnToggleExternalLogWindow -- opens the separate logging window

playback_controls = [self.ui.button_play_pause, self.ui.button_reverse, self.ui.horizontalSpacer_5]

speed_controls = [self.ui.button_slow_down, self.ui.doubleSpinBoxPlaybackSpeed, self.ui.toolButton_SpeedBurstEnabled, self.ui.button_speed_up, self.ui.horizontalSpacer_6]

mark_controls = [self.ui.button_mark_start, self.ui.button_mark_end, self.ui.horizontalSpacer_2]

epoch_controls = [self.ui.frame_Epoch, self.ui.btnJumpToPrevious, self.ui.comboActiveJumpTargetSeries, self.ui.btnJumpToUnused, self.ui.btnCurrentIntervals_Extra, self.ui.btnCurrentIntervals_Remove, self.ui.btnCurrentIntervals_Customize,
    self.ui.btnJumpToNext, self.ui.horizontalSpacer_2]

jump_to_destination_controls = [self.ui.frame_JumpToDestination, self.ui.spinBoxJumpDestination, self.ui.btnJumpToDestination]

move_controls = [self.ui.btnSkipLeft, self.ui.btnLeft, self.ui.spinBoxFrameJumpMultiplier, self.ui.btnRight, self.ui.btnSkipRight, self.ui.btnJoystickMove, self.ui.horizontalSpacer_3]

debug_log_controls = [self.ui.txtLogLine, self.ui.btnToggleExternalLogWindow]

standalone_extra_controls = [self.ui.btnHelp, self.ui.btnToggleRightSidebar]


# Joystick/Move controls _____________________________________________________________________________________________ #
self.ui.btnJoystickMove
.on_joystick_delta_state_changed
.sig_joystick_delta_occured (float, float)



# Dock/Track Controls ________________________________________________________________________________________________ #
btnAddDockedTrack





"""
## Define the .ui file path
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'Spike3DRasterBottomPlaybackControlBarBase.ui')

@metadata_attributes(short_name=None, tags=['logging', 'widget'], input_requires=[], output_provides=[], uses=['LoggingOutputWidget', 'LoggingBaseClass'], used_by=[], creation_date='2025-01-06 12:04', related_items=[])
class Spike3DRasterBottomPlaybackControlBar(ComboBoxCtrlOwningMixin, QWidget):
    """ A playback bar with buttons loaded from a Qt .ui file. """
    # _logger = None  # ensure the attribute exists at class scope
    
    play_pause_toggled = QtCore.pyqtSignal(bool) # returns bool indicating whether is_playing
    jump_left = QtCore.pyqtSignal()
    jump_right = QtCore.pyqtSignal()
    reverse_toggled = QtCore.pyqtSignal(bool) # returns bool indicating whether is_reversed
        
    # Jump Target Items
    jump_target_left = QtCore.pyqtSignal(str)
    jump_target_right = QtCore.pyqtSignal(str)
    jump_series_selection_changed = QtCore.pyqtSignal(str)
    
    jump_specific_time = QtCore.pyqtSignal(float)
    jump_specific_time_window = QtCore.pyqtSignal(float, float)
    
    # Series Target Actions
    series_remove_pressed = QtCore.pyqtSignal(str)    
    series_customize_pressed = QtCore.pyqtSignal(str)

    series_clear_all_pressed = QtCore.pyqtSignal()
    series_add_pressed = QtCore.pyqtSignal()

    sig_joystick_delta_occured = QtCore.pyqtSignal(float, float) # dx, dy

    sigToggleRightSidebarVisibility = QtCore.pyqtSignal(bool)

    sigAddDockedTrackRequested = QtCore.pyqtSignal()


    ## Editing Start/End Window
    sigManualEditWindowStartEndToggled = QtCore.pyqtSignal(bool) # whether the user is manually editing


    def __init__(self, parent=None):
        # super().__init__(parent=parent) # Call the inherited classes __init__ method
        QWidget.__init__(self, parent=parent)
        self.ui = uic.loadUi(uiFile, self) # Load the .ui file
        self.ui.connections = ConnectionsContainer() 
        self.params = VisualizationParameters(name='Spike3DRasterBottomPlaybackControlBar', debug_print=False)  
        
        # # Auto
        # self.ui = Ui_RootWidget()
        # self.ui.setupUi(self) # builds the design from the .ui onto this widget.
        
        self._logger = LoggingBaseClass(log_records=[])
        
        self.initUI()
        self.show() # Show the GUI

    def initUI(self):
        """ setup the UI
        """
   

        move_controls = [self.ui.btnSkipLeft, self.ui.btnLeft, self.ui.spinBoxFrameJumpMultiplier, self.ui.btnRight, self.ui.btnSkipRight] # , self.ui.horizontalSpacer_3
        # debug_log_controls = [self.ui.txtLogLine, self.ui.btnToggleExternalLogWindow]
        standalone_extra_controls = [self.ui.btnHelp]
        
        controls_to_hide = [self.ui.slider_progress, self.ui.button_full_screen, self.ui.btnCurrentIntervals_Customize, *standalone_extra_controls] # , *move_controls


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
        
        self._INIT_UI_initialize_jump_time_edit()
        self._INIT_UI_initialize_active_window_time_double_spinboxes()
        

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

        ## Hide any controls in `controls_to_hide`
        for a_ctrl in controls_to_hide:
            a_ctrl.hide()
        
        # debug_log_controls _________________________________________________________________________________________________ #
        self.ui._attached_log_window = None
        self.ui.connections['_logger_sigLogUpdated'] = self._logger.sigLogUpdated.connect(self.on_log_updated)
        self.ui.connections['_logger_sigLogUpdateFinished'] = self._logger.sigLogUpdateFinished.connect(self.on_log_update_finished)


        # debug_log_controls = [self.ui.txtLogLine, self.ui.btnToggleExternalLogWindow]
        self._format_button_toggle_log_window()
        self.ui.connections['btnToggleExternalLogWindow_pressed'] = self.ui.btnToggleExternalLogWindow.pressed.connect(self.toggle_log_window)

        # Help/Utility Controls and Buttons __________________________________________________________________________________ #
        self.ui.connections['btnHelp_pressed'] = self.ui.btnHelp.pressed.connect(self.on_help_button_pressed)
        
        # pg.JoystickButton
        self.ui.connections['btnJoystickMove_sigStateChanged'] = self.ui.btnJoystickMove.sigStateChanged.connect(self.on_joystick_delta_state_changed)
        
        self.ui.connections['btnAddDockedTrack_pressed'] = self.ui.btnAddDockedTrack.pressed.connect(self.on_add_docked_track_widget_button_pressed)

        self.ui.btnToggleRightSidebar.pressed.connect(self.on_right_sidebar_toggle_button_pressed)

    def on_joystick_delta_state_changed(self, joystick_ctrl, new_state):
        print(f"on_joystick_delta_state_changed(joystick_ctrl: {joystick_ctrl} new_state: {new_state})")
        # new_state = joystick_ctrl.getState()
        dx, dy = new_state
        print(f'\tdx: {dx}, dy: {dy}')
        if ((abs(dx) > 0) or (abs(dy) > 0)):
            self.sig_joystick_delta_occured.emit(dx, dy)
        
        # x += dx * 1e-3
        # y += dy * 1e-3



    @function_attributes(short_name=None, tags=['format', 'button'], input_requires=[], output_provides=[], uses=[], used_by=['_format_button_reversed', '_format_button_toggle_log_window'], creation_date='2025-01-06 08:39', related_items=[])
    def _format_boolean_toggle_button(self, button):
        """ formats the any checkable button based on whether it is checked or not """
        if button.isChecked():
            # setting background color to an energized blue with white text
            button.setStyleSheet("color : rgb(255, 255, 255); background-color : rgb(85, 170, 255);")
        # if it is unchecked
        else:
            # set background color back to default (dark grey) with grey text
            button.setStyleSheet("color : rgb(180, 180, 180); background-color : rgb(65, 60, 54)")
            

        
    # def __str__(self):
    #      return 
    
    # Called when the play/pause button is clicked:
    def play_pause(self):
        # THIS IS NOT TRUE: the play_pause_model uses inverted logic, so we negate the current state value to determine if is_playing
        is_playing = self.ui.play_pause_model.getState()
        self.log_print(f'is_playing: {is_playing}')
        self.play_pause_toggled.emit(is_playing)

    def on_jump_left(self):
        # Skip back some frames
        self.jump_left.emit()
        if self.params.debug_print:
            self.log_print(f'on_jump_left()')
        # self.shift_animation_frame_val(-5)
        pass

    def on_jump_right(self):
        # Skip forward some frames
        self.jump_right.emit()
        if self.params.debug_print:
            self.log_print(f'on_jump_left()')
        # self.shift_animation_frame_val(5)
        pass        

    def on_jump_specified_hour_min_sec(self):
        self.on_jump_time_editing_finished()
        
    
    def on_reverse_held(self):
        # Change the direction of playback by changing the sign of the updating.
        # if button is checked    
        self.is_playback_reversed = self.ui.button_reverse.isChecked()
        self._format_button_reversed()

        self.reverse_toggled.emit(self.is_playback_reversed)

    def _format_button_reversed(self):
        """ formats the reverse button based on whether it is checked or not """
        self._format_boolean_toggle_button(button=self.ui.button_reverse)

    # ==================================================================================================================== #
    # Jump Target Controls                                                                                                 #
    # ==================================================================================================================== #
    # @property
    # def combo_jump_target_series(self):
    #     return self.ui.comboActiveJumpTargetSeries

    @classmethod
    def total_fractional_seconds(cls, hours: int = 0, minutes: int = 0, seconds: int = 0, milliseconds: int = 0) -> float:
        """ reciprocal of `.decompose_fractional_seconds(...)`
        Calculate total fractional seconds from hours, minutes, seconds, and milliseconds.

        Args:
            hours (int): Number of hours (default: 0).
            minutes (int): Number of minutes (default: 0).
            seconds (int): Number of seconds (default: 0).
            milliseconds (int): Number of milliseconds (default: 0).

        Returns:
            float: Total time in fractional seconds.
        """
        return (hours * 3600) + (minutes * 60) + seconds + (milliseconds / 1000)

    @classmethod
    def decompose_fractional_seconds(cls, total_seconds: float) -> tuple[int, int, int, int]:
        """ reciprocal of `.total_fractional_seconds(...)`
        Decompose total fractional seconds into hours, minutes, seconds, and milliseconds.

        Args:
            total_seconds (float): Total time in fractional seconds.

        Returns:
            tuple[int, int, int, int]: A tuple containing hours, minutes, seconds, and milliseconds.
        """
        hours = int(total_seconds // 3600)
        total_seconds %= 3600
        minutes = int(total_seconds // 60)
        total_seconds %= 60
        seconds = int(total_seconds)
        milliseconds = int((total_seconds - seconds) * 1000)
        return hours, minutes, seconds, milliseconds

    @property
    def time_edit(self):
        """The time_edit property."""
        if (not hasattr(self, 'ui')) or (not hasattr(self.ui, 'jumpToHourMinSecTimeEdit')):
            return None # not initialized yet
        return self.ui.jumpToHourMinSecTimeEdit

    @property
    def time_fractional_seconds(self) -> float:
        """The time_fractional_seconds property."""
        time_fractional_seconds = None
        try:
            # self.time_edit.blockSignals(True)        
            time = self.time_edit.time()  # Get the QTime object
            time_fractional_seconds = self.total_fractional_seconds(hours=time.hour(), minutes=time.minute(), seconds=time.second(), milliseconds=time.msec())
            # return time_fractional_seconds
        except AttributeError:
            time_fractional_seconds = None
        except Exception as e:
            raise
        # finally:
        #     self.time_edit.blockSignals(False)
        return time_fractional_seconds
    @time_fractional_seconds.setter
    def time_fractional_seconds(self, value):
        time_tuple = self.decompose_fractional_seconds(value)
        assert len(time_tuple) == 4
        # Create a QTime object from the tuple
        time_obj = QTime(time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3])
        try:
            # Set the QTimeEdit's time
            self.time_edit.setTime(time_obj)
        except Exception as e:
            raise
        finally:
            self.time_edit.blockSignals(False)
        

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
        ## Capture the previous selection: `AttributeError: 'Spike3DRasterBottomPlaybackControlBar' object has no attribute 'ui'` 
        # NOTE: has `self.comboActiveJumpTargetSeries` but not `self.ui.comboActiveJumpTargetSeries`
        try:
            selected_index, selected_item_text = self.get_current_combo_item_selection(self.ui.comboActiveJumpTargetSeries, debug_print=False)
        except AttributeError as e:
            # handle missing `self.ui`
            # self.ui.comboActiveJumpTargetSeries
            selected_index, selected_item_text = self.get_current_combo_item_selection(self.comboActiveJumpTargetSeries, debug_print=False)
            pass
            # Alternatively, return (-1, '')
        
        return (selected_index, selected_item_text)


    def update_jump_target_series_options(self, new_options):
        """ called to update the list of jump-target options """
        return self._tryUpdateComboItemsUi(self.ui.comboActiveJumpTargetSeries, new_options)

    def _tryUpdateComboItemsUi(self, curr_combo_box, new_options):
        updated_list = KeysListAccessingMixin.get_keys_list(new_options)
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
            self.log_print(f'WARNING: could not select any default items because the list was empty.')
            found_desired_index = None
        return found_desired_index


    def _update_series_action_buttons(self, has_valid_series_selection: bool):
        """ conditionally update whether the buttons are enabled based on whether we have a valid series selection. """
        self.ui.btnJumpToPrevious.setEnabled(has_valid_series_selection)
        self.ui.btnJumpToNext.setEnabled(has_valid_series_selection)
        self.ui.btnCurrentIntervals_Remove.setEnabled(has_valid_series_selection)
        self.ui.btnCurrentIntervals_Customize.setEnabled(has_valid_series_selection)
        # self.ui.btnCurrentIntervals_Extra.setEnabled(has_valid_series_selection)


    @pyqtExceptionPrintingSlot(object)
    def on_rendered_intervals_list_changed(self, interval_list_owning_object):
        """ called when the list of rendered intervals changes """
        self.update_jump_target_series_options(interval_list_owning_object.list_all_rendered_intervals(debug_print=False))
        if self.params.debug_print:
            self.log_print(f'on_rendered_intervals_list_changed(interval_list_owning_object: {interval_list_owning_object})')

    @pyqtExceptionPrintingSlot(str)
    def on_jump_combo_series_changed(self, series_name):
        if self.params.debug_print:
            self.log_print(f'on_jump_combo_series_changed(series_name: {series_name})')
        curr_jump_series_name = self.current_selected_jump_target_series_name # 'PBEs'
        self._update_series_action_buttons(self.has_valid_current_target_series_name) # enable/disable the action buttons
        self.jump_series_selection_changed.emit(curr_jump_series_name)

    @QtCore.pyqtSlot()
    def on_jump_next_series_item(self):
        """ seeks the current active_time_Window to the start of the next epoch event (for the epoch event series specified in the bottom bar) 

            By default, snap the start of the active_time_window to the start of the next epoch event
        """
        curr_jump_series_name = self.current_selected_jump_target_series_name # 'PBEs'
        self.log_print(f'on_jump_next_series_item(): curr_jump_series_name: {curr_jump_series_name}')
        self.jump_target_right.emit(curr_jump_series_name)

    @QtCore.pyqtSlot()
    def on_jump_prev_series_item(self):
        """ seeks the current active_time_Window to the start of the next epoch event (for the epoch event series specified in the bottom bar) 

            By default, snap the start of the active_time_window to the start of the next epoch event
        """
        curr_jump_series_name = self.current_selected_jump_target_series_name # 'PBEs'
        if self.params.debug_print:
            self.log_print(f'on_jump_prev_series_item(): curr_jump_series_name: {curr_jump_series_name}')
        self.jump_target_left.emit(curr_jump_series_name)


    @QtCore.pyqtSlot()
    def on_series_remove_button_pressed(self):
        """ 
        """
        curr_series_name = self.current_selected_jump_target_series_name # 'PBEs'
        if self.params.debug_print:
            self.log_print(f'on_series_remove_button_pressed(): curr_series_name: {curr_series_name}')
        self.series_remove_pressed.emit(curr_series_name)

    @QtCore.pyqtSlot()
    def on_series_customize_button_pressed(self):
        """ 
        """
        curr_series_name = self.current_selected_jump_target_series_name # 'PBEs'
        if self.params.debug_print:
            self.log_print(f'on_series_customize_button_pressed(): curr_series_name: {curr_series_name}')
        self.series_customize_pressed.emit(curr_series_name)


    @QtCore.pyqtSlot()
    def on_action_clear_all_pressed(self):
        """ 
        """
        if self.params.debug_print:
            self.log_print(f'on_action_clear_all_pressed()')
        self.series_clear_all_pressed.emit()

    @QtCore.pyqtSlot()
    def on_action_request_add_intervals_pressed(self):
        """ 
        """
        self.log_print(f'Spike3DRasterBottomPlaybackControlBar.on_action_request_add_intervals_pressed()')
        self.series_add_pressed.emit()


    # @QtCore.pyqtSlot()
    # def on_series_extra_button_pressed(self):
    #     """ 
    #     """
    #     print(f'on_series_extra_button_pressed()')
    #     # self.series_remove_pressed.emit(curr_series_name)
    

    def _INIT_UI_initialize_jump_time_edit(self):
        """ sets up `jumpToHourMinSecTimeEdit` and `btnJumpToSpecifiedTime`
        
        """
        # Set initial stylesheet
        self.set_jump_time_light_grey_style()
        
        self.ui.btnJumpToSpecifiedTime.clicked.connect(self.on_jump_specified_hour_min_sec)

        # Connect signals to handle focus and editing states
        self.ui.jumpToHourMinSecTimeEdit.editingFinished.connect(self.on_jump_time_editing_finished)
        self.ui.jumpToHourMinSecTimeEdit.installEventFilter(self)
        
        

    def on_jump_time_editing_finished(self):
        """Editing of the time has finished """
        time_fractional_seconds: float = self.time_fractional_seconds # self.total_fractional_seconds(hours=time.hour(), minutes=time.minute(), seconds=time.second(), milliseconds=time.msec())
        self.log_print(f'time_fractional_seconds: {time_fractional_seconds}')
        
        self.ui.jumpToHourMinSecTimeEdit.clearFocus()
        self.set_jump_time_light_grey_style()

        ## emit the event
        self.jump_specific_time.emit(time_fractional_seconds)
        
        

    def set_jump_time_light_grey_style(self):
        """Set the light grey inactive style."""
        self.ui.jumpToHourMinSecTimeEdit.setStyleSheet("""
            QTimeEdit {
                color: lightgrey;
            }
        """)

    def set_jump_time_white_style(self):
        """Set the white active style."""
        self.ui.jumpToHourMinSecTimeEdit.setStyleSheet("""
            QTimeEdit {
                color: white;
            }
        """)
        

    # ==================================================================================================================== #
    # Start/End Double Spin Boxes                                                                                          #
    # ==================================================================================================================== #
    #     
    def _INIT_UI_initialize_active_window_time_double_spinboxes(self):
        """ sets up `doubleSpinBox_ActiveWindowStartTime` and `doubleSpinBox_ActiveWindowEndTime`
        
        """
        # Connect toggle button to toggle edit mode
        self.ui.btnEditNumberField_Revert.setVisible(False) ## hide
        self.ui.btnEditNumberField_Toggle.toggled.connect(self.on_edit_number_field_toggle_changed)
        self.ui.btnEditNumberField_Revert.pressed.connect(self.on_edit_number_field_revert_button_pressed)
        
        # Connect signals to handle focus and editing states
        self.ui.doubleSpinBox_ActiveWindowStartTime.editingFinished.connect(self.on_active_window_start_time_editing_finished)
        self.ui.doubleSpinBox_ActiveWindowStartTime.installEventFilter(self)
        
        self.ui.doubleSpinBox_ActiveWindowEndTime.editingFinished.connect(self.on_active_window_end_time_editing_finished)
        self.ui.doubleSpinBox_ActiveWindowEndTime.installEventFilter(self)
        
        # Initialize in non-editable mode
        self.on_start_end_doubleSpinBox_edit_mode_changed(False)
        

    @pyqtExceptionPrintingSlot()
    def on_edit_number_field_revert_button_pressed(self):
        """Handles when the edit number field toggle button is toggled
        
        """
        self.log_print(f'on_edit_number_field_revert_button_pressed()')
        
        # Format the toggle button based on checked state
        self._format_boolean_toggle_button(button=self.ui.btnEditNumberField_Toggle)
        
        # Update the editability of the spinboxes
        self.on_start_end_doubleSpinBox_edit_mode_changed(is_checked)
        
        # Emit signal to notify other components
        self.sigManualEditWindowStartEndToggled.emit(is_checked)
        
        # If editing is enabled, set focus to the start time spinbox
        if is_checked:
            self.ui.doubleSpinBox_ActiveWindowStartTime.setFocus()
            

    @pyqtExceptionPrintingSlot(bool)
    def on_edit_number_field_toggle_changed(self, is_checked):
        """Handles when the edit number field toggle button is toggled
        
        Args:
            is_checked (bool): Whether the button is toggled/checked
        """
        self.log_print(f'on_edit_number_field_toggle_changed(is_checked: {is_checked})')
        
        # Format the toggle button based on checked state
        self._format_boolean_toggle_button(button=self.ui.btnEditNumberField_Toggle)
        
        # Update the editability of the spinboxes
        self.on_start_end_doubleSpinBox_edit_mode_changed(is_checked)
        
        # Emit signal to notify other components
        self.sigManualEditWindowStartEndToggled.emit(is_checked)
        
        # If editing is enabled, set focus to the start time spinbox
        if is_checked:
            self.ui.doubleSpinBox_ActiveWindowStartTime.setFocus()


    def on_start_end_doubleSpinBox_edit_mode_changed(self, are_controls_editable: bool):
        """ called to enable user editing of the two doubleSpinBox controls for start/end times 
        """
        print(f'Spike3DRasterBottomPlaybackControlBar.on_start_end_doubleSpinBox_edit_mode_changed(are_controls_editable: {are_controls_editable})')
        if not are_controls_editable:
            # Clear focus from both spinboxes
            self.ui.doubleSpinBox_ActiveWindowStartTime.clearFocus()
            self.ui.doubleSpinBox_ActiveWindowEndTime.clearFocus()
            # Deselect any selected text using the underlying QLineEdit
            self.ui.doubleSpinBox_ActiveWindowStartTime.lineEdit().deselect()
            self.ui.doubleSpinBox_ActiveWindowEndTime.lineEdit().deselect()
                    
        if are_controls_editable:
            focus_policy = QtCore.Qt.ClickFocus
        else:
            ## not editable
            focus_policy = QtCore.Qt.NoFocus
            
        self.ui.doubleSpinBox_ActiveWindowStartTime.setReadOnly(not are_controls_editable)
        self.ui.doubleSpinBox_ActiveWindowStartTime.setKeyboardTracking(are_controls_editable)
        self.ui.doubleSpinBox_ActiveWindowStartTime.setFocusPolicy(focus_policy)
        
        self.ui.doubleSpinBox_ActiveWindowEndTime.setReadOnly(not are_controls_editable)
        self.ui.doubleSpinBox_ActiveWindowEndTime.setKeyboardTracking(are_controls_editable)
        self.ui.doubleSpinBox_ActiveWindowEndTime.setFocusPolicy(focus_policy)


    def on_active_window_start_time_editing_finished(self):
        """Editing of the time has finished """
        start_t_seconds: float = float(self.ui.doubleSpinBox_ActiveWindowStartTime.value())
        end_t_seconds: float = float(self.ui.doubleSpinBox_ActiveWindowEndTime.value())
        self.log_print(f'start_t_seconds: {start_t_seconds}, end_t_seconds: {end_t_seconds}')
        
        self.ui.doubleSpinBox_ActiveWindowStartTime.clearFocus()
        ## emit the event
        self.jump_specific_time_window.emit(start_t_seconds, end_t_seconds)


    def on_active_window_end_time_editing_finished(self):
        """Editing of the time has finished """
        start_t_seconds: float = float(self.ui.doubleSpinBox_ActiveWindowStartTime.value())
        end_t_seconds: float = float(self.ui.doubleSpinBox_ActiveWindowEndTime.value())
        self.log_print(f'start_t_seconds: {start_t_seconds}, end_t_seconds: {end_t_seconds}')
        
        self.ui.doubleSpinBox_ActiveWindowEndTime.clearFocus()
        ## emit the event
        self.jump_specific_time_window.emit(start_t_seconds, end_t_seconds)
        

        
        

    @pyqtExceptionPrintingSlot(float, float)
    def on_window_changed(self, start_t, end_t):
        if self.params.debug_print:
            self.log_print(f'Spike3DRasterBottomPlaybackControlBar.on_time_window_changed(start_t: {start_t}, end_t: {end_t})')
        # need to block signals:
        self.ui.doubleSpinBox_ActiveWindowStartTime.blockSignals(True)
        self.ui.doubleSpinBox_ActiveWindowEndTime.blockSignals(True)
        if (start_t is not None):
            self.ui.doubleSpinBox_ActiveWindowStartTime.setValue(start_t)
            # self.ui.jumpToHourMinSecTimeEdit.blockSignals(True)
            # self.ui.jumpToHourMinSecTimeEdit.setValue(
            self.time_fractional_seconds = start_t
            # start_time_hour_min_sec_tuple = self.decompose_fractional_seconds(start_t)
            # assert len(start_time_hour_min_sec_tuple) == 4
            # # Create a QTime object from the tuple
            # start_time_obj = QTime(start_time_hour_min_sec_tuple[0], start_time_hour_min_sec_tuple[1], start_time_hour_min_sec_tuple[2], start_time_hour_min_sec_tuple[3])
            # # Set the QTimeEdit's time
            # self.time_edit.setTime(start_time_obj)
            # self.ui.jumpToHourMinSecTimeEdit.blockSignals(False)

        if (end_t is not None):
            self.ui.doubleSpinBox_ActiveWindowEndTime.setValue(end_t)
        self.ui.doubleSpinBox_ActiveWindowStartTime.blockSignals(False) # unblock the signals when done
        self.ui.doubleSpinBox_ActiveWindowEndTime.blockSignals(False)
        # self.ui.doubleSpinBox_ActiveWindowStartTime.setValue(start_t)
        # self.ui.doubleSpinBox_ActiveWindowEndTime.setValue(end_t)
        if self.params.debug_print:
            self.log_print(f'\tdone.')
        

    # ==================================================================================================================== #
    # Debug Logging Controls                                                                                               #
    # ==================================================================================================================== #
    # debug_log_controls = [self.ui.txtLogLine, self.ui.btnToggleExternalLogWindow]

    @property
    def logger(self) -> LoggingBaseClass:
        """The logger property."""
        if not hasattr(self, '_logger'):
            return None # not initialized yet
        return self._logger
    @logger.setter
    def logger(self, value: LoggingBaseClass):
        self._logger = value
        
    @property
    def attached_log_window(self) -> Optional[LoggingOutputWidget]:
        """The attached_log_window property."""
        try:
            if not hasattr(self, '_attached_log_window'):
                return None # not initialized yet
            return self._attached_log_window
        except (AttributeError, NameError):
            return None
        


    @pyqtExceptionPrintingSlot(object)
    def on_log_updated(self, logger):
        if self.params.debug_print:
            print(f'Spike3DRasterBottomPlaybackControlBar.on_log_updated(logger: {logger})')
        # logger: LoggingBaseClass
        target_text: str = logger.get_flattened_log_text(flattening_delimiter='|', limit_to_n_most_recent=3)
        self.ui.txtLogLine.setText(target_text)
        ## don't need to update the connected window, as it will update itself
        

    @pyqtExceptionPrintingSlot()
    def on_log_update_finished(self):
        if self.params.debug_print:
            print(f'Spike3DRasterBottomPlaybackControlBar.on_log_update_finished()')
        # logger: LoggingBaseClass
        target_text: str = self.logger.get_flattened_log_text(flattening_delimiter='|', limit_to_n_most_recent=3)
        self.ui.txtLogLine.setText(target_text)
        ## don't need to update the connected window, as it will update itself
        

    def _format_button_toggle_log_window(self):
        """ formats the toggle_log_window button based on whether it is checked or not """
        self._format_boolean_toggle_button(button=self.ui.btnToggleExternalLogWindow)
        

    def toggle_log_window(self):
        """ shows/hides the multi-line log window widget 
        """
        is_external_window_opened: bool = self.ui.btnToggleExternalLogWindow
        if self.params.debug_print:
            print(f'is_external_window_opened: {is_external_window_opened}')
        if (self.ui._attached_log_window is None) and (is_external_window_opened):
            ## open a new one
            self.ui._attached_log_window = LoggingOutputWidget()
            self._logger.sigLogUpdated.connect(self.ui._attached_log_window.on_log_updated)
            self.ui.connections['_attached_log_window'] = {'_logger_sigLogUpdated': self._logger.sigLogUpdated.connect(self.ui._attached_log_window.on_log_updated),
                                                           '_logger_sigLogUpdateFinished': self._logger.sigLogUpdateFinished.connect(self.ui._attached_log_window.on_log_update_finished),
            }
            self.ui._attached_log_window.on_log_updated(self.logger)
            self.ui._attached_log_window.show()
        else:
            if self.params.debug_print:
                print(f'hide.')
            self.ui._attached_log_window.hide()

        self._format_button_toggle_log_window()
        
    @function_attributes(short_name=None, tags=['logging'], input_requires=[], output_provides=[], uses=[], used_by=['add_log_lines'], creation_date='2025-01-06 11:26', related_items=[])
    def add_log_line(self, new_line: str, allow_split_newlines: bool = True, defer_log_changed_event:bool=False):
        """ adds an additional entry to the log """
        if self.params.debug_print:
            print(f'.add_log_line(...): self.logger: {self.logger.get_flattened_log_text()}')
        self.logger.add_log_line(new_line=new_line, allow_split_newlines=allow_split_newlines, defer_log_changed_event=defer_log_changed_event)
            
    @function_attributes(short_name=None, tags=['logging'], input_requires=[], output_provides=[], uses=['add_log_line'], used_by=['log_print'], creation_date='2025-01-06 11:26', related_items=[])
    def add_log_lines(self, new_lines: List[str], allow_split_newlines: bool = True, defer_log_changed_event:bool=False):
        """ adds an additional entries to the log """
        if self.params.debug_print:
            print(f'.add_log_lines(...): self.logger: {self.logger.get_flattened_log_text()}')
        self.logger.add_log_lines(new_lines=new_lines, allow_split_newlines=allow_split_newlines, defer_log_changed_event=defer_log_changed_event)
                    
    @function_attributes(short_name=None, tags=['logging', 'print'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-06 11:25', related_items=[])
    def log_print(self, *args):
        """ adds an additional entry to the log """
        print(*args)
        if self.params.debug_print:
            self.add_log_lines(new_lines=args, defer_log_changed_event=False)


    # ==================================================================================================================== #
    # Help/Utility Buttons                                                                                                 #
    # ==================================================================================================================== #
    def on_help_button_pressed(self):
        self.log_print(f'on_help_button_pressed()')
        
    def on_right_sidebar_toggle_button_pressed(self):
        print(f'Spike3DRasterBottomPlaybackControlBar.on_right_sidebar_toggle_button_pressed():')
        should_sidebar_be_visible: bool = self.ui.btnToggleRightSidebar.isChecked()
        print(f'\tshould_sidebar_be_visible: {should_sidebar_be_visible}')
        
        self.sigToggleRightSidebarVisibility.emit(should_sidebar_be_visible)

    def on_add_docked_track_widget_button_pressed(self):
        """ btnAddDockedTrack """
        self.log_print(f'on_add_docked_track_widget_button_pressed()')
        self.sigAddDockedTrackRequested.emit()

    # ==================================================================================================================== #
    # eventFilter                                                                                                          #
    # ==================================================================================================================== #


    def eventFilter(self, source, event):
        """Handle focus events to change styles dynamically."""
        if source == self.ui.jumpToHourMinSecTimeEdit:
            if event.type() == event.FocusIn:
                self.set_jump_time_white_style()
            elif event.type() == event.FocusOut:
                self.set_jump_time_light_grey_style()
                
        elif source == self.time_edit and event.type() == event.KeyPress:
            # """Handle Enter key to finalize and lose focus."""
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.time_edit.clearFocus()  # Finalize and lose focus
                return True  # Mark event as handled
            
        elif (source == self.ui.doubleSpinBox_ActiveWindowStartTime) or (source == self.ui.doubleSpinBox_ActiveWindowEndTime):
            if event.type() == event.KeyPress:
                # """Handle Enter key to finalize and lose focus."""
                if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                    source.clearFocus()  # Finalize and lose focus
                    return True  # Mark event as handled
                else: 
                    ## other key presses, such as type numbers and such
                    pass

        if self.params.debug_print:
            print(f'Spike3DRasterBottomPlaybackControlBar.eventFilter(source: {source}, event: {event})')
            
        return super().eventFilter(source, event)
    

@metadata_attributes(short_name=None, tags=['bottom', 'ui', 'owner'], input_requires=[], output_provides=[], uses=[], used_by=['Spike3DRasterWindowWidget'], creation_date='2025-01-07 00:00', related_items=[])
class SpikeRasterBottomFrameControlsMixin(LoggingBaseClassLoggerOwningMixin):
    """ renders the UI controls for the Spike3DRaster_Vedo class 
        Follows Conventions outlined in ModelViewMixin Conventions.md
        
        Used in `Spike3DRasterWindowWidget`
        
    """
    
    # @QtCore.Property(object) # # Note that this ia *pyqt*Property, meaning it's available to pyqt
    @property
    def bottom_playback_control_bar_widget(self) -> Spike3DRasterBottomPlaybackControlBar:
        """The bottom_playback_control_bar_widget property."""
        return self.ui.bottomPlaybackControlBarWidget
        
    @property
    def bottom_playback_control_bar_logger(self) -> LoggingBaseClass:
        """The logger property."""
        return self.bottom_playback_control_bar_widget.logger

    # @property
    # def logger(self) -> LoggingBaseClass:
    #     """The logger property."""
    #     if not hasattr(self, '_logger'):
    #         return None # not initialized yet
    #     return self._logger
    # @logger.setter
    # def logger(self, value: LoggingBaseClass):
    #     self._logger = value
        
    # @property
    # def attached_log_window(self) -> Optional[LoggingOutputWidget]:
    #     """The attached_log_window property."""
    #     try:
    #         if not hasattr(self, '_attached_log_window'):
    #             return None # not initialized yet
    #         return self._attached_log_window
    #     except (AttributeError, NameError):
    #         return None


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
        
        controls_frame = Spike3DRasterBottomPlaybackControlBar() # Initialize new controls class from the Spike3DRasterBottomPlaybackControlBar class.
        controls_layout = controls_frame.layout() # Get the layout
        
        self.SpikeRasterBottomFrameControlsMixin_connectSignals(controls_frame)

        
        return controls_frame, controls_layout


    @QtCore.pyqtSlot()
    def SpikeRasterBottomFrameControlsMixin_on_destroy(self):
        """ perfrom teardown/destruction of anything that needs to be manually removed or released """
        # TODO: NOT CALLED
        pass

    @pyqtExceptionPrintingSlot(float, float)
    def SpikeRasterBottomFrameControlsMixin_on_window_update(self, new_start=None, new_end=None):
        """ called to perform updates when the active window changes. Redraw, recompute data, etc. 
        called from: `.update_animation(...)`
        
        """
        # Called the Implementor's update_window(...) function
        print(f'SpikeRasterBottomFrameControlsMixin_on_window_update(new_start: {new_start}, new_end: {new_end}')
        #TODO 2023-11-21 18:49: - [ ] Doesn't work :[
        # need to block signals:
        # doubleSpinBox_ActiveWindowStartTime.blockSignals(True)
        # doubleSpinBox_ActiveWindowEndTime.blockSignals(True)
        # if new_start is not None:
        #     self.ui.doubleSpinBox_ActiveWindowStartTime.setValue(new_start)
        # if new_end is not None:
        #     self.ui.doubleSpinBox_ActiveWindowEndTime.setValue(new_end)
        # doubleSpinBox_ActiveWindowStartTime.blockSignals(False) # unblock the signals when done
        # doubleSpinBox_ActiveWindowEndTime.blockSignals(False)
        self.bottom_playback_control_bar_widget.on_window_changed(new_start, new_end)
        # if new_start is not None:
        #     ## update the jump time when it scrolls
        #     ## should rate-limit it:
        #     # self.time_fractional_seconds = new_start
        #     # doubleSpinBox_ActiveWindowStartTime
        #     pass
        
        pass
    
    # ## Update Functions:
    # @pyqtExceptionPrintingSlot(bool)
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
        

    # @pyqtExceptionPrintingSlot(bool)
    # def on_reverse_held(self, is_reversed):
    #     print(f'SpikeRasterBottomFrameControlsMixin.on_reverse_held(is_reversed: {is_reversed})')
    #     pass


    # ==================================================================================================================== #
    # LoggingBaseClassLoggerOwningMixin Implementation                                                                     #
    # ==================================================================================================================== #
    


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
