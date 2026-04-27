# Interactive3dSpikeBehaviorOptionsWidget.py
"""
A PyQt-based options widget that drives the `InteractivePlaceCellDataExplorer`
("3d_interactive_spike_and_behavior_browser") in real time.

It exposes toggles and spinboxes for:
    - visibility of historical / recent spike dots
    - visibility of the trajectory trail and current-position marker
    - duration (seconds) of the recent (trail) and historical spike windows
    - point-size and opacity ranges of the fading trajectory trail
    - a manual refresh button that re-renders the meshes at the current slider position

The widget operates entirely through the explorer's existing public API:
    explorer.params, explorer.spikes_main_historical, explorer.spikes_main_recent_only,
    explorer.animal_location_trail, explorer.animal_current_location_point,
    explorer.on_slider_update_mesh, explorer.on_active_window_update_mesh
"""
import numpy as np
from typing import Optional

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot

from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphoplacecellanalysis.PhoPositionalData.plotting.visualization_window import VisualizationWindow


@metadata_attributes(short_name=None, tags=['gui', 'qt', 'widget', 'options', '3d', 'interactive', 'spikes'], input_requires=[], output_provides=[], uses=['InteractivePlaceCellDataExplorer'], used_by=['_display_3d_interactive_spike_and_behavior_browser'], creation_date='2026-04-27 05:40', related_items=[])
class Interactive3dSpikeBehaviorOptionsWidget(QtWidgets.QWidget):
    """ Real-time options panel for the 3D interactive spike-and-behavior browser.

    Usage:
        from pyphoplacecellanalysis.GUI.Qt.Widgets.Interactive3dSpikeBehaviorOptionsWidget.Interactive3dSpikeBehaviorOptionsWidget import Interactive3dSpikeBehaviorOptionsWidget
        optionsWidget = Interactive3dSpikeBehaviorOptionsWidget.build_for_explorer(ipspikesDataExplorer)
        optionsWidget.show()
    """

    sigOptionsChanged = pyqtSignal(dict)


    def __init__(self, explorer, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent=parent)
        self.explorer = explorer
        self._suspend_signals = False
        self._build_ui()
        self._initialize_from_explorer()
        self._connect_signals()


    @classmethod
    def build_for_explorer(cls, explorer) -> "Interactive3dSpikeBehaviorOptionsWidget":
        """ Build a new widget bound to the provided InteractivePlaceCellDataExplorer instance. """
        widget = cls(explorer=explorer)
        widget.setWindowTitle("3D Spike & Behavior Options")
        return widget


    def _build_ui(self):
        outerLayout = QtWidgets.QVBoxLayout(self)
        outerLayout.setContentsMargins(6, 6, 6, 6)
        outerLayout.setSpacing(6)

        spikesGroup = QtWidgets.QGroupBox("Spikes")
        spikesForm = QtWidgets.QFormLayout(spikesGroup)
        self.chkHistoricalSpikes = QtWidgets.QCheckBox("Show historical spikes")
        self.chkRecentSpikes = QtWidgets.QCheckBox("Show recent (active) spikes")
        self.spnHistoricalDuration = QtWidgets.QDoubleSpinBox()
        self.spnHistoricalDuration.setRange(0.5, 4096.0)
        self.spnHistoricalDuration.setDecimals(2)
        self.spnHistoricalDuration.setSingleStep(1.0)
        self.spnHistoricalDuration.setSuffix(" s")
        spikesForm.addRow(self.chkHistoricalSpikes)
        spikesForm.addRow(self.chkRecentSpikes)
        spikesForm.addRow("Historical window:", self.spnHistoricalDuration)
        outerLayout.addWidget(spikesGroup)

        trajGroup = QtWidgets.QGroupBox("Trajectory")
        trajForm = QtWidgets.QFormLayout(trajGroup)
        self.chkTrajectoryTrail = QtWidgets.QCheckBox("Show trajectory trail")
        self.chkCurrentPosition = QtWidgets.QCheckBox("Show current position")
        self.spnTrailDuration = QtWidgets.QDoubleSpinBox()
        self.spnTrailDuration.setRange(0.5, 120.0)
        self.spnTrailDuration.setDecimals(2)
        self.spnTrailDuration.setSingleStep(0.5)
        self.spnTrailDuration.setSuffix(" s")
        self.spnTrailMinSize = QtWidgets.QDoubleSpinBox()
        self.spnTrailMinSize.setRange(0.0, 50.0)
        self.spnTrailMinSize.setDecimals(2)
        self.spnTrailMinSize.setSingleStep(0.1)
        self.spnTrailMaxSize = QtWidgets.QDoubleSpinBox()
        self.spnTrailMaxSize.setRange(0.0, 50.0)
        self.spnTrailMaxSize.setDecimals(2)
        self.spnTrailMaxSize.setSingleStep(0.1)
        self.spnTrailMinOpacity = QtWidgets.QDoubleSpinBox()
        self.spnTrailMinOpacity.setRange(0.0, 1.0)
        self.spnTrailMinOpacity.setDecimals(2)
        self.spnTrailMinOpacity.setSingleStep(0.05)
        self.spnTrailMaxOpacity = QtWidgets.QDoubleSpinBox()
        self.spnTrailMaxOpacity.setRange(0.0, 1.0)
        self.spnTrailMaxOpacity.setDecimals(2)
        self.spnTrailMaxOpacity.setSingleStep(0.05)
        sizeRow = QtWidgets.QHBoxLayout()
        sizeRow.addWidget(self.spnTrailMinSize)
        sizeRow.addWidget(QtWidgets.QLabel(" -> "))
        sizeRow.addWidget(self.spnTrailMaxSize)
        opacityRow = QtWidgets.QHBoxLayout()
        opacityRow.addWidget(self.spnTrailMinOpacity)
        opacityRow.addWidget(QtWidgets.QLabel(" -> "))
        opacityRow.addWidget(self.spnTrailMaxOpacity)
        trajForm.addRow(self.chkTrajectoryTrail)
        trajForm.addRow(self.chkCurrentPosition)
        trajForm.addRow("Trail duration:", self.spnTrailDuration)
        trajForm.addRow("Trail size (start->end):", sizeRow)
        trajForm.addRow("Trail opacity (start->end):", opacityRow)
        outerLayout.addWidget(trajGroup)

        actionsRow = QtWidgets.QHBoxLayout()
        self.btnRefresh = QtWidgets.QPushButton("Refresh")
        self.btnRefresh.setToolTip("Rebuild meshes at the current slider position")
        actionsRow.addStretch(1)
        actionsRow.addWidget(self.btnRefresh)
        outerLayout.addLayout(actionsRow)

        outerLayout.addStretch(1)


    def _initialize_from_explorer(self):
        """ Populate UI controls from the explorer's current state. Signals are suspended during init. """
        self._suspend_signals = True
        try:
            params = self.explorer.params

            enable_historical = bool(params.get('enable_historical_spikes', True))
            enable_recent = bool(params.get('enable_recent_spikes', True))
            self.chkHistoricalSpikes.setChecked(enable_historical)
            self.chkRecentSpikes.setChecked(enable_recent)

            historical_window = params.get('longer_spikes_window', None)
            if historical_window is not None and historical_window.duration_seconds is not None:
                self.spnHistoricalDuration.setValue(float(historical_window.duration_seconds))
            else:
                self.spnHistoricalDuration.setValue(1024.0)

            recent_window = params.get('recent_spikes_window', None)
            if recent_window is not None and recent_window.duration_seconds is not None:
                self.spnTrailDuration.setValue(float(recent_window.duration_seconds))
            else:
                self.spnTrailDuration.setValue(10.0)

            trail_size_values = params.get('active_trail_size_values', None)
            if trail_size_values is not None and len(trail_size_values) >= 2:
                self.spnTrailMinSize.setValue(float(trail_size_values[0]))
                self.spnTrailMaxSize.setValue(float(trail_size_values[-1]))
            else:
                self.spnTrailMinSize.setValue(1.2)
                self.spnTrailMaxSize.setValue(0.4)

            trail_opacity_values = params.get('active_trail_opacity_values', None)
            if trail_opacity_values is not None and len(trail_opacity_values) >= 2:
                self.spnTrailMinOpacity.setValue(float(trail_opacity_values[0]))
                self.spnTrailMaxOpacity.setValue(float(trail_opacity_values[-1]))
            else:
                self.spnTrailMinOpacity.setValue(0.0)
                self.spnTrailMaxOpacity.setValue(0.6)

            self.chkTrajectoryTrail.setChecked(self._actor_visibility(self.explorer.animal_location_trail, default=True))
            self.chkCurrentPosition.setChecked(self._actor_visibility(self.explorer.animal_current_location_point, default=True))
        finally:
            self._suspend_signals = False


    def _connect_signals(self):
        self.chkHistoricalSpikes.toggled.connect(self._on_historical_spikes_toggled)
        self.chkRecentSpikes.toggled.connect(self._on_recent_spikes_toggled)
        self.chkTrajectoryTrail.toggled.connect(self._on_trajectory_trail_toggled)
        self.chkCurrentPosition.toggled.connect(self._on_current_position_toggled)
        self.spnTrailDuration.valueChanged.connect(self._on_trail_duration_changed)
        self.spnHistoricalDuration.valueChanged.connect(self._on_historical_duration_changed)
        self.spnTrailMinSize.valueChanged.connect(self._on_trail_size_range_changed)
        self.spnTrailMaxSize.valueChanged.connect(self._on_trail_size_range_changed)
        self.spnTrailMinOpacity.valueChanged.connect(self._on_trail_opacity_range_changed)
        self.spnTrailMaxOpacity.valueChanged.connect(self._on_trail_opacity_range_changed)
        self.btnRefresh.clicked.connect(self._on_refresh_clicked)
        # Re-apply checkbox-driven visibility after every mesh rebuild (e.g. on slider drag),
        # since the explorer's add_mesh(name=...) recreates each actor with default visibility=True.
        try:
            self.explorer.sigOnUpdateMeshes.connect(self._on_explorer_meshes_updated)
        except Exception as e:
            print(f"Interactive3dSpikeBehaviorOptionsWidget: could not connect sigOnUpdateMeshes: {e}")


    @staticmethod
    def _actor_visibility(actor, default: bool = True) -> bool:
        if actor is None:
            return default
        try:
            return bool(actor.GetVisibility())
        except Exception:
            return default


    @staticmethod
    def _set_actor_visibility(actor, visible: bool):
        if actor is None:
            return
        try:
            actor.SetVisibility(int(bool(visible)))
        except Exception:
            try:
                actor.visibility = bool(visible)
            except Exception:
                pass


    def _render_now(self):
        try:
            if self.explorer.p is not None:
                self.explorer.p.render()
        except Exception as e:
            print(f"Interactive3dSpikeBehaviorOptionsWidget: render failed: {e}")


    def _rebuild_at_current_slider(self):
        """ Re-trigger the explorer's mesh-rebuild at the current slider position so updated params take effect. """
        try:
            slider_wrapper = self.explorer.active_timestamp_slider_wrapper
        except Exception:
            slider_wrapper = None
        try:
            if slider_wrapper is not None:
                self.explorer.on_slider_update_mesh(self.explorer.active_timestamp_slider_curr_index)
            else:
                t_start, t_stop = self.explorer.active_timestamp_slider_curr_start_stop_times
                self.explorer.on_active_window_update_mesh(t_start, t_stop, enable_position_mesh_updates=True, render=True)
        except Exception as e:
            print(f"Interactive3dSpikeBehaviorOptionsWidget: rebuild failed: {e}")


    def _emit_changed(self, **kwargs):
        if self._suspend_signals:
            return
        try:
            self.sigOptionsChanged.emit(dict(kwargs))
        except Exception:
            pass


    @pyqtSlot(bool)
    def _on_historical_spikes_toggled(self, checked: bool):
        if self._suspend_signals:
            return
        # Only flip params.enable_historical_spikes on CHECK (so a previously-disabled rebuild path turns back on).
        # On UNCHECK, leave params.enable_historical_spikes alone and rely on actor SetVisibility(0). This avoids a
        # NameError in InteractivePlaceCellDataExplorer.on_active_window_update_mesh, where `flattened_spike_times`
        # is only defined inside the historical block but referenced from the recent-spikes block.
        if checked:
            self.explorer.params.enable_historical_spikes = True
        self._set_actor_visibility(self.explorer.spikes_main_historical, checked)
        self._render_now()
        self._emit_changed(enable_historical_spikes=bool(checked))


    @pyqtSlot(bool)
    def _on_recent_spikes_toggled(self, checked: bool):
        if self._suspend_signals:
            return
        if checked:
            self.explorer.params.enable_recent_spikes = True
        self._set_actor_visibility(self.explorer.spikes_main_recent_only, checked)
        self._render_now()
        self._emit_changed(enable_recent_spikes=bool(checked))


    @pyqtSlot(float, float)
    def _on_explorer_meshes_updated(self, t_start: float, t_stop: float):
        """ Re-apply current checkbox visibility state to every actor after the explorer rebuilds meshes. """
        self._reapply_visibility_state()


    def _reapply_visibility_state(self):
        self._set_actor_visibility(self.explorer.spikes_main_historical, self.chkHistoricalSpikes.isChecked())
        self._set_actor_visibility(self.explorer.spikes_main_recent_only, self.chkRecentSpikes.isChecked())
        self._set_actor_visibility(self.explorer.animal_location_trail, self.chkTrajectoryTrail.isChecked())
        self._set_actor_visibility(self.explorer.animal_current_location_point, self.chkCurrentPosition.isChecked())


    @pyqtSlot(bool)
    def _on_trajectory_trail_toggled(self, checked: bool):
        if self._suspend_signals:
            return
        self._set_actor_visibility(self.explorer.animal_location_trail, checked)
        self._render_now()
        self._emit_changed(show_trajectory_trail=bool(checked))


    @pyqtSlot(bool)
    def _on_current_position_toggled(self, checked: bool):
        if self._suspend_signals:
            return
        self._set_actor_visibility(self.explorer.animal_current_location_point, checked)
        self._render_now()
        self._emit_changed(show_current_position=bool(checked))


    @pyqtSlot(float)
    def _on_trail_duration_changed(self, new_value: float):
        if self._suspend_signals:
            return
        params = self.explorer.params
        sampling_rate = self.explorer.active_session.position.sampling_rate
        new_recent_window = VisualizationWindow(duration_seconds=float(new_value), sampling_rate=sampling_rate)
        params.recent_spikes_window = new_recent_window
        params.curr_view_window_length_samples = new_recent_window.duration_num_frames
        self._regenerate_trail_arrays()
        try:
            params.active_epoch_position_linear_indicies = np.arange(np.size(self.explorer.active_session.position.time))
            params.pre_computed_window_sample_indicies = new_recent_window.build_sliding_windows(params.active_epoch_position_linear_indicies)
        except Exception as e:
            print(f"Interactive3dSpikeBehaviorOptionsWidget: failed to recompute pre_computed_window_sample_indicies: {e}")
        self._rebuild_at_current_slider()
        self._emit_changed(trail_duration_seconds=float(new_value))


    @pyqtSlot(float)
    def _on_historical_duration_changed(self, new_value: float):
        if self._suspend_signals:
            return
        params = self.explorer.params
        sampling_rate = self.explorer.active_session.position.sampling_rate
        params.longer_spikes_window = VisualizationWindow(duration_seconds=float(new_value), sampling_rate=sampling_rate)
        self._rebuild_at_current_slider()
        self._emit_changed(historical_duration_seconds=float(new_value))


    @pyqtSlot(float)
    def _on_trail_size_range_changed(self, _new_value: float):
        if self._suspend_signals:
            return
        self._regenerate_trail_arrays(only_size=True)
        self._rebuild_at_current_slider()
        self._emit_changed(trail_size_min=float(self.spnTrailMinSize.value()), trail_size_max=float(self.spnTrailMaxSize.value()))


    @pyqtSlot(float)
    def _on_trail_opacity_range_changed(self, _new_value: float):
        if self._suspend_signals:
            return
        self._regenerate_trail_arrays(only_opacity=True)
        self._rebuild_at_current_slider()
        self._emit_changed(trail_opacity_min=float(self.spnTrailMinOpacity.value()), trail_opacity_max=float(self.spnTrailMaxOpacity.value()))


    @pyqtSlot()
    def _on_refresh_clicked(self):
        self._rebuild_at_current_slider()


    def closeEvent(self, event):
        try:
            self.explorer.sigOnUpdateMeshes.disconnect(self._on_explorer_meshes_updated)
        except Exception:
            pass
        super().closeEvent(event)


    def _regenerate_trail_arrays(self, only_size: bool = False, only_opacity: bool = False):
        """ Regenerate the trail size/opacity linspace arrays from the spinbox values, sized to the current recent_spikes_window. """
        params = self.explorer.params
        recent_window = params.get('recent_spikes_window', None)
        if recent_window is None or recent_window.duration_num_frames is None:
            return
        n = int(recent_window.duration_num_frames)
        if n < 2:
            n = 2
        if not only_opacity:
            params.active_trail_size_values = np.linspace(float(self.spnTrailMinSize.value()), float(self.spnTrailMaxSize.value()), n)
        if not only_size:
            params.active_trail_opacity_values = np.linspace(float(self.spnTrailMinOpacity.value()), float(self.spnTrailMaxOpacity.value()), n)
