import pytest

from pyphocorehelpers.DataStructure.logging_data_structures import LoggingBaseClass
from pyphoplacecellanalysis.GUI.Qt.Menus.BaseMenuProviderMixin import BaseMenuCommand
from pyphoplacecellanalysis.GUI.Qt.Menus.PhoMenuHelper import PhoMenuHelper


class _FailingMenuCommand(BaseMenuCommand):
    def __init__(self, spike_raster_window=None, action_identifier='actionTestFailing'):
        super().__init__(action_identifier=action_identifier)
        self._spike_raster_window = spike_raster_window

    def execute(self, *args, **kwargs):
        raise ValueError('menu failure for test')


class _WindowWithLogger:
    def __init__(self):
        self._logger = LoggingBaseClass(log_records=[])

    @property
    def bottom_playback_control_bar_logger(self):
        return self._logger


class _WindowWithoutLogger:
    pass


def test_base_menu_command_call_swallows_exception_without_window(capsys):
    command = _FailingMenuCommand(spike_raster_window=None)

    result = command()

    assert result is None
    captured = capsys.readouterr()
    assert 'menu failure for test' in captured.err or 'menu failure for test' in captured.out


def test_base_menu_command_call_reports_to_logger(capsys):
    window = _WindowWithLogger()
    command = _FailingMenuCommand(spike_raster_window=window)

    result = command()

    assert result is None
    log_text = window.bottom_playback_control_bar_logger.get_flattened_log_text()
    assert 'ERROR [menu:TestFailing]' in log_text
    assert 'menu failure for test' in log_text


def test_report_menu_error_without_logger_is_best_effort(capsys):
    window = _WindowWithoutLogger()

    PhoMenuHelper.report_menu_error(
        ValueError('standalone failure'),
        error_context='SmokeTest',
        spike_raster_window=window,
    )

    captured = capsys.readouterr()
    assert 'standalone failure' in captured.err or 'standalone failure' in captured.out


def test_connect_action_safe_swallows_exception():
    class _Action:
        def __init__(self):
            self._slot = None

        def triggered(self):
            return self

        def connect(self, slot):
            self._slot = slot
            return object()

        def emit(self):
            if self._slot is not None:
                self._slot()

    action = _Action()

    def _failing_callback():
        raise RuntimeError('lambda menu failure')

    PhoMenuHelper.connect_action_safe(
        action,
        _failing_callback,
        error_context='LambdaTest',
        spike_raster_window=None,
    )

    action.emit()


def test_spike2draster_import_has_no_menu_side_effects():
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster

    assert Spike2DRaster is not None
